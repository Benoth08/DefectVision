"""Service FastAPI : sondes, prediction, erreurs, metriques Prometheus, rechargement."""

from __future__ import annotations

import importlib

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_part


def _png(seed: int = 0, defect: bool = False) -> bytes:
    ok, buf = cv2.imencode(".png", make_part(np.random.default_rng(seed), 64, defect))
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="module")
def client(trained_project):
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_SOURCE", "local")
    mp.setenv("MODEL_DIR", str(trained_project / "models"))
    mp.setenv("ADMIN_TOKEN", "jeton-de-test")
    mp.setenv("MAX_UPLOAD_BYTES", "200000")
    mp.delenv("MLFLOW_TRACKING_URI", raising=False)
    import defectvision.serving.app as app_module

    app_module = importlib.reload(app_module)
    with TestClient(app_module.app) as test_client:
        yield test_client
    mp.undo()


def test_probes(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["ready"]
    assert client.get("/models").json()["version"].startswith("dv-")


@pytest.mark.parametrize("model", ["supervised", "autoencoder", "hybrid"])
def test_predict_each_model(client, model):
    r = client.post("/predict", params={"model": model}, files={"file": ("piece.png", _png(1, True), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == model and body["label"] in ("OK", "DEFAUT")
    assert body["filename"] == "piece.png"
    assert "heatmaps" not in body


def test_predict_with_explanations(client):
    r = client.post("/predict", params={"explain": "true"}, files={"file": ("p.png", _png(2), "image/png")})
    assert r.status_code == 200
    heatmaps = r.json()["heatmaps"]
    assert set(heatmaps) == {"gradcam", "anomaly"}


def test_batch(client):
    files = [("files", (f"p{i}.png", _png(i, i % 2 == 0), "image/png")) for i in range(3)]
    r = client.post("/predict/batch", params={"model": "hybrid"}, files=files)
    assert r.status_code == 200
    assert r.json()["count"] == 3


def test_rejects_bad_inputs(client):
    assert client.post("/predict", files={"file": ("x.png", b"pas une image", "image/png")}).status_code == 400
    big = b"0" * 200_001
    assert client.post("/predict", files={"file": ("x.png", big, "image/png")}).status_code == 413
    r = client.post("/predict", params={"model": "inconnu"}, files={"file": ("p.png", _png(), "image/png")})
    assert r.status_code == 422


def _samples(text: str) -> dict[str, list]:
    from prometheus_client.parser import text_string_to_metric_families

    out: dict[str, list] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            out.setdefault(sample.name, []).append(sample)
    return out


def test_metrics_exposed(client):
    client.post("/predict", files={"file": ("p.png", _png(3), "image/png")})
    samples = _samples(client.get("/metrics").text)
    predictions = {(s.labels["model"], s.labels["label"]): s.value for s in samples["defectvision_predictions_total"]}
    assert sum(v for (model, _), v in predictions.items() if model == "hybrid") >= 1
    assert samples["defectvision_model_loaded"][0].value == 1.0
    assert samples["defectvision_input_mean_intensity_count"][0].value >= 1
    requests_ok = [s for s in samples["defectvision_http_requests_total"]
                   if s.labels == {"method": "POST", "route": "/predict", "status": "200"}]
    assert requests_ok and requests_ok[0].value >= 1


def test_service_does_not_need_training_libraries():
    """L'image de service n'embarque ni pandas, ni scikit-learn, ni matplotlib."""
    import os
    import subprocess
    import sys

    from conftest import PROJECT

    code = ("import sys, defectvision.serving.app; "
            "bad = [m for m in ('pandas', 'sklearn', 'matplotlib') if m in sys.modules]; "
            "assert not bad, bad")
    env = {**os.environ, "PYTHONPATH": str(PROJECT / "src"), "MODEL_SOURCE": "local"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_reload_requires_token(client):
    assert client.post("/admin/reload").status_code == 403
    assert client.post("/admin/reload", headers={"X-Admin-Token": "faux"}).status_code == 403
    r = client.post("/admin/reload", headers={"X-Admin-Token": "jeton-de-test"})
    assert r.status_code == 200 and r.json()["reloaded"]
