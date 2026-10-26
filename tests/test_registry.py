"""Enregistrement MLflow de bout en bout (base SQLite temporaire, sans serveur)."""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import pytest

from conftest import InProject, small_params

mlflow = pytest.importorskip("mlflow")


@pytest.fixture()
def tracking(trained_project, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    return trained_project


def test_register_promotes_and_serves_the_champion(tracking):
    from defectvision.registry.register import register

    with InProject(tracking):
        first = register(small_params(), Path("models"), "essai", "defectvision-essai", promote=True)
        second = register(small_params(), Path("models"), "essai", "defectvision-essai", promote=True)
        assert first["promoted"] is first["gates_passed"]
        if first["gates_passed"]:
            assert first["reason"] == "premier champion"
            # Meme jeu de modeles, meme F2 : la nouvelle version devient champion.
            assert second["promoted"] and second["previous_champion"] == first["registered_version"]

        client = mlflow.MlflowClient()
        challenger = client.get_model_version_by_alias("defectvision-essai", "challenger")
        assert int(challenger.version) == second["registered_version"]

        model = mlflow.pyfunc.load_model(f"models:/defectvision-essai/{second['registered_version']}")
        path = pd.read_csv("data/splits/blind_test.csv")["path"].iloc[0]
        frame = pd.DataFrame({"image_base64": [base64.b64encode(Path(path).read_bytes()).decode()],
                              "model": ["hybrid"]})
        out = model.predict(frame)
        assert out.loc[0, "label"] in ("OK", "DEFAUT")
