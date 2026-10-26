"""Pipeline complet sur donnees synthetiques : chaque etape produit ce qu'attend la suivante."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from conftest import InProject, make_part, small_params
from defectvision.inference.loading import MODEL_NAMES
from defectvision.inference.predictor import Predictor
from defectvision.utils import read_json


def test_pipeline_outputs(trained_project):
    root = trained_project
    for rel in ("data/interim/index.csv", "data/splits/train.csv", "data/splits/val.csv",
                "data/splits/blind_test.csv", "models/supervised/model.pt", "models/autoencoder/model.pt",
                "models/hybrid/hybrid.json", "models/manifest.json", "reports/metrics/blind_test.json",
                "reports/predictions/blind_test.csv", "reports/plots/supervised_history.csv"):
        assert (root / rel).exists(), rel
    figures = {p.name for p in (root / "reports/figures").glob("*.png")}
    assert {"dataset_examples.png", "roc_pr_curves.png", "score_space.png", "confusion_matrices.png",
            "metrics_ci.png", "autoencoder_maps.png", "training_curves.png"} <= figures


def test_thresholds_come_from_validation(trained_project):
    root = trained_project
    meta = read_json(root / "models/supervised/meta.json")
    assert meta["val_metrics"]["recall"] >= meta["target_recall"]
    ae = read_json(root / "models/autoencoder/meta.json")
    assert ae["val_metrics"]["recall"] >= ae["target_recall"]
    assert ae["trained_on"].startswith("pieces saines")
    report = read_json(root / "reports/metrics/blind_test.json")
    assert report["thresholds_chosen_on"] == "validation"
    assert set(report["models"]) == set(MODEL_NAMES)
    for m in report["models"].values():
        assert m["ci95"]["recall"]["low"] <= m["recall"] <= m["ci95"]["recall"]["high"]
    assert "passed" in report["gates"]


def test_autoencoder_is_trained_on_healthy_parts_only(trained_project):
    import pandas as pd

    root = trained_project
    train = pd.read_csv(root / "data/splits/train.csv")
    ae = read_json(root / "models/autoencoder/meta.json")
    assert ae["train_images"] == int((train["label"] == 0).sum())


def test_predictor_answers_for_each_model(trained_project):
    predictor = Predictor(trained_project / "models")
    gray = make_part(np.random.default_rng(9), 64, defect=True)
    for name in MODEL_NAMES:
        result = predictor.predict(gray, name, explain=True)
        assert result["label"] in ("OK", "DEFAUT")
        assert result["model_version"].startswith("dv-")
        assert result["heatmaps"]
    hybrid = predictor.predict(gray, "hybrid")
    assert {"p_supervised", "anomaly_score", "anomaly_z", "source"} <= set(hybrid["details"])
    assert isinstance(hybrid["review_required"], bool)
    with pytest.raises(ValueError):
        predictor.predict(gray, "inconnu")


def test_conv_autoencoder_variant_trains(trained_project, tmp_path):
    from pathlib import Path

    from defectvision.training.train_autoencoder import train_autoencoder

    params = copy.deepcopy(small_params())
    params["autoencoder"]["architecture"] = "conv"
    with InProject(trained_project):
        meta = train_autoencoder(params, trained_project, Path("data/splits"), tmp_path,
                                 tmp_path / "m.json", tmp_path / "h.csv")
    assert meta["architecture"] == "conv" and meta["tensor_mode"] == "gray"
    assert (tmp_path / "autoencoder" / "model.pt").exists()
