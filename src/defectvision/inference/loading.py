"""Lecture et ecriture du dossier de modeles.

Organisation (identique en local, dans DVC et dans le registre MLflow) :

    models/
      supervised/   model.pt  meta.json
      autoencoder/  model.pt  meta.json
      hybrid/       hybrid.json
      manifest.json             (version, empreintes, metriques)
"""

from __future__ import annotations

from pathlib import Path

import torch

from defectvision.models.autoencoder import build_autoencoder
from defectvision.models.classifier import DefectClassifier
from defectvision.models.hybrid import HybridModel
from defectvision.utils import read_json, write_json

SUPERVISED = "supervised"
AUTOENCODER = "autoencoder"
HYBRID = "hybrid"
MODEL_NAMES = (SUPERVISED, AUTOENCODER, HYBRID)


def save_torch_model(model: torch.nn.Module, meta: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, out_dir / "model.pt")
    write_json(out_dir / "meta.json", meta)


def _state(path: Path, device) -> dict:
    return torch.load(path, map_location=device, weights_only=True)


def load_supervised(models_dir: str | Path, device="cpu") -> tuple[DefectClassifier, dict]:
    folder = Path(models_dir) / SUPERVISED
    meta = read_json(folder / "meta.json")
    # pretrained=False : les poids viennent du fichier, aucun telechargement.
    model = DefectClassifier(num_classes=2, pretrained=False)
    model.load_state_dict(_state(folder / "model.pt", device))
    return model.to(device).eval(), meta


def load_autoencoder(models_dir: str | Path, device="cpu") -> tuple[torch.nn.Module, dict]:
    folder = Path(models_dir) / AUTOENCODER
    meta = read_json(folder / "meta.json")
    model = build_autoencoder(meta["config"], pretrained=False)
    model.load_state_dict(_state(folder / "model.pt", device))
    return model.to(device).eval(), meta


def load_hybrid(models_dir: str | Path) -> HybridModel:
    return HybridModel.load(Path(models_dir) / HYBRID / "hybrid.json")
