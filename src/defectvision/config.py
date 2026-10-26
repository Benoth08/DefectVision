"""Chemins standard du projet et lecture de params.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Chemins relatifs a la racine du projet (DVC lance les etapes depuis la racine).
RAW_DIR = Path("data/raw/casting")
INDEX_CSV = Path("data/interim/index.csv")
SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
METRICS_DIR = REPORTS_DIR / "metrics"
PLOTS_DIR = REPORTS_DIR / "plots"
FIGURES_DIR = REPORTS_DIR / "figures"

SPLIT_NAMES = ("train", "val", "blind_test")
LABEL_NAMES = {0: "OK", 1: "DEFAUT"}


def project_root() -> Path:
    """Racine du projet : variable DEFECTVISION_ROOT, sinon dossier courant."""
    return Path(os.environ.get("DEFECTVISION_ROOT", Path.cwd())).resolve()


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Lit params.yaml (ou le fichier donne)."""
    path = Path(path) if path else project_root() / "params.yaml"
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def split_csv(split: str, splits_dir: str | Path = SPLITS_DIR) -> Path:
    """Chemin du fichier CSV d'un decoupage (train, val ou blind_test)."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"decoupage inconnu : {split}")
    return Path(splits_dir) / f"{split}.csv"
