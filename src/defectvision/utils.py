"""Outils communs : graines, peripherique de calcul, JSON, empreintes."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    """Fixe les graines (Python, NumPy, PyTorch) pour des runs reproductibles."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto"):
    """Renvoie le peripherique torch : 'auto' choisit le GPU s'il existe."""
    import torch

    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def num_workers(default: int) -> int:
    """Nombre de processus de chargement (surcharge : DV_NUM_WORKERS)."""
    return int(os.environ.get("DV_NUM_WORKERS", default))


def configure_torch_threads() -> None:
    """Limite les threads CPU de torch si TORCH_NUM_THREADS est defini."""
    value = os.environ.get("TORCH_NUM_THREADS")
    if value:
        import torch

        torch.set_num_threads(int(value))


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    """Ecrit un JSON lisible, en UTF-8, en creant le dossier parent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(_to_builtin(obj), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
