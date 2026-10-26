"""Outils partages par les etapes d'entrainement."""

from __future__ import annotations

import csv
import platform
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from defectvision.utils import num_workers


def make_loader(dataset, batch_size: int, workers: int, sampler=None, shuffle: bool = False,
                generator: torch.Generator | None = None, train: bool = False) -> DataLoader:
    """train=True : on ecarte le dernier lot incomplet (une BatchNorm ne s'entraine pas sur une seule image)."""
    workers = num_workers(workers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        generator=generator if sampler is None else None,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=train and len(dataset) > batch_size,
    )


def clone_state(model: torch.nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def write_history(path: str | Path, rows: list[dict]) -> None:
    """Historique par epoch au format CSV (lu par 'dvc plots' et par MLflow)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_info(device: torch.device) -> dict:
    info = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info
