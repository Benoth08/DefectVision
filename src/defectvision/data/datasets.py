"""Jeux de donnees PyTorch et tirage equilibre."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from defectvision.data.augment import augment
from defectvision.data.preprocessing import preprocess_from_config, read_gray
from defectvision.data.tensors import to_tensor


class CastingDataset(Dataset):
    """Images d'un decoupage (CSV path,label), pretraitees une fois puis gardees en memoire."""

    def __init__(
        self,
        frame: pd.DataFrame,
        root: str | Path,
        image_size: int,
        preprocessing: dict,
        tensor_mode: str = "rgb",
        augment_cfg: dict | None = None,
        augment_mode: str = "none",
    ):
        self.paths = frame["path"].tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.root = Path(root)
        self.image_size = image_size
        self.preprocessing = preprocessing
        self.tensor_mode = tensor_mode
        self.augment_cfg = augment_cfg or {}
        self.augment_mode = augment_mode if augment_cfg is not None else "none"
        # Le pretraitement (CLAHE + redimensionnement) est deterministe : on le
        # fait une seule fois. ~50 Ko par image en 224x224.
        self._cache = [self._load(i) for i in range(len(self.paths))]

    def _load(self, idx: int) -> np.ndarray:
        gray = read_gray(self.root / self.paths[idx])
        return preprocess_from_config(gray, self.image_size, self.preprocessing)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        gray = self._cache[idx]
        if self.augment_mode != "none":
            # Graine tiree du generateur torch : differente a chaque epoch et
            # dans chaque processus de chargement, reproductible avec la graine globale.
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            gray = augment(gray, rng, self.augment_cfg, self.augment_mode)
        return to_tensor(gray, self.tensor_mode), torch.tensor(self.labels[idx], dtype=torch.long)


def balanced_sampler(labels: list[int], generator: torch.Generator | None = None) -> WeightedRandomSampler:
    """Tirage avec remise ou chaque classe a la meme probabilite d'etre tiree."""
    labels_arr = np.asarray(labels)
    counts = np.bincount(labels_arr, minlength=2).astype(float)
    weights = 1.0 / counts[labels_arr]
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(labels_arr),
        replacement=True,
        generator=generator,
    )


def read_split(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)
