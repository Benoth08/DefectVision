"""Conversion image grise -> tenseur (sans pandas : utilise aussi par le service)."""

from __future__ import annotations

import numpy as np
import torch

from defectvision.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD

# 'rgb' = 3 canaux normalises ImageNet (ResNet) ; 'gray' = 1 canal dans [0, 1].
TENSOR_MODES = ("rgb", "gray")


def to_tensor(gray: np.ndarray, mode: str) -> torch.Tensor:
    """Image grise uint8 (H, W) -> tenseur (C, H, W) float32."""
    img = torch.from_numpy(np.ascontiguousarray(gray)).float().div_(255.0)
    if mode == "gray":
        return img.unsqueeze(0)
    if mode == "rgb":
        img = img.unsqueeze(0).expand(3, -1, -1)
        mean = torch.from_numpy(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.from_numpy(IMAGENET_STD).view(3, 1, 1)
        return (img - mean) / std
    raise ValueError(f"mode de tenseur inconnu : {mode}")


def to_batch(grays: list[np.ndarray], mode: str) -> torch.Tensor:
    return torch.stack([to_tensor(g, mode) for g in grays])
