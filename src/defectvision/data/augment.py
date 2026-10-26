"""Augmentations d'entrainement (NumPy / OpenCV), appliquees apres le pretraitement.

Reprend la chaine du notebook d'origine (symetries, rotation avec bord
reflechi, luminosite/contraste, bruit gaussien, flou) sans dependance externe.
"""

from __future__ import annotations

import cv2
import numpy as np


def _rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def geometric(gray: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """Symetries et petite rotation : la piece reste une piece saine."""
    if rng.random() < cfg.get("hflip_p", 0.5):
        gray = gray[:, ::-1]
    if rng.random() < cfg.get("vflip_p", 0.5):
        gray = gray[::-1, :]
    if rng.random() < cfg.get("rotate_p", 0.5):
        limit = cfg.get("rotate_limit", 15)
        gray = _rotate(np.ascontiguousarray(gray), rng.uniform(-limit, limit))
    return np.ascontiguousarray(gray)


def photometric(gray: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """Luminosite/contraste, bruit et flou."""
    img = gray.astype(np.float32) / 255.0
    if rng.random() < cfg.get("brightness_contrast_p", 0.5):
        limit = cfg.get("brightness_contrast_limit", 0.2)
        alpha = 1.0 + rng.uniform(-limit, limit)
        beta = rng.uniform(-limit, limit)
        img = img * alpha + beta
    if rng.random() < cfg.get("noise_p", 0.3):
        img = img + rng.normal(0.0, cfg.get("noise_std", 0.03), img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    out = (img * 255.0 + 0.5).astype(np.uint8)
    if rng.random() < cfg.get("blur_p", 0.2):
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def augment(gray: np.ndarray, rng: np.random.Generator, cfg: dict, mode: str = "full") -> np.ndarray:
    """mode : 'none', 'geometric' ou 'full'."""
    if mode == "none":
        return gray
    gray = geometric(gray, rng, cfg)
    if mode == "full":
        gray = photometric(gray, rng, cfg)
    return gray
