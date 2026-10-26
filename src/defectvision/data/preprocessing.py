"""Pretraitement commun a l'entrainement et au service d'inference.

Une seule fonction (preprocess_gray) est utilisee partout : l'image vue par
le modele en production est exactement celle vue a l'entrainement.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convertit une image BGR, BGRA ou deja grise en niveaux de gris uint8."""
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """CLAHE : rehausse le contraste local, ce qui fait ressortir les defauts fins."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(gray)


def preprocess_gray(
    image: np.ndarray,
    size: int,
    clahe: bool = True,
    clip_limit: float = 2.0,
    tile_grid: int = 8,
) -> np.ndarray:
    """Gris -> CLAHE (optionnel) -> redimensionnement carre. Renvoie uint8 (size, size)."""
    gray = to_gray(image)
    if clahe:
        gray = apply_clahe(gray, clip_limit, tile_grid)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def preprocess_from_config(image: np.ndarray, size: int, cfg: dict) -> np.ndarray:
    """Variante qui lit les reglages dans la section 'preprocessing' de params.yaml."""
    return preprocess_gray(
        image,
        size,
        clahe=bool(cfg.get("clahe", True)),
        clip_limit=float(cfg.get("clahe_clip_limit", 2.0)),
        tile_grid=int(cfg.get("clahe_tile_grid", 8)),
    )


def read_gray(path: str | Path) -> np.ndarray:
    """Lit une image sur disque en niveaux de gris (gere les chemins non ASCII)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"image illisible : {path}")
    return image


def decode_image(data: bytes) -> np.ndarray:
    """Decode des octets (JPEG, PNG, BMP...) en niveaux de gris uint8."""
    if not data:
        raise ValueError("fichier vide")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("image illisible")
    return to_gray(image)
