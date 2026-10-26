"""Scores par lot : probabilite de defaut (supervise) et score d'anomalie (autoencodeur)."""

from __future__ import annotations

import numpy as np
import torch

from defectvision.models.autoencoder import reduce_maps


def score_config(meta_or_cfg: dict) -> dict:
    """Reglages de reduction carte -> score, lus dans meta.json ou params.yaml."""
    cfg = meta_or_cfg.get("score", meta_or_cfg)
    return {
        "reduction": cfg.get("score_reduction", cfg.get("reduction", "max")),
        "topk_ratio": float(cfg.get("score_topk_ratio", cfg.get("topk_ratio", 0.01))),
        "blur_sigma": float(cfg.get("score_blur_sigma", cfg.get("blur_sigma", 4.0))),
    }


@torch.no_grad()
def supervised_scores(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for images, _ in loader:
        out.append(model.predict_proba(images.to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


@torch.no_grad()
def autoencoder_scores(model, loader, device, cfg: dict) -> np.ndarray:
    model.eval()
    cfg = score_config(cfg)
    out = []
    for images, _ in loader:
        maps = model.anomaly_map(images.to(device))
        out.append(reduce_maps(maps, cfg["reduction"], cfg["topk_ratio"], cfg["blur_sigma"]).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


@torch.no_grad()
def autoencoder_loss(model, loader, device) -> float:
    """Perte de reconstruction moyenne (sert a l'arret anticipe sur des pieces saines)."""
    model.eval()
    total, count = 0.0, 0
    for images, _ in loader:
        images = images.to(device)
        total += float(model.loss(images)) * images.shape[0]
        count += images.shape[0]
    return total / max(count, 1)
