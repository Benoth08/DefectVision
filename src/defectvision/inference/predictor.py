"""Predicteur unique pour les trois modeles (utilise par l'API et par MLflow)."""

from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

import numpy as np
import torch

from defectvision.data.preprocessing import preprocess_from_config
from defectvision.data.tensors import to_batch
from defectvision.explain.heatmaps import encode_png, grad_cam, overlay
from defectvision.inference.loading import (
    AUTOENCODER,
    HYBRID,
    MODEL_NAMES,
    SUPERVISED,
    load_autoencoder,
    load_hybrid,
    load_supervised,
)
from defectvision.inference.scoring import score_config
from defectvision.models.autoencoder import gaussian_blur, reduce_maps
from defectvision.utils import read_json


def _b64(image: np.ndarray) -> str:
    return base64.b64encode(encode_png(image)).decode("ascii")


class Predictor:
    """Charge le dossier de modeles et repond aux demandes de prediction."""

    def __init__(self, models_dir: str | Path, device: str = "cpu"):
        self.models_dir = Path(models_dir)
        self.device = torch.device(device)
        self.supervised, self.sup_meta = load_supervised(self.models_dir, self.device)
        self.autoencoder, self.ae_meta = load_autoencoder(self.models_dir, self.device)
        self.hybrid = load_hybrid(self.models_dir)
        self.score_cfg = score_config(self.ae_meta)
        manifest = self.models_dir / "manifest.json"
        self.manifest = read_json(manifest) if manifest.exists() else {}
        # Grad-CAM pose des crochets sur le modele : un verrou evite que deux
        # requetes simultanees se melangent.
        self._lock = threading.Lock()

    @property
    def version(self) -> str:
        return str(self.manifest.get("version", "inconnue"))

    def describe(self) -> dict:
        return {
            "version": self.version,
            "models": {
                SUPERVISED: {
                    "architecture": self.sup_meta.get("architecture"),
                    "threshold": self.sup_meta.get("threshold"),
                    "image_size": self.sup_meta.get("image_size"),
                },
                AUTOENCODER: {
                    "architecture": self.ae_meta.get("architecture"),
                    "threshold": self.ae_meta.get("threshold"),
                    "image_size": self.ae_meta.get("image_size"),
                },
                HYBRID: {"strategy": self.hybrid.strategy, "threshold": self.hybrid.threshold},
            },
            "metrics": self.manifest.get("metrics", {}),
        }

    def _tensor(self, gray: np.ndarray, meta: dict, mode: str) -> tuple[torch.Tensor, np.ndarray]:
        processed = preprocess_from_config(gray, int(meta["image_size"]), meta["preprocessing"])
        return to_batch([processed], mode).to(self.device), processed

    def _supervised(self, gray: np.ndarray, explain: bool) -> dict:
        x, processed = self._tensor(gray, self.sup_meta, "rgb")
        p = float(self.supervised.predict_proba(x)[0])
        out = {"p_supervised": p}
        if explain:
            cam = grad_cam(self.supervised, x, class_idx=1)
            out["gradcam_png"] = _b64(overlay(processed, cam))
        return out

    def _autoencoder(self, gray: np.ndarray, explain: bool) -> dict:
        x, processed = self._tensor(gray, self.ae_meta, self.autoencoder.tensor_mode)
        with torch.no_grad():
            maps = self.autoencoder.anomaly_map(x)
            cfg = self.score_cfg
            score = float(reduce_maps(maps, cfg["reduction"], cfg["topk_ratio"], cfg["blur_sigma"])[0])
        out = {"anomaly_score": score}
        if explain:
            # Echelle absolue : 1 = niveau du seuil de decision de l'autoencodeur.
            blurred = gaussian_blur(maps, cfg["blur_sigma"])[0, 0].cpu().numpy()
            heat = blurred / max(float(self.ae_meta["threshold"]), 1e-12)
            out["anomaly_png"] = _b64(overlay(processed, np.clip(heat, 0, 1)))
        return out

    def explain_arrays(self, gray: np.ndarray) -> dict:
        """Cartes brutes pour les figures : Grad-CAM, carte d'anomalie (1 = seuil), reconstruction."""
        with self._lock:
            x, sup_img = self._tensor(gray, self.sup_meta, "rgb")
            cam = grad_cam(self.supervised, x, class_idx=1)
            xa, ae_img = self._tensor(gray, self.ae_meta, self.autoencoder.tensor_mode)
            with torch.no_grad():
                maps = self.autoencoder.anomaly_map(xa)
                blurred = gaussian_blur(maps, self.score_cfg["blur_sigma"])[0, 0].cpu().numpy()
                recon = self.autoencoder.reconstruct(xa)
        out = {
            "supervised_image": sup_img,
            "gradcam": cam,
            "autoencoder_image": ae_img,
            "anomaly": blurred / max(float(self.ae_meta["threshold"]), 1e-12),
        }
        if recon is not None:
            out["reconstruction"] = (recon[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return out

    def predict(self, gray: np.ndarray, model: str = HYBRID, explain: bool = False) -> dict:
        if model not in MODEL_NAMES:
            raise ValueError(f"modele inconnu : {model} (attendu : {', '.join(MODEL_NAMES)})")
        start = time.perf_counter()
        with self._lock:
            details: dict = {}
            if model in (SUPERVISED, HYBRID):
                details.update(self._supervised(gray, explain))
            if model in (AUTOENCODER, HYBRID):
                details.update(self._autoencoder(gray, explain))

        heatmaps = {k.removesuffix("_png"): details.pop(k) for k in list(details) if k.endswith("_png")}
        review = None
        if model == SUPERVISED:
            score, threshold = details["p_supervised"], float(self.sup_meta["threshold"])
        elif model == AUTOENCODER:
            score, threshold = details["anomaly_score"], float(self.ae_meta["threshold"])
            details["anomaly_z"] = float(self.hybrid.ae_z(score))
        else:
            decision = self.hybrid.decide(details["p_supervised"], details["anomaly_score"])
            score, threshold = float(decision["probability"][0]), float(self.hybrid.threshold)
            details["anomaly_z"] = float(self.hybrid.ae_z(details["anomaly_score"]))
            details["source"] = str(decision["source"][0])
            review = bool(decision["review"][0])

        is_defect = score >= threshold
        result = {
            "model": model,
            "label": "DEFAUT" if is_defect else "OK",
            "is_defect": bool(is_defect),
            "score": float(score),
            "threshold": threshold,
            "review_required": review,
            "details": details,
            "model_version": self.version,
            "inference_ms": round((time.perf_counter() - start) * 1000, 2),
        }
        if heatmaps:
            result["heatmaps"] = heatmaps
        return result
