"""Grad-CAM et superposition de cartes de chaleur sur l'image."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def grad_cam(model, x: torch.Tensor, class_idx: int = 1, target_layer=None) -> np.ndarray:
    """Grad-CAM pour une image (1, C, H, W). Renvoie une carte (h, w) dans [0, 1].

    Les crochets (hooks) sont poses puis retires a chaque appel : le modele
    reste utilisable normalement ensuite.
    """
    layer = target_layer or model.backbone.layer4[-1].conv2
    store: dict[str, torch.Tensor] = {}

    def save_activation(_module, _inp, out):
        store["act"] = out

    def save_gradient(_module, _grad_in, grad_out):
        store["grad"] = grad_out[0]

    handles = [layer.register_forward_hook(save_activation), layer.register_full_backward_hook(save_gradient)]
    was_training = model.training
    try:
        model.eval()
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            output = model(x)
            model.zero_grad(set_to_none=True)
            output[0, class_idx].backward()
        weights = store["grad"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * store["act"]).sum(dim=1)).squeeze(0).detach().cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype(np.float32)


def overlay(gray: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpose une carte [0, 1] (palette 'inferno') sur une image grise. Renvoie BGR uint8."""
    h, w = gray.shape[:2]
    heat = cv2.resize(np.clip(heat, 0, 1).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray
    return cv2.addWeighted(base, 1 - alpha, colored, alpha, 0)


def encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("encodage PNG impossible")
    return buf.tobytes()
