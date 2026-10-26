"""Etape DVC 'train_autoencoder' : autoencodeur entraine sur les seules pieces saines.

- Entrainement : images OK du train uniquement (aucun defaut vu).
- Arret anticipe : perte de reconstruction sur les pieces OK de la validation
  (critere sans etiquette de defaut).
- Seuil : choisi sur toute la validation (OK + defauts), comme pour le
  supervise. C'est le seul usage d'etiquettes de defaut, indispensable pour
  calibrer un score d'anomalie qui n'a pas d'echelle propre.
- Normalisation robuste du score (mediane et dispersion des pieces OK de la
  validation) : utilisee par le modele hybride.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from defectvision.config import METRICS_DIR, MODELS_DIR, PLOTS_DIR, SPLITS_DIR, load_params, project_root, split_csv
from defectvision.data.datasets import CastingDataset, read_split
from defectvision.evaluation.metrics import binary_metrics, select_threshold
from defectvision.inference.loading import AUTOENCODER, save_torch_model
from defectvision.inference.scoring import autoencoder_loss, autoencoder_scores, score_config
from defectvision.models.autoencoder import build_autoencoder
from defectvision.training.common import clone_state, make_loader, run_info, write_history
from defectvision.utils import configure_torch_threads, get_device, seed_everything, write_json

AE_CONFIG_KEYS = ("architecture", "latent_channels", "hidden_channels", "ssim_weight")


def robust_stats(scores: np.ndarray) -> tuple[float, float]:
    """Mediane et ecart-type robuste (MAD x 1,4826) ; repli sur l'ecart-type si MAD nul."""
    center = float(np.median(scores))
    scale = float(1.4826 * np.median(np.abs(scores - center)))
    if scale <= 0:
        scale = float(np.std(scores)) or 1.0
    return center, scale


def train_autoencoder(params: dict, root: Path, splits_dir: Path, models_dir: Path,
                      metrics_path: Path, history_path: Path) -> dict:
    configure_torch_threads()
    seed = int(params["seed"])
    seed_everything(seed)
    cfg, aug = params["autoencoder"], params["augmentation"]
    # Pretraitement propre a l'autoencodeur s'il est precise (ex. sans CLAHE), sinon le commun.
    pre = {**params["preprocessing"], **cfg.get("preprocessing", {})}
    target_recall = float(params["evaluation"]["target_recall"])
    device = get_device(params["runtime"]["device"])
    workers = int(params["runtime"]["num_workers"])
    size, batch = int(cfg["image_size"]), int(cfg["batch_size"])

    model = build_autoencoder(cfg).to(device)
    mode = model.tensor_mode

    train_df = read_split(split_csv("train", splits_dir))
    val_df = read_split(split_csv("val", splits_dir))
    train_ok = train_df[train_df["label"] == 0]
    val_ok = val_df[val_df["label"] == 0]

    train_ds = CastingDataset(train_ok, root, size, pre, mode, aug, cfg.get("augment", "geometric"))
    train_plain_ds = CastingDataset(train_ok, root, size, pre, mode)
    val_ok_ds = CastingDataset(val_ok, root, size, pre, mode)
    val_ds = CastingDataset(val_df, root, size, pre, mode)
    val_y = np.asarray(val_ds.labels)
    print(f"[OK] autoencodeur '{cfg['architecture']}' sur {device} ; "
          f"train OK={len(train_ds)} ; validation={len(val_ds)} (dont OK={len(val_ok_ds)})")

    generator = torch.Generator().manual_seed(seed)
    train_loader = make_loader(train_ds, batch, workers, shuffle=True, generator=generator, train=True)
    val_ok_loader = make_loader(val_ok_ds, batch, workers)
    val_loader = make_loader(val_ds, batch, workers)

    if hasattr(model, "fit_normalization"):
        plain_loader = make_loader(train_plain_ds, batch, workers)
        model.fit_normalization(images.to(device) for images, _ in plain_loader)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable, lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    score_cfg = score_config(cfg)
    patience = int(cfg.get("patience", 1000))

    history: list[dict] = []
    best_loss, best_state, best_epoch, stale = float("inf"), clone_state(model), 0, 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        start = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(images)
            loss.backward()
            optimizer.step()
            total += loss.item() * images.shape[0]
            count += images.shape[0]
        train_loss = total / max(count, 1)
        val_ok_loss = autoencoder_loss(model, val_ok_loader, device)
        # Suivi seulement (pas de selection) : separation OK / defaut sur la validation.
        scores = autoencoder_scores(model, val_loader, device, score_cfg)
        auc = binary_metrics(val_y, scores, float(np.median(scores)))["roc_auc"]
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_ok_loss": round(val_ok_loss, 6),
            "val_roc_auc": round(auc, 6),
            "seconds": round(time.perf_counter() - start, 2),
        })
        if epoch == 1 or epoch % 5 == 0:
            print(f"  epoch {epoch:3d} | train={train_loss:.5f} | val OK={val_ok_loss:.5f} | AUC val={auc:.4f}")
        if val_ok_loss < best_loss - 1e-7:
            best_loss, best_state, best_epoch, stale = val_ok_loss, clone_state(model), epoch, 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  arret anticipe a l'epoch {epoch} ({patience} epochs sans progres)")
                break

    model.load_state_dict(best_state)
    scores = autoencoder_scores(model, val_loader, device, score_cfg)
    confidence = params["evaluation"].get("recall_confidence")
    threshold = select_threshold(val_y, scores, target_recall, confidence)
    val_metrics = binary_metrics(val_y, scores, threshold)
    center, scale = robust_stats(scores[val_y == 0])
    meta = {
        "model": AUTOENCODER,
        "architecture": cfg["architecture"],
        "config": {k: cfg[k] for k in AE_CONFIG_KEYS if k in cfg},
        "image_size": size,
        "preprocessing": pre,
        "tensor_mode": mode,
        "score": score_cfg,
        "threshold": threshold,
        "target_recall": target_recall,
        "recall_confidence": confidence,
        "ok_score_center": center,
        "ok_score_scale": scale,
        "best_epoch": best_epoch,
        "trained_on": "pieces saines (OK) du train uniquement",
        "train_images": len(train_ds),
        "val_metrics": val_metrics,
        **run_info(device),
    }
    save_torch_model(model, meta, models_dir / AUTOENCODER)
    write_history(history_path, history)
    write_json(metrics_path, {"best_epoch": best_epoch, "val": val_metrics})
    print(f"[OK] seuil={threshold:.6f} rappel={val_metrics['recall']:.3f} precision={val_metrics['precision']:.3f} "
          f"AUC={val_metrics['roc_auc']:.4f}")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--splits", default=str(SPLITS_DIR))
    parser.add_argument("--models", default=str(MODELS_DIR))
    parser.add_argument("--metrics", default=str(METRICS_DIR / "autoencoder_val.json"))
    parser.add_argument("--history", default=str(PLOTS_DIR / "autoencoder_history.csv"))
    args = parser.parse_args(argv)
    train_autoencoder(load_params(args.params), project_root(), Path(args.splits), Path(args.models),
                      Path(args.metrics), Path(args.history))


if __name__ == "__main__":
    main()
