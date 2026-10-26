"""Etape DVC 'train_supervised' : ResNet-18 en deux phases.

Phase 1 : le ResNet est gele, seule la tete apprend (lr eleve).
Phase 2 : les derniers blocs sont degeles (lr faible, decroissance cosinus).

Desequilibre des classes : par defaut, tirage equilibre (chaque lot ~50/50).
Selection du meilleur epoch : precision moyenne (aire sous la courbe
precision-rappel) sur la validation, insensible au seuil et au desequilibre.
Seuil final : precision maximale sous la contrainte rappel >= cible, sur la validation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from defectvision.config import METRICS_DIR, MODELS_DIR, PLOTS_DIR, SPLITS_DIR, load_params, project_root, split_csv
from defectvision.data.datasets import CastingDataset, balanced_sampler, read_split
from defectvision.evaluation.metrics import binary_metrics, select_threshold
from defectvision.inference.loading import SUPERVISED, save_torch_model
from defectvision.inference.scoring import supervised_scores
from defectvision.models.classifier import DefectClassifier
from defectvision.training.common import clone_state, make_loader, run_info, write_history
from defectvision.utils import configure_torch_threads, get_device, seed_everything, write_json


def _train_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total, count, tp, positives = 0.0, 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.shape[0]
        count += images.shape[0]
        preds = outputs.argmax(dim=1)
        tp += int(((preds == 1) & (labels == 1)).sum())
        positives += int((labels == 1).sum())
    return total / max(count, 1), tp / max(positives, 1)


@torch.no_grad()
def _val_loss(model, loader, criterion, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        total += float(criterion(model(images), labels)) * images.shape[0]
        count += images.shape[0]
    return total / max(count, 1)


def train_supervised(params: dict, root: Path, splits_dir: Path, models_dir: Path,
                     metrics_path: Path, history_path: Path) -> dict:
    configure_torch_threads()
    seed = int(params["seed"])
    seed_everything(seed)
    cfg, pre, aug = params["supervised"], params["preprocessing"], params["augmentation"]
    target_recall = float(params["evaluation"]["target_recall"])
    device = get_device(params["runtime"]["device"])
    workers = int(params["runtime"]["num_workers"])
    size, batch = int(cfg["image_size"]), int(cfg["batch_size"])

    train_df = read_split(split_csv("train", splits_dir))
    val_df = read_split(split_csv("val", splits_dir))
    train_ds = CastingDataset(train_df, root, size, pre, "rgb", aug, "full")
    val_ds = CastingDataset(val_df, root, size, pre, "rgb")
    val_y = np.asarray(val_ds.labels)

    generator = torch.Generator().manual_seed(seed)
    balancing = cfg.get("balancing", "sampler")
    sampler = balanced_sampler(train_ds.labels, generator) if balancing == "sampler" else None
    train_loader = make_loader(train_ds, batch, workers, sampler=sampler, shuffle=True, generator=generator,
                               train=True)
    val_loader = make_loader(val_ds, batch, workers)

    class_weights = None
    if balancing == "class_weight":
        counts = np.bincount(train_ds.labels, minlength=2).astype(float)
        class_weights = torch.tensor(len(train_ds) / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = DefectClassifier(num_classes=2, pretrained=bool(cfg["pretrained"])).to(device)
    print(f"[OK] ResNet-18 sur {device} ; train={len(train_ds)} val={len(val_ds)} ; equilibrage={balancing}")

    phases = [
        ("phase1", int(cfg["epochs_phase1"]), float(cfg["lr_phase1"])),
        ("phase2", int(cfg["epochs_phase2"]), float(cfg["lr_phase2"])),
    ]
    patience = int(cfg.get("patience", 1000))
    history: list[dict] = []
    best_ap, best_state, best_epoch = -1.0, clone_state(model), 0
    epoch = 0
    for phase, n_epochs, lr in phases:
        if n_epochs <= 0:
            continue
        if phase == "phase1":
            model.freeze_backbone()
        else:
            trainable = model.unfreeze_top_layers(int(cfg["unfreeze_layers"]))
            print(f"  parametres entrainables : {trainable:,}")
        optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs) if phase == "phase2" else None
        stale = 0
        for _ in range(n_epochs):
            epoch += 1
            start = time.perf_counter()
            train_loss, train_recall = _train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = _val_loss(model, val_loader, criterion, device)
            probs = supervised_scores(model, val_loader, device)
            m = binary_metrics(val_y, probs, 0.5)
            history.append({
                "epoch": epoch,
                "phase": phase,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "train_recall": round(train_recall, 6),
                "val_recall": round(m["recall"], 6),
                "val_f2": round(m["f2"], 6),
                "val_average_precision": round(m["average_precision"], 6),
                "val_roc_auc": round(m["roc_auc"], 6),
                "seconds": round(time.perf_counter() - start, 2),
            })
            print(f"  [{phase}] epoch {epoch:2d} | train loss={train_loss:.4f} | val loss={val_loss:.4f} "
                  f"AP={m['average_precision']:.4f} rappel@0.5={m['recall']:.3f}")
            if m["average_precision"] > best_ap + 1e-6:
                best_ap, best_state, best_epoch, stale = m["average_precision"], clone_state(model), epoch, 0
            else:
                stale += 1
            if scheduler is not None:
                scheduler.step()
            if stale >= patience:
                print(f"  arret anticipe ({phase}) : {patience} epochs sans progres")
                break

    model.load_state_dict(best_state)
    probs = supervised_scores(model, val_loader, device)
    confidence = params["evaluation"].get("recall_confidence")
    threshold = select_threshold(val_y, probs, target_recall, confidence)
    val_metrics = binary_metrics(val_y, probs, threshold)
    meta = {
        "model": SUPERVISED,
        "architecture": "resnet18",
        "image_size": size,
        "preprocessing": pre,
        "tensor_mode": "rgb",
        "threshold": threshold,
        "target_recall": target_recall,
        "recall_confidence": confidence,
        "best_epoch": best_epoch,
        "balancing": balancing,
        "val_metrics": val_metrics,
        **run_info(device),
    }
    save_torch_model(model, meta, models_dir / SUPERVISED)
    write_history(history_path, history)
    write_json(metrics_path, {"best_epoch": best_epoch, "val": val_metrics})
    print(f"[OK] seuil={threshold:.4f} rappel={val_metrics['recall']:.3f} precision={val_metrics['precision']:.3f} "
          f"AP={val_metrics['average_precision']:.4f}")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--splits", default=str(SPLITS_DIR))
    parser.add_argument("--models", default=str(MODELS_DIR))
    parser.add_argument("--metrics", default=str(METRICS_DIR / "supervised_val.json"))
    parser.add_argument("--history", default=str(PLOTS_DIR / "supervised_history.csv"))
    args = parser.parse_args(argv)
    train_supervised(load_params(args.params), project_root(), Path(args.splits), Path(args.models),
                     Path(args.metrics), Path(args.history))


if __name__ == "__main__":
    main()
