"""Etape DVC 'fit_hybrid' : fusion des scores supervise et non supervise.

Les deux modeles sont deja entraines ; on calcule leurs scores sur la
validation, puis :
- 'stacking' : regression logistique (classes equilibrees) sur
  [logit(p_supervise), z_anomalie]. Le seuil est choisi sur des predictions
  hors echantillon (validation croisee stratifiee) pour ne pas etre optimiste,
  puis le modele est reappris sur toute la validation ;
- 'or' : defaut si l'une des branches depasse son seuil ; le seuil commun
  decale les deux branches ensemble pour viser le meme rappel.

Les deux strategies sont mesurees ; params.yaml choisit celle qui est deployee.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from defectvision.config import METRICS_DIR, MODELS_DIR, SPLITS_DIR, load_params, project_root, split_csv
from defectvision.data.datasets import CastingDataset, read_split
from defectvision.evaluation.metrics import binary_metrics, select_threshold
from defectvision.inference.loading import HYBRID, load_autoencoder, load_supervised
from defectvision.inference.scoring import autoencoder_scores, supervised_scores
from defectvision.models.hybrid import HybridModel, logit
from defectvision.training.common import make_loader
from defectvision.utils import configure_torch_threads, get_device, seed_everything, write_json


def compute_scores(frame, root: Path, models_dir: Path, params: dict) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Probabilites supervisees et scores d'anomalie d'un decoupage."""
    device = get_device(params["runtime"]["device"])
    workers = int(params["runtime"]["num_workers"])
    sup, sup_meta = load_supervised(models_dir, device)
    ae, ae_meta = load_autoencoder(models_dir, device)
    sup_ds = CastingDataset(frame, root, int(sup_meta["image_size"]), sup_meta["preprocessing"], "rgb")
    ae_ds = CastingDataset(frame, root, int(ae_meta["image_size"]), ae_meta["preprocessing"], ae.tensor_mode)
    p_sup = supervised_scores(sup, make_loader(sup_ds, 32, workers), device)
    s_ae = autoencoder_scores(ae, make_loader(ae_ds, 16, workers), device, ae_meta)
    return p_sup, s_ae, sup_meta, ae_meta


def fit_hybrid(params: dict, root: Path, splits_dir: Path, models_dir: Path, metrics_path: Path) -> HybridModel:
    configure_torch_threads()
    seed = int(params["seed"])
    seed_everything(seed)
    cfg = params["hybrid"]
    target_recall = float(params["evaluation"]["target_recall"])
    confidence = params["evaluation"].get("recall_confidence")

    val_df = read_split(split_csv("val", splits_dir))
    y = val_df["label"].to_numpy().astype(int)
    p_sup, s_ae, sup_meta, ae_meta = compute_scores(val_df, root, models_dir, params)

    novelty_quantile = float(cfg.get("novelty_quantile", 0.99))
    base = {
        "ae_center": float(ae_meta["ok_score_center"]),
        "ae_scale": float(ae_meta["ok_score_scale"]),
        "supervised_threshold": float(sup_meta["threshold"]),
        "autoencoder_threshold": float(ae_meta["threshold"]),
        "review_margin": float(cfg.get("review_margin", 0.1)),
        "novelty_threshold": float(np.quantile(s_ae[y == 0], novelty_quantile)),
    }

    # --- Strategie 'stacking' ---
    probe = HybridModel(strategy="stacking", **base)
    features = np.column_stack([logit(p_sup), probe.ae_z(s_ae)])
    folds = min(int(cfg.get("cv_folds", 5)), int(np.bincount(y).min()))
    lr = LogisticRegression(C=float(cfg.get("C", 1.0)), class_weight="balanced", max_iter=1000)
    oof = cross_val_predict(
        lr, features, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed), method="predict_proba"
    )[:, 1]
    stacking_threshold = select_threshold(y, oof, target_recall, confidence)
    lr.fit(features, y)
    stacking = HybridModel(
        strategy="stacking",
        threshold=stacking_threshold,
        coef_supervised=float(lr.coef_[0, 0]),
        coef_autoencoder=float(lr.coef_[0, 1]),
        intercept=float(lr.intercept_[0]),
        extra={"cv_folds": folds, "C": float(cfg.get("C", 1.0))},
        **base,
    )

    # --- Strategie 'or' ---
    or_model = HybridModel(strategy="or", **base)
    or_scores = or_model.probability(p_sup, s_ae)
    or_model.threshold = select_threshold(y, or_scores, target_recall, confidence)

    candidates = {"stacking": (stacking, oof), "or": (or_model, or_scores)}
    strategy = cfg.get("strategy", "stacking")
    if strategy not in candidates:
        raise ValueError(f"strategie inconnue : {strategy}")
    chosen, _ = candidates[strategy]

    report = {
        "deployed_strategy": strategy,
        "val": {
            name: binary_metrics(y, scores, model.threshold) for name, (model, scores) in candidates.items()
        },
        "note": "stacking : metriques sur predictions hors echantillon (validation croisee)",
    }
    (models_dir / HYBRID).mkdir(parents=True, exist_ok=True)
    chosen.save(models_dir / HYBRID / "hybrid.json")
    write_json(metrics_path, report)
    for name, m in report["val"].items():
        print(f"  {name:8s} | rappel={m['recall']:.3f} precision={m['precision']:.3f} "
              f"F2={m['f2']:.3f} AP={m['average_precision']:.4f}")
    print(f"[OK] strategie deployee : {strategy} (seuil={chosen.threshold:.4f})")
    return chosen


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--splits", default=str(SPLITS_DIR))
    parser.add_argument("--models", default=str(MODELS_DIR))
    parser.add_argument("--metrics", default=str(METRICS_DIR / "hybrid_val.json"))
    args = parser.parse_args(argv)
    fit_hybrid(load_params(args.params), project_root(), Path(args.splits), Path(args.models), Path(args.metrics))


if __name__ == "__main__":
    main()
