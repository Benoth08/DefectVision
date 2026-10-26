"""Etape DVC 'evaluate' : mesure finale des trois modeles sur le blind test.

Le blind test n'a servi a rien d'autre : ni entrainement, ni choix d'epoch,
ni choix de seuil (tous faits sur la validation). Il est lu une seule fois ici.

Sorties :
- reports/metrics/blind_test.json : metriques + intervalles de confiance 95 %
  (bootstrap stratifie), routage du modele hybride, conditions de promotion ;
- reports/predictions/blind_test.csv : scores et decisions par image ;
- models/manifest.json : version du jeu de modeles, empreintes, metriques ;
- reports/figures/*.png : figures du README.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from defectvision.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
    load_params,
    project_root,
    split_csv,
)
from defectvision.data.datasets import read_split
from defectvision.data.preprocessing import read_gray
from defectvision.evaluation import figures
from defectvision.evaluation.metrics import binary_metrics, bootstrap_ci
from defectvision.inference.loading import AUTOENCODER, HYBRID, MODEL_NAMES, SUPERVISED, load_hybrid
from defectvision.inference.predictor import Predictor
from defectvision.training.fit_hybrid import compute_scores
from defectvision.utils import configure_torch_threads, read_json, seed_everything, sha256_file, write_json

SUMMARY_KEYS = ("recall", "precision", "f2", "false_alarm_rate", "balanced_accuracy", "roc_auc", "average_precision")


def check_gates(results: dict, gates: dict) -> dict:
    hybrid = results[HYBRID]
    checks = {
        "recall": {"value": hybrid["recall"], "min": float(gates["min_recall"])},
        "precision": {"value": hybrid["precision"], "min": float(gates["min_precision"])},
    }
    for check in checks.values():
        check["passed"] = check["value"] >= check["min"]
    return {"model": HYBRID, "checks": checks, "passed": all(c["passed"] for c in checks.values())}


def cpu_latency(models_dir: Path, frame: pd.DataFrame, root: Path, n: int = 30) -> dict:
    """Temps d'inference par image sur CPU (un thread de service, sans explication)."""
    predictor = Predictor(models_dir, device="cpu")
    grays = [read_gray(root / p) for p in frame["path"].head(n)]
    out = {}
    for model in MODEL_NAMES:
        predictor.predict(grays[0], model)  # echauffement
        times = []
        for gray in grays:
            start = time.perf_counter()
            predictor.predict(gray, model)
            times.append((time.perf_counter() - start) * 1000)
        out[model] = {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
                      "images": len(times)}
    return out


def write_manifest(models_dir: Path, results: dict, val: dict) -> dict:
    files = {
        p.relative_to(models_dir).as_posix(): sha256_file(p)
        for p in sorted(models_dir.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = {
        "version": f"dv-{digest[:12]}",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
        "metrics": {
            "blind_test": {name: {k: results[name][k] for k in SUMMARY_KEYS} for name in MODEL_NAMES},
            "val": val,
        },
    }
    write_json(models_dir / "manifest.json", manifest)
    return manifest


def make_figures(params: dict, root: Path, splits_dir: Path, models_dir: Path, figures_dir: Path,
                 blind: pd.DataFrame, scores: dict, thresholds: dict, results: dict, hybrid) -> None:
    y = blind["label"].to_numpy().astype(int)
    train = read_split(split_csv("train", splits_dir))
    figures.dataset_examples(train, root, figures_dir / "dataset_examples.png")
    example = train[train["label"] == 1]["path"].iloc[0]
    figures.clahe_effect(read_gray(root / example), params["preprocessing"], figures_dir / "preprocessing_clahe.png")
    split_report = REPORTS_DIR / "split_report.json"
    if split_report.exists():
        figures.split_distribution(read_json(split_report), figures_dir / "split_distribution.png")
    sup_hist, ae_hist = PLOTS_DIR / "supervised_history.csv", PLOTS_DIR / "autoencoder_history.csv"
    if sup_hist.exists() and ae_hist.exists():
        figures.training_curves(pd.read_csv(sup_hist), pd.read_csv(ae_hist), figures_dir / "training_curves.png")

    figures.score_space(scores[SUPERVISED], scores[AUTOENCODER], y, hybrid, figures_dir / "score_space.png")
    figures.score_distributions(scores, thresholds, y, figures_dir / "score_distributions.png")
    figures.roc_pr_curves(scores, thresholds, y, results, figures_dir / "roc_pr_curves.png")
    figures.confusion_matrices(results, figures_dir / "confusion_matrices.png")
    figures.metrics_with_ci(results, figures_dir / "metrics_ci.png", int(params["evaluation"]["bootstrap_samples"]))

    # Exemples d'explication : defauts detectes (Grad-CAM) et melange OK / defauts (autoencodeur).
    predictor = Predictor(models_dir, device="cpu")
    # blind a un index 0..n-1 : l'indice d'une ligne est aussi sa position dans les scores.
    detected = np.flatnonzero((y == 1) & (scores[SUPERVISED] >= thresholds[SUPERVISED]))[:5]
    cam_items = []
    for pos in detected:
        arrays = predictor.explain_arrays(read_gray(root / blind.loc[pos, "path"]))
        cam_items.append({**arrays, "p_supervised": float(scores[SUPERVISED][pos])})
    if cam_items:
        figures.gradcam_examples(cam_items, figures_dir / "gradcam_examples.png")

    ae_items = []
    picks = list(np.flatnonzero(y == 0)[:2]) + list(np.flatnonzero(y == 1)[:3])
    for pos in picks:
        arrays = predictor.explain_arrays(read_gray(root / blind.loc[pos, "path"]))
        ae_items.append({**arrays, "label": int(y[pos]), "anomaly_score": float(scores[AUTOENCODER][pos]),
                         "threshold": float(thresholds[AUTOENCODER])})
    figures.autoencoder_examples(ae_items, figures_dir / "autoencoder_maps.png")


def evaluate(params: dict, root: Path, splits_dir: Path, models_dir: Path, metrics_path: Path,
             predictions_path: Path, figures_dir: Path, with_figures: bool = True) -> dict:
    configure_torch_threads()
    seed = int(params["seed"])
    seed_everything(seed)
    ev = params["evaluation"]

    blind = read_split(split_csv("blind_test", splits_dir)).reset_index(drop=True)
    y = blind["label"].to_numpy().astype(int)
    p_sup, s_ae, sup_meta, ae_meta = compute_scores(blind, root, models_dir, params)
    hybrid = load_hybrid(models_dir)
    decision = hybrid.decide(p_sup, s_ae)

    scores = {SUPERVISED: p_sup, AUTOENCODER: s_ae, HYBRID: decision["probability"]}
    thresholds = {SUPERVISED: float(sup_meta["threshold"]), AUTOENCODER: float(ae_meta["threshold"]),
                  HYBRID: float(hybrid.threshold)}
    results = {}
    for name in MODEL_NAMES:
        m = binary_metrics(y, scores[name], thresholds[name])
        m["ci95"] = bootstrap_ci(y, scores[name], thresholds[name], int(ev["bootstrap_samples"]),
                                 float(ev["ci_level"]), seed)
        results[name] = m

    defect = y == 1
    sources = pd.Series(decision["source"])
    routing = {
        "review_rate": float(decision["review"].mean()),
        "review_count": int(decision["review"].sum()),
        "review_rate_ok_parts": float(decision["review"][~defect].mean()) if (~defect).any() else 0.0,
        "review_rate_defects": float(decision["review"][defect].mean()) if defect.any() else 0.0,
        "novel_count": int(decision["novel"].sum()),
        "source_counts": {k: int(v) for k, v in sources.value_counts().items()},
        "defects_seen_only_by_autoencoder": int((defect & (sources == "non_supervise")).sum()),
        "defects_seen_only_by_supervised": int((defect & (sources == "supervise")).sum()),
        "defects_missed_by_both_branches": int((defect & (sources == "aucun")).sum()),
    }
    report = {
        "split": "blind_test",
        "images": int(len(y)),
        "defects": int(defect.sum()),
        "thresholds_chosen_on": "validation",
        "target_recall": float(ev["target_recall"]),
        "models": results,
        "hybrid_routing": routing,
        "gates": check_gates(results, params["gates"]),
        "cpu_latency": cpu_latency(models_dir, blind, root),
    }
    write_json(metrics_path, report)

    predictions = blind[["path", "label"]].assign(
        p_supervised=np.round(p_sup, 6),
        anomaly_score=np.round(s_ae, 6),
        p_hybrid=np.round(decision["probability"], 6),
        pred_supervised=(p_sup >= thresholds[SUPERVISED]).astype(int),
        pred_autoencoder=(s_ae >= thresholds[AUTOENCODER]).astype(int),
        pred_hybrid=decision["is_defect"].astype(int),
        hybrid_source=decision["source"],
        review_required=decision["review"].astype(int),
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False, lineterminator="\n")

    val = {SUPERVISED: {k: sup_meta["val_metrics"][k] for k in SUMMARY_KEYS},
           AUTOENCODER: {k: ae_meta["val_metrics"][k] for k in SUMMARY_KEYS}}
    hybrid_val = METRICS_DIR / "hybrid_val.json"
    if hybrid_val.exists():
        val[HYBRID] = {k: read_json(hybrid_val)["val"][hybrid.strategy][k] for k in SUMMARY_KEYS}
    manifest = write_manifest(models_dir, results, val)

    if with_figures:
        make_figures(params, root, splits_dir, models_dir, figures_dir, blind, scores, thresholds, results, hybrid)

    for name, m in results.items():
        ci = m["ci95"]["recall"]
        print(f"  {name:12s} | rappel={m['recall']:.3f} [{ci['low']:.3f}-{ci['high']:.3f}] "
              f"precision={m['precision']:.3f} F2={m['f2']:.3f} AUC={m['roc_auc']:.4f}")
    print(f"[OK] version {manifest['version']} ; promotion possible : {report['gates']['passed']}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--splits", default=str(SPLITS_DIR))
    parser.add_argument("--models", default=str(MODELS_DIR))
    parser.add_argument("--metrics", default=str(METRICS_DIR / "blind_test.json"))
    parser.add_argument("--predictions", default=str(REPORTS_DIR / "predictions" / "blind_test.csv"))
    parser.add_argument("--figures", default=str(FIGURES_DIR))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    evaluate(load_params(args.params), project_root(), Path(args.splits), Path(args.models), Path(args.metrics),
             Path(args.predictions), Path(args.figures), with_figures=not args.no_figures)


if __name__ == "__main__":
    main()
