"""Journalise un run complet dans MLflow, enregistre le jeu de modeles et le promeut.

Appele par Airflow apres 'dvc repro' (ou a la main) :

    python -m defectvision.registry.register --promote

- un run MLflow par execution du pipeline : parametres, historiques par
  epoch, metriques de validation et du blind test (avec intervalles), figures,
  dvc.lock ;
- le dossier models/ est enregistre comme modele 'pyfunc' dans le registre
  (nom par defaut : defectvision) et recoit l'alias 'challenger' ;
- avec --promote, l'alias 'champion' passe a cette version si les conditions
  de params.yaml (gates) sont remplies et si le F2 du modele hybride sur le
  blind test n'est pas inferieur a celui du champion actuel.

Les etapes DVC n'ecrivent rien dans MLflow : le pipeline reste reproductible
hors ligne, et un entrainement fait ailleurs (GPU distant) s'enregistre de la
meme facon.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from importlib import metadata
from pathlib import Path

import pandas as pd

from defectvision.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR, load_params
from defectvision.utils import read_json, write_json

COMPARISON_METRIC = "f2"
RUNTIME_PACKAGES = ("torch", "torchvision", "numpy", "opencv-python-headless", "pandas", "pyyaml")


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = str(value)
    return out


def git_commit() -> str:
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "inconnu"


def pinned_requirements() -> list[str]:
    reqs = []
    for pkg in RUNTIME_PACKAGES:
        try:
            reqs.append(f"{pkg}=={metadata.version(pkg).split('+')[0]}")
        except metadata.PackageNotFoundError:
            continue
    return reqs


def _log_history(mlflow, path: Path, prefix: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    history = pd.read_csv(path)
    for _, row in history.iterrows():
        step = int(row["epoch"])
        for col, value in row.items():
            if col in ("epoch", "phase") or not isinstance(value, int | float):
                continue
            mlflow.log_metric(f"{prefix}.{col}", float(value), step=step)


def _metric_items(report: dict) -> dict[str, float]:
    out = {}
    for name, m in report["models"].items():
        for key in ("recall", "precision", "f1", "f2", "false_alarm_rate", "balanced_accuracy", "mcc",
                    "roc_auc", "average_precision", "threshold"):
            out[f"blind.{name}.{key}"] = float(m[key])
        for key, ci in m.get("ci95", {}).items():
            out[f"blind.{name}.{key}.ci_low"] = float(ci["low"])
            out[f"blind.{name}.{key}.ci_high"] = float(ci["high"])
    for key, value in report.get("hybrid_routing", {}).items():
        if isinstance(value, int | float):
            out[f"blind.hybrid.routing.{key}"] = float(value)
    for name, lat in report.get("cpu_latency", {}).items():
        out[f"cpu_latency.{name}.p50_ms"] = float(lat["p50_ms"])
        out[f"cpu_latency.{name}.p95_ms"] = float(lat["p95_ms"])
    return out


def register(params: dict, models_dir: Path, experiment: str, model_name: str, promote: bool,
             tolerance: float = 0.0) -> dict:
    import mlflow
    import mlflow.pyfunc
    from mlflow import MlflowClient

    from defectvision.registry.pyfunc_model import DefectVisionModel

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise RuntimeError("MLFLOW_TRACKING_URI n'est pas defini")
    manifest = read_json(models_dir / "manifest.json")
    blind = read_json(METRICS_DIR / "blind_test.json")
    version = manifest["version"]

    mlflow.set_experiment(experiment)
    client = MlflowClient()
    with mlflow.start_run(run_name=f"pipeline-{version}") as run:
        mlflow.set_tags({"defectvision.version": version, "git_commit": git_commit(),
                         "hybrid.strategy": params["hybrid"]["strategy"],
                         "autoencoder.architecture": params["autoencoder"]["architecture"]})
        mlflow.log_params(flatten(params))
        for name in ("supervised", "autoencoder"):
            meta = read_json(models_dir / name / "meta.json")
            mlflow.log_metrics({f"val.{name}.{k}": float(v) for k, v in meta["val_metrics"].items()
                                if isinstance(v, int | float)})
        hybrid_val = METRICS_DIR / "hybrid_val.json"
        if hybrid_val.exists():
            for strategy, m in read_json(hybrid_val)["val"].items():
                mlflow.log_metrics({f"val.hybrid_{strategy}.{k}": float(v) for k, v in m.items()
                                    if isinstance(v, int | float)})
        _log_history(mlflow, PLOTS_DIR / "supervised_history.csv", "train.supervised")
        _log_history(mlflow, PLOTS_DIR / "autoencoder_history.csv", "train.autoencoder")
        mlflow.log_metrics(_metric_items(blind))
        mlflow.log_metric("gates.passed", float(blind["gates"]["passed"]))

        if FIGURES_DIR.exists():
            mlflow.log_artifacts(str(FIGURES_DIR), "figures")
        mlflow.log_artifacts(str(METRICS_DIR), "metrics")
        for extra in (REPORTS_DIR / "split_report.json", REPORTS_DIR / "data_audit.json",
                      Path("dvc.lock"), Path("params.yaml")):
            if extra.exists():
                mlflow.log_artifact(str(extra), "pipeline")

        src = Path(__file__).resolve().parents[1]
        kwargs = {
            "python_model": DefectVisionModel(),
            "artifacts": {"models": str(models_dir)},
            "code_paths": [str(src)],
            "pip_requirements": pinned_requirements(),
        }
        if int(mlflow.__version__.split(".")[0]) >= 3:
            info = mlflow.pyfunc.log_model(name="bundle", **kwargs)
        else:
            info = mlflow.pyfunc.log_model(artifact_path="bundle", **kwargs)
        run_id = run.info.run_id

    mv = mlflow.register_model(info.model_uri, model_name)
    hybrid = blind["models"]["hybrid"]
    tags = {
        "defectvision.version": version,
        "run_id": run_id,
        "gates_passed": str(blind["gates"]["passed"]),
        f"blind_hybrid_{COMPARISON_METRIC}": f"{hybrid[COMPARISON_METRIC]:.6f}",
        "blind_hybrid_recall": f"{hybrid['recall']:.6f}",
        "blind_hybrid_precision": f"{hybrid['precision']:.6f}",
    }
    for key, value in tags.items():
        client.set_model_version_tag(model_name, mv.version, key, value)
    client.set_registered_model_alias(model_name, "challenger", mv.version)

    decision = {"registered_version": int(mv.version), "run_id": run_id, "bundle_version": version,
                "gates_passed": bool(blind["gates"]["passed"]), "promoted": False, "reason": "promotion non demandee"}
    if promote:
        try:
            champion = client.get_model_version_by_alias(model_name, "champion")
            champ_value = float(champion.tags.get(f"blind_hybrid_{COMPARISON_METRIC}", "-1"))
        except Exception:
            champion, champ_value = None, -1.0
        # Meme precision que l'etiquette stockee sur la version du champion (6 decimales).
        new_value = round(float(hybrid[COMPARISON_METRIC]), 6)
        if not blind["gates"]["passed"]:
            decision["reason"] = "conditions de qualite non remplies (gates)"
        elif new_value + tolerance < champ_value:
            decision["reason"] = f"F2 {new_value:.4f} inferieur au champion ({champ_value:.4f})"
        else:
            client.set_registered_model_alias(model_name, "champion", mv.version)
            decision.update(promoted=True, reason=f"F2 {new_value:.4f} >= champion ({champ_value:.4f})"
                            if champion else "premier champion")
        decision["previous_champion"] = int(champion.version) if champion else None
    return decision


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--models", default=str(MODELS_DIR))
    parser.add_argument("--experiment", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "defectvision"))
    parser.add_argument("--model-name", default=os.environ.get("MLFLOW_MODEL_NAME", "defectvision"))
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--report", default=str(REPORTS_DIR / "registry.json"))
    args = parser.parse_args(argv)
    decision = register(load_params(args.params), Path(args.models), args.experiment, args.model_name, args.promote)
    write_json(args.report, decision)
    print(f"[OK] version {decision['registered_version']} enregistree ; promue : {decision['promoted']} "
          f"({decision['reason']})")


if __name__ == "__main__":
    main()
