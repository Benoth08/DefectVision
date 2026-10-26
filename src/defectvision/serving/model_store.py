"""Chargement et rechargement a chaud du jeu de modeles servi par l'API."""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path

from defectvision.inference.predictor import Predictor
from defectvision.serving import monitoring
from defectvision.serving.settings import Settings

log = logging.getLogger("defectvision.serving")


def _find_models_dir(root: Path) -> Path:
    """Le registre MLflow range le dossier models/ sous artifacts/ : on cherche son manifeste."""
    for manifest in sorted(root.rglob("manifest.json")):
        if (manifest.parent / "supervised" / "meta.json").exists():
            return manifest.parent
    raise FileNotFoundError(f"aucun jeu de modeles dans {root}")


class ModelStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.predictor: Predictor | None = None
        self.source: str | None = None
        self.last_error: str | None = None
        self._lock = threading.Lock()

    def _from_mlflow(self) -> tuple[Predictor, str]:
        import mlflow
        from mlflow import MlflowClient

        s = self.settings
        mv = MlflowClient().get_model_version_by_alias(s.mlflow_model_name, s.mlflow_model_alias)
        target = s.cache_dir / f"v{mv.version}"
        if not target.exists():
            tmp = s.cache_dir / f".v{mv.version}.part"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{s.mlflow_model_name}/{mv.version}", dst_path=str(tmp)
            )
            tmp.rename(target)
        predictor = Predictor(_find_models_dir(target), device=s.device)
        return predictor, f"mlflow:{s.mlflow_model_name}@{s.mlflow_model_alias} (v{mv.version})"

    def _from_local(self) -> tuple[Predictor, str]:
        return Predictor(self.settings.model_dir, device=self.settings.device), f"local:{self.settings.model_dir}"

    def load(self) -> bool:
        """Charge un jeu de modeles ; garde l'ancien si le nouveau echoue."""
        attempts = []
        if self.settings.mlflow_enabled:
            attempts.append(self._from_mlflow)
        if self.settings.model_source in ("auto", "local"):
            attempts.append(self._from_local)
        errors = []
        for attempt in attempts:
            try:
                predictor, source = attempt()
            except Exception as exc:  # le service doit demarrer meme sans modele
                errors.append(f"{attempt.__name__}: {exc}")
                log.warning("chargement impossible (%s) : %s", attempt.__name__, exc)
                continue
            with self._lock:
                self.predictor, self.source, self.last_error = predictor, source, None
            monitoring.set_model_info(predictor.version, source)
            monitoring.MODEL_RELOADS.labels("ok").inc()
            log.info("jeu de modeles %s charge depuis %s", predictor.version, source)
            return True
        self.last_error = " ; ".join(errors) or "aucune source de modeles configuree"
        monitoring.MODEL_RELOADS.labels("erreur").inc()
        if self.predictor is None:
            monitoring.set_model_info(None, None)
        return False
