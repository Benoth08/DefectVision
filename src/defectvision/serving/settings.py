"""Reglages du service, lus dans les variables d'environnement."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # auto : registre MLflow (alias champion) puis, a defaut, dossier local.
    model_source: str = field(default_factory=lambda: _env("MODEL_SOURCE", "auto"))
    model_dir: Path = field(default_factory=lambda: Path(_env("MODEL_DIR", "/models")))
    mlflow_model_name: str = field(default_factory=lambda: _env("MLFLOW_MODEL_NAME", "defectvision"))
    mlflow_model_alias: str = field(default_factory=lambda: _env("MLFLOW_MODEL_ALIAS", "champion"))
    cache_dir: Path = field(default_factory=lambda: Path(_env("MODEL_CACHE_DIR", "/tmp/defectvision-models")))
    admin_token: str = field(default_factory=lambda: _env("ADMIN_TOKEN", ""))
    max_upload_bytes: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))))
    max_batch_files: int = field(default_factory=lambda: int(_env("MAX_BATCH_FILES", "16")))
    root_path: str = field(default_factory=lambda: _env("ROOT_PATH", ""))
    device: str = field(default_factory=lambda: _env("DEVICE", "cpu"))

    @property
    def mlflow_enabled(self) -> bool:
        return self.model_source in ("auto", "mlflow") and bool(os.environ.get("MLFLOW_TRACKING_URI"))
