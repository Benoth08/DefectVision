"""Schemas des reponses de l'API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ModelName = Literal["supervised", "autoencoder", "hybrid"]


class Prediction(BaseModel):
    filename: str | None = None
    model: ModelName
    label: Literal["OK", "DEFAUT"]
    is_defect: bool
    score: float = Field(description="Probabilite de defaut (supervise, hybride) ou score d'anomalie (autoencodeur)")
    threshold: float = Field(description="Seuil de decision, choisi sur la validation")
    review_required: bool | None = Field(
        None, description="Modele hybride : piece a faire revoir par un operateur"
    )
    details: dict[str, float | str]
    model_version: str
    inference_ms: float
    heatmaps: dict[str, str] | None = Field(None, description="Cartes de chaleur PNG encodees en base64")


class BatchPrediction(BaseModel):
    model: ModelName
    count: int
    defects: int
    predictions: list[Prediction]


class Health(BaseModel):
    status: str


class Readiness(BaseModel):
    ready: bool
    model_version: str | None
    source: str | None
    error: str | None = None


class ReloadResult(BaseModel):
    reloaded: bool
    model_version: str | None
    source: str | None
    error: str | None = None
