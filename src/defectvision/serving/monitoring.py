"""Metriques Prometheus exposees sur /metrics.

Trois familles :
- service : requetes, codes de retour, latence ;
- modele : predictions par classe, distribution des scores, revues operateur,
  branche declenchante du modele hybride ;
- donnees d'entree : luminosite et contraste moyens (derive de l'eclairage ou
  de la camera), taille des images.
"""

from __future__ import annotations

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5, 5.0)
PROB_BUCKETS = tuple(round(x, 2) for x in np.linspace(0.05, 1.0, 20))
Z_BUCKETS = (-3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 50)
INTENSITY_BUCKETS = tuple(range(16, 257, 16))

HTTP_REQUESTS = Counter(
    "defectvision_http_requests_total", "Requetes HTTP recues", ["method", "route", "status"]
)
HTTP_LATENCY = Histogram(
    "defectvision_http_request_duration_seconds", "Duree de traitement des requetes HTTP", ["route"],
    buckets=LATENCY_BUCKETS,
)
INFERENCE_LATENCY = Histogram(
    "defectvision_inference_duration_seconds", "Duree d'inference par image", ["model"], buckets=LATENCY_BUCKETS
)
PREDICTIONS = Counter("defectvision_predictions_total", "Predictions rendues", ["model", "label"])
SCORES = Histogram(
    "defectvision_score", "Score de decision (probabilite de defaut)", ["model"], buckets=PROB_BUCKETS
)
ANOMALY_Z = Histogram(
    "defectvision_anomaly_z", "Score d'anomalie normalise (z) de l'autoencodeur", buckets=Z_BUCKETS
)
REVIEWS = Counter("defectvision_review_required_total", "Pieces renvoyees a l'operateur", ["model"])
HYBRID_SOURCE = Counter(
    "defectvision_hybrid_source_total", "Branche ayant declenche l'alerte du modele hybride", ["source"]
)
PREDICTION_ERRORS = Counter("defectvision_prediction_errors_total", "Requetes de prediction refusees", ["reason"])
INPUT_MEAN = Histogram(
    "defectvision_input_mean_intensity", "Luminosite moyenne des images recues (0-255)", buckets=INTENSITY_BUCKETS
)
INPUT_CONTRAST = Histogram(
    "defectvision_input_contrast", "Ecart-type des niveaux de gris des images recues",
    buckets=(8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128),
)
MODEL_LOADED = Gauge("defectvision_model_loaded", "1 si un jeu de modeles est charge")
MODEL_INFO = Gauge("defectvision_model_info", "Jeu de modeles en service", ["version", "source"])
MODEL_RELOADS = Counter("defectvision_model_reloads_total", "Rechargements du jeu de modeles", ["result"])


def observe_input(gray: np.ndarray) -> None:
    INPUT_MEAN.observe(float(gray.mean()))
    INPUT_CONTRAST.observe(float(gray.std()))


def observe_prediction(result: dict) -> None:
    model = result["model"]
    PREDICTIONS.labels(model, result["label"]).inc()
    INFERENCE_LATENCY.labels(model).observe(result["inference_ms"] / 1000)
    details = result["details"]
    if model != "autoencoder":
        SCORES.labels(model).observe(result["score"])
    if "anomaly_z" in details:
        ANOMALY_Z.observe(details["anomaly_z"])
    if result.get("review_required"):
        REVIEWS.labels(model).inc()
    if "source" in details:
        HYBRID_SOURCE.labels(details["source"]).inc()


def set_model_info(version: str | None, source: str | None) -> None:
    MODEL_INFO.clear()
    if version:
        MODEL_INFO.labels(version, source or "inconnue").set(1)
        MODEL_LOADED.set(1)
    else:
        MODEL_LOADED.set(0)
