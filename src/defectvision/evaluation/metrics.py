"""Metriques insensibles au desequilibre, choix du seuil et intervalles de confiance.

Vocabulaire CND : rappel = POD (probabilite de detection) ; taux de fausses
alarmes = proportion de pieces saines rejetees a tort.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

THRESHOLD_FREE = ("roc_auc", "average_precision")


def confusion(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    tn = int((~y & ~pred).sum())
    fn = int((y & ~pred).sum())
    return tp, fp, tn, fn


def _div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def fbeta(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    return _div((1 + b2) * precision * recall, b2 * precision + recall)


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    """Metriques a un seuil donne, plus les metriques independantes du seuil."""
    y, score = np.asarray(y).astype(int), np.asarray(score, dtype=float)
    pred = score >= threshold
    tp, fp, tn, fn = confusion(y, pred)
    recall = _div(tp, tp + fn)
    precision = _div(tp, tp + fp)
    specificity = _div(tn, tn + fp)
    both = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold),
        "recall": recall,
        # Rappel demontre avec 95 % de confiance (borne inferieure de Clopper-Pearson).
        "recall_lower_95": float(recall_lower_bound(tp, tp + fn, 0.95)) if tp + fn else 0.0,
        "precision": precision,
        "specificity": specificity,
        "false_alarm_rate": _div(fp, fp + tn),
        "f1": fbeta(precision, recall, 1.0),
        "f2": fbeta(precision, recall, 2.0),
        "balanced_accuracy": (recall + specificity) / 2,
        "mcc": float(matthews_corrcoef(y, pred)) if both else 0.0,
        "accuracy": _div(tp + tn, len(y)),
        "roc_auc": float(roc_auc_score(y, score)) if both else float("nan"),
        "average_precision": float(average_precision_score(y, score)) if both else float("nan"),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": int(len(y)),
        "prevalence": float(y.mean()) if len(y) else 0.0,
    }


def recall_lower_bound(tp: np.ndarray | int, positives: int, confidence: float) -> np.ndarray:
    """Borne inferieure unilaterale de Clopper-Pearson du rappel (POD) au niveau de confiance donne."""
    tp = np.asarray(tp, dtype=float)
    alpha = 1.0 - confidence
    lower = beta.ppf(alpha, tp, positives - tp + 1)
    return np.where(tp > 0, np.nan_to_num(lower), 0.0)


def select_threshold(y: np.ndarray, score: np.ndarray, target_recall: float,
                     confidence: float | None = None) -> float:
    """Seuil qui maximise la precision sous une contrainte de rappel (POD).

    - confidence=None : rappel observe >= target_recall ;
    - confidence=0.95 : borne inferieure de Clopper-Pearson a 95 % du rappel
      >= target_recall (critere "POD 95/95" du controle non destructif) : le
      rappel vise est demontre malgre la taille limitee de l'echantillon.
      Si aucun seuil ne peut le demontrer (trop peu de defauts), on exige un
      rappel observe de 100 %.

    Les candidats sont les scores observes (pas de grille arbitraire). A
    precision egale, on garde le seuil le plus BAS : il detecte davantage de
    defauts (un defaut manque coute plus qu'une fausse alarme).
    """
    y, score = np.asarray(y).astype(int), np.asarray(score, dtype=float)
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("aucun defaut dans l'echantillon de calibration")
    order = np.argsort(-score, kind="mergesort")
    s_sorted, y_sorted = score[order], y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    # On ne garde que la derniere position de chaque valeur de score (seuils distincts).
    last = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    tp, fp, thresholds = tp[last], fp[last], s_sorted[last]
    precision = tp / (tp + fp)
    if confidence is None:
        ok = tp / positives >= target_recall - 1e-12
    else:
        ok = recall_lower_bound(tp, positives, confidence) >= target_recall - 1e-12
        if not ok.any():
            ok = tp == positives
    best = np.flatnonzero(ok & np.isclose(precision, precision[ok].max(), rtol=0, atol=1e-12))
    # thresholds est decroissant : le dernier indice est le seuil le plus bas.
    return float(thresholds[best[-1]])


def bootstrap_ci(
    y: np.ndarray,
    score: np.ndarray,
    threshold: float,
    n_samples: int = 2000,
    level: float = 0.95,
    seed: int = 42,
    keys: tuple[str, ...] = ("recall", "precision", "f2", "false_alarm_rate", "balanced_accuracy", "mcc",
                             "roc_auc", "average_precision"),
) -> dict[str, dict[str, float]]:
    """Intervalles de confiance par bootstrap stratifie (reechantillonnage dans chaque classe)."""
    y, score = np.asarray(y).astype(int), np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    values: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_samples):
        idx = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        m = binary_metrics(y[idx], score[idx], threshold)
        for k in keys:
            values[k].append(m[k])
    alpha = (1 - level) / 2
    return {
        k: {"low": float(np.quantile(v, alpha)), "high": float(np.quantile(v, 1 - alpha))}
        for k, v in values.items()
    }
