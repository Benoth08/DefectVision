"""Metriques, choix du seuil et intervalles de confiance."""

from __future__ import annotations

import numpy as np
import pytest

from defectvision.evaluation.metrics import binary_metrics, bootstrap_ci, recall_lower_bound, select_threshold


def brute_force_threshold(y, score, target):
    best_t, best_p = None, -1.0
    for t in sorted(set(score), reverse=True):
        m = binary_metrics(y, score, t)
        if m["recall"] >= target and m["precision"] > best_p:
            best_t, best_p = t, m["precision"]
    return best_t


@pytest.mark.parametrize("seed", range(5))
def test_select_threshold_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 120)
    score = np.round(np.clip(y * 0.4 + rng.normal(0.3, 0.2, 120), 0, 1), 2)  # avec ex aequo
    t = select_threshold(y, score, 0.95)
    m = binary_metrics(y, score, t)
    assert m["recall"] >= 0.95
    assert m["precision"] == pytest.approx(binary_metrics(y, score, brute_force_threshold(y, score, 0.95))["precision"])


def test_select_threshold_prefers_more_recall_at_equal_precision():
    # Precision 1 de 0.6 a 0.9 : le seuil 0.6 garde la precision et detecte tous les defauts.
    y = np.array([1] * 20 + [0] * 5)
    score = np.array([0.9] * 18 + [0.7, 0.6] + [0.1] * 5)
    t = select_threshold(y, score, 0.9)
    assert t == pytest.approx(0.6)
    assert binary_metrics(y, score, t)["recall"] == 1.0


def test_recall_lower_bound_matches_the_29_of_29_rule():
    # Regle classique du CND : 29 detections sur 29 demontrent une POD de 90 % a 95 % de confiance.
    assert float(recall_lower_bound(29, 29, 0.95)) == pytest.approx(0.05 ** (1 / 29))
    assert float(recall_lower_bound(29, 29, 0.95)) > 0.90 > float(recall_lower_bound(28, 29, 0.95))


def test_select_threshold_pod_95_95_is_more_demanding():
    rng = np.random.default_rng(3)
    y = np.array([1] * 120 + [0] * 80)
    score = np.where(y == 1, rng.normal(0.75, 0.12, 200), rng.normal(0.35, 0.12, 200))
    plain = select_threshold(y, score, 0.95)
    strict = select_threshold(y, score, 0.95, confidence=0.95)
    assert strict <= plain
    tp = int(((score >= strict) & (y == 1)).sum())
    assert float(recall_lower_bound(tp, 120, 0.95)) >= 0.95


def test_select_threshold_falls_back_to_full_recall_when_unprovable():
    y = np.array([1] * 5 + [0] * 5)
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.3, 0.5, 0.2, 0.1, 0.1, 0.1])
    t = select_threshold(y, score, 0.95, confidence=0.95)
    assert binary_metrics(y, score, t)["recall"] == 1.0


def test_binary_metrics_known_values():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1])
    m = binary_metrics(y, score, 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (3, 1, 5, 1)
    assert m["recall"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(0.75)
    assert m["false_alarm_rate"] == pytest.approx(1 / 6)
    assert m["balanced_accuracy"] == pytest.approx((0.75 + 5 / 6) / 2)


def test_bootstrap_is_reproducible_and_brackets_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    score = y * 0.5 + rng.normal(0.25, 0.2, 200)
    a = bootstrap_ci(y, score, 0.5, n_samples=200, seed=1)
    b = bootstrap_ci(y, score, 0.5, n_samples=200, seed=1)
    assert a == b
    recall = binary_metrics(y, score, 0.5)["recall"]
    assert a["recall"]["low"] <= recall <= a["recall"]["high"]
