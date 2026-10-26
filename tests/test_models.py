"""Formes de sortie et comportements des modeles."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from defectvision.models.autoencoder import ConvAutoencoder, FeatureAutoencoder, reduce_maps, ssim_map
from defectvision.models.classifier import DefectClassifier
from defectvision.models.hybrid import HybridModel


def test_classifier_outputs_and_freezing():
    model = DefectClassifier(pretrained=False).eval()
    x = torch.randn(2, 3, 32, 32)
    assert model(x).shape == (2, 2)
    p = model.predict_proba(x)
    assert p.shape == (2,) and torch.all((p >= 0) & (p <= 1))
    model.freeze_backbone()
    head = sum(p.numel() for p in model.backbone.fc.parameters())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == head
    assert model.unfreeze_top_layers(2) > head


def test_conv_autoencoder_shapes():
    model = ConvAutoencoder(latent_channels=8).eval()
    x = torch.rand(2, 1, 64, 64)
    assert model(x).shape == x.shape
    maps = model.anomaly_map(x)
    assert maps.shape == (2, 1, 64, 64) and torch.all(maps >= 0)
    assert model.loss(x).ndim == 0


def test_feature_autoencoder_keeps_backbone_frozen_and_in_eval():
    model = FeatureAutoencoder(pretrained=False, hidden_channels=16, latent_channels=8)
    model.train()
    assert not model.stem.training and not model.layer3.training
    stats_before = model.stem[1].running_mean.clone()
    model.fit_normalization(torch.randn(2, 3, 32, 32) for _ in range(2))
    assert torch.equal(stats_before, model.stem[1].running_mean)
    x = torch.randn(2, 3, 32, 32)
    assert model.anomaly_map(x).shape == (2, 1, 32, 32)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith(("encoder", "decoder")) for n in trainable)


def test_ssim_identity_is_one():
    x = torch.rand(1, 1, 32, 32)
    assert torch.allclose(ssim_map(x, x), torch.ones_like(x), atol=1e-4)


@pytest.mark.parametrize("reduction", ["max", "topk_mean", "mean"])
def test_reduce_maps_orders_by_anomaly(reduction):
    maps = torch.zeros(2, 1, 32, 32)
    maps[1, 0, 10:14, 10:14] = 1.0
    scores = reduce_maps(maps, reduction, 0.05, 1.0)
    assert scores[1] > scores[0]


def _hybrid(strategy: str, **kw) -> HybridModel:
    base = dict(ae_center=1.0, ae_scale=0.5, supervised_threshold=0.3, autoencoder_threshold=2.0)
    base.update(kw)
    return HybridModel(strategy=strategy, **base)


def test_or_strategy_is_a_logical_or():
    model = _hybrid("or", threshold=0.5)
    p = np.array([0.1, 0.5, 0.1, 0.9])
    s = np.array([1.0, 1.0, 3.0, 3.0])
    d = model.decide(p, s)
    assert list(d["is_defect"]) == [False, True, True, True]
    assert list(d["source"]) == ["aucun", "supervise", "non_supervise", "les_deux"]
    # Une anomalie vue par le seul autoencodeur part en revue operateur.
    assert d["review"][2]


def test_review_uses_the_novelty_threshold():
    """Seule une anomalie forte, que le supervise ne voit pas, part en revue."""
    model = _hybrid("or", threshold=0.5, novelty_threshold=4.0, review_margin=0.0)
    d = model.decide(np.array([0.1, 0.1]), np.array([3.0, 5.0]))
    assert list(d["source"]) == ["non_supervise", "non_supervise"]
    assert list(d["review"]) == [False, True]


def test_stacking_probability_increases_with_each_score():
    model = _hybrid("stacking", coef_supervised=1.0, coef_autoencoder=0.8, intercept=-1.0)
    low = model.probability(np.array([0.2]), np.array([1.0]))
    assert model.probability(np.array([0.8]), np.array([1.0])) > low
    assert model.probability(np.array([0.2]), np.array([3.0])) > low


def test_hybrid_save_load_roundtrip(tmp_path):
    model = _hybrid("stacking", coef_supervised=1.2, coef_autoencoder=0.4, intercept=-0.3, threshold=0.42)
    model.save(tmp_path / "hybrid.json")
    assert HybridModel.load(tmp_path / "hybrid.json") == model
