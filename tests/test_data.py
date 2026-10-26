"""Pretraitement, hachage perceptuel, index et decoupage sans fuite."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from conftest import InProject, make_part
from defectvision.data.augment import augment
from defectvision.data.build_index import build_index, describe
from defectvision.data.hashing import (
    connected_groups,
    detail_correlation,
    detail_map,
    image_hashes,
    min_hamming,
    symmetric_min_hamming,
)
from defectvision.data.preprocessing import decode_image, preprocess_gray, to_gray
from defectvision.data.split import assign_splits, balanced_weights, split_dataset


def test_preprocess_shape_and_type():
    img = make_part(np.random.default_rng(1), 80)
    out = preprocess_gray(img, 32, clahe=True)
    assert out.shape == (32, 32) and out.dtype == np.uint8


def test_to_gray_accepts_color_and_alpha():
    bgr = np.dstack([make_part(np.random.default_rng(2), 40)] * 3)
    bgra = np.dstack([bgr, np.full(bgr.shape[:2], 255, np.uint8)])
    assert to_gray(bgr).shape == (40, 40)
    assert to_gray(bgra).shape == (40, 40)


def test_decode_image_roundtrip_and_errors():
    img = make_part(np.random.default_rng(3), 50)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    assert np.array_equal(decode_image(buf.tobytes()), img)
    with pytest.raises(ValueError):
        decode_image(b"pas une image")
    with pytest.raises(ValueError):
        decode_image(b"")


def test_augment_keeps_shape_and_is_reproducible():
    img = make_part(np.random.default_rng(4), 32)
    cfg = {"rotate_p": 1.0, "noise_p": 1.0, "blur_p": 1.0, "brightness_contrast_p": 1.0}
    a = augment(img, np.random.default_rng(7), cfg, "full")
    b = augment(img, np.random.default_rng(7), cfg, "full")
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert np.array_equal(augment(img, np.random.default_rng(7), cfg, "none"), img)


def test_hash_is_invariant_to_flips_and_rotations():
    rng = np.random.default_rng(5)
    img = make_part(rng, 64)
    variants = [np.fliplr(img), np.flipud(img), np.rot90(img, 1), np.rot90(img, 3)]
    query = np.stack([image_hashes(np.ascontiguousarray(v)) for v in variants])
    ref = image_hashes(img)[None, 0, :]
    assert min_hamming(query, ref).max() == 0
    other = image_hashes(make_part(rng, 64))
    assert min_hamming(other[None], ref)[0, 0] > 20


def test_connected_groups_links_transitively():
    dist = np.array([[0, 2, 50, 50], [2, 0, 3, 50], [50, 3, 0, 50], [50, 50, 50, 0]])
    roots = connected_groups(dist, max_distance=5)
    assert roots[0] == roots[1] == roots[2] != roots[3]


def test_symmetric_distance_matrix():
    rng = np.random.default_rng(6)
    hashes = np.stack([image_hashes(make_part(rng, 64)) for _ in range(5)])
    dist = symmetric_min_hamming(hashes)
    assert np.array_equal(dist, dist.T) and np.all(np.diag(dist) == 0)


def test_detail_correlation_recognises_the_same_part():
    rng = np.random.default_rng(8)
    img = make_part(rng, 128)
    same = detail_correlation(detail_map(np.rot90(img).copy(), 64), detail_map(img, 64))
    other = detail_correlation(detail_map(make_part(rng, 128), 64), detail_map(img, 64))
    assert same > 0.95 and other < 0.5


def test_file_names_do_not_decide_the_source(tmp_path):
    path = tmp_path / "casting_data" / "casting_data" / "train" / "def_front" / "cast_def_0_5121.jpeg"
    assert describe(path, tmp_path)["source"] == "augmented_300"
    path = tmp_path / "casting_512x512" / "casting_512x512" / "ok_front" / "cast_ok_0_1.jpeg"
    assert describe(path, tmp_path) == {"label": 0, "source": "originals_512", "kaggle_split": ""}


@pytest.fixture(scope="module")
def index_and_audit(synthetic_project):
    with InProject(synthetic_project):
        from pathlib import Path

        return build_index(Path("data/raw/casting"), synthetic_project, 16, 12, 20, 0.5)


def test_index_groups_duplicates_and_audits_kaggle_split(index_and_audit):
    index, audit = index_and_audit
    originals = index[index["source"] == "originals_512"]
    assert len(originals) == 36 + 2 + 44
    # Le miroir et la copie rejoignent le groupe de leur original.
    group = originals.set_index(originals["path"].str.rsplit("/", n=1).str[-1])["group"]
    assert group["cast_ok_0_0.jpeg"] == group["cast_ok_0_mirror.jpeg"] == group["cast_ok_0_copy.jpeg"]
    assert audit["originals"]["exact_duplicates"] == 1
    # Le jeu synthetique "augmente" a des copies tournees de part et d'autre de son decoupage : la fuite est vue.
    kaggle = audit["augmented"]["kaggle_split"]
    assert kaggle["test_with_near_duplicate_in_train"] > 0
    assert 0.0 <= audit["augmented"]["matched_ratio"] <= 1.0


def test_split_has_no_leak_and_is_stratified(index_and_audit):
    index, _ = index_and_audit
    frames, report = split_dataset(index, 0.2, 0.2, "sel-test")
    groups = {k: set(v["group"]) for k, v in frames.items()}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["blind_test"]
    assert not groups["val"] & groups["blind_test"]
    for name in ("val", "blind_test"):
        ratio = report["splits"][name]["defect_ratio"]
        assert abs(ratio - 44 / 81) < 0.12
    # Tirage equilibre : chaque classe pese 50 % dans le train.
    share = report["splits"]["train"]["sampler_share"]
    assert share["OK"] == pytest.approx(0.5) and share["DEFAUT"] == pytest.approx(0.5)


def test_blind_test_is_frozen_when_val_ratio_changes(index_and_audit):
    index, _ = index_and_audit
    a, _ = split_dataset(index, 0.15, 0.2, "sel-test")
    b, _ = split_dataset(index, 0.3, 0.2, "sel-test")
    assert list(a["blind_test"]["path"]) == list(b["blind_test"]["path"])
    c, _ = split_dataset(index, 0.15, 0.2, "autre-sel")
    assert list(a["blind_test"]["path"]) != list(c["blind_test"]["path"])


def test_assign_splits_rejects_bad_ratios():
    df = pd.DataFrame({"label": [0, 1], "group": ["a", "b"]})
    with pytest.raises(ValueError):
        assign_splits(df, 0.6, 0.5, "s")


def test_balanced_weights_sum_equally():
    labels = pd.Series([0] * 30 + [1] * 70)
    w = balanced_weights(labels)
    assert w[labels == 0].sum() == pytest.approx(w[labels == 1].sum())
