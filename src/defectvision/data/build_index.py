"""Etape DVC 'index' : inventaire des images, groupes de quasi-doublons et audit de fuite.

Le jeu Kaggle contient deux ensembles :
- casting_512x512 : 1 300 photos originales, sans augmentation ;
- casting_data (300x300) : 7 348 images deja augmentees a partir des memes pieces,
  deja reparties par Kaggle en train/test.

Le projet decoupe les originaux (source de verite), par groupes de quasi-doublons.
L'ensemble augmente sert a mesurer la fuite qu'aurait provoquee le decoupage Kaggle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from defectvision.config import INDEX_CSV, RAW_DIR, REPORTS_DIR, load_params, project_root
from defectvision.data.hashing import (
    detail_correlation,
    detail_map,
    image_hashes,
    min_hamming,
    symmetric_min_hamming,
    union_find,
)
from defectvision.data.preprocessing import read_gray
from defectvision.utils import sha256_file, write_json

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".bmp"}
LABEL_FOLDERS = {"ok_front": 0, "def_front": 1}
ORIGINALS = "originals_512"
AUGMENTED = "augmented_300"


def describe(path: Path, raw_dir: Path) -> dict | None:
    """Etiquette et provenance d'une image, deduites de son chemin."""
    label = LABEL_FOLDERS.get(path.parent.name.lower())
    if label is None:
        return None
    # Dossiers seulement : des fichiers augmentes s'appellent par exemple cast_def_0_5121.jpeg.
    folders = [p.lower() for p in path.relative_to(raw_dir).parts[:-1]]
    if any(p.startswith("casting_512") for p in folders):
        source = ORIGINALS
    elif any(p.startswith("casting_data") for p in folders):
        source = AUGMENTED
    else:
        source = "other"
    kaggle_split = "train" if "train" in folders else "test" if "test" in folders else ""
    return {"label": label, "source": source, "kaggle_split": kaggle_split}


def discover(raw_dir: Path, root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        info = describe(path, raw_dir)
        if info is None:
            continue
        rows.append({"path": path.resolve().relative_to(root).as_posix(), **info})
    if not rows:
        raise RuntimeError(f"aucune image trouvee dans {raw_dir}")
    return pd.DataFrame(rows)


def hash_images(df: pd.DataFrame, root: Path, hash_size: int) -> tuple[pd.DataFrame, np.ndarray, dict[int, np.ndarray]]:
    """Empreintes de toutes les images, et cartes de detail fin des originaux."""
    hashes, shas, widths, heights, details = [], [], [], [], {}
    for pos, (rel, source) in enumerate(zip(df["path"], df["source"], strict=True)):
        gray = read_gray(root / rel)
        hashes.append(image_hashes(gray, hash_size))
        shas.append(sha256_file(root / rel))
        heights.append(gray.shape[0])
        widths.append(gray.shape[1])
        if source == ORIGINALS:
            details[pos] = detail_map(gray)
    df = df.assign(sha256=shas, width=widths, height=heights)
    return df, np.stack(hashes), details


def percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    qs = [0, 1, 5, 25, 50, 75, 95, 100]
    return {f"p{q}": float(np.percentile(values, q)) for q in qs}


def group_originals(
    df: pd.DataFrame,
    hashes: np.ndarray,
    details: list[np.ndarray],
    max_distance: int,
    candidate_distance: int,
    min_correlation: float,
) -> tuple[pd.DataFrame, dict]:
    """Groupes de photos d'une meme piece parmi les originaux, avec une cle stable (empreinte).

    Deux niveaux de lien :
    - sur : distance dHash <= max_distance (doublon visuel net) ;
    - verifie : distance <= candidate_distance ET correlation du detail fin
      >= min_correlation (meme texture de surface, donc probablement la meme piece).
    Les groupes sont les composantes connexes de ces liens.
    """
    dist = symmetric_min_hamming(hashes)
    upper = np.triu(np.ones_like(dist, dtype=bool), k=1)
    sure = np.argwhere(upper & (dist <= max_distance))
    candidates = np.argwhere(upper & (dist > max_distance) & (dist <= candidate_distance))
    verified = [(i, j) for i, j in candidates if detail_correlation(details[i], details[j]) >= min_correlation]
    roots = union_find(len(df), [*map(tuple, sure), *verified])
    masked = dist.astype(float)
    np.fill_diagonal(masked, np.inf)
    nn = masked.min(axis=1) if len(df) > 1 else np.array([np.inf])

    df = df.assign(_root=roots, nn_distance=nn)
    keys = df.groupby("_root")["sha256"].min().map(lambda s: f"g_{s[:12]}")
    df["group"] = df["_root"].map(keys)
    df = df.drop(columns="_root")

    sizes = df.groupby("group").size()
    labels_per_group = df.groupby("group")["label"].nunique()
    stats = {
        "images": int(len(df)),
        "links": {"sure": int(len(sure)), "candidates": int(len(candidates)), "verified": int(len(verified))},
        "groups": int(sizes.size),
        "groups_with_several_images": int((sizes > 1).sum()),
        "images_in_shared_groups": int(sizes[sizes > 1].sum()),
        "largest_group": int(sizes.max()),
        "group_sizes": {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
        "groups_with_conflicting_labels": int((labels_per_group > 1).sum()),
        "exact_duplicates": int(df["sha256"].duplicated().sum()),
        "nearest_neighbor_hamming": percentiles(nn[np.isfinite(nn)]),
    }
    return df, stats


def audit_augmented(
    df: pd.DataFrame,
    hashes: np.ndarray,
    originals: pd.DataFrame,
    original_hashes: np.ndarray,
    max_distance: int,
) -> tuple[pd.DataFrame, dict]:
    """Mesure la fuite du decoupage Kaggle (train/test des images augmentees).

    Le rattachement d'une image augmentee a son original est tente a titre
    d'information : les augmentations de Kaggle (rotations quelconques,
    recadrages) le rendent peu fiable, d'ou l'usage des originaux pour le
    decoupage du projet.
    """
    stats: dict = {"images": int(len(df))}
    if df.empty:
        return df, stats

    to_orig = min_hamming(hashes, original_hashes[:, 0, :])
    nearest = to_orig.argmin(axis=1)
    nearest_dist = to_orig[np.arange(len(df)), nearest]
    matched = nearest_dist <= max_distance
    df = df.assign(
        nn_distance=nearest_dist.astype(float),
        group=np.where(matched, originals["group"].to_numpy()[nearest], ""),
    )
    stats["matched_to_an_original"] = int(matched.sum())
    stats["matched_ratio"] = float(matched.mean())
    stats["distance_to_nearest_original"] = percentiles(nearest_dist)

    train_mask = (df["kaggle_split"] == "train").to_numpy()
    test_mask = (df["kaggle_split"] == "test").to_numpy()
    if train_mask.any() and test_mask.any():
        nn_test = min_hamming(hashes[test_mask], hashes[train_mask][:, 0, :]).min(axis=1)
        n_test = int(test_mask.sum())
        stats["kaggle_split"] = {
            "train_images": int(train_mask.sum()),
            "test_images": n_test,
            # Jumeau : meme image au pixel pres, a une symetrie pres (distance 0 sur 256 bits).
            "test_with_twin_in_train": int((nn_test == 0).sum()),
            "test_with_twin_in_train_ratio": float((nn_test == 0).mean()),
            "test_within_4_bits_of_train": int((nn_test <= 4).sum()),
            "test_within_4_bits_of_train_ratio": float((nn_test <= 4).mean()),
            "test_with_near_duplicate_in_train": int((nn_test <= max_distance).sum()),
            "test_with_near_duplicate_in_train_ratio": float((nn_test <= max_distance).mean()),
            "distance_test_to_nearest_train": percentiles(nn_test),
        }
    return df, stats


def build_index(raw_dir: Path, root: Path, hash_size: int, max_distance: int, candidate_distance: int,
                min_correlation: float) -> tuple[pd.DataFrame, dict]:
    df = discover(raw_dir, root)
    df, hashes, details = hash_images(df, root, hash_size)

    is_orig = (df["source"] == ORIGINALS).to_numpy()
    if not is_orig.any():
        raise RuntimeError("aucune image originale (dossier casting_512x512) trouvee")
    orig_positions = np.flatnonzero(is_orig)
    originals, orig_stats = group_originals(
        df[is_orig].reset_index(drop=True), hashes[is_orig], [details[p] for p in orig_positions],
        max_distance, candidate_distance, min_correlation,
    )

    is_aug = (df["source"] == AUGMENTED).to_numpy()
    augmented, aug_stats = audit_augmented(
        df[is_aug].reset_index(drop=True), hashes[is_aug], originals, hashes[is_orig], max_distance
    )

    index = pd.concat([originals, augmented], ignore_index=True)
    columns = ["path", "source", "kaggle_split", "label", "sha256", "width", "height", "group", "nn_distance"]
    index = index[columns].sort_values(["source", "path"]).reset_index(drop=True)

    counts = index.groupby(["source", "label"]).size().unstack(fill_value=0)
    audit = {
        "hash": {"type": "dHash", "bits": hash_size * hash_size, "symmetries": 8},
        "near_duplicate_max_hamming": max_distance,
        "verified_link": {"max_hamming": candidate_distance, "min_detail_correlation": min_correlation},
        "counts": {src: {"OK": int(row.get(0, 0)), "DEFAUT": int(row.get(1, 0))} for src, row in counts.iterrows()},
        "originals": orig_stats,
        "augmented": aug_stats,
    }
    return index, audit


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--raw", default=str(RAW_DIR))
    parser.add_argument("--out", default=str(INDEX_CSV))
    parser.add_argument("--report", default=str(REPORTS_DIR / "data_audit.json"))
    args = parser.parse_args(argv)

    params = load_params(args.params)["data"]
    index, audit = build_index(
        Path(args.raw), project_root(), int(params["hash_size"]), int(params["near_duplicate_max_hamming"]),
        int(params["verified_link_max_hamming"]), float(params["verified_link_min_detail_correlation"]),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.out, index=False, lineterminator="\n")
    write_json(args.report, audit)
    orig = audit["originals"]
    print(f"[OK] {len(index)} images indexees ; originaux : {orig['images']} images, {orig['groups']} groupes "
          f"(liens surs={orig['links']['sure']}, verifies={orig['links']['verified']}/{orig['links']['candidates']})")
    kaggle = audit["augmented"].get("kaggle_split")
    if kaggle:
        print(f"Audit du decoupage Kaggle : {kaggle['test_with_twin_in_train_ratio']:.1%} des images de test ont "
              f"un jumeau dans le train, {kaggle['test_with_near_duplicate_in_train_ratio']:.1%} un quasi-doublon")


if __name__ == "__main__":
    main()
