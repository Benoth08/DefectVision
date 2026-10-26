"""Etape DVC 'split' : train / validation / blind test, sans fuite et stratifie.

Regles :
- l'unite de decoupage est le groupe de quasi-doublons (une piece ne peut pas
  se trouver dans deux decoupages) ;
- chaque classe est decoupee separement (stratification) ;
- l'ordre des groupes vient d'un hachage (sel + cle du groupe) : le blind test
  ne depend que du sel et de son ratio, il reste fige si l'on change le reste ;
- le train recoit une colonne 'weight' (inverse de la frequence de sa classe)
  qui sert au tirage equilibre : chaque lot d'entrainement est ~50/50, quel
  que soit le desequilibre du jeu.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from defectvision.config import INDEX_CSV, LABEL_NAMES, REPORTS_DIR, SPLITS_DIR, load_params
from defectvision.data.build_index import ORIGINALS
from defectvision.utils import sha256_file, write_json

COLUMNS = ["path", "label", "group", "sha256"]


def _rank(key: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:16], 16)


def _take(groups: pd.Series, order: list[str], target: int) -> list[str]:
    """Prend des groupes dans l'ordre jusqu'a atteindre target images (au plus pres)."""
    taken, total = [], 0
    for key in order:
        if total >= target:
            break
        size = int(groups[key])
        # On s'arrete si ajouter le groupe eloigne davantage de la cible.
        if total > 0 and abs(total + size - target) > abs(total - target):
            continue
        taken.append(key)
        total += size
    return taken


def assign_splits(df: pd.DataFrame, val_ratio: float, blind_ratio: float, salt: str) -> pd.Series:
    """Renvoie la colonne 'split' (train, val, blind_test) de chaque image."""
    if not 0 < val_ratio < 1 or not 0 < blind_ratio < 1 or val_ratio + blind_ratio >= 1:
        raise ValueError("ratios invalides")
    split = pd.Series("train", index=df.index)
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        sizes = sub.groupby("group").size()
        n = int(sizes.sum())
        blind_order = sorted(sizes.index, key=lambda g: _rank(g, f"{salt}:blind"))
        blind = set(_take(sizes, blind_order, round(blind_ratio * n)))
        rest = [g for g in sizes.index if g not in blind]
        val_order = sorted(rest, key=lambda g: _rank(g, f"{salt}:val"))
        val = set(_take(sizes, val_order, round(val_ratio * n)))
        split[sub.index[sub["group"].isin(blind)]] = "blind_test"
        split[sub.index[sub["group"].isin(val)]] = "val"
    return split


def balanced_weights(labels: pd.Series) -> pd.Series:
    """Poids n / (n_classes * n_c) : chaque classe pese autant dans le tirage."""
    counts = labels.value_counts()
    return labels.map(lambda c: len(labels) / (len(counts) * counts[c]))


def split_dataset(index: pd.DataFrame, val_ratio: float, blind_ratio: float, salt: str):
    data = index[index["source"] == ORIGINALS].copy()

    # Retire les groupes dont les images ont des etiquettes contradictoires.
    conflicts = data.groupby("group")["label"].nunique()
    conflicting = set(conflicts[conflicts > 1].index)
    excluded_conflicts = int(data["group"].isin(conflicting).sum())
    data = data[~data["group"].isin(conflicting)]

    # Un seul exemplaire par fichier identique (meme contenu binaire).
    before = len(data)
    data = data.drop_duplicates("sha256")
    excluded_duplicates = before - len(data)

    data["split"] = assign_splits(data, val_ratio, blind_ratio, salt)
    frames = {name: data[data["split"] == name][COLUMNS].sort_values("path") for name in ("train", "val", "blind_test")}
    frames["train"] = frames["train"].assign(weight=balanced_weights(frames["train"]["label"]).round(6))

    report = {
        "unit": "groupe de quasi-doublons",
        "salt": salt,
        "ratios": {"val": val_ratio, "blind_test": blind_ratio},
        "excluded": {"conflicting_label_groups": excluded_conflicts, "exact_duplicates": excluded_duplicates},
        "splits": {},
    }
    for name, frame in frames.items():
        counts = frame["label"].value_counts()
        entry = {
            "images": int(len(frame)),
            "groups": int(frame["group"].nunique()),
            **{LABEL_NAMES[c]: int(counts.get(c, 0)) for c in (0, 1)},
            "defect_ratio": float(frame["label"].mean()) if len(frame) else 0.0,
        }
        if name == "train":
            weight = frame.groupby("label")["weight"].sum()
            entry["sampler_share"] = {LABEL_NAMES[c]: float(weight.get(c, 0) / weight.sum()) for c in (0, 1)}
        report["splits"][name] = entry

    # Controles d'absence de fuite.
    groups = {name: set(f["group"]) for name, f in frames.items()}
    shas = {name: set(f["sha256"]) for name, f in frames.items()}
    pairs = [("train", "val"), ("train", "blind_test"), ("val", "blind_test")]
    report["checks"] = {
        "shared_groups": {f"{a}/{b}": len(groups[a] & groups[b]) for a, b in pairs},
        "shared_files": {f"{a}/{b}": len(shas[a] & shas[b]) for a, b in pairs},
    }
    if any(report["checks"]["shared_groups"].values()) or any(report["checks"]["shared_files"].values()):
        raise RuntimeError("fuite detectee entre decoupages")
    return frames, report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--index", default=str(INDEX_CSV))
    parser.add_argument("--out", default=str(SPLITS_DIR))
    parser.add_argument("--report", default=str(REPORTS_DIR / "split_report.json"))
    args = parser.parse_args(argv)

    params = load_params(args.params)["split"]
    index = pd.read_csv(args.index, keep_default_na=False)
    frames, report = split_dataset(
        index, float(params["val_ratio"]), float(params["blind_test_ratio"]), str(params["salt"])
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
        report["splits"][name]["sha256"] = sha256_file(out / f"{name}.csv")
    write_json(args.report, report)
    for name, entry in report["splits"].items():
        print(f"{name:10s} : {entry['images']:5d} images (OK={entry['OK']}, DEFAUT={entry['DEFAUT']})")


if __name__ == "__main__":
    main()
