"""Jeu de donnees synthetique et pipeline complet en miniature (CPU, quelques secondes).

Les images imitent la structure du jeu Kaggle : originaux dans
casting_512x512/, images augmentees dans casting_data/{train,test}/.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

PROJECT = Path(__file__).resolve().parents[1]


def make_part(rng: np.random.Generator, size: int = 64, defect: bool = False) -> np.ndarray:
    """Disque clair (la piece) sur fond sombre, texture aleatoire, trous sombres si defaut."""
    img = np.zeros((size, size), np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    img += rng.uniform(20, 60) + rng.uniform(-20, 20) * (xx / size) + rng.uniform(-20, 20) * (yy / size)
    center = (size // 2 + int(rng.integers(-3, 4)), size // 2 + int(rng.integers(-3, 4)))
    radius = int(size * rng.uniform(0.3, 0.4))
    mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2
    img[mask] = rng.uniform(140, 190)
    for _ in range(8):
        x0, y0 = (int(v) for v in rng.integers(0, size, 2))
        cv2.rectangle(img, (x0, y0), (x0 + int(rng.integers(3, 10)), y0 + int(rng.integers(3, 10))),
                      float(rng.uniform(60, 220)), -1)
    img += rng.normal(0, 4, img.shape)
    if defect:
        for _ in range(int(rng.integers(2, 4))):
            pos = (center[0] + int(rng.integers(-radius // 2, radius // 2 + 1)),
                   center[1] + int(rng.integers(-radius // 2, radius // 2 + 1)))
            cv2.circle(img, pos, max(2, size // 14), float(rng.uniform(0, 25)), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpeg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    path.write_bytes(buf.tobytes())


def build_synthetic_raw(raw: Path, n_ok: int = 36, n_def: int = 44, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    orig_dir = raw / "casting_512x512" / "casting_512x512"
    originals = {0: [], 1: []}
    for label, n, folder, prefix in ((0, n_ok, "ok_front", "cast_ok"), (1, n_def, "def_front", "cast_def")):
        for i in range(n):
            img = make_part(rng, 64, defect=bool(label))
            write_image(orig_dir / folder / f"{prefix}_0_{i}.jpeg", img)
            originals[label].append(img)
    # Quasi-doublon (miroir) et doublon exact : doivent rester dans le meme decoupage.
    write_image(orig_dir / "ok_front" / "cast_ok_0_mirror.jpeg", np.fliplr(originals[0][0]))
    write_image(orig_dir / "ok_front" / "cast_ok_0_copy.jpeg", originals[0][0])

    # Jeu "augmente" 300x300 (ici 48x48) : test et train partagent des pieces d'origine.
    aug_dir = raw / "casting_data" / "casting_data"
    for label, folder in ((0, "ok_front"), (1, "def_front")):
        for i, img in enumerate(originals[label][:12]):
            small = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
            split = "test" if i % 3 == 0 else "train"
            write_image(aug_dir / split / folder / f"aug_{label}_{i}.jpeg", np.rot90(small, i % 4))
            if split == "test":
                write_image(aug_dir / "train" / folder / f"aug_{label}_{i}_bis.jpeg", small)
    return {"n_ok": n_ok + 2, "n_def": n_def}


def small_params() -> dict:
    """params.yaml du projet, reduit pour des tests rapides sur CPU."""
    with open(PROJECT / "params.yaml", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    params = copy.deepcopy(params)
    params["supervised"].update(image_size=32, pretrained=False, batch_size=8, epochs_phase1=1,
                                epochs_phase2=1, patience=5)
    params["autoencoder"].update(image_size=32, pretrained=False, batch_size=8, epochs=2, patience=5)
    params["evaluation"]["bootstrap_samples"] = 50
    params["runtime"].update(num_workers=0, device="cpu")
    return params


@pytest.fixture(scope="session")
def synthetic_project(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("project")
    build_synthetic_raw(root / "data" / "raw" / "casting")
    with open(root / "params.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(small_params(), handle, sort_keys=False)
    return root


class InProject:
    """Se place a la racine du projet synthetique le temps d'un bloc (chemins relatifs du pipeline)."""

    def __init__(self, root: Path):
        self.root = root

    def __enter__(self):
        self.previous_cwd = os.getcwd()
        self.previous_env = os.environ.get("DEFECTVISION_ROOT")
        os.chdir(self.root)
        os.environ["DEFECTVISION_ROOT"] = str(self.root)
        return self

    def __exit__(self, *exc):
        os.chdir(self.previous_cwd)
        if self.previous_env is None:
            os.environ.pop("DEFECTVISION_ROOT", None)
        else:
            os.environ["DEFECTVISION_ROOT"] = self.previous_env


@pytest.fixture(scope="session")
def trained_project(synthetic_project: Path) -> Path:
    """Execute toutes les etapes du pipeline (sauf le telechargement) via leurs points d'entree."""
    from defectvision.data import build_index, split
    from defectvision.evaluation import evaluate
    from defectvision.training import fit_hybrid, train_autoencoder, train_supervised

    with InProject(synthetic_project):
        build_index.main([])
        split.main([])
        train_supervised.main([])
        train_autoencoder.main([])
        fit_hybrid.main([])
        evaluate.main([])
    return synthetic_project
