"""Test de fumee de la pile deployee : envoie des images d'un decoupage a l'API et resume les reponses.

    docker compose run --rm trainer python scripts/smoke_test.py --url http://nginx/api --n 60

Verifie que chaque reponse est valide, compare la decision a l'etiquette connue et mesure la latence vue
par le client. Sert aussi a alimenter les tableaux de bord Grafana.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import requests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://nginx/api")
    parser.add_argument("--split", default="data/splits/blind_test.csv")
    parser.add_argument("--n", type=int, default=60, help="nombre d'images par modele")
    parser.add_argument("--models", default="hybrid,supervised,autoencoder")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    health = requests.get(f"{args.url}/ready", timeout=10)
    print(f"/ready -> {health.status_code} {health.json()}")
    if health.status_code != 200:
        return 1

    frame = pd.read_csv(args.split).sample(frac=1.0, random_state=args.seed).head(args.n)
    failures = 0
    for model in args.models.split(","):
        correct, latencies = 0, []
        for path, label in zip(frame["path"], frame["label"], strict=True):
            start = time.perf_counter()
            with open(path, "rb") as handle:
                response = requests.post(f"{args.url}/predict", params={"model": model},
                                         files={"file": (Path(path).name, handle, "image/jpeg")}, timeout=60)
            latencies.append((time.perf_counter() - start) * 1000)
            if response.status_code != 200:
                failures += 1
                print(f"  {model} {path}: HTTP {response.status_code} {response.text[:200]}")
                continue
            correct += int(response.json()["is_defect"] == bool(label))
        print(f"{model:12s} | {len(frame)} images | justes={correct} ({correct / len(frame):.1%}) | "
              f"latence client mediane={statistics.median(latencies):.0f} ms")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
