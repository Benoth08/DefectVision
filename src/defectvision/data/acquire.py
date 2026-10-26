"""Etape DVC 'acquire' : telecharge et extrait le jeu Casting Product (Kaggle).

Le telechargement public de Kaggle ne demande pas de compte. L'empreinte
SHA-256 de l'archive est ecrite dans le rapport ; si params.yaml en fixe une
(data.archive_sha256), elle est verifiee.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from defectvision.config import RAW_DIR, REPORTS_DIR, load_params
from defectvision.utils import write_json

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".bmp"}


def download(url: str, destination: Path, chunk_size: int = 1 << 20) -> str:
    """Telecharge url dans destination et renvoie le SHA-256 du fichier."""
    request = urllib.request.Request(url, headers={"User-Agent": "defectvision/1.0"})
    digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=120) as response, open(destination, "wb") as out:
        for chunk in iter(lambda: response.read(chunk_size), b""):
            out.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, out_dir: Path) -> int:
    """Extrait l'archive en refusant tout chemin qui sortirait de out_dir."""
    out_dir = out_dir.resolve()
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (out_dir / member.filename).resolve()
            if out_dir not in target.parents and target != out_dir:
                raise ValueError(f"chemin refuse dans l'archive : {member.filename}")
            zf.extract(member, out_dir)
            count += 1
    return count


def summarize(raw_dir: Path) -> dict[str, int]:
    """Compte les images par dossier (chemin relatif a raw_dir)."""
    counts: Counter[str] = Counter()
    for path in raw_dir.rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            counts[path.parent.relative_to(raw_dir).as_posix()] += 1
    return dict(sorted(counts.items()))


def acquire(url: str, raw_dir: Path, expected_sha256: str = "") -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "archive.zip"
        print(f"Telechargement : {url}")
        sha = download(url, archive)
        size = archive.stat().st_size
        print(f"Archive : {size / 1e6:.1f} Mo, sha256={sha}")
        if expected_sha256 and sha != expected_sha256:
            raise RuntimeError(f"empreinte inattendue : {sha} (attendue {expected_sha256})")
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        raw_dir.mkdir(parents=True)
        n_members = safe_extract(archive, raw_dir)
    folders = summarize(raw_dir)
    return {
        "url": url,
        "archive_sha256": sha,
        "archive_bytes": size,
        "members": n_members,
        "images_per_folder": folders,
        "images_total": sum(folders.values()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--out", default=str(RAW_DIR))
    parser.add_argument("--report", default=str(REPORTS_DIR / "acquire.json"))
    args = parser.parse_args(argv)

    params = load_params(args.params)["data"]
    report = acquire(params["download_url"], Path(args.out), params.get("archive_sha256", ""))
    write_json(args.report, report)
    print(f"[OK] {report['images_total']} images extraites dans {args.out}")


if __name__ == "__main__":
    main()
