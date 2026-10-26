"""Hachage perceptuel (dHash) insensible aux symetries, pour reperer les quasi-doublons.

Une meme piece peut apparaitre plusieurs fois (retournee, tournee de 90 degres,
legerement retouchee). Si ces copies tombent dans des decoupages differents,
le score du test est trop optimiste (fuite de donnees). On compare donc chaque
image aux autres sous ses 8 symetries du carre (4 rotations x miroir).
"""

from __future__ import annotations

import cv2
import numpy as np


def dhash_bits(gray: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """dHash : signe du gradient horizontal sur une vignette (hash_size+1) x hash_size."""
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).ravel()


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """Range un vecteur de bits dans des mots uint64 (complete par des zeros)."""
    n_words = -(-bits.size // 64)
    padded = np.zeros(n_words * 64, dtype=np.uint8)
    padded[: bits.size] = bits
    return np.packbits(padded).view(">u8").astype(np.uint64)


def dihedral_variants(gray: np.ndarray) -> list[np.ndarray]:
    """Les 8 symetries du carre : rotations de 0, 90, 180, 270 degres, avec et sans miroir."""
    variants = []
    for flipped in (gray, np.fliplr(gray)):
        for k in range(4):
            variants.append(np.ascontiguousarray(np.rot90(flipped, k)))
    return variants


def image_hashes(gray: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """Empreintes des 8 symetries : tableau (8, n_mots) uint64. La ligne 0 est l'image telle quelle."""
    return np.stack([pack_bits(dhash_bits(v, hash_size)) for v in dihedral_variants(gray)])


def min_hamming(query: np.ndarray, reference: np.ndarray, chunk: int | None = None) -> np.ndarray:
    """Distance de Hamming minimale sur les 8 symetries.

    query : (N, 8, W) empreintes des symetries des images requetes ;
    reference : (M, W) empreintes canoniques (ligne 0) des images de reference.
    Renvoie une matrice (N, M) d'entiers. Le calcul avance par blocs de lignes
    pour rester sous ~160 Mo de memoire.
    """
    n, m = query.shape[0], reference.shape[0]
    if chunk is None:
        chunk = max(1, int(2e7 // (query.shape[1] * m * query.shape[2])))
    out = np.empty((n, m), dtype=np.int32)
    for start in range(0, n, chunk):
        block = query[start : start + chunk]
        xor = block[:, :, None, :] ^ reference[None, None, :, :]
        dist = np.bitwise_count(xor).sum(axis=-1, dtype=np.int32)
        out[start : start + chunk] = dist.min(axis=1)
    return out


def symmetric_min_hamming(hashes: np.ndarray, chunk: int | None = None) -> np.ndarray:
    """Distances entre toutes les paires d'un meme ensemble (matrice symetrique, diagonale = 0)."""
    dist = min_hamming(hashes, hashes[:, 0, :], chunk=chunk)
    dist = np.minimum(dist, dist.T)
    np.fill_diagonal(dist, 0)
    return dist


def union_find(n: int, pairs) -> np.ndarray:
    """Composantes connexes d'un graphe a n sommets donne par ses aretes (i, j).

    Renvoie pour chaque element l'indice de la racine de son groupe.
    """
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


def connected_groups(dist: np.ndarray, max_distance: int) -> np.ndarray:
    """Composantes connexes du graphe 'distance <= max_distance'."""
    rows, cols = np.nonzero(np.triu(dist <= max_distance, k=1))
    return union_find(dist.shape[0], zip(rows.tolist(), cols.tolist(), strict=True))


def detail_map(gray: np.ndarray, size: int = 256, sigma: float = 3.0) -> np.ndarray:
    """Detail fin normalise (passe-haut) : la texture de surface propre a chaque piece.

    La forme generale de la roue est commune a toutes les pieces ; seul le
    detail fin (grain de fonderie, rayures, taches) distingue deux pieces.
    """
    f = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    f -= cv2.GaussianBlur(f, (0, 0), sigma)
    return ((f - f.mean()) / (f.std() + 1e-6)).astype(np.float32)


def detail_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation maximale entre deux cartes de detail, sur les 8 symetries de la premiere."""
    return max(float((v * b).mean()) for v in dihedral_variants(a))
