"""Modele hybride : combine le score supervise et le score d'anomalie.

Idee : le modele supervise reconnait bien les defauts qu'il a deja vus ;
l'autoencodeur, entraine sur des pieces saines, signale tout ecart a la
normale, y compris un type de defaut jamais rencontre. Le modele hybride
fusionne les deux scores :

- strategie 'stacking' (par defaut) : regression logistique sur
  [logit(p_supervise), z_anomalie], apprise sur la validation ;
- strategie 'or' : defaut des que l'une des deux branches depasse son seuil.

Chaque decision indique aussi quelle branche l'a declenchee et si la piece
doit etre revue par un operateur (anomalie vue par l'autoencodeur seul, ou
score proche du seuil).

Ce module n'utilise que NumPy : le service d'inference n'a pas besoin de
scikit-learn.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from defectvision.utils import read_json, write_json

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


@dataclass
class HybridModel:
    strategy: str
    # Normalisation robuste du score d'anomalie (mediane et dispersion des pieces saines).
    ae_center: float
    ae_scale: float
    # Seuils propres a chaque branche (choisis sur la validation).
    supervised_threshold: float
    autoencoder_threshold: float
    # Seuil de la probabilite fusionnee.
    threshold: float = 0.5
    # Coefficients de la regression logistique (strategie 'stacking').
    coef_supervised: float = 0.0
    coef_autoencoder: float = 0.0
    intercept: float = 0.0
    review_margin: float = 0.1
    # Seuil de nouveaute : anomalie forte (centile eleve des pieces saines de la validation).
    # Une piece au-dessus, que le supervise ne signale pas, part en revue operateur.
    novelty_threshold: float | None = None
    extra: dict = field(default_factory=dict)

    def ae_z(self, s_ae: np.ndarray) -> np.ndarray:
        return (np.asarray(s_ae, dtype=float) - self.ae_center) / max(self.ae_scale, EPS)

    def probability(self, p_sup: np.ndarray, s_ae: np.ndarray) -> np.ndarray:
        """Probabilite fusionnee de defaut."""
        if self.strategy == "stacking":
            x = self.coef_supervised * logit(p_sup) + self.coef_autoencoder * self.ae_z(s_ae) + self.intercept
            return sigmoid(x)
        if self.strategy == "or":
            # 0.5 exactement quand une branche est a son seuil : seuil 0.5 <=> regle OU.
            a = logit(p_sup) - logit(self.supervised_threshold)
            b = self.ae_z(s_ae) - self.ae_z(self.autoencoder_threshold)
            return sigmoid(np.maximum(a, b))
        raise ValueError(f"strategie inconnue : {self.strategy}")

    def decide(self, p_sup: np.ndarray, s_ae: np.ndarray) -> dict[str, np.ndarray]:
        p_sup = np.atleast_1d(np.asarray(p_sup, dtype=float))
        s_ae = np.atleast_1d(np.asarray(s_ae, dtype=float))
        prob = self.probability(p_sup, s_ae)
        is_defect = prob >= self.threshold
        sup_flag = p_sup >= self.supervised_threshold
        ae_flag = s_ae >= self.autoencoder_threshold
        source = np.select(
            [sup_flag & ae_flag, sup_flag, ae_flag], ["les_deux", "supervise", "non_supervise"], "aucun"
        )
        # Revue operateur : anomalie forte que le supervise ne voit pas (defaut peut-etre
        # d'un type inconnu) ou decision trop proche du seuil.
        novel = s_ae >= (self.novelty_threshold if self.novelty_threshold is not None else self.autoencoder_threshold)
        near = np.abs(prob - self.threshold) < self.review_margin
        review = (novel & ~sup_flag) | near
        return {"probability": prob, "is_defect": is_defect, "source": source, "review": review, "novel": novel}

    def save(self, path: str | Path) -> None:
        write_json(path, asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> HybridModel:
        return cls(**read_json(path))
