"""Figures des rapports et du README (PNG, fond clair, palette accessible).

Conventions graphiques :
- classes : OK en gris (contexte), Defaut en rouge (ce que l'on cherche) ;
- modeles : bleu (supervise), orange (autoencodeur), aqua (hybride) ;
- traits fins, grilles discretes, pas de double axe, legende des qu'il y a
  plusieurs series ; cartes de chaleur en 'inferno' avec barre d'echelle.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from defectvision.data.preprocessing import apply_clahe, read_gray  # noqa: E402
from defectvision.models.hybrid import logit  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CLASS_COLORS = {0: "#898781", 1: "#e34948"}
CLASS_LABELS = {0: "OK", 1: "Défaut"}
MODEL_COLORS = {"supervised": "#2a78d6", "autoencoder": "#eb6834", "hybrid": "#1baf7a"}
MODEL_LABELS = {
    "supervised": "Supervisé (ResNet-18)",
    "autoencoder": "Non supervisé (autoencodeur)",
    "hybrid": "Hybride",
}
SEQUENTIAL = LinearSegmentedColormap.from_list("dv_blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.titleweight": "bold",
}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _suptitle(fig, title: str, subtitle: str | None = None) -> None:
    """Titre et sous-titre au-dessus de la figure, a distance fixe en points (quelle que soit sa hauteur)."""
    from matplotlib.transforms import offset_copy

    sub_offset = 6 if subtitle else 0
    title_tr = offset_copy(fig.transFigure, fig=fig, x=0, y=sub_offset + (18 if subtitle else 8), units="points")
    fig.text(0.01, 1.0, title, ha="left", va="bottom", color=INK, fontsize=13, fontweight="bold",
             transform=title_tr)
    if subtitle:
        sub_tr = offset_copy(fig.transFigure, fig=fig, x=0, y=sub_offset, units="points")
        fig.text(0.01, 1.0, subtitle, ha="left", va="bottom", color=INK_2, fontsize=9.5, transform=sub_tr)


def _image_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- donnees


def dataset_examples(frame: pd.DataFrame, root: Path, path: Path, n: int = 5, seed: int = 0) -> None:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6), layout="constrained")
        for row, label in enumerate((0, 1)):
            sample = frame[frame["label"] == label].sample(n=n, random_state=seed)
            for col, rel in enumerate(sample["path"]):
                ax = axes[row, col]
                ax.imshow(read_gray(root / rel), cmap="gray", vmin=0, vmax=255)
                _image_axis(ax)
                if col == 0:
                    ax.set_ylabel(CLASS_LABELS[label], color=INK, fontsize=11, fontweight="bold")
        _suptitle(fig, "Pièces moulées (roues de pompe) : conformes et défectueuses",
                  "Photos originales 512×512 du jeu Casting Product, prises dans le train")
        _save(fig, path)


def clahe_effect(gray: np.ndarray, pre: dict, path: Path) -> None:
    enhanced = apply_clahe(gray, float(pre["clahe_clip_limit"]), int(pre["clahe_tile_grid"]))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2))
        for ax, img, title in zip(axes, (gray, enhanced), ("Image originale", "Après CLAHE"), strict=True):
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title)
            _image_axis(ax)
        _suptitle(fig, "Prétraitement : rehaussement du contraste local (CLAHE)",
                  "Les porosités et les bavures ressortent mieux avant le passage dans les modèles")
        fig.tight_layout()
        _save(fig, path)


def split_distribution(report: dict, path: Path) -> None:
    names = [("train", "Train"), ("val", "Validation"), ("blind_test", "Blind test")]
    x = np.arange(len(names))
    width = 0.34
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        for offset, label, key in ((-width / 2 - 0.01, 0, "OK"), (width / 2 + 0.01, 1, "DEFAUT")):
            values = [report["splits"][n][key] for n, _ in names]
            bars = ax.bar(x + offset, values, width, color=CLASS_COLORS[label], label=CLASS_LABELS[label])
            ax.bar_label(bars, padding=3, color=INK_2, fontsize=9)
        ax.set_xticks(x, [label for _, label in names])
        ax.set_ylabel("Nombre d'images")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right")
        share = report["splits"]["train"].get("sampler_share", {})
        subtitle = "Découpage par groupes de quasi-doublons, stratifié par classe"
        if share:
            subtitle += (f" ; tirage d'entraînement équilibré : OK {share['OK'] * 100:.0f} % / "
                         f"Défaut {share['DEFAUT'] * 100:.0f} %")
        _suptitle(fig, "Répartition des images", subtitle)
        fig.tight_layout()
        _save(fig, path)


# --------------------------------------------------------------------------- entrainement


def training_curves(sup_history: pd.DataFrame, ae_history: pd.DataFrame, path: Path) -> None:
    blue, orange = MODEL_COLORS["supervised"], MODEL_COLORS["autoencoder"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8))
        ax = axes[0]
        ax.plot(sup_history["epoch"], sup_history["train_loss"], color=blue, alpha=0.45, label="Train")
        ax.plot(sup_history["epoch"], sup_history["val_loss"], color=blue, label="Validation")
        phase2 = sup_history[sup_history["phase"] == "phase2"]
        if not phase2.empty:
            start = phase2["epoch"].iloc[0] - 0.5
            for a in axes[:2]:
                a.axvline(start, color=MUTED, linewidth=1)
            ax.text(start + 0.2, ax.get_ylim()[1], "fine-tuning", color=INK_2, va="top", fontsize=8.5)
        ax.set_title("Supervisé : perte")
        ax.set_xlabel("Epoch")
        ax.legend()

        ax = axes[1]
        ax.plot(sup_history["epoch"], sup_history["val_average_precision"], color=blue)
        ax.set_title("Supervisé : précision moyenne (val.)")
        ax.set_xlabel("Epoch")

        ax = axes[2]
        ax.plot(ae_history["epoch"], ae_history["train_loss"], color=orange, alpha=0.45, label="Train (OK)")
        ax.plot(ae_history["epoch"], ae_history["val_ok_loss"], color=orange, label="Validation (OK)")
        ax.set_yscale("log")
        ax.set_title("Autoencodeur : perte de reconstruction")
        ax.set_xlabel("Epoch")
        ax.legend()

        ax = axes[3]
        ax.plot(ae_history["epoch"], ae_history["val_roc_auc"], color=orange)
        ax.set_title("Autoencodeur : ROC AUC (val., suivi)")
        ax.set_xlabel("Epoch")
        _suptitle(fig, "Courbes d'entraînement",
                  "Supervisé : meilleur epoch choisi sur la précision moyenne ; "
                  "autoencodeur : arrêt sur la perte des pièces saines")
        fig.tight_layout()
        _save(fig, path)


# --------------------------------------------------------------------------- explicabilite


def _colorbar(fig, axes, label: str, vmax: float = 1.0) -> None:
    """Barre d'echelle partagee, a droite de toute la grille (mise en page 'constrained')."""
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, vmax))
    cbar = fig.colorbar(sm, ax=axes, shrink=0.55, aspect=25, pad=0.015)
    cbar.set_label(label, color=INK_2)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=AXIS, labelcolor=INK_2)


def gradcam_examples(items: list[dict], path: Path) -> None:
    from defectvision.explain.heatmaps import overlay

    n = len(items)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, n, figsize=(2.6 * n + 0.8, 5.4), squeeze=False, layout="constrained")
        for col, item in enumerate(items):
            axes[0, col].imshow(item["supervised_image"], cmap="gray", vmin=0, vmax=255)
            axes[0, col].set_title(f"P(défaut) = {item['p_supervised']:.0%}", fontsize=9.5)
            axes[1, col].imshow(_rgb(overlay(item["supervised_image"], item["gradcam"])))
            for row in (0, 1):
                _image_axis(axes[row, col])
        axes[0, 0].set_ylabel("Image", color=INK, fontweight="bold")
        axes[1, 0].set_ylabel("Grad-CAM", color=INK, fontweight="bold")
        _colorbar(fig, axes.ravel().tolist(), "Attention relative (max = 1)")
        _suptitle(fig, "Explicabilité du modèle supervisé (Grad-CAM)",
                  "Défauts du blind test détectés : zones qui ont le plus pesé dans la décision")
        _save(fig, path)


def autoencoder_examples(items: list[dict], path: Path) -> None:
    from defectvision.explain.heatmaps import overlay

    has_recon = all("reconstruction" in it for it in items)
    cols = 3 if has_recon else 2
    n = len(items)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(cols, n, figsize=(2.6 * n + 0.8, 2.75 * cols + 0.4), squeeze=False,
                                 layout="constrained")
        for col, item in enumerate(items):
            img = item["autoencoder_image"]
            axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=255)
            verdict = "anomalie" if item["anomaly_score"] >= item["threshold"] else "normal"
            axes[0, col].set_title(f"{CLASS_LABELS[item['label']]} → {verdict}\n"
                                   f"score/seuil = {item['anomaly_score'] / item['threshold']:.2f}", fontsize=9)
            row = 1
            if has_recon:
                axes[1, col].imshow(item["reconstruction"], cmap="gray", vmin=0, vmax=255)
                row = 2
            axes[row, col].imshow(_rgb(overlay(img, np.clip(item["anomaly"], 0, 1))))
            for r in range(cols):
                _image_axis(axes[r, col])
        axes[0, 0].set_ylabel("Image", color=INK, fontweight="bold")
        if has_recon:
            axes[1, 0].set_ylabel("Reconstruction", color=INK, fontweight="bold")
        axes[cols - 1, 0].set_ylabel("Carte d'anomalie", color=INK, fontweight="bold")
        _colorbar(fig, axes.ravel().tolist(), "Erreur de reconstruction (1 = seuil)")
        _suptitle(fig, "Détection non supervisée : autoencodeur entraîné sur pièces saines",
                  "Blind test : la carte montre où la pièce s'écarte de l'aspect normal appris")
        _save(fig, path)


# --------------------------------------------------------------------------- scores


def score_space(p_sup: np.ndarray, s_ae: np.ndarray, y: np.ndarray, hybrid, path: Path) -> None:
    from matplotlib.lines import Line2D

    x = np.clip(logit(p_sup), -14, 14)
    z = hybrid.ae_z(s_ae)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.8, 5.9))
        handles = []
        for label in (0, 1):
            m = y == label
            ax.scatter(x[m], z[m], s=26, color=CLASS_COLORS[label], edgecolors=SURFACE, linewidths=0.8, zorder=3)
            handles.append(Line2D([], [], linestyle="none", marker="o", markersize=7, color=CLASS_COLORS[label],
                                  label=f"{CLASS_LABELS[label]} ({int(m.sum())} pièces)"))
        xs = np.linspace(x.min() - 1, x.max() + 1, 200)
        t = float(logit(hybrid.threshold))
        if hybrid.strategy == "stacking" and abs(hybrid.coef_autoencoder) > 1e-9:
            ys = (t - hybrid.intercept - hybrid.coef_supervised * xs) / hybrid.coef_autoencoder
            ax.plot(xs, ys, color=INK, linewidth=1.6, zorder=4)
        elif hybrid.strategy == "or":
            xt = float(logit(hybrid.supervised_threshold)) + t
            zt = float(hybrid.ae_z(hybrid.autoencoder_threshold)) + t
            ax.plot([xs.min(), xt, xt], [zt, zt, z.min() - 1], color=INK, linewidth=1.6, zorder=4)
        handles.append(Line2D([], [], color=INK, linewidth=1.6, label="Frontière du modèle hybride"))
        ax.set_ylim(z.min() - 1, z.max() + 1)
        ax.set_xlim(xs.min(), xs.max())
        x_sup = float(logit(hybrid.supervised_threshold))
        z_ae = float(hybrid.ae_z(hybrid.autoencoder_threshold))
        ax.axvline(x_sup, color=MODEL_COLORS["supervised"], linewidth=1)
        ax.axhline(z_ae, color=MODEL_COLORS["autoencoder"], linewidth=1)
        label_box = {"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none", "alpha": 0.9}
        ax.annotate("seuil supervisé", (x_sup, 1), xycoords=("data", "axes fraction"), xytext=(4, -4),
                    textcoords="offset points", va="top", color=INK_2, fontsize=8.5, bbox=label_box, zorder=6)
        ax.annotate("seuil autoencodeur", (1, z_ae), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                    textcoords="offset points", ha="right", color=INK_2, fontsize=8.5, bbox=label_box, zorder=6)
        ax.set_xlabel("Score supervisé : logit P(défaut)")
        ax.set_ylabel("Score d'anomalie normalisé (z)")
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
        _suptitle(fig, "Espace des deux scores sur le blind test",
                  "Chaque point est une pièce ; le modèle hybride combine les deux axes")
        fig.tight_layout()
        _save(fig, path)


def score_distributions(scores: dict, thresholds: dict, y: np.ndarray, path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(scores), figsize=(5.0 * len(scores), 3.8))
        for ax, (name, values) in zip(np.atleast_1d(axes), scores.items(), strict=True):
            values = np.asarray(values, dtype=float)
            bins = np.linspace(values.min(), values.max(), 40)
            for label in (0, 1):
                ax.hist(values[y == label], bins=bins, color=CLASS_COLORS[label], alpha=0.55,
                        label=CLASS_LABELS[label], edgecolor=SURFACE, linewidth=0.6)
            ax.axvline(thresholds[name], color=INK, linewidth=1.4)
            ax.annotate("seuil", (thresholds[name], 1), xycoords=("data", "axes fraction"),
                        xytext=(4, -4), textcoords="offset points", va="top", color=INK_2, fontsize=8.5)
            ax.set_yscale("log")
            ax.set_title(MODEL_LABELS[name])
            ax.set_xlabel("Score d'anomalie" if name == "autoencoder" else "P(défaut)")
        np.atleast_1d(axes)[0].set_ylabel("Nombre d'images (échelle log)")
        np.atleast_1d(axes)[-1].legend(loc="upper center")
        _suptitle(fig, "Distribution des scores sur le blind test",
                  "Seuils fixés sur la validation (POD 95/95 : rappel ≥ 95 % démontré à 95 % de confiance), "
                  "jamais sur le blind test")
        fig.tight_layout()
        _save(fig, path)


def roc_pr_curves(scores: dict, thresholds: dict, y: np.ndarray, results: dict, path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12.2, 4.9))
        ax_roc.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
        for name, values in scores.items():
            color = MODEL_COLORS[name]
            m = results[name]
            fpr, tpr, _ = roc_curve(y, values)
            ax_roc.plot(fpr, tpr, color=color, label=f"{MODEL_LABELS[name]} — AUC {m['roc_auc']:.3f}")
            ax_roc.scatter([m["false_alarm_rate"]], [m["recall"]], s=64, color=color, edgecolors=SURFACE,
                           linewidths=2, zorder=5)
            prec, rec, _ = precision_recall_curve(y, values)
            ax_pr.plot(rec, prec, color=color, label=f"{MODEL_LABELS[name]} — AP {m['average_precision']:.3f}")
            ax_pr.scatter([m["recall"]], [m["precision"]], s=64, color=color, edgecolors=SURFACE,
                          linewidths=2, zorder=5)
        ax_roc.set_xlabel("Taux de fausses alarmes")
        ax_roc.set_ylabel("Rappel (POD)")
        ax_roc.set_title("Courbe ROC")
        ax_roc.legend(loc="lower right")
        prevalence = float(np.mean(y))
        ax_pr.axhline(prevalence, color=AXIS, linewidth=1)
        ax_pr.set_xlabel("Rappel (POD)")
        ax_pr.set_ylabel("Précision")
        ax_pr.set_title("Courbe précision-rappel")
        ax_pr.legend(loc="lower left")
        for ax in (ax_roc, ax_pr):
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.02)
        _suptitle(fig, "Comparaison des trois modèles sur le blind test",
                  "Le point marque le seuil retenu sur la validation")
        fig.tight_layout()
        _save(fig, path)


def confusion_matrices(results: dict, path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(results), figsize=(4.1 * len(results), 3.9))
        for ax, (name, m) in zip(np.atleast_1d(axes), results.items(), strict=True):
            cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
            ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=cm.max())
            for i in range(2):
                for j in range(2):
                    share = cm[i, j] / max(cm[i].sum(), 1)
                    dark = cm[i, j] > 0.55 * cm.max()
                    ax.text(j, i, f"{cm[i, j]}\n{share:.1%}", ha="center", va="center", fontsize=11,
                            color="#ffffff" if dark else INK)
            ax.set_xticks([0, 1], ["OK", "Défaut"])
            ax.set_yticks([0, 1], ["OK", "Défaut"])
            ax.set_xlabel("Prédit")
            ax.set_ylabel("Réel")
            ax.set_title(MODEL_LABELS[name])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
        _suptitle(fig, "Matrices de confusion sur le blind test",
                  "Pourcentages par ligne : rappel et taux de fausses alarmes")
        fig.tight_layout()
        _save(fig, path)


def metrics_with_ci(results: dict, path: Path, n_boot: int = 2000) -> None:
    metrics = [
        ("recall", "Rappel (POD)"),
        ("precision", "Précision"),
        ("f2", "F2"),
        ("balanced_accuracy", "Exactitude équilibrée"),
        ("false_alarm_rate", "Taux de fausses alarmes"),
    ]
    names = list(results)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.0), sharey=True)
        for ax, (key, title) in zip(axes, metrics, strict=True):
            for i, name in enumerate(names):
                m, ci = results[name][key], results[name]["ci95"][key]
                ax.plot([ci["low"], ci["high"]], [i, i], color=MODEL_COLORS[name], linewidth=2)
                ax.scatter([m], [i], s=64, color=MODEL_COLORS[name], edgecolors=SURFACE, linewidths=2, zorder=5)
                ax.annotate(f"{m:.3f}", (m, i), xytext=(0, 8), textcoords="offset points", ha="center",
                            color=INK_2, fontsize=8.5)
            ax.set_title(title, fontsize=10)
            ax.set_ylim(-0.6, len(names) - 0.4)
            ax.grid(axis="y", visible=False)
            ax.margins(x=0.25)
        axes[0].set_yticks(range(len(names)), [MODEL_LABELS[n] for n in names])
        axes[0].invert_yaxis()
        _suptitle(fig, "Métriques au seuil retenu, avec intervalle de confiance à 95 %",
                  f"Bootstrap stratifié ({n_boot:,} rééchantillonnages) sur le blind test".replace(",", " "))
        fig.tight_layout()
        _save(fig, path)
