# DefectVision — Controle Qualite Industriel par Vision IA

> Systeme de detection de defauts sur pieces industrielles par Deep Learning (ResNet-18 + Grad-CAM)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![CV](https://img.shields.io/badge/Domain-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Portfolio%20PoC-orange)

---

## Le probleme metier

Le controle qualite visuel en industrie (aeronautique, automobile, metallurgie) repose sur des operateurs humains dont la vigilance chute apres 2 heures d'inspection continue. Le taux de defauts non detectes varie de 5 a 15% selon la fatigue. Un defaut non detecte en aeronautique peut couter des vies ; un rappel produit coute **100 a 1000 fois** plus cher qu'une detection en ligne.

## La solution

Un systeme d'**assistance a l'operateur** (pas de remplacement) qui :
1. **Classifie** chaque piece en OK ou Defaut via un CNN (ResNet-18 en transfer learning)
2. **Explique** sa decision via Grad-CAM (carte de chaleur montrant les zones d'attention)
3. **Optimise** le seuil de decision pour maximiser la detection (rappel >= 95%)

## Architecture

```
Image piece  -->  Pretraitement (CLAHE + Resize)  -->  Augmentation (Albumentations)
                                                              |
                                                              v
                                                   ResNet-18 (Transfer Learning)
                                                     Phase 1 : backbone gele
                                                     Phase 2 : fine-tuning top layers
                                                              |
                                                              v
                                                    P(OK) / P(Defaut)
                                                              |
                                              +---------------+---------------+
                                              |                               |
                                              v                               v
                                       Seuil optimal                    Grad-CAM
                                    (rappel >= 95%)               (explicabilite visuelle)
                                              |                               |
                                              v                               v
                                      Decision finale              Carte de chaleur
                                                              superposee sur l'image
```

## Resultats

| Metrique | Valeur |
|---|---|
| Rappel (POD - Probability of Detection) | > 95% |
| Precision | > 90% |
| F2-Score | > 93% |
| ROC AUC | > 0.98 |
| Temps d'inference (CPU) | < 100 ms/image |

*Les valeurs exactes dependent du run ; les chiffres ci-dessus sont indicatifs sur le dataset Casting Product.*

## Dataset

**Casting Product Image Data** — Dataset industriel reel contenant ~7 000 images de pieces moulees (pompes a eau de submersible) classees en OK et Defaut.

Source : [Kaggle - Real Life Industrial Dataset of Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

## Stack technique

| Composant | Technologie | Justification |
|---|---|---|
| Pretraitement | OpenCV (CLAHE) | Rehaussement de contraste local, critique en CND |
| Augmentation | Albumentations | Plus rapide que torchvision, support bboxes natif |
| Modele CNN | ResNet-18 (torchvision) | Transfer learning efficace avec peu de donnees |
| Entrainement | PyTorch | Flexibilite, controle fin du freeze/unfreeze |
| Explicabilite | Grad-CAM custom | Carte d'attention pour la confiance operateur |
| Metriques | scikit-learn | Rappel, F2, ROC, matrice de confusion |
| Interface | Streamlit | Upload + prediction + visualisation Grad-CAM |

## Structure du projet

```
DefectVision/
├── README.md
├── requirements.txt
├── defect_vision.py                 # Notebook Colab complet
├── cours_projet2.md                 # Cours pedagogique detaille
└── best_model.pt                    # Poids du modele (genere apres entrainement)
```

## Installation et execution

```bash
# Cloner le repo
git clone https://github.com/<votre-username>/DefectVision.git
cd DefectVision

# Installer les dependances
pip install -r requirements.txt

# Lancer le notebook (Colab recommande pour GPU gratuit)
# Ouvrir defect_vision.py dans Google Colab
# Runtime > Change runtime type > T4 GPU
```

## Concepts cles implementes

### Transfer Learning en 2 phases
- **Phase 1 (Feature Extraction)** : Backbone ResNet gele, seule la tete FC apprend. LR = 1e-3.
- **Phase 2 (Fine-Tuning)** : Les 2 dernieres couches convolutives sont degelees. LR = 1e-5.

Cette strategie permet d'obtenir de bons resultats avec seulement quelques centaines d'images.

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
Technique de rehaussement de contraste local indispensable en imagerie industrielle. Equivalent 2D d'un controle automatique de gain (AGC) avec limitation de dynamique.

### Seuil de decision optimise
Au lieu du seuil par defaut (0.5), on cherche le seuil qui **maximise la precision sous contrainte de rappel >= 95%**. En CND, un faux negatif (defaut non detecte) est infiniment plus grave qu'un faux positif (re-inspection inutile).

### Grad-CAM (Gradient-weighted Class Activation Mapping)
Visualisation des zones de l'image qui ont le plus contribue a la decision du modele. Indispensable pour la confiance de l'operateur et la validation metier.

## Metriques CND (vocabulaire metier)

| Terme IA | Terme CND | Signification |
|---|---|---|
| Rappel (Recall) | POD (Probability of Detection) | % de defauts reels detectes |
| Faux negatif | Defaut non detecte | Risque critique |
| Faux positif | Fausse alarme | Cout de re-inspection |
| F2-Score | -- | Metrique composite, 2x le poids au rappel |

## Extensions possibles

- **YOLOv8** pour la localisation (bounding box) des defauts, pas seulement la classification
- **Segmentation (U-Net)** pour le contour exact des defauts
- **Anomaly Detection (PatchCore)** si aucune image de defaut n'est disponible
- **Export ONNX** pour l'inference optimisee sur camera industrielle
- **MLflow** pour le tracking des experiences

## Auteur

Projet realise dans le cadre d'un portfolio Data Science / IA industrielle.
Profil : Docteur-Ingenieur, expertise CND et capteurs, reconversion IA.

---

*Ce projet demontre la maitrise du transfer learning avec peu de donnees, la comprehension du cout asymetrique des erreurs en industrie, et l'importance de l'explicabilite (Grad-CAM) pour le deploiement reel.*
