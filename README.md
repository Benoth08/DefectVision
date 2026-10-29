# DefectVision — Détection de défauts sur pièces de fonderie

Trois modèles pour dire si une pièce est **saine** ou **défectueuse** :

- **Supervisé** : ResNet-18 pré-entraîné, affiné sur des pièces saines et défectueuses.
- **Non supervisé** : autoencodeur entraîné **uniquement sur des pièces saines**. Une pièce qu'il reconstruit mal est suspecte.
- **Hybride** : régression logistique qui combine les deux scores.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange)
![MLOps](https://img.shields.io/badge/MLOps-DVC%20%7C%20MLflow%20%7C%20Airflow-green)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)

![Pièces conformes et défectueuses](reports/figures/dataset_examples.png)

---

## Résultats

Blind test de **195 pièces jamais vues** (117 défauts). Seuils fixés sur la validation.

| Modèle | Rappel | Précision | Fausses alarmes | AUC ROC | Défauts manqués |
|---|---|---|---|---|---|
| Supervisé — ResNet-18 | 95,7 % | 99,1 % | 1,3 % | 0,998 | 5 |
| Autoencodeur | 95,7 % | 78,9 % | 38,5 % | 0,942 | 5 |
| **Hybride** | **97,4 %** | **99,1 %** | **1,3 %** | **0,999** | **3** |

### Courbes ROC et précision-rappel

![Courbes ROC et précision-rappel](reports/figures/roc_pr_curves.png)

### Métriques avec intervalles de confiance (bootstrap, 2 000 tirages)

![Métriques avec intervalles de confiance](reports/figures/metrics_ci.png)

### Matrices de confusion

![Matrices de confusion](reports/figures/confusion_matrices.png)

### Comment l'hybride combine les deux scores

Chaque point est une pièce. Axe horizontal : score du supervisé. Axe vertical : score de l'autoencodeur.

![Espace des deux scores](reports/figures/score_space.png)

![Distribution des scores](reports/figures/score_distributions.png)

### Explicabilité

**Grad-CAM** (supervisé) : zones qui ont pesé dans la décision.

![Grad-CAM](reports/figures/gradcam_examples.png)

**Carte d'anomalie** (autoencodeur) : zones qui s'écartent d'une pièce saine.

![Cartes d'anomalie](reports/figures/autoencoder_maps.png)

### Entraînement

![Courbes d'entraînement](reports/figures/training_curves.png)

### Données

Découpage par groupes de photos d'une même pièce, sans fuite entre train, validation et blind test.

![Répartition des images](reports/figures/split_distribution.png)

![Effet du CLAHE](reports/figures/preprocessing_clahe.png)

---

## MLOps

```mermaid
flowchart LR
    AF[Airflow<br/>DAG hebdomadaire] -->|dvc repro| P
    subgraph P[Pipeline DVC]
        A[acquire] --> I[index] --> S[split]
        S --> TS[train_supervised]
        S --> TA[train_autoencoder]
        TS --> H[fit_hybrid]
        TA --> H
        H --> E[evaluate]
    end
    E -->|quality gate| R[MLflow<br/>alias champion]
    R --> API[FastAPI]
    NG[nginx] --> API
    PR[Prometheus] -->|/metrics| API
    PR --> GF[Grafana]
```

| Outil | Rôle |
|---|---|
| **DVC** | versionne données et modèles ; `dvc repro` ne relance que les étapes modifiées |
| **MLflow** | suivi des runs, registre des modèles (alias `champion` / `challenger`) |
| **Airflow** | réentraîne, évalue, vérifie la quality gate (rappel ≥ 95 %, précision ≥ 80 %), enregistre et recharge l'API |
| **FastAPI + nginx** | API de prédiction derrière une passerelle (limite de débit, taille max) |
| **Prometheus + Grafana** | latence, trafic, taux de défauts, dérive de luminosité et de contraste, alertes |
| **Docker Compose** | toute la pile en microservices |

### Grafana

![Tableau de bord Grafana](docs/images/grafana_dashboard.png)

### Alertes Prometheus

![Alertes Prometheus](docs/images/prometheus_alertes.png)

### MLflow

![Runs MLflow](docs/images/mlflow_runs.png)

![Courbes de perte dans MLflow](docs/images/mlflow_run_metriques.png)

![Registre MLflow](docs/images/mlflow_registre.png)

### Airflow

![DAG Airflow](docs/images/airflow_dag.png)

### API

![Documentation de l'API](docs/images/api_docs.png)

---

## Démarrage rapide

```bash
cp .env.example .env
docker compose build
docker compose --profile jobs build trainer
docker compose run --rm trainer dvc repro        # données + modèles + métriques
docker compose up -d
docker compose run --rm trainer python -m defectvision.registry.register --promote
```

```bash
curl -F "file=@piece.jpeg" "http://localhost:8088/api/predict?model=hybrid"
```

| Service | Adresse |
|---|---|
| API | http://localhost:8088/api/docs |
| Grafana | http://localhost:8088/grafana/ |
| MLflow | http://127.0.0.1:5500 |
| Airflow | http://127.0.0.1:8089 |

*Les choix détaillés sont dans [docs/DECISIONS.md](docs/DECISIONS.md).*
