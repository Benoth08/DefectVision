# Journal des décisions

Chaque choix fait pendant l'industrialisation, avec sa raison et, quand il y en a une, la mesure qui l'appuie.
Les chiffres viennent de `reports/` (générés par le pipeline) sauf mention contraire.

## Données

### D1. Le découpage se fait sur les 1 300 photos originales, pas sur les 7 348 images 300×300

Le jeu Kaggle contient deux ensembles : `casting_512x512` (1 300 photos originales, 519 OK et 781 défauts)
et `casting_data` (7 348 images 300×300 déjà augmentées par l'auteur, réparties en train/test).

L'audit (`reports/data_audit.json`, étape `index`) mesure la fuite du découpage fourni par Kaggle :
**9,0 % des images de son test ont un jumeau exact dans son train** (même image à une symétrie près,
distance dHash 0 sur 256 bits) et 33,7 % un quasi-doublon (≤ 12 bits). Un score mesuré sur ce test
est donc optimiste. Les images augmentées ne peuvent pas non plus être rattachées de façon fiable à
leur original (distance médiane 40 bits : rotations quelconques, recadrages) ; on ne peut donc pas les
répartir proprement par pièce. Le projet découpe les originaux et fait ses propres augmentations,
uniquement sur le train.

### D2. L'unité de découpage est le groupe de photos d'une même pièce

Même parmi les « originaux », une pièce est souvent photographiée plusieurs fois (autre orientation,
léger décalage). Deux photos sont liées :

- **lien sûr** : distance dHash ≤ 12 bits sur 256, sous les 8 symétries du carré ;
- **lien vérifié** : distance ≤ 20 bits **et** corrélation du détail fin (image passe-haut 256 px) ≥ 0,5.

Les groupes sont les composantes connexes de ces liens. Résultat : 1 039 groupes, 380 images
partageant un groupe, plus grand groupe de 25 images, aucun groupe aux étiquettes contradictoires.

Pourquoi deux niveaux : sur ce jeu, toutes les roues se ressemblent et la distance dHash au plus proche
voisin forme un continuum (1 à 48 bits), sans écart net entre « même pièce » et « pièce semblable ».
L'inspection visuelle de paires (planches générées pendant l'audit) montre la même pièce jusqu'à ~11 bits,
et encore souvent entre 13 et 20 bits. La corrélation du détail fin (grain de fonderie, taches) dépasse
0,5 pour la majorité des doublons nets et pour moins de 1 % des paires éloignées (30 à 60 bits).
Le critère est volontairement prudent : mieux vaut regrouper deux pièces semblables que laisser fuir une
même pièce dans le blind test.

### D3. Trois jeux : train (70 %), validation (15 %), blind test (15 %)

- Stratification par classe, groupe par groupe.
- **Blind test figé** : l'ordre des groupes vient d'un hachage (sel `defectvision-v1` + clé du groupe) ;
  le blind test ne dépend que de ce sel et de son ratio. Changer `val_ratio` ou un hyperparamètre ne
  le modifie pas (test `test_blind_test_is_frozen_when_val_ratio_changes`).
- Le blind test n'est lu que par l'étape `evaluate`. Choix d'epoch, seuils et fusion hybride se font
  sur la validation.
- Contrôles automatiques : aucun groupe ni fichier partagé entre deux jeux (l'étape échoue sinon).

### D4. Train insensible au déséquilibre

Le jeu est déséquilibré dans l'autre sens que d'habitude : 60 % de défauts. Le train porte une colonne
`weight` (inverse de la fréquence de la classe) et l'entraînement supervisé tire ses lots avec
`WeightedRandomSampler` : chaque lot est en moyenne à 50/50 (`sampler_share` dans
`reports/split_report.json`). Les métriques suivies sont insensibles à la prévalence ou la rendent
explicite : rappel (POD), précision, F2, exactitude équilibrée, MCC, AUC ROC, précision moyenne.
Variante possible par `supervised.balancing: class_weight` (perte pondérée).

## Modèles

### M1. Supervisé : ResNet-18, deux phases, comme le notebook d'origine

Phase 1 (tête seule, lr 1e-3, 5 epochs) puis phase 2 (deux derniers blocs dégelés, lr 1e-5, 10 epochs,
cosinus) : valeurs du notebook conservées. `TARGET_SIZE` et `NUM_EPOCHS` n'y étaient pas définis :
224 px (taille native de ResNet) et 5 + 10 epochs.

Changement : le meilleur epoch était choisi sur le rappel au seuil 0,5, critère qui favorise un modèle
qui déclare tout défectueux. Il est choisi ici sur la **précision moyenne** (aire sous la courbe
précision-rappel), indépendante du seuil.

### M2. Seuils : POD 95/95 sur la validation, précision maximale

Le notebook choisissait le seuil qui maximise la précision sous la contrainte « rappel observé ≥ 95 % »
sur la validation. Cette règle a été appliquée telle quelle d'abord (sur les scores observés, sans
grille à 0,01, pour les trois modèles). **Première évaluation sur le blind test** (conservée dans
`reports/experiments/blind_test_regle_seuil_haut.json`) : rappel 0,932 (supervisé), 0,897
(autoencodeur), 0,923 (hybride), tous sous 95 %, et la condition de promotion échouait.

Cause : avec 117 défauts en validation, viser exactement 95 % de rappel place le seuil au bord de la
contrainte ; sur des pièces nouvelles, le rappel retombe naturellement en dessous (le seuil est
optimisé sur l'échantillon même qui le juge).

Règle retenue : la pratique du contrôle non destructif, une **POD 95/95** (dans l'esprit du critère
a90/95 de la MIL-HDBK-1823A). Le seuil maximise la précision sous la contrainte « borne inférieure de
Clopper-Pearson à 95 % du rappel ≥ 95 % » : sur 117 défauts, cela exige d'en détecter au moins 116.
À précision égale, le seuil le plus bas est gardé. Paramètre : `evaluation.recall_confidence`
(supprimer la ligne revient à l'ancienne règle).

**Transparence** : cette règle a été adoptée après avoir vu la première évaluation du blind test.
Le blind test a donc servi deux fois, et les chiffres finaux sont un peu moins indépendants qu'une
première lecture. Seuls les seuils ont changé : architecture, entraînement et données sont identiques
(mêmes graines, mêmes poids). Pour une mesure de nouveau totalement indépendante, il faut un nouveau
blind test (nouvelles pièces, ou nouveau `split.salt` après une nouvelle collecte).

### M3. Non supervisé : autoencodeur de caractéristiques, entraîné sur les seules pièces saines

L'autoencodeur reconstruit les caractéristiques intermédiaires (couches 2 et 3) d'un ResNet-18 gelé,
par des convolutions 1×1 ; l'erreur de reconstruction donne une carte d'anomalie et un score d'image
(maximum de la carte lissée). Il n'apprend que sur les images OK du train (361 images) ; l'arrêt
anticipé utilise la perte des images OK de la validation. Les étiquettes de défaut ne servent qu'à
calibrer le seuil, ce qu'un score d'anomalie exige (il n'a pas d'échelle propre).

Une variante sur pixels (`autoencoder.architecture: conv`, perte L2 + SSIM) est fournie et testée.

**Choix des réglages, sur la validation uniquement** (197 images, dont 80 OK ; critère : précision
moyenne, comme pour le modèle supervisé ; le blind test n'a pas servi). Mesuré sur GPU L4 le 2026-10-23 :

| Variante | AUC ROC | Précision moyenne | Précision au rappel ≥ 95 % |
|---|---|---|---|
| A. caractéristiques, CLAHE, latent 64, 224 px, 60 epochs (départ) | 0,870 | 0,921 | 0,696 |
| B. comme A, 200 epochs | 0,867 | 0,919 | 0,704 |
| C. comme B, latent 32 | 0,894 | 0,936 | 0,700 |
| D. comme B, score = moyenne du 1 % le plus anormal | 0,876 | 0,923 | 0,720 |
| E. comme B, **sans CLAHE** | 0,945 | 0,964 | 0,762 |
| F. comme B, 320 px | 0,920 | 0,953 | 0,740 |
| G. pixels (conv), CLAHE, 256 px, 300 epochs | 0,929 | 0,951 | 0,758 |
| H. sans CLAHE, latent 32 | 0,945 | 0,962 | 0,836 |
| **I. sans CLAHE, latent 32, 320 px (retenue)** | **0,966** | **0,979** | 0,849 |
| J. pixels (conv), sans CLAHE, 256 px, 300 epochs | 0,956 | 0,969 | 0,882 |
| K. sans CLAHE, latent 16 | 0,932 | 0,957 | 0,774 |

Enseignement principal : le CLAHE, utile au modèle supervisé, nuit à l'autoencodeur. Il amplifie la
texture normale de la fonte, que l'autoencodeur ne sait pas reconstruire et prend pour une anomalie.
L'autoencodeur a donc son propre réglage de prétraitement (`autoencoder.preprocessing.clahe: false`),
enregistré avec le modèle et appliqué à l'identique par l'API.

### M4. Hybride : régression logistique sur les deux scores (stacking)

Entrées : `logit(p_supervisé)` et le score d'anomalie normalisé (médiane et dispersion robuste des
pièces OK de la validation). Régression logistique à classes équilibrées, apprise sur la validation ;
le seuil est choisi sur des prédictions **hors échantillon** (validation croisée stratifiée à 5 plis)
pour ne pas être optimiste. La règle « OU » (défaut si l'une des branches dépasse son seuil) est
calculée aussi et comparée ; `hybrid.strategy` choisit celle qui est déployée.

Chaque décision indique la branche qui l'a déclenchée (`supervise`, `non_supervise`, `les_deux`,
`aucun`) et demande une **revue opérateur** dans deux cas : la probabilité fusionnée est à moins de
0,1 du seuil, ou l'autoencodeur voit une **anomalie forte** que le supervisé ne signale pas (défaut
peut-être d'un type inconnu). « Forte » veut dire au-dessus du 99e centile des scores des pièces
saines de la validation (`hybrid.novelty_quantile`). Une première version utilisait le seuil de
l'autoencodeur (réglé pour 95 % de rappel) : elle renvoyait trop de pièces saines en revue (constaté
lors de la vérification MLflow sur le pod). C'est la logique de l'article cité : le supervisé nomme
les défauts connus, le non supervisé couvre l'inattendu.

## Industrialisation

### I1. Les étapes DVC n'écrivent pas dans MLflow ; l'enregistrement est une étape séparée

`dvc repro` reste reproductible hors ligne (pas de serveur nécessaire), et un entraînement fait ailleurs
(GPU distant) s'enregistre de la même façon. `python -m defectvision.registry.register --promote`
journalise ensuite un run complet (paramètres, historiques par epoch, métriques validation et blind
test avec intervalles, figures, `dvc.lock`) et enregistre le dossier `models/` comme modèle `pyfunc`
autonome (code inclus).

### I2. Registre MLflow : alias `challenger` et `champion`

Chaque version enregistrée reçoit `challenger`. Elle devient `champion` si les conditions de
`params.yaml` (`gates` : rappel ≥ 95 % et précision ≥ 80 % pour l'hybride sur le blind test) sont
remplies et si son F2 n'est pas inférieur à celui du champion en place. L'API charge le champion au
démarrage et sur `POST /admin/reload` (appelé par Airflow), avec repli sur `./models`.

Limite assumée : comparer les versions successives sur le même blind test finit par l'épuiser (on
choisit, sans le vouloir, ce qui lui convient). À terme,
renouveler le blind test avec des pièces nouvelles (le sel `split.salt` permet d'en tirer un autre).

### I3. Airflow : `DockerOperator` sur une image `trainer`, `LocalExecutor`

Chaque étape tourne dans un conteneur éphémère de l'image `trainer`, sur le dépôt monté. `dvc repro
<étape>` ne refait que ce qui a changé : une exécution sans changement passe en quelques secondes.
`LocalExecutor` plutôt que Celery : un seul nœud, pas de Redis, moins de mémoire. Les tâches DVC
s'enchaînent (DVC verrouille le dépôt). Planification hebdomadaire, DAG en pause à la création.

### I4. Augmentations en NumPy/OpenCV au lieu d'Albumentations

Albumentations a été archivé en 2025 ; son successeur est sous double licence AGPL/commerciale. La
chaîne du notebook (symétries, rotation à bord réfléchi, luminosité/contraste, bruit, flou) est
réécrite en quelques lignes, sans dépendance.

### I5. Service sans bibliothèques d'entraînement

L'image `api` n'embarque ni pandas, ni scikit-learn, ni matplotlib (vérifié par un test) : la fusion
hybride est réimplémentée en NumPy à partir des coefficients appris.

### I6. Supervision

Prometheus collecte l'API (trafic, latence, prédictions par classe, distribution des scores, score
d'anomalie, revues opérateur, branche déclenchante, luminosité et contraste des images reçues) et
nginx (exportateur). Règles d'alerte : API muette, aucun modèle chargé, latence p95 > 1 s, erreurs > 5 %,
dérive du taux de défauts ou de la luminosité par rapport à la moyenne sur 7 jours, trop de revues
opérateur. Grafana est provisionné (source et tableau versionnés).

### I7. Ports et mémoire

La machine de développement partage une VM Docker de 3 Go avec d'autres projets : chaque service a une
limite mémoire, Airflow est dans un profil séparé, et les ports par défaut (8088, 5500, 8089, 9091)
évitent ceux déjà pris. MLflow, Airflow et Prometheus n'écoutent que sur 127.0.0.1 ; seul nginx est
exposé.

### I7 bis. MLflow : sans moteur de tâches de fond, et avec les noms d'hôtes autorisés

- Le serveur MLflow 3.17 démarre par défaut un moteur de tâches (huey) pour l'évaluation GenAI et les
  traitements planifiés : plusieurs processus Python supplémentaires, ~650 Mo mesurés pour un seul worker.
  Le projet n'en a pas l'usage : `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` (lu dans le code de
  `mlflow/server/__init__.py`) et une limite de 512 Mo.
- Depuis MLflow 3.5, un middleware refuse toute requête dont l'en-tête `Host` n'est pas `localhost`
  (réponse 403, « invalid Host header »). Or l'API et les tâches du pipeline appellent le serveur par son nom
  de service, `mlflow:5000`. Constaté lors de la démonstration complète, corrigé par
  `MLFLOW_SERVER_ALLOWED_HOSTS` (liste explicite, sans joker global).

### I8. Certificats

Sur un poste dont l'antivirus ou le proxy inspecte le HTTPS, `pip` échoue dans les conteneurs. Les
Dockerfiles ajoutent les certificats `.crt` déposés dans `docker/certs/` (ignorés par git) aux
autorités reconnues. La vérification TLS n'est jamais désactivée.

### I9. DVC dans des conteneurs éphémères, sur un dépôt monté depuis Windows

Constaté sur la machine de développement (Docker Desktop, dépôt sous `C:\Users\...`, antivirus actif) :
à travers le montage, DVC traite environ 5 à 9 fichiers par seconde. `dvc pull` des 8 648 images prend
donc près d'une demi-heure, alors que l'extraction de la même archive directement sur Windows prend 23 s.
Trois mesures en découlent :

- les empreintes déjà calculées par DVC (`/var/tmp/dvc`) vivent dans un volume nommé
  `defectvision-dvc-site-cache`, monté par le service `trainer` et par chaque tâche du DAG ; sans lui,
  chaque conteneur éphémère relirait toutes les images ;
- `core.checksum_jobs = 16` : les lectures se font en parallèle, ce qui compense la latence du montage ;
- `.dvcignore` exclut `__pycache__` et `*.pyc`. Sans cela, l'empreinte d'un dossier de code dépendait de
  la machine qui avait exécuté les tests. C'est ce qui s'est produit : le pod avait laissé ses fichiers
  `.pyc` dans l'empreinte de l'étape `evaluate`.

Sur un hôte Linux, le montage est natif et ces lenteurs disparaissent.

### I10. Ce qui a été vérifié, et où (23-24/10/2026)

| Vérification | Où | Résultat |
|---|---|---|
| 54 tests (données, métriques, modèles, pipeline, API, registre, DAG), `ruff` | serveur Linux (Python 3.12) ; DAG avec Airflow 3.2.1 | tous réussis |
| Pipeline DVC complet (`dvc repro`) | GPU L4 | 292 s, `dvc status` propre |
| `dvc repro --single-item evaluate` avec le code final | GPU RTX 3090 | métriques identiques, `dvc.lock` mis à jour |
| MLflow : enregistrement, promotion, `pyfunc`, API servie depuis le registre | serveur Linux, MLflow 3.17.2 | réussi |
| `nginx -t`, `promtool check config`, `promtool check rules` | serveur Linux | valides (8 règles) |
| Démonstration de bout en bout (nginx, API, MLflow, Prometheus, exportateur, Grafana, Airflow) | serveur Linux, processus | réussie, captures du README |
| Images Docker `api`, `trainer`, `mlflow` | poste Windows (Docker Desktop) | construites |
| Conteneurs `postgres`, `mlflow` (santé), `trainer` (`dvc fetch`) | poste Windows | fonctionnent |
| Pile Compose complète, DAG avec `DockerOperator` | poste Windows | **non vérifié** : la VM Docker (3 Go, partagée avec d'autres projets) s'est figée le 23/10 vers 23 h, et elle ne doit pas être redémarrée sans accord |

La démonstration a révélé deux réglages indispensables à MLflow 3.17, ajoutés au Compose (I7 bis). Pour
finir la vérification sur le poste, une fois Docker de nouveau disponible :

```bash
docker compose up -d && docker compose run --rm trainer python -m defectvision.registry.register --promote
docker compose run --rm trainer python scripts/smoke_test.py --url http://nginx/api
docker compose --profile airflow up -d      # puis déclencher defectvision_training dans l'interface
```

### I11. Le modèle supervisé reste un ResNet-18 (pas de YOLOX)

Le jeu Casting Product n'a que des étiquettes d'image (OK/défaut), sans boîte englobante : un
détecteur d'objets comme YOLOX ne peut pas y être entraîné sans annotation supplémentaire. Le projet
garde la classification ResNet-18 du notebook et ajoute la localisation par Grad-CAM et par la carte
d'anomalie de l'autoencodeur.
