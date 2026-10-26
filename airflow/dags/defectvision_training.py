"""DAG Airflow : reentrainement DefectVision de bout en bout.

Chaque etape tourne dans un conteneur ephemere de l'image 'trainer'
(DockerOperator), sur le depot monte en /workspace :

  dvc_pull -> acquire/index/split -> train_supervised -> train_autoencoder
           -> fit_hybrid -> evaluate -> quality_gate -> register_mlflow
           -> dvc_push -> reload_api

- 'dvc repro <etape>' ne refait que ce qui a change (donnees, code ou
  parametres) : une execution sans changement passe en quelques secondes ;
- quality_gate lit reports/metrics/blind_test.json et arrete le DAG si les
  conditions de params.yaml (gates) ne sont pas remplies ;
- register_mlflow enregistre le jeu de modeles et le promeut 'champion' ;
- reload_api demande a l'API de charger le nouveau champion.

Variables d'environnement (voir .env.example) : DEFECTVISION_HOST_PROJECT_DIR
(chemin du depot vu par le demon Docker), DEFECTVISION_TRAINER_IMAGE,
DEFECTVISION_DOCKER_NETWORK, DEFECTVISION_API_URL, ADMIN_TOKEN.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import DAG, task
from airflow.sdk.exceptions import AirflowFailException
from docker.types import Mount

HOST_PROJECT_DIR = os.environ.get("DEFECTVISION_HOST_PROJECT_DIR", "")
if not HOST_PROJECT_DIR:
    # Visible dans l'interface Airflow (erreur d'import du DAG) plutot qu'en cours d'execution.
    raise RuntimeError("DEFECTVISION_HOST_PROJECT_DIR n'est pas defini (voir .env.example)")
LOCAL_PROJECT_DIR = os.environ.get("DEFECTVISION_LOCAL_PROJECT_DIR", "/opt/defectvision")
TRAINER_IMAGE = os.environ.get("DEFECTVISION_TRAINER_IMAGE", "defectvision-trainer:1.0.0")
NETWORK = os.environ.get("DEFECTVISION_DOCKER_NETWORK", "defectvision")
API_URL = os.environ.get("DEFECTVISION_API_URL", "http://api:8000")
TRAINER_MEM_LIMIT = os.environ.get("DEFECTVISION_TRAINER_MEM_LIMIT", "2g")
DVC_SITE_CACHE_VOLUME = os.environ.get("DEFECTVISION_DVC_SITE_CACHE_VOLUME", "defectvision-dvc-site-cache")

TRAINER_ENV = {
    "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    "DV_NUM_WORKERS": os.environ.get("DV_NUM_WORKERS", "2"),
    "TORCH_NUM_THREADS": os.environ.get("TORCH_NUM_THREADS", "4"),
    "TORCH_HOME": "/workspace/.cache/torch",
    "GIT_COMMIT": "{{ dag_run.conf.get('git_commit', '') if dag_run and dag_run.conf else '' }}",
}

default_args = {
    "owner": "defectvision",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def trainer_task(task_id: str, command: str, **kwargs) -> DockerOperator:
    """Une etape du pipeline dans un conteneur 'trainer' ephemere."""
    return DockerOperator(
        task_id=task_id,
        image=TRAINER_IMAGE,
        command=["bash", "-lc", command],
        working_dir="/workspace",
        mounts=[
            Mount(source=HOST_PROJECT_DIR, target="/workspace", type="bind"),
            # Empreintes DVC deja calculees, partagees entre les conteneurs ephemeres.
            Mount(source=DVC_SITE_CACHE_VOLUME, target="/var/tmp/dvc", type="volume"),
        ],
        environment=TRAINER_ENV,
        network_mode=NETWORK,
        docker_url="unix://var/run/docker.sock",
        auto_remove="force",
        mount_tmp_dir=False,
        mem_limit=TRAINER_MEM_LIMIT,
        do_xcom_push=False,
        **kwargs,
    )


with DAG(
    dag_id="defectvision_training",
    description="Reentrainement, evaluation sur blind test, enregistrement MLflow et deploiement",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["defectvision", "mlops", "vision"],
    doc_md=__doc__,
) as dag:
    dvc_pull = trainer_task("dvc_pull", "dvc pull --allow-missing || echo 'cache DVC vide ou incomplet'")
    prepare_data = trainer_task("prepare_data", "dvc repro split")
    train_supervised = trainer_task("train_supervised", "dvc repro train_supervised",
                                    execution_timeout=timedelta(hours=3))
    train_autoencoder = trainer_task("train_autoencoder", "dvc repro train_autoencoder",
                                     execution_timeout=timedelta(hours=3))
    fit_hybrid = trainer_task("fit_hybrid", "dvc repro fit_hybrid")
    evaluate = trainer_task("evaluate", "dvc repro evaluate")

    @task
    def quality_gate() -> dict:
        """Lit le rapport du blind test ; echoue si les conditions de promotion ne sont pas remplies."""
        path = os.path.join(LOCAL_PROJECT_DIR, "reports", "metrics", "blind_test.json")
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        gates = report["gates"]
        summary = {name: {k: round(m[k], 4) for k in ("recall", "precision", "f2")}
                   for name, m in report["models"].items()}
        print(json.dumps({"gates": gates, "blind_test": summary}, indent=2, ensure_ascii=False))
        if not gates["passed"]:
            raise AirflowFailException(f"conditions de qualite non remplies : {gates['checks']}")
        return summary

    register = trainer_task("register_mlflow", "python -m defectvision.registry.register --promote")
    dvc_push = trainer_task("dvc_push", "dvc push || echo 'pas de remote DVC joignable'")

    @task
    def reload_api() -> dict:
        """Demande a l'API de charger le champion MLflow."""
        import requests

        response = requests.post(f"{API_URL}/admin/reload",
                                 headers={"X-Admin-Token": os.environ.get("ADMIN_TOKEN", "")}, timeout=300)
        response.raise_for_status()
        result = response.json()
        print(json.dumps(result, ensure_ascii=False))
        if not result.get("reloaded"):
            raise AirflowFailException(f"rechargement refuse : {result.get('error')}")
        return result

    gate = quality_gate()
    # Les etapes DVC partagent un verrou de depot : elles s'enchainent.
    dvc_pull >> prepare_data >> train_supervised >> train_autoencoder >> fit_hybrid >> evaluate >> gate
    gate >> register >> dvc_push >> reload_api()
