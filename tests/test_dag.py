"""Structure du DAG Airflow (saute si Airflow n'est pas installe)."""

from __future__ import annotations

import importlib.util
import os

import pytest

from conftest import PROJECT

# Le dossier airflow/ du depot est importable comme paquet vide : on teste le vrai SDK.
pytest.importorskip("airflow.sdk")
pytest.importorskip("airflow.providers.docker")


@pytest.fixture(scope="module")
def dag():
    os.environ.setdefault("DEFECTVISION_HOST_PROJECT_DIR", "/tmp/defectvision")
    spec = importlib.util.spec_from_file_location("dv_dag", PROJECT / "airflow/dags/defectvision_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dag


def test_dag_tasks_and_order(dag):
    expected = ["dvc_pull", "prepare_data", "train_supervised", "train_autoencoder", "fit_hybrid", "evaluate",
                "quality_gate", "register_mlflow", "dvc_push", "reload_api"]
    assert set(dag.task_ids) == set(expected)
    for upstream, downstream in zip(expected, expected[1:], strict=False):
        assert downstream in dag.get_task(upstream).downstream_task_ids


def test_dag_never_runs_twice_at_once(dag):
    assert dag.max_active_runs == 1
    assert dag.catchup is False
