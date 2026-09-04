"""Initialisation minimale de MLflow en stockage local."""

from __future__ import annotations

from pathlib import Path


def configure_mlflow(repo_root: Path, experiment_name: str):
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow est absent. Installer training/requirements-training.txt avant le smoke test."
        ) from exc
    from mlflow.tracking import MlflowClient

    database = (repo_root / "mlflow.db").resolve()
    artifacts = (repo_root / "mlartifacts").resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{database.as_posix()}")
    client = MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(experiment_name, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(experiment_name)
    return mlflow, database
