"""Utilitários do MLflow Model Registry (aliases champion / challenger)."""

from __future__ import annotations

import mlflow
from mlflow.exceptions import MlflowException

from credit_risk.settings import settings


def client() -> mlflow.MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.MlflowClient()


def get_alias_version(alias: str):
    try:
        return client().get_model_version_by_alias(settings.model_name, alias)
    except MlflowException:
        return None


def load_model(alias: str = "champion"):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.sklearn.load_model(f"models:/{settings.model_name}@{alias}")


def predict_pd(model, df):
    """PD usando as features com que AQUELE modelo foi treinado (contrato do próprio artefato).

    Desacopla versões: um champion antigo continua funcionando mesmo após o código mudar a lista
    de features de um novo challenger.
    """
    cols = list(getattr(model, "feature_names_in_", []))
    if not cols:
        from credit_risk.ml.features import FEATURES

        cols = FEATURES
    return model.predict_proba(df[cols])[:, 1]


def load_reference(alias: str = "champion"):
    """Amostra de referência (treino) salva junto do run que gerou a versão do modelo."""
    import pandas as pd

    mv = get_alias_version(alias)
    if mv is None:
        return None
    path = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="monitoring/reference.parquet")
    return pd.read_parquet(path)
