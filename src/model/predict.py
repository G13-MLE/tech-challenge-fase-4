"""Predição em lote para snapshots de produção.

Carrega o modelo persistido e gera probabilidades de inadimplência,
usado na Etapa 2 para medir a degradação ao longo do tempo.
"""

import logging

import joblib
import numpy as np
import pandas as pd

from src.core.dataset import MODEL_PATH
from src.observability.logging import log_event

logger = logging.getLogger(__name__)

__all__ = ["load_model", "predict_proba"]


def load_model() -> dict:
    """Carrega o pacote de modelo persistido.

    Returns:
        Dicionário com estimador e nomes das features.

    Raises:
        FileNotFoundError: Se o artefato não existir.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {MODEL_PATH}. Rode `make treino` primeiro."
        )
    return joblib.load(MODEL_PATH)


def predict_proba(frame: pd.DataFrame, bundle: dict | None = None) -> np.ndarray:
    """Gera probabilidades de inadimplência para um lote.

    Args:
        frame: DataFrame com as features (sem a coluna alvo).
        bundle: Pacote de modelo; se None, carrega do disco.

    Returns:
        Array de probabilidades da classe positiva.
    """
    if bundle is None:
        bundle = load_model()
    features = [column for column in bundle["features"] if column in frame.columns]
    probabilities = bundle["model"].predict_proba(frame[features])[:, 1]
    # FR-018: registra nome/versão do modelo nos logs de inferência.
    if bundle.get("model_name"):
        log_event(
            "inferencia",
            "success",
            n_rows=int(len(frame)),
            model_name=bundle.get("model_name"),
            model_version=bundle.get("model_version"),
        )
    return probabilities
