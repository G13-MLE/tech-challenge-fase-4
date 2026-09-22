"""Avaliação de performance do modelo por snapshot de produção.

Mede ROC-AUC, precision e recall para evidenciar a degradação
silenciosa ao longo do tempo (Etapa 2).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)

__all__ = ["evaluate_frame"]


def evaluate_frame(frame: pd.DataFrame, probabilities: np.ndarray, target: str) -> dict[str, float]:
    """Calcula métricas de classificação para um snapshot.

    Args:
        frame: DataFrame com a coluna alvo.
        probabilities: Probabilidades previstas pelo modelo.
        target: Nome da coluna alvo.

    Returns:
        Dicionário de métricas.
    """
    labels = frame[target].to_numpy()
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }
    logger.info("Avaliação: ROC-AUC %.4f.", metrics["roc_auc"])
    return metrics
