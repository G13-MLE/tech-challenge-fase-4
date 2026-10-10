"""Avaliação de performance do modelo por snapshot de produção.

Mede ROC-AUC, precision, recall, Brier score e calibração para evidenciar
a degradação silenciosa ao longo do tempo (Etapa 2 / Concept Drift).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

__all__ = ["evaluate_frame", "calibration_bins"]


def calibration_bins(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    """Calcula a curva de calibração (bin → previsto vs observado).

    Args:
        labels: Rótulos verdadeiros (0/1).
        probabilities: Probabilidades previstas da classe positiva.
        n_bins: Número de bins de confiança.

    Returns:
        Lista de bins com ``mean_predicted``, ``fraction_positive`` e ``count``.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.digitize(probabilities, edges[1:-1])
    curve: list[dict[str, float]] = []
    for index in range(n_bins):
        mask = bins == index
        count = int(mask.sum())
        if count == 0:
            continue
        curve.append(
            {
                "bin": float((edges[index] + edges[index + 1]) / 2),
                "mean_predicted": float(probabilities[mask].mean()),
                "fraction_positive": float(labels[mask].mean()),
                "count": count,
            }
        )
    return curve


def evaluate_frame(frame: pd.DataFrame, probabilities: np.ndarray, target: str) -> dict:
    """Calcula métricas de classificação e calibração para um snapshot.

    Args:
        frame: DataFrame com a coluna alvo.
        probabilities: Probabilidades previstas pelo modelo.
        target: Nome da coluna alvo.

    Returns:
        Dicionário de métricas (inclui ``brier`` e ``calibration``).
    """
    labels = frame[target].to_numpy()
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "calibration": calibration_bins(labels, probabilities),
    }
    logger.info("Avaliação: ROC-AUC %.4f, Brier %.4f.", metrics["roc_auc"], metrics["brier"])
    return metrics
