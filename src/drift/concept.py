"""Análise de Concept Drift: mudança em P(y|X) e degradação de performance.

Enquanto o Data Drift mede P(X), o Concept Drift mede P(y|X). Aqui
comparamos a performance do modelo entre snapshots para evidenciar a
degradação silenciosa ao longo do tempo.
"""

import logging

import pandas as pd

from src.core.dataset import TARGET_COLUMN
from src.core.params import load_params
from src.model.evaluate import evaluate_frame
from src.model.predict import load_model, predict_proba

logger = logging.getLogger(__name__)

__all__ = ["evaluate_snapshots"]


def evaluate_snapshots(snapshots: list[pd.DataFrame]) -> list[dict[str, float]]:
    """Avalia o modelo em cada snapshot de produção ao longo do tempo.

    Args:
        snapshots: Lista de DataFrames de produção em ordem temporal.

    Returns:
        Lista de métricas por snapshot (índice = mês).
    """
    bundle = load_model()
    target = load_params().data.target_column
    results = []
    for index, snapshot in enumerate(snapshots, start=1):
        probabilities = predict_proba(snapshot, bundle)
        metrics = evaluate_frame(snapshot, probabilities, target if target else TARGET_COLUMN)
        metrics["month"] = index
        results.append(metrics)
    return results
