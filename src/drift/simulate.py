"""Simulação de Data Drift e Concept Drift ao longo do tempo.

Etapa 2. A partir do dataset de referência, gera snapshots mensais de
produção com distribuições deslocadas (Data Drift) e com mudança em
P(y|X) (Concept Drift). Determinístico via seed fixa.
"""

import logging

import numpy as np
import pandas as pd

from src.core.dataset import PRODUCTION_DIR, REFERENCE_DATA_PATH, load_parquet, save_parquet
from src.core.params import load_params

logger = logging.getLogger(__name__)

__all__ = ["simulate_snapshots"]


def _shift_column(
    series: pd.Series, direction: str, amount: float, rng: np.random.Generator
) -> pd.Series:
    """Desloca a distribuição de uma coluna numérica.

    Args:
        series: Série original.
        direction: 'increase' ou 'decrease'.
        amount: Magnitude relativa do deslocamento.
        rng: Gerador aleatório determinístico.

    Returns:
        Série deslocada.
    """
    noise = rng.normal(0, series.std() * 0.05, len(series))
    sign = 1.0 if direction == "increase" else -1.0
    return series + sign * series.std() * amount + noise


def simulate_snapshots() -> list:
    """Gera snapshots mensais de produção com drift crescente.

    Returns:
        Lista de caminhos dos snapshots gravados.
    """
    params = load_params()
    reference = load_parquet(REFERENCE_DATA_PATH)
    rng = np.random.default_rng(params.drift.seed)
    target = params.data.target_column
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for month in range(1, params.drift.n_snapshots + 1):
        snapshot = reference.copy()
        progress = month / params.drift.n_snapshots
        amount = params.drift.intensity * progress
        for column, direction in params.drift.shifted_features.items():
            if column in snapshot.columns:
                snapshot[column] = _shift_column(snapshot[column], direction, amount, rng)
        # Concept drift: inverte rótulos de uma fração crescente (P(y|X) muda).
        flip_mask = rng.random(len(snapshot)) < params.drift.concept_drift_strength * progress
        snapshot.loc[flip_mask, target] = 1 - snapshot.loc[flip_mask, target]
        path = PRODUCTION_DIR / f"prod_month_{month:02d}.parquet"
        save_parquet(snapshot, path)
        paths.append(path)
        logger.info("Snapshot %s gerado (%d linhas).", path.name, len(snapshot))
    return paths
