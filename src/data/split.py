"""Separação entre dataset de referência (treino) e produção.

A separação deve ser determinística (seed fixa) e sem vazamento entre
os conjuntos. O dataset de produção é gerado na Etapa 2.
"""

import logging

import pandas as pd

from src.core.dataset import REFERENCE_DATA_PATH, load_parquet
from src.core.params import load_params

logger = logging.getLogger(__name__)

__all__ = ["split_reference"]


def split_reference(frame: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o dataset de referência em treino e validação.

    Args:
        frame: DataFrame de entrada; se None, carrega o de referência.

    Returns:
        Tupla (treino, validação).
    """
    if frame is None:
        frame = load_parquet(REFERENCE_DATA_PATH)
    params = load_params()
    shuffled = frame.sample(frac=1.0, random_state=params.data.seed).reset_index(drop=True)
    cut = int(len(shuffled) * params.data.reference_train_size)
    train, holdout = shuffled.iloc[:cut], shuffled.iloc[cut:]
    logger.info("Split: %d treino / %d validação.", len(train), len(holdout))
    return train.reset_index(drop=True), holdout.reset_index(drop=True)
