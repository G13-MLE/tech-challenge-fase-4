"""Geração de um dataset sintético de crédito (fallback offline).

Cria um dataset com esquema compatível ao UCI 350 para desenvolvimento
e testes sem download. Não substitui dados reais — apenas permite
rodar o pipeline localmente e no CI.
"""

import logging

import numpy as np
import pandas as pd

from src.core.dataset import DATA_RAW_PATH

logger = logging.getLogger(__name__)

__all__ = ["generate", "main"]

N_ROWS = 5000
SEED = 42


def generate(destination=DATA_RAW_PATH, n_rows: int = N_ROWS, seed: int = SEED):
    """Gera um CSV sintético com o esquema do dataset de crédito.

    Args:
        destination: Caminho de destino do CSV.
        n_rows: Número de linhas a gerar.
        seed: Seed para reprodutibilidade.

    Returns:
        Caminho do arquivo gerado.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "ID": np.arange(1, n_rows + 1),
            "LIMIT_BAL": rng.integers(10000, 800000, n_rows).astype(float),
            "SEX": rng.integers(1, 3, n_rows),
            "EDUCATION": rng.integers(1, 5, n_rows),
            "MARRIAGE": rng.integers(1, 4, n_rows),
            "AGE": rng.integers(18, 80, n_rows),
            "PAY_0": rng.integers(-2, 9, n_rows),
            "BILL_AMT1": rng.normal(50000, 30000, n_rows),
            "PAY_AMT1": rng.integers(0, 100000, n_rows).astype(float),
            "default payment next month": rng.integers(0, 2, n_rows),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    logger.info("Dataset sintético gerado: %s (%d linhas).", destination, n_rows)
    return destination


def main() -> None:
    """Ponto de entrada da geração do dataset sintético."""
    generate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
