"""Fixtures compartilhadas entre os testes.

Gera um DataFrame sintético com o esquema do dataset de crédito para
que os testes rodem sem depender de download ou de dados reais.
"""

import numpy as np
import pandas as pd
import pytest

SEED = 42
N_ROWS = 400

FEATURE_COLUMNS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "BILL_AMT1",
    "PAY_AMT1",
]


def make_credit_frame(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Cria um DataFrame sintético compatível com o contrato de crédito.

    Args:
        n_rows: Número de linhas.
        seed: Seed para reprodutibilidade.

    Returns:
        DataFrame com features e coluna alvo.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "LIMIT_BAL": rng.integers(10000, 800000, n_rows).astype(float),
            "SEX": rng.integers(1, 3, n_rows),
            "EDUCATION": rng.integers(1, 5, n_rows),
            "MARRIAGE": rng.integers(1, 4, n_rows),
            "AGE": rng.integers(18, 80, n_rows),
            "PAY_0": rng.integers(-2, 9, n_rows),
            "BILL_AMT1": rng.normal(50000, 30000, n_rows),
            "PAY_AMT1": rng.integers(0, 100000, n_rows).astype(float),
            "default_next_month": rng.integers(0, 2, n_rows),
        }
    )


@pytest.fixture()
def credit_frame() -> pd.DataFrame:
    """DataFrame sintético válido conforme o contrato."""
    return make_credit_frame()


@pytest.fixture()
def bad_frame() -> pd.DataFrame:
    """DataFrame sintético com violações do contrato."""
    frame = make_credit_frame()
    frame.loc[0:2, "AGE"] = 15
    frame.loc[3:5, "LIMIT_BAL"] = -100.0
    frame.loc[6, "EDUCATION"] = 99
    return frame
