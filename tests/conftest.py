"""Fixtures compartilhadas entre os testes.

Gera um DataFrame sintético com o esquema do dataset de crédito para
que os testes rodem sem depender de download ou de dados reais.
"""

import numpy as np
import pandas as pd
import pytest

SEED = 42
N_ROWS = 400
SMALL_N_ROWS = 8
DRIFT_SEED = 43

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


def make_drifted_frame(n_rows: int = N_ROWS, seed: int = DRIFT_SEED) -> pd.DataFrame:
    """Cria um DataFrame de produção com Data Drift (features deslocadas).

    Desloca ``LIMIT_BAL``/``AGE``/``PAY_0`` para cima, preservando o
    schema e o domínio do contrato, para simular snapshots de produção
    usados nos testes de drift e monitoramento.

    Args:
        n_rows: Número de linhas.
        seed: Seed para reprodutibilidade.

    Returns:
        DataFrame com as features deslocadas.
    """
    frame = make_credit_frame(n_rows=n_rows, seed=seed)
    frame["LIMIT_BAL"] = frame["LIMIT_BAL"] * 1.5
    frame["AGE"] = np.clip(frame["AGE"] + 10, 18, 100)
    frame["PAY_0"] = np.clip(frame["PAY_0"] + 2, -2, 8)
    return frame


@pytest.fixture()
def credit_frame() -> pd.DataFrame:
    """DataFrame sintético válido conforme o contrato."""
    return make_credit_frame()


@pytest.fixture()
def reference_frame() -> pd.DataFrame:
    """DataFrame de referência fixo (seed=42) para comparações de drift."""
    return make_credit_frame()


@pytest.fixture()
def small_valid_frame() -> pd.DataFrame:
    """DataFrame pequeno e válido conforme o contrato (reuso em contratos)."""
    return make_credit_frame(n_rows=SMALL_N_ROWS)


@pytest.fixture()
def production_frame() -> pd.DataFrame:
    """DataFrame de produção com Data Drift (seed fixa, features deslocadas)."""
    return make_drifted_frame()


@pytest.fixture()
def bad_frame() -> pd.DataFrame:
    """DataFrame sintético com violações do contrato."""
    frame = make_credit_frame()
    frame.loc[0:2, "AGE"] = 15
    frame.loc[3:5, "LIMIT_BAL"] = -100.0
    frame.loc[6, "EDUCATION"] = 99
    return frame
