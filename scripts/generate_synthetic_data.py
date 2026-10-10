"""Geração de um dataset sintético de crédito (fallback offline).

Cria um dataset com esquema compatível ao UCI 350 para desenvolvimento
e testes sem download. Diferente de um gerador sem sinal (ROC-AUC ≈ 0.5),
o alvo aqui é derivado de um **escore latente de risco** correlacionado
com as features, produzindo ROC-AUC ≈ 0.65–0.80 e taxa de positivos ≈ 22%.

Não substitui dados reais — apenas permite rodar o pipeline localmente
e no CI (fallback offline determinístico).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.dataset import DATA_RAW_PATH

logger = logging.getLogger(__name__)

__all__ = ["generate", "latent_risk_score", "main"]

N_ROWS = 5000
SEED = 42
RAW_TARGET_COLUMN = "default payment next month"
TARGET_POSITIVE_RATE = 0.22


def latent_risk_score(frame: pd.DataFrame) -> np.ndarray:
    """Calcula o escore latente de risco (logit) a partir das features.

    A relação é monotônica e interpretável: inadimplência aumenta com o
    status de pagamento recente (``PAY_0``) e com a fatura (``BILL_AMT1``),
    e diminui com o limite de crédito (``LIMIT_BAL``) e o pagamento
    recente (``PAY_AMT1``).

    Args:
        frame: DataFrame com as features de crédito.

    Returns:
        Array de escores latentes (logits) por linha.
    """
    limit_bal = frame["LIMIT_BAL"].to_numpy(dtype=float)
    pay_0 = frame["PAY_0"].to_numpy(dtype=float)
    bill = frame["BILL_AMT1"].to_numpy(dtype=float)
    pay_amt = frame["PAY_AMT1"].to_numpy(dtype=float)
    age = frame["AGE"].to_numpy(dtype=float)

    return (
        -0.6
        + 0.55 * pay_0
        + 0.9 * (bill / 1e5)
        - 0.8 * (limit_bal / 3e5)
        - 0.6 * (pay_amt / 5e4)
        + 0.35 * ((age - 35.0) / 20.0)
    )


def generate(destination=DATA_RAW_PATH, n_rows: int = N_ROWS, seed: int = SEED):
    """Gera um CSV sintético com o esquema e sinal do dataset de crédito.

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
        }
    )
    logits = latent_risk_score(frame)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    # Calibra a probabilidade para atingir a taxa de positivos alvo (~22%).
    scale = TARGET_POSITIVE_RATE / max(float(probabilities.mean()), 1e-9)
    calibrated = np.clip(probabilities * scale, 0.0, 0.99)
    frame[RAW_TARGET_COLUMN] = (rng.random(n_rows) < calibrated).astype(int)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    logger.info(
        "Dataset sintético gerado: %s (%d linhas, taxa positiva %.3f).",
        destination,
        n_rows,
        float(frame[RAW_TARGET_COLUMN].mean()),
    )
    return destination


def main() -> None:
    """Ponto de entrada da geração do dataset sintético."""
    generate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
