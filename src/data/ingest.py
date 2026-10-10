"""Ingestão: carrega o dataset bruto, padroniza colunas e remove PII.

Etapa 1. Converte o CSV bruto em `data/reference/reference.parquet`,
renomeando a coluna alvo e descartando a coluna de identificação (PII).
"""

import logging

from src.core.dataset import (
    DATA_RAW_PATH,
    ID_COLUMN,
    REFERENCE_DATA_PATH,
    save_parquet,
)
from src.core.params import load_params
from src.observability.logging import TaskTimer

logger = logging.getLogger(__name__)

__all__ = ["ingest", "main"]


def ingest() -> None:
    """Executa a ingestão do dataset bruto para o formato de referência.

    Raises:
        FileNotFoundError: Se o arquivo bruto não existir.
    """
    import pandas as pd

    if not DATA_RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset bruto não encontrado: {DATA_RAW_PATH}. Rode `make data-download`."
        )
    params = load_params()
    frame = pd.read_csv(DATA_RAW_PATH)
    frame = frame.rename(columns={"default payment next month": params.data.target_column})
    frame = frame.drop(columns=[ID_COLUMN], errors="ignore")
    save_parquet(frame, REFERENCE_DATA_PATH)
    logger.info("Ingestão concluída: %d linhas em %s.", len(frame), REFERENCE_DATA_PATH)


def main() -> None:
    """Ponto de entrada do estágio de ingestão."""
    with TaskTimer("ingestao"):
        ingest()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
