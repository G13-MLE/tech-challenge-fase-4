"""Validação do lote contra o contrato de dados.

Bloqueia a ingestão (exit code não-zero) quando o lote viola o contrato,
escrevendo um relatório JSON com as regras violadas.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import pandera.errors as pa_errors

from src.contracts.schema import CREDIT_CONTRACT
from src.core.dataset import (
    REFERENCE_DATA_PATH,
    VALIDATION_RESULT_PATH,
    atomic_write_text,
    load_parquet,
)

logger = logging.getLogger(__name__)

__all__ = ["validate_batch", "main"]


def validate_batch(path: str | None = None) -> dict:
    """Valida um lote de dados contra o contrato de crédito.

    Args:
        path: Caminho do Parquet a validar; se None, usa o de referência.

    Returns:
        Relatório com status e violações encontradas.
    """
    data_path = REFERENCE_DATA_PATH if path is None else Path(path)
    frame = load_parquet(data_path)
    assert isinstance(frame, pd.DataFrame)
    try:
        CREDIT_CONTRACT.validate(frame, lazy=True)
    except pa_errors.SchemaErrors as err:
        report = {
            "status": "failed",
            "file": str(data_path),
            "n_rows": int(len(frame)),
            "n_violations": int(len(err.failure_cases)),
            "violations": json.loads(err.failure_cases.to_json(orient="records")),
        }
        atomic_write_text(VALIDATION_RESULT_PATH, json.dumps(report, indent=2))
        logger.error("Contrato violado: %d falhas em %s.", report["n_violations"], data_path)
        return report
    report = {
        "status": "passed",
        "file": str(data_path),
        "n_rows": int(len(frame)),
        "n_violations": 0,
        "violations": [],
    }
    atomic_write_text(VALIDATION_RESULT_PATH, json.dumps(report, indent=2))
    logger.info("Contrato validado com sucesso: %d linhas.", len(frame))
    return report


def main() -> None:
    """Ponto de entrada do estágio de validação.

    Raises:
        SystemExit: Código 1 se o contrato for violado (bloqueia a ingestão).
    """
    report = validate_batch()
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
