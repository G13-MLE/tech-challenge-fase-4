"""Validação do lote contra o contrato de dados.

Bloqueia a ingestão (exit code não-zero) quando o lote viola o contrato,
escrevendo um relatório JSON com as regras violadas e as dimensões de
qualidade (FR-003). A validação é lazy: todas as violações são coletadas.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pandera.errors as pa_errors

from src.contracts.quality import compute_quality
from src.contracts.schema import CREDIT_CONTRACT
from src.core.dataset import (
    DEDUP_REPORT_PATH,
    REFERENCE_DATA_PATH,
    VALIDATION_RESULT_PATH,
    atomic_write_text,
    load_parquet,
)
from src.observability.logging import TaskTimer

logger = logging.getLogger(__name__)

__all__ = ["validate_batch", "main"]


def _attach_dedup(quality: dict) -> None:
    """Anexa o resumo de deduplicação ao bloco de qualidade, se existir."""
    if DEDUP_REPORT_PATH.exists():
        quality["dedup"] = json.loads(DEDUP_REPORT_PATH.read_text(encoding="utf-8"))


def validate_batch(path: str | None = None) -> dict:
    """Valida um lote de dados contra o contrato de crédito.

    Args:
        path: Caminho do Parquet a validar; se None, usa o de referência.

    Returns:
        Relatório com status, violações e dimensões de qualidade.
    """
    data_path = REFERENCE_DATA_PATH if path is None else Path(path)
    frame = load_parquet(data_path)
    assert isinstance(frame, pd.DataFrame)
    quality = compute_quality(frame)
    _attach_dedup(quality)
    try:
        CREDIT_CONTRACT.validate(frame, lazy=True)
    except pa_errors.SchemaErrors as err:
        report = {
            "status": "failed",
            "file": str(data_path),
            "n_rows": int(len(frame)),
            "n_violations": int(len(err.failure_cases)),
            "violations": json.loads(err.failure_cases.to_json(orient="records")),
            "quality": quality,
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
        "quality": quality,
    }
    atomic_write_text(VALIDATION_RESULT_PATH, json.dumps(report, indent=2))
    logger.info("Contrato validado com sucesso: %d linhas.", len(frame))
    return report


def main(argv: list[str] | None = None) -> None:
    """Ponto de entrada do estágio de validação.

    Args:
        argv: Argumentos de linha de comando; o primeiro é o caminho
            opcional do Parquet (padrão: dataset de referência). Se None,
            usa ``sys.argv[1:]``.

    Raises:
        SystemExit: Código 1 se o contrato for violado (bloqueia a ingestão).
    """
    arguments = sys.argv[1:] if argv is None else argv
    path = arguments[0] if arguments else None
    with TaskTimer("validacao"):
        report = validate_batch(path)
        if report["status"] != "passed":
            sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
