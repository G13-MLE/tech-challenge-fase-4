"""Ingestão: carrega o dataset bruto, padroniza colunas e remove PII.

Etapa 1. Converte o CSV bruto em `data/reference/reference.parquet`,
renomeando a coluna alvo, descartando a coluna de identificação (PII) e
**deduplicando** (SHA-256 exatos + MinHash/LSH near-duplicates) **antes**
de qualquer separação treino/validação (FR-003). O resumo da deduplicação
é gravado em ``reports/validation/dedup.json``.
"""

import json
import logging

from src.core.dataset import (
    DATA_RAW_PATH,
    DEDUP_REPORT_PATH,
    HARMONIZATION_REPORT_PATH,
    ID_COLUMN,
    REFERENCE_DATA_PATH,
    atomic_write_text,
    save_parquet,
)
from src.core.params import load_params
from src.data.dedup import deduplicate
from src.observability.logging import TaskTimer

logger = logging.getLogger(__name__)

__all__ = ["ingest", "main"]

# Colunas canônicas usadas pelo contrato/modelo (subconjunto do UCI 350).
CANONICAL_FEATURES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "BILL_AMT1",
    "PAY_AMT1",
]
# Colunas que o contrato exige como float (o UCI as traz como inteiro).
_FLOAT_FEATURES = ("LIMIT_BAL", "BILL_AMT1", "PAY_AMT1")
# Códigos não documentados no UCI 350 (0, 5, 6) mapeados para "others"
# (EDUCATION=4, MARRIAGE=3), respeitando o domínio do contrato (FR-001).
_EDUCATION_HARMONIZATION = {0: 4, 5: 4, 6: 4}
_MARRIAGE_HARMONIZATION = {0: 3}


def _harmonize_dtypes_and_categories(frame):
    """Ajusta dtypes e códigos de categoria aos domínios do contrato.

    O UCI 350 usa inteiros para colunas monetárias e contém códigos de
    categoria não documentados em ``EDUCATION`` (0/5/6) e ``MARRIAGE`` (0);
    ambos são harmonizados para o domínio do contrato (FR-001) sem alterar
    a semântica documentada (``others``).

    Em vez de mutar silenciosamente, retorna também um **registro
    auditável** da harmonização (mapeamento + contagens por código),
    persistido em ``reports/validation/harmonization.json`` e anexado ao
    relatório de validação (clarificação #5: sem mutação silenciosa).

    Args:
        frame: DataFrame canônico carregado do bruto.

    Returns:
        Tupla ``(frame_harmonizado, registro_de_harmonizacao)``.
    """
    columns_audit: dict[str, dict] = {}
    for column, mapping in (
        ("EDUCATION", _EDUCATION_HARMONIZATION),
        ("MARRIAGE", _MARRIAGE_HARMONIZATION),
    ):
        if column not in frame.columns:
            continue
        before = frame[column].value_counts().to_dict()
        remapped = {int(code): int(count) for code, count in before.items() if code in mapping}
        frame[column] = frame[column].replace(mapping).astype(int)
        columns_audit[column] = {
            "mapping": {str(code): target for code, target in mapping.items()},
            "remapped_counts": {str(code): count for code, count in remapped.items()},
            "n_remapped": int(sum(remapped.values())),
        }
    for column in _FLOAT_FEATURES:
        if column in frame.columns:
            frame[column] = frame[column].astype(float)
    harmonization = {
        "policy": "code harmonization to documented domain (no semantic change)",
        "columns": columns_audit,
    }
    return frame, harmonization


def ingest() -> int:
    """Executa a ingestão do dataset bruto para o formato de referência.

    Projeta as colunas canônicas, harmoniza dtypes/categorias ao contrato
    do FR-001 (registrando a decisão de forma auditável), aplica
    deduplicação (exata + near-duplicate) antes de persistir o dataset de
    referência, garantindo que nenhuma separação posterior
    (treino/validação) compartilhe linhas duplicadas.

    Returns:
        Número de linhas persistidas no dataset de referência.

    Raises:
        FileNotFoundError: Se o arquivo bruto não existir.
        ValueError: Se alguma coluna canônica estiver ausente no bruto.
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
    columns = [*CANONICAL_FEATURES, params.data.target_column]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Colunas canônicas ausentes no dataset bruto: {missing}.")
    frame = frame[columns].copy()
    frame, harmonization = _harmonize_dtypes_and_categories(frame)
    deduplicated, dedup_report = deduplicate(frame)
    save_parquet(deduplicated, REFERENCE_DATA_PATH)
    atomic_write_text(DEDUP_REPORT_PATH, json.dumps(dedup_report, indent=2))
    atomic_write_text(HARMONIZATION_REPORT_PATH, json.dumps(harmonization, indent=2))
    logger.info(
        "Ingestão concluída: %d linhas em %s (%d removidas na dedup).",
        len(deduplicated),
        REFERENCE_DATA_PATH,
        dedup_report["total_removed"],
    )
    return int(len(deduplicated))


def main() -> None:
    """Ponto de entrada do estágio de ingestão."""
    with TaskTimer("ingestao") as timer:
        n_rows = ingest()
        timer.record(n_rows=n_rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
