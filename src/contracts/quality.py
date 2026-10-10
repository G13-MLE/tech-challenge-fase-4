"""Dimensões de qualidade de dados do lote (FR-003).

Mede completude, unicidade, temporalidade, consistência e validade e
registra a decisão por coluna. Política **flag-and-keep** (clarificação
#5): nenhum achado muta os dados; cada coluna recebe a decisão padrão
``keep`` e transformações são opt-in via configuração.
"""

import numpy as np
import pandas as pd
import pandera.errors as pa_errors
import pandera.pandas as pa

from src.contracts.schema import CREDIT_CONTRACT

__all__ = ["compute_quality", "DEFAULT_DECISION"]

DEFAULT_DECISION = "keep"


def _completeness(frame: pd.DataFrame) -> float:
    """Fração de células não nulas (1.0 = sem ausências)."""
    if frame.size == 0:
        return 1.0
    return float(1.0 - frame.isna().to_numpy().mean())


def _uniqueness(frame: pd.DataFrame) -> float:
    """Fração de linhas únicas (1.0 = sem duplicatas exatas)."""
    if len(frame) == 0:
        return 1.0
    return float(len(frame.drop_duplicates()) / len(frame))


def _column_conformance(frame: pd.DataFrame, column: pa.Column) -> pd.Series:
    """Máscara booleana de células conformes a uma coluna do contrato."""
    name = column.name
    if name is None or name not in frame.columns:
        return pd.Series(False, index=frame.index)
    series = frame[name]
    mask = pd.Series(True, index=frame.index)
    if not column.nullable:
        mask &= series.notna()
    for check in column.checks:
        result = check(series)
        output = pd.Series(np.asarray(result.check_output).ravel(), index=frame.index)
        mask &= output.fillna(False)
    return mask


def _consistency(frame: pd.DataFrame) -> float:
    """Fração de células conformes às regras de domínio do contrato."""
    if frame.empty:
        return 1.0
    total = len(frame) * len(CREDIT_CONTRACT.columns)
    if total == 0:
        return 1.0
    conforming = 0
    for column in CREDIT_CONTRACT.columns.values():
        try:
            conforming += int(_column_conformance(frame, column).sum())
        except Exception:  # noqa: BLE001 - dtype inválido torna a coluna não conforme
            continue
    return float(conforming / total)


def _validity(frame: pd.DataFrame) -> tuple[float, int]:
    """Fração de linhas válidas e o número de linhas inválidas.

    Usa o contrato em runtime (fonte única de verdade) com validação
    lazy para contar todas as linhas que violam alguma regra.
    """
    if len(frame) == 0:
        return 1.0, 0
    try:
        CREDIT_CONTRACT.validate(frame, lazy=True)
    except pa_errors.SchemaErrors as err:
        failure_cases = err.failure_cases
        if "index" in failure_cases.columns:
            rows = failure_cases.loc[failure_cases["schema_context"] == "Column", "index"]
            rows = pd.to_numeric(rows, errors="coerce").dropna().astype(int)
            n_invalid = int(rows.nunique())
        else:  # pragma: no cover - contrato atual só tem checks de coluna
            n_invalid = len(frame)
        n_invalid = min(n_invalid, len(frame))
        return float(1.0 - n_invalid / len(frame)), n_invalid
    return 1.0, 0


def compute_quality(frame: pd.DataFrame) -> dict:
    """Calcula as dimensões de qualidade do lote (FR-003).

    Args:
        frame: DataFrame do lote a avaliar.

    Returns:
        Bloco ``quality`` do ``ValidationReport`` com completude,
        unicidade, temporalidade, consistência, validade e as decisões
        por coluna (padrão ``keep``, sem mutação).
    """
    validity, n_invalid = _validity(frame)
    n_rows = int(len(frame))
    n_duplicates = int(len(frame) - len(frame.drop_duplicates())) if n_rows else 0
    return {
        "completeness": round(_completeness(frame), 6),
        "uniqueness": round(_uniqueness(frame), 6),
        "timeliness": "not_applicable",
        "consistency": round(_consistency(frame), 6),
        "validity": round(validity, 6),
        "n_rows": n_rows,
        "n_columns": int(frame.shape[1]),
        "n_duplicate_rows": n_duplicates,
        "n_invalid_rows": n_invalid,
        "decisions": {str(column): DEFAULT_DECISION for column in frame.columns},
    }
