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
from src.core.params import load_params
from src.data.missingness import diagnose_missingness
from src.data.outliers import detect_outliers

__all__ = ["compute_quality", "compute_timeliness", "DEFAULT_DECISION"]

DEFAULT_DECISION = "keep"

# Colunas que expressam "atualidade" do dado (freshness) quando presentes.
_TIMELINESS_COLUMNS = ("EVENT_TIMESTAMP", "timestamp", "snapshot_date", "date")


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


def compute_timeliness(frame: pd.DataFrame) -> dict[str, object]:
    """Avalia a atualidade (freshness) do lote, se houver coluna temporal.

    O dataset de crédito canônico não possui coluna de timestamp, então a
    temporalidade é estruturalmente **não aplicável** — o frescor é
    garantido no nível do pipeline (DVC + snapshots mensais). Quando uma
    coluna temporal reconhecida existir, reporta seu intervalo.

    Args:
        frame: DataFrame do lote.

    Returns:
        Dicionário ``{status, reason}`` ou ``{status, column, min, max}``.
    """
    for column in _TIMELINESS_COLUMNS:
        if column in frame.columns:
            series = pd.to_datetime(frame[column], errors="coerce")
            return {
                "status": "measured",
                "column": column,
                "min": str(series.min()),
                "max": str(series.max()),
            }
    return {
        "status": "not_applicable",
        "reason": "lote sem coluna temporal; frescor garantido por DVC/snapshots mensais",
    }


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
        unicidade, temporalidade, consistência, validade, diagnóstico de
        ausência (MCAR/MAR/MNAR), detecção de outliers (≥3 métodos) e as
        decisões por coluna (padrão ``keep``, sem mutação).
    """
    validity, n_invalid = _validity(frame)
    n_rows = int(len(frame))
    n_duplicates = int(len(frame) - len(frame.drop_duplicates())) if n_rows else 0
    overrides = load_params().quality.decisions
    decisions = {
        str(column): overrides.get(str(column), DEFAULT_DECISION) for column in frame.columns
    }
    return {
        "completeness": round(_completeness(frame), 6),
        "uniqueness": round(_uniqueness(frame), 6),
        "timeliness": compute_timeliness(frame),
        "consistency": round(_consistency(frame), 6),
        "validity": round(validity, 6),
        "n_rows": n_rows,
        "n_columns": int(frame.shape[1]),
        "n_duplicate_rows": n_duplicates,
        "n_invalid_rows": n_invalid,
        "missingness": diagnose_missingness(frame),
        "outliers": detect_outliers(frame),
        "decisions": decisions,
    }
