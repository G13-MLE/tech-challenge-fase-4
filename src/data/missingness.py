"""Diagnóstico de ausências (MCAR/MAR/MNAR) — FR-003, R12.

Para cada coluna com valores ausentes, classifica o mecanismo de
ausência:

- **MNAR**: a ausência associa-se ao alvo (``P(missing | y)`` difere).
- **MAR**: a ausência associa-se a outras variáveis observadas.
- **MCAR**: nenhuma associação detectada com alvo ou demais colunas.
- **none**: sem valores ausentes.

Política flag-and-keep (clarificação #5): nenhum valor é imputado; a
decisão por coluna tem padrão ``keep`` e transformações são opt-in.
"""

import logging

import pandas as pd
from scipy import stats

__all__ = ["diagnose_missingness", "MECHANISMS"]

logger = logging.getLogger(__name__)

MECHANISMS = ("none", "MCAR", "MAR", "MNAR", "undefined")
_ALPHA = 0.05
_DEFAULT_DECISION = "keep"
_MIN_GROUP = 5


def _association_pvalue(indicator: pd.Series, other: pd.Series) -> float | None:
    """p-valor da associação entre a máscara de ausência e outra coluna.

    Usa χ² sobre a tabela de contingência; variáveis numéricas contínuas
    são discretizadas em quantis para viabilizar o teste.
    """
    observed = other.dropna()
    indicator = indicator.loc[observed.index]
    if indicator.nunique() < 2 or observed.nunique() < 2:
        return None
    values = observed
    if pd.api.types.is_numeric_dtype(values) and values.nunique() > 10:
        values = pd.qcut(values, q=min(_MIN_GROUP, values.nunique()), duplicates="drop")
    try:
        table = pd.crosstab(indicator, values)
        if table.shape[0] < 2 or table.shape[1] < 2 or table.to_numpy().size == 0:
            return None
        _, pvalue, _, _ = stats.chi2_contingency(table)
        return float(pvalue)
    except (ValueError, ZeroDivisionError):
        return None


def _classify(frame: pd.DataFrame, column: str, target_column: str | None) -> str:
    """Classifica o mecanismo de ausência de uma coluna."""
    indicator = frame[column].isna()
    n_missing = int(indicator.sum())
    if n_missing == 0:
        return "none"
    if n_missing == len(frame):
        return "undefined"

    if target_column and target_column in frame.columns and target_column != column:
        pvalue = _association_pvalue(indicator, frame[target_column])
        if pvalue is not None and pvalue < _ALPHA:
            return "MNAR"

    for other in frame.columns:
        if other == column:
            continue
        pvalue = _association_pvalue(indicator, frame[other])
        if pvalue is not None and pvalue < _ALPHA:
            return "MAR"
    return "MCAR"


def diagnose_missingness(
    frame: pd.DataFrame, target_column: str | None = "default_next_month"
) -> dict:
    """Diagnostica a ausência de valores por coluna (MCAR/MAR/MNAR).

    Args:
        frame: DataFrame do lote.
        target_column: Coluna alvo usada no teste de MNAR.

    Returns:
        Relatório com ``missing_rate``, ``n_missing``, ``mechanism`` e
        ``decision`` (padrão ``keep``) por coluna, e um resumo de contagem
        por mecanismo.
    """
    n_rows = int(len(frame))
    columns: dict[str, dict] = {}
    summary: dict[str, int] = dict.fromkeys(MECHANISMS, 0)
    for column in frame.columns:
        n_missing = int(frame[column].isna().sum())
        mechanism = _classify(frame, column, target_column)
        summary[mechanism] = summary.get(mechanism, 0) + 1
        columns[str(column)] = {
            "missing_rate": round(n_missing / n_rows, 6) if n_rows else 0.0,
            "n_missing": n_missing,
            "mechanism": mechanism,
            "decision": _DEFAULT_DECISION,
        }
    return {
        "n_rows": n_rows,
        "target_column": target_column,
        "columns": columns,
        "mechanism_summary": summary,
    }
