"""Detecção de outliers com múltiplos métodos — FR-003, R11/R12.

Aplica ao menos três métodos:

- **Z-score** (univariado, ``|z| > 3``);
- **IQR** (univariado, fora de ``[Q1 - 1.5·IQR, Q3 + 1.5·IQR]``);
- **LOF** (multivariado, ``LocalOutlierFactor``);
- **Isolation Forest** (multivariado, ``sklearn``).

Política flag-and-keep (clarificação #5): outliers são **sinalizados**,
não removidos; a decisão por coluna tem padrão ``keep``.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

__all__ = ["detect_outliers"]

logger = logging.getLogger(__name__)

Z_THRESHOLD = 3.0
IQR_FACTOR = 1.5
CONTAMINATION = "auto"
_MIN_ROWS_MULTIVARIATE = 20
_DEFAULT_DECISION = "keep"


def _numeric_columns(frame: pd.DataFrame, columns: list[str] | None) -> list[str]:
    if columns is not None:
        return [column for column in columns if column in frame.columns]
    return [
        str(column)
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) and column != "default_next_month"
    ]


def _iqr_mask(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - IQR_FACTOR * iqr) | (series > q3 + IQR_FACTOR * iqr)


def _zscore_mask(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(False, index=series.index)
    return ((series - series.mean()).abs() / std) > Z_THRESHOLD


def detect_outliers(frame: pd.DataFrame, columns: list[str] | None = None, seed: int = 42) -> dict:
    """Detecta outliers por múltiplos métodos e sinaliza sem remover.

    Args:
        frame: DataFrame do lote.
        columns: Colunas numéricas a avaliar; se None, infere as numéricas.
        seed: Seed para os métodos estocásticos (Isolation Forest).

    Returns:
        Relatório com contagens por método, bloco multivariado e decisão
        por coluna (padrão ``keep``).
    """
    numeric_columns = _numeric_columns(frame, columns)
    n_rows = int(len(frame))
    per_column: dict[str, dict] = {}
    for column in numeric_columns:
        series = frame[column].astype(float)
        zscore_mask = _zscore_mask(series).fillna(False)
        iqr_mask = _iqr_mask(series).fillna(False)
        per_column[column] = {
            "n_outliers_zscore": int(zscore_mask.sum()),
            "n_outliers_iqr": int(iqr_mask.sum()),
            "rate_zscore": round(float(zscore_mask.mean()), 6) if n_rows else 0.0,
            "rate_iqr": round(float(iqr_mask.mean()), 6) if n_rows else 0.0,
        }

    multivariate = {
        "n_outliers_lof": 0,
        "n_outliers_isolation_forest": 0,
        "rate_lof": 0.0,
        "rate_isolation_forest": 0.0,
    }
    matrix = frame[numeric_columns].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(matrix) >= _MIN_ROWS_MULTIVARIATE and matrix.shape[1] >= 1:
        scaled = (matrix - matrix.mean()) / matrix.std(ddof=0).replace(0, 1.0)
        lof = LocalOutlierFactor(contamination=CONTAMINATION)
        lof_labels = lof.fit_predict(scaled)
        iso = IsolationForest(contamination=CONTAMINATION, random_state=seed)
        iso_labels = iso.fit_predict(scaled)
        n_lof = int((lof_labels == -1).sum())
        n_iso = int((iso_labels == -1).sum())
        denominator = len(matrix)
        multivariate = {
            "n_outliers_lof": n_lof,
            "n_outliers_isolation_forest": n_iso,
            "rate_lof": round(n_lof / denominator, 6),
            "rate_isolation_forest": round(n_iso / denominator, 6),
        }
    else:
        logger.warning(
            "Linhas insuficientes para métodos multivariados (%d < %d).",
            len(matrix),
            _MIN_ROWS_MULTIVARIATE,
        )

    return {
        "methods": ["zscore", "iqr", "lof", "isolation_forest"],
        "thresholds": {
            "zscore": Z_THRESHOLD,
            "iqr_factor": IQR_FACTOR,
            "contamination": CONTAMINATION,
        },
        "n_rows": n_rows,
        "numeric_columns": numeric_columns,
        "columns": per_column,
        "multivariate": multivariate,
        "decisions": dict.fromkeys(numeric_columns, _DEFAULT_DECISION),
    }
