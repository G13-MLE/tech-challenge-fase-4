"""Testes estatísticos de drift: PSI e Kolmogorov-Smirnov.

PSI é a métrica primária de decisão (effect size); KS é reportado
como evidência secundária. Gera `reports/drift/statistical_report.json`.
"""

import json
import logging

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.core.dataset import (
    STATISTICAL_REPORT_PATH,
    TARGET_COLUMN,
    atomic_write_text,
)
from src.core.params import load_params

logger = logging.getLogger(__name__)

__all__ = ["compute_psi", "compute_ks", "build_statistical_report"]


def compute_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Calcula o Population Stability Index entre duas distribuições.

    Args:
        expected: Distribuição de referência.
        actual: Distribuição de produção.
        buckets: Número de buckets do histograma.

    Returns:
        Valor de PSI (>= 0).
    """
    breakpoints = np.linspace(0, 100, buckets + 1)
    cuts = np.percentile(expected, breakpoints)
    cuts = np.unique(cuts)
    expected_counts, _ = np.histogram(expected, bins=cuts)
    actual_counts, _ = np.histogram(actual, bins=cuts)
    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def compute_ks(expected: pd.Series, actual: pd.Series) -> dict[str, float]:
    """Executa o teste de Kolmogorov-Smirnov entre duas distribuições.

    Args:
        expected: Distribuição de referência.
        actual: Distribuição de produção.

    Returns:
        Dicionário com estatística e p-valor.
    """
    result = ks_2samp(expected, actual)
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}


def build_statistical_report(reference: pd.DataFrame, production: pd.DataFrame) -> dict:
    """Constrói o relatório estatístico de drift para todas as features numéricas.

    Args:
        reference: Dataset de referência.
        production: Snapshot de produção.

    Returns:
        Relatório com PSI, KS e classificação por feature.
    """
    thresholds = load_params().drift_thresholds
    features = [c for c in reference.columns if c != TARGET_COLUMN]
    results = {}
    for column in features:
        if not pd.api.types.is_numeric_dtype(reference[column]):
            continue
        psi = compute_psi(reference[column], production[column])
        if psi < thresholds.psi_stable:
            severity = "stable"
        elif psi < thresholds.psi_moderate:
            severity = "moderate"
        else:
            severity = "severe"
        results[column] = {
            "psi": psi,
            "ks": compute_ks(reference[column], production[column]),
            "severity": severity,
        }
    report = {
        "n_features": len(results),
        "n_drifted": sum(1 for r in results.values() if r["severity"] != "stable"),
        "features": results,
    }
    atomic_write_text(STATISTICAL_REPORT_PATH, json.dumps(report, indent=2))
    logger.info(
        "Relatório estatístico: %d features, %d com drift.",
        report["n_features"],
        report["n_drifted"],
    )
    return report
