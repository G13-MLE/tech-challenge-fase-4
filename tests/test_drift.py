"""Testes dos testes estatísticos de drift (Etapa 2)."""

import numpy as np
import pandas as pd

from src.drift.stats import (
    benjamini_hochberg,
    bonferroni,
    build_statistical_report,
    classify_feature,
    compute_chi2,
    compute_ks,
    compute_psi,
    compute_psi_categorical,
)


def test_psi_low_for_same_distribution():
    """PSI deve ser baixo quando a distribuição é a mesma."""
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(0, 1, 2000))
    assert compute_psi(series, series) < 0.05


def test_psi_high_for_shifted_distribution():
    """PSI deve ser alto com deslocamento severo de distribuição."""
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(0, 1, 2000))
    shifted = pd.Series(rng.normal(5, 1, 2000))
    assert compute_psi(reference, shifted) > 0.25


def test_psi_categorical_detects_new_category():
    """PSI categórico detecta uma categoria nunca vista (drift severo)."""
    reference = pd.Series([1, 1, 2, 2, 3, 3, 4, 4] * 50)
    production = pd.Series([1, 2, 3, 5, 5, 5, 5, 5] * 50)
    assert compute_psi_categorical(reference, production) > 0.25
    assert compute_psi_categorical(reference, reference) < 0.01


def test_chi2_and_cramers_v_categorical():
    """χ² detecta dependência entre categoria e população."""
    reference = pd.Series([1] * 200 + [2] * 200)
    production = pd.Series([2] * 300 + [1] * 100)
    result = compute_chi2(reference, production)
    assert result["statistic"] > 0
    assert result["pvalue"] < 0.05
    assert result["cramers_v"] > 0
    assert "method" in result


def test_chi2_low_counts_falls_back_to_monte_carlo_or_fisher():
    """Contagens esperadas baixas acionam Fisher/Monte Carlo (guarda)."""
    reference = pd.Series([1, 2, 2, 3, 3, 3])
    production = pd.Series([1, 1, 2, 3, 4, 4])
    result = compute_chi2(reference, production)
    assert result["method"] in {"fisher_exact", "monte_carlo", "degenerate"}


def test_classify_feature_routes_columns():
    """Colunas inteiras com poucos níveis são categóricas; contínuas, numéricas."""
    assert classify_feature(pd.Series([1, 2, 3, 4])) == "categorical"
    assert classify_feature(pd.Series(np.random.default_rng(0).normal(0, 1, 100))) == "numeric"


def test_multiple_comparison_corrections():
    """FDR e Bonferroni marcam apenas os p-valores suficientemente pequenos."""
    pvalues = [0.001, 0.008, 0.02, 0.4, 0.9]
    fdr = benjamini_hochberg(pvalues, alpha=0.05)
    bonf = bonferroni(pvalues, alpha=0.05)
    assert fdr[0] and fdr[1]
    assert not fdr[4]
    assert bonf[0]
    assert not bonf[3]


def test_statistical_report_flags_breach_and_history_fields():
    """O relatório expõe is_breach, bandas de PSI e correção de comparações."""
    rng = np.random.default_rng(7)
    reference = pd.DataFrame(
        {
            "LIMIT_BAL": rng.normal(0, 1, 800),
            "AGE": rng.integers(18, 80, 800),
            "SEX": rng.integers(1, 3, 800),
            "default_next_month": rng.integers(0, 2, 800),
        }
    )
    production = reference.copy()
    production["LIMIT_BAL"] = production["LIMIT_BAL"] + 6.0
    production["SEX"] = 1
    report = build_statistical_report(reference, production, write=False)
    assert report["is_breach"] is True
    assert report["max_psi"] > 0.25
    assert "multiple_comparison" in report
    assert report["features"]["LIMIT_BAL"]["severity"] == "severe"


def test_ks_detects_shift():
    """KS deve detectar diferença entre distribuições deslocadas."""
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(0, 1, 2000))
    shifted = pd.Series(rng.normal(5, 1, 2000))
    result = compute_ks(reference, shifted)
    assert result["statistic"] > 0.5
    assert result["pvalue"] < 0.05
