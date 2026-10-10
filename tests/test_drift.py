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


def test_severity_boundary_matches_breach():
    """Um PSI exatamente no limiar é moderate (não severe sem breach)."""
    from src.drift.stats import _severity

    assert _severity(0.25, 0.1, 0.25) == "moderate"
    assert _severity(0.250001, 0.1, 0.25) == "severe"
    assert _severity(0.0999, 0.1, 0.25) == "stable"


def test_drift_gate_fr024_and_detector_modes():
    """O gate do FR-024 falha em breach; o modo detector exige a detecção."""
    from scripts.drift_gate import evaluate_detector, evaluate_gate

    breach = {"max_psi": 0.4, "is_breach": True}
    clean = {"max_psi": 0.05, "is_breach": False}
    assert evaluate_gate(breach, 0.25)[0] is False
    assert evaluate_gate(clean, 0.25)[0] is True
    assert evaluate_detector(breach, 0.25)[0] is True
    assert evaluate_detector(clean, 0.25)[0] is False
    assert evaluate_detector(clean, 0.25, {"suite_passed": False})[0] is True


def test_build_test_suite_result_preserves_suite_verdict(tmp_path, monkeypatch):
    """O veredito do Evidently Test Suite não é mascarado pelo veredito PSI."""
    import src.drift.detect as detect_module

    monkeypatch.setattr(detect_module, "DRIFT_TEST_RESULT_PATH", tmp_path / "drift.json")
    monkeypatch.setattr(
        detect_module,
        "_evidently_test_results",
        lambda reference, production: {"tests": [], "n_failed": 1, "passed": False},
    )
    reference = pd.DataFrame({"A": [1, 2, 3]})
    production = pd.DataFrame({"A": [4, 5, 6]})
    statistical = {"is_breach": True, "max_psi": 0.5, "features": {"A": {"severity": "severe"}}}
    result = detect_module.build_test_suite_result(reference, production, statistical)
    assert result["suite_passed"] is False
    assert result["suite_n_failed"] == 1
    assert result["is_breach"] is True
    assert result["passed"] is False
    assert (tmp_path / "drift.json").exists()
