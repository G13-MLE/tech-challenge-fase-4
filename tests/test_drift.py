"""Testes dos testes estatísticos de drift (Etapa 2)."""

from pathlib import Path

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
from tests.conftest import make_credit_frame


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
    assert "psi_provenance" in result
    assert (tmp_path / "drift.json").exists()


def test_evidently_html_flags_dataset_drift(tmp_path):
    """O HTML do Evidently marca o dataset como drift (não contradiz o PSI)."""
    from src.drift.detect import build_evidently_report

    reference = make_credit_frame(n_rows=300, seed=1)
    production = reference.copy()
    production["LIMIT_BAL"] = production["LIMIT_BAL"] * 3.0

    html_path = build_evidently_report(reference, production, html_path=tmp_path / "drift.html")
    content = Path(html_path).read_text(encoding="utf-8")

    assert "Drift is detected for" in content
    assert "Dataset Drift is NOT detected" not in content
    assert "LIMIT_BAL" in content


def test_render_statistical_summary_is_readable():
    """O resumo estatístico em Markdown evidencia a mudança por PSI/KS."""
    from src.drift.report import render_statistical_summary

    report = {
        "month": 2,
        "is_breach": True,
        "n_drifted": 1,
        "max_psi": 1.29,
        "psi_bands": {"stable": 0.1, "moderate": 0.25},
        "history": [
            {
                "month": 1,
                "features": {"LIMIT_BAL": {"psi": 0.05, "severity": "stable"}},
            },
            {
                "month": 2,
                "features": {"LIMIT_BAL": {"psi": 1.29, "severity": "severe"}},
            },
        ],
        "features": {
            "LIMIT_BAL": {
                "type": "numeric",
                "psi": 1.29,
                "severity": "severe",
                "ks": {"statistic": 0.22, "pvalue": 0.0},
            }
        },
        "multivariate": {
            "mmd2": 0.07,
            "mmd2_threshold": 0.004,
            "energy": 0.3,
            "energy_threshold": 0.04,
            "drifted": True,
        },
        "drift_types": {"data_drift": True, "concept_drift": True},
    }
    performance = [
        {"month": 1, "roc_auc": 0.66, "reference_auc": 0.76, "concept_drift": 0.10},
        {"month": 2, "roc_auc": 0.61, "reference_auc": 0.76, "concept_drift": 0.15},
    ]

    markdown = render_statistical_summary(report, performance)

    assert markdown.startswith("# Relatório estatístico")
    assert "| LIMIT_BAL |" in markdown
    assert "severe" in markdown
    assert "KS=" in markdown
    assert "MMD²" in markdown
    assert "| M2 |" in markdown
    assert "concept drift" in markdown
