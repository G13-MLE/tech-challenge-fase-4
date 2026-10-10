"""Testes da detecção multivariada de drift (Etapa 2, FR-011)."""

import numpy as np
import pandas as pd

from src.drift.multivariate import (
    build_multivariate_report,
    energy_distance,
    invert_correlation,
    median_heuristic_gamma,
    mmd2_unbiased,
)
from src.drift.stats import compute_ks, compute_psi

TARGET = "default_next_month"


def _correlated_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Frame com correlação positiva entre LIMIT_BAL e AGE."""
    rng = np.random.default_rng(seed)
    limit = rng.normal(0, 1, n)
    age = 0.7 * limit + np.sqrt(1 - 0.49) * rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "LIMIT_BAL": limit,
            "AGE": age,
            "PAY_0": rng.integers(-2, 9, n),
            "BILL_AMT1": rng.normal(0, 1, n),
            "PAY_AMT1": rng.normal(0, 1, n),
            TARGET: rng.integers(0, 2, n),
        }
    )


def test_mmd2_near_zero_for_same_distribution():
    """MMD² é próximo de zero entre amostras da mesma distribuição."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, (200, 3))
    b = rng.normal(0, 1, (200, 3))
    gamma = median_heuristic_gamma(a, b)
    assert mmd2_unbiased(a, b, gamma) < 0.02


def test_mmd2_large_for_shifted_distribution():
    """MMD² é grande quando as distribuições estão deslocadas."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, (200, 3))
    b = rng.normal(3, 1, (200, 3))
    gamma = median_heuristic_gamma(a, b)
    assert mmd2_unbiased(a, b, gamma) > mmd2_unbiased(a, a.copy(), gamma)


def test_energy_distance_positive_for_shift():
    """A Energy distance é positiva e maior para distribuições deslocadas."""
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, (100, 2))
    b = rng.normal(3, 1, (100, 2))
    assert energy_distance(a, b) > energy_distance(a, a.copy())


def test_invert_correlation_is_invisible_to_marginal_tests_but_flagged():
    """A inversão de correlação preserva marginais (PSI/KS≈0) mas o MMD² flagra."""
    reference = _correlated_frame()
    inverted = invert_correlation(reference, "LIMIT_BAL", "AGE")

    assert np.corrcoef(reference["LIMIT_BAL"], reference["AGE"])[0, 1] > 0.5
    assert np.corrcoef(inverted["LIMIT_BAL"], inverted["AGE"])[0, 1] < -0.5
    assert compute_psi(reference["LIMIT_BAL"], inverted["LIMIT_BAL"]) < 0.01
    assert compute_ks(reference["AGE"], inverted["AGE"])["pvalue"] > 0.05

    report = build_multivariate_report(reference, inverted)
    assert report["mmd2"] > report["mmd2_threshold_p95"]
    assert report["drifted"] is True


def test_multivariate_report_bootstrap_thresholds():
    """O relatório expõe limiares bootstrap p95/p99 e o veredito."""
    reference = _correlated_frame(seed=3)
    production = _correlated_frame(seed=4)
    report = build_multivariate_report(reference, production)
    assert report["mmd2_threshold_p95"] <= report["mmd2_threshold_p99"]
    assert report["n_bootstrap"] >= 1
    assert set(report["features"]) >= {"LIMIT_BAL", "AGE"}
