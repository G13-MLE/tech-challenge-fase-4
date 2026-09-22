"""Testes dos testes estatísticos de drift (Etapa 2)."""

import numpy as np
import pandas as pd

from src.drift.stats import compute_ks, compute_psi


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


def test_ks_detects_shift():
    """KS deve detectar diferença entre distribuições deslocadas."""
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(0, 1, 2000))
    shifted = pd.Series(rng.normal(5, 1, 2000))
    result = compute_ks(reference, shifted)
    assert result["statistic"] > 0.5
    assert result["pvalue"] < 0.05
