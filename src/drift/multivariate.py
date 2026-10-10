"""Detecção multivariada de drift: MMD² e Energy distance.

Testes univariados (PSI/KS) medem marginais e são **cegos** a mudanças na
estrutura de dependência entre features. Aqui implementamos o MMD²
(``Maximum Mean Discrepancy``, estimador não-viesado com kernel RBF e
``gamma`` pela heurística da mediana), a Energy distance e um limiar nulo
calibrado por bootstrap (p95/p99). Incluímos também a simulação de uma
**inversão de correlação** (``LIMIT_BAL × AGE``) que preserva as marginais
— invisível ao PSI/KS, porém detectável multivaridamente (FR-011).
"""

import logging

import numpy as np
import pandas as pd

from src.core.dataset import TARGET_COLUMN
from src.core.params import load_params
from src.drift.stats import classify_feature

logger = logging.getLogger(__name__)

__all__ = [
    "mmd2_unbiased",
    "energy_distance",
    "median_heuristic_gamma",
    "mmd2_bootstrap_threshold",
    "energy_bootstrap_threshold",
    "invert_correlation",
    "build_multivariate_report",
]

_MAX_SAMPLES = 400
_BOOTSTRAP_SEED = 42


def _pairwise_sq_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distancias euclidianas ao quadrado entre dois conjuntos de linhas."""
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * a @ b.T, 0.0)


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distancias euclidianas entre dois conjuntos de linhas."""
    return np.sqrt(_pairwise_sq_distances(a, b))


def _rbf_kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    """Kernel RBF (gaussiano) entre dois conjuntos de linhas."""
    return np.exp(-gamma * _pairwise_sq_distances(a, b))


def median_heuristic_gamma(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula ``gamma`` pela heurística da mediana.

    ``gamma = 1 / (2 * mediana_distancia²)`` sobre os dados combinados.

    Args:
        a: Primeiro conjunto de amostras (n_features).
        b: Segundo conjunto de amostras (n_features).

    Returns:
        Valor de ``gamma`` (> 0).
    """
    combined = np.vstack([a, b])
    distances = _pairwise_distances(combined, combined)
    median = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
    return 1.0 / (2.0 * max(median**2, 1e-9))


def mmd2_unbiased(a: np.ndarray, b: np.ndarray, gamma: float) -> float:
    """Estimador não-viesado do MMD² com kernel RBF.

    Args:
        a: Primeiro conjunto de amostras.
        b: Segundo conjunto de amostras.
        gamma: Parâmetro do kernel RBF.

    Returns:
        MMD² (>= 0 truncado em 0 quando o estimador é negativo).
    """
    m, n = len(a), len(b)
    if m < 2 or n < 2:
        return 0.0
    kxx = _rbf_kernel(a, a, gamma)
    kyy = _rbf_kernel(b, b, gamma)
    kxy = _rbf_kernel(a, b, gamma)
    term_x = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    term_y = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    value = term_x + term_y - 2.0 * kxy.mean()
    return float(max(value, 0.0))


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Energy distance (Szekely) entre dois conjuntos de amostras.

    Args:
        a: Primeiro conjunto de amostras.
        b: Segundo conjunto de amostras.

    Returns:
        Energy distance (>= 0 truncada em 0).
    """
    if len(a) < 1 or len(b) < 1:
        return 0.0
    d_ab = _pairwise_distances(a, b).mean()
    d_aa = _pairwise_distances(a, a).mean()
    d_bb = _pairwise_distances(b, b).mean()
    return float(max(2.0 * d_ab - d_aa - d_bb, 0.0))


def _standardize_numeric(
    reference: pd.DataFrame, production: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Seleciona features numéricas e as padroniza pelas estatísticas da referência."""
    numeric = [
        column
        for column in reference.columns
        if column != TARGET_COLUMN and classify_feature(reference[column]) == "numeric"
    ]
    ref_values = reference[numeric].to_numpy(dtype=float)
    prod_values = production[numeric].to_numpy(dtype=float)
    mean = ref_values.mean(axis=0)
    std = ref_values.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (ref_values - mean) / std, (prod_values - mean) / std, numeric


def _subsample(array: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    """Subamostra linhas para manter o custo O(n²) sob controle."""
    if len(array) <= max_samples:
        return array
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(array), size=max_samples, replace=False)
    return array[np.sort(indices)]


def mmd2_bootstrap_threshold(
    reference_standardized: np.ndarray,
    gamma: float,
    n_bootstrap: int,
    quantile: float,
    max_samples: int = _MAX_SAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> float:
    """Calibra o limiar nulo do MMD² por bootstrap.

    Divide a referência em duas metades aleatórias (mesma distribuição) e
    calcula o MMD² ``n_bootstrap`` vezes; o limiar é o quantil solicitado.

    Args:
        reference_standardized: Referência padronizada (features numéricas).
        gamma: Parâmetro do kernel RBF.
        n_bootstrap: Número de reamostragens.
        quantile: Quantil do limiar nulo (ex.: 0.95 → p95).
        max_samples: Máximo de linhas por reamostragem.
        seed: Seed determinística.

    Returns:
        Valor de limiar (float).
    """
    rng = np.random.default_rng(seed)
    sample = _subsample(reference_standardized, max_samples, seed)
    half = len(sample) // 2
    if half < 2:
        return 0.0
    statistics = []
    for _ in range(n_bootstrap):
        permuted = sample[rng.permutation(len(sample))]
        statistics.append(mmd2_unbiased(permuted[:half], permuted[half : 2 * half], gamma))
    return float(np.quantile(statistics, quantile))


def energy_bootstrap_threshold(
    reference_standardized: np.ndarray,
    n_bootstrap: int,
    quantile: float,
    max_samples: int = _MAX_SAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> float:
    """Calibra o limiar nulo da Energy distance por bootstrap.

    Args:
        reference_standardized: Referência padronizada (features numéricas).
        n_bootstrap: Número de reamostragens.
        quantile: Quantil do limiar nulo.
        max_samples: Máximo de linhas por reamostragem.
        seed: Seed determinística.

    Returns:
        Valor de limiar (float).
    """
    rng = np.random.default_rng(seed)
    sample = _subsample(reference_standardized, max_samples, seed)
    half = len(sample) // 2
    if half < 2:
        return 0.0
    statistics = []
    for _ in range(n_bootstrap):
        permuted = sample[rng.permutation(len(sample))]
        statistics.append(energy_distance(permuted[:half], permuted[half : 2 * half]))
    return float(np.quantile(statistics, quantile))


def invert_correlation(frame: pd.DataFrame, feature_a: str, feature_b: str) -> pd.DataFrame:
    """Inverte a correlação entre duas features preservando as marginais.

    Reatribui os valores de ``feature_b`` em ordem inversa ao ranking de
    ``feature_a``, tornando a correlação negativa sem alterar a
    distribuição marginal de nenhuma das duas (invisível ao PSI/KS).

    Args:
        frame: Snapshot onde a inversão será aplicada.
        feature_a: Primeira feature (mantida na ordem original).
        feature_b: Segunda feature (valores reatribuídos).

    Returns:
        Cópia de ``frame`` com a correlação invertida.
    """
    inverted = frame.copy()
    if feature_a not in inverted.columns or feature_b not in inverted.columns:
        return inverted
    order = np.argsort(inverted[feature_a].to_numpy())
    sorted_b = np.sort(inverted[feature_b].to_numpy())
    inverted.iloc[order, inverted.columns.get_loc(feature_b)] = sorted_b[::-1]
    return inverted


def build_multivariate_report(reference: pd.DataFrame, production: pd.DataFrame) -> dict:
    """Constrói o relatório multivariado (MMD²/Energy) de um snapshot.

    Args:
        reference: Dataset de referência.
        production: Snapshot de produção.

    Returns:
        Dicionário com MMD², limiares bootstrap (p95/p99), Energy distance
        e o veredito ``drifted``.
    """
    thresholds = load_params().drift_thresholds
    ref_std, prod_std, numeric = _standardize_numeric(reference, production)
    ref_std = _subsample(ref_std, _MAX_SAMPLES, _BOOTSTRAP_SEED)
    prod_std = _subsample(prod_std, _MAX_SAMPLES, _BOOTSTRAP_SEED)
    gamma = median_heuristic_gamma(ref_std, prod_std)
    mmd2 = mmd2_unbiased(ref_std, prod_std, gamma)
    primary_quantile = thresholds.multivariate_bootstrap_quantile
    upper_quantile = max(0.99, primary_quantile)
    p95 = mmd2_bootstrap_threshold(
        ref_std, gamma, thresholds.multivariate_n_bootstrap, primary_quantile
    )
    p99 = mmd2_bootstrap_threshold(
        ref_std, gamma, thresholds.multivariate_n_bootstrap, upper_quantile
    )
    energy = energy_distance(ref_std, prod_std)
    energy_threshold = energy_bootstrap_threshold(
        ref_std, thresholds.multivariate_n_bootstrap, primary_quantile
    )
    report = {
        "features": numeric,
        "n_samples": {"reference": int(len(ref_std)), "production": int(len(prod_std))},
        "gamma": float(gamma),
        "mmd2": float(mmd2),
        "mmd2_threshold_p95": float(p95),
        "mmd2_threshold_p99": float(p99),
        "mmd2_threshold": float(p95),
        "mmd2_threshold_quantile": float(primary_quantile),
        "energy": float(energy),
        "energy_threshold": float(energy_threshold),
        "drifted": bool(mmd2 > p95),
        "n_bootstrap": int(thresholds.multivariate_n_bootstrap),
    }
    logger.info(
        "Multivariado: MMD²=%.5f (p95=%.5f), drifted=%s.",
        report["mmd2"],
        report["mmd2_threshold"],
        report["drifted"],
    )
    return report
