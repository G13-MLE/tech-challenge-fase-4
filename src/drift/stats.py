"""Testes estatísticos de drift: PSI, Kolmogorov-Smirnov e χ².

PSI é a métrica primária de decisão (effect size). Para features numéricas
reportamos também Kolmogorov-Smirnov; para categóricas, χ² de independência
com correção de múltiplas comparações (FDR/Bonferroni) e correção de
Cramér's V. ``is_breach`` é verdadeiro quando **qualquer feature única**
tem PSI > 0.25 (drift severo). Gera ``reports/drift/statistical_report.json``.
"""

import json
import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, ks_2samp

from src.core.dataset import (
    STATISTICAL_REPORT_PATH,
    TARGET_COLUMN,
    atomic_write_text,
)
from src.core.params import load_params

logger = logging.getLogger(__name__)

__all__ = [
    "compute_psi",
    "compute_psi_categorical",
    "compute_ks",
    "compute_chi2",
    "classify_feature",
    "benjamini_hochberg",
    "bonferroni",
    "build_statistical_report",
]

# Uma feature numérica inteira com poucos níveis distintos é tratada como
# categórica (ex.: PAY_0, SEX, EDUCATION, MARRIAGE).
_MAX_CATEGORICAL_LEVELS = 20
_LOW_EXPECTED_COUNT = 5.0
_MONTE_CARLO_ITERATIONS = 1000


def classify_feature(series: pd.Series) -> str:
    """Classifica uma coluna como "numeric" ou "categorical".

    Args:
        series: Série de uma feature.

    Returns:
        ``"categorical"`` para dtypes não numéricos ou inteiros com poucos
        níveis distintos; ``"numeric"`` caso contrário.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return "categorical"
    if pd.api.types.is_integer_dtype(series):
        if series.nunique(dropna=True) <= _MAX_CATEGORICAL_LEVELS:
            return "categorical"
        return "numeric"
    return "numeric"


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


def compute_psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    """Calcula PSI por proporção de categorias (alinhando o domínio).

    Args:
        expected: Série categórica de referência.
        actual: Série categórica de produção.

    Returns:
        Valor de PSI (>= 0).
    """
    categories = sorted(set(expected.dropna().unique()) | set(actual.dropna().unique()))
    expected_pct = expected.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    actual_pct = actual.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    expected_pct = np.clip(expected_pct.to_numpy(), 1e-6, None)
    actual_pct = np.clip(actual_pct.to_numpy(), 1e-6, None)
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


def _monte_carlo_chi2_pvalue(
    expected: pd.Series,
    actual: pd.Series,
    observed_statistic: float,
    seed: int = 42,
) -> float:
    """p-valor por permutação para o χ² quando as contagens são baixas."""
    combined = np.concatenate([expected.to_numpy(), actual.to_numpy()])
    sizes = [len(expected), len(actual)]
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(_MONTE_CARLO_ITERATIONS):
        permuted = rng.permutation(combined)
        a = pd.Series(permuted[: sizes[0]])
        b = pd.Series(permuted[sizes[0] :])
        categories = sorted(set(pd.unique(a)) | set(pd.unique(b)))
        observed = np.array(
            [
                a.value_counts().reindex(categories, fill_value=0).to_numpy(),
                b.value_counts().reindex(categories, fill_value=0).to_numpy(),
            ]
        )
        observed = observed[:, observed.sum(axis=0) > 0]
        if observed.shape[1] < 2:
            continue
        stat = float(chi2_contingency(observed, correction=False)[0])
        if stat >= observed_statistic:
            count += 1
    return float((count + 1) / (_MONTE_CARLO_ITERATIONS + 1))


def compute_chi2(expected: pd.Series, actual: pd.Series) -> dict[str, float | str]:
    """Teste χ² de independência para features categóricas.

    Protege contra contagens esperadas baixas: para tabelas 2×2 usa o
    teste exato de Fisher; caso contrário, quando ``expected_min < 5``,
    usa um p-valor por permutação (Monte Carlo).

    Args:
        expected: Série categórica de referência.
        actual: Série categórica de produção.

    Returns:
        Dicionário com estatística, p-valor, Cramér's V, contagem mínima
        esperada e o método efetivamente usado.
    """
    categories = sorted(set(expected.dropna().unique()) | set(actual.dropna().unique()))
    observed = np.array(
        [
            expected.value_counts().reindex(categories, fill_value=0).to_numpy(),
            actual.value_counts().reindex(categories, fill_value=0).to_numpy(),
        ],
        dtype=float,
    )
    observed = observed[:, observed.sum(axis=0) > 0]
    n_obs = observed.sum()
    if observed.shape[1] < 2 or n_obs == 0:
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "cramers_v": 0.0,
            "expected_min": 0.0,
            "method": "degenerate",
        }
    statistic, _, _, expected_counts = chi2_contingency(observed, correction=False)
    expected_min = float(expected_counts.min())
    method = "chi2"
    pvalue = float(chi2_contingency(observed, correction=False)[1])
    if observed.shape == (2, 2):
        pvalue = float(fisher_exact(observed)[1])
        method = "fisher_exact"
    elif expected_min < _LOW_EXPECTED_COUNT:
        logger.warning(
            "χ² com contagem esperada mínima < %.1f (min=%.2f); usando p-valor Monte Carlo.",
            _LOW_EXPECTED_COUNT,
            expected_min,
        )
        pvalue = _monte_carlo_chi2_pvalue(expected, actual, float(statistic))
        method = "monte_carlo"
    dof = min(observed.shape) - 1
    cramers_v = float(np.sqrt(statistic / (n_obs * dof))) if dof > 0 else 0.0
    return {
        "statistic": float(statistic),
        "pvalue": pvalue,
        "cramers_v": cramers_v,
        "expected_min": expected_min,
        "method": method,
    }


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Correção de múltiplas comparações de Benjamini-Hochberg (FDR).

    Args:
        pvalues: Lista de p-valores.
        alpha: Nível de significância.

    Returns:
        Lista booleana indicando quais hipóteses permanecem significativas.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = np.argsort(pvalues)
    ranked = np.array(pvalues)[order]
    thresholds = alpha * (np.arange(1, m + 1) / m)
    below = ranked <= thresholds
    if not below.any():
        return [False] * m
    largest = np.max(np.nonzero(below)[0])
    significant = np.zeros(m, dtype=bool)
    significant[order[: largest + 1]] = True
    return significant.tolist()


def bonferroni(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Correção de Bonferroni para múltiplas comparações.

    Args:
        pvalues: Lista de p-valores.
        alpha: Nível de significância.

    Returns:
        Lista booleana indicando quais hipóteses permanecem significativas.
    """
    m = len(pvalues)
    if m == 0:
        return []
    return [bool(value <= alpha / m) for value in pvalues]


def _severity(psi: float, stable: float, moderate: float) -> str:
    """Classifica o PSI nas bandas stable/moderate/severe.

    Bandas do desafio: ``stable < 0.1``, ``moderate 0.1–0.25`` e
    ``severe > 0.25``. O limite ``moderate`` é inclusivo para permanecer
    consistente com ``is_breach`` (``max_psi > moderate``), evitando que
    um valor exatamente no limiar seja rotulado ``severe`` sem ser breach.
    """
    if psi < stable:
        return "stable"
    if psi <= moderate:
        return "moderate"
    return "severe"


def build_statistical_report(
    reference: pd.DataFrame, production: pd.DataFrame, write: bool = True
) -> dict:
    """Constrói o relatório estatístico de drift para todas as features.

    Args:
        reference: Dataset de referência.
        production: Snapshot de produção.
        write: Se True, persiste o relatório em disco (atômico).

    Returns:
        Relatório com PSI, KS/χ², severidade, correção de múltiplas
        comparações e ``is_breach`` (qualquer feature com PSI > 0.25).
    """
    thresholds = load_params().drift_thresholds
    features = [column for column in reference.columns if column != TARGET_COLUMN]
    results: dict[str, dict] = {}
    pvalues: list[float] = []
    for column in features:
        kind = classify_feature(reference[column])
        psi = (
            compute_psi(reference[column], production[column])
            if kind == "numeric"
            else compute_psi_categorical(reference[column], production[column])
        )
        entry: dict[str, object] = {
            "type": kind,
            "psi": psi,
            "severity": _severity(psi, thresholds.psi_stable, thresholds.psi_moderate),
        }
        if kind == "numeric":
            ks = compute_ks(reference[column], production[column])
            entry["ks"] = ks
            pvalues.append(ks["pvalue"])
        else:
            chi2 = compute_chi2(reference[column], production[column])
            entry["chi2"] = chi2
            pvalues.append(float(chi2["pvalue"]))
        results[column] = entry

    fdr = benjamini_hochberg(pvalues)
    bonf = bonferroni(pvalues)
    for index, column in enumerate(features):
        results[column]["corrected_significant"] = bool(fdr[index])
        results[column]["bonferroni_significant"] = bool(bonf[index])

    max_psi = max((entry["psi"] for entry in results.values()), default=0.0)
    is_breach = bool(max_psi > thresholds.psi_moderate)
    report = {
        "n_features": len(results),
        "n_drifted": sum(1 for r in results.values() if r["severity"] != "stable"),
        "n_severe": sum(1 for r in results.values() if r["severity"] == "severe"),
        "max_psi": float(max_psi),
        "psi_bands": {"stable": thresholds.psi_stable, "moderate": thresholds.psi_moderate},
        "multiple_comparison": {
            "method": "fdr_bh",
            "alpha": 0.05,
            "n_significant_fdr": int(sum(fdr)),
            "n_significant_bonferroni": int(sum(bonf)),
        },
        "is_breach": is_breach,
        "features": results,
    }
    if write:
        atomic_write_text(STATISTICAL_REPORT_PATH, json.dumps(report, indent=2))
    logger.info(
        "Relatório estatístico: %d features, %d com drift, breach=%s.",
        report["n_features"],
        report["n_drifted"],
        report["is_breach"],
    )
    return report
