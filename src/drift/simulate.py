"""Simulação de Data Drift e Concept Drift ao longo do tempo.

Etapa 2. A partir do dataset de referência, gera snapshots mensais de
produção aplicando: (a) **Data Drift** — deslocamento crescente das
distribuições de features importantes; (b) **Concept Drift** — mudança
monotônica de ``P(y|X)`` preservando ``P(Y)`` (via ``src/drift/concept.py``);
e (c) uma **categoria nunca antes vista** em um snapshot de produção.
Tudo determinístico via seed fixa.

O dataset de produção é mantido **estritamente separado** da referência.
"""

import logging

import numpy as np
import pandas as pd

from src.core.dataset import PRODUCTION_DIR, REFERENCE_DATA_PATH, load_parquet, save_parquet
from src.core.params import load_params
from src.drift.concept import apply_concept_drift, fit_concept_model, prior_label_drift
from src.drift.multivariate import invert_correlation

logger = logging.getLogger(__name__)

__all__ = ["simulate_snapshots", "inject_unseen_category"]

# Categoria sintética inexistente no domínio de treino (EDUCATION ∈ {1,2,3,4}).
UNSEEN_CATEGORY = {"EDUCATION": 5}
UNSEEN_CATEGORY_SHARE = 0.10


def _shift_column(
    series: pd.Series, direction: str, amount: float, rng: np.random.Generator
) -> pd.Series:
    """Desloca a distribuição de uma coluna numérica.

    Colunas inteiras são arredondadas para preservar o tipo/domínio.

    Args:
        series: Série original.
        direction: "increase" ou "decrease".
        amount: Magnitude relativa do deslocamento.
        rng: Gerador aleatório determinístico.

    Returns:
        Série deslocada.
    """
    noise = rng.normal(0, series.std() * 0.05, len(series))
    sign = 1.0 if direction == "increase" else -1.0
    shifted = series + sign * series.std() * amount + noise
    if pd.api.types.is_integer_dtype(series):
        return np.rint(shifted).astype(int)
    return shifted


def inject_unseen_category(
    frame: pd.DataFrame,
    column: str = "EDUCATION",
    value: int = UNSEEN_CATEGORY["EDUCATION"],
    share: float = UNSEEN_CATEGORY_SHARE,
    seed: int = 0,
) -> pd.DataFrame:
    """Injeta uma categoria nunca vista em uma fração das linhas.

    Args:
        frame: Snapshot onde a categoria será injetada.
        column: Coluna categórica alvo da injeção.
        value: Valor da categoria nunca antes vista.
        share: Fração de linhas que receberá a nova categoria.
        seed: Seed determinística.

    Returns:
        Cópia do snapshot com a categoria injetada.
    """
    if column not in frame.columns:
        return frame
    injected = frame.copy()
    rng = np.random.default_rng(seed)
    n_rows = len(injected)
    n_inject = max(1, int(round(n_rows * share)))
    indices = rng.choice(n_rows, size=n_inject, replace=False)
    injected.iloc[indices, injected.columns.get_loc(column)] = value
    logger.info(
        "Categoria inédita %s=%s injetada em %d linhas de %s.",
        column,
        value,
        n_inject,
        column,
    )
    return injected


def simulate_snapshots() -> list:
    """Gera snapshots mensais de produção com Data + Concept Drift.

    Returns:
        Lista de caminhos dos snapshots gravados (ordem temporal).
    """
    params = load_params()
    reference = load_parquet(REFERENCE_DATA_PATH)
    rng = np.random.default_rng(params.drift.seed)
    target = params.data.target_column
    n_snapshots = params.drift.n_snapshots
    concept_model = fit_concept_model(reference, target)
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for month in range(1, n_snapshots + 1):
        snapshot = reference.copy()
        progress = month / n_snapshots
        amount = params.drift.intensity * progress
        for column, direction in params.drift.shifted_features.items():
            if column in snapshot.columns:
                snapshot[column] = _shift_column(snapshot[column], direction, amount, rng)
        # Data drift categórico: categoria nunca antes vista no último snapshot.
        if month == n_snapshots:
            snapshot = inject_unseen_category(snapshot, seed=params.drift.seed + month)
            # Inversão de correlação (LIMIT_BAL × AGE): preserva as marginais
            # (invisível ao PSI/KS) mas detectável multivaridamente (MMD²).
            snapshot = invert_correlation(snapshot, "LIMIT_BAL", "AGE")
        # Concept drift: deslocamento monotônico de P(y|X) preservando P(Y).
        snapshot = apply_concept_drift(
            snapshot,
            concept_model,
            progress=progress,
            strength=params.drift.concept_drift_strength,
            target=target,
        )
        path = PRODUCTION_DIR / f"prod_month_{month:02d}.parquet"
        save_parquet(snapshot, path)
        paths.append(path)
        drift = prior_label_drift(reference, snapshot, target)
        logger.info(
            "Snapshot %s gerado (%d linhas, P(Y)=%.3f, label drift=%+.4f).",
            path.name,
            len(snapshot),
            drift["production_positive_rate"],
            drift["label_drift"],
        )
    return paths
