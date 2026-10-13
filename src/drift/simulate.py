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
from pathlib import Path

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


def simulate_snapshots(
    destination_dir: Path | None = None,
    intensity: float | None = None,
    concept_drift_strength: float | None = None,
    n_snapshots: int | None = None,
    seed: int | None = None,
    reference: pd.DataFrame | None = None,
) -> list[Path]:
    """Gera snapshots mensais de produção com Data + Concept Drift.

    Parâmetros opcionais têm como default o comportamento atual do pipeline
    (``params.drift.*`` e ``data/production/``), de modo que a assinatura
    permanece retrocompatível. O demo executável (US7) usa esses overrides
    para escrever snapshots isolados em ``data/demo/`` reutilizando **um
    único** caminho de simulação.

    Args:
        destination_dir: Diretório dos snapshots (default
            ``data/production/``).
        intensity: Intensidade global do Data Drift (default
            ``drift.intensity``).
        concept_drift_strength: Magnitude do Concept Drift (default
            ``drift.concept_drift_strength``).
        n_snapshots: Número de meses simulados (default ``drift.n_snapshots``).
        seed: Seed determinística (default ``drift.seed``).
        reference: DataFrame de referência; se None, carrega o artefato.

    Returns:
        Lista de caminhos dos snapshots gravados (ordem temporal).
    """
    params = load_params()
    if reference is None:
        reference = load_parquet(REFERENCE_DATA_PATH)
    seed_value = params.drift.seed if seed is None else seed
    n_months = params.drift.n_snapshots if n_snapshots is None else n_snapshots
    drift_intensity = params.drift.intensity if intensity is None else intensity
    concept_strength = (
        params.drift.concept_drift_strength
        if concept_drift_strength is None
        else concept_drift_strength
    )
    destination = destination_dir or PRODUCTION_DIR
    rng = np.random.default_rng(seed_value)
    target = params.data.target_column
    concept_model = fit_concept_model(reference, target)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for month in range(1, n_months + 1):
        snapshot = reference.copy()
        progress = month / n_months
        amount = drift_intensity * progress
        for column, direction in params.drift.shifted_features.items():
            if column in snapshot.columns:
                snapshot[column] = _shift_column(snapshot[column], direction, amount, rng)
        # Data drift categórico: categoria nunca antes vista no último snapshot.
        if month == n_months:
            snapshot = inject_unseen_category(snapshot, seed=seed_value + month)
            # Inversão de correlação (LIMIT_BAL × AGE): preserva as marginais
            # (invisível ao PSI/KS) mas detectável multivaridamente (MMD²).
            snapshot = invert_correlation(snapshot, "LIMIT_BAL", "AGE")
        # Concept drift: deslocamento monotônico de P(y|X) preservando P(Y).
        snapshot = apply_concept_drift(
            snapshot,
            concept_model,
            progress=progress,
            strength=concept_strength,
            target=target,
        )
        path = destination / f"prod_month_{month:02d}.parquet"
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
