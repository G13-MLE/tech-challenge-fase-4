"""Concept Drift: mudança em ``P(y|X)`` preservando ``P(Y)``.

Enquanto o Data Drift mede ``P(X)``, o Concept Drift mede ``P(y|X)``. Aqui
simulamos a mudança de conceito **deslocando de forma monotônica o logit**
de decisão (rotação dos pesos latentes), **sem** inversão aleatória de
rótulos — a taxa de positivos ``P(Y)`` é preservada por construção (o
limiar é reescalado ao quantil da taxa-base). Também discriminamos
label/prior drift (``P(Y)``) de concept drift (``P(y|X)``) e avaliamos a
degradação de performance por snapshot.

Implementação numérica em ``src/drift/concept.py``; orquestração do
estágio em ``src/drift/run.py`` (FR-009, FR-010, FR-013).
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.dataset import TARGET_COLUMN
from src.model.evaluate import evaluate_frame
from src.model.predict import load_model, predict_proba

logger = logging.getLogger(__name__)

__all__ = [
    "ConceptModel",
    "fit_concept_model",
    "apply_concept_drift",
    "prior_label_drift",
    "evaluate_snapshots",
]

# Direções "espúrias" (não-direcionais do risco real) usadas na rotação.
_SPURIOUS_WEIGHTS = {
    "AGE": 1.0,
    "SEX": 0.6,
    "MARRIAGE": 0.6,
    "EDUCATION": 0.6,
    "PAY_0": -0.6,
    "BILL_AMT1": -0.6,
    "PAY_AMT1": -0.4,
}


@dataclass
class ConceptModel:
    """Estado latente usado para simular concept drift.

    Attributes:
        features: Ordem das features usadas.
        mean: Média de referência por feature (padronização).
        std: Desvio-padrão de referência por feature.
        coefficients: Coeficientes latentes (logit) ajustados na referência.
        base_rate: Taxa de positivos de referência ``P(Y)``.
        rotation: Direção de rotação normalizada dos coeficientes.
    """

    features: list[str]
    mean: np.ndarray
    std: np.ndarray
    coefficients: np.ndarray
    base_rate: float
    rotation: np.ndarray


def _standardize(frame: pd.DataFrame, model: ConceptModel) -> np.ndarray:
    """Padroniza as features de ``frame`` com as estatísticas da referência."""
    values = frame[model.features].to_numpy(dtype=float)
    return (values - model.mean) / np.where(model.std == 0, 1.0, model.std)


def _rotation_direction(features: list[str], magnitude: float) -> np.ndarray:
    """Constrói a direção determinística de rotação dos coeficientes."""
    weights = np.array([_SPURIOUS_WEIGHTS.get(name, 0.2) for name in features], dtype=float)
    norm = np.linalg.norm(weights)
    if norm == 0:
        weights = np.ones(len(features))
        norm = np.linalg.norm(weights)
    return weights / norm * magnitude


def fit_concept_model(reference: pd.DataFrame, target: str = TARGET_COLUMN) -> ConceptModel:
    """Ajusta o modelo latente de concept drift sobre a referência.

    Ajusta uma regressão logística para obter os coeficientes (logit) que
    reproduzem ``P(y|X)`` na referência e define a base ``P(Y)``.

    Args:
        reference: Dataset de referência (com o alvo).
        target: Nome da coluna alvo.

    Returns:
        Instância de ``ConceptModel`` pronta para gerar concept drift.
    """
    from sklearn.linear_model import LogisticRegression

    features = [column for column in reference.columns if column != target]
    values = reference[features].to_numpy(dtype=float)
    labels = reference[target].to_numpy()
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    standardized = (values - mean) / np.where(std == 0, 1.0, std)
    estimator = LogisticRegression(max_iter=2000)
    estimator.fit(standardized, labels)
    coefficients = estimator.coef_.ravel()
    rotation = _rotation_direction(features, float(np.linalg.norm(coefficients)))
    return ConceptModel(
        features=features,
        mean=mean,
        std=std,
        coefficients=coefficients,
        base_rate=float(labels.mean()),
        rotation=rotation,
    )


def _calibrate_intercept(logits: np.ndarray, base_rate: float) -> float:
    """Encontra o intercepto que preserva ``P(Y) = base_rate`` por bisseção."""
    low, high = -30.0, 30.0
    for _ in range(80):
        mid = (low + high) / 2.0
        mean_probability = float((1.0 / (1.0 + np.exp(-(logits + mid)))).mean())
        if mean_probability < base_rate:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def apply_concept_drift(
    frame: pd.DataFrame,
    model: ConceptModel,
    progress: float,
    strength: float,
    target: str = TARGET_COLUMN,
    seed: int = 42,
) -> pd.DataFrame:
    """Relabela ``frame`` com um deslocamento monotônico de ``P(y|X)``.

    Rotaciona os coeficientes latentes em ``strength * progress`` e gera
    novos rótulos a partir do logit resultante com o intercepto calibrado
    para **preservar ``P(Y)``** — não há inversão aleatória dos rótulos
    originais, e o resultado é determinístico (seed fixa).

    Args:
        frame: Snapshot de produção (features já deslocadas).
        model: Modelo latente ajustado na referência.
        progress: Progresso temporal em ``[0, 1]``.
        strength: Magnitude do concept drift.
        target: Nome da coluna alvo.
        seed: Seed determinística da amostragem de rótulos.

    Returns:
        Cópia de ``frame`` com o alvo relabelado.
    """
    drifted = frame.copy()
    standardized = _standardize(drifted, model)
    coefficients = model.coefficients + strength * progress * model.rotation
    logits = standardized @ coefficients
    intercept = _calibrate_intercept(logits, model.base_rate)
    probabilities = 1.0 / (1.0 + np.exp(-(logits + intercept)))
    rng = np.random.default_rng(seed)
    drifted[target] = (rng.random(len(probabilities)) < probabilities).astype(int)
    return drifted


def prior_label_drift(
    reference: pd.DataFrame, production: pd.DataFrame, target: str = TARGET_COLUMN
) -> dict[str, float]:
    """Mede label/prior drift ``P(Y)`` entre referência e produção.

    Args:
        reference: Dataset de referência.
        production: Snapshot de produção.
        target: Nome da coluna alvo.

    Returns:
        Dicionário com as taxas e a diferença (label drift).
    """
    reference_rate = float(reference[target].mean())
    production_rate = float(production[target].mean())
    return {
        "reference_positive_rate": reference_rate,
        "production_positive_rate": production_rate,
        "label_drift": production_rate - reference_rate,
    }


def evaluate_snapshots(snapshots: list[pd.DataFrame]) -> list[dict]:
    """Avalia o modelo em cada snapshot de produção ao longo do tempo.

    Args:
        snapshots: Lista de DataFrames de produção em ordem temporal.

    Returns:
        Lista de métricas por snapshot (inclui mês, Brier, calibração,
        label drift e concept drift ``P(y|X)``).
    """
    bundle = load_model()
    target = TARGET_COLUMN
    results: list[dict] = []
    reference_auc: float | None = None
    for index, snapshot in enumerate(snapshots, start=1):
        probabilities = predict_proba(snapshot, bundle)
        metrics = evaluate_frame(snapshot, probabilities, target)
        if reference_auc is None:
            reference_auc = metrics["roc_auc"]
        metrics["month"] = index
        metrics["reference_auc"] = reference_auc
        metrics["concept_drift"] = float(reference_auc - metrics["roc_auc"])
        results.append(metrics)
    return results
