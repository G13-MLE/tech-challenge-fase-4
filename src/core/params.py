"""Carregamento centralizado dos parâmetros do pipeline.

Lê o arquivo `configs/params.yaml` e disponibiliza os valores como
modelos Pydantic tipados para os estágios do pipeline.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

__all__ = [
    "PARAMS_PATH",
    "DataParams",
    "TrainParams",
    "DriftParams",
    "DriftThresholds",
    "Params",
    "load_params",
]

PARAMS_PATH = Path("configs/params.yaml")


class DataParams(BaseModel):
    """Parâmetros da ingestão e separação dos dados.

    Attributes:
        target_column: Nome da coluna alvo binária.
        id_column: Nome da coluna de identificação (PII).
        reference_train_size: Proporção do dataset de referência usada para treino.
        seed: Seed fixa para reprodutibilidade.
    """

    target_column: str = "default_next_month"
    id_column: str = "ID"
    reference_train_size: float = Field(gt=0, lt=1, default=0.8)
    seed: int = Field(ge=0, default=42)


class TrainParams(BaseModel):
    """Parâmetros do estágio de treinamento.

    Attributes:
        seed: Seed fixa para reprodutibilidade.
        test_size: Proporção do conjunto de teste.
        classifier: Tipo de classificador ('xgboost' ou 'logistic_regression').
        class_weight: Estratégia de peso das classes ('balanced' ou None).
        logistic_regression_C: Inverso da regularização da regressão logística.
        logistic_regression_max_iter: Máximo de iterações da regressão logística.
        xgboost_n_estimators: Número de árvores do XGBoost.
        xgboost_max_depth: Profundidade máxima das árvores do XGBoost.
        xgboost_learning_rate: Taxa de aprendizado do XGBoost.
        xgboost_subsample: Fração de amostras por árvore do XGBoost.
        xgboost_colsample_bytree: Fração de features por árvore do XGBoost.
        cv_folds: Número de folds para cross-validation (0 desativa).
    """

    seed: int = Field(ge=0, default=42)
    test_size: float = Field(gt=0, lt=1, default=0.2)
    classifier: Literal["xgboost", "logistic_regression"] = "xgboost"
    class_weight: str | None = "balanced"
    logistic_regression_C: float = Field(gt=0, default=0.5)  # noqa: N815
    logistic_regression_max_iter: int = Field(ge=100, default=2000)
    xgboost_n_estimators: int = Field(ge=1, default=300)
    xgboost_max_depth: int = Field(ge=1, default=4)
    xgboost_learning_rate: float = Field(gt=0, le=1, default=0.05)
    xgboost_subsample: float = Field(gt=0, le=1, default=0.8)
    xgboost_colsample_bytree: float = Field(gt=0, le=1, default=0.8)
    cv_folds: int = Field(ge=0, default=5)


class DriftParams(BaseModel):
    """Parâmetros da simulação de drift.

    Attributes:
        seed: Seed fixa para a simulação determinística.
        n_snapshots: Número de snapshots mensais de produção.
        intensity: Intensidade global do drift (0 = sem drift, 1 = forte).
        shifted_features: Features deslocadas e direção do deslocamento.
        concept_drift_strength: Fração de mudança em P(y|X).
    """

    seed: int = Field(ge=0, default=42)
    n_snapshots: int = Field(ge=2, default=6)
    intensity: float = Field(ge=0, le=1, default=0.5)
    shifted_features: dict[str, str] = Field(default_factory=dict)
    concept_drift_strength: float = Field(ge=0, le=1, default=0.3)


class DriftThresholds(BaseModel):
    """Limiares de decisão para alertas de drift e degradação.

    Attributes:
        psi_stable: Limite superior de PSI para distribuição estável.
        psi_moderate: Limite superior de PSI para drift moderado.
        auc_drop_alert: Queda de ROC-AUC que dispara alerta de degradação.
        ks_pvalue_alert: p-valor do teste de Kolmogorov-Smirnov para alerta.
    """

    psi_stable: float = Field(gt=0, default=0.1)
    psi_moderate: float = Field(gt=0, default=0.25)
    auc_drop_alert: float = Field(gt=0, le=1, default=0.05)
    ks_pvalue_alert: float = Field(gt=0, le=1, default=0.05)


class Params(BaseModel):
    """Parâmetros completos do pipeline.

    Attributes:
        data: Parâmetros de dados.
        train: Parâmetros de treino.
        drift: Parâmetros de simulação de drift.
        drift_thresholds: Limiares de drift.
    """

    data: DataParams
    train: TrainParams
    drift: DriftParams
    drift_thresholds: DriftThresholds


@lru_cache
def load_params() -> Params:
    """Carrega os parâmetros do arquivo configs/params.yaml.

    O resultado é cacheado por invocação (lru_cache). Cada estágio
    do pipeline executa em processo separado, portanto o cache não
    causa problemas de staleness em execução normal. Não chame
    cache_clear() em código de produção.

    Returns:
        Params: Parâmetros tipados e validados do pipeline.

    Raises:
        FileNotFoundError: Se o arquivo de parâmetros não existir.
    """
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"Arquivo de parâmetros não encontrado: {PARAMS_PATH}")
    with PARAMS_PATH.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return Params(**raw)
