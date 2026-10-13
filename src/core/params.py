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
    "DatasetParams",
    "QualityParams",
    "TrainParams",
    "DriftParams",
    "DriftThresholds",
    "SLOParams",
    "AlertsParams",
    "RetrainParams",
    "FairnessParams",
    "DemoParams",
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


class DatasetParams(BaseModel):
    """Parâmetros da fonte do dataset de referência.

    Attributes:
        source: Fonte do dataset ('uci' real ou 'synthetic' fallback offline/CI).
    """

    source: Literal["uci", "synthetic"] = "uci"


class QualityParams(BaseModel):
    """Parâmetros das verificações de qualidade de dados.

    Attributes:
        near_dup_threshold: Limiar de similaridade Jaccard (MinHash/LSH) para
            considerar duas linhas como near-duplicates.
        decisions: Registro auditável da decisão pretendida por coluna —
            ``keep``, ``remove`` ou ``transform``. É **descritivo**: as
            remoções/transformações **não** são aplicadas automaticamente
            (política flag-and-keep, FR-004), evitando mutação silenciosa;
            qualquer transformação efetiva deve ser implementada de forma
            explícita pelo operador.
    """

    near_dup_threshold: float = Field(gt=0, le=1, default=0.9)
    decisions: dict[str, Literal["keep", "remove", "transform"]] = Field(default_factory=dict)


class TrainParams(BaseModel):
    """Parâmetros do estágio de treinamento.

    Attributes:
        seed: Seed fixa para reprodutibilidade (também usada pelo estimador).
        classifier: Tipo de classificador ('xgboost' ou 'logistic_regression').
            A proporção treino/validação vem de ``data.reference_train_size``
            (via ``split_reference``).
        class_weight: Estratégia de peso das classes ('balanced' ou None).
        logistic_regression_C: Inverso da regularização da regressão logística.
        logistic_regression_max_iter: Máximo de iterações da regressão logística.
        xgboost_n_estimators: Número de árvores do XGBoost.
        xgboost_max_depth: Profundidade máxima das árvores do XGBoost.
        xgboost_learning_rate: Taxa de aprendizado do XGBoost.
        xgboost_subsample: Fração de amostras por árvore do XGBoost.
        xgboost_colsample_bytree: Fração de features por árvore do XGBoost.
    """

    seed: int = Field(ge=0, default=42)
    classifier: Literal["xgboost", "logistic_regression"] = "xgboost"
    class_weight: str | None = "balanced"
    logistic_regression_C: float = Field(gt=0, default=0.5)  # noqa: N815
    logistic_regression_max_iter: int = Field(ge=100, default=2000)
    xgboost_n_estimators: int = Field(ge=1, default=300)
    xgboost_max_depth: int = Field(ge=1, default=4)
    xgboost_learning_rate: float = Field(gt=0, le=1, default=0.05)
    xgboost_subsample: float = Field(gt=0, le=1, default=0.8)
    xgboost_colsample_bytree: float = Field(gt=0, le=1, default=0.8)


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
    concept_drift_strength: float = Field(ge=0, le=2, default=1.5)


class DriftThresholds(BaseModel):
    """Limiares de decisão para alertas de drift e degradação.

    Attributes:
        psi_stable: Limite superior de PSI para distribuição estável.
        psi_moderate: Limite superior de PSI para drift moderado.
        auc_drop_alert: Queda de ROC-AUC que dispara alerta de degradação.
        multivariate_n_bootstrap: Reamostragens bootstrap do limiar nulo multivariado.
        multivariate_bootstrap_quantile: Quantil do limiar nulo (ex.: 0.95 = p95).
    """

    psi_stable: float = Field(gt=0, default=0.1)
    psi_moderate: float = Field(gt=0, default=0.25)
    auc_drop_alert: float = Field(gt=0, le=1, default=0.05)
    multivariate_n_bootstrap: int = Field(ge=1, default=200)
    multivariate_bootstrap_quantile: float = Field(gt=0, lt=1, default=0.95)


class SLOParams(BaseModel):
    """Data SLOs e Error Budget usados pelo estágio de monitoramento.

    Attributes:
        accuracy_min: Acurácia mínima aceitável do modelo.
        psi_max: PSI máximo tolerado por feature.
        ks_pvalue_min: p-valor mínimo tolerado no teste KS.
        missing_rate_max: Taxa máxima tolerada de valores ausentes.
    """

    accuracy_min: float = Field(ge=0, le=1, default=0.85)
    psi_max: float = Field(gt=0, default=0.25)
    ks_pvalue_min: float = Field(ge=0, le=1, default=0.05)
    missing_rate_max: float = Field(ge=0, le=1, default=0.01)


class AlertsParams(BaseModel):
    """Regras de consolidação anti-fadiga de alertas (FR-016).

    Attributes:
        breach_features: Nº de features que devem cruzar juntas para consolidar.
        consolidation_windows: Nº de janelas consecutivas para uma feature crítica.
    """

    breach_features: int = Field(ge=1, default=2)
    consolidation_windows: int = Field(ge=1, default=3)


class RetrainParams(BaseModel):
    """Gatilho objetivo de re-treino (FR-017).

    Attributes:
        consecutive_critical_windows: Nº de meses consecutivos em severidade
            crítica para disparar o re-treino.
    """

    consecutive_critical_windows: int = Field(ge=1, default=2)


class FairnessParams(BaseModel):
    """Guardrails de fairness por subgrupo (FR-017, FR-026).

    Attributes:
        disparate_impact_min: Razão de impacto dispar mínima (regra dos 80%).
        equal_opportunity_max: Diferença máxima de equal opportunity (TPR).
    """

    disparate_impact_min: float = Field(gt=0, le=1, default=0.8)
    equal_opportunity_max: float = Field(ge=0, le=1, default=0.10)


class DemoParams(BaseModel):
    """Configuração do demo executável ``make demo-drift`` (US7).

    Encoda o cenário de um único demo (bloco ``demo:`` em
    ``configs/params.yaml``), com override opcional por CLI. ``scenario`` é
    **apenas rótulo de exibição** — não seleciona presets (FR-032).

    Attributes:
        scenario: Rótulo do cenário exibido no resumo (identificador apenas).
        n_snapshots: Número de meses (snapshots) simulados no demo (≥ 2).
        intensity: Intensidade do Data Drift do demo (amplificada para o choque).
        concept_drift_strength: Magnitude do Concept Drift ``P(y|X)``.
        output_dir: Diretório isolado de evidência do demo.
        seed: Seed determinística (default = ``drift.seed`` se ausente).
    """

    scenario: str = Field(min_length=1, default="economic_shock")
    n_snapshots: int = Field(ge=2, default=6)
    intensity: float = Field(ge=0, le=1, default=0.8)
    concept_drift_strength: float = Field(ge=0, le=2, default=1.5)
    output_dir: str = Field(min_length=1, default="reports/drift/demo")
    seed: int = Field(ge=0, default=42)


class Params(BaseModel):
    """Parâmetros completos do pipeline.

    Attributes:
        data: Parâmetros de dados.
        dataset: Fonte do dataset de referência.
        quality: Parâmetros de qualidade de dados.
        train: Parâmetros de treino.
        drift: Parâmetros de simulação de drift.
        drift_thresholds: Limiares de drift.
        slo: Data SLOs e Error Budget.
        alerts: Regras de consolidação de alertas.
        retrain: Gatilho de re-treino.
        fairness: Guardrails de fairness.
        demo: Cenário do demo executável (US7).
    """

    data: DataParams
    dataset: DatasetParams
    quality: QualityParams
    train: TrainParams
    drift: DriftParams
    drift_thresholds: DriftThresholds
    slo: SLOParams
    alerts: AlertsParams
    retrain: RetrainParams
    fairness: FairnessParams
    demo: DemoParams = Field(default_factory=DemoParams)


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
