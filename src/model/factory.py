"""Factory de modelos e wrapper com metadados de treino."""

from dataclasses import dataclass

__all__ = ["TrainedModel"]


@dataclass
class TrainedModel:
    """Modelo treinado com metadados do pipeline.

    Attributes:
        estimator: Estimador scikit-learn/XGBoost treinado.
        feature_names: Ordem das features usadas no treino.
        metrics: Métricas de avaliação no conjunto de teste.
    """

    estimator: object
    feature_names: list[str]
    metrics: dict[str, float]
