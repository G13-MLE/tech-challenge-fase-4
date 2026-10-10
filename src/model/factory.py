"""Factory de modelos do pipeline.

Centraliza a construção do estimador configurado em ``configs/params.yaml``
(``xgboost`` ou ``logistic_regression``) para que treino e re-treino
compartilhem a mesma definição (FR-006).
"""

from src.core.params import load_params

__all__ = ["build_estimator"]


def build_estimator(name: str | None = None) -> object:
    """Constrói o estimador configurado em ``configs/params.yaml``.

    Args:
        name: Classificador a instanciar ("xgboost" ou
            "logistic_regression"); se None, usa o valor de
            ``configs/params.yaml``.

    Returns:
        Estimador scikit-learn compatível (XGBoost ou LogisticRegression).

    Raises:
        ValueError: Se o classificador solicitado for desconhecido.
    """
    params = load_params().train
    classifier = name or params.classifier
    if classifier == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=params.xgboost_n_estimators,
            max_depth=params.xgboost_max_depth,
            learning_rate=params.xgboost_learning_rate,
            subsample=params.xgboost_subsample,
            colsample_bytree=params.xgboost_colsample_bytree,
            random_state=params.seed,
            eval_metric="logloss",
        )
    if classifier == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=params.logistic_regression_C,
            max_iter=params.logistic_regression_max_iter,
            class_weight=params.class_weight,
            random_state=params.seed,
        )
    raise ValueError(f"Classificador desconhecido: {classifier!r}.")
