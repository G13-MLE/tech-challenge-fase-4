"""Treino do classificador de risco de crédito.

Etapa 1. Treina um classificador (XGBoost ou LogisticRegression, conforme
`configs/params.yaml`) sobre o dataset de referência, persiste o artefato
em `models/model.joblib` e registra as métricas de teste.
"""

import json
import logging

import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.core.dataset import (
    MODEL_PATH,
    REFERENCE_DATA_PATH,
    TRAIN_METRICS_PATH,
    atomic_write_text,
    load_parquet,
)
from src.core.params import load_params

logger = logging.getLogger(__name__)

__all__ = ["build_estimator", "train", "main"]


def build_estimator() -> object:
    """Constrói o estimador configurado em `configs/params.yaml`.

    Returns:
        Estimador scikit-learn compatível (XGBoost ou LogisticRegression).
    """
    params = load_params().train
    if params.classifier == "xgboost":
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
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=params.logistic_regression_C,
        max_iter=params.logistic_regression_max_iter,
        class_weight=params.class_weight,
        random_state=params.seed,
    )


def train(frame: pd.DataFrame | None = None) -> dict[str, float]:
    """Treina o classificador e persiste o artefato e as métricas.

    Args:
        frame: DataFrame de entrada; se None, carrega o de referência.

    Returns:
        Dicionário de métricas de teste.
    """
    if frame is None:
        frame = load_parquet(REFERENCE_DATA_PATH)
    params = load_params().train
    target = load_params().data.target_column
    features = frame.drop(columns=[target])
    labels = frame[target]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=params.test_size,
        random_state=params.seed,
        stratify=labels,
    )
    model = build_estimator()
    model.fit(x_train, y_train)  # type: ignore[attr-defined]
    probabilities = model.predict_proba(x_test)[:, 1]  # type: ignore[attr-defined]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(features.columns)}, MODEL_PATH)
    atomic_write_text(TRAIN_METRICS_PATH, json.dumps(metrics, indent=2))
    logger.info("Treino concluído. ROC-AUC: %.4f.", metrics["roc_auc"])
    return metrics


def main() -> None:
    """Ponto de entrada do estágio de treino."""
    train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
