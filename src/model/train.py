"""Treino do classificador de risco de crédito.

Etapa 1. Treina um classificador (XGBoost ou LogisticRegression, conforme
``configs/params.yaml``) sobre o dataset de referência, persiste o artefato
em ``models/model.joblib`` (estimador + lista de features) e registra as
métricas de teste, incluindo Brier score e calibração.
"""

import json
import logging

import joblib
import pandas as pd
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.core.dataset import (
    MODEL_PATH,
    REFERENCE_DATA_PATH,
    TRAIN_METRICS_PATH,
    atomic_write_text,
    load_parquet,
)
from src.core.params import load_params
from src.model.evaluate import calibration_bins
from src.model.factory import build_estimator
from src.observability.logging import TaskTimer

logger = logging.getLogger(__name__)

__all__ = ["build_estimator", "train", "main"]


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
        "brier": float(brier_score_loss(y_test, probabilities)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "classifier": params.classifier,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(features.columns)}, MODEL_PATH)
    metrics["calibration"] = calibration_bins(y_test.to_numpy(), probabilities)
    atomic_write_text(TRAIN_METRICS_PATH, json.dumps(metrics, indent=2))
    logger.info("Treino concluído. ROC-AUC: %.4f.", metrics["roc_auc"])
    return metrics


def main() -> None:
    """Ponto de entrada do estágio de treino."""

    with TaskTimer("treino"):
        train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
