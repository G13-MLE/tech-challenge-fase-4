"""Treino do classificador de risco de crédito.

Etapa 1. Treina um classificador (XGBoost ou LogisticRegression, conforme
``configs/params.yaml``) sobre o dataset de referência, persiste o artefato
em ``models/model.joblib`` (estimador + lista de features + identidade do
modelo) e registra as métricas de teste, incluindo Brier score e calibração.

A separação treino/validação usa ``src/data/split.py`` (``split_reference``)
— a mesma separação determinística coberta pelos testes de vazamento
(FR-007) — em vez de reimplementar o split no estágio de treino.
"""

import hashlib
import json
import logging

import joblib
import pandas as pd
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score

from src.core.dataset import (
    MODEL_PATH,
    REFERENCE_DATA_PATH,
    TRAIN_METRICS_PATH,
    atomic_write_text,
    load_parquet,
)
from src.core.params import load_params
from src.data.split import split_reference
from src.model.evaluate import calibration_bins
from src.model.factory import build_estimator
from src.observability.logging import TaskTimer

logger = logging.getLogger(__name__)

__all__ = ["build_estimator", "train", "main"]


def _model_version(classifier: str, features: list[str]) -> str:
    """Deriva um identificador determinístico do modelo (classificador + features)."""
    payload = json.dumps({"classifier": classifier, "features": features}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


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
    features = [column for column in frame.columns if column != target]

    train_frame, test_frame = split_reference(frame)
    x_train, y_train = train_frame[features], train_frame[target]
    x_test, y_test = test_frame[features], test_frame[target]

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
    bundle = {
        "model": model,
        "features": list(features),
        "model_name": params.classifier,
        "model_version": _model_version(params.classifier, list(features)),
    }
    joblib.dump(bundle, MODEL_PATH)
    metrics["calibration"] = calibration_bins(y_test.to_numpy(), probabilities)
    atomic_write_text(TRAIN_METRICS_PATH, json.dumps(metrics, indent=2))
    logger.info("Treino concluído. ROC-AUC: %.4f.", metrics["roc_auc"])
    return metrics


def main() -> None:
    """Ponto de entrada do estágio de treino."""

    with TaskTimer("treino") as timer:
        metrics = train()
        timer.record(n_rows=metrics["n_train"] + metrics["n_test"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
