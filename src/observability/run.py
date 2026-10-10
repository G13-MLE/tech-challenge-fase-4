"""Orquestra a Etapa 3: SLO, alertas, fairness e tracking.

Consome os relatórios de drift/treino, agrega as métricas de saúde,
consolida alertas (anti-fadiga), avalia Data SLOs + Error Budget, mede
fairness por subgrupo, decide o re-treino com guardrails e registra o
modelo no MLflow. Escreve ``reports/monitoring/run_summary.json`` e
log estruturado com ``run_id``/``trace_id`` (FR-015–FR-022).
"""

import json
import logging

from src.core.dataset import (
    DRIFT_REPORT_PATH,
    FAIRNESS_DIR,
    MODEL_PATH,
    PERFORMANCE_PATH,
    PRODUCTION_DIR,
    REFERENCE_DATA_PATH,
    RETRAIN_DECISION_PATH,
    RUN_SUMMARY_PATH,
    SLO_REPORT_PATH,
    STATISTICAL_REPORT_PATH,
    TARGET_COLUMN,
    TRAIN_METRICS_PATH,
    VALIDATION_RESULT_PATH,
    atomic_write_text,
    load_parquet,
)
from src.model.predict import load_model, predict_proba
from src.observability.alerts import build_alerts
from src.observability.fairness import build_fairness_report, build_fairness_time_series
from src.observability.logging import LOG_PATH, TaskTimer
from src.observability.metrics import aggregate_health
from src.observability.retrain import decide_retrain
from src.observability.slo import MonitoringPipeline
from src.observability.tracking import log_model_run

logger = logging.getLogger(__name__)

__all__ = ["run", "main"]


def _load_json(path):
    """Carrega um JSON se existir."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _missing_rate(validation: dict) -> float:
    """Deriva a taxa de ausência do relatório de validação."""
    completeness = validation.get("quality", {}).get("completeness", 1.0)
    return float(1.0 - completeness)


def _evaluate_fairness() -> dict:
    """Avalia fairness por subgrupo no último snapshot e ao longo dos meses."""
    paths = sorted(PRODUCTION_DIR.glob("prod_month_*.parquet"))
    if not paths or not MODEL_PATH.exists():
        return {}
    bundle = load_model()
    frames = [load_parquet(path) for path in paths]
    probabilities = [predict_proba(frame, bundle) for frame in frames]
    build_fairness_time_series(frames, probabilities, TARGET_COLUMN)
    return build_fairness_report(frames[-1], probabilities[-1], TARGET_COLUMN)


def run() -> dict:
    """Executa a consolidação de observabilidade.

    Returns:
        Resumo de saúde do pipeline com alertas, SLO, retrain e fairness.
    """
    statistical = _load_json(STATISTICAL_REPORT_PATH)
    performance = _load_json(PERFORMANCE_PATH)
    if not isinstance(performance, list):
        performance = []
    train_metrics = _load_json(TRAIN_METRICS_PATH)
    validation = _load_json(VALIDATION_RESULT_PATH)

    health = aggregate_health(statistical, performance, LOG_PATH)
    alerts = build_alerts(statistical, health, history=statistical.get("history"))
    slo_report = MonitoringPipeline().evaluate(
        statistical, performance, alerts, missing_rate=_missing_rate(validation)
    )
    fairness = _evaluate_fairness()
    retrain = decide_retrain(statistical, train_metrics, fairness)
    health["slo_status"] = slo_report["status_geral"]

    tracking = _track(train_metrics, statistical)

    summary = {
        "health": health,
        "alerts": alerts,
        "n_alerts": len(alerts),
        "slo": slo_report,
        "retrain_decision": retrain["decision"],
        "fairness_passed": fairness.get("passes"),
        "tracking": tracking,
    }
    atomic_write_text(RUN_SUMMARY_PATH, json.dumps(summary, indent=2))
    logger.info(
        "Observabilidade consolidada: %d alertas, SLO=%s, retrain=%s.",
        len(alerts),
        slo_report["status_geral"],
        retrain["decision"],
    )
    return summary


def _track(train_metrics: dict, statistical: dict) -> dict:
    """Registra o modelo/artefatos no MLflow (fallback silencioso)."""
    if not MODEL_PATH.exists():
        return {"tracked": False}
    bundle = load_model()
    artifacts = [
        STATISTICAL_REPORT_PATH,
        PERFORMANCE_PATH,
        DRIFT_REPORT_PATH,
        SLO_REPORT_PATH,
        RETRAIN_DECISION_PATH,
        FAIRNESS_DIR / "fairness_report.json",
    ]
    features = bundle["features"]
    input_example = load_parquet(REFERENCE_DATA_PATH)[features].head(5)
    result = log_model_run(
        bundle=bundle,
        metrics=train_metrics,
        params={"n_features": len(features), "max_psi": statistical.get("max_psi", 0.0)},
        input_example=input_example,
        artifacts=artifacts,
        alias="champion",
        dataset_path=REFERENCE_DATA_PATH,
        run_name="monitoramento",
    )
    if result.get("tracked") and result.get("model_version"):
        try:
            import mlflow

            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias("credit_scoring", "champion", result["model_version"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Alias champion não definido (%s).", exc)
    return result


def main() -> None:
    """Ponto de entrada do estágio de monitoramento."""
    with TaskTimer("monitoramento"):
        run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
