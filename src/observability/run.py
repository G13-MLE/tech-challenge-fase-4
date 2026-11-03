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
    HEALTH_HTML_PATH,
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
from src.core.params import load_params
from src.data.split import split_reference
from src.model.factory import build_estimator
from src.model.predict import load_model, predict_proba
from src.observability.alerts import build_alerts
from src.observability.fairness import (
    build_fairness_report,
    build_fairness_time_series,
    evaluate_fairness,
)
from src.observability.logging import LOG_PATH, TaskTimer
from src.observability.metrics import aggregate_health
from src.observability.report_html import render_health_html
from src.observability.retrain import decide_retrain
from src.observability.slo import MonitoringPipeline
from src.observability.tracking import MODEL_NAME, log_model_run

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


def _evaluate_challenger() -> tuple[float | None, dict | None]:
    """Treina e avalia um challenger (re-treino ingênuo sobre dados driftados).

    Simula um re-treino feito sobre o último snapshot de produção (já
    contaminado pelo drift) e o avalia no holdout de **referência**. Esse
    re-treino ingênuo tende a degradar o ROC-AUC, e o guardrail de
    promoção deve bloqueá-lo (FR-017, SC-007).

    Returns:
        Tupla ``(challenger_auc, fairness)``; ``(None, None)`` se não houver
        snapshots ou dataset de referência.
    """
    from sklearn.metrics import roc_auc_score

    paths = sorted(PRODUCTION_DIR.glob("prod_month_*.parquet"))
    if not paths or not REFERENCE_DATA_PATH.exists():
        return None, None
    reference = load_parquet(REFERENCE_DATA_PATH)
    target = load_params().data.target_column
    features = [column for column in reference.columns if column != target]
    _, holdout = split_reference(reference)
    production = load_parquet(paths[-1])
    model = build_estimator()
    model.fit(production[features], production[target])  # type: ignore[attr-defined]
    probabilities = model.predict_proba(holdout[features])[:, 1]  # type: ignore[attr-defined]
    challenger_auc = float(roc_auc_score(holdout[target], probabilities))
    challenger_fairness = evaluate_fairness(holdout, probabilities, target)
    logger.info("Challenger (re-treino em drift): ROC-AUC %.4f.", challenger_auc)
    return challenger_auc, challenger_fairness


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
    challenger_auc, challenger_fairness = _evaluate_challenger()
    retrain = decide_retrain(
        statistical,
        train_metrics,
        fairness_report=({"by_group": challenger_fairness} if challenger_fairness else fairness),
        challenger_auc=challenger_auc,
    )
    health["slo_status"] = slo_report["status_geral"]

    tracking = _track(train_metrics, statistical, retrain)

    n_rows = len(load_parquet(REFERENCE_DATA_PATH)) if REFERENCE_DATA_PATH.exists() else 0
    summary = {
        "health": health,
        "alerts": alerts,
        "n_alerts": len(alerts),
        "slo": slo_report,
        "retrain_decision": retrain["decision"],
        "challenger_auc": challenger_auc,
        "fairness_passed": fairness.get("passes"),
        "fairness_mitigation": {
            "strategy": fairness.get("mitigation", {}).get("strategy"),
            "passes_before": fairness.get("mitigation", {}).get("passes_before"),
            "passes_after": fairness.get("mitigation", {}).get("passes_after"),
        },
        "tracking": tracking,
        "n_rows": int(n_rows),
    }
    atomic_write_text(RUN_SUMMARY_PATH, json.dumps(summary, indent=2))
    fairness_path = FAIRNESS_DIR / "fairness_report.json"
    fairness_evidence = _load_json(fairness_path)
    atomic_write_text(
        HEALTH_HTML_PATH,
        render_health_html(summary, statistical, performance, slo_report, fairness_evidence),
    )
    logger.info(
        "Observabilidade consolidada: %d alertas, SLO=%s, retrain=%s.",
        len(alerts),
        slo_report["status_geral"],
        retrain["decision"],
    )
    return summary


def _apply_registry_aliases(version: str) -> None:
    """Define os aliases do Model Registry (FR-020), sem promoção automática (FR-017).

    O modelo recém-treinado é sempre registrado como ``challenger``. O
    ``champion`` **nunca** é trocado automaticamente: mantém o vigente (ou
    faz *bootstrap* do primeiro modelo quando ainda não houver champion).
    A promoção de um challenger elegível é uma ação explícita.
    """
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(MODEL_NAME, "challenger", version)
        try:
            client.get_model_version_by_alias(MODEL_NAME, "champion")
        except Exception:  # noqa: BLE001 - sem champion prévio: bootstrap
            client.set_registered_model_alias(MODEL_NAME, "champion", version)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Aliases champion/challenger não definidos (%s).", exc)


def _track(train_metrics: dict, statistical: dict, retrain: dict) -> dict:
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
        alias="challenger",
        dataset_path=REFERENCE_DATA_PATH,
        run_name="monitoramento",
    )
    if result.get("tracked") and result.get("model_version"):
        _apply_registry_aliases(result["model_version"])
        if retrain["guardrails"].get("eligible"):
            logger.info(
                "Challenger elegível à promoção (guardrails OK); promoção exige ação "
                "explícita (FR-017)."
            )
    return result


def main() -> None:
    """Ponto de entrada do estágio de monitoramento."""
    with TaskTimer("monitoramento") as timer:
        summary = run()
        timer.record(n_rows=summary.get("n_rows"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
