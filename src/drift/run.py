"""Orquestra a Etapa 2: simulação, detecção e avaliação de drift.

Gera os snapshots (Data + Concept Drift), roda os testes PSI/KS/χ² com
correção de múltiplas comparações, o teste multivariado (MMD²/Energy), o
relatório Evidently (PSI/χ²), o gate de drift e mede a degradação de
performance (Concept Drift) por mês. Todos os relatórios são escritos
atomicamente (FR-008–FR-014).
"""

import json
import logging

from src.core.dataset import (
    PERFORMANCE_PATH,
    REFERENCE_DATA_PATH,
    STATISTICAL_REPORT_PATH,
    STATISTICAL_SUMMARY_PATH,
    TRAIN_METRICS_PATH,
    atomic_write_text,
    load_parquet,
)
from src.core.params import load_params
from src.drift.concept import (
    evaluate_concept_control,
    evaluate_snapshots,
    fit_concept_model,
    prior_label_drift,
)
from src.drift.detect import build_evidently_report, build_test_suite_result
from src.drift.multivariate import build_multivariate_report
from src.drift.report import render_statistical_summary
from src.drift.simulate import simulate_snapshots
from src.drift.stats import build_statistical_report

logger = logging.getLogger(__name__)

__all__ = ["run", "main"]


def _history_entry(month: int, report: dict) -> dict:
    """Resume um relatório mensal para a série temporal de PSI."""
    features = report["features"]
    return {
        "month": month,
        "n_drifted": report["n_drifted"],
        "n_severe": report["n_severe"],
        "max_psi": report["max_psi"],
        "is_breach": report["is_breach"],
        "features": {
            name: {"psi": entry["psi"], "severity": entry["severity"]}
            for name, entry in features.items()
        },
    }


def run() -> dict:
    """Executa o pipeline completo de drift.

    Returns:
        Dicionário com caminhos e resultados de performance.
    """
    reference = load_parquet(REFERENCE_DATA_PATH)
    baseline_auc: float | None = None
    if TRAIN_METRICS_PATH.exists():
        baseline_auc = float(
            json.loads(TRAIN_METRICS_PATH.read_text(encoding="utf-8")).get("roc_auc", 0.0)
        )
    paths = simulate_snapshots()
    snapshots = [load_parquet(path) for path in paths]

    history: list[dict] = []
    last_report: dict = {}
    for month, snapshot in enumerate(snapshots, start=1):
        report = build_statistical_report(reference, snapshot, write=False)
        history.append(_history_entry(month, report))
        last_report = report

    last_snapshot = snapshots[-1]
    last_report["history"] = history
    last_report["month"] = len(snapshots)
    last_report["multivariate"] = build_multivariate_report(reference, last_snapshot)

    # Discrimina os três tipos de drift.
    label = prior_label_drift(reference, last_snapshot)
    performance = evaluate_snapshots(snapshots, baseline_auc=baseline_auc)
    # Controle do gerador de rótulos: isola o Concept Drift real do custo de
    # substituir os rótulos originais pelo substituto logístico (progress=0).
    params = load_params()
    concept_control = evaluate_concept_control(
        snapshots,
        fit_concept_model(reference),
        strength=params.drift.concept_drift_strength,
    )
    label_by_month = {
        month: prior_label_drift(reference, snapshot)
        for month, snapshot in enumerate(snapshots, start=1)
    }
    for entry, control_auc in zip(performance, concept_control, strict=True):
        drift = label_by_month[entry["month"]]
        entry["p_positive"] = drift["production_positive_rate"]
        entry["label_drift"] = drift["label_drift"]
        entry["control_auc"] = control_auc
        entry["concept_drift_adjusted"] = float(control_auc - entry["roc_auc"])
    last_report["drift_types"] = {
        "data_drift": bool(last_report["n_drifted"] > 0),
        "label_drift": bool(abs(label["label_drift"]) > 0.02),
        "concept_drift": bool(performance[-1]["concept_drift_adjusted"] > 0.0),
        "multivariate_drift": bool(last_report["multivariate"]["drifted"]),
    }
    atomic_write_text(STATISTICAL_REPORT_PATH, json.dumps(last_report, indent=2))
    atomic_write_text(PERFORMANCE_PATH, json.dumps(performance, indent=2))
    atomic_write_text(
        STATISTICAL_SUMMARY_PATH, render_statistical_summary(last_report, performance)
    )

    build_evidently_report(reference, last_snapshot)
    build_test_suite_result(reference, last_snapshot, last_report)

    summary = {
        "n_snapshots": len(snapshots),
        "n_rows": int(sum(len(snapshot) for snapshot in snapshots)),
        "statistical_report": str(STATISTICAL_REPORT_PATH),
        "statistical_summary": str(STATISTICAL_SUMMARY_PATH),
        "performance_report": str(PERFORMANCE_PATH),
        "is_breach": last_report["is_breach"],
        "auc_last_month": performance[-1]["roc_auc"],
        "auc_drop": performance[-1]["concept_drift"],
        "auc_drop_adjusted": performance[-1]["concept_drift_adjusted"],
    }
    logger.info("Drift concluído com %d snapshots.", len(snapshots))
    return summary


def main() -> None:
    """Ponto de entrada do estágio de drift."""
    from src.observability.logging import TaskTimer

    with TaskTimer("drift") as timer:
        summary = run()
        timer.record(n_rows=summary.get("n_rows"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
