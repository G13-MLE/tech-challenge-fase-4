"""Orquestra a Etapa 2: simulação, detecção e avaliação de drift.

Gera os snapshots, roda os testes PSI/KS, produz o relatório Evidently
e mede a degradação de performance (Concept Drift).
"""

import json
import logging

from src.core.dataset import (
    STATISTICAL_REPORT_PATH,
    atomic_write_text,
    load_parquet,
)
from src.drift.concept import evaluate_snapshots
from src.drift.detect import build_evidently_report
from src.drift.simulate import simulate_snapshots
from src.drift.stats import build_statistical_report

logger = logging.getLogger(__name__)

PERFORMANCE_PATH = STATISTICAL_REPORT_PATH.parent / "performance.json"

__all__ = ["run", "main"]


def run() -> dict:
    """Executa o pipeline completo de drift.

    Returns:
        Dicionário com caminhos e resultados de performance.
    """
    from src.core.dataset import REFERENCE_DATA_PATH

    reference = load_parquet(REFERENCE_DATA_PATH)
    paths = simulate_snapshots()
    snapshots = [load_parquet(path) for path in paths]
    build_evidently_report(reference, snapshots[-1])
    build_statistical_report(reference, snapshots[-1])
    performance = evaluate_snapshots(snapshots)
    atomic_write_text(PERFORMANCE_PATH, json.dumps(performance, indent=2))
    summary = {
        "n_snapshots": len(snapshots),
        "statistical_report": str(STATISTICAL_REPORT_PATH),
        "performance_by_month": performance,
    }
    logger.info("Drift concluído com %d snapshots.", len(snapshots))
    return summary


def main() -> None:
    """Ponto de entrada do estágio de drift."""
    run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
