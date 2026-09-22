"""Detecção de drift com Evidently AI.

Gera um relatório HTML comparando o dataset de referência com um
snapshot de produção. IMPORTANTE: o método de drift é configurado
explicitamente para PSI/KS — o default do Evidently para n > 1000 é
Wasserstein, que não é a métrica exigida pelo desafio.
"""

import logging

from src.core.dataset import DRIFT_REPORT_PATH

logger = logging.getLogger(__name__)

__all__ = ["build_evidently_report"]


def build_evidently_report(reference, production) -> str:
    """Gera o relatório HTML de drift com Evidently.

    Args:
        reference: DataFrame de referência.
        production: DataFrame de produção.

    Returns:
        Caminho do HTML gerado.
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    data_definition = DataDefinition()
    reference_ds = Dataset.from_pandas(reference, data_definition=data_definition)
    production_ds = Dataset.from_pandas(production, data_definition=data_definition)
    report = Report([DataDriftPreset(num_method="psi", cat_method="psi")])
    snapshot = report.run(production_ds, reference_ds)
    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(DRIFT_REPORT_PATH))
    logger.info("Relatório Evidently salvo em %s.", DRIFT_REPORT_PATH)
    return str(DRIFT_REPORT_PATH)
