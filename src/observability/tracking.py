"""Tracking de experimentos e ciclo de vida do modelo com MLflow.

Registra parâmetros, métricas e artefatos (relatórios de drift) de cada
execução, permitindo rastrear a evolução da degradação do modelo.
"""

import logging
import os
from contextlib import contextmanager

from src.core.config import get_settings

logger = logging.getLogger(__name__)

__all__ = ["tracking_run"]


@contextmanager
def tracking_run(name: str, params: dict | None = None):
    """Context manager de uma execução MLflow.

    Se o MLflow não estiver disponível, faz fallback silencioso
    (a etapa não deve quebrar só por ausência do tracking server).

    Args:
        name: Nome da execução.
        params: Parâmetros a registrar.

    Yields:
        Objeto de run do MLflow (ou None em fallback).
    """
    settings = get_settings()
    os.environ.setdefault("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)
    try:
        import mlflow

        with mlflow.start_run(run_name=name) as run:
            if params:
                mlflow.log_params({key: str(value) for key, value in params.items()})
            yield run
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow indisponível (%s); seguindo sem tracking.", exc)
        yield None
