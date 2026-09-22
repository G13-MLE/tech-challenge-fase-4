"""Logger JSON-lines para observabilidade do pipeline.

Centraliza os logs de execução (sucesso/falha, latência, volume de
dados) em `logs/pipeline.ndjson`, alimentando o dashboard e o
ambiente de tracking.
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["LOG_PATH", "log_event", "TaskTimer"]

LOG_PATH = Path("logs/pipeline.ndjson")


def log_event(task: str, status: str, **fields: object) -> None:
    """Registra um evento estruturado em JSON-lines.

    Args:
        task: Nome da tarefa/estágio do pipeline.
        status: Status do evento (started | success | failed).
        **fields: Campos adicionais (latency, n_rows, message, etc.).
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = {"task": task, "status": status, "timestamp": time.time(), **fields}
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")
    logger.info("Evento registrado: %s %s", task, status)


class TaskTimer:
    """Context manager que mede e registra a latência de uma tarefa."""

    def __init__(self, task: str) -> None:
        """Inicializa o medidor para a tarefa informada.

        Args:
            task: Nome da tarefa/estágio.
        """
        self.task = task
        self.start = 0.0

    def __enter__(self) -> "TaskTimer":
        """Marca o início e registra o evento de início.

        Returns:
            A própria instância.
        """
        self.start = time.perf_counter()
        log_event(self.task, "started")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Registra o término com latência e status."""
        latency_ms = (time.perf_counter() - self.start) * 1000
        status = "failed" if exc_type else "success"
        log_event(self.task, status, latency_ms=round(latency_ms, 2))
