"""Logger JSON-lines para observabilidade do pipeline.

Centraliza os logs de execução (sucesso/falha, latência, volume de dados,
modelo) em ``logs/pipeline.ndjson``, com correlação por ``run_id``
(pipeline) e ``trace_id`` (execução) para rastreabilidade fim-a-fim
(FR-018, FR-019). Alimenta o dashboard e o exporter Prometheus.
"""

import json
import logging
import os
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "LOG_PATH",
    "log_event",
    "TaskTimer",
    "new_run_id",
    "set_correlation",
    "get_correlation",
    "resolve_run_id",
]

LOG_PATH = Path("logs/pipeline.ndjson")

_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)


def new_run_id() -> str:
    """Gera um novo identificador de execução de pipeline.

    Returns:
        String UUID4 curta usada como ``run_id``.
    """
    return uuid.uuid4().hex[:12]


def resolve_run_id() -> str:
    """Resolve o ``run_id`` corrente, reaproveitando o do ambiente se houver.

    Returns:
        Identificador de pipeline (compartilhado entre estágios via
        ``PIPELINE_RUN_ID``).
    """
    current = _run_id.get() or os.environ.get("PIPELINE_RUN_ID")
    if not current:
        current = new_run_id()
    os.environ.setdefault("PIPELINE_RUN_ID", current)
    _run_id.set(current)
    return current


def set_correlation(run_id: str | None = None, trace_id: str | None = None) -> None:
    """Define o contexto de correlação (``run_id`` e/ou ``trace_id``).

    Args:
        run_id: Identificador do pipeline (compartilhado por todos os estágios).
        trace_id: Identificador da execução/estágio atual.
    """
    if run_id is not None:
        _run_id.set(run_id)
    if trace_id is not None:
        _trace_id.set(trace_id)
    elif _trace_id.get() is None:
        _trace_id.set(uuid.uuid4().hex[:12])


def get_correlation() -> dict[str, str | None]:
    """Retorna o contexto de correlação atual.

    Returns:
        Dicionário com ``run_id`` e ``trace_id``.
    """
    return {"run_id": _run_id.get(), "trace_id": _trace_id.get()}


def log_event(task: str, status: str, **fields: object) -> None:
    """Registra um evento estruturado em JSON-lines.

    Campos aceitos incluem ``latency_ms``, ``n_rows``, ``message``,
    ``model_name`` e ``model_version``. ``run_id``/``trace_id`` são
    adicionados automaticamente a partir do contexto de correlação.

    Args:
        task: Nome da tarefa/estágio do pipeline.
        status: Status do evento (started | success | failed).
        **fields: Campos adicionais do evento.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    correlation = get_correlation()
    event = {
        "task": task,
        "status": status,
        "timestamp": time.time(),
        "run_id": correlation["run_id"],
        "trace_id": correlation["trace_id"],
        **fields,
    }
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")
    logger.info("Evento registrado: %s %s", task, status)


class TaskTimer:
    """Context manager que mede e registra a latência de uma tarefa."""

    def __init__(self, task: str, run_id: str | None = None, **fields: object) -> None:
        """Inicializa o medidor para a tarefa informada.

        Args:
            task: Nome da tarefa/estágio.
            run_id: Identificador de pipeline; propaga para o contexto.
            **fields: Campos extras incluídos no evento de término.
        """
        self.task = task
        self.run_id = run_id
        self.fields = fields
        self.start = 0.0

    def record(self, **fields: object) -> None:
        """Adiciona campos extras ao evento de término (ex.: ``n_rows``).

        Permite registrar métricas conhecidas apenas após o início da
        tarefa (ex.: volume processado) no evento ``success``/``failed``.

        Args:
            **fields: Campos a acrescentar ao evento de término.
        """
        self.fields.update(fields)

    def __enter__(self) -> "TaskTimer":
        """Marca o início e registra o evento de início.

        Returns:
            A própria instância.
        """
        run_id = self.run_id or os.environ.get("PIPELINE_RUN_ID")
        if run_id is None:
            run_id = resolve_run_id()
        set_correlation(run_id=run_id)
        self.start = time.perf_counter()
        log_event(self.task, "started")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Registra o término com latência, status e campos extras."""
        latency_ms = (time.perf_counter() - self.start) * 1000
        status = "failed" if exc_type else "success"
        log_event(self.task, status, latency_ms=round(latency_ms, 2), **self.fields)
