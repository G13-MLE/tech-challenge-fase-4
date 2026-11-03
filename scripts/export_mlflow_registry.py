"""Exporta o estado do MLflow Registry para evidência versionada (Etapa 3).

O `mlruns/` é ignorado pelo Git (banco do MLflow), então um avaliador não
consegue ver os runs/models apenas clonando o repositório. Este script grava
um retrato determinístico do Registry — versões, aliases `champion`/
`challenger`, run IDs e métricas — em
``reports/monitoring/mlflow_registry.json``.

Requer o servidor MLflow no ar (``make mlflow-up``). É uma exportação de
leitura: não cria runs nem altera o Registry.
"""

import json
import logging
import os
from pathlib import Path

import mlflow

from src.core.dataset import atomic_write_text
from src.observability.tracking import MODEL_NAME

logger = logging.getLogger(__name__)

__all__ = ["export_registry", "main"]

REGISTRY_EXPORT_PATH = Path("reports/monitoring/mlflow_registry.json")


def _version_payload(version) -> dict:
    """Resume uma versão do Registry em campos serializáveis."""
    return {
        "version": version.version,
        "run_id": version.run_id,
        "status": version.status,
        "creation_timestamp": version.creation_timestamp,
        "current_stage": version.current_stage,
        "aliases": list(getattr(version, "aliases", []) or []),
    }


def export_registry(path=REGISTRY_EXPORT_PATH) -> dict:
    """Lê o Registry e grava o retrato em JSON de forma atômica.

    Args:
        path: Caminho do JSON de saída.

    Returns:
        Dicionário exportado (também gravado em ``path``).
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient()

    versions = sorted(
        client.search_model_versions(f"name='{MODEL_NAME}'"),
        key=lambda version: int(version.version),
    )
    aliases: dict[str, str] = {}
    for alias in ("champion", "challenger"):
        try:
            aliases[alias] = client.get_model_version_by_alias(MODEL_NAME, alias).version
        except Exception:  # noqa: BLE001 - alias ausente é estado válido
            aliases[alias] = None

    latest = versions[-1] if versions else None
    metrics: dict[str, float] = {}
    if latest is not None:
        run = client.get_run(latest.run_id)
        metrics = {key: float(value) for key, value in run.data.metrics.items()}

    payload = {
        "tracking_uri": uri,
        "model_name": MODEL_NAME,
        "aliases": aliases,
        "n_versions": len(versions),
        "latest_version": latest.version if latest else None,
        "latest_run_id": latest.run_id if latest else None,
        "latest_metrics": metrics,
        "versions": [_version_payload(version) for version in versions],
    }
    atomic_write_text(path, json.dumps(payload, indent=2))
    logger.info("Registry exportado: %s (v%s).", path, payload["latest_version"])
    return payload


def main() -> None:
    """Ponto de entrada da exportação do Registry."""
    export_registry()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
