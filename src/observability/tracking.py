"""Tracking de experimentos e ciclo de vida do modelo com MLflow.

Registra parâmetros, métricas e artefatos (relatórios de drift/validação),
assina o modelo (``infer_signature``), registra um ``input_example`` e o
hash SHA-256 do dataset de entrada, publica no Model Registry e aplica
aliases ``champion``/``challenger``. Aplica *gates* por artefato. Faz
fallback silencioso quando o MLflow não está disponível (FR-021, FR-022).
"""

import hashlib
import logging
import os
from contextlib import contextmanager
from pathlib import Path

from src.core.config import get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "tracking_run",
    "dataset_digest",
    "artifact_gate",
    "log_model_run",
    "MODEL_NAME",
]

MODEL_NAME = "credit_scoring"
EXPERIMENT_NAME = "credit_drift_monitor"
_SKOPS_TRUSTED_TYPES = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]


def _configure_mlflow_env() -> None:
    """Configura o cliente MLflow para falhar rápido quando offline."""
    settings = get_settings()
    os.environ.setdefault("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")


def dataset_digest(path: Path) -> str:
    """Calcula o digest SHA-256 de um arquivo de dataset.

    Args:
        path: Caminho do arquivo.

    Returns:
        Hash hexadecimal; string vazia se o arquivo não existir.
    """
    if not path.exists():
        logger.warning("Dataset %s inexistente; digest vazio.", path)
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_input_dataset(path: Path):
    """Constrói um ``mlflow.data.Dataset`` a partir do arquivo do dataset.

    O campo ``digest`` do MLflow tem limite de 36 caracteres e não aceita
    um SHA-256 integral; o hash completo é registrado como parâmetro da
    run (``dataset_sha256``) e o ``Dataset`` usa o digest nativo do MLflow.

    Args:
        path: Caminho do dataset (Parquet ou CSV).

    Returns:
        ``mlflow.data.Dataset`` ou ``None`` se o arquivo não existir.
    """
    if not path.exists():
        logger.warning("Dataset %s inexistente; log_input ignorado.", path)
        return None
    import pandas as pd
    from mlflow.data.pandas_dataset import from_pandas

    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return from_pandas(frame, source=str(path), name=path.name)


def artifact_gate(paths: list[Path]) -> dict:
    """Verifica se todos os artefatos obrigatórios existem (gate).

    Args:
        paths: Lista de caminhos de artefatos requeridos.

    Returns:
        Dicionário ``{passed, missing}``.
    """
    missing = [str(path) for path in paths if not path.exists()]
    return {"passed": not missing, "missing": missing}


@contextmanager
def tracking_run(name: str, params: dict | None = None, experiment: str = EXPERIMENT_NAME):
    """Context manager de uma execução MLflow.

    Se o MLflow não estiver disponível, faz fallback silencioso (a etapa
    não deve quebrar só por ausência do tracking server). O fallback cobre
    apenas falhas na abertura da run; exceções lançadas pelo corpo do
    ``with`` propagam normalmente (a run é encerrada como FAILED). Um
    gerador de contextmanager não pode dar um segundo ``yield`` após
    receber um ``throw`` — isso levanta ``RuntimeError`` e mascara o erro
    original.

    Args:
        name: Nome da execução.
        params: Parâmetros a registrar.
        experiment: Nome do experimento MLflow.

    Yields:
        Objeto de run do MLflow (ou None em fallback).
    """
    _configure_mlflow_env()
    mlflow = None
    run = None
    try:
        import mlflow

        mlflow.set_experiment(experiment)
        run = mlflow.start_run(run_name=name)
        if params:
            mlflow.log_params({key: str(value) for key, value in params.items()})
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow indisponível (%s); seguindo sem tracking.", exc)
        if mlflow is not None and run is not None:
            mlflow.end_run(status="FAILED")
        yield None
        return
    finished = True
    try:
        yield run
    except BaseException:
        finished = False
        raise
    finally:
        mlflow.end_run(status="FINISHED" if finished else "FAILED")


def log_model_run(
    bundle: dict,
    metrics: dict,
    params: dict | None = None,
    input_example=None,
    artifacts: list[Path] | None = None,
    alias: str = "challenger",
    dataset_path: Path | None = None,
    run_name: str = "treino",
) -> dict:
    """Publica o modelo e artefatos no MLflow com Registry e aliases.

    Args:
        bundle: Dicionário com ``model`` e ``features``.
        metrics: Métricas a registrar.
        params: Parâmetros a registrar.
        input_example: Exemplo de entrada para o ``input_example``.
        artifacts: Artefatos a anexar.
        alias: Alias de registro (``champion``/``challenger``).
        dataset_path: Dataset de entrada (para registrar o digest SHA-256).
        run_name: Nome da execução.

    Returns:
        Dicionário ``{tracked, model_version, uri, gates}``.
    """
    result: dict = {"tracked": False, "model_version": None, "uri": None, "gates": {}}
    artifacts = artifacts or []
    result["gates"] = artifact_gate(artifacts)
    _configure_mlflow_env()
    try:
        import mlflow
        from mlflow.models import infer_signature

        estimator = bundle["model"]
        with tracking_run(run_name, params=params) as run:
            if run is None:
                return result
            numeric_metrics = {
                key: float(value) for key, value in metrics.items() if _is_number(value)
            }
            mlflow.log_metrics(numeric_metrics)
            if input_example is not None:
                predictions = estimator.predict(input_example)
                signature = infer_signature(input_example, predictions)
            else:
                signature = None
            if dataset_path is not None:
                dataset_path = Path(dataset_path)
                digest = dataset_digest(dataset_path)
                dataset = _build_input_dataset(dataset_path)
                if dataset is not None:
                    mlflow.log_input(dataset, context="training")
                if digest:
                    mlflow.log_param("dataset_sha256", digest)
            model_info = mlflow.sklearn.log_model(
                sk_model=estimator,
                name="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=MODEL_NAME,
                skops_trusted_types=_SKOPS_TRUSTED_TYPES,
            )
            for artifact in artifacts:
                if artifact.exists():
                    mlflow.log_artifact(str(artifact))
            result.update(
                tracked=True,
                model_version=str(getattr(model_info, "registered_model_version", "")),
                uri=str(model_info.model_uri),
            )
        try:
            client = mlflow.tracking.MlflowClient()
            version = result.get("model_version")
            if version:
                client.set_registered_model_alias(MODEL_NAME, alias, version)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Não foi possível definir o alias %s (%s).", alias, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow indisponível (%s); seguindo sem tracking.", exc)
    return result


def _is_number(value: object) -> bool:
    """Indica se um valor é numérico (registrável como métrica)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)
