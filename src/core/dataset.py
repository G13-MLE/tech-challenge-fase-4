"""Caminhos e utilitários de persistência do pipeline.

Centraliza os caminhos de dados, modelo e relatórios usados pelos
estágios do DVC, evitando literais espalhados pelo código. Todas as
escritas são atômicas (write-to-temp-then-rename).
"""

from pathlib import Path

import pandas as pd

__all__ = [
    "DATA_RAW_PATH",
    "REFERENCE_DATA_PATH",
    "PRODUCTION_DIR",
    "MODEL_PATH",
    "VALIDATION_RESULT_PATH",
    "TRAIN_METRICS_PATH",
    "DRIFT_REPORT_PATH",
    "STATISTICAL_REPORT_PATH",
    "RUN_SUMMARY_PATH",
    "SLO_REPORT_PATH",
    "RETRAIN_DECISION_PATH",
    "PROFILE_HTML_PATH",
    "FAIRNESS_DIR",
    "PERFORMANCE_PATH",
    "ID_COLUMN",
    "TARGET_COLUMN",
    "load_parquet",
    "save_parquet",
    "atomic_write_text",
    "atomic_write_bytes",
]

DATA_RAW_PATH = Path("data/raw/reference.csv")
REFERENCE_DATA_PATH = Path("data/reference/reference.parquet")
PRODUCTION_DIR = Path("data/production")
MODEL_PATH = Path("models/model.joblib")

VALIDATION_RESULT_PATH = Path("reports/validation/result.json")
PROFILE_HTML_PATH = Path("reports/validation/profile.html")
TRAIN_METRICS_PATH = Path("reports/train_metrics.json")
DRIFT_REPORT_PATH = Path("reports/drift/drift_report.html")
STATISTICAL_REPORT_PATH = Path("reports/drift/statistical_report.json")
PERFORMANCE_PATH = Path("reports/drift/performance.json")
SLO_REPORT_PATH = Path("reports/monitoring/slo_report.json")
RETRAIN_DECISION_PATH = Path("reports/monitoring/retrain_decision.json")
RUN_SUMMARY_PATH = Path("reports/monitoring/run_summary.json")
FAIRNESS_DIR = Path("reports/fairness")

ID_COLUMN = "ID"
TARGET_COLUMN = "default_next_month"


def load_parquet(path: Path) -> pd.DataFrame:
    """Carrega um DataFrame a partir de um arquivo Parquet.

    Args:
        path: Caminho do arquivo Parquet.

    Returns:
        DataFrame carregado.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Persiste um DataFrame em Parquet de forma atômica.

    Escreve em arquivo temporário e renomeia, evitando arquivos
    parciais em caso de falha.

    Args:
        df: DataFrame a ser persistido.
        path: Caminho do arquivo Parquet de saída.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.parquet")
    try:
        df.to_parquet(tmp_path, index=False)
        tmp_path.rename(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, content: str) -> None:
    """Escreve conteúdo textual de forma atômica.

    Args:
        path: Caminho do arquivo de saída.
        content: Conteúdo a ser escrito.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.rename(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Escreve conteúdo binário de forma atômica.

    Args:
        path: Caminho do arquivo de saída.
        content: Conteúdo binário a ser escrito.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_bytes(content)
        tmp_path.rename(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
