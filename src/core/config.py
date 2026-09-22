"""Configurações centralizadas do projeto via Pydantic Settings.

Carrega variáveis de ambiente do arquivo `.env` (quando existente) e
disponibiliza acesso tipado às configurações em toda a aplicação.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """Configurações da aplicação.

    Attributes:
        app_name: Nome da aplicação.
        app_env: Ambiente de execução (development | production).
        log_level: Nível de log (DEBUG | INFO | WARNING | ERROR).
        model_path: Caminho do artefato de modelo treinado.
        reference_data_path: Caminho do dataset de referência (treino).
        production_data_dir: Diretório dos snapshots de produção (drift).
        reports_dir: Diretório base dos relatórios gerados.
        mlflow_tracking_uri: URI do servidor/backend de tracking MLflow.
        mlflow_port: Porta do servidor MLflow (Docker).
        dashboard_port: Porta do dashboard Streamlit.
        prometheus_port: Porta do Prometheus.
        grafana_port: Porta do Grafana.
        drift_psi_stable: Limite superior de PSI para distribuição estável.
        drift_psi_moderate: Limite superior de PSI para drift moderado.
        drift_auc_drop_alert: Queda de ROC-AUC que dispara alerta.
        kaggle_username: Usuário do Kaggle para download do dataset.
        kaggle_key: Chave de API do Kaggle (armazenada como segredo).
        dvc_onedrive_remote_url: URL do remote DVC (OneDrive), opcional.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "credit-drift-monitor"
    app_env: str = "development"
    log_level: str = "INFO"
    model_path: Path = Path("models/model.joblib")
    reference_data_path: Path = Path("data/reference/reference.parquet")
    production_data_dir: Path = Path("data/production")
    reports_dir: Path = Path("reports")
    model_backend: Literal["joblib", "onnx"] = "joblib"
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_port: int = 5001
    dashboard_port: int = 8501
    prometheus_port: int = 9090
    grafana_port: int = 3000
    drift_psi_stable: float = 0.1
    drift_psi_moderate: float = 0.25
    drift_auc_drop_alert: float = 0.05
    kaggle_username: str = ""
    kaggle_key: SecretStr = SecretStr("")
    dvc_onedrive_remote_url: str = ""


@lru_cache
def get_settings() -> Settings:
    """Retorna a instância única (cacheada) das configurações.

    Returns:
        Settings: Configurações carregadas do ambiente.
    """
    return Settings()
