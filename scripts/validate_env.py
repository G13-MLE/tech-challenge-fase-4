"""Validação do ambiente de desenvolvimento.

Verifica Python, dependências, diretórios essenciais e variáveis de
ambiente (fonte do dataset e porta do exporter Prometheus) antes de
rodar o pipeline. Uso: `make verify`.
"""

import sys
from pathlib import Path

from dotenv import dotenv_values

__all__ = ["main"]

REQUIRED_DIRS = ["configs", "src", "docker"]
REQUIRED_FILES = ["configs/params.yaml", "pyproject.toml", "Makefile", ".env.example"]

# Variáveis documentadas em `.env.example` que precisam existir.
REQUIRED_ENV_KEYS = ["DATASET_SOURCE", "PROMETHEUS_EXPORTER_PORT"]

# Chaves que nunca devem conter valor commitado.
SECRET_ENV_KEYS = ["KAGGLE_KEY"]

ALLOWED_DATASET_SOURCES = {"uci", "synthetic"}
DEFAULT_DATASET_SOURCE = "uci"
DEFAULT_EXPORTER_PORT = 8000
PORT_MIN, PORT_MAX = 1, 65535

ENV_EXAMPLE_PATH = Path(".env.example")
ENV_PATH = Path(".env")


def _validate_dataset_source(value: str, problems: list[str]) -> None:
    if value not in ALLOWED_DATASET_SOURCES:
        allowed = " | ".join(sorted(ALLOWED_DATASET_SOURCES))
        problems.append(f"DATASET_SOURCE inválido: '{value}'; esperado: {allowed}.")


def _validate_exporter_port(value: str, problems: list[str]) -> None:
    try:
        port = int(value)
    except ValueError:
        problems.append(f"PROMETHEUS_EXPORTER_PORT não é inteiro: '{value}'.")
        return
    if not (PORT_MIN <= port <= PORT_MAX):
        problems.append(
            f"PROMETHEUS_EXPORTER_PORT fora do intervalo {PORT_MIN}-{PORT_MAX}: {port}."
        )


def _validate_env(problems: list[str]) -> None:
    documented = dotenv_values(ENV_EXAMPLE_PATH)
    for key in REQUIRED_ENV_KEYS:
        if key not in documented:
            problems.append(f"Variável ausente em .env.example: {key}")

    for key in SECRET_ENV_KEYS:
        if documented.get(key):
            problems.append(
                f"Segredo não vazio em .env.example: {key} (nunca commitar credenciais)."
            )

    values = dotenv_values(ENV_PATH) if ENV_PATH.is_file() else documented
    _validate_dataset_source(values.get("DATASET_SOURCE") or DEFAULT_DATASET_SOURCE, problems)
    _validate_exporter_port(
        values.get("PROMETHEUS_EXPORTER_PORT") or str(DEFAULT_EXPORTER_PORT), problems
    )


def main() -> int:
    """Executa as checagens do ambiente.

    Returns:
        Código de saída (0 = OK, 1 = problema).
    """
    problems: list[str] = []
    if sys.version_info[:2] != (3, 12):
        problems.append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} detectado; esperado 3.12."
        )
    for directory in REQUIRED_DIRS:
        if not Path(directory).is_dir():
            problems.append(f"Diretório ausente: {directory}")
    for file in REQUIRED_FILES:
        if not Path(file).is_file():
            problems.append(f"Arquivo ausente: {file}")
    if ENV_EXAMPLE_PATH.is_file():
        _validate_env(problems)
    if problems:
        for problem in problems:
            print(f"[ERRO] {problem}")
        return 1
    print("[OK] Ambiente validado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
