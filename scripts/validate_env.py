"""Validação do ambiente de desenvolvimento.

Verifica Python, dependências e diretórios essenciais antes de rodar
o pipeline. Uso: `make verify`.
"""

import sys
from pathlib import Path

__all__ = ["main"]

REQUIRED_DIRS = ["configs", "src", "docker"]
REQUIRED_FILES = ["configs/params.yaml", "pyproject.toml", "Makefile"]


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
    if problems:
        for problem in problems:
            print(f"[ERRO] {problem}")
        return 1
    print("[OK] Ambiente validado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
