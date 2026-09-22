"""Download do dataset de crédito e conversão para o formato bruto.

Etapa 1. Baixa o dataset UCI "Default of Credit Card Clients" (ID 350)
e normaliza a coluna alvo. Requer acesso à internet ou o arquivo já
presente em `data/raw/`.
"""

import logging
import urllib.request
from pathlib import Path

from src.core.dataset import DATA_RAW_PATH

logger = logging.getLogger(__name__)

__all__ = ["download", "main"]

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"


def download(destination: Path = DATA_RAW_PATH) -> Path:
    """Baixa o dataset de crédito para o caminho bruto.

    Args:
        destination: Caminho de destino do CSV bruto.

    Returns:
        Caminho do arquivo baixado.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Baixando dataset de %s...", UCI_URL)
    urllib.request.urlretrieve(UCI_URL, destination)  # noqa: S310
    logger.info("Dataset salvo em %s.", destination)
    return destination


def main() -> None:
    """Ponto de entrada do download do dataset."""
    download()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
