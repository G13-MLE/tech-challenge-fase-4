"""Download do dataset de crédito e conversão para o formato bruto.

Etapa 1. Baixa o dataset real UCI "Default of Credit Card Clients"
(ID 350, ~30k amostras), extrai o ZIP interno e parseia a planilha
``.xls`` (que possui duas linhas de cabeçalho — usamos ``header=1``),
escrevendo ``data/raw/reference.csv`` com os nomes de coluna originais.

O downloader é tolerante a falhas de rede: se o CSV bruto já existir,
nada é baixado (idempotente). Como fallback opcional o dataset pode ser
obtido via ``kagglehub``. Para ambientes offline/CI use
``scripts/generate_synthetic_data.py``.
"""

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.core.dataset import DATA_RAW_PATH, atomic_write_text

logger = logging.getLogger(__name__)

__all__ = ["download", "parse_uci_xls", "normalize_columns", "main"]

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

# Nome canônico da coluna alvo no arquivo bruto (renomeada na ingestão).
RAW_TARGET_COLUMN = "default payment next month"

_TARGET_ALIASES = {
    "default payment next month": RAW_TARGET_COLUMN,
    "default.payment.next.month": RAW_TARGET_COLUMN,
    "default_next_month": RAW_TARGET_COLUMN,
}


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normaliza os nomes de coluna do UCI 350 e o nome do alvo.

    Args:
        frame: DataFrame bruto com nomes de coluna possivelmente variantes.

    Returns:
        DataFrame com nomes normalizados e alvo renomeado para o canônico.
    """
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame.rename(
        columns={key: value for key, value in _TARGET_ALIASES.items() if key in frame.columns}
    )


def parse_uci_xls(source: bytes | Path) -> pd.DataFrame:
    """Parseia a planilha do UCI 350 a partir de bytes ou caminho.

    A planilha possui duas linhas de cabeçalho; a primeira descreve os
    tipos e a segunda os nomes das colunas, portanto ``header=1``.

    Args:
        source: Bytes do arquivo ``.xls``/``.xlsx`` ou caminho local.

    Returns:
        DataFrame com as colunas originais do dataset UCI 350.

    Raises:
        ValueError: Se nenhuma planilha ``.xls``/``.xlsx`` for encontrada.
    """
    excel_source = io.BytesIO(source) if isinstance(source, bytes) else source
    frame = pd.read_excel(excel_source, header=1)
    return normalize_columns(frame)


def _extract_xls(zip_bytes: bytes) -> pd.DataFrame:
    """Extrai o ZIP do UCI 350 e parseia a planilha interna."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith((".xls", ".xlsx"))]
        if not members:
            raise ValueError("Nenhuma planilha .xls/.xlsx encontrada no ZIP do UCI 350.")
        with archive.open(members[0]) as handle:
            return parse_uci_xls(handle.read())


def _download_zip(url: str = UCI_URL) -> bytes:
    """Baixa o conteúdo binário do ZIP do UCI 350."""
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        return response.read()


def _from_kagglehub() -> pd.DataFrame:
    """Fallback opcional: obtém o dataset via ``kagglehub``."""
    import kagglehub

    path = Path(kagglehub.dataset_download("uciml/default-of-credit-card-clients-dataset"))
    csv_files = sorted(path.glob("**/*.csv"))
    if not csv_files:
        raise ValueError("kagglehub não retornou nenhum CSV.")
    frame = pd.read_csv(csv_files[0])
    return normalize_columns(frame)


def download(destination: Path = DATA_RAW_PATH, *, force: bool = False) -> Path:
    """Adquire o dataset UCI 350 e o persiste em ``destination``.

    Args:
        destination: Caminho de destino do CSV bruto.
        force: Se True, rebaixa mesmo que ``destination`` já exista.

    Returns:
        Caminho do arquivo CSV gravado.

    Raises:
        RuntimeError: Se o download e o fallback kagglehub falharem.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force and destination.stat().st_size > 0:
        logger.info("Dataset já presente em %s; download ignorado.", destination)
        return destination
    try:
        logger.info("Baixando dataset UCI 350 de %s...", UCI_URL)
        frame = _extract_xls(_download_zip())
    except Exception as exc:  # noqa: BLE001 - fallback explícito
        logger.warning("Falha no download direto (%s); tentando kagglehub.", exc)
        frame = _from_kagglehub()
    atomic_write_text(destination, frame.to_csv(index=False))
    logger.info("Dataset salvo em %s (%d linhas).", destination, len(frame))
    return destination


def main() -> None:
    """Ponto de entrada do download do dataset."""
    download()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
