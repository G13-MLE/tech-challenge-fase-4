"""Perfilagem de dados → ``reports/validation/profile.html`` (FR-003, R7).

Gera o HTML de profiling via Evidently ``DataSummaryPreset`` (dependência
já presente). Se o Evidently falhar, cai em um fallback simples baseado
em ``pandas.describe``. A escrita é atômica.
"""

import html
import logging
from pathlib import Path

import pandas as pd

from src.core.dataset import PROFILE_HTML_PATH, normalize_text_file

__all__ = ["generate_profile"]

logger = logging.getLogger(__name__)


def _fallback_html(frame: pd.DataFrame) -> str:
    """HTML mínimo de estatísticas descritivas usado como fallback."""
    described = frame.describe(include="all").transpose()
    table = described.to_html(escape=True)
    missing = frame.isna().mean().round(6)
    missing_table = missing.to_frame("missing_rate").to_html(escape=True)
    title = html.escape("Perfil de dados (fallback pandas)")
    return (
        "<!DOCTYPE html><html lang='pt-br'><head><meta charset='utf-8'>"
        f"<title>{title}</title></head><body>"
        f"<h1>{title}</h1>"
        f"<p>Linhas: {len(frame)} | Colunas: {frame.shape[1]}</p>"
        "<h2>Estatísticas descritivas</h2>"
        f"{table}"
        "<h2>Taxa de ausência por coluna</h2>"
        f"{missing_table}"
        "</body></html>"
    )


def generate_profile(frame: pd.DataFrame, destination: Path = PROFILE_HTML_PATH) -> Path:
    """Gera o HTML de perfilagem do lote.

    Args:
        frame: DataFrame a perfilar.
        destination: Caminho do HTML de saída.

    Returns:
        Caminho do HTML gerado.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(".tmp.html")
    try:
        from evidently import Dataset, Report
        from evidently.presets import DataSummaryPreset

        snapshot = Report([DataSummaryPreset()]).run(Dataset.from_pandas(frame))
        snapshot.save_html(str(tmp_path))
        normalize_text_file(tmp_path)
        tmp_path.rename(destination)
        logger.info("Perfil Evidently gerado: %s.", destination)
    except Exception:  # noqa: BLE001 - fallback garantido para CI offline
        logger.warning("Evidently indisponível; usando fallback pandas.", exc_info=True)
        from src.core.dataset import atomic_write_text

        atomic_write_text(destination, _fallback_html(frame))
    finally:
        tmp_path.unlink(missing_ok=True)
    return destination


def main() -> None:
    """Gera o perfil do dataset de referência (entrada de linha de comando)."""
    from src.core.dataset import REFERENCE_DATA_PATH, load_parquet

    generate_profile(load_parquet(REFERENCE_DATA_PATH))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
