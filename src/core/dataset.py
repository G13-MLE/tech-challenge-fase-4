"""Caminhos e utilitários de persistência do pipeline.

Centraliza os caminhos de dados, modelo e relatórios usados pelos
estágios do DVC, evitando literais espalhados pelo código. Todas as
escritas são atômicas (write-to-temp-then-rename).
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

__all__ = [
    "DATA_RAW_PATH",
    "REFERENCE_DATA_PATH",
    "PRODUCTION_DIR",
    "MODEL_PATH",
    "VALIDATION_RESULT_PATH",
    "BAD_BATCH_RESULT_PATH",
    "DEDUP_REPORT_PATH",
    "HARMONIZATION_REPORT_PATH",
    "TRAIN_METRICS_PATH",
    "DRIFT_REPORT_PATH",
    "STATISTICAL_REPORT_PATH",
    "DRIFT_TEST_RESULT_PATH",
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
    "atomic_path",
    "normalize_text_file",
    "sanitize_cjk",
]

DATA_RAW_PATH = Path("data/raw/reference.csv")
REFERENCE_DATA_PATH = Path("data/reference/reference.parquet")
PRODUCTION_DIR = Path("data/production")
MODEL_PATH = Path("models/model.joblib")

VALIDATION_RESULT_PATH = Path("reports/validation/result.json")
BAD_BATCH_RESULT_PATH = Path("reports/validation/bad_batch_result.json")
DEDUP_REPORT_PATH = Path("reports/validation/dedup.json")
HARMONIZATION_REPORT_PATH = Path("reports/validation/harmonization.json")
PROFILE_HTML_PATH = Path("reports/validation/profile.html")
TRAIN_METRICS_PATH = Path("reports/train_metrics.json")
DRIFT_REPORT_PATH = Path("reports/drift/drift_report.html")
STATISTICAL_REPORT_PATH = Path("reports/drift/statistical_report.json")
DRIFT_TEST_RESULT_PATH = Path("reports/drift/drift_test_result.json")
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

    Garante um newline final no arquivo, de modo que o artefato gerado
    permaneça estável sob o hook ``end-of-file-fixer`` e o ``dvc.lock``
    continue idempotente.

    Args:
        path: Caminho do arquivo de saída.
        content: Conteúdo a ser escrito.
    """
    if not content.endswith("\n"):
        content = content + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.rename(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


_CJK_PATTERN = re.compile(
    "["
    "\u3000-\u303f"  # CJK Symbols and Punctuation
    "\u3040-\u30ff"  # Hiragana and Katakana
    "\u3100-\u312f"  # Bopomofo
    "\u31c0-\u31ef"  # CJK Strokes
    "\u3200-\u32ff"  # Enclosed CJK Letters and Months
    "\u3300-\u33ff"  # CJK Compatibility
    "\u3400-\u4dbf"  # CJK Unified Ideographs Extension A
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\uf900-\ufaff"  # CJK Compatibility Ideographs
    "\ufe30-\ufe4f"  # CJK Compatibility Forms
    "\uff00-\uffef"  # Halfwidth and Fullwidth Forms
    "\U00020000-\U0002ebef"  # CJK Unified Ideographs Extensions B-F
    "\U0002f800-\U0002fa1f"  # CJK Compatibility Ideographs Supplement
    "]"
)


def sanitize_cjk(text: str) -> str:
    """Substitui caracteres CJK (chinês/japonês) por espaço.

    O bundle frontend embutido pelo Evidently carrega dados de localização
    em diversos idiomas, incluindo chinês e japonês. Como os relatórios do
    projeto são em português, esses caracteres são ruído; eles são trocados
    por um espaço ASCII em vez de removidos porque a remoção pura deixa
    expressões regulares inválidas no JavaScript gerado: uma âncora ``^``
    seguida de um quantificador órfão (``^??``) dispara "Nothing to repeat".
    A troca por espaço preserva a sintaxe do bundle e elimina qualquer
    caractere CJK.

    Args:
        text: Conteúdo textual a sanitizar.

    Returns:
        O texto sem nenhum caractere CJK.
    """
    return _CJK_PATTERN.sub(" ", text)


def normalize_text_file(path: Path) -> None:
    """Remove espaços ao final das linhas e garante newline final único.

    Aplica em um arquivo já escrito o mesmo efeito dos hooks de pre-commit
    ``trailing-whitespace`` e ``end-of-file-fixer``, mantendo os artefatos
    gerados (ex.: HTML do Evidently) estáveis e o ``dvc.lock`` idempotente.
    Os caracteres CJK remanescentes (dados de localização embutidos pelo
    bundle do Evidently) também são neutralizados; ver :func:`sanitize_cjk`.

    Args:
        path: Arquivo de texto a normalizar (idealmente um temporário antes
            do rename atômico).
    """
    content = sanitize_cjk(path.read_text(encoding="utf-8"))
    lines = [line.rstrip() for line in content.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


@contextmanager
def atomic_path(path: Path, suffix: str = ".tmp") -> Iterator[Path]:
    """Fornece um caminho temporário e o renomeia atomicamente ao sair.

    Útil para bibliotecas que escrevem diretamente em um caminho (ex.: o
    ``save_html`` do Evidently). O consumidor escreve em ``<path>`` + ``suffix``
    e, se o bloco concluir sem exceção, o arquivo é renomeado para ``path``
    (write-to-temp-then-rename); em caso de erro, o temporário é removido.

    Args:
        path: Caminho final do arquivo.
        suffix: Sufixo do arquivo temporário (default ``.tmp``).

    Yields:
        Caminho temporário onde o conteúdo deve ser escrito.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + suffix)
    try:
        yield tmp_path
        tmp_path.rename(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
