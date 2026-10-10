"""Testes de dados e dataset (Etapa 1/2, FR-006/FR-007).

Cobre a normalização do downloader UCI 350, o sinal preditivo do gerador
sintético (fallback offline), a separação determinística e a ausência de
vazamento entre treino e validação.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from scripts.download_data import RAW_TARGET_COLUMN, normalize_columns, parse_uci_xls
from scripts.generate_synthetic_data import generate
from src.data.dedup import deduplicate, row_hash
from src.data.split import split_reference
from tests.conftest import make_credit_frame


def test_normalize_columns_renames_target_aliases():
    """O downloader normaliza nomes de coluna e o alvo para o canônico."""
    frame = pd.DataFrame(
        {"ID": [1, 2], "LIMIT_BAL": [100.0, 200.0], "default.payment.next.month": [0, 1]}
    )
    normalized = normalize_columns(frame)
    assert "default.payment.next.month" not in normalized.columns
    assert RAW_TARGET_COLUMN in normalized.columns


def test_parse_uci_xls_uses_second_header_row(monkeypatch):
    """A planilha do UCI 350 usa duas linhas de cabeçalho (``header=1``)."""
    captured: dict[str, object] = {}

    def fake_read_excel(source, header=0):
        captured["header"] = header
        return pd.DataFrame({"ID": [1], "default payment next month": [0]})

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)
    frame = parse_uci_xls(b"fake-bytes")
    assert captured["header"] == 1
    assert RAW_TARGET_COLUMN in frame.columns


def test_synthetic_generator_has_predictive_signal(tmp_path):
    """O dataset sintético de fallback tem sinal (ROC-AUC > 0.5)."""
    path = tmp_path / "synthetic.csv"
    generate(path, n_rows=3000, seed=42)
    frame = pd.read_csv(path)
    features = frame.drop(columns=["ID", RAW_TARGET_COLUMN])
    labels = frame[RAW_TARGET_COLUMN]
    assert 0.10 < float(labels.mean()) < 0.40
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    assert auc > 0.5


def test_split_is_deterministic():
    """A separação referência/validação é determinística (mesma seed)."""
    frame = make_credit_frame(n_rows=200, seed=11)
    train_a, holdout_a = split_reference(frame)
    train_b, holdout_b = split_reference(frame)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(holdout_a, holdout_b)
    assert len(train_a) + len(holdout_a) == len(frame)


def test_split_has_no_leakage_after_dedup():
    """Sem vazamento: nenhuma linha compartilhada entre treino e validação."""
    frame = make_credit_frame(n_rows=200, seed=11)
    deduplicated, _ = deduplicate(frame, use_near_dup=False)
    train, holdout = split_reference(deduplicated)
    assert set(row_hash(train)).isdisjoint(set(row_hash(holdout)))


def test_synthetic_frame_matches_contract_dtypes(tmp_path):
    """O frame sintético preserva o domínio do contrato (sem valores inválidos)."""
    path = tmp_path / "synthetic.csv"
    generate(path, n_rows=1000, seed=7)
    frame = pd.read_csv(path)
    assert frame["AGE"].between(18, 100).all()
    assert frame["SEX"].isin([1, 2]).all()
    assert np.isfinite(frame["LIMIT_BAL"]).all()
