# Métricas de Saúde — Pipeline de Credit Scoring (Etapa 3)

Este documento é a fonte canônica das métricas operacionais e de saúde do modelo. Os nomes são
idênticos aos expostos pelo exporter Prometheus (`src/observability/exporter.py`) e às métricas de log
(`logs/pipeline.ndjson`) consumidas pelo dashboard Streamlit.

## Convenções

- `run_id` correlaciona todos os estágios de uma execução de pipeline; `trace_id` correlaciona uma
  execução/estágio. Ambos são propagados via `PIPELINE_RUN_ID` e registrados em cada evento.
- Logs em JSON-lines: todo evento traz `{task, status, timestamp, latency_ms, n_rows, run_id, trace_id}`.
  Nos eventos `started` os campos `latency_ms`/`n_rows` vêm zerados (ainda não medidos) e são
  preenchidos nos eventos de término (`success`/`failed`). Eventos de inferência acrescentam
  `model_name`/`model_version` (FR-018).
- Relatórios em `reports/**/*.json` são a fonte do exporter (padrão *proxy*: lidos no momento do scrape).

## Métricas operacionais (exporter / dashboard)

| Métrica | Tipo (Prometheus) | Labels | Fonte | Descrição | Limiar/alerta |
| ------- | ----------------- | ------ | ----- | --------- | ------------- |
| `pipeline_task_status` | Gauge | `task` | `logs/pipeline.ndjson` | 1 = sucesso, 0 = falha na última execução | 0 → alerta de falha |
| `pipeline_task_latency_ms` | Gauge | `task` | `logs/pipeline.ndjson` | Latência do estágio (ms) | monitoramento |
| `pipeline_rows_processed` | Gauge | `task` | `logs/pipeline.ndjson` | Linhas processadas pelo estágio | volume |
| `drift_features_total` | Gauge | — | `statistical_report.json` | Nº de features avaliadas | — |
| `drift_features_drifted` | Gauge | `severity` | `statistical_report.json` | Features por severidade (stable/moderate/severe) | severe > 0 |
| `drift_psi` | Gauge | `feature`, `month` | `statistical_report.json` (history) | PSI por feature por mês | > 0.25 = breach |
| `model_roc_auc` | Gauge | `month` | `performance.json` | ROC-AUC por snapshot | queda > 0.05 |
| `model_auc_drop` | Gauge | — | `performance.json` | Queda de AUC vs baseline | ≥ `auc_drop_alert` (0.05) |
| `drift_alerts_total` | Gauge | `type` | `run_summary.json` | Nº de alertas por tipo | anti-fadiga |
| `slo_error_budget` | Gauge | `slo` | `slo_report.json` | `valor − piso` (positivo = folga) | < 0 = breach |
| `slo_status` | Gauge | `slo` | `slo_report.json` | 1 = pass, 0 = breach | 0 → ação |

## Data SLOs e Error Budget

Definidos em `configs/params.yaml` → `slo` e avaliados por `src/observability/slo.py`.

| SLO | Limite | Comparação | Fonte |
| --- | ------ | ---------- | ----- |
| `accuracy_min` | 0.85 | `valor ≥ limite` | acurácia do último snapshot (`performance.json`) |
| `psi_max` | 0.25 | `valor ≤ limite` | `max_psi` do relatório estatístico |
| `ks_pvalue_min` | 0.05 | `valor ≥ limite` | menor p-valor KS/χ² |
| `missing_rate_max` | 0.01 | `valor ≤ limite` | `1 − completeness` (`validation/result.json`) |

**Error Budget** = folga em relação ao piso/teto. Positivo = dentro do SLO; negativo = ultrapassado.
`status_geral` = `critical` (breach em `psi_max`/`accuracy_min`), `degraded` (outros breaches) ou `ok`.

## Métricas de drift e modelo (Etapa 2)

- **PSI** (primário): bandas `stable < 0.1`, `moderate ≤ 0.25`, `severe > 0.25`. **Qualquer feature** com
  PSI > 0.25 é um *breach* (SLO + gate de CI).
- **KS** (numérico) e **χ²/Cramér's V** (categórico), com correção de múltiplas comparações (FDR-BH e
  Bonferroni).
- **MMD² / Energy** (multivariado): mudanças na dependência entre features invisíveis às marginais.
- **ROC-AUC / Precision / Recall / Brier / Calibração** por snapshot para evidenciar Concept Drift.

## Alertas (anti-fadiga)

Consolidação (`src/observability/alerts.py`): alerta quando `breach_features` (≥2) features entram em
drift **juntas** no mesmo mês, OU uma **feature crítica** (PSI > 0.25) permanece severa por
`consolidation_windows` (≥3) meses consecutivos. O **re-treino** só dispara em `consecutive_critical_windows`
(N=2) meses críticos — regra distinta e posterior à consolidação de alertas.

## Guardrails de promoção (re-treino)

`src/observability/retrain.py`: challenger promovível somente se **AUC ≥ champion** E **disparate impact
≥ 0.8** E **equal opportunity difference ≤ 0.10**. O challenger **nunca** é promovido automaticamente.
