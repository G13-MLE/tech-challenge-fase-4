# Demonstração de drift — `economic_shock`

- **Meses:** 6
- **Seed:** 42

## PSI por feature deslocada

| Feature | M1 | M2 | M3 | M4 | M5 | M6 |
| --- | --- | --- | --- | --- | --- | --- |
| LIMIT_BAL | 0.136 moderate | 0.606 severe | 1.452 severe | 2.429 severe | 3.603 severe | 4.097 severe |
| AGE | 0.037 stable | 0.185 moderate | 0.522 severe | 1.358 severe | 1.850 severe | 2.550 severe |
| PAY_0 | 0.000 stable | 0.000 stable | 0.058 stable | 1.284 severe | 2.130 severe | 2.130 severe |

## ROC-AUC por mês

Baseline de referência: **0.7613**.

| Mês | ROC-AUC | Queda vs baseline |
| --- | --- | --- |
| M1 | 0.6558 | 0.1055 |
| M2 | 0.6393 | 0.1220 |
| M3 | 0.6370 | 0.1243 |
| M4 | 0.6241 | 0.1372 |
| M5 | 0.6048 | 0.1565 |
| M6 | 0.6111 | 0.1502 |

## Tipos de drift

| Tipo | Detectado |
| --- | --- |
| data_drift | True |
| label_drift | False |
| concept_drift | True |
| multivariate_drift | True |

## Alertas consolidados (8)

- **[high] data_drift** — 2 features com drift severo juntas no mês 3.
- **[high] data_drift** — 3 features com drift severo juntas no mês 4.
- **[high] data_drift** — 3 features com drift severo juntas no mês 5.
- **[high] data_drift** — 4 features com drift severo juntas no mês 6.
- **[critical] data_drift** — Feature crítica LIMIT_BAL em severidade severa por 5 meses consecutivos.
- **[critical] data_drift** — Feature crítica AGE em severidade severa por 4 meses consecutivos.
- **[critical] data_drift** — Feature crítica PAY_0 em severidade severa por 3 meses consecutivos.
- **[high] model_degradation** — Queda de ROC-AUC de 0.1502 vs baseline (limiar 0.05).

## Evidência isolada

- `summary_json`: `reports/drift/demo/demo_summary.json`
- `summary_md`: `reports/drift/demo/demo_summary.md`
- `statistical_report`: `reports/drift/demo/statistical_report.json`
- `performance`: `reports/drift/demo/performance.json`
- `drift_report_html`: `reports/drift/demo/drift_report.html`
- `snapshots_dir`: `data/demo`
