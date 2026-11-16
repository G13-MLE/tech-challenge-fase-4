# Roteiro do Vídeo STAR (≤ 5 minutos)

**Formato**: captura de tela com narração (Situation, Task, Action, Result). Duração alvo: **4–5 min**.

## Lista de telas / sequência

| Tempo | Tela | Narração (pontos-chave) |
| ----- | ---- | ----------------------- |
| 0:00–0:30 | **Situação** — slide do problema | Fintech com modelo de crédito; choques econômicos → degradação silenciosa. Referência vs produção. |
| 0:30–1:00 | **Tarefa** — objetivos | 4 etapas: contratos, drift, observabilidade, governança. Requisitos: PSI-first, fairness, LGPD. |
| 1:00–1:45 | **Ação 1** — contratos | Mostrar `src/contracts/schema.py` e rodar `python -m src.contracts.validate data/corrupted/bad_batch.parquet` → **exit 1**. Abrir `reports/validation/result.json`. |
| 1:45–2:45 | **Ação 2** — drift | `make demo-drift` (demo autocontido: simula o choque econômico e imprime o resumo do drift no terminal; evidência em `reports/drift/demo/`) — ou `make pipeline` para o pipeline completo. Abrir `reports/drift/demo/demo_summary.md` e `reports/drift/statistical_report.json` (PSI severo em `LIMIT_BAL/AGE/PAY_0`, `is_breach=true`), `reports/drift/drift_report.html` (Evidently) e `performance.json` (queda de AUC). Explicar Data vs Concept vs dependência (MMD²). |
| 2:45–3:30 | **Ação 3** — observabilidade | `make mlflow-up` (runs + alias `champion`); `make dashboard-up` (SLO, alertas, heatmap de PSI); `make monitoring-up` (Prometheus/Grafana + `/metrics` do exporter). Logs com `run_id`. |
| 3:30–4:15 | **Ação 4** — governança | Mostrar `docs/governance_lgpd.md`, `docs/causal_dag.md`, `reports/fairness/` e os guardrails de re-treino. Explicar PII, base legal e direito de revisão. |
| 4:15–4:45 | **Resultado** | Pipeline reprodutível (`dvc repro` idempotente), CI com gates de contrato e drift, evidências versionadas. Métricas: ROC-AUC, PSI, fairness. |
| 4:45–5:00 | **Encerramento** | Limitações e próximos passos (retreino automatizado com aprovação, monitoramento de feedback loop). |

## Checklist de gravação

- [x] Resolução 1080p, áudio limpo, sem dados pessoais na tela.
- [x] Duração ≤ 5 min (usar 1.5x se necessário).
- [x] Cobrir explicitamente: contratos bloqueando, PSI-first, fairness, LGPD.
- [x] Publicar (YouTube unlisted / Loom / Drive) e colar o link no `README.md`.

> **Link público:** https://drive.google.com/file/d/1HOeQgDW37jiqwuCBBzlwzpKgJG_wWFGz/view?usp=sharing
