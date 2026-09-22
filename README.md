# Tech Challenge — Fase 4: Observabilidade, Drift e Governança

Camada de **sustentação e confiabilidade** para um modelo de *Credit Scoring* em produção. O projeto
intercepta dados ruins com *data contracts*, simula a passagem do tempo (Data Drift + Concept Drift),
detecta a degradação com testes estatísticos (PSI / Kolmogorov-Smirnov) e Evidently AI, expõe a saúde do
pipeline em um dashboard de observabilidade e documenta governança (LGPD) e análise causal.

> **Status:** estrutura inicial (scaffold). As seções marcadas com `TODO` serão preenchidas ao longo das etapas.

## Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Como Executar](#como-executar)
- [Etapas do Projeto](#etapas-do-projeto)
- [Governança e LGPD](#governança-e-lgpd)
- [Análise Causal](#análise-causal)
- [Métricas de Saúde](#métricas-de-saúde)
- [Vídeo STAR](#vídeo-star)

## Visão Geral

Uma fintech de crédito possui um modelo de risco rodando em produção. Mudanças econômicas (inflação,
novos perfis de clientes) fazem a população atual divergir da população de treino, degradando o modelo
de forma **silenciosa**. Este repositório constrói a camada de detecção e governança em torno de um
classificador base.

- **Referência** (`data/reference/`) — usado no treino.
- **Produção** (`data/production/`) — snapshots sintéticos com drift, onde a degradação é medida.

## Arquitetura

Pipeline DVC com estágios em português: `ingestao → validacao → treino → drift → monitoramento`.

```text
src/
├── core/            # Settings, Params, caminhos e I/O atômico
├── data/            # ingestão e split referência/produção
├── contracts/       # schema Pandera + validação que bloqueia lotes inválidos
├── model/           # treino (XGBoost/LogReg), predição e avaliação
├── drift/           # simulação, PSI/KS, Evidently e concept drift
└── observability/   # logs JSON, métricas, tracking MLflow e alertas
dashboard/app.py     # dashboard Streamlit de saúde do pipeline
```

## Como Executar

Requer [`uv`](https://docs.astral.sh/uv/) e Python 3.12 (`uv` instala automaticamente via `.python-version`).

```bash
make setup            # uv sync + .env + pre-commit
make data-synthetic   # ou `make data-download` para o dataset UCI real
make pipeline         # ingestao -> validacao -> treino -> drift -> monitoramento
make mlflow-up        # MLflow UI em http://localhost:5001
make dashboard-up     # Dashboard em http://localhost:8501
make test && make lint
```

## Etapas do Projeto

| Etapa | Descrição | Peso | Evidência |
| ----- | --------- | ---- | --------- |
| 1 — Validação e Contratos | Contrato de dados rígido + modelo baseline | 25% | `reports/validation/`, `models/` |
| 2 — Drift | Simulação Data/Concept Drift + PSI/KS + Evidently | 25% | `reports/drift/drift_report.html` |
| 3 — Observabilidade | Logs, métricas, MLflow e dashboard | 20% | `reports/monitoring/`, `logs/` |
| 4 — Governança | LGPD, viés, análise causal e vídeo STAR | 15% + 15% | este README |

## Governança e LGPD

TODO: tratar os quatro pilares exigidos — tratamento de PII, base legal para decisão de crédito,
plano de retenção e mitigação de vieses. Detalhamento em `docs/governance_lgpd.md`.

## Análise Causal

TODO: DAG causal do problema (choque macro → renda/emprego → comportamento → drift → decisão errada).
Detalhamento em `docs/causal_dag.md`.

## Métricas de Saúde

TODO: documentar as métricas operacionais (status das tarefas, latência, volume processado, alertas de
falha/drift, ROC-AUC por snapshot). Ver `docs/metrics.md`.

## Vídeo STAR

TODO: link do vídeo (≤ 5 min) — Situation, Task, Action, Result. Roteiro em `docs/video_script.md`.

## Licença

MIT — ver [LICENSE](LICENSE).
