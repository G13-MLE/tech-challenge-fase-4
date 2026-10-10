# Tech Challenge — Fase 4: Observabilidade, Drift e Governança

Camada de **sustentação e confiabilidade** para um modelo de *Credit Scoring* em produção. O projeto
intercepta dados ruins com *data contracts*, simula a passagem do tempo (Data + Concept Drift), detecta a
degradação com testes estatísticos (PSI-first / Kolmogorov-Smirnov / χ²) e multivariados (MMD² / Energy),
gera relatório Evidently AI, expõe a saúde do pipeline em um dashboard e no `/metrics` (Prometheus/Grafana),
e documenta governança (LGPD), análise causal e fairness.

> **Status:** implementação completa (Etapas 1–4). `make test` e `make lint` verdes; pipeline DVC
> reprodutível (`dvc repro` idempotente); CI com gates de contrato e drift.

## Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Como Executar](#como-executar)
- [Etapas do Projeto](#etapas-do-projeto)
- [Observabilidade](#observabilidade)
- [Governança e LGPD](#governança-e-lgpd)
- [Model Card](#model-card)
- [DPIA (resumo)](#dpia-resumo)
- [Análise Causal](#análise-causal)
- [Métricas de Saúde](#métricas-de-saúde)
- [Vídeo STAR](#vídeo-star)
- [Critérios de Avaliação](#critérios-de-avaliação)

## Visão Geral

Uma fintech de crédito possui um modelo de risco rodando em produção. Mudanças econômicas (inflação,
novos perfis de clientes) fazem a população atual divergir da população de treino, degradando o modelo
de forma **silenciosa**. Este repositório constrói a camada de detecção e governança em torno de um
classificador base.

- **Referência** (`data/reference/`) — usado no treino (dataset público **UCI 350**, ~30k; ou fallback
  sintético com sinal para execução offline/CI).
- **Produção** (`data/production/`) — snapshots mensais sintéticos com Data + Concept Drift.
- **Referência e produção nunca se misturam** (separação estrita; teste de vazamento).

## Arquitetura

Pipeline DVC com estágios em português: `ingestao → validacao → treino → drift → monitoramento`.

```text
src/
├── core/            # Settings (env), Params (YAML), caminhos e I/O atômico
├── data/            # ingestão, split, dedup, missingness, outliers
├── contracts/       # schema Pandera + validação que BLOQUEIA lotes inválidos (exit 1)
├── model/           # factory, treino (XGBoost/LogReg), predição e avaliação
├── drift/           # simulate, stats (PSI/KS/χ²/FDR), multivariate (MMD²/Energy),
│                    #   concept (P(y|X)), detect (Evidently), run
└── observability/   # logging (run_id/trace_id), metrics, alerts, slo, retrain,
                     #   fairness, tracking (MLflow), exporter (Prometheus), run
dashboard/app.py     # dashboard Streamlit (SLO, PSI heatmap, alertas, Evidently)
scripts/             # download_data, generate_synthetic_data, make_bad_batch, drift_gate, validate_env
docker/              # Dockerfile + docker-compose + Grafana provisioning as-code
configs/             # params.yaml + prometheus.yml
docs/                # metrics.md, governance_lgpd.md, data_dictionary.md, causal_dag.md, video_script.md
reports/             # evidências versionadas: validation/, drift/, monitoring/, fairness/
```

## Como Executar

Requer [`uv`](https://docs.astral.sh/uv/) e Python 3.12 (`uv` instala automaticamente via `.python-version`).

```bash
make setup            # uv sync + .env + pre-commit
make data-synthetic   # fallback offline (CI) — ou `make data-download` para o UCI 350 real
make pipeline         # ingestao -> validacao -> treino -> drift -> monitoramento (dvc repro)
make test && make lint
make mlflow-up        # MLflow UI em http://localhost:5001
make dashboard-up     # Dashboard em http://localhost:8501
make monitoring-up    # Prometheus (9090) + Grafana (3000) + exporter /metrics (8000)
```

## Etapas do Projeto

| Etapa | Descrição | Peso | Evidência |
| ----- | --------- | ---- | --------- |
| 1 — Validação e Contratos | Contrato de dados rígido (bloqueia lotes) + modelo baseline | 25% | [`reports/validation/`](reports/validation/), `models/`, [`src/contracts/`](src/contracts/) |
| 2 — Drift | Simulação Data/Concept Drift + PSI/KS/χ² + MMD² + Evidently | 25% | [`reports/drift/`](reports/drift/) |
| 3 — Observabilidade | Logs JSON, métricas, SLO, MLflow, dashboard e `/metrics` | 20% | [`reports/monitoring/`](reports/monitoring/), [`docs/metrics.md`](docs/metrics.md) |
| 4 — Governança | LGPD, viés/fairness, análise causal e vídeo STAR | 15% + 15% | [`reports/fairness/`](reports/fairness/), [docs](#governança-e-lgpd) |

## Observabilidade

- **Logs JSON-lines** (`logs/pipeline.ndjson`) com correlação `run_id`/`trace_id` por execução.
- **Data SLOs + Error Budget** (`src/observability/slo.py`) → `reports/monitoring/slo_report.json`.
- **Alertas anti-fadiga** (`src/observability/alerts.py`): ≥2 features juntas OU 1 feature crítica por
  ≥3 meses consecutivos.
- **Re-treino com guardrails** (`src/observability/retrain.py`): PSI ≥ 0.25 por N=2 meses → challenger
  (nunca promovido automaticamente).
- **MLflow Registry** com `signature`, `input_example`, digest SHA-256 do dataset e aliases
  `champion`/`challenger`.
- **Exporter Prometheus** (`src/observability/exporter.py`) em `exporter:8000` + Grafana as-code.

### Métricas Principais (pipeline atual, dados de exemplo)

| Métrica | Valor |
| ------- | ----- |
| ROC-AUC baseline (referência) | ~0.71 |
| ROC-AUC último snapshot | ~0.67 (queda ≈ 0.03) |
| PSI máx. (`PAY_0`) | ~2.8 (**breach** > 0.25) |
| MMD² (multivariado) | flag de drift de dependência (`LIMIT_BAL × AGE`) |
| Label drift `P(Y)` | ≈ 0 (preservado na simulação de concept drift) |

## Governança e LGPD

Tratamento de **PII** (`ID` direto → removido; `SEX`/`AGE`/`EDUCATION`/`MARRIAGE` quasi-identificadores),
**base legal** (LGPD art. 7º, X / art. 11) para decisão de crédito, **retenção** por ciclo de vida e
**mitigação de vieses** (disparate impact ≥ 0.8 e equal opportunity ≤ 0.10). O titular tem direito à
**revisão de decisões automatizadas** (art. 20), viabilizado pelo versionamento no MLflow. Dados
sintéticos são usados como salvaguarda em dev/CI.

Detalhes: [`docs/governance_lgpd.md`](docs/governance_lgpd.md) e [`docs/data_dictionary.md`](docs/data_dictionary.md).

## Model Card

| Item | Descrição |
| ---- | --------- |
| **Modelo** | Classificador binário de risco (XGBoost, fallback LogReg) — `models/model.joblib` |
| **Intended use** | Priorização/apoio à decisão de crédito; **não** é decisão final automatizada |
| **Features** | `LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, BILL_AMT1, PAY_AMT1` |
| **Target** | `default_next_month` (1 = inadimplente) |
| **Treino** | UCI 350 (referência); split determinístico 80/20, dedup antes do split |
| **Métricas** | ROC-AUC, Precision, Recall, Brier, calibração (+ PSI/KS/χ² e fairness em produção) |
| **Limitações** | Dataset histórico de Taiwan; pode não generalizar a outras populações/tempos; risco de proxy em `SEX`/`MARRIAGE`/`EDUCATION` |
| **Fairness** | Guardrails por `SEX` e faixas de `AGE` monitorados a cada snapshot |

## DPIA (resumo)

- **Finalidade**: análise de risco de crédito (legítimo interesse, art. 7º, X).
- **Dados**: financeiros agregados + quasi-identificadores; nenhum dado é publicado por linha.
- **Necessidade/proporcionalidade**: apenas features com valor preditivo; `ID` removido.
- **Riscos**: viés discriminatório, reidentificação, decisão opaca.
- **Mitigações**: fairness por subgrupo, agregação em relatórios, registry/versionamento, revisão humana.
- **Titular**: direito de revisão de decisão automatizada (art. 20).

## Análise Causal

DAG causal (choque macro → renda/emprego → comportamento → drift → decisão errada → dano financeiro e de
fairness), com suposições e confundidores, em [`docs/causal_dag.md`](docs/causal_dag.md).

## Métricas de Saúde

Definições, fontes e limiares de todas as métricas operacionais/modelo em [`docs/metrics.md`](docs/metrics.md),
alinhadas aos nomes do exporter Prometheus.

## Vídeo STAR

Link do vídeo (≤ 5 min — Situation, Task, Action, Result): **TODO — inserir link público**.
Roteiro completo em [`docs/video_script.md`](docs/video_script.md).

## Critérios de Avaliação

- [x] **1. Validação/Contratos** — contrato rígido bloqueia lotes inválidos (exit 1) e mede qualidade
      (dimensões, dedup antes do split, missingness, outliers, profiling).
- [x] **2. Drift** — snapshots determinísticos com Data + Concept Drift; detecção **PSI-first** + KS/χ²
      com FDR, multivariado (MMD²/Energy) e relatório Evidently.
- [x] **3. Observabilidade** — logs correlacionados, SLO/Error Budget, alertas anti-fadiga, MLflow
      Registry, dashboard e `/metrics`.
- [x] **4. Governança** — LGPD/PII, análise causal, fairness com guardrails e Model Card/DPIA.
- [x] **5. Reprodutibilidade** — DVC pipeline (estágios PT), `dvc repro` idempotente, CI com gates de
      contrato e drift; evidências versionadas em `reports/`.

## Licença

MIT — ver [LICENSE](LICENSE).
