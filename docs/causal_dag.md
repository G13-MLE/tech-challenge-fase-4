# Análise Causal — Degradação silenciosa do modelo de crédito

O DAG causal abaixo explica **por que** o modelo degrada em produção e onde o pipeline de observabilidade
atua. Ele conecta o choque macroeconômico ao dano financeiro e de fairness, passando pelo drift.

## DAG (Directed Acyclic Graph)

```mermaid
flowchart TD
    M["Choque macroeconômico\n(inflação, juros, desemprego)"] --> I["Renda e emprego do titular"]
    M --> B["Disposição/necessidade de crédito"]
    I --> X["Comportamento financeiro\n(uso do limite, atrasos, pagamentos)"]
    B --> X
    X --> PX["Distribuição de entrada P(X)\n(Data/Covariate Drift)"]
    M --> PY["Relação P(y|X)\n(Concept Drift)"]
    X --> PY
    PX --> D["Degradação do modelo\n(ROC-AUC ↓, Brier ↑)"]
    PY --> D
    D --> DEC["Decisão de crédito enviesada/errada"]
    DEC --> HARM["Dano financeiro\n(perda por inadimplência)"]
    DEC --> FAIR["Dano de fairness\n(recusa indevida por subgrupo)"]
    DEC --> MON["Detecção:\nPSI/KS/χ², MMD², Evidently"]
    MON --> ALERT["Alertas + SLO + Re-treino\ncom guardrails"]
    ALERT -.->|mitiga| D

    classDef macro fill:#ffe8cc,stroke:#d9480f;
    classDef drift fill:#e7f5ff,stroke:#1971c2;
    classDef harm fill:#ffe3e3,stroke:#c92a2a;
    classDef ctrl fill:#ebfbee,stroke:#2f9e44;
    class M,I,B,X macro;
    class PX,PY,D drift;
    class DEC,HARM,FAIR harm;
    class MON,ALERT ctrl;
```

## Caminhos causais principais

1. **Data Drift**: `Choque macro → Renda/emprego → Comportamento (P(X) muda) → Degradação`.
   A população de produção deixa de parecer com a de treino (ex.: `LIMIT_BAL`, `AGE`, `PAY_0` deslocam).
2. **Concept Drift**: `Choque macro → P(y|X) muda` — a **relação** entre comportamento e inadimplência se
   altera (mesmo perfil, risco diferente). Não muda `P(Y)` por si só, mas o mapeamento — por isso o
   pipeline proíbe inversão aleatória de rótulos.
3. **Drift de dependência**: correlações entre features se invertem (ex.: `LIMIT_BAL × AGE`) sem alterar
   as marginais — invisível a PSI/KS, detectável por **MMD²/Energy**.

## Suposições

- O choque macro é **exógeno** ao modelo (não é uma consequência das decisões do próprio modelo).
- As features observadas são **proxies** do comportamento financeiro subjacente.
- O rótulo `default_next_month` é medido sem viés sistemático (não há *label bias* relevante).
- Não há *feedback loop* forte de curto prazo entre decisões de crédito e o comportamento observado
  (violação desse pressuposto exigiria modelagem adicional).

## Confundidores

| Confundidor | Efeito | Mitigação |
| ----------- | ------ | --------- |
| Sazonalidade (meses de gasto) | muda `P(X)` sem mudar risco | snapshots mensais; análise temporal |
| Mudança de política de crédito | muda composição da carteira | monitorar `LIMIT_BAL` e mix de perfil |
| Rótulos atrasados (inadimplência futura) | atrasa a observação do Concept Drift | janela de avaliação por mês |
| Variáveis omitidas (score externo, bureau) | risco não explicado | documentar limites do modelo |

## Contrafactual e resposta

- **Se** o pipeline não existisse, o dano ocorreria de forma silenciosa (sem alerta) até a perda
  materializar.
- **Com** o pipeline, o drift é detectado cedo (PSI-first), consolidado em alertas anti-fadiga, o
  re-treino é acionado por regra objetiva e os guardrails de fairness bloqueiam promoção de modelos
  enviesados.
