# =============================================================================
# Tech Challenge Fase 4 - Makefile
# =============================================================================

PYTHON := uv run python

# ---------------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------------
.PHONY: \
	help \
	sync setup verify \
	test lint \
	pipeline ingestao validacao treino drift monitoramento demo-drift \
	data-download data-synthetic \
	dvc-remote \
	docker-build \
	mlflow-up mlflow-down \
	dashboard-up dashboard-down \
	monitoring-up monitoring-down

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo "Tech Challenge Fase 4 - Comandos Disponíveis"
	@echo ""
	@echo "Setup:"
	@echo "  make verify - Validar ambiente (scripts/validate_env.py)"
	@echo "  make sync   - Sincronizar dependências (uv sync)"
	@echo "  make setup  - Configurar ambiente (.env, deps, pre-commit)"
	@echo ""
	@echo "Qualidade:"
	@echo "  make test - Rodar testes (pytest)"
	@echo "  make lint - Verificar código com ruff"
	@echo ""
	@echo "Pipeline DVC:"
	@echo "  make data-download  - Baixar dataset de crédito e registrar no DVC"
	@echo "  make data-synthetic - Gerar dataset sintético e registrar no DVC"
	@echo "  make pipeline       - Reexecutar pipeline DVC completo (dvc repro -v)"
	@echo "  make ingestao       - Reexecutar stage de ingestão"
	@echo "  make validacao      - Reexecutar stage de validação (data contracts)"
	@echo "  make treino         - Reexecutar stage de treino"
	@echo "  make drift          - Reexecutar stage de drift"
	@echo "  make monitoramento  - Reexecutar stage de monitoramento"
	@echo "  make demo-drift     - Demo autocontido de drift (choque econômico + resumo no terminal)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Construir imagem Docker da aplicação"
	@echo "  make mlflow-up      - Subir MLflow tracking server (Docker)"
	@echo "  make mlflow-down    - Parar MLflow"
	@echo "  make dashboard-up   - Subir dashboard Streamlit (Docker)"
	@echo "  make dashboard-down - Parar dashboard"
	@echo "  make monitoring-up  - Subir stack de monitoramento (perfil monitoring)"
	@echo "  make monitoring-down - Parar stack de monitoramento"
	@echo ""

# ---------------------------------------------------------------------------
# Setup e ambiente
# ---------------------------------------------------------------------------
sync:
	uv sync

setup: sync
	@echo "Configurando ambiente..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "[OK] .env criado a partir de .env.example"; fi
	@uv run pre-commit install
	@$(MAKE) dvc-remote
	@echo "Setup concluído!"

verify:
	@echo "Validando ambiente..."
	uv run python scripts/validate_env.py

# ---------------------------------------------------------------------------
# Qualidade
# ---------------------------------------------------------------------------
test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

# ---------------------------------------------------------------------------
# Pipeline DVC
# ---------------------------------------------------------------------------
pipeline:
	@echo "Reexecutando pipeline DVC completo (saída live)..."
	PIPELINE_RUN_ID=$$(uv run python -c "from src.observability.logging import new_run_id; print(new_run_id())") uv run dvc repro -v

ingestao:
	uv run dvc repro ingestao -v

validacao:
	uv run dvc repro validacao -v

treino:
	uv run dvc repro treino -v

drift:
	uv run dvc repro drift -v

monitoramento:
	uv run dvc repro monitoramento -v

demo-drift:
	@echo "Executando demo de drift (choque econômico simulado; evidência isolada)..."
	uv run python -m src.drift.demo $(DEMO_ARGS)

dvc-remote:
	@URL=$$(grep DVC_ONEDRIVE_REMOTE_URL .env 2>/dev/null | cut -d= -f2-); \
	if [ -n "$$URL" ]; then \
		uv run dvc remote add -f onedrive "$$URL"; \
		echo "[OK] Remote DVC OneDrive configurado."; \
	else \
		echo "[SKIP] DVC_ONEDRIVE_REMOTE_URL vazio — remote opcional, pipeline roda self-contained."; \
	fi

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
data-download:
	@echo "Baixando dataset de crédito..."
	PYTHONPATH=. uv run python scripts/download_data.py
	uv run dvc add data/raw/reference.csv
	git add data/raw/reference.csv.dvc
	@echo "[OK] dataset baixado e registrado no DVC."

data-synthetic:
	@echo "Gerando dataset sintético..."
	PYTHONPATH=. uv run python scripts/generate_synthetic_data.py
	uv run dvc add data/raw/reference.csv
	git add data/raw/reference.csv.dvc
	@echo "[OK] dataset sintético gerado e registrado no DVC."

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build:
	@echo "Construindo imagem Docker..."
	@rm -rf /tmp/credit-monitor-build && mkdir -p /tmp/credit-monitor-build
	@cp -r pyproject.toml uv.lock src /tmp/credit-monitor-build/
	docker build -t credit-monitor -f docker/Dockerfile /tmp/credit-monitor-build
	@rm -rf /tmp/credit-monitor-build
	@echo "[OK] Imagem construída. Modelos/dados são montados via volume no docker-compose."

mlflow-up:
	@echo "Subindo MLflow tracking server..."
	docker compose -f docker/docker-compose.yml --env-file .env --profile mlflow up -d
	@echo "MLflow UI: http://localhost:$$(grep MLFLOW_PORT .env 2>/dev/null | cut -d= -f2 || echo 5001)"

mlflow-down:
	docker compose -f docker/docker-compose.yml --profile mlflow down

dashboard-up:
	@echo "Subindo dashboard Streamlit..."
	docker compose -f docker/docker-compose.yml --env-file .env --profile dashboard up -d --build
	@echo "Dashboard: http://localhost:$$(grep DASHBOARD_PORT .env 2>/dev/null | cut -d= -f2 || echo 8501)"

dashboard-down:
	docker compose -f docker/docker-compose.yml --profile dashboard down

# ---------------------------------------------------------------------------
# Monitoramento (Prometheus + Grafana)
# ---------------------------------------------------------------------------
monitoring-up:
	@echo "Subindo stack de monitoramento (Prometheus + Grafana)..."
	docker compose -f docker/docker-compose.yml --env-file .env --profile monitoring up -d
	@echo "Prometheus: http://localhost:$$(grep PROMETHEUS_PORT .env 2>/dev/null | cut -d= -f2 || echo 9090)"
	@echo "Grafana:    http://localhost:$$(grep GRAFANA_PORT .env 2>/dev/null | cut -d= -f2 || echo 3000)"

monitoring-down:
	@echo "Parando stack de monitoramento..."
	docker compose -f docker/docker-compose.yml --profile monitoring down
