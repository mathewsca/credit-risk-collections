.DEFAULT_GOAL := help
COMPOSE := docker compose
RUN := $(COMPOSE) exec -T airflow /opt/project-venv/bin/python -m

help: ## Lista os comandos
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

up: ## Sobe toda a stack (Postgres, Airflow, MLflow, API, Streamlit)
	$(COMPOSE) up -d --build

down: ## Derruba a stack (mantém volumes)
	$(COMPOSE) down

reset: ## Derruba a stack e apaga volumes (banco e artefatos)
	$(COMPOSE) down -v

pipeline: ## Dispara a DAG completa no Airflow e acompanha
	$(COMPOSE) exec -T airflow airflow dags trigger credit_risk_pipeline

shock: ## Dispara a DAG com o cenário de choque (demonstra drift -> retreino)
	$(COMPOSE) exec -T airflow airflow dags trigger credit_risk_pipeline -c '{"scenario": "shock"}'

dbt-docs: ## Gera a documentação e a linhagem do dbt
	$(COMPOSE) exec -T airflow bash -c "cd /opt/project/dbt && /opt/project-venv/bin/dbt docs generate --profiles-dir . --target-path /opt/project/dbt/target"

setup: ## Ambiente local de desenvolvimento (sem Docker)
	python -m venv .venv && .venv/bin/pip install -e ".[api,app,dbt,dev]" && .venv/bin/pre-commit install

test: ## Testes unitários
	pytest

lint: ## Lint e formatação
	ruff check . && ruff format --check .

app: ## Roda o Streamlit local lendo o snapshot Parquet
	streamlit run app/Home.py
