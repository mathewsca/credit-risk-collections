"""Pipeline de risco de crédito: ERP -> warehouse -> monitoramento -> (re)treino condicional -> scoring.

    extract_erp -> ingest_raw -> dbt_build -> monitor_and_decide ─┬─> train_challenger -> promote ─┐
                                                                  └─> skip_training ───────────────┴─> score -> export_snapshot -> reload_api

O retreino NÃO roda por calendário: roda quando o monitoramento detecta drift, queda de desempenho,
descalibração ou ausência de modelo em produção (ou quando forçado via parâmetro).
"""

from __future__ import annotations

import os

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import Param, dag, task

PY = os.getenv("PROJECT_PYTHON", "/opt/project-venv/bin/python")
VENV_BIN = os.path.dirname(PY)


@dag(
    dag_id="credit_risk_pipeline",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 1, 1, tz="America/Sao_Paulo"),
    catchup=False,
    max_active_runs=1,
    tags=["credit-risk", "mlops"],
    params={
        "scenario": Param(
            "base", enum=["base", "shock"], description="'shock' simula crise a partir de out/2024"
        ),
        "force_retrain": Param(False, type="boolean"),
    },
    doc_md=__doc__,
    default_args={
        "retries": 3,
        "retry_delay": pendulum.duration(seconds=30),
        "retry_exponential_backoff": 2.0,  # 30s, 60s, 120s: absorve indisponibilidades transitórias
    },
)
def credit_risk_pipeline():
    extract_erp = BashOperator(
        task_id="extract_erp",
        bash_command=f"{PY} -m credit_risk.generator.generate --scenario {{{{ params.scenario }}}}",
        doc_md="Simula a extração do ERP (gera os Parquet da landing zone).",
    )

    @task.external_python(python=PY, expect_airflow=False)
    def ingest_raw() -> dict:
        from credit_risk.ingest import run

        return run()

    dbt_build = BashOperator(
        task_id="dbt_build",
        bash_command=(
            f"cd /opt/project/dbt && {VENV_BIN}/dbt build --profiles-dir . "
            "--target-path /tmp/dbt/target --log-path /tmp/dbt/logs"
        ),
        doc_md="staging -> marts -> ml_features, com todos os testes de dados (falha = pipeline para).",
    )

    @task.branch_external_python(python=PY, expect_airflow=False)
    def monitor_and_decide(force: str) -> str:
        from credit_risk.ml.monitor import run

        result = run()
        print(result)
        return "train_challenger" if result["retrain"] or force == "True" else "skip_training"

    @task.external_python(python=PY, expect_airflow=False)
    def train_challenger() -> dict:
        from credit_risk.ml.train import run

        s = run()
        return {
            "version": s.get("model_version"),
            "selected": s["selected"],
            "test": s["test"][s["selected"]]["metrics"],
        }

    @task.external_python(python=PY, expect_airflow=False)
    def promote() -> dict:
        from credit_risk.ml.promote import run

        d = run()
        return {k: (str(v) if k == "decided_at" else v) for k, v in d.items()}

    skip_training = EmptyOperator(task_id="skip_training")

    @task.external_python(python=PY, expect_airflow=False, trigger_rule="none_failed_min_one_success")
    def score() -> int:
        from credit_risk.ml.score import run

        return len(run())

    @task.external_python(python=PY, expect_airflow=False)
    def export_snapshot() -> None:
        from credit_risk.export import run

        run()

    reload_api = BashOperator(
        task_id="reload_api",
        bash_command="curl -fsS -X POST http://api:8000/reload || echo 'API indisponível (ok em execução local)'",
    )

    decision = monitor_and_decide(force="{{ params.force_retrain }}")
    extract_erp >> ingest_raw() >> dbt_build >> decision
    decision >> train_challenger() >> promote() >> (scored := score())
    decision >> skip_training >> scored
    scored >> export_snapshot() >> reload_api


credit_risk_pipeline()
