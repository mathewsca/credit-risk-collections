"""API de scoring de risco de crédito (FastAPI).

Carrega o modelo com alias `champion` do MLflow Model Registry. Para usar uma nova versão
promovida, basta chamar POST /reload (sem rebuild da imagem).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit_risk.ml.registry import predict_pd
from credit_risk.ml.score import risk_band

MODEL_NAME = os.getenv("MODEL_NAME", "credit-risk-pd")
STATE: dict = {}


def _load() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mv = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
    STATE["model"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    STATE["version"] = mv.version


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        _load()
    except Exception as exc:  # noqa: BLE001 - API sobe mesmo sem modelo; /health indica o estado
        STATE["error"] = str(exc)
    yield


app = FastAPI(
    title="Credit Risk PD API",
    description="Probabilidade de atraso > 30 dias (PD) e faixa de risco de um título a receber.",
    version="1.0.0",
    lifespan=lifespan,
)


class Titulo(BaseModel):
    vlr_nominal: float = Field(..., gt=0, examples=[850.0])
    numero_parcela: int = Field(..., ge=1, examples=[2])
    qtd_parcelas: int = Field(..., ge=1, examples=[3])
    prazo_dias: int = Field(..., ge=0, examples=[60])
    mes_vencimento: int = Field(..., ge=1, le=12, examples=[3])
    tempo_cliente_anos: float = Field(..., ge=0, examples=[4.5])
    hist_qtd_titulos: int = Field(0, ge=0, examples=[35])
    hist_taxa_late30: float = Field(0.0, ge=0, le=1, examples=[0.03])
    hist_qtd_late30_180d: int = Field(0, ge=0, examples=[0])
    hist_max_dias_atraso: int = Field(0, examples=[12])
    hist_dias_desde_ultimo_late30: int = Field(9999, ge=0, examples=[9999])
    exp_qtd_titulos_abertos: int = Field(0, ge=0, examples=[4])
    exp_vlr_aberto: float = Field(0.0, ge=0, examples=[2400.0])
    atual_qtd_vencidos: int = Field(0, ge=0, description="Títulos vencidos e não pagos na data de emissão")
    atual_vlr_vencido: float = Field(0.0, ge=0)
    atual_max_dias_vencido: int = Field(0, ge=0)
    rec_media_dias_atraso_90d: float = Field(
        0.0, description="Atraso médio dos pagamentos nos últimos 90 dias"
    )
    rec_qtd_pagamentos_90d: int = Field(0, ge=0)
    tipo_origem: Literal["DUP", "BOL", "CHQ"] = "DUP"
    porte: str = Field("03 - Médio", examples=["03 - Médio"])
    uf_cliente: str = Field("GO", min_length=2, max_length=2)
    divisao: str = "Atacado"
    uf_filial: str = Field("GO", min_length=2, max_length=2)
    sem_representante: Literal["True", "False"] = "False"


class Predicao(BaseModel):
    pd: float
    faixa_risco: str
    perda_esperada: float
    model_version: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if "model" in STATE else "sem_modelo",
        "model_version": STATE.get("version"),
        "erro": STATE.get("error"),
    }


@app.post("/reload")
def reload() -> dict:
    _load()
    STATE.pop("error", None)
    return {"model_version": STATE["version"]}


@app.post("/predict", response_model=list[Predicao])
def predict(titulos: list[Titulo]) -> list[Predicao]:
    if "model" not in STATE:
        raise HTTPException(503, "Modelo indisponível: rode o pipeline de treino/promoção.")
    X = pd.DataFrame([t.model_dump() for t in titulos])
    p = predict_pd(STATE["model"], X)
    bands = risk_band(p)
    return [
        Predicao(
            pd=round(float(pi), 6),
            faixa_risco=b,
            perda_esperada=round(float(pi * t.vlr_nominal), 2),
            model_version=str(STATE["version"]),
        )
        for pi, b, t in zip(p, bands, titulos, strict=True)
    ]
