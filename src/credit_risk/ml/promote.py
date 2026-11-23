"""Promoção challenger -> champion com gate de qualidade (champion/challenger).

Os dois modelos são avaliados na MESMA janela out-of-time mais recente. O challenger só substitui
o champion se:
  * passar no piso absoluto (ROC-AUC >= 0,70), e
  * não for pior em discriminação (PR-AUC >= champion - 0,005), e
  * não piorar a calibração em mais de 5% (Brier).
Toda decisão fica registrada em ml.model_decisions (auditoria).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from credit_risk.ml.features import TARGET, load_features, temporal_split
from credit_risk.ml.metrics import evaluate
from credit_risk.ml.registry import client, get_alias_version, load_model, predict_pd
from credit_risk.settings import settings

MIN_ROC_AUC = 0.70
PR_AUC_TOLERANCE = 0.005
BRIER_TOLERANCE = 1.05


def decide(chal: dict, champ: dict | None) -> tuple[bool, str]:
    if chal["roc_auc"] < MIN_ROC_AUC:
        return False, f"challenger abaixo do piso de ROC-AUC ({chal['roc_auc']:.3f} < {MIN_ROC_AUC})"
    if champ is None:
        return True, "sem champion em produção"
    if chal["pr_auc"] < champ["pr_auc"] - PR_AUC_TOLERANCE:
        return False, f"PR-AUC pior ({chal['pr_auc']:.4f} vs {champ['pr_auc']:.4f})"
    if chal["brier"] > champ["brier"] * BRIER_TOLERANCE:
        return False, f"calibração pior (Brier {chal['brier']:.5f} vs {champ['brier']:.5f})"
    return True, f"challenger igual ou melhor (PR-AUC {chal['pr_auc']:.4f} vs {champ['pr_auc']:.4f})"


def run(features: pd.DataFrame | None = None) -> dict:
    chal_v = get_alias_version("challenger")
    if chal_v is None:
        raise RuntimeError("Nenhum challenger registrado. Rode o treino antes.")
    champ_v = get_alias_version("champion")
    if champ_v is not None and champ_v.version == chal_v.version:
        return {"promoted": False, "reason": "challenger já é o champion", "version": chal_v.version}

    test = temporal_split(load_features(features)).test
    y = test[TARGET].astype(int).to_numpy()
    chal = evaluate(y, predict_pd(load_model("challenger"), test))
    champ = evaluate(y, predict_pd(load_model("champion"), test)) if champ_v else None
    promoted, reason = decide(chal, champ)
    if promoted:
        client().set_registered_model_alias(settings.model_name, "champion", chal_v.version)

    decision = {
        "decided_at": datetime.now(UTC).replace(tzinfo=None),
        "challenger_version": int(chal_v.version),
        "champion_version": int(champ_v.version) if champ_v else None,
        "challenger_pr_auc": chal["pr_auc"],
        "challenger_roc_auc": chal["roc_auc"],
        "challenger_brier": chal["brier"],
        "champion_pr_auc": champ["pr_auc"] if champ else None,
        "champion_brier": champ["brier"] if champ else None,
        "promoted": promoted,
        "reason": reason,
    }
    try:
        from credit_risk.db import copy_df

        copy_df(pd.DataFrame([decision]), "model_decisions", schema="ml", mode="append")
    except Exception as exc:  # noqa: BLE001 - auditoria não deve derrubar a promoção
        print(f"[promote] aviso: não registrou auditoria ({exc})")
    print(f"[promote] {'PROMOVIDO' if promoted else 'mantido'}: v{chal_v.version} | {reason}")
    return decision


if __name__ == "__main__":
    run()
