# Model card: `credit-risk-pd`

*Formato inspirado em Mitchell et al. (2019), "Model Cards for Model Reporting", FAT\*.*

## Visão geral
| | |
|---|---|
| **Tarefa** | Classificação binária: o título a receber vai atrasar **mais de 30 dias**? |
| **Saída** | PD (probabilidade de inadimplência) calibrada, faixa de risco A–E e perda esperada (PD × saldo) |
| **Modelo em produção** | LightGBM (busca em 3 configurações, `deterministic=True`) com calibração isotônica |
| **Baseline obrigatório** | Regressão logística (L2, C = 0,5), que assume a produção sempre que o LightGBM não for significativamente melhor na validação ([detalhes](model_selection.md)) |
| **Registro** | MLflow Model Registry, `credit-risk-pd@champion`, serialização segura (skops) com allowlist de tipos |

## Uso pretendido
- **Sim:** priorizar a cobrança **preventiva** de títulos em aberto que ainda não completaram 30 dias de atraso; apoiar o dimensionamento da equipe de cobrança.
- **Não:** recusar ou conceder crédito de forma automática; precificar juros; decisões sobre pessoas físicas. **O modelo não passou por avaliação de vieses para esses usos**.

## Dados
Dados **sintéticos** com mecanismo causal conhecido (ver [data card](data_card.md)). Separação temporal por safra de vencimento: treino (mais antigo), validação (3 meses seguintes) e teste out-of-time (últimos 3 meses com rótulo maturado).

## Features (14 numéricas + 5 categóricas)
Todas calculadas **point-in-time** na data de emissão (`dbt/models/ml/ml_features.sql`):

| Grupo | Features |
|---|---|
| Título | valor nominal, nº da parcela, qtd. de parcelas, mês de vencimento, tipo de origem |
| Cliente | porte, UF, tempo de relacionamento |
| Histórico maturado | qtd. de títulos, taxa de atraso > 30d, atrasos > 30d nos últimos 180 dias, maior atraso |
| Situação na emissão | títulos em aberto, saldo em aberto, **títulos vencidos e não pagos** |
| Comportamento recente | atraso médio e qtd. de pagamentos nos últimos 90 dias |

**Removidas por redundância** (|ρ de Spearman| > 0,9): dias desde o último atraso, valor e dias máximos vencidos, prazo em dias e UF da filial (93,8% igual à UF do cliente). A colinearidade gerava coeficientes com sinais contraditórios.

## Métricas
Ver [README › Resultados](../README.md#-resultados) e `reports/metrics.json` (gerado a cada treino, com IC 95% por bootstrap).

## Monitoramento e manutenção
| Sinal | Limiar | Ação |
|---|---|---|
| PSI do score | > 0,25 | Retreinar |
| PSI de features (exceto calendário) | > 0,25 | Retreinar |
| Queda de ROC-AUC na safra mais recente | > 0,05 vs. teste | Retreinar |
| Observado / esperado (calibração no agregado) | fora de [0,67; 1,5] | Retreinar |
| Promoção do challenger | ROC-AUC ≥ 0,70; PR-AUC ≥ champion − 0,005; Brier ≤ champion × 1,05 | Senão, champion mantido e decisão auditada |

## Limitações e riscos
- **Defasagem do rótulo:** o desempenho real só é conhecido 31 dias após o vencimento. O PSI serve como alerta antecipado.
- **Rupturas de regime:** logo após um choque não há rótulos do novo regime. O gate impede promover um modelo pior, mas a recalibração precisa de algumas safras (ver o cenário de choque no README). Próximo passo natural: recalibração do intercepto (Steyerberg, 2019, *Clinical Prediction Models*, cap. 20).
- **LGD = 100%** na perda esperada: premissa conservadora e explícita.
- **Distância até o oráculo:** parte da informação (o regime latente do cliente no mês do vencimento) não é observável na emissão. É um limite do problema, não do algoritmo.
