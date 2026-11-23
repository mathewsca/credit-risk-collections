# Por que estes modelos? Evidências para a escolha

> Princípio: a escolha do algoritmo é uma **hipótese a ser testada**, não uma preferência. A literatura define os candidatos; os dados deste problema decidem entre eles, com um teste estatístico.

## 1. Candidatos e o que a evidência externa diz

### Regressão logística (baseline obrigatório)
| Evidência | O que mostra |
|---|---|
| Lessmann, Baesens, Seow & Thomas (2015). *Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research*. **EJOR** 247(1), 124–136. [doi:10.1016/j.ejor.2015.05.030](https://doi.org/10.1016/j.ejor.2015.05.030) | O maior benchmark de credit scoring (41 classificadores, 8 bases reais) mostra que a regressão logística continua sendo a referência do setor. Ensembles a superam, mas o ganho precisa ser avaliado pelo impacto financeiro, e não só pelo AUC |
| Siddiqi (2017). *Intelligent Credit Scoring*, Wiley | A logística é o padrão de mercado para scorecards: é interpretável, auditável e aceita por reguladores |
| Basel Committee (2005). *Studies on the Validation of Internal Rating Systems* (WP 14) | Validação baseada em discriminação, calibração e estabilidade. Um modelo simples bem calibrado é o ponto de partida |

**Papel no projeto:** é o piso que qualquer modelo mais complexo precisa superar **com significância estatística**.

### Gradient Boosting (LightGBM): o candidato "estado da arte" para dados tabulares
| Evidência | O que mostra |
|---|---|
| Gunnarsson, vanden Broucke, Baesens, Óskarsdóttir & Lemahieu (2021). *Deep learning for credit scoring: Do or don't?* **EJOR** 295(1), 292–305. [doi:10.1016/j.ejor.2021.03.006](https://doi.org/10.1016/j.ejor.2021.03.006) | Em várias bases de crédito, o **XGBoost (boosting) teve o melhor desempenho**; redes profundas (MLP, DBN) **não superaram** modelos rasos e custaram muito mais. Conclusão dos autores: preferir boosting |
| Grinsztajn, Oyallon & Varoquaux (2022). *Why do tree-based models still outperform deep learning on typical tabular data?* **NeurIPS** Datasets & Benchmarks. [arXiv:2207.08815](https://arxiv.org/abs/2207.08815) | Em 45 bases tabulares de tamanho médio (~10 mil amostras), **modelos de árvore seguem como estado da arte**. Redes têm dificuldade com features não informativas e funções irregulares, que são características típicas de dados de crédito |
| Shwartz-Ziv & Armon (2021). *Tabular Data: Deep Learning is Not All You Need*. [arXiv:2106.03253](https://arxiv.org/abs/2106.03253) | O XGBoost supera TabNet, NODE e outros modelos profundos, **inclusive nas bases dos artigos que os propuseram**, e exige bem menos ajuste |
| McElfresh et al. (2023). *When Do Neural Nets Outperform Boosted Trees on Tabular Data?* **NeurIPS**. [arXiv:2305.02997](https://arxiv.org/abs/2305.02997) | 19 algoritmos em 176 bases: GBDTs lidam melhor com features **assimétricas e de cauda pesada**, como valores monetários e contagens. Em muitas bases, **ajustar bem um GBDT importa mais do que trocar de família** de modelo |
| Rubachev et al. (2024). *TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks*. [arXiv:2406.19380](https://arxiv.org/abs/2406.19380) | Em bases industriais com **separação temporal**, GBDT e MLPs simples lideram. **Separações temporais mudam o ranking** dos métodos em relação às aleatórias, o que justifica nosso protocolo out-of-time |
| Ke et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. **NeurIPS** | Escolha de LightGBM em vez de XGBoost: desempenho equivalente com treino mais rápido (histogramas, GOSS e EFB), útil para retreinos frequentes no pipeline |

### Evidência da prática (Kaggle e repositórios)
| Fonte | O que mostra |
|---|---|
| Kaggle: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (2018, mais de 7 mil times) | A solução vencedora foi um *stack* de dezenas de modelos ([post do time vencedor](https://www.kaggle.com/c/home-credit-default-risk/discussion/64480)), e as soluções abertas de referência são baseadas em LightGBM ([neptune-ai/open-solution-home-credit](https://github.com/minerva-ml/open-solution-home-credit); [kozodoi/Kaggle_Home_Credit](https://github.com/kozodoi/Kaggle_Home_Credit)). **O diferencial dos vencedores foi a engenharia de features sobre o histórico de pagamentos**, não o algoritmo |
| Kaggle: [AMEX Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction) (2022) | Problema de inadimplência com séries de comportamento por cliente. As soluções públicas de topo usam GBDTs (LightGBM com DART) como base ([exemplo](https://github.com/Nicolas-Bolouri/LGBM-Solution-Amex-Kaggle)) |
| Kaggle: [Home Credit – Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) (2024) | A própria métrica da competição pune a **perda de desempenho ao longo do tempo**: estabilidade virou critério explícito de qualidade. Isso fundamenta o nosso monitoramento por safra e o PSI |

**Lições para o projeto:** em competições e em produção, o maior ganho vem de **features de comportamento recente do cliente**, o que confirma-se aqui (seção 3).

### Por que NÃO deep learning, AutoML ou modelos de fronteira?
- **Deep learning**: evidência consistente de que **não** supera boosting em dados tabulares de crédito (Gunnarsson et al., 2021; Grinsztajn et al., 2022), com custo e opacidade maiores.
- **TabPFN** (McElfresh et al., 2023): competitivo, mas limitado a poucos milhares de linhas de treino. Esse caso tem mais de 100 mil.
- **Simplicidade como requisito**: em crédito, explicabilidade e auditabilidade são exigências de negócio e regulatórias, com auditoria constante. Cada componente a mais precisa se justificar com ganho mensurável.

## 2. Protocolo de decisão

1. **Separação temporal** train < valid < test (out-of-time), seguindo a evidência do TabReD de que separações aleatórias distorcem conclusões.
2. Busca pequena de hiperparâmetros do LightGBM, avaliada na **validação**.
3. **Teste de superioridade pareado**: bootstrap pareado (1.000 reamostragens) da diferença de PR-AUC (LightGBM − logística) na validação.
   - LightGBM só é escolhido se o **limite inferior do IC 95% for > 0**.
   - Caso contrário, prevalece a **regra da parcimônia**: modelo mais simples, interpretável e barato de manter.
4. **Calibração isotônica** na validação (Niculescu-Mizil & Caruana, 2005): boosting tende a gerar probabilidades mal calibradas, e a PD alimenta um cálculo financeiro (perda esperada).
5. O **teste** é usado uma única vez, para reportar com IC 95% (bootstrap).
6. **Oráculo**: como os dados são sintéticos com mecanismo conhecido, comparamos com a PD verdadeira, o limite superior teórico.

## 3. Resultado e interpretação

| | ROC-AUC (IC 95%) | PR-AUC (IC 95%) |
|---|---|---|
| Regressão logística | 0,802 [0,780; 0,824] | 0,228 [0,198; 0,257] |
| LightGBM calibrado | 0,799 [0,778; 0,817] | 0,219 [0,190; 0,252] |
| Oráculo (PD verdadeira) | 0,967 | 0,795 |

**Decisão pelo protocolo:** na validação, o LightGBM superou a logística com significância (ΔPR-AUC = +0,063, IC 95% pareado [+0,033; +0,089], 100% das reamostragens a favor), então foi selecionado.

**O que o teste mostrou:** no trimestre seguinte (out–dez/2024), os dois **empatam** (ICs sobrepostos). Isso é coerente com a literatura (Lessmann et al., 2015; McElfresh et al., 2023): a vantagem do boosting em dados tabulares costuma ser pequena e instável entre períodos. Não altera-se a regra depois de ver o teste (isso seria *p-hacking*). A consequência prática é monitorar por safra, o que o pipeline faz.

**Achado principal: features importam mais que algoritmo.** Na primeira versão, que usava apenas o histórico de rótulos já maturados, o ROC-AUC era de 0,72. Ao incluir sinais **tempestivos e legítimos na data de emissão** (títulos vencidos em aberto, comportamento de pagamento dos últimos 90 dias), o ROC-AUC chegou a 0,80. A troca de algoritmo pouco mudou o resultado. Isso reproduz, em escala menor, a lição do Kaggle Home Credit.

**Variância entre safras:** o PR-AUC mensal oscila bastante (cerca de 180 eventos por mês). Por isso todas as métricas são reportadas com IC 95% e o monitoramento acompanha o desempenho safra a safra.
