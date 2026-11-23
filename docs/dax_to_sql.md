# Migração das regras de negócio: Power BI (DAX/Power Query) → SQL versionado (dbt)

No relatório original, a lógica ficava espalhada em três lugares: SQL embutido no Power Query, colunas calculadas e medidas DAX. Isso dificulta revisão, teste e reuso. Na reconstrução, **toda a regra vive no dbt**: versionada no git, testada a cada execução e documentada com linhagem.

| Regra | Antes | Agora (dbt) | Observação |
|---|---|---|---|
| Título em atraso = vencimento anterior à data de pagamento (ou a hoje, se em aberto) | SQL no Power Query, com `getdate()` | `fct_titulos.flg_atraso` | A data "de hoje" virou uma **data de snapshot** (`var('snapshot_date')`), então os resultados são **reprodutíveis** e não mudam conforme o dia da atualização |
| Valor em atraso = saldo em aberto dos títulos em atraso | SQL no Power Query | `fct_titulos.vlr_atraso` | — |
| Índice de inadimplência = Σ valor em atraso ÷ Σ valor nominal | Medida DAX | `mart_inadimplencia_mensal.indice_inadimplencia` e app | Mesma definição de negócio, agregada na granularidade do visual |
| Quantidade de títulos | Medida DAX (`DISTINCTCOUNT`) | `count(*)` | A unicidade de `cod_titulo` é garantida por teste `unique` |
| Produtos presentes em títulos vencidos | Tabela ponte + medida DAX | `mart_produtos_atraso` | A ponte virou um join com os itens da nota |
| Agrupamento de filiais para filtro | Coluna calculada DAX | `dim_empresa.regiao` | Atributo de dimensão, e não lógica no relatório |
| Filtros de tipo e situação do título | SQL no Power Query | Contrato de dados na ingestão (`pandera`) + `accepted_values` no dbt | A regra virou **teste**: se a origem mudar, o pipeline falha em vez de mostrar números errados |

## Novas regras (não existiam no original)

| Regra | Modelo | Motivo |
|---|---|---|
| `faixa_aging` | `fct_titulos` | Gestão da carteira por idade do atraso (prática padrão de contas a receber) |
| `is_late30` + `label_disponivel` | `fct_titulos` | Alvo do modelo de PD; o rótulo só existe 31 dias após o vencimento |
| Features point-in-time | `ml_features` | Histórico do cliente sem vazamento de informação do futuro (teste `assert_no_label_leakage`) |
| `tipo_origem` × `tipo_atual` | `stg_erp__titulos` | `TERC` (cobrança terceirizada) só existe **depois** do calote. Usar o tipo atual no modelo seria vazamento |
