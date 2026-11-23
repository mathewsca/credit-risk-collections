# Camada Power BI

O relatório Power BI consome o **mesmo warehouse** do app Streamlit. Toda a regra de negócio vive no dbt; o Power BI só agrega, sem lógica duplicada em Power Query.

## Montagem (cerca de 15 minutos)

1. Suba a stack (`docker compose up -d`) e rode o pipeline uma vez.
2. Power BI Desktop → **Obter dados → Banco de dados PostgreSQL**
   - Servidor: `localhost:5432` · Banco: `warehouse`
   - Credencial de banco de dados: `bi_reader` / `bi_reader` (somente leitura)
   - Modo: **Import**
3. Selecione as tabelas:
   - `marts.fct_titulos`, `marts.dim_empresa`, `marts.dim_cliente`, `marts.dim_representante`, `marts.dim_data`
   - `marts.mart_produtos_atraso`
   - `ml.scores`
4. Relacionamentos (modelo estrela, filtro em direção única):

   | De (muitos) | Para (um) |
   |---|---|
   | `fct_titulos[cod_empresa]` | `dim_empresa[cod_empresa]` |
   | `fct_titulos[cod_cliente]` | `dim_cliente[cod_cliente]` |
   | `fct_titulos[cod_representante]` | `dim_representante[cod_representante]` |
   | `fct_titulos[fk_data_vencimento]` | `dim_data[pk_data]` |
   | `scores[cod_empresa]` | `dim_empresa[cod_empresa]` |

5. Marque `dim_data` como **tabela de datas** (coluna `data`).
6. Crie a tabela `_Medidas` e cole as medidas de [`measures.dax`](measures.dax).
7. Crie o parâmetro de campo numérico **Capacidade** (50 a 5.000, incremento de 50) para a medida `Cobertura da Fila`.
8. Visuais: siga as páginas do app (Visão Geral, Carteira, Títulos, Risco).
9. **Arquivo → Salvar como → Projeto do Power BI (.pbip)** dentro desta pasta e faça o commit.
   O formato PBIP salva o modelo em TMDL (texto), o que permite revisar mudanças de medidas e relacionamentos em pull requests.

> O `.pbix` original **não** é versionado (`.gitignore`), porque contém dados reais.
