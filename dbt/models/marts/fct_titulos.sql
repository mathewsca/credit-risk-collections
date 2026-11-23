-- Fato de títulos a receber. Grão: parcela.
-- Regras replicadas da origem (SQL embutido no Power Query do relatório original):
--   ATRASO    = vencimento < coalesce(quitação, hoje)
--   VLRATRASO = se ATRASO então nominal - pago, senão 0
-- Índice de inadimplência (medida DAX "Indice Inad") = sum(vlr_atraso) / sum(vlr_nominal)
{% set snapshot = "'" ~ var('snapshot_date') ~ "'::date" %}
with t as (
    select * from {{ ref('stg_erp__titulos') }}
),

nf as (
    select cod_nota, cod_representante, qtd_parcelas from {{ ref('stg_erp__notas_fiscais') }}
),

base as (
    select
        t.*,
        coalesce(nf.cod_representante, -1)          as cod_representante,
        nf.qtd_parcelas,
        coalesce(t.dt_quitacao, {{ snapshot }})     as dt_referencia,
        t.vlr_nominal - t.vlr_pago                  as vlr_aberto
    from t
    left join nf using (cod_nota)
)

select
    base.*,
    to_char(dt_vencimento, 'YYYYMMDD')::int                                  as fk_data_vencimento,
    (dt_vencimento < dt_referencia)                                          as flg_atraso,
    case when dt_vencimento < dt_referencia then vlr_aberto else 0 end       as vlr_atraso,
    greatest(dt_referencia - dt_vencimento, 0)                               as dias_atraso,
    case
        when status = 'Q' then 'Quitado'
        when dt_vencimento >= {{ snapshot }} then 'A vencer'
        when {{ snapshot }} - dt_vencimento <= 30 then '01-30 dias'
        when {{ snapshot }} - dt_vencimento <= 60 then '31-60 dias'
        when {{ snapshot }} - dt_vencimento <= 90 then '61-90 dias'
        when {{ snapshot }} - dt_vencimento <= 180 then '91-180 dias'
        else '180+ dias'
    end                                                                      as faixa_aging,
    -- Alvo do modelo: atraso > 30 dias (pago após 30 dias, ou ainda em aberto após 30 dias)
    (dt_referencia - dt_vencimento) > 30                                     as is_late30,
    -- O rótulo só é conhecido 31 dias após o vencimento (não rotular como "bom" o que ainda não maturou)
    (dt_vencimento + 31 <= {{ snapshot }})                                   as label_disponivel
from base
