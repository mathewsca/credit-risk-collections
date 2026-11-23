-- Features point-in-time para o modelo de PD (probabilidade de atraso > 30 dias).
-- Regra de ouro: toda feature usa SOMENTE informação disponível na data de emissão do título.
--   * histórico de atraso > 30d: apenas títulos cujo rótulo já era conhecido (vencimento + 31 <= emissão)
--   * situação atual: títulos já vencidos e ainda não pagos NA DATA DE EMISSÃO (sinal mais tempestivo)
--   * comportamento recente: atraso médio dos pagamentos feitos nos 90 dias anteriores à emissão
--   * dias de atraso: sempre truncados na data de emissão (nunca usa quitações futuras)
--   * tipo do título: tipo de ORIGEM (o tipo atual "TERC" só existe depois do calote -> vazamento)
-- Um único self-join (títulos anteriores do mesmo cliente) + agregações com FILTER.
with t as (
    select * from {{ ref('fct_titulos') }}
),

pares as (
    select
        cur.cod_titulo,
        cur.dt_emissao,
        prev.cod_titulo     as prev_titulo,
        prev.dt_vencimento  as prev_venc,
        prev.dt_quitacao    as prev_quit,
        prev.vlr_nominal    as prev_valor,
        prev.is_late30      as prev_late30,
        -- o título anterior estava pago na data de emissão do atual?
        (prev.dt_quitacao is not null and prev.dt_quitacao <= cur.dt_emissao) as prev_pago_na_emissao
    from t as cur
    inner join t as prev
        on prev.cod_cliente = cur.cod_cliente
        and prev.dt_emissao < cur.dt_emissao
),

agg as (
    select
        cod_titulo,
        -- histórico maturado (rótulo conhecido)
        count(*) filter (where prev_venc + 31 <= dt_emissao)                              as hist_qtd_titulos,
        avg(prev_late30::int) filter (where prev_venc + 31 <= dt_emissao)                 as hist_taxa_late30,
        count(*) filter (where prev_late30 and prev_venc + 31 <= dt_emissao
                         and prev_venc >= dt_emissao - 180)                               as hist_qtd_late30_180d,
        max(least(coalesce(prev_quit, dt_emissao), dt_emissao) - prev_venc)
            filter (where prev_venc < dt_emissao)                                         as hist_max_dias_atraso,
        dt_emissao - max(prev_venc) filter (where prev_late30 and prev_venc + 31 <= dt_emissao)
                                                                                          as hist_dias_desde_ultimo_late30,
        -- situação na data de emissão
        count(*) filter (where not prev_pago_na_emissao)                                  as exp_qtd_titulos_abertos,
        sum(prev_valor) filter (where not prev_pago_na_emissao)                           as exp_vlr_aberto,
        count(*) filter (where not prev_pago_na_emissao and prev_venc < dt_emissao)       as atual_qtd_vencidos,
        sum(prev_valor) filter (where not prev_pago_na_emissao and prev_venc < dt_emissao) as atual_vlr_vencido,
        max(dt_emissao - prev_venc) filter (where not prev_pago_na_emissao and prev_venc < dt_emissao)
                                                                                          as atual_max_dias_vencido,
        -- comportamento de pagamento recente (90 dias)
        avg(prev_quit - prev_venc) filter (where prev_pago_na_emissao
                                           and prev_quit >= dt_emissao - 90)              as rec_media_dias_atraso_90d,
        count(*) filter (where prev_pago_na_emissao and prev_quit >= dt_emissao - 90)     as rec_qtd_pagamentos_90d
    from pares
    group by cod_titulo, dt_emissao
)

select
    t.cod_titulo,
    t.cod_cliente,
    t.cod_empresa,
    t.dt_emissao,
    t.dt_vencimento,
    date_trunc('month', t.dt_vencimento)::date                                as safra_vencimento,
    -- características do título
    t.vlr_nominal::float                                                      as vlr_nominal,
    t.numero_parcela,
    t.qtd_parcelas,
    (t.dt_vencimento - t.dt_emissao)                                          as prazo_dias,
    t.tipo_origem,
    extract(month from t.dt_vencimento)::int                                  as mes_vencimento,
    (t.cod_representante = -1)                                                as sem_representante,
    -- cliente
    c.porte,
    c.uf                                                                      as uf_cliente,
    round(((t.dt_emissao - c.cliente_desde) / 365.25)::numeric, 2)::float     as tempo_cliente_anos,
    -- filial
    e.divisao,
    e.uf                                                                      as uf_filial,
    -- histórico maturado
    coalesce(a.hist_qtd_titulos, 0)                                           as hist_qtd_titulos,
    coalesce(a.hist_taxa_late30, 0)::float                                    as hist_taxa_late30,
    coalesce(a.hist_qtd_late30_180d, 0)                                       as hist_qtd_late30_180d,
    greatest(coalesce(a.hist_max_dias_atraso, 0), 0)                          as hist_max_dias_atraso,
    coalesce(a.hist_dias_desde_ultimo_late30, 9999)                           as hist_dias_desde_ultimo_late30,
    -- situação na emissão
    coalesce(a.exp_qtd_titulos_abertos, 0)                                    as exp_qtd_titulos_abertos,
    coalesce(a.exp_vlr_aberto, 0)::float                                      as exp_vlr_aberto,
    coalesce(a.atual_qtd_vencidos, 0)                                         as atual_qtd_vencidos,
    coalesce(a.atual_vlr_vencido, 0)::float                                   as atual_vlr_vencido,
    coalesce(a.atual_max_dias_vencido, 0)                                     as atual_max_dias_vencido,
    -- comportamento recente
    coalesce(a.rec_media_dias_atraso_90d, 0)::float                           as rec_media_dias_atraso_90d,
    coalesce(a.rec_qtd_pagamentos_90d, 0)                                     as rec_qtd_pagamentos_90d,
    -- alvo e metadados
    t.is_late30,
    t.label_disponivel,
    t.status,
    t.vlr_aberto::float                                                       as vlr_aberto
from t
inner join {{ ref('dim_cliente') }} as c using (cod_cliente)
inner join {{ ref('dim_empresa') }} as e using (cod_empresa)
left join agg as a using (cod_titulo)
