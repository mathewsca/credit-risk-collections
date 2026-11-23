-- Visão gerencial: KPIs por filial e mês de vencimento (alimenta "Índice por Mês / Filial / UF").
select
    date_trunc('month', f.dt_vencimento)::date                  as mes_vencimento,
    f.cod_empresa,
    e.nome_filial,
    e.uf                                                        as uf_filial,
    e.regiao,
    count(*)                                                    as qtd_titulos,
    sum(f.vlr_nominal)                                          as vlr_nominal,
    sum(f.vlr_pago)                                             as vlr_pago,
    sum(f.vlr_atraso)                                           as vlr_atraso,
    sum(f.vlr_atraso) / nullif(sum(f.vlr_nominal), 0)           as indice_inadimplencia,
    avg(case when f.label_disponivel then f.is_late30::int end) as taxa_late30
from {{ ref('fct_titulos') }} as f
inner join {{ ref('dim_empresa') }} as e using (cod_empresa)
group by 1, 2, 3, 4, 5
