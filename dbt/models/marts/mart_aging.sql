-- Carteira em aberto por faixa de atraso (aging list) e filial.
select
    f.cod_empresa,
    e.nome_filial,
    f.faixa_aging,
    count(*)            as qtd_titulos,
    sum(f.vlr_aberto)   as vlr_aberto
from {{ ref('fct_titulos') }} as f
inner join {{ ref('dim_empresa') }} as e using (cod_empresa)
where f.status = 'A'
group by 1, 2, 3
