-- "Top produtos em atraso": em quantos títulos vencidos e não pagos cada produto aparece.
-- Produtos identificados apenas por código e linha genérica.
select
    p.cod_produto,
    p.linha,
    count(distinct f.cod_titulo)   as qtd_titulos_atraso,
    sum(f.vlr_atraso)              as vlr_atraso_titulos
from {{ ref('fct_titulos') }} as f
inner join {{ ref('stg_erp__notas_itens') }} as i using (cod_nota)
inner join {{ ref('dim_produto') }} as p using (cod_produto)
where f.vlr_atraso > 0
group by 1, 2
