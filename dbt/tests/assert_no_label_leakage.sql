-- Teste singular anti-vazamento: o primeiro título de cada cliente não pode ter histórico,
-- e nenhum título pode "ver" mais títulos maturados do que existiam antes da sua emissão.
with f as (
    select
        cod_titulo,
        cod_cliente,
        dt_emissao,
        hist_qtd_titulos,
        row_number() over (partition by cod_cliente order by dt_emissao, cod_titulo) as ordem
    from {{ ref('ml_features') }}
),

maturados_antes as (
    select f.cod_titulo, count(t.cod_titulo) as n_possivel
    from f
    left join {{ ref('fct_titulos') }} as t
        on t.cod_cliente = f.cod_cliente
        and t.dt_vencimento + 31 <= f.dt_emissao
    group by f.cod_titulo
)

select f.*
from f
inner join maturados_antes as m using (cod_titulo)
where (f.ordem = 1 and f.hist_qtd_titulos > 0)
   or f.hist_qtd_titulos > m.n_possivel
