-- Produtos identificados apenas por código e linha genérica (sem descrição).
select
    produto_id::int                 as cod_produto,
    linha,
    preco_unitario::numeric(12, 2)  as preco_unitario
from {{ source('erp', 'produtos') }}
