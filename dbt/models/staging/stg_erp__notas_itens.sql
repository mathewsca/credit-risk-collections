select
    nota_id::bigint      as cod_nota,
    produto_id::int      as cod_produto,
    quantidade::int      as quantidade
from {{ source('erp', 'notas_itens') }}
