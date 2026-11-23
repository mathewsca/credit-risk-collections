select
    cliente_id::bigint                         as cod_cliente,
    razao_social,
    coalesce(porte, 'Nao Cadastrado')          as porte,
    uf,
    cidade,
    cliente_desde::date                        as cliente_desde,
    filial_id::int                             as cod_empresa_carteira,
    representante_id::int                      as cod_representante_carteira
from {{ source('erp', 'clientes') }}
