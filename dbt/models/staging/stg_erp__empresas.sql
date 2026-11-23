select
    filial_id::int     as cod_empresa,
    nome_filial,
    divisao,
    uf,
    cidade
from {{ source('erp', 'filiais') }}
