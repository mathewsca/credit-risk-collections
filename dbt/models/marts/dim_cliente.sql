select
    cod_cliente,
    razao_social,
    porte,
    uf,
    cidade,
    cliente_desde,
    cod_empresa_carteira,
    cod_representante_carteira
from {{ ref('stg_erp__clientes') }}
