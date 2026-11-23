select cod_produto, linha, preco_unitario
from {{ ref('stg_erp__produtos') }}
