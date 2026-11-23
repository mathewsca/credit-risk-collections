select cod_representante, nome_representante, cod_empresa, dt_admissao
from {{ ref('stg_erp__representantes') }}
union all
-- membro desconhecido: vendas de balcão sem representante (evita perder linhas em joins)
select -1, 'Sem representante', null, null
