select
    cod_empresa,
    nome_filial,
    divisao,
    uf,
    cidade,
    case
        when uf in ('GO', 'DF', 'MT', 'MS') then 'Centro-Oeste'
        when uf in ('MG', 'SP', 'RJ', 'ES') then 'Sudeste'
        when uf in ('AM', 'AC', 'RO', 'PA', 'TO', 'RR', 'AP') then 'Norte'
        when uf in ('PR', 'SC', 'RS') then 'Sul'
        else 'Nordeste'
    end as regiao
from {{ ref('stg_erp__empresas') }}
