select
    representante_id::int  as cod_representante,
    nome                   as nome_representante,
    filial_id::int         as cod_empresa,
    dt_admissao::date      as dt_admissao
from {{ source('erp', 'representantes') }}
