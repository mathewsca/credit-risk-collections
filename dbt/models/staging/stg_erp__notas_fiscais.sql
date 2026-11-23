select
    nota_id::bigint                as cod_nota,
    numero_documento::bigint       as numero_nota,
    filial_id::int                 as cod_empresa,
    cliente_id::bigint             as cod_cliente,
    representante_id::int          as cod_representante,
    dt_emissao::date               as dt_emissao,
    valor_total::numeric(14, 2)    as vlr_total,
    qtd_parcelas::int              as qtd_parcelas
from {{ source('erp', 'notas') }}
