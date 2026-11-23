-- Títulos a receber: tipagem e nomes padronizados.
select
    titulo_id::bigint                      as cod_titulo,
    nota_id::bigint                        as cod_nota,
    filial_id::int                         as cod_empresa,
    cliente_id::bigint                     as cod_cliente,
    numero_documento::bigint               as numero_titulo,
    parcela::int                           as numero_parcela,
    tipo_origem,
    tipo_atual,
    dt_emissao::date                       as dt_emissao,
    dt_vencimento::date                    as dt_vencimento,
    dt_pagamento::date                     as dt_quitacao,
    valor_nominal::numeric(14, 2)          as vlr_nominal,
    valor_juros::numeric(14, 2)            as vlr_juro,
    valor_pago::numeric(14, 2)             as vlr_pago,
    case situacao when 'PAGO' then 'Q' else 'A' end as status
from {{ source('erp', 'titulos') }}
