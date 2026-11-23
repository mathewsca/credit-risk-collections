# Data card: contas a receber sintéticas da Distribuidora simulada

## Origem e privacidade
- **100% sintético.** Nenhum registro, nome, documento, código ou valor real é usado ou publicado.
- Os parâmetros do gerador (`config/generator.yaml`) têm **ordens de grandeza plausíveis** para o atacado/varejo brasileiro: taxa de atraso > 30 dias em torno de 2%, maioria dos pagamentos até o vencimento e forte persistência do atraso no mesmo cliente.
- **Anonimização por construção:**
  - IDs **aleatórios e não sequenciais**, com quantidade de dígitos própria (título com 12, nota com 11, cliente com 10...).
  - Nomes de pessoas e empresas gerados pelo `Faker` (pt_BR), em formato composto.
  - Produtos **somente por código** e linha genérica (A–H), sem descrição.
  - Geografia (Centro-Oeste/Sudeste) e calendário (2023–2024) próprios.
  - Valores monetários com **perturbação aleatória de ±1–5% por nota**.
- A empresa é fictícia.

## Mecanismo gerador (verdade conhecida)
1. **Cliente**: porte, UF, tempo de relacionamento, fragilidade estrutural (latente) e heterogeneidade não observável.
2. **Regime mensal (cadeia de Markov)**: adimplente ↔ em dificuldade. O hazard de entrada depende do perfil e da sazonalidade (jan–mar); a persistência é de 80% ao mês, com duração média de 5 meses.
3. **Título**: no regime "em dificuldade", 88% dos títulos atrasam; no adimplente, o atraso é raro e depende de valor, número da parcela, tipo do título e de um choque por nota.
4. **Pagamento**: mistura de pagamento antecipado, no prazo, com atraso curto, com atraso longo e nunca pago.
5. **Pós-evento**: títulos não pagos há mais de 90 dias viram `TERC` (cobrança terceirizada). É uma armadilha de vazamento intencional e documentada: o modelo usa `tipo_origem`.

Como o gerador salva a **PD verdadeira** de cada título (`data/ground_truth/`, fora do git), dá para medir a distância entre o modelo e o **classificador oráculo**, algo impossível com dados reais.

## Volumes (cenário base)
30 filiais · 140 representantes · 6.000 clientes · 2.500 produtos · cerca de 78 mil notas · cerca de 175 mil títulos · cerca de 520 mil itens.

## Cenário de choque (`--scenario shock`)
A partir de out/2024: ticket médio +25% (inflação) e maior entrada em dificuldade, sobretudo em GO e MG. Serve para demonstrar a detecção de drift e o comportamento do gate de promoção.

## Limitações
- O regime de crise é homogêneo dentro de cada UF; na realidade há contágio setorial e por cidade.
- Não há renegociação de dívidas nem pagamentos parciais.
