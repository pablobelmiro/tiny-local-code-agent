# CPU Small-Model Agent — Design Spec

Data: 2026-09-22

## Contexto e objetivo

Este é um fork de aprendizado do [`avbiswas/neural-code`](https://github.com/avbiswas/neural-code),
um agente de código de terminal minimalista (loop de tools + LLM). O objetivo
não é construir um produto: é um projeto pessoal para aprender como um agente
de código funciona por dentro, adaptando-o para rodar com modelos pequenos
(2B/8B/~14B) em CPU via Ollama (ou qualquer endpoint compatível com a API da
OpenAI — container, API remota etc.), com visibilidade de uso de tokens e um
limite forte de gasto de tokens por sessão.

O código-base do `neuralcode` já cobre boa parte disso: endpoint configurável
via `BASE_URL`/`API_KEY`/`MODEL`, tracking de `usage` (prompt/completion/
reasoning/cached tokens) exibido na UI, compactação automática de contexto, e
um conjunto de tools (bash, leitura/escrita de arquivo, edição) com permissões
e sandbox.

Este spec cobre apenas as duas peças que faltam:

1. Perfis de modelo prontos para rodar em CPU com modelos pequenos.
2. Um limite forte (hard budget) de tokens por sessão.

## Fora de escopo (por agora)

- Extração de `<think>` / raciocínio de modelos locais que não usam o campo
  `reasoning_tokens` da OpenAI (ex.: tags customizadas de alguns modelos
  open-weight). Fica como experimento futuro.
- Budget por turno (só por sessão, ver decisão abaixo).
- Qualquer mudança na lógica de tools, sandbox ou permissões existente.
- Benchmarking formal de qualidade dos modelos pequenos em tool-calling —
  isso será avaliado na prática rodando o agente, não faz parte da spec.

## Componente 1: perfis de modelo (`profiles.py` + `models.yaml`)

### Arquivo `models.yaml` (raiz do projeto)

Lista de perfis nomeados, cada um com os campos usados hoje em `config.py`
mais o novo `token_budget`:

```yaml
profiles:
  small-2b:
    model: qwen2.5-coder:1.5b
    context_window: 32000
    token_budget: 60000
  medium-8b:
    model: qwen2.5-coder:7b
    context_window: 32000
    token_budget: 120000
  large-14b:
    model: qwen2.5-coder:14b
    context_window: 32000
    token_budget: 200000
```

`base_url` e `api_key` **não** entram no perfil — continuam vindo só de env
vars (`BASE_URL`/`API_KEY`), já que isso é sobre onde o backend roda (local,
container, remoto), não sobre qual modelo. Um perfil descreve o modelo, não a
infraestrutura.

`context_window` nos perfis é conservador (32k) porque modelos pequenos
rodando em CPU via Ollama costumam ser configurados com contexto reduzido por
padrão para manter velocidade aceitável.

### `neuralcode/profiles.py`

- `load_profiles() -> dict`: lê `models.yaml` da raiz do projeto (mesmo nível
  de `pyproject.toml`). Se o arquivo não existir, retorna `{}` (não é erro —
  perfis são opcionais).
- `resolve_profile(name: str | None) -> dict | None`: retorna o dict do
  perfil pelo nome, ou `None` se `name` for `None` ou não encontrado.
- Nenhuma validação elaborada de schema — é um projeto pessoal; erros de
  YAML mal formado propagam naturalmente (fail-fast, sem silenciar).

### Mudanças em `config.py`

- Novo argumento de CLI `--profile` em `agent.py`, e/ou env var `PROFILE`.
- Ordem de resolução para `MODEL`, `CONTEXT_WINDOW`, `TOKEN_BUDGET`: env var
  explícita (se setada) > valor do perfil ativo (se houver) > default atual
  do código.
- `BASE_URL`/`API_KEY` continuam exclusivamente de env vars, como já é hoje
  (mantém o contrato "endpoint configurável" pedido: local, container ou API
  remota, sem mudança de código).

## Componente 2: limite forte de tokens (`budget.py`)

### Decisão: budget por sessão, hard-stop com confirmação

Um único contador acumulado (`prompt_tokens + completion_tokens` de todas as
chamadas ao LLM na sessão atual). Quando esse total cruza `TOKEN_BUDGET`
(resolvido como no Componente 1), o agente:

1. Mostra na UI quantos tokens foram usados e o teto configurado.
2. Pede confirmação (`y/N`) antes de fazer a próxima chamada ao LLM.
3. Se o usuário disser não, o turno atual é abortado sem chamar o LLM
   (mensagem do usuário já ficou salva na sessão, pode ser retomada depois).
4. Se disser sim, o agente segue normalmente e só pergunta de novo depois de
   ultrapassar o próximo múltiplo do budget (ex.: a cada +50% do budget
   original acima do teto), para não interromper a cada turno depois que o
   usuário já aceitou continuar.

Um budget de por-turno foi considerado e descartado por simplicidade (ver
pergunta ao usuário) — pode ser adicionado depois se o hard-stop por sessão
se mostrar insuficiente na prática.

### `neuralcode/budget.py`

- Estado simples em módulo (mesmo padrão de outros módulos do projeto, que
  não usa classes para estado de sessão): `_total_tokens = 0`,
  `_last_warned_at = 0`.
- `track(usage: dict) -> None`: soma `usage["prompt_tokens"] +
  usage["completion_tokens"]` ao total.
- `remaining() -> int | None`: `TOKEN_BUDGET - total`, ou `None` se não há
  budget configurado (perfil/env ausente — budget é opcional, como os
  perfis).
- `should_warn() -> bool`: `True` quando `total` cruzou `TOKEN_BUDGET` e
  ainda não avisamos para o múltiplo atual.
- `mark_warned() -> None`: atualiza `_last_warned_at` para o múltiplo atual.

### Mudanças em `agent.py`

- Após `ui.usage(usage)`, chamar `budget.track(usage)`.
- No topo do loop interno (antes de `call_llm`), checar
  `budget.should_warn()`; se `True`, chamar `ui.budget_warning(...)` (nova
  função em `ui.py`) e pedir confirmação via `input()` (mesmo padrão simples
  já usado em `ui.ask()`). Se negado, `break` do loop interno sem chamar o
  LLM.

## UI (`ui.py`)

- `usage(usage: dict)` ganha uma linha extra: budget usado, ex.
  `"budget: 45,200 / 120,000 tokens (37%)"`, só exibida quando há budget
  configurado (`budget.remaining() is not None`).
- Campos `reasoning_tokens`/`cached_tokens` continuam sendo exibidos só
  quando não são `None` (comportamento já existente via `getattr(...,
  None)` em `llm.py` — nenhuma mudança necessária aqui, só confirmando que
  funciona com Ollama, que tipicamente não popula esses campos).
- Nova função `budget_warning(total, limit)` — imprime aviso e faz a
  pergunta de confirmação (chamada por `agent.py`, não pelo próprio
  `ui.py` fazendo I/O escondido).

## Testes

Dado o caráter de projeto de aprendizado pessoal (sem CI configurada), os
testes cobrem só a lógica pura, sem mockar o LLM real:

- `test_profiles.py`: parsing de `models.yaml`, fallback quando arquivo não
  existe, resolução de perfil por nome inválido (`None`).
- `test_budget.py`: acumulação de `track()`, `should_warn()` cruzando o
  limite, não repetir aviso no mesmo múltiplo, `remaining()` quando budget
  não configurado.

Validação end-to-end é manual: rodar `neuralcode` contra o Ollama local do
usuário com cada perfil e observar tool-calling, uso de tokens e o aviso de
budget funcionando.

## Perfis iniciais sugeridos

| Perfil | Modelo Ollama | Tamanho | Uso |
|---|---|---|---|
| `small-2b` | `qwen2.5-coder:1.5b` | ~2B | Testes rápidos, tarefas triviais |
| `medium-8b` | `qwen2.5-coder:7b` | ~7-8B | Bom equilíbrio qualidade/velocidade em CPU |
| `large-14b` | `qwen2.5-coder:14b` | ~14B | Melhor qualidade de tool-calling disponível, mais lento em CPU |

`qwen2.5-coder` foi escolhido pela família por ser tanto pequena/leve quanto
orientada a código e com bom suporte a tool-calling — relevante já que
modelos pequenos genéricos tendem a ser fracos em seguir schemas de tool
call.
