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

Este spec cobre as peças que faltam:

1. Perfis de modelo prontos para rodar em CPU com modelos pequenos.
2. Um limite forte (hard budget) de tokens por sessão, com reset de contexto
   ao ser confirmado.
3. Um subdiretório de trabalho por sessão dentro do repositório, escolhido
   interativamente no início do agente.

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
4. Se disser sim, o agente **limpa o contexto**: `messages` volta a conter só
   `{"role": "system", "content": SYSTEM_PROMPT}` (equivalente a reiniciar a
   conversa), o contador de budget é zerado, e o agente continua no mesmo
   subdiretório de trabalho (Componente 3) — os arquivos criados/editados
   até ali permanecem, só a conversa é resetada. Isso é diferente de
   `/compact` (que resume o histórico mantendo continuidade): aqui é um
   corte limpo, porque o objetivo é permitir sessões longas de trabalho no
   mesmo diretório mesmo com um budget pequeno por "janela" de conversa.
   O usuário é avisado explicitamente de que o contexto foi limpo.

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
- `reset() -> None`: zera `_total_tokens` e `_last_warned_at` (chamado após
  o reset de contexto ser confirmado).

### Mudanças em `agent.py`

- Após `ui.usage(usage)`, chamar `budget.track(usage)`.
- No topo do loop interno (antes de `call_llm`), checar
  `budget.should_warn()`; se `True`, chamar `ui.budget_warning(...)` (nova
  função em `ui.py`) e pedir confirmação via `input()` (mesmo padrão simples
  já usado em `ui.ask()`).
  - Se negado: `break` do loop interno sem chamar o LLM (turno abortado).
  - Se confirmado: `messages = [{"role": "system", "content": SYSTEM_PROMPT}]`,
    `budget.reset()`, `session.save(messages)` (grava o corte como uma nova
    sessão continuando no mesmo diretório — reaproveita `session.compacted`
    ou um helper equivalente), avisa na UI que o contexto foi limpo, e segue
    o loop normalmente a partir daí.

## Componente 3: subdiretório de trabalho por sessão (`workdir.py`)

### Motivação

Modelos pequenos vão estourar o budget de tokens com frequência. Para que
isso não signifique perder o diretório de trabalho (código/arquivos que o
agente já criou), cada sessão opera dentro de um subdiretório dedicado em
`sessions/<id>/` na raiz do fork. Ao estourar o budget e confirmar (ver
Componente 2), só a conversa é resetada — o subdiretório e seus arquivos
continuam. Uma nova execução do agente pode reaproveitar o mesmo
subdiretório (abrindo uma nova conversa nele, ou retomando uma antiga via
`/sessions`/`--resume` como já existe) ou criar um subdiretório novo.

### Problema técnico a resolver

Hoje `sandbox.py` (`PROJECT = Path.cwd().resolve()`) e `session.py`
(`SESSION_DIR` derivado de `Path.cwd()`) calculam o diretório do projeto
**no momento do import do módulo**. Isso precisa acontecer **depois** que o
agente decidiu e trocou (`os.chdir`) para o subdiretório da sessão, não
antes. A correção: tornar esses valores preguiçosos (computados na hora do
uso, não como constante de módulo), para que reflitam o `cwd` já
atualizado quando `wrap()`/`save()`/`load()` etc. rodam.

- `sandbox.py`: `PROJECT` e a string `PROFILE` (que depende de `PROJECT`)
  passam a ser calculados dentro de `wrap()` a cada chamada, em vez de
  constantes de módulo.
- `session.py`: `SESSION_DIR` vira uma função `_session_dir()` chamada onde
  hoje `SESSION_DIR` é usado (`save`, `path_for`, `all_sessions`).
  `PROJECT` (o slug do caminho) é computado dentro dela.

Nenhuma outra lógica dessas duas peças muda — isso é só destravar o
comportamento de import-time para call-time.

### `neuralcode/workdir.py` (novo)

- `SESSIONS_DIR_NAME = "sessions"`.
- `list_existing(root: Path) -> list[Path]`: subdiretórios diretos de
  `root/sessions/`, mais recentes primeiro (por mtime), cada um anotado com
  o título da sessão mais recente ali dentro (reaproveitando
  `session.title`/`session.all_sessions`, já que após o chdir esses módulos
  enxergam aquele diretório).
- `choose_or_create(root: Path) -> Path`: usa `ui.pick` (já existe, usado em
  `/sessions` e `/rewind`) para listar os subdiretórios existentes com uma
  opção extra "novo diretório de sessão" no topo. Escolher um existente
  retorna aquele `Path`; escolher "novo" cria
  `root/sessions/<YYYYMMDD-HHMMSS>/` (mesmo formato de timestamp que
  `session.CURRENT` já usa) e retorna o `Path` recém-criado.

### Mudanças em `agent.py`

- Logo no início de `main()`, antes de qualquer outra coisa que dependa do
  diretório de trabalho: `repo_root = Path.cwd()`, depois
  `session_dir = workdir.choose_or_create(repo_root)`, depois
  `os.chdir(session_dir)`.
- `sandbox.name()` no banner (`ui.banner`) já reflete corretamente o novo
  `cwd` sem mudança adicional, já que passa a computar `PROJECT` sob
  demanda.
- Escolher um subdiretório **existente** não reabre automaticamente a
  última conversa — inicia uma sessão nova nesse diretório (comportamento
  atual sem `--resume`). Para continuar uma conversa anterior naquele
  mesmo diretório, o usuário usa `/sessions` (lista as sessões daquele
  `cwd`, que é exatamente o subdiretório escolhido) ou `--resume`.

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
  não configurado, `reset()` zerando o estado.
- `test_workdir.py`: `list_existing` com diretório `sessions/` ausente
  (retorna vazio) e com subdiretórios presentes (ordem por mtime);
  `choose_or_create` criando um novo subdiretório com o formato de nome
  esperado (usando `tmp_path` do pytest, sem tocar no repositório real).

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
