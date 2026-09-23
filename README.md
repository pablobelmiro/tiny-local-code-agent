# tiny-local-code-agent

A minimal terminal coding agent, extended to run against small local models
(2B-14B) on CPU through [Ollama](https://ollama.com) or any OpenAI-compatible
endpoint. This is a personal learning project: a portfolio for experimenting
with how a coding agent works under the hood, and how far it holds up when
the model behind it is small and running on a laptop CPU instead of a large
hosted model.

## Origin

This project started as a fork of
[`avbiswas/neural-code`](https://github.com/avbiswas/neural-code), a minimal
coding agent harness built to teach how the pieces of a coding agent fit
together (tool loop, permissions, sandboxing, context compaction, sessions).
Full credit for the original design and implementation goes to that project —
[the accompanying walkthrough video](https://youtu.be/Lu1UWqVTbQg) is a great
starting point for understanding the base agent this one builds on.

## What's added on top

- **Model profiles** (`models.yaml` + `PROFILE` env var) — named presets for
  small CPU-friendly models (`small-2b`, `medium-8b`, `large-14b`), each
  bundling a model name, a conservative context window, and a token budget.
- **A hard per-session token budget** — once usage crosses the configured
  budget, the agent asks whether to clear the conversation and keep going
  (rather than silently growing forever or hard-crashing).
- **A working directory per session** (`sessions/<timestamp>/`) — chosen
  interactively at startup, so a budget-triggered context reset clears the
  conversation but keeps every file the agent has created or edited in that
  directory. You can resume an existing session directory or start a fresh
  one each run.

## Requirements

- Python 3.10+
- An OpenAI-compatible endpoint. For local CPU use, that's
  [Ollama](https://ollama.com) (`ollama serve`, or a container running it).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Usage

Point the agent at your local Ollama (or any OpenAI-compatible endpoint) and
pick a model profile:

```bash
export BASE_URL=http://localhost:11434/v1
export API_KEY=ollama          # Ollama ignores the value, but the client needs one set
export PROFILE=small-2b        # or medium-8b / large-14b, from models.yaml

# make sure the model is pulled in Ollama first, e.g.:
#   ollama pull qwen2.5-coder:1.5b

.venv/bin/neuralcode
# equivalently: .venv/bin/python -m neuralcode.agent
```

On startup you'll be asked to pick a session working directory — either an
existing one under `sessions/`, or a new one. The agent's tools (bash, file
read/write/edit) all operate inside that directory.

Useful env vars, all optional beyond `BASE_URL`/`API_KEY`:

| Variable | Purpose |
|---|---|
| `PROFILE` | Named profile from `models.yaml` (`small-2b`, `medium-8b`, `large-14b`) |
| `MODEL` | Overrides the profile's model |
| `CONTEXT_WINDOW` | Overrides the profile's context window |
| `TOKEN_BUDGET` | Overrides the profile's token budget (hard per-session limit) |

Slash commands available in the chat: `/sessions` (reopen a past chat in the
current working directory), `/rewind` (go back in the conversation),
`/compact` (manually summarize a long conversation), `/reload` (re-read
`MODEL`/`BASE_URL`/`PROFILE`/etc. from the environment and rebuild the LLM
client, without restarting or losing the conversation - handy for switching
models mid-session; export the new values first, then run `/reload`).

## Design notes

See `docs/superpowers/specs/` and `docs/superpowers/plans/` for the design
spec and implementation plan behind the profile/budget/workdir extensions
added on top of the original `neural-code` base.
