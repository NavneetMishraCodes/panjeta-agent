# Panjeta Agent

> A personal agent built to understand commands, use tools, and perform local
> computer tasks. Run from the terminal, provider-agnostic, tool-calling
> agent loop on top of a clean LLM abstraction.

## 🚧 Status

**Early development, working foundation.** Panjeta currently provides:

* A provider-independent LLM abstraction (`BaseLLM`) with **two working
  providers**: OpenRouter (via the official `openai` SDK against OpenRouter's
  OpenAI-compatible API) and Google Gemini (via the official `google-genai`
  SDK).
* A generic tool-calling **Agent loop** that asks the model to plan, executes
  requested tool calls through a controlled registry, and feeds results back
  until the model gives a final answer.
* A local terminal entry point whose startup message is generated in plain
  Python — no LLM call, no tokens spent.
* A **controlled file-manager tool family** (`list_directory`, `read_file`,
  `create_file`, `copy_file`, `move_file`, `rename_file`, `delete_file`)
  operating strictly inside a configurable filesystem sandbox
  (`PANJETA_FILE_ROOT`).

**Not yet implemented** (documented goals only): persistent memory, WhatsApp
integration, OCR, PDF/image content extraction, embeddings/vector search,
browser automation, GUI/keyboard/mouse automation, and arbitrary computer
control.

## 🛠️ Tech Stack

* **Python 3.14** (project runs in a local `venv`).
* **`openai`** — official OpenAI SDK, used *only* because OpenRouter exposes
  an OpenAI-compatible API (not an OpenAI-the-company integration).
* **`google-genai`** — official Google Gemini SDK.
* **`python-dotenv`** — loads `.env` for local development.
* **Standard-library `unittest`** for all tests — no test framework
  dependency.
* No AI frameworks (no LangChain, LangGraph, LlamaIndex), no vector
  databases, no memory systems. Dependencies are added only when necessary.

## 🧠 Architecture

Plain-text local flow of `python -m src.main`:

```text
User instruction (typed locally)
        ↓
PANJETA AGENT banner / prompt   ← generated locally, no LLM, no tokens
        ↓
src/agent/agent.py  (Agent.run - tool-calling loop, max 8 iterations)
        ↓
src/llm/base.py  (BaseLLM / Message / ToolCall / ToolDefinition / LLMResponse)
        ↓
OpenRouterLLM  OR  GoogleLLM          (selected via create_llm)
        ↓
ToolRegistry  →  calculator | search_files | file-manager tools   (all tool execution)
        ↓
Final answer printed locally
```

### Layers

* **`src/llm/base.py`** — provider-independent types (`Message`,
  `ToolDefinition`, `ToolCall`, `LLMResponse`), the `BaseLLM` interface, and
  `LLMConfigError` (generic configuration failure shared by all providers).
* **`src/llm/openrouter.py`** — `OpenRouterLLM`: OpenAI-compatible adapter
  pointed at `https://openrouter.ai/api/v1`. Converts everything to/from the
  generic types inside the adapter.
* **`src/llm/google.py`** — `GoogleLLM`: official `google-genai` Gemini
  adapter. Converts Gemini function calls, contents, system instructions,
  and tool schemas to/from the generic types inside the adapter. Function
  responses are matched back to their calls by id (plus name).
* **`src/llm/factory.py`** — `create_llm("openrouter" | "google")` —
  selects a provider without the Agent knowing which one it is talking to.
* **`src/agent/agent.py`** — the `Agent` loop — provider-agnostic; it only
  sees `BaseLLM` and `ToolRegistry`.
* **`src/tools/`** — `Tool` registry (`Tool`/`ToolRegistry`) with `calculator`
  (safe expression evaluator, no `eval`, injection-proof), `search_files`
  (read-only filename/metadata search, contents never read), and the
  file-manager family in `src/tools/file_manager.py`, guarded by the
  sandbox in `src/tools/paths.py`. Tools never run shell commands.
* **`src/ui.py`** — minimal local terminal interface (banner, prompt,
  output). No LLM involvement.
* **`src/main.py`** — interactive entry point.

### Filesystem tools & safety boundary

The file-manager tools operate **only** inside the Panjeta filesystem root:

* Selected with the `PANJETA_FILE_ROOT` environment variable (absolute path
  recommended). When unset, a safe default of `<project folder>/panjeta_files`
  is used (created on first use).
* Every path is resolved against this root; `..` traversal is rejected
  outright, absolute paths are accepted only when they resolve inside the
  root, and symlink escapes are detected by canonicalising the deepest
  existing ancestor of the target.
* `read_file` reads plain text only (binary refused), truncated at 50 KB;
  `list_directory` returns at most 200 entries.
* `create_file` refuses to overwrite silently (`overwrite=true` required);
  copy/move refuse to clobber existing destinations; parents are never
  created automatically.
* `delete_file` is destructive and requires an explicit `confirm=true`
  argument; directories are refused (no recursive deletion).

Supported task examples (the LLM decides which tools to call — nothing is
hard-coded in `main.py`):

```text
List the files in my Panjeta folder.
Read notes/todo.txt.
Create notes/today.txt with today's task list.
Copy notes/a.txt to backup/a.txt.
Rename notes/old.txt to notes/new.txt.
Delete junk.txt.
```

### Provider selection

Priority: `--provider` flag → `PANJETA_LLM_PROVIDER` env var → default
`openrouter`.

```text
python -m src.main --provider google
python -m src.main --provider openrouter
```

## 🔑 Configuration

Copy `.env.example` to `.env` and fill in the provider you want to use:

```dotenv
OPENROUTER_API_KEY=        # https://openrouter.ai/keys
OPENROUTER_MODEL=

GOOGLE_API_KEY=              # https://aistudio.google.com/apikey
GOOGLE_MODEL=                # e.g. gemini-2.0-flash
```

Secrets are never printed. `.env` is gitignored.

Optional filesystem sandbox configuration:

```dotenv
# Every file-manager tool stays strictly inside this root.
# Default when unset: <project folder>/panjeta_files
PANJETA_FILE_ROOT=
```

Manual real-API smoke scripts (not part of the automated suite):

```text
venv\Scripts\python.exe -m scripts.smoke_agent
venv\Scripts\python.exe -m scripts.smoke_search <root> [query]
```

## 🧪 Tests

```text
venv\Scripts\python.exe -m unittest discover -s tests -v
```

**186 tests, all passing (offline, deterministic).** The providers are tested
with scripted fakes and the filesystem tools run inside temporary sandbox
directories — no API keys or network needed.

## ⚠️ Current Limitations

* Agent memory is per-run only; nothing persists between runs.
* Filesystem tools are sandboxed to `PANJETA_FILE_ROOT` by design; they
  handle **files only** — no directory creation/moving/deletion, no
  binary/PDF/image content parsing, no search over file contents.
* No approval UI yet: safety comes from explicit, validated tools and the
  `delete_file` `confirm=true` requirement.
* No web access, no WhatsApp, no OCR/PDF, no browser or GUI automation, and
  no arbitrary computer control.

## 🗺️ Direction

Panjeta is being built toward a general-purpose local computer agent — but
**this phase provides no arbitrary computer control**, only explicitly
registered, validated tools. Next candidates: directory-level file-manager
operations (create/move/delete directories, if justified), a real
approval/permission layer for destructive operations, and persistent session
memory — each behind the same `ToolRegistry`.
