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

**Not yet implemented** (documented goals only): persistent memory, WhatsApp
integration, OCR, PDF/image content extraction, embeddings/vector search,
browser automation, and the full computer/file-manager tool family.

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
ToolRegistry  →  calculator | search_files   (all tool execution)
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
  (safe expression evaluator, no `eval`, injection-proof) and `search_files`
  (read-only filename/metadata search, contents never read).
* **`src/ui.py`** — minimal local terminal interface (banner, prompt,
  output). No LLM involvement.
* **`src/main.py`** — interactive entry point.

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

Manual real-API smoke scripts (not part of the automated suite):

```text
venv\Scripts\python.exe -m scripts.smoke_agent
venv\Scripts\python.exe -m scripts.smoke_search <root> [query]
```

## 🧪 Tests

```text
venv\Scripts\python.exe -m unittest discover -s tests -v
```

**107 tests, all passing (offline, deterministic).** The Google and OpenRouter
providers are tested with scripted fakes — no API keys or network needed.

## ⚠️ Current Limitations

* Agent memory is per-run only; nothing persists between runs.
* Tools are limited to `calculator` and `search_files`; no content reading
  (PDF/OCR/image), web access, or file modification yet.
* No approvals/permission UI — safety comes from the small set of explicit,
  validated tools.

## 🗺️ Direction

Panjeta is being built toward a general-purpose local computer agent. The
next major tool family is file-manager functionality (list/read/create/copy/
move/rename/delete) as independent registered tools — and eventually
content-aware features (PDFs, images, WhatsApp-style documents). The current
architecture keeps each of those additions a separate registered tool behind
the same `ToolRegistry`.
