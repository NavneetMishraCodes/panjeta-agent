# Panjeta Agent

> A personal agent built to understand commands, use tools, and perform local
> computer tasks. Run from the terminal, provider-agnostic, tool-calling
> agent loop on top of a clean LLM abstraction.

## 🚧 Status

**Early development, working foundation.** The core principle holds throughout:
the **LLM is only the planner** — it requests actions — while Panjeta itself
validates, approves, executes, observes, and remembers the work. Panjeta
currently provides:

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
  `create_file`, `copy_file`, `move_file`, `rename_file`, `delete_file`,
  `create_directory`, `delete_directory`, `move_directory`) operating
  strictly inside a configurable filesystem sandbox (`PANJETA_FILE_ROOT`).
* A **real human-approval layer**: destructive tools (the two deletions, and
  `create_file` with `overwrite=true`) pause and ask the actual user at the
  console — `Delete 'notes/old.txt'? [y/N]` — and default to DENY. LLM text is
  never treated as approval.
* **Persistent session state**: the interactive conversation is stored in a
  bounded, versioned JSON file and restored on the next launch. History is
  capped by both message count and encoded size, corrupted or hand-edited
  files degrade to a clean new session, and trimming can never leave a tool
  result orphaned from its assistant tool call (pairing is repaired on
  load and on every persisted turn).
* **Hardened runtime (Phase 9)**: the filesystem sandbox cannot be escaped
  through `..`, absolute paths, the root's own ancestors, or symlink/junction
  redirects; provider failures (SDK errors, malformed payloads, unusable tool
  calls, empty answers) are reported as one clear local error and the
  interactive session stays alive. All of it is covered by the offline suite,
  including an end-to-end harness.

**Not yet implemented** (documented goals only): long-term semantic memory
(Panjeta restores recent conversation context, not human-like memory),
WhatsApp integration, OCR, PDF/image content extraction, embeddings/vector
search, browser automation, GUI/keyboard/mouse automation, recursive
directory deletion, and arbitrary computer control.

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
SessionStore.load  (data/session.json)   ← optional, offline, gitignored
        ↓
src/agent/agent.py  (Agent.run - tool-calling loop, max 8 iterations)
        ↓
src/llm/base.py  (BaseLLM / Message / ToolCall / ToolDefinition / LLMResponse)
        ↓
OpenRouterLLM  OR  GoogleLLM          (selected via create_llm)
        ↓
ToolRegistry → file-manager | calculator | search_files   (all tool execution)
        |            destructive tools pass the human-approval gate first
        ↓
Final answer printed locally
        ↓
SessionStore.save  (bounded history restored on the next launch)
```

### Layers

* **`src/llm/base.py`** — provider-independent types (`Message`,
  `ToolDefinition`, `ToolCall`, `LLMResponse`), the `BaseLLM` interface, and
  the provider-independent failures all providers share: `LLMConfigError`
  (bad configuration), `LLMRequestError` (an SDK/transport failure or a
  malformed provider payload), and `LLMToolCallError` (a tool call that
  cannot be normalized). Providers raise these at their own boundary and
  scrub credentials from the text.
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
  sees `BaseLLM` and `ToolRegistry`. A provider failure (SDK/transport
  error, malformed response, unnormalizable tool call, or an empty answer)
  is converted at this single boundary into one `AgentLLMError`, so the
  interactive loop can report a clear error and keep running without the
  core agent knowing which provider is configured. Nothing is persisted for
  a failed turn.
* **`src/tools/`** — `Tool` registry (`Tool`/`ToolRegistry`) with `calculator`
  (safe expression evaluator, no `eval`, injection-proof), `search_files`
  (read-only filename/metadata search, contents never read), and the
  file-manager family in `src/tools/file_manager.py`, guarded by the
  sandbox in `src/tools/paths.py`. Tools never run shell commands.
* **`src/approval.py`** — the human-approval layer: the `Approver`
  protocol, the console implementation (`ConsoleApprover`, explicit
  `y`/`yes` only, default deny), and `ApprovalDeniedError`. The
  `ToolRegistry` consults the approver before executing any tool marked
  `requires_approval`. LLM providers and the Agent's reasoning are never
  involved in approval decisions.
* **`src/session.py`** — `SessionStore`: a versioned, bounded JSON session
  file (default `<project>/data/session.json`, gitignored;
  `PANJETA_SESSION_FILE` overrides the location). Saving is atomic; a
  missing file starts a new session, and malformed/incompatible files are
  logged and replaced rather than crashing Panjeta. This is conversational
  context only — **not** semantic memory.
* **`src/ui.py`** — minimal local terminal interface (banner, prompt,
  output). No LLM involvement.
* **`src/main.py`** — interactive entry point.

### Filesystem tools & safety boundary

The file-manager tools (files **and** directories) operate **only** inside
the Panjeta filesystem root:

* Selected with the `PANJETA_FILE_ROOT` environment variable (absolute path
  recommended). When unset, a safe default of `<project folder>/panjeta_files`
  is used (created on first use).
* Every path is resolved against this root; `..` traversal is rejected
  outright, absolute paths are accepted only when they resolve **inside** the
  root, and the containment test never runs the other way round: the root's
  own parent and grandparent (and anything else above the root) are rejected
  for reads, listings, writes, and destinations alike.
* Symlink **and** Windows junction escapes are caught by canonicalising the
  existing prefix of the target and requiring the final resolved path to stay
  inside the canonical root, so a link that points out of the sandbox is
  refused even for a not-yet-existing destination. Directory walks never
  descend into links (no cycles, no escapes).
* `read_file` reads plain text only (binary refused), truncated at 50 KB.
  `list_directory` returns at most 200 entries; an optional `recursive=true`
  mode is depth-capped at 3 levels, entry-capped at 200, and never follows
  symlinks (default is non-recursive).
* `create_file` refuses to overwrite silently (`overwrite=true` required);
  `overwrite=true` is destructive, so it goes through the same human-approval
  gate as deletions (`Overwrite file 'notes/old.txt'? [y/N]`, default deny).
  Copy/move refuse to clobber existing destinations; `create_directory`
  creates only the requested directory (never a parent tree);
  `move_directory` relocates a whole directory and refuses to move it into
  itself or its own subtree.
* **Destructive operations** (`delete_file`, `delete_directory`, and
  `create_file` with `overwrite=true`) require real human approval at the
  console: Panjeta prints e.g. `Delete 'notes/old.txt'? [y/N]` and only an
  explicit `y`/`yes` proceeds — anything else, including blank input, denies
  and nothing is touched. The two deletions additionally require the
  tool-level `confirm=true` argument, so the model has to state its intent.
  `delete_directory` deletes EMPTY directories only; recursive deletion does
  not exist in Panjeta.

Supported task examples (the LLM decides which tools to call — nothing is
hard-coded in `main.py`):

```text
List the files in my Panjeta folder.
Read notes/todo.txt.
Create notes/today.txt with today's task list.
Copy notes/a.txt to backup/a.txt.
Rename notes/old.txt to notes/new.txt.
Create a folder called homework.
Delete junk.txt.          ← Panjeta asks you [y/N] before deleting
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

Optional session file location (conversation persistence):

```dotenv
# Default when unset: <project folder>/data/session.json (gitignored)
PANJETA_SESSION_FILE=
```

Start with a clean conversation instead of restoring the previous one:

```text
python -m src.main --fresh-session        # or the alias: --fresh
```

Session controls (purely local commands — no provider, no API key needed):

```text
python -m src.main session status   # file path, format version, message count
python -m src.main session reset    # delete the stored session file
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

**351 tests, all passing offline and deterministically** (2 skipped, see
below). The providers are tested with scripted fakes, the filesystem tools
run inside temporary sandbox directories, approval is tested with scripted
approvers and stdin, and session persistence is tested against temporary JSON
files — no API keys or network needed.

The suite includes a permanent offline **end-to-end harness**
(`tests/test_e2e.py`): a `ScriptedLLM` drives the real `Agent` → real
`ToolRegistry` → real tools → a real temporary filesystem, covering the
calculator, `search_files`, the whole file-manager family (create, read, copy,
move, rename, delete, directory create/delete/move), an approved deletion, a
denied deletion, provider-failure handling, and sandbox-boundary rejection.
It verifies integration rather than replacing the unit tests.

Sandbox-escape regressions live in `tests/test_paths.py` (path resolution) and
`tests/test_file_manager.py` (`SandboxEscapeRegressionTest`, which also asserts
that nothing outside the root was created, read, or changed).

The two skips are the file-symlink tests on a machine whose platform does not
grant the symlink privilege (Windows reports `WinError 1314`). Directory
escape tests still run, using a junction fallback, and the skip is raised only
for that genuine capability limit (a dedicated `LinkCapabilityError`), so an
actual link-helper bug fails the suite instead of silently skipping.

## ⚠️ Current Limitations

* The session restores **recent conversation context only** (bounded at 200
  messages and 200,000 encoded bytes; the oldest are dropped, and a trimmed
  window is repaired so tool calls/results always stay paired). This is
  persistent session state —
  *not* long-term semantic memory and *not* human-like memory. No
  summarization or embeddings exist.
* Filesystem tools are sandboxed to `PANJETA_FILE_ROOT` by design; there is
  no binary/PDF/image content parsing and no search over file contents.
* `search_files` is intentionally **not** sandboxed: finding files the user
  points it at anywhere on the computer is its purpose. It is read-only
  (names and metadata only, links never followed) and bounded to
  `MAX_SCANNED_ENTRIES` (50,000) entries per search, and the result says when
  the scan stopped early. A search rooted at a very large tree can still take
  noticeable time; a depth-limited walk/cancellation is a Phase 10
  reliability item rather than a Phase 9 change.
* `delete_directory` deletes empty directories only; recursive deletion
  does not exist anywhere in Panjeta.
* Approval happens at the local console only (no remote/GUI approval), and
  there is no per-tool permission configuration yet.
* No web access, no WhatsApp, no OCR/PDF, no browser or GUI automation, and
  no arbitrary computer control.

## 🗺️ Direction

Panjeta is being built toward a general-purpose local computer agent — but
**this project provides no arbitrary computer control**, only explicitly
registered, validated tools behind the sandbox, the human-approval layer,
and the registry. The foundation (sandbox + approval + sessions +
`ToolRegistry`) is now in place for broader capabilities next: content-aware
features (PDFs, images, WhatsApp-style documents), richer tool families,
and configurable permissions — each added as a separate registered tool.
