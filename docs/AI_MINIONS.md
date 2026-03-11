# AI Sidecars

This repository now includes a unified sidecar dispatcher for delegated read-heavy work such as:

- implementation summarization
- second-pass code review
- design/spec reduction
- draft patch planning

The unified entrypoint is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1
```

Available providers:

- `gemini`
- `claude`
- `gpt`

All providers are read-only helpers by contract. Codex remains responsible for edits, verification, and final judgment.

## Actions

Health check across all providers:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action doctor
```

Provider-specific auth:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action auth -Provider gemini
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action auth -Provider claude
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action auth -Provider gpt
```

No-browser auth is supported for:

- `gemini` via `-NoBrowser`
- `gpt` via `-NoBrowser` which maps to Codex device auth

Example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action auth -Provider gpt -NoBrowser
```

Claude currently exposes browser login only through its CLI. If your Claude/Anthropic account uses Google SSO, complete that step in the browser during `claude auth login`.

## Delegate A Task

Analysis:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 `
  -Action delegate `
  -Provider gemini `
  -Mode analysis `
  -Task "Summarize the relevant implementation and risks for this task." `
  -ContextPath README.md,ROADMAP.md `
  -OutputFormat text
```

Review:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 `
  -Action delegate `
  -Provider gpt `
  -Mode review `
  -Task "Review these changes for bugs, regressions, and weak assumptions." `
  -ContextPath path/to/file1,path/to/file2 `
  -OutputFormat text
```

Claude with injected context:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 `
  -Action delegate `
  -Provider claude `
  -Mode analysis `
  -InjectContext `
  -Task "Summarize the relevant implementation and risks for this task." `
  -ContextPath AGENTS.md,scripts/ai-minion.ps1 `
  -OutputFormat text
```

The example includes `-InjectContext` to make the intent explicit. For Claude, the current wrapper still forces inline context whenever you provide `-ContextPath`.

## Output Artifacts

When you do not provide `-OutputPath`, artifacts are written under:

```text
.tool-cache/ai-minions/<provider>/
```

Each delegated run stores:

- the exact prompt as `*.prompt.txt`
- the final extracted answer as `*.txt` or `*.json`
- GPT raw event streams as `*.raw.jsonl`
- Claude transcript copies as `*.session.jsonl`

## Provider Notes

Gemini:

- The existing provider-specific wrapper remains `scripts/gemini-sidecar.ps1`.
- `scripts/ai-minion.ps1` forwards auth and normal delegated runs to that wrapper.
- Gemini-specific setup details remain in `docs/GEMINI_MINION.md`.

Claude:

- `claude auth status --json` is used for health checks.
- Claude inlines requested `-ContextPath` file contents into the delegated prompt because the wrapper disables Claude tools for read-only runs.
- If you omit `-InjectContext`, the wrapper still forces inline context for Claude and prints a note so the provider-specific behavior is visible.
- Passing `-InjectContext` to Claude is currently redundant, but it can make your command intent clearer when you want provider-agnostic examples.
- On this machine, `claude -p` sometimes exits with an empty stdout body even when the response succeeds.
- The wrapper works around that by assigning a session id and extracting the final assistant message from `~/.claude/projects/.../<session-id>.jsonl`.

GPT:

- GPT sidecar is implemented with the official OpenAI Codex CLI (`@openai/codex`).
- Browser auth uses ChatGPT sign-in. If your ChatGPT account uses Google login, that browser flow uses the same account path.
- `-NoBrowser` maps to Codex device auth.
- On this machine the Windows shim was missing, so the wrapper launches `node ...@openai/codex/bin/codex.js` directly.

## Reliability Tips

- Keep `-ContextPath` narrow. In practice, 1-5 focused files produce better results than broad directory-scale requests.
- If a delegated run times out or returns weak output, retry once with a smaller file set and a sharper task prompt before switching providers.
- Prefer `gemini` or `gpt` for routine delegated work.
- Use `claude` as an additional reviewer or fallback when you want a second opinion from a different model family.
