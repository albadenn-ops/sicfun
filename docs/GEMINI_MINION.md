# Gemini Sidecar

This repository now includes an optional Gemini CLI sidecar for delegated read-heavy work such as:

- benchmark/result summarization
- second-pass code review
- design/spec reduction
- draft patch planning

The default path is read-only delegation. The sidecar is meant to reduce prompt load on the primary coding agent, not replace final verification.

If you want the unified multi-provider front door for Gemini, Claude, and GPT/Codex, use `scripts/ai-minion.ps1` and see `docs/AI_MINIONS.md`.

## One-Time Setup

Gemini CLI is installed globally through npm:

```powershell
npm install -g @google/gemini-cli
```

Start Google login from the repository:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gemini-sidecar.ps1 -Action auth
```

If Gemini hits the browser consent bug or the browser does not open, the wrapper now retries manual auth automatically. You can still force manual auth yourself:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gemini-sidecar.ps1 -Action auth -NoBrowser
```

Notes:

- The wrapper boots the CLI from its Node entrypoint instead of the generated Windows shim. On this machine, the shim tries to invoke `-S` directly and fails.
- If `~/.gemini/settings.json` does not exist yet, the wrapper creates a minimal file with `security.auth.selectedType = "oauth-personal"` before launching the interactive login flow.
- Headless delegation requires a configured auth type, so run the auth step once before `-Action delegate`.
- Browser mode still asks `Do you want to continue? [Y/n]:` before Gemini attempts launch. If you do not answer `Y`, nothing will open.
- If browser auth exits with Gemini's consent error (`exit code 41`), the wrapper retries with `NO_BROWSER=true`.
- `-NoBrowser` sets `NO_BROWSER=true` so Gemini prints the auth URL and asks for the returned authorization code directly in the terminal.

## Health Check

Inspect local setup:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gemini-sidecar.ps1 -Action doctor
```

This reports:

- `node` and `npm` discovery
- Gemini CLI entrypoint and version
- `~/.gemini/settings.json`
- current selected auth type
- whether the shared `AI_ENTRYPOINT.md` contract is present
- whether the repo-level `GEMINI.md` overlay is present

## Delegate A Task

Read-only analysis:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gemini-sidecar.ps1 `
  -Action delegate `
  -Mode analysis `
  -Task "Summarize the latest exact-mode hall benchmark changes." `
  -ContextPath docs/OPERATOR_RUNBOOK.md,ROADMAP.md `
  -OutputFormat text
```

Code review pass:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gemini-sidecar.ps1 `
  -Action delegate `
  -Mode review `
  -Task "Review the hall max runner changes for regressions or unsafe assumptions." `
  -ContextPath scripts/run-playing-hall-max.ps1,src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala `
  -OutputFormat json
```

Draft patch planning only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gemini-sidecar.ps1 `
  -Action delegate `
  -Mode draft-patch `
  -Task "Propose the smallest patch to make the hall runner emit a more useful failure summary." `
  -ContextPath scripts/run-playing-hall-max.ps1 `
  -OutputFormat text
```

## Output Artifacts

When you do not provide `-OutputPath`, the wrapper writes artifacts under:

```text
.tool-cache/gemini-sidecar/
```

For each delegate run it saves:

- the exact prompt sent to Gemini as `*.prompt.txt`
- the raw Gemini response as `*.txt`, `*.json`, or `*.jsonl`

`.tool-cache/` is already ignored by git.

## Usage Notes

- Delegated runs are read-only by contract through the prompt, the shared `AI_ENTRYPOINT.md` contract, and the `GEMINI.md` role overlay, not by Gemini CLI approval mode.
- `AI_ENTRYPOINT.md` carries the common SICFUN sidecar rules. `GEMINI.md` adds the Gemini-specific low-trust support role.
- If you need Gemini to inspect paths outside the repository root, pass them through `-IncludeDirectories`.
- Gemini CLI may emit status lines such as `Loaded cached credentials.`; the wrapper preserves successful runs instead of treating that stderr noise as failure.

## Reliability Tips

- Keep `-ContextPath` narrow. In practice, 1-5 focused files produce better output than broad directory-scale requests.
- If a delegate run times out or returns weak output, retry with a smaller file set and a sharper task prompt before giving up.
- For larger investigations, split the work into multiple passes such as:
  - `analysis` on the small context needed to understand the area
  - `review` on the exact files you touched or the exact diff under scrutiny
- Use Gemini as a bounded support worker and extractor, not as the source of final judgment. The primary agent should still inspect code, run tests, and decide what lands.
