# Codex Agent Instructions

## Linked Worker Requirement

For any non-trivial task in this repository, use one delegated sidecar before finalizing your answer or patch.

Default rule:
- Use [`scripts/ai-minion.ps1`](scripts/ai-minion.ps1) for repo exploration, second-pass review, benchmark/result summarization, and patch planning.
- Available linked workers are `gemini`, `claude`, and `gpt`.
- Treat every provider as a read-only helper. Codex remains responsible for edits, verification, and final judgment.
- Prefer `claude` for planning, scope control, contradiction hunting, and fact-checking when available.
- Prefer `gpt` for implementation-oriented breakdowns, execution planning after scope is set, and code-aware review.
- Use `gemini` for bounded repo exploration, extraction, summarization, repetitive support work, and explicit execution tasks. Treat Gemini as low-trust for inference.
- If the chosen provider is unavailable or auth is broken, state that briefly, retry once with fewer files and a sharper task, then switch providers instead of blocking.
- If all providers are unavailable, proceed with local analysis and state that no sidecar review was obtained.

Suggested workflow:
1. Inspect the repo slice yourself first and identify the smallest useful context files.
2. Run `scripts/ai-minion.ps1` in the matching mode on a narrow context:
   - `analysis` for exploration or summarization
   - `review` for bug/regression hunting
   - `draft-patch` for change planning
3. If the first provider times out or returns shallow output, retry once with fewer files and a more specific task instead of broadening context.
4. Read the sidecar output, verify the claims against the repository, then proceed with your own work.
5. For substantial edits, prefer a second narrow `review` pass on the touched files before finalizing, potentially from a different provider.

Canonical commands:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 `
  -Action delegate `
  -Provider gemini `
  -Mode analysis `
  -Task "Summarize the relevant implementation and risks for this task." `
  -ContextPath README.md,ROADMAP.md `
  -OutputFormat text
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 `
  -Action delegate `
  -Provider gpt `
  -Mode review `
  -Task "Review these changes for bugs, regressions, and weak assumptions." `
  -ContextPath path/to/file1,path/to/file2 `
  -OutputFormat text
```

Notes:
- The unified dispatcher writes prompts and outputs under `.tool-cache/ai-minions/<provider>/`.
- Shared sidecar rules live in `AI_ENTRYPOINT.md`. Provider-specific role overlays live in `GEMINI.md`, `CLAUDE.md`, and `GPT.md`.
- `scripts/gemini-sidecar.ps1` still exists for Gemini-specific auth/setup details; the unified dispatcher forwards normal Gemini flows to it.
- Claude inlines requested `-ContextPath` file contents for delegated runs, even without `-InjectContext`, because its delegated path runs without file-reading tools. Passing `-InjectContext` to Claude is currently redundant but still useful for making that intent explicit in the command line. The dispatcher prints a note when it has to force that behavior.
- Prefer focused `-ContextPath` inputs over broad directory dumps.
- Best signal usually comes from 1-5 files. Large context sets are more likely to time out or produce vague summaries.
- For large tasks, chain multiple narrow delegate calls by module or concern instead of one oversized request.
- Treat sidecar output as a second set of eyes. Do not let it replace local code reading, tests, or final judgment.
- For trivial tasks like a single factual command or obvious one-line edit, delegated sidecars are optional.
