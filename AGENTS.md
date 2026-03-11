# Codex Agent Instructions

## Gemini Minion Requirement

For any non-trivial task in this repository, use Gemini as a delegated sidecar before finalizing your answer or patch.

Default rule:
- Use [`scripts/gemini-sidecar.ps1`](scripts/gemini-sidecar.ps1) for repo exploration, second-pass review, benchmark/result summarization, and patch planning.
- Treat Gemini as a read-only helper. Codex remains responsible for edits, verification, and final judgment.
- If Gemini sidecar is unavailable or auth is broken, state that briefly and continue without blocking the task.

Suggested workflow:
1. Identify the task type and the smallest useful context files.
2. Run Gemini once in the matching mode:
   - `analysis` for exploration or summarization
   - `review` for bug/regression hunting
   - `draft-patch` for change planning
3. Read Gemini's output, verify the claims against the repository, then proceed with your own work.

Canonical commands:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gemini-sidecar.ps1 `
  -Action delegate `
  -Mode analysis `
  -Task "Summarize the relevant implementation and risks for this task." `
  -ContextPath README.md,ROADMAP.md `
  -OutputFormat text
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gemini-sidecar.ps1 `
  -Action delegate `
  -Mode review `
  -Task "Review these changes for bugs, regressions, and weak assumptions." `
  -ContextPath path/to/file1,path/to/file2 `
  -OutputFormat text
```

Notes:
- The sidecar writes prompts and outputs under `.tool-cache/gemini-sidecar/`.
- Prefer focused `-ContextPath` inputs over broad directory dumps.
- For trivial tasks like a single factual command or obvious one-line edit, Gemini is optional.
