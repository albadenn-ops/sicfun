# Codex Agent Instructions

## Linked Worker Requirement

For any non-trivial task in this repository, use one delegated sidecar before finalizing your answer or patch.

## Code Navigation

Prefer repo-local navigation tools before broad raw file reads, especially to avoid RAM-heavy exploration:

- Prefer the MCP servers declared in `.mcp.json` when they are usable in the current client or shell.
- If your current client or toolchain exposes `repomapper`, use it for cheap broad orientation when you do not yet know where to look.
- If your current client or toolchain exposes `jcodemunch`, use it for file trees, symbol lookup, targeted text search, and focused retrieval.
- If your current client or toolchain exposes `jdocmunch`, use it for docs; otherwise use focused doc reads/searches instead of loading large documents wholesale.
- Use `rg` first for cheap literal search when shell access is available and you are looking for plain text rather than broad repo orientation.
- Prefer targeted retrieval over reading entire large files or large directory sweeps when the narrower tool can answer the question.
- Do not assume tool-specific repo-local wrapper scripts exist in the current checkout. `scripts/ai/import-ai-nav.ps1` is only a PowerShell convenience layer for CLI-based sessions when `repomapper` and `jcodemunch` are installed on `PATH`.
- If `.jcodemunch-index/` or `.tool-cache/repomapper/` exist, treat them as local caches only; do not assume they are present, fresh, or shared across chats.
- If a native MCP client is not exposed but `uvx` is available, use the `.mcp.json` commands directly from the shell:
  - `uvx jcodemunch-mcp index <repo-root> --no-ai-summaries`
  - `uvx jdocmunch-mcp index-local --path docs --name <repo-id>`

See [`docs/ai/ai-code-navigation.md`](docs/ai/ai-code-navigation.md) for workflow guidance and repo-specific notes.

Default rule:
- Use [`scripts/ai/ai-minion.ps1`](scripts/ai/ai-minion.ps1) for second-pass review, benchmark/result summarization, and patch planning after local navigation has narrowed the scope.
- Available linked workers are `gemini`, `claude`, and `gpt`.
- Treat every provider as a read-only helper. Codex remains responsible for edits, verification, and final judgment.
- Prefer `claude` for planning, scope control, contradiction hunting, and fact-checking when available.
- Prefer `gpt` for implementation-oriented breakdowns, execution planning after scope is set, and code-aware review.
- Use `gemini` for bounded repo exploration, extraction, summarization, repetitive support work, and explicit execution tasks. Treat Gemini as low-trust for inference.
- If the chosen provider is unavailable or auth is broken, state that briefly, retry once with fewer files and a sharper task, then switch providers instead of blocking.
- If all providers are unavailable, proceed with local analysis and state that no sidecar review was obtained.

Suggested workflow:
1. Inspect the repo slice yourself first and identify the smallest useful context files.
2. Run `scripts/ai/ai-minion.ps1` in the matching mode on a narrow context:
   - `analysis` for exploration or summarization
   - `review` for bug/regression hunting
   - `draft-patch` for change planning
3. If the first provider times out or returns shallow output, retry once with fewer files and a more specific task instead of broadening context.
4. Read the sidecar output, verify the claims against the repository, then proceed with your own work.
5. For substantial edits, prefer a second narrow `review` pass on the touched files before finalizing, potentially from a different provider.

Canonical commands:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai/ai-minion.ps1 `
  -Action delegate `
  -Provider gemini `
  -Mode analysis `
  -Task "Summarize the relevant implementation and risks for this task." `
  -ContextPath README.md,ROADMAP.md `
  -OutputFormat text
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai/ai-minion.ps1 `
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
- `scripts/ai/gemini-sidecar.ps1` still exists for Gemini-specific auth/setup details; the unified dispatcher forwards normal Gemini flows to it.
- Claude inlines requested `-ContextPath` file contents for delegated runs, even without `-InjectContext`, because its delegated path runs without file-reading tools. Passing `-InjectContext` to Claude is currently redundant but still useful for making that intent explicit in the command line. The dispatcher prints a note when it has to force that behavior.
- Prefer focused `-ContextPath` inputs over broad directory dumps.
- Best signal usually comes from 1-5 files. Large context sets are more likely to time out or produce vague summaries.
- For large tasks, chain multiple narrow delegate calls by module or concern instead of one oversized request.
- Treat sidecar output as a second set of eyes. Do not let it replace local code reading, tests, or final judgment.
- For trivial tasks like a single factual command or obvious one-line edit, delegated sidecars are optional.

## Project Structure and Conventions

This repository is a Scala 3.8.1 project using sbt as the build tool.

- Source code: `src/main/scala/`
- Test code: `src/test/scala/`
- Scripts: `scripts/` (PowerShell scripts for various tasks)
- Documentation: `docs/`
- Generated data: `data/` (recreate as needed, not committed)
- Build configuration: `build.sbt`

Key build commands:
- Compile: `sbt compile`
- Test: `sbt test`
- Run: `sbt run`

Scala compiler options (from `build.sbt`):
- `-deprecation`, `-feature`, `-unchecked`
- `-Wunused:imports,privates,locals`
- `-Werror` (warnings as errors)
- `-new-syntax`, `-indent`

Test framework: MUnit (`org.scalameta %% "munit" % "1.2.2" % Test)

Native code: `src/main/native/` with C++ sources built via clang++. Tracked DLLs for performant computation (built via `sbt nativeBuild`):
- CPU: `sicfun_native_cpu.dll`, `sicfun_cfr_native.dll`, `sicfun_bayes_native.dll`, `sicfun_ddre_native.dll`, `sicfun_postflop_native.dll`, `sicfun_pomcp_native.dll`
- CUDA: `sicfun_bayes_cuda.dll`, `sicfun_cfr_cuda.dll`, `sicfun_ddre_cuda.dll`, `sicfun_postflop_cuda.dll`, `sicfun_gpu_kernel.dll`
- OpenCL: `sicfun_opencl_kernel.dll`

Resource generation: sbt tasks like `generateHeadsUpTable` for equity tables.

For web UI development, see `docs/site-preview-hybrid/` and scripts in `scripts/packaged-hand-history-web/`.

Reference `build.sbt` for detailed settings and tasks.
