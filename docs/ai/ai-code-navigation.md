# AI Code Navigation

This repo prefers targeted navigation over broad raw file reads.

Primary tools when they are available in the current client or toolchain:

- `RepoMapper` for broad repository maps
- `jCodeMunch` for symbol-level search and retrieval
- `jDocMunch` for doc search and outlines

The repo no longer vendors RepoMapper or jCodeMunch source trees or tool-specific wrapper scripts. Use the interface your current client provides, such as MCP tools or external CLIs on `PATH`.

This checkout also provides MCP server definitions in `.mcp.json` for:

- `jcodemunch` via `uvx jcodemunch-mcp`
- `jdocmunch` via `uvx jdocmunch-mcp`

If `.tool-cache/repomapper` or `.jcodemunch-index/` exist, treat them as opportunistic local caches, not durable agent memory.

## Local helper

- `scripts/import-ai-nav.ps1` defines PowerShell aliases that forward to external `repomapper` and `jcodemunch` commands. Missing commands are reported when you invoke the alias.
- There are no guaranteed repo-local wrappers such as `scripts/repomap.ps1` or `scripts/jcodemunch.ps1` in the current checkout.

## When to use which

- Use `RepoMapper` when you do not yet know where to look and your current client or toolchain exposes it.
- Use `jCodeMunch` when you know a symbol name or want symbol-level retrieval.
- Use `jDocMunch` for doc-section search and outlines when it is available.
- Use `rg` first for the cheapest literal/text search when shell access is available.
- If `RepoMapper` is unavailable, fall back to `jCodeMunch` or `rg` instead of broad raw file reads.

## Recommended workflow

1. Start with a broad map if the area is unfamiliar and a repo-map tool is available.
2. If no broad map tool is available, use `jCodeMunch` or `rg` to narrow the area before reading files.
3. Read only the specific implementation or section you need.
4. Use targeted text search for strings, errors, comments, and other non-symbol text.

If your session does not expose MCP calls directly but `uvx` is installed, initialize the local indexes from the shell first:

```powershell
uvx jcodemunch-mcp index C:\Users\alexl\code\math\untitled --no-ai-summaries
uvx jdocmunch-mcp index-local --path C:\Users\alexl\code\math\untitled\docs --name untitled-docs
```

The current local repo index created from this checkout reports as `local/untitled-a84ca850`.

Those shell commands refresh the local indexes and caches declared by `.mcp.json`. If your session still does not expose query-time MCP tools after indexing, fall back to `rg` plus focused file reads rather than loading large files wholesale.

## PowerShell helpers

If you have external `repomapper` and `jcodemunch` commands on `PATH`, import the repo-local helper functions into the current shell:

```powershell
. .\scripts\import-ai-nav.ps1
```

This adds:

- `rmap` -> `repomapper` CLI
- `jcm` -> raw `jcodemunch` CLI
- `jfindsym` -> symbol search
- `jgetsym` -> symbol retrieval
- `jfindtxt` -> text search
- `jtree` -> file tree
- `joutline` -> file outline

Examples:

```powershell
jfindsym runConfig -Language scala
jgetsym "src/main/scala/sicfun/holdem/TexasHoldemPlayingHall.scala::runConfig#function" -Verify
jfindtxt "playing hall failed"
jtree
joutline build.sbt
```

## Measured results in this repo

These are historical measurements from March 7, 2026, when this repo still carried the local navigation toolchain:

- `RepoMapper` produced a useful repo map over 361 files in about 2022 tokens.
- `jCodeMunch` indexed 172 files with 2044 symbols.
- Indexed language counts: 150 Scala, 10 Java, 9 C++, 3 Python.
- `search-symbols runConfig --language scala` returned six candidates in about 820 tokens.
- `get-symbol "src/main/scala/sicfun/holdem/TexasHoldemPlayingHall.scala::runConfig#function"` returned the full function in about 2589 tokens.
- Reading the full `TexasHoldemPlayingHall.scala` file costs about 17379 tokens.
- `build.sbt` indexed as Scala and exposed task/settings symbols such as `headsUpTableMode`, `generateHeadsUpTable`, and `Compile / resourceGenerators`.
- Simple key searches such as `headsUpTableMode` resolved to the key declaration instead of returning both the declaration and the plain `:=` assignment.

The practical takeaway is:

- `RepoMapper` is the cheaper first pass for orientation.
- `jCodeMunch` is the cheaper second pass for exact symbol retrieval.

## Repo-specific notes

- Commit `8ced6f7` removed the vendored RepoMapper and jCodeMunch sources and their wrapper scripts.
- `scripts/import-ai-nav.ps1` is a thin convenience layer over externally provided `repomapper` and `jcodemunch` CLIs.
- `.mcp.json` is the preferred source of truth for MCP-backed navigation in this checkout.
- In shell-only sessions, prefer the `uvx jcodemunch-mcp ...` and `uvx jdocmunch-mcp ...` commands from `.mcp.json` over ad hoc tool discovery.
- Cache directories may exist from prior local runs, but treat them as stale until verified.
