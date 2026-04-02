# AI Code Navigation

This repo prefers targeted navigation over broad raw file reads.

Primary tools when they are available in the current client or toolchain:

- `RepoMapper` for broad repository maps
- `jCodeMunch` for symbol-level search and retrieval
- `jDocMunch` for doc search and outlines

The repo no longer vendors RepoMapper or jCodeMunch source trees or tool-specific wrapper scripts. Use the interface your current client provides, such as MCP tools or external CLIs on `PATH`.

If `.tool-cache/repomapper` or `.jcodemunch-index/` exist, treat them as opportunistic local caches, not durable agent memory.

## Local helper

- `scripts/import-ai-nav.ps1` defines PowerShell aliases that forward to external `repomapper` and `jcodemunch` commands. Missing commands are reported when you invoke the alias.
- There are no guaranteed repo-local wrappers such as `scripts/repomap.ps1` or `scripts/jcodemunch.ps1` in the current checkout.

## When to use which

- Use `RepoMapper` when you do not yet know where to look.
- Use `jCodeMunch` when you know a symbol name or want symbol-level retrieval.
- Use `jDocMunch` for doc-section search and outlines when it is available.
- Use `rg` first for the cheapest literal/text search when shell access is available.

## Recommended workflow

1. Start with a broad map if the area is unfamiliar.
2. Narrow with `file-tree`, `search-symbols`, or an equivalent targeted query.
3. Read only the specific implementation or section you need.
4. Use targeted text search for strings, errors, comments, and other non-symbol text.

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
- Cache directories may exist from prior local runs, but treat them as stale until verified.
