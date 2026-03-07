# AI Code Navigation

This repo has two repo-local code navigation tools:

- `RepoMapper` for broad repository maps
- `jCodeMunch` for symbol-level search and retrieval

Both are wired through local wrappers and vendored source trees.

RepoMapper cache data is written under `.tool-cache/repomapper`.
The local jCodeMunch index is written under `.jcodemunch-index`.

## Local entry points

- `scripts/repomap.ps1`
- `scripts/jcodemunch.ps1`

## Vendored sources

- `third_party/repomapper`
- `third_party/jcodemunch_mcp`

## When to use which

- Use `RepoMapper` when you do not yet know where to look.
- Use `jCodeMunch` when you know a symbol name or want symbol-level retrieval.
- Use `rg` first for the cheapest literal/text search when shell access is available.

## Recommended workflow

1. Start with a broad map if the area is unfamiliar.
2. Narrow with `file-tree` or `search-symbols`.
3. Read only the specific implementation with `get-symbol`.
4. Use `search-text` for strings, errors, comments, and other non-symbol text.

## Useful commands

```powershell
.\scripts\repomap.ps1 --other-files src project scripts build.sbt README.md ROADMAP.md --chat-files build.sbt README.md --mentioned-files src/main/scala/sicfun/holdem/TexasHoldemPlayingHall.scala --mentioned-idents TexasHoldemPlayingHall --map-tokens 2048 --force-refresh

.\scripts\jcodemunch.ps1 list-repos

.\scripts\jcodemunch.ps1 file-tree --repo local/untitled --path-prefix src/main/scala/sicfun/holdem

.\scripts\jcodemunch.ps1 search-symbols runConfig --repo local/untitled --language scala --max-results 10

.\scripts\jcodemunch.ps1 search-text "playing hall failed" --repo local/untitled --file-pattern "*.scala" --max-results 5

.\scripts\jcodemunch.ps1 get-symbol "src/main/scala/sicfun/holdem/TexasHoldemPlayingHall.scala::runConfig#function" --repo local/untitled --verify
```

## Measured results in this repo

Measured on March 7, 2026 after local Scala support and Windows byte-offset fixes were applied to the vendored `jCodeMunch`.

- `RepoMapper` produced a useful repo map over 361 files in about 2022 tokens.
- `jCodeMunch` now indexes 171 files with 2023 symbols.
- Indexed language counts: 149 Scala, 10 Java, 9 C++, 3 Python.
- `search-symbols runConfig --language scala` returned six candidates in about 820 tokens.
- `get-symbol "src/main/scala/sicfun/holdem/TexasHoldemPlayingHall.scala::runConfig#function"` returned the full function in about 2589 tokens.
- Reading the full `TexasHoldemPlayingHall.scala` file costs about 17379 tokens.

The practical takeaway is:

- `RepoMapper` is the cheaper first pass for orientation.
- `jCodeMunch` is the cheaper second pass for exact symbol retrieval.

## Repo-specific notes

- The vendored `jCodeMunch` includes local patches for Scala support.
- The vendored `jCodeMunch` also preserves raw file newlines so symbol byte offsets verify correctly on Windows.
- `build.sbt` is discoverable but currently yields no symbols in the local `jCodeMunch` index.
- `RepoMapper` is vendored with its `queries/` directory because the packaged install omitted those files.
