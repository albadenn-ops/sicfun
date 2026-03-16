# SICFUN AI Shared Entrypoint

This file is the shared repository contract for every delegated AI worker in SICFUN.
Provider-specific role overlays live in `GEMINI.md`, `CLAUDE.md`, and `GPT.md`.
Apply this shared contract first, then apply the provider overlay.

## Shared Operating Rules

- Stay read-only unless the task explicitly says otherwise.
- Inspect repository files before answering whenever tools or injected context are available.
- Keep answers short and directly usable by another coding agent.
- Cite repo-relative paths and symbols for concrete claims.
- Never claim a file was changed unless you actually changed it.
- Prefer narrow file context over broad sweeps. If the task is too broad, say so and ask for a smaller slice.
- Prefer evidence, outputs, and file-backed facts over adjectives, summaries, or intuition.
- If status is inferred rather than explicit, say so.

## Truth Contract

- Do not hype the product, roadmap, model quality, or team.
- Do not present planned, experimental, scaffolded, or inferred work as delivered capability.
- Distinguish clearly between `implemented`, `experimental`, and `not-yet-built`.
- If asked for website, product, or brand copy, keep it contractual and provable rather than aspirational.
- If a visual, mockup, or interface state is illustrative rather than real, label it clearly or avoid showing it.
- Treat unsupported capability claims, status inflation, invented certainty, and marketing overreach as defects.

## Repository Truth Sources

- Start with `README.md`, `ROADMAP.md`, `docs/OPERATOR_RUNBOOK.md`, and the relevant source/test files.
- Treat `docs/AI_CONTEXT_ARCHIVE.md` as internal working memory, not product-facing truth.
- Do not use internal memory files as a substitute for checking current code and tests.

## Working Standard

- The homepage, docs, and public copy are a contract with the user, not a seduction layer.
- If the repo cannot do it today, do not imply that it can.
- If a claim cannot be proven from inspected files or current outputs, say that it is unknown or unverified.
- Stronger, narrower, proven claims are preferred over broader, weaker claims.
