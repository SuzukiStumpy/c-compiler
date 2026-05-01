# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`UBIQUITOUS_LANGUAGE.md`** at the repo root — the canonical vocabulary for this project. Use the terms defined here; avoid the aliases listed in the "Aliases to avoid" column.
- **`docs/adr/`** — read ADRs that touch the area you're about to work in.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront.

## File structure

Single-context repo:

```
/
├── UBIQUITOUS_LANGUAGE.md
├── docs/adr/
│   └── 0001-*.md
└── compiler/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `UBIQUITOUS_LANGUAGE.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap worth adding.

## Maintaining UBIQUITOUS_LANGUAGE.md

When a new domain term is resolved or an ambiguity is clarified in conversation, update `UBIQUITOUS_LANGUAGE.md` in the same PR/commit. Add it to the appropriate table with a definition and any aliases to avoid.

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders) — but worth reopening because…_
