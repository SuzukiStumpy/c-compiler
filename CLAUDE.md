# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A homegrown C compiler written in Python supporting a subset of the C language. `compiler/compiler.py` is the main entry point.

## Environment

- Python 3.13, managed with `uv`
- The `.venv` is auto-activated in VS Code (`launch_venv.sh`)
- Install deps: `uv sync`

## Commands

```bash
# Run tests
uv run pytest

# Run a single test file
uv run pytest test/test_foo.py

# Run a single test
uv run pytest test/test_foo.py::test_name

# Lint (check)
uv run ruff check .

# Lint (auto-fix)
uv run ruff check --fix .

# Type check
uv run pyrefly check

# Verify build
uv run test
```

## Code style

- Ruff enforces: `ANN` (type annotations), `E/F/W` (pycodestyle/pyflakes), `C` (convention), `I` (isort)
- Line length: 120
- All functions must have type annotations (ANN rules are enabled)
- All modules, types and functions should have docstrings attached explaining their usage.  In addition, functions should have parameters, return values and exceptions explained in the docstring.  Use Google-style formatting.

## Architecture

Tests go in `test/`. Pytest is configured to look there and ignores the `compiler/` source directory for test discovery (`norecursedirs` + `addopts = "--ignore=compiler"`).

The `compiler` package is installed in editable mode into `.venv` via the `c_compiler.pth` path file, so `import compiler` works without reinstalling after edits.

## Agent skills

### Issue tracker

Issues live in GitHub Issues (`SuzukiStumpy/c-compiler`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout — `UBIQUITOUS_LANGUAGE.md` (vocabulary) and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
