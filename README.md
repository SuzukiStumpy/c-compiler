# c-compiler

A homegrown C compiler written in Python, supporting a growing subset of the C language. The compiler takes a `.c` source file through the full pipeline from preprocessing to a native executable.

## Supported C subset

The compiler currently handles:

- `int`-typed functions with a `return` statement
- Integer constants and arithmetic: `+`, `-`, `*`, `/`, `%`
- Bitwise operators: `~`, `&`, `|`, `^`, `<<`, `>>`
- Logical operators with short-circuit evaluation: `!`, `&&`, `||`
- Relational operators: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Unary negation (`-`) and bitwise complement (`~`)
- Prefix increment and decrement: `++x`, `--x`
- Postfix increment and decrement: `x++`, `x--`
- Conditional (ternary) expressions: `<cond> ? <then> : <else>`
- Local variable declarations (`int x;`, `int x = <expr>;`)
- Variable assignment (`x = <expr>`)
- Compound assignment: `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=`
- Expression statements (expressions evaluated for side effects)
- `if` / `else` statements
- `goto` and labeled statements
- Compound (block) statements with nested variable scoping (`{ ... }`)

## Prerequisites

- **Python 3.13** — managed via [`uv`](https://github.com/astral-sh/uv)
- **`ruff`** — enforced by the pre-commit hooks (installed via `uv sync`)
- **`pyrefly`** — type checker (installed via `uv sync`)
- **`gcc`** or **`clang`** (with a `gcc` shim) — used for preprocessing and final assembly/linking; must be available on `PATH`

The toolchain is designed for Linux or WSL environments.

## Dev installation

Clone the repository, then from a terminal in the project root:

```bash
uv sync
```

This creates a virtual environment under `.venv`, installs all dependencies (including dev tools), and registers the `compiler` entry point in editable mode so changes to `compiler/` take effect immediately without reinstalling.

VS Code users: the `.venv` is auto-activated via `launch_venv.sh`.

## Tool installation

To install the compiler for use outside the development environment:

```bash
uv tool install .
```

## Usage

```
Usage: compiler [OPTIONS] SOURCE

  Compile SOURCE from C to a native executable.

  Pass a stage flag to stop the pipeline early and inspect intermediate
  output. Without a flag the full pipeline runs and an executable is produced.

Options:
  --lex       Stop after lexing.
  --parse     Stop after parsing; print the AST.
  --validate  Stop after semantic analysis; print the resolved AST.
  --tacky     Stop after TAC generation; print the TAC intermediate representation.
  --codegen   Stop after code generation; emit assembly but do not link.
  --help      Show this message and exit.
```

### Example

```bash
# Compile a C source file to an executable
compiler hello.c

# Inspect the token stream only
compiler --lex hello.c

# Inspect the abstract syntax tree
compiler --parse hello.c

# Inspect the AST after variable resolution
compiler --validate hello.c

# Inspect the three-address code intermediate representation
compiler --tacky hello.c

# Generate assembly but stop before linking
compiler --codegen hello.c
```

## Pipeline

The compiler runs six stages in sequence:

| Stage | CLI flag | Description |
| ----- | -------- | ----------- |
| Preprocessing | *(automatic)* | Runs `gcc -E` to expand macros; produces a `.i` file |
| Lexing | `--lex` | Converts the preprocessed file into a token stream |
| Parsing | `--parse` | Converts the token stream into an abstract syntax tree (AST) |
| Semantic analysis | `--validate` | Verifies variable declarations and resolves all names to unique identifiers |
| TAC generation | `--tacky` | Flattens the AST into a three-address-code (TAC) intermediate representation |
| Code generation | `--codegen` | Translates TAC into an x86-64 assembly AST and emits a `.s` file |
| Assembly and linking | *(default)* | Passes the `.s` file to `gcc` to produce a native executable |

## Development

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest test/test_semantic_analysis.py

# Lint (check only)
uv run ruff check .

# Lint (auto-fix)
uv run ruff check --fix .

# Type check
uv run pyrefly check
```
