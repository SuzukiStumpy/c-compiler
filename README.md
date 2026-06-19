# c-compiler

A homegrown C compiler written in Python, supporting a substantial subset of the C11 language. The compiler takes a `.c` source file through the full pipeline from preprocessing to a native x86-64 executable.

## Supported C subset

### Types

| Type | Notes |
|------|-------|
| `int` | 32-bit signed integer |
| `long` / `long long` | 64-bit signed integer (both are 8 bytes on x86-64 Linux) |
| `unsigned int` | 32-bit unsigned integer |
| `unsigned long` / `unsigned long long` | 64-bit unsigned integer (both are 8 bytes on x86-64 Linux) |
| `short` / `signed short` / `unsigned short` | 16-bit integer; promotes to `int` for arithmetic (C11 §6.3.1.1) |
| `char` / `signed char` / `unsigned char` | 8-bit character types |
| `__int128` / `unsigned __int128` | GCC 128-bit extension; static variables and zero-initialisation supported; see note below |
| `float` | 32-bit IEEE 754 floating-point; `f`/`F` suffix on literals |
| `double` | 64-bit IEEE 754 floating-point |
| `long double` | 80-bit x87 extended precision; `L`/`l` suffix on literals; passed/returned via stack (MEMORY class) |
| `void` | Return type and pointer target |
| Pointers (`T *`) | Pointer arithmetic; null pointer constants |
| Arrays (`T[N]`) | Fixed-size arrays; compound initialisers; designated initialisers (`{[n] = val}`); array decay |
| `struct` | Named/anonymous structs; nested structs; anonymous struct/union members (field promotion, C11 §6.7.2.1); compound initialisers; designated initialisers (`{.field = val}`); bit-fields (`int x : 3`) |
| `union` | Named/anonymous unions; anonymous struct/union members (field promotion); bit-field members |
| `enum` | Enumeration types; enumerator constants are `int`-typed compile-time values |
| `_Bool` | 1-byte unsigned integer; any nonzero value converts to 1 (C11 §6.3.1.2); `true`/`false` via `<stdbool.h>` |
| Function pointers | `int (*fp)(int, int)` style; call via `fp(args)` or `(*fp)(args)`; pass/return by value |

Type specifiers `signed` and `unsigned` may be combined with `int`, `long`, `long long`, `short`, and `char` in all standard permutations. The `LL`/`ll` suffix produces a `long long` constant; `ULL`/`LLU`/`ull`/`llu` produces an `unsigned long long` constant; `f`/`F` produces a `float` constant; `L`/`l` produces a `long double` constant.

Integer literals are accepted in all four C bases:

| Prefix | Base | Example |
|--------|------|---------|
| *(none)* | Decimal | `255` |
| `0` | Octal | `0377` |
| `0x` / `0X` | Hexadecimal | `0xFF` |
| `0b` / `0B` | Binary (GCC extension) | `0b11111111` |

All type suffixes (`u`, `U`, `l`, `L`, `ll`, `LL` and combinations) apply to every base.

> **`__int128` note** — The type is fully recognised through the pipeline (lexer → parser → semantic analysis → TAC → assembly emission). Static variables with constant initialisers are emitted as two consecutive `.quad` directives (little-endian lo/hi). Local-variable arithmetic, multi-word instruction sequences (`adcq`/`sbbq`), and the SysV AMD64 two-register calling convention for 128-bit values are not yet implemented; any expression-level use of `__int128` beyond zero-initialisation will produce incorrect or incomplete code.

### Expressions

- Integer, floating-point, and character constants
- String literals (decay to `char *`; embedded in read-only data)
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Bitwise: `~`, `&`, `|`, `^`, `<<`, `>>`
- Logical (short-circuit): `!`, `&&`, `||`
- Relational: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Prefix and postfix `++` / `--`
- Conditional (ternary): `cond ? then : else`
- Assignment `=` and compound assignment `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=`
- Explicit type casts
- `sizeof` on expressions and on type names
- Address-of `&` and pointer dereference `*`
- Array subscript `arr[i]`
- Struct and union member access: `.` and `->`
- Function calls (direct and through function pointers)

Usual arithmetic conversions and integer promotions follow C11 sec. 6.3.1.

### Statements

- `return`
- Expression statements
- `if` / `else`
- `while`, `do`/`while`
- `for` (declaration or expression initialisers)
- `switch` with `case` and `default` labels
- `break` and `continue`
- `goto` and labeled statements
- Compound blocks `{ ... }` with nested variable scoping

### Declarations

- Local and global variable declarations with optional initialisers
- Compound initialisers for arrays and structs (`{ ... }`), including designated initialisers (`{.field = val}` for structs/unions, `{[n] = val}` for arrays)
- Function declarations and definitions (multiple per translation unit)
- Variadic function declarations (`int printf(char *fmt, ...)`) — calling variadic functions is fully supported; System V AMD64 ABI `%al` register is set to the number of floating-point arguments used
- Struct and union declarations and definitions, including bit-fields (`int x : 3;`, `unsigned int : 5;`, `int : 0;`)
- `enum` declarations and definitions; enumerator constants fold to `int` values at semantic analysis time; enumerator values may be integer literals, unary-negated literals, or references to previously-declared enumerators in the same definition
- Storage classes: `static` and `extern`
- `typedef` declarations with scope-aware shadowing; typedef names resolve to concrete types at parse time
- Type qualifiers: `const`, `volatile`, `restrict` — see below for enforcement details
- Function specifiers: `inline` — parsed and stored on the function declaration; no inlining is performed; `_Noreturn` — parsed and stored on the function declaration; no contract enforcement is performed
- `_Static_assert(expr, msg)` at file scope and block scope — constant expression evaluated at compile time; a zero result is a compile-time error with the supplied message; no runtime code is emitted

### Semantic analysis

The semantic analysis phase performs full name resolution and type checking:

- Unique renaming of every declared identifier
- C11 usual arithmetic conversions and pointer-compatibility rules
- Linkage resolution for `static` and `extern` identifiers
- `switch` validation: integer-only controlling expressions; no duplicate `case` values; no floating-point case labels
- Variadic call sites: extra arguments beyond the declared parameters are type-checked individually; `va_list`/`va_start`/`va_arg`/`va_end` are not yet supported (Tier 2)
- Type qualifier enforcement: assignment to `const`-qualified lvalues raises `SemanticError`; `restrict` on non-pointer types raises `SemanticError`

### Type qualifiers

All three C11 type qualifiers (`const`, `volatile`, `restrict`) are fully supported.
Qualifiers may appear in any position where C11 allows them: before or after a type specifier, and after a `*` in a pointer declarator.
Qualified types are preserved in the AST as `QualifiedType` wrapper nodes and enforced throughout the pipeline.

| Qualifier | C11 reference | Compiler behaviour |
|---|---|---|
| `const` | sec. 6.7.3 | Assignment to a `const`-qualified lvalue (variable, pointer-to-const dereference) is a `SemanticError`. `const` global variables are placed in `.rodata`. |
| `volatile` | sec. 6.7.3 | Reads and writes to `volatile`-qualified variables are never eliminated or reordered by the copy-propagation or dead-store-elimination optimisation passes. |
| `restrict` | sec. 6.7.3 | Applying `restrict` to a non-pointer type is a `SemanticError`. The aliasing hint is tracked in the type system; no alias-based optimisations are performed beyond what the standard aliasing rules already permit. |

Standard library headers that use these qualifiers (e.g. `const char *` in `<string.h>`) parse and type-check without error.

## Prerequisites

- **Python 3.13** - managed via [`uv`](https://github.com/astral-sh/uv)
- **`ruff`** - style linter (installed via `uv sync`)
- **`pyrefly`** - type checker (installed via `uv sync`)
- **`gcc`** or **`clang`** (with a `gcc` shim) - used for preprocessing and final assembly/linking; must be on `PATH`

The toolchain targets Linux and WSL. macOS is partially supported (function-name underscore prefixes are applied automatically).

## Dev installation

Clone the repository, then from the project root:

```bash
uv sync
```

This creates `.venv`, installs all dependencies (including dev tools), and registers the `compiler` entry point in editable mode so changes to `compiler/` take effect immediately without reinstalling.

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
  -c                            Don't link; produce an object file.
  -S                            Don't assemble or link; produce the assembly file.
  -I DIR                        Add DIR to the preprocessor include search path (repeatable).
  -D NAME[=VALUE]               Define a preprocessor macro (repeatable).
  -U NAME                       Undefine a preprocessor macro (repeatable).
  -l NAME                       Link library NAME (repeatable).
  --lex                         Stop after lexing.
  --parse                       Stop after parsing; print the AST.
  --validate                    Stop after semantic analysis; print the resolved AST.
  --tacky                       Stop after TAC generation; print the TAC representation.
  --codegen                     Stop after code generation; emit assembly but do not link.
  --fold-constants              Optimisation: constant folding.
  --propagate-copies            Optimisation: copy propagation.
  --eliminate-unreachable-code  Optimisation: unreachable code elimination.
  --eliminate-dead-stores       Optimisation: dead store elimination.
  --optimize                    Enable all optimisations.
  --help                        Show this message and exit.
```

The stage flags are mutually exclusive. `-c`, `-S`, `-l`, `-I`, `-D`, and `-U` may be combined freely.

### Examples

```bash
# Compile to an executable
compiler hello.c

# Link against the math library
compiler -lm hello.c

# Add an include directory and define a macro
compiler -I/usr/local/include -DNDEBUG hello.c

# Undefine a macro
compiler -UDEBUG hello.c

# Compile to an object file only
compiler -c hello.c

# Produce an assembly file only
compiler -S hello.c

# Inspect the token stream
compiler --lex hello.c

# Inspect the AST after parsing
compiler --parse hello.c

# Inspect the AST after semantic analysis
compiler --validate hello.c

# Inspect the TAC intermediate representation
compiler --tacky hello.c

# Emit assembly without linking
compiler --codegen hello.c

# Compile with all optimisations enabled
compiler --optimize hello.c
```

## Pipeline

The compiler runs the following stages in sequence:

| Stage | CLI flag | Description |
|-------|----------|-------------|
| Preprocessing | *(automatic)* | Runs `gcc -E` to expand macros; produces a `.i` file |
| Lexing | `--lex` | Converts the preprocessed source into a token stream |
| Parsing | `--parse` | Converts tokens into an abstract syntax tree (AST) |
| Semantic analysis | `--validate` | Resolves names, checks types, validates control flow |
| Optimisation | *(optional flags)* | TAC-level passes: constant folding, copy propagation, dead store elimination, unreachable code elimination |
| TAC generation | `--tacky` | Flattens the AST into three-address-code (TAC) |
| Code generation | `--codegen` | Translates TAC into an x86-64 assembly AST; emits a `.s` file |
| Register allocation | *(automatic)* | Graph-colouring allocation for GP and FP registers; Briggs conservative coalescing |
| Assembly and linking | *(default)* | Passes the `.s` file to `gcc`; produces an object file (`-c`), assembly file (`-S`), or native executable |

## Optimisations

All optimisation passes operate on the TAC representation and run in a fixed-point loop until no further changes occur.

| Flag | Description |
|------|-------------|
| `--fold-constants` | Evaluates constant expressions at compile time |
| `--propagate-copies` | Replaces uses of copy-assigned temporaries with their source values |
| `--eliminate-unreachable-code` | Removes basic blocks with no predecessors in the control-flow graph |
| `--eliminate-dead-stores` | Removes assignments whose values are never subsequently read |
| `--optimize` | Enables all four passes above |

Register allocation runs unconditionally at the assembly level, using graph colouring with Briggs conservative coalescing to minimise spills for both the general-purpose and floating-point register banks.

## What is currently missing (C11)

The sections below describe C11 features that are not yet implemented. References are to ISO/IEC 9899:2011.

### Integer types

`int`, `long`, `long long`, `unsigned int`, `unsigned long`, `unsigned long long`, `short`, `unsigned short`, and `char` (all three signed/plain/unsigned variants) are fully supported. `__int128` / `unsigned __int128` (a GCC extension) are partially supported — see the type table note above.

### Floating-point types

| Missing type | Notes |
|---|---|
| `_Complex` / `_Imaginary` | Complex and imaginary arithmetic (sec. 6.2.5 p11) |

`float` (32-bit), `double` (64-bit), and `long double` (80-bit x87 extended precision) are all supported. All arithmetic, relational, and conversion operations between `float`, `double`, `long double`, and integer types are implemented, including the SysV AMD64 ABI calling conventions (XMM registers for `float`/`double`; stack (MEMORY class) for `long double`).

### Atomic types and thread-local storage

C11 introduced a concurrency model alongside `<stdatomic.h>`:

- **`_Atomic`** type qualifier and type specifier (sec. 6.7.2.4) - lock-free atomic reads, writes, and RMW operations.
- **`_Thread_local`** storage-class specifier (sec. 6.7.1) - per-thread static storage.

Neither is recognised.

### Alignment

- **`_Alignas`** specifier (sec. 6.7.5) - over-align a declaration.
- **`_Alignof`** operator (sec. 6.5.3.4) - query alignment of a type.

### Declarations and specifiers

| Missing feature | C11 reference | Notes |
|---|---|---|
| `_Generic` selection | sec. 6.5.1.1 | Type-generic macro building block used by `<tgmath.h>`. |

### Initialisers

- **Compound literals** (sec. 6.5.2.5) - `(struct Point){1, 2}` as an rvalue expression. The cast-like syntax is not parsed.

### Enum limitations

`enum` is supported at both file scope and block scope, including:

- **Inline definitions** — `enum Color { RED } x;` declares the enum and a variable in one statement.
- **Anonymous enums** — `enum { A, B };` (no tag) registers enumerators without requiring a tag name.

The following are not yet implemented:

- **Full constant-expression enumerator values** — enumerator values must be integer literals, unary-negated literals, or references to a previously-declared enumerator in the same definition. Expressions involving `sizeof`, bitwise shifts, or cross-enum references are not evaluated.

### Struct and union features

- **Flexible array members** (sec. 6.7.2.1 p18) - `struct S { int n; char buf[]; };`. The trailing incomplete-array member pattern is not parsed.

### Variadic functions

`...` in a parameter list (sec. 6.7.6), `<stdarg.h>` (`va_list`, `va_start`, `va_arg`, `va_end`, `va_copy`) and the `__builtin_va_*` intrinsics are all absent. Variadic functions cannot be declared or called.

### Variable-length arrays

VLAs (sec. 6.7.6.2) - arrays whose size is a runtime expression - were mandatory in C99, made optional in C11 (Annex J.2), and removed from C17. This compiler does not implement them; all array sizes must be compile-time integer constant expressions.

### Multi-file compilation

The compiler accepts exactly one source file per invocation (`nargs=1`). There is no support for:

- Compiling multiple `.c` files in a single invocation

`-I`, `-D`, and `-U` are supported and forwarded directly to the GCC preprocessor. Separate compilation of individual files followed by linking with a system linker does work, because each `.c` file is compiled to a standard ELF object file.

### Standard library

No standard library is bundled. External libraries can be linked via the `-l` flag, but only if they were compiled with a compatible ABI. `typedef` names in preprocessed headers (e.g. `size_t`, `ptrdiff_t`, `uint32_t`) are now recognised, though other missing features (variadic functions) may still prevent some headers from parsing cleanly.

---

## Development

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest test/semantic_analysis/test_type_checking.py

# Run a single test by name
uv run pytest test/semantic_analysis/test_type_checking.py::TestDoubleOperatorRestrictions

# Lint (check)
uv run ruff check .

# Lint (auto-fix)
uv run ruff check --fix .

# Type check
uv run pyrefly check
```
