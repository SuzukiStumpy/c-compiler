# Architecture

Python-based C compiler targeting x86-64 Linux (SysV ABI). Implements a subset of C11/C17.

## Pipeline

Preprocessing (GCC) → Lex → Parse → Semantic Analysis → TAC Generation → [Optimisation] → Code Generation → Emit → Assemble/Link

Each stage is gated by a `Stage` bitmask flag. Passing `--lex`/`--parse`/`--validate`/`--tacky`/`--codegen` stops the pipeline early and prints the corresponding intermediate representation.

Preprocessor passthrough flags (`-I`, `-D`, `-U`) are forwarded directly to the GCC `-E` invocation; no pipeline stages are affected.

```
compiler/compiler.py      CLI (click) + pipeline orchestrator
compiler/subprocesses.py  GCC preprocessing + assembler/linker invocation
compiler/lexer.py         Preprocessed .i file → token stream
compiler/parser.py        Token stream → AST (Program)
compiler/semantic_analysis.py  AST → resolved AST + symbol table + struct/enum tables
compiler/tac_generator.py Resolved AST → TAC program
compiler/optimiser/       TAC-level optimisation passes (fixed-point loop)
compiler/code_generator.py TAC program → assembly AST
compiler/emitter.py       Assembly AST → .s file
```

## Shared inter-stage data

Three tables flow through the pipeline as out-of-band state (not embedded in tree nodes):

| Name | Type alias | Defined in | Purpose |
|---|---|---|---|
| `symbol_table` | `SymbolTable` | `compiler/types/symbols.py` | Maps resolved identifier → type + attrs (LocalAttr / StaticAttr / FunAttr / ConstantAttr / EnumConstAttr) |
| `struct_type_table` | `StructTypeTable` | `compiler/types/symbols.py` | Maps struct/union tag → member layout (offsets, sizes, alignment, bitfields) |
| `enum_type_table` | `EnumTypeTable` | `compiler/types/symbols.py` | Maps enum tag → enumerator name list |

`resolve_with_symbols` (semantic analysis) builds all three. Downstream stages receive them as explicit parameters.

## Syntax node packages

All node types are immutable frozen dataclasses. Sum types use ABC base classes.

### `compiler/syntax/ast/` — C AST

| Submodule | Contents |
|---|---|
| `types.py` | `Type` hierarchy: `IntType`, `LongType`, `UIntType`, `ULongType`, `ShortType`, `UShortType`, `Char`, `SChar`, `UChar`, `BoolType`, `FloatType`, `DoubleType`, `LongDoubleType`, `Int128Type`, `UInt128Type`, `LongLongType`, `ULongLongType`, `VoidType`, `FunType`, `Pointer`, `Array`, `Structure`, `Union`, `EnumType`, `QualifiedType`; `Array.size: int | None` where `None` denotes an incomplete/flexible array |
| `expressions.py` | `Exp` hierarchy: `Constant`, `String`, `Var`, `Cast`, `Unary`, `Binary`, `Assignment`, `CompoundAssignment`, `Conditional`, `FunctionCall`, `Dereference`, `AddrOf`, `Subscript`, `SizeOf`, `SizeOfT`, `AlignOf`, `AlignOfT`, `Dot`, `Arrow`, `CompoundLiteral`, `Initializer`, `SingleInit`, `CompoundInit`, `MemberDesignatorInit`, `IndexDesignatorInit`, `VaStart`, `VaArg`, `VaEnd`, `VaCopy` |
| `statements.py` | `Statement` hierarchy: `Return`, `Expression`, `If`, `Goto`, `Label`, `Compound`, `Break`, `Continue`, `While`, `DoWhile`, `For`, `Switch`, `Case`, `Default`, `Null` |
| `declarations.py` | `Declaration` hierarchy + `Program`, `Function`, `Block`, `BlockItem`, `VariableDeclaration`, `FunDecl`, `VarDecl`, `StructDecl`, `UnionDecl`, `EnumDecl`, `TypedefDecl`, `StaticAssert`, `Parameter`; `MemberDeclaration` carries optional `nested_decl: StructDecl | UnionDecl | None` for anonymous struct/union members; `EnumeratorDeclaration.value_exp: Exp | None` (constant expression or `None` for auto-increment) |
| `constants.py` | `Const` hierarchy: `ConstInt`, `ConstLong`, `ConstUInt`, `ConstULong`, `ConstDouble`, `ConstFloat`, `ConstLongDouble`, `ConstChar`, `ConstUChar`, `ConstShort`, `ConstUShort`, `ConstLongLong`, `ConstULongLong`, `ConstInt128`, `ConstUInt128`, `ConstBool` |
| `operators.py` | `UnaryOperator` and `BinaryOperator` enums |
| `pretty_print.py` | `pretty_print(program)` — renders any AST/TAC/ASM node tree as text |

### `compiler/syntax/tac/` — Three-Address Code

| Submodule | Contents |
|---|---|
| `instructions.py` | TAC instruction nodes (including `VaStart` for variadic register-save-area setup) |
| `values.py` | `Val` variants (`Constant`, `Var`) |
| `operators.py` | TAC operator enums |
| `declarations.py` | `Program`, `Function`, `StaticVariable`, `StaticConstant` |

### `compiler/syntax/asm/` — Assembly AST

| Submodule | Contents |
|---|---|
| `instructions.py` | Assembly instruction nodes (`Mov`, `Binary`, `Unary`, `Idiv`, `Div`, `Cdq`, `Cmp`, `Jmp`, `JmpCC`, `SetCC`, `Label`, `Ret`, `Call`, `IndirectCall`, `Lea`, `Push`, `Pop`, `Movsx`, `MoveZeroExtend`, `AllocateStack`, `DeallocateStack`, `Cvttsd2si`, `Cvtsi2sd`, `Cvttss2si`, `Cvtsi2ss`, `Cvtss2sd`, `Cvtsd2ss`) |
| `operands.py` | `Operand` hierarchy: `Imm`, `Register`, `Pseudo`, `Memory`, `Indexed`, `Data`, `PseudoMem`; `GPRegisters`/`FPRegisters` enums; `Width` enum; `ALLOCATABLE_GP`/`ALLOCATABLE_FP` |
| `values.py` | `AssemblyType` variants: `Byte`, `Longword`, `Quadword`, `Double` |
| `operators.py` | Assembly operator enums |
| `declarations.py` | `Program`, `Function`, `StaticVariable`, `StaticConstant` |

## Optimiser (`compiler/optimiser/`)

### TAC-level passes (`tac_main.py`)

Applied per-function in a fixed-point loop until stable:

1. **Constant folding** (`constant_folding.py`) — fold constant expressions at TAC level
2. **Unreachable code elimination** (`unreachable_code_elimination.py`) — remove dead basic blocks
3. **Copy propagation** (`copy_propagation.py`) — forward copy assignments
4. **Dead store elimination** (`dead_store_elimination.py`) — remove unused writes

All passes operate on a CFG (`cfg.py`). `worklist.py` provides a generic worklist algorithm.

### Assembly-level: register allocation (`asm_main.py`)

Two independent passes (GP registers, FP/XMM registers), each running:

1. Build interference graph via backward liveness analysis (CFG-based dataflow)
2. Conservative coalescing loop (Briggs + George tests) — merge copy-related pseudos
3. Graph colouring (Chaitin-style simplification/select, caller-saved preferred)
4. Spill handling — uncoloured pseudos remain as `Pseudo` operands for the fixup pass in `code_generator.py`

`interference_graph.py` — undirected graph with spill costs and colour map.
`disjoint_set.py` — union-find for coalescing merges (`compiler/types/disjoint_set.py`).

## Types sub-package (`compiler/types/`)

| File | Contents |
|---|---|
| `stages.py` | `Stage` bitmask enum (`LEX`, `PARSE`, `VALIDATE`, `TACKY`, `CODEGEN`, `LINK`, `ALL`) |
| `tokens.py` | `TokenType` enum (all lexer token kinds) |
| `symbols.py` | `SymbolTableEntry`, `SymbolAttrs` variants, `StaticInit` variants, `InitialValue` variants, `MemberEntry`, `StructEntry`, `EnumEntry`, type aliases `SymbolTable` / `StructTypeTable` / `EnumTypeTable` |
| `optimizations.py` | `Optimizations` dataclass (four bool flags) |
| `disjoint_set.py` | Generic union-find used by register allocator |

## Supported C subset

- **Integer types**: `char`, `signed char`, `unsigned char`, `short`, `unsigned short`, `int`, `unsigned int`, `long`, `unsigned long`, `long long`, `unsigned long long`, `__int128`, `unsigned __int128`, `_Bool`
- **Floating-point**: `float`, `double`, `long double` (x87 80-bit, stored as 16 bytes on Linux)
- **Derived types**: pointers, arrays (fixed-size), structs, unions (including anonymous struct/union members with field promotion, C11 §6.7.2.1), enums (named, anonymous `enum { … }`, inline `enum E { … } var`), function pointers, bitfields
- **Type qualifiers**: `const`, `volatile`, `restrict` (parsed; restrict has no semantic effect)
- **Function specifiers**: `inline` (parsed; stored as `Function.is_inline`; no inlining performed — flag is informational only); `_Noreturn` (parsed; stored as `Function.is_noreturn`; no contract enforcement — flag is informational only)
- **Static assertions**: `_Static_assert(expr, msg)` at file scope and block scope; constant expression is evaluated during semantic analysis; a zero result raises a `SemanticError` with the supplied message; no runtime code is emitted
- **Storage classes**: `static`, `extern`, `typedef` (`auto` and `register` are not yet recognised)
- **Control flow**: `if`/`else`, `while`, `do`/`while`, `for`, `switch`/`case`/`default`, `break`, `continue`, `goto`, labels
- **Operators**: full arithmetic, bitwise, logical, comparison, pre/post increment/decrement, address-of, dereference, subscript, `sizeof`, `_Alignof`, member access (`.` and `->`)
- **Variadic functions**: `va_list`, `va_start`, `va_arg`, `va_end`, `va_copy` via `<stdarg.h>`; SysV AMD64 ABI register-save-area (176 bytes, 16-byte aligned) and overflow argument area; GP and XMM register paths in `va_arg`
- **Alignment specifiers**: `_Alignas(expr)` and `_Alignas(type)` on variable declarations; SA evaluates to `VariableDeclaration.align_as: int`; overrides `.balign` for static variables and stack-slot alignment for locals; `_Alignof(type)` / `_Alignof expr` returns `unsigned long` alignment at compile time
- **Initializers**: scalar and compound (aggregate) with `static` constant evaluation; designated initialisers (`{.member = val}` for structs/unions, `{[n] = val}` for arrays)
- **String literals** (stored as hidden static objects in `.rodata`)

## Target

x86-64 Linux, SysV ABI. GCC is invoked for preprocessing and final assembly/linking.

## Test structure

```
test/
├── conftest.py                      # shared fixtures
├── fixtures/                        # sample C source files
├── samples/                         # additional sample sources
├── test_lexer.py                    # unit tests: lexer
├── test_lexer_error.py
├── test_compiler_driver.py          # integration: full pipeline
├── test_code_generator.py
├── test_emitter.py
├── test_pretty_print.py
├── test_asm_nodes.py
├── test_optimiser.py
├── test_subprocesses.py
├── parser/                          # parser tests
├── semantic_analysis/               # SA tests (variables, types, structs, etc.)
│   └── helpers.py
├── tac_generator/                   # TAC generation tests
│   └── helpers.py
└── optimiser/                       # optimiser pass tests
```

Subdirectory test packages use a local `helpers.py`; no `__init__.py` needed (pytest auto-discovers).

`writing-a-c-compiler-tests/` — official acceptance test suite from the *Writing a C Compiler* book, invoked via `test_compiler <path-to-compiler> --chapter N --stage XXXXX`.

## Key architectural invariants

- AST/TAC/ASM nodes are **immutable frozen dataclasses** — never mutated after construction.
- Typedef names are **resolved in the parser** to concrete `Type` nodes; no typedef names reach semantic analysis or beyond.
- `symbol_table` and `struct_type_table` flow as **explicit parameters**; they are never embedded in tree nodes.
- The optimiser operates on a **fixed-point loop** — passes repeat until the function body stabilises.
- Register allocation produces a `(rewritten_instructions, gp_callee_saved, fp_callee_saved)` triple; the code generator inserts callee-save push/pop from this list.
- `Pseudo` operands that survive register allocation (spilled) are rewritten to `Memory` (stack) operands by a fixup pass in `code_generator.py`.
