# Ubiquitous Language

## Compilation pipeline

| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **C source file** | The original `.c` file authored by the user | Source, input file |
| **Preprocessed file** | A `.i` file produced by the GCC preprocessor from a C source file | Intermediate file |
| **Assembly file** | A `.s` file produced by the code generator, suitable for assembling and linking | Output file |
| **Executable** | The final binary produced by assembling and linking the assembly file | Binary, output |
| **Compilation stage** | One discrete step in the pipeline (lexing, parsing, semantic analysis, TAC generation, code generation) | Phase, pass |

## Pipeline stages

| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **Preprocessing** | The stage that runs `gcc -E` to expand macros and produce a preprocessed file | Pre-processing |
| **Lexing** | The stage that converts a preprocessed file into a token stream | Tokenisation, scanning |
| **Parsing** | The stage that converts a token stream into an abstract syntax tree | — |
| **Semantic analysis** | The VALIDATE stage that walks the parsed AST to verify that all variable references are in scope and to assign each declared variable a unique internal name; raises `SemanticError` on any violation | Variable resolution, validation |
| **TAC generation** | The TACKY stage that translates the semantically-validated AST into a three-address-code intermediate representation by flattening nested expression trees into sequences of atomic instructions | Tacky generation, IR generation |
| **Code generation** | The stage that converts a TAC program into an assembly AST | Codegen |
| **Assembly and linking** | The stage that runs `gcc` on an assembly file to produce an executable | Assembling, linking |

## Tokens

| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **Token** | A single classified unit of source text produced by the lexer, carrying type, value, and source position | Lexeme |
| **Token type** | The classification of a token (e.g. `RETURN`, `IDENTIFIER`, `LPAREN`) | Token kind, token class |
| **Token value** | The exact substring of source text matched by the token | Lexeme, matched text |
| **Keyword** | A token whose type corresponds to a reserved C word (`int`, `void`, `return`) | Reserved word |
| **Identifier** | A token representing a user-defined name that is not a keyword | Name, symbol |
| **Integer constant** | A token representing a numeric literal composed of decimal digits | Number, numeric constant, literal |
| **Token stream** | The ordered list of tokens produced by the lexer for a given preprocessed file | Token list, token sequence |

## Source position

| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **Source position** | The combination of filename, line number, and column number that locates a token or error in the original C source file | Location, offset |
| **Line number** | The 1-indexed line within the original C source file at which a token appears | Line, row |
| **Column number** | The 1-indexed character position within a line at which a token begins | Column, offset, char position |
| **Linemarker** | A GCC preprocessor directive of the form `# <line> "<file>"` that records the mapping from preprocessed content back to the original source file | Line directive, line marker |

## Error reporting

| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **Lexer error** | An error raised when the lexer encounters a character that cannot begin any valid token | Lex error, tokenisation error |
| **Parse error** | An error raised when the parser encounters a token that does not conform to the grammar | — |
| **Semantic error** | An error raised by the semantic analyser when the program contains a constraint violation such as a duplicate variable declaration or a reference to an undeclared variable | — |
| **Code generation error** | An error raised when the code generator encounters an AST or TAC construct it cannot translate | — |
| **Error context** | The ±2 lines of source surrounding an error line, displayed alongside the error message | Context, surrounding lines |
| **Source context** | See *error context* | — |
| **Underline** | The `^~~~~` indicator printed beneath the offending token to show its column and span | Highlight, caret |
| **Plain-text formatting** | Error output rendered without ANSI colour codes, suitable for testing and non-terminal output | — |
| **Terminal formatting** | Error output rendered with ANSI colour codes via `click`, used when writing to a terminal | Colour formatting, coloured output |
| **Diagnostic** | Any structured error or warning emitted by the compiler to describe a problem in the source | Error output, message |

## Abstract Syntax Tree

| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **Abstract Syntax Tree** | The tree structure produced by the parser from a token stream, representing the grammatical structure of the C source | Parse tree, syntax tree |
| **AST node** | A single element in the abstract syntax tree, representing one syntactic construct | Node, tree node |
| **Translation unit** | A complete C source file; the concept modelled by the `Program` AST node | Module, compilation unit |
| **Program** | The root AST node representing a translation unit; holds one or more function definitions | Root node |
| **Function definition** | An AST node representing a complete C function, including its return type, name, parameter list, and body | Function, function declaration |
| **Parameter** | An AST node representing a single named, typed argument in a function's signature | Argument, arg |
| **Function body** | The ordered sequence of block items that make up a function definition | Block, function block |
| **Block item** | A sum-type wrapper for items that may appear in a function body; each is either a statement (`S`) or a declaration (`D`) | Body item |
| **Statement** | An executable instruction appearing in a function body; one of: return statement, expression statement, or null statement | Instruction, command |
| **Return statement** | The statement form `return <exp>;` that transfers control back to the caller with a value | Return, return expression |
| **Expression statement** | A statement consisting of a single expression evaluated for its side effects; the expression's value is discarded | — |
| **Null statement** | An empty placeholder statement with no effect | Empty statement |
| **Declaration** | An AST node representing a variable declaration, carrying the variable name and an optional initialiser expression | Variable declaration |
| **Expression** | A syntactic construct that evaluates to a value; one of: constant, variable, unary operation (including prefix/postfix increment and decrement), binary operation, or assignment | Value, rvalue |
| **Integer constant expression** | The expression form consisting of a single integer literal | Literal expression, number expression |
| **Variable expression** | The expression form consisting of a reference to a declared variable by name | Variable reference |
| **Unary expression** | The expression form consisting of a unary operator applied to a single sub-expression | — |
| **Binary expression** | The expression form consisting of a binary operator applied to two sub-expressions | — |
| **Assignment expression** | The expression form `<lvalue> = <rvalue>` that stores the value of `rvalue` into the storage location named by `lvalue` | — |
| **lvalue** | An expression that denotes a storage location and may appear on the left-hand side of an assignment; currently only a variable expression | Left-hand value |
| **rvalue** | An expression whose computed value is used directly; does not denote a storage location | Right-hand value |
| **Type** | A C type annotation used in declarations (e.g. as a return type or parameter type) | Type annotation, data type |
| **Sum type** | An AST node category with multiple named variants (e.g. `Statement`, `Exp`, `Type`); represented as an abstract base class | Variant type, union type |
| **Product type** | An AST node with a fixed, named set of fields and exactly one form (e.g. `Parameter`, `Function`, `Program`); represented as a frozen dataclass | Record type, struct |
| **ASDL** | Zephyr Abstract Syntax Description Language; the notation used in this project to formally specify the grammar of AST nodes | — |

## Variables

| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **Variable** | A named storage location declared within a function body with an optional initial value | Identifier (when referring to a storage location) |
| **Variable resolution** | The semantic analysis pass that verifies every variable reference has a corresponding in-scope declaration and rewrites all names to unique internal names | Name resolution, scoping |
| **Symbol table** | The per-function mapping from source-level variable names to their unique internal names, maintained during variable resolution | Name table, scope map |
| **Unique internal name** | The compiler-generated identifier of the form `.V<name>.<n>` assigned to each declared variable; the leading `.V` prefix prevents collisions with any valid C identifier | Mangled name, renamed variable |
| **Initialiser** | The optional expression whose value is stored into a variable at the point of declaration | Initial value, init expression |
| **Scope** | The region of a function body in which a declared variable is visible; currently limited to the entire function body (no nested block scopes) | — |
| **Duplicate declaration** | A semantic error raised when a variable name is declared more than once within the same scope | Redeclaration |
| **Undeclared variable** | A semantic error raised when a variable is referenced before any declaration is in scope | Undefined variable |

## Operators

### Unary operators

| Term | C token | Form | Definition |
| ---- | ------- | ---- | ---------- |
| **Complement** | `~` | prefix | Bitwise complement: flips every bit of the operand |
| **Negate** | `-` | prefix | Arithmetic negation: computes the two's-complement negative of the operand |
| **Not** | `!` | prefix | Logical negation: produces 1 if the operand is zero, 0 otherwise |
| **PreIncrement** | `++` | prefix | Increments the operand by 1 and evaluates to the new value |
| **PreDecrement** | `--` | prefix | Decrements the operand by 1 and evaluates to the new value |
| **PostIncrement** | `++` | postfix | Evaluates to the operand's current value, then increments it by 1 |
| **PostDecrement** | `--` | postfix | Evaluates to the operand's current value, then decrements it by 1 |

### Compound assignment operators

Compound assignments are syntactic sugar desugared by the parser into a plain `Assignment` wrapping a `Binary` operation. They do not introduce new AST nodes.

| Token | Desugars to |
| ----- | ----------- |
| `+=` | `x = x + <rhs>` |
| `-=` | `x = x - <rhs>` |
| `*=` | `x = x * <rhs>` |
| `/=` | `x = x / <rhs>` |
| `%=` | `x = x % <rhs>` |
| `&=` | `x = x & <rhs>` |
| `\|=` | `x = x \| <rhs>` |
| `^=` | `x = x ^ <rhs>` |
| `<<=` | `x = x << <rhs>` |
| `>>=` | `x = x >> <rhs>` |

### Binary operators — arithmetic

| Term | C token | Definition |
| ---- | ------- | ---------- |
| **Add** | `+` | Adds two integer values |
| **Subtract** | `-` | Subtracts the right operand from the left |
| **Multiply** | `*` | Multiplies two integer values |
| **Divide** | `/` | Integer division of the left operand by the right |
| **Remainder** | `%` | Remainder after integer division |

### Binary operators — bitwise

| Term | C token | Definition |
| ---- | ------- | ---------- |
| **LeftShift** | `<<` | Shifts the left operand's bits left by the right operand's count |
| **RightShift** | `>>` | Arithmetic right shift: shifts the left operand's bits right by the right operand's count |
| **BitwiseAnd** | `&` | Bitwise AND of two values |
| **BitwiseXor** | `^` | Bitwise exclusive OR of two values |
| **BitwiseOr** | `\|` | Bitwise OR of two values |

### Binary operators — logical

| Term | C token | Definition | Note |
| ---- | ------- | ---------- | ---- |
| **And** | `&&` | Logical AND: 1 if both operands are non-zero, 0 otherwise | Short-circuits: right operand not evaluated if left is zero |
| **Or** | `\|\|` | Logical OR: 1 if either operand is non-zero, 0 otherwise | Short-circuits: right operand not evaluated if left is non-zero |

### Binary operators — relational

| Term | C token | Definition |
| ---- | ------- | ---------- |
| **Equal** | `==` | 1 if both operands are equal, 0 otherwise |
| **NotEqual** | `!=` | 1 if the operands are unequal, 0 otherwise |
| **LessThan** | `<` | 1 if the left operand is strictly less than the right, 0 otherwise |
| **LessOrEqual** | `<=` | 1 if the left operand is less than or equal to the right, 0 otherwise |
| **GreaterThan** | `>` | 1 if the left operand is strictly greater than the right, 0 otherwise |
| **GreaterOrEqual** | `>=` | 1 if the left operand is greater than or equal to the right, 0 otherwise |

## Three-Address Code (TAC)

Three-address code is the intermediate representation produced by TAC generation. It is a flat, linear sequence of atomic instructions; nested expression trees are flattened by introducing temporary variables.

| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **TAC program** | The root TAC node, holding a single TAC function; the TAC equivalent of an AST `Program` | — |
| **TAC function** | A TAC node representing a function's complete, flattened instruction sequence | — |
| **TAC instruction** | An atomic operation in a TAC function body; one of: `Return`, `Unary`, `Binary`, `Copy`, `Jump`, `JumpIfZero`, `JumpIfNotZero`, or `Label` | TAC node, IR instruction |
| **TAC value** | An operand in a TAC instruction; either a `Constant` (an integer literal) or a `Var` (a named variable or temporary) | TAC operand |
| **Temporary variable** | A TAC `Var` named `tmp.<n>`, created by the TAC generator to hold the result of a sub-expression; not present in the source | Temp, synthetic variable |
| **Instruction flattening** | The process of converting a nested AST expression tree into a linear sequence of TAC instructions, each computing a single operation into a temporary | Lowering, linearisation |
| **Copy instruction** | A TAC instruction that copies a value from a source operand to a destination operand; used to initialise variables and implement assignment | Move, assignment instruction |
| **Label** | A TAC instruction that marks a named position in the instruction sequence, used as the target of jump instructions; also used in the assembly AST for the same purpose | — |
| **Jump** | A TAC instruction that unconditionally transfers control to a named label | Goto |
| **JumpIfZero** | A TAC instruction that transfers control to a named label if a condition operand evaluates to zero | Conditional branch (false) |
| **JumpIfNotZero** | A TAC instruction that transfers control to a named label if a condition operand evaluates to non-zero | Conditional branch (true) |
| **Short-circuit evaluation** | The TAC pattern for `&&` and `\|\|` that emits a conditional jump after each operand, skipping the remaining operands when the result is already determined | — |

## Relationships

- A **C source file** is transformed by **preprocessing** into a **preprocessed file**.
- A **preprocessed file** is transformed by **lexing** into a **token stream**.
- A **token stream** is transformed by **parsing** into an **abstract syntax tree**.
- An **abstract syntax tree** is transformed by **semantic analysis** into a resolved **abstract syntax tree** in which every variable name has been replaced by a **unique internal name**.
- A resolved **abstract syntax tree** is transformed by **TAC generation** into a **TAC program**.
- A **TAC program** is transformed by **code generation** into an assembly AST.
- An assembly AST is emitted and passed to **assembly and linking** to produce an **executable**.
- A **token stream** contains zero or more **tokens**, each carrying a **token type**, a **token value**, and a **source position**.
- An **abstract syntax tree** is rooted at a **Program** node, which holds one or more **function definitions**.
- A **function definition** has a **type** (return type), a name, zero or more **parameters**, and a **function body**.
- A **parameter** has a **type** and a name.
- A **function body** is an ordered sequence of **block items**; each **block item** wraps either a **statement** (`S`) or a **declaration** (`D`).
- A **declaration** carries a variable name and an optional **initialiser** expression; after **semantic analysis**, its name is a **unique internal name**.
- A **return statement** holds an **expression**.
- An **expression statement** holds an **expression** evaluated for its side effects.
- An **assignment expression** requires its left-hand side to be an **lvalue** (currently only a **variable expression**).
- A **source position** references the original **C source file**, not the **preprocessed file**, via **linemarkers**.
- A **keyword** is a **token** whose **token type** has a distinct enum member (e.g. `RETURN`) — it is never classified as an **identifier**.
- A **lexer error** carries a **source position** and an **error context** drawn from the original **C source file**.
- **Semantic analysis** maintains a **symbol table** for each function; a **semantic error** is raised on a **duplicate declaration** or an **undeclared variable**.
- A **TAC function** body is a sequence of **TAC instructions**; each instruction operates on **TAC values** (constants and named variables).
- **Temporary variables** are created during **instruction flattening** and do not correspond to any source-level name.
- **Short-circuit evaluation** for `&&` and `||` is implemented in TAC using **JumpIfZero** and **JumpIfNotZero** instructions targeting generated **labels**.

## Example dialogue

> **Dev:** "The lexer received the **preprocessed file** — does the **token**'s filename point to the `.i` file or the `.c` file?"

> **Domain expert:** "Always the original **C source file**. The lexer reads **linemarkers** to track which file each line came from, so every **source position** references the file the user actually wrote."

> **Dev:** "What if `returned` appears in the source — is that a `RETURN` **keyword** or an **identifier**?"

> **Domain expert:** "An **identifier**. A **keyword** only matches the exact reserved word with a word boundary. `returned` matches the **identifier** pattern because the word boundary check fails after `return`."

> **Dev:** "And if the lexer hits a `$` — what does the **diagnostic** look like?"

> **Domain expert:** "A **lexer error** with the **source position** of the `$`, plus two lines of **error context** above and below. The **terminal formatting** colours it red and draws an **underline** beneath the character. The **plain-text formatting** does the same without colour — that's what we assert against in tests."

> **Dev:** "If I write `int x = x;`, does the **semantic analyser** reject it?"

> **Domain expert:** "No. The **scope** of `x` begins at its declarator, before the `=`, so the `x` on the right-hand side is the *same* `x` (a **self-referential initialiser**). It is undefined behaviour at runtime but not a **semantic error** — the **symbol table** already contains `x` when the **initialiser** is resolved."

> **Dev:** "How does `a && b` get represented in the TAC?"

> **Domain expert:** "The TAC generator emits the instructions for `a`, then a **JumpIfZero** to a false **label**, then the instructions for `b`, then another **JumpIfZero** to the same false label, then a **copy** of 1 into a **temporary variable**, then an unconditional **jump** to an end label, then the false label, then a **copy** of 0 into the temporary. That's **short-circuit evaluation** — if `a` is zero, the instructions for `b` are never reached."

## Flagged ambiguities

- **"source"** was used loosely throughout to mean both the **C source file** (user-authored `.c`) and the **preprocessed file** (`.i` passed to the lexer). These are distinct artifacts at different pipeline stages. Prefer the explicit terms.
- **"number" / "numeric constant" / "integer constant"** all appeared for the same token type. Canonical term: **integer constant**. The `TokenType` enum member is `NUMBER` for historical reasons — when discussing the domain concept, use *integer constant*.
- **"compile"** was used both for the overall pipeline and for a specific internal function. As a domain term, *compilation* refers to the full pipeline from C source file to executable. The internal function name is an implementation detail.
- **"position"** and **"location"** were used interchangeably. Canonical term: **source position**.
- **"TAC"** and **"tacky"** both appear — "tacky" is the CLI flag name (`--tacky`) and stage name; **"TAC"** (three-address code) is the domain term for the intermediate representation itself. Prefer **TAC** when discussing the concept; use *tacky* only when referring to the CLI flag or the `Stage.TACKY` bitmask member.
