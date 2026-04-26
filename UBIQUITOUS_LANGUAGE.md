# Ubiquitous Language

## Compilation pipeline

| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **C source file** | The original `.c` file authored by the user | Source, input file |
| **Preprocessed file** | A `.i` file produced by the GCC preprocessor from a C source file | Intermediate file |
| **Assembly file** | A `.s` file produced by the code generator, suitable for assembling and linking | Output file |
| **Executable** | The final binary produced by assembling and linking the assembly file | Binary, output |
| **Compilation stage** | One discrete step in the pipeline (lexing, parsing, code generation) | Phase, pass |

## Pipeline stages
                                                         
| Term | Definition | Aliases to avoid |
| ---- | ---------- | ---------------- |
| **Preprocessing** | The stage that runs `gcc -E` to expand macros and produce a preprocessed file | Pre-processing |
| **Lexing** | The stage that converts a preprocessed file into a token stream | Tokenisation, scanning |
| **Parsing** | The stage that converts a token stream into an abstract syntax tree | — |
| **Code generation** | The stage that converts an abstract syntax tree into an assembly file | Codegen |
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
| **Function body** | The ordered sequence of statements that make up a function definition | Block, function block |
| **Statement** | An executable instruction appearing in a function body | Instruction, command |
| **Return statement** | The statement form `return <exp>;` that transfers control back to the caller with a value | Return, return expression |
| **Expression** | A syntactic construct that evaluates to a value | Value, rvalue |
| **Integer constant expression** | The expression form consisting of a single integer literal | Literal expression, number expression |
| **Type** | A C type annotation used in declarations (e.g. as a return type or parameter type) | Type annotation, data type |
| **Sum type** | An AST node category with multiple named variants (e.g. `Statement`, `Exp`, `Type`); represented as an abstract base class | Variant type, union type |
| **Product type** | An AST node with a fixed, named set of fields and exactly one form (e.g. `Parameter`, `Function`, `Program`); represented as a frozen dataclass | Record type, struct |
| **ASDL** | Zephyr Abstract Syntax Description Language; the notation used in this project to formally specify the grammar of AST nodes | — |

## Relationships

- A **C source file** is transformed by **preprocessing** into a **preprocessed file**.
- A **preprocessed file** is transformed by **lexing** into a **token stream**.
- A **token stream** is transformed by **parsing** into an **abstract syntax tree**.
- A **token stream** contains zero or more **tokens**, each carrying a **token type**, a **token value**, and a **source position**.
- An **abstract syntax tree** is rooted at a **Program** node, which holds one or more **function definitions**.
- A **function definition** has a **type** (return type), a name, zero or more **parameters**, and a **function body**.
- A **parameter** has a **type** and a name.
- A **function body** is an ordered sequence of **statements**.
- A **return statement** holds an **expression**; in the current grammar, an **expression** is always an **integer constant expression**.
- A **source position** references the original **C source file**, not the **preprocessed file**, via **linemarkers**.
- A **keyword** is a **token** whose **token type** has a distinct enum member (e.g. `RETURN`) — it is never classified as an **identifier**.
- A **lexer error** carries a **source position** and an **error context** drawn from the original **C source file**.

## Example dialogue

> **Dev:** "The lexer received the **preprocessed file** — does the **token**'s filename point to the `.i` file or the `.c` file?"

> **Domain expert:** "Always the original **C source file**. The lexer reads **linemarkers** to track which file each line came from, so every **source position** references the file the user actually wrote."

> **Dev:** "What if `returned` appears in the source — is that a `RETURN` **keyword** or an **identifier**?"

> **Domain expert:** "An **identifier**. A **keyword** only matches the exact reserved word with a word boundary. `returned` matches the **identifier** pattern because the word boundary check fails after `return`."

> **Dev:** "And if the lexer hits a `$` — what does the **diagnostic** look like?"

> **Domain expert:** "A **lexer error** with the **source position** of the `$`, plus two lines of **error context** above and below. The **terminal formatting** colours it red and draws an **underline** beneath the character. The **plain-text formatting** does the same without colour — that's what we assert against in tests."

## Flagged ambiguities

- **"source"** was used loosely throughout to mean both the **C source file** (user-authored `.c`) and the **preprocessed file** (`.i` passed to the lexer). These are distinct artifacts at different pipeline stages. Prefer the explicit terms.
- **"number" / "numeric constant" / "integer constant"** all appeared for the same token type. Canonical term: **integer constant**. The `TokenType` enum member is `NUMBER` for historical reasons — when discussing the domain concept, use *integer constant*.
- **"compile"** was used both for the overall pipeline and for a specific internal function. As a domain term, *compilation* refers to the full pipeline from C source file to executable. The internal function name is an implementation detail.
- **"position"** and **"location"** were used interchangeably. Canonical term: **source position**.
