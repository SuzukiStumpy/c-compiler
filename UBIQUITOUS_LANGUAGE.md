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

## Relationships

- A **C source file** is transformed by **preprocessing** into a **preprocessed file**.
- A **preprocessed file** is transformed by **lexing** into a **token stream**.
- A **token stream** contains zero or more **tokens**, each carrying a **token type**, a **token value**, and a **source position**.
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
