"""Reads in a preprocessed source file and produces a token stream"""

import re
from pathlib import Path

from compiler.types.tokens import Token, TokenType
from compiler.utils.exceptions import LexerError

_TOKEN_PATTERNS: list[tuple[str, str]] = [
    # Keywords — must precede IDENTIFIER
    ("INT", r"int\b"),
    ("VOID", r"void\b"),
    ("RETURN", r"return\b"),
    # Identifiers and literals
    ("IDENTIFIER", r"[a-zA-Z_]\w*\b"),
    ("NUMBER", r"\d+\b"),
    # Punctuation
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("SEMICOLON", r";"),
    # Boolan Operations
    ("BITWISE_NOT", r"~"),
    # Unary Operators
    ("DECREMENT", r"--"),
    ("MINUS", r"-"),  # This is also a binary operator, which we disambiguate in the parser
    # Binary Operators
    ("PLUS", r"\+"),
    ("TIMES", r"\*"),
    ("DIVIDE", r"/"),
    ("PERCENT", r"%"),  # Modulo operator
    # Shift operators must precede any future single < / > relational tokens
    ("LSHIFT", r"<<"),
    ("RSHIFT", r">>"),
    ("CARET", r"\^"),
    # Logical operators — multi-character tokens must precede their single-char prefixes
    ("AND", r"&&"),
    ("AMPERSAND", r"&"),
    ("OR", r"\|\|"),
    ("PIPE", r"\|"),
    ("EQ", r"=="),
    ("NEQ", r"!="),
    ("LEQ", r"<="),
    ("GEQ", r">="),
    ("NOT", r"!"),
    ("LT", r"<"),
    ("GT", r">"),
]

_TOKEN_REGEX = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_PATTERNS),
    re.ASCII,
)

_LINEMARKER_RE = re.compile(r'# (\d+) "([^"]*)"')
_WHITESPACE_RE = re.compile(r"\s+")


def lexer(source: Path) -> list[Token]:
    """Tokenise a GCC-preprocessed C source file.

    Reads the preprocessed file, strips GCC linemarkers (updating source
    position tracking), skips whitespace, and matches each remaining substring
    against the token pattern table. Returns the complete token stream.

    Args:
        source: Path to a preprocessed ``.i`` file produced by ``gcc -E``.

    Returns:
        A list of tokens in source order. Each token's ``filename``, ``line``,
        and ``col`` fields reference the original C source file as reported by
        the linemarkers in ``source``.

    Raises:
        LexerError: If any character at the current position cannot begin a
            valid token. The error carries the source position and ±2 lines of
            context from the original C source file.
    """
    text = source.read_text()
    source_lines = text.splitlines()

    tokens: list[Token] = []
    pos = 0
    current_file = source.name
    current_line = 1
    current_col = 1

    while pos < len(text):
        # Linemarkers only appear at the start of a line
        if pos == 0 or text[pos - 1] == "\n":
            if lm := _LINEMARKER_RE.match(text, pos):
                current_line = int(lm.group(1))
                current_file = lm.group(2)
                end = text.find("\n", pos)
                pos = end + 1 if end != -1 else len(text)
                current_col = 1
                continue

        if ws := _WHITESPACE_RE.match(text, pos):
            consumed = ws.group()
            newlines = consumed.count("\n")
            if newlines:
                current_line += newlines
                current_col = len(consumed) - consumed.rfind("\n")
            else:
                current_col += len(consumed)
            pos += len(consumed)
            continue

        tok = _TOKEN_REGEX.match(text, pos)
        if not tok:
            bad = text[pos]
            try:
                error_lines = Path(current_file).read_text().splitlines()
            except OSError:
                error_lines = source_lines
            raise LexerError(
                message=f"unexpected character {bad!r}",
                filename=current_file,
                line=current_line,
                col=current_col,
                token_value=bad,
                source_lines=error_lines,
            )

        assert tok.lastgroup is not None
        value = tok.group()
        tokens.append(
            Token(type=TokenType[tok.lastgroup], value=value, line=current_line, col=current_col, filename=current_file)
        )
        current_col += len(value)
        pos += len(value)

    return tokens
