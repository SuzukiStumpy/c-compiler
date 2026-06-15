"""Token types and the Token dataclass used throughout the compiler pipeline."""

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """Classification of a token produced by the lexer.

    Each member corresponds to one syntactic category. Keywords have distinct
    members (e.g. ``RETURN``) rather than sharing a generic ``KEYWORD`` member,
    so the parser can match on type alone without inspecting the token value.
    New token types are added here as additional language features are supported.
    """

    # Keywords - must be matched before IDENTIFIER in the lexer
    INT = auto()
    VOID = auto()
    RETURN = auto()
    IF = auto()
    ELSE = auto()
    GOTO = auto()
    DO = auto()
    WHILE = auto()
    FOR = auto()
    BREAK = auto()
    CONTINUE = auto()
    SWITCH = auto()
    CASE = auto()
    DEFAULT = auto()
    STATIC = auto()
    EXTERN = auto()
    LONG = auto()
    SIGNED = auto()
    UNSIGNED = auto()
    DOUBLE = auto()
    CHAR = auto()
    SHORT = auto()
    SIZEOF = auto()
    STRUCT = auto()
    UNION = auto()
    TYPEDEF = auto()
    INT128 = auto()
    # Synthetic type-specifier sentinel – not produced by the lexer; used internally
    # by the parser to represent the two-token ``long long`` specifier as a single
    # entry in the type-specifier frozenset.
    LONG_LONG = auto()
    # Identifiers and literals
    IDENTIFIER = auto()
    NUMBER = auto()
    LONGLONG = auto()
    ULONGLONG = auto()
    LONGINT = auto()
    UINT = auto()
    ULONG = auto()
    FPDOUBLE = auto()
    CONSTCHAR = auto()
    STRING = auto()
    # Punctuation
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    LBRACE = auto()  # {
    RBRACE = auto()  # }
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    SEMICOLON = auto()  # ;
    PERIOD = auto()  # .
    ARROW = auto()  # ->
    # Boolean Operators
    BITWISE_NOT = auto()  # ~
    # Compound Assignment Operators
    PLUS_ASSIGN = auto()  # +=
    MINUS_ASSIGN = auto()  # -=
    MUL_ASSIGN = auto()  # *=
    DIV_ASSIGN = auto()  # /=
    MOD_ASSIGN = auto()  # %=
    BITWISE_AND_ASSIGN = auto()  # &=
    BITWISE_OR_ASSIGN = auto()  # \|=
    BITWISE_XOR_ASSIGN = auto()  # \^=
    LSHIFT_ASSIGN = auto()  # <<=
    RSHIFT_ASSIGN = auto()  # >>=
    # Unary Operators
    MINUS = auto()  # This is also a binary operator, which we disambiguate in the parser
    DECREMENT = auto()  # --
    INCREMENT = auto()  # ++
    # Binary Operators
    PLUS = auto()  # +
    ASTERISK = auto()  # *
    DIVIDE = auto()  # /
    PERCENT = auto()  # % Modulo operator
    LSHIFT = auto()  # <<
    RSHIFT = auto()  # >>
    AMPERSAND = auto()  # &
    CARET = auto()  # ^
    PIPE = auto()  # |
    # Logical Operator Tokens
    NOT = auto()  # !
    AND = auto()  # &&
    OR = auto()  # ||
    EQ = auto()  # ==
    NEQ = auto()  # !=
    LT = auto()  # <
    GT = auto()  # >
    LEQ = auto()  # <=
    GEQ = auto()  # >=
    ASSIGNMENT = auto()  # =
    # Conditional Expressions
    QUERY = auto()  # ?
    COLON = auto()  # :
    COMMA = auto()  # ,


@dataclass(frozen=True, slots=True)
class Token:
    """An immutable unit of source text produced by the lexer.

    Each token carries its classification, matched text, and the exact source
    position at which it was found in the original C source file. Tokens are
    the primary interface between the lexer and the parser.

    Attributes:
        type: The classification of this token.
        value: The exact substring of source text that was matched.
        line: 1-indexed line number in the original C source file.
        col: 1-indexed column number at which this token begins.
        filename: Path to the original C source file, resolved via linemarkers.
    """

    type: TokenType
    value: str
    line: int
    col: int
    filename: str
