"""Shared helpers for the parser test package.

Provides token constants and lightweight AST-building utilities used across
the themed test modules.  Each test builds a token stream directly (without
invoking the lexer) so that parser behaviour is tested in isolation.
"""

import pytest  # noqa: F401 - re-exported for test modules that don't import it directly

from compiler.parser import parse  # noqa: F401
from compiler.syntax.ast import (  # noqa: F401
    AddrOf,
    Array,
    Block,
    BlockItem,
    Break,
    Case,
    Cast,
    Char,
    Compound,
    CompoundInit,
    Constant,
    ConstInt,
    ConstLong,
    Continue,
    D,
    Default,
    Dereference,
    DoWhile,
    Expression,
    For,
    Function,
    FunctionCall,
    FunDecl,
    FunType,
    Goto,
    If,
    InitDecl,
    InitExp,
    IntType,
    Label,
    LongType,
    Null,
    Parameter,
    Pointer,
    Program,
    Return,
    S,
    SChar,
    SingleInit,
    Statement,
    StorageClass,
    String,
    Switch,
    UChar,
    VarDecl,
    VariableDeclaration,
    While,
)
from compiler.syntax.ast.expressions import Exp
from compiler.syntax.ast.operators import (  # noqa: F401
    Add,
    And,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Complement,
    Divide,
    Equal,
    GreaterOrEqual,
    GreaterThan,
    LeftShift,
    LessOrEqual,
    LessThan,
    Multiply,
    Negate,
    Not,
    NotEqual,
    Or,
    PostDecrement,
    PostIncrement,
    PreDecrement,
    PreIncrement,
    Remainder,
    RightShift,
    Subtract,
)
from compiler.types.tokens import Token, TokenType
from compiler.utils.exceptions import ParseError  # noqa: F401


def _tok(
    token_type: TokenType,
    value: str,
    line: int = 1,
    col: int = 1,
    filename: str = "test.c",
) -> Token:
    """Construct a Token with positional defaults for use in tests.

    Args:
        token_type: The classification of the token.
        value: The source text of the token.
        line: 1-indexed line number (default 1).
        col: 1-indexed column number (default 1).
        filename: Source filename (default 'test.c').

    Returns:
        An immutable ``Token`` instance.
    """
    return Token(type=token_type, value=value, line=line, col=col, filename=filename)


# ---------------------------------------------------------------------------
# Helpers for common token sequences
# ---------------------------------------------------------------------------

_INT = _tok(TokenType.INT, "int")
_VOID = _tok(TokenType.VOID, "void")
_RETURN = _tok(TokenType.RETURN, "return")
_LPAREN = _tok(TokenType.LPAREN, "(")
_RPAREN = _tok(TokenType.RPAREN, ")")
_LBRACE = _tok(TokenType.LBRACE, "{")
_RBRACE = _tok(TokenType.RBRACE, "}")
_SEMICOLON = _tok(TokenType.SEMICOLON, ";")
_BITWISE_NOT = _tok(TokenType.BITWISE_NOT, "~")
_MINUS = _tok(TokenType.MINUS, "-")
_PLUS = _tok(TokenType.PLUS, "+")
_ASTERISK = _tok(TokenType.ASTERISK, "*")
_DIVIDE = _tok(TokenType.DIVIDE, "/")
_PERCENT = _tok(TokenType.PERCENT, "%")
_LSHIFT = _tok(TokenType.LSHIFT, "<<")
_RSHIFT = _tok(TokenType.RSHIFT, ">>")
_AMPERSAND = _tok(TokenType.AMPERSAND, "&")
_CARET = _tok(TokenType.CARET, "^")
_PIPE = _tok(TokenType.PIPE, "|")
_NOT = _tok(TokenType.NOT, "!")
_AND = _tok(TokenType.AND, "&&")
_INCREMENT = _tok(TokenType.INCREMENT, "++")
_DECREMENT = _tok(TokenType.DECREMENT, "--")
_OR = _tok(TokenType.OR, "||")
_EQ = _tok(TokenType.EQ, "==")
_NEQ = _tok(TokenType.NEQ, "!=")
_LT = _tok(TokenType.LT, "<")
_GT = _tok(TokenType.GT, ">")
_LEQ = _tok(TokenType.LEQ, "<=")
_GEQ = _tok(TokenType.GEQ, ">=")
_IF = _tok(TokenType.IF, "if")
_ELSE = _tok(TokenType.ELSE, "else")
_GOTO = _tok(TokenType.GOTO, "goto")
_BREAK = _tok(TokenType.BREAK, "break")
_CONTINUE = _tok(TokenType.CONTINUE, "continue")
_WHILE = _tok(TokenType.WHILE, "while")
_DO = _tok(TokenType.DO, "do")
_FOR = _tok(TokenType.FOR, "for")
_SWITCH = _tok(TokenType.SWITCH, "switch")
_CASE = _tok(TokenType.CASE, "case")
_DEFAULT = _tok(TokenType.DEFAULT, "default")
_ASSIGNMENT = _tok(TokenType.ASSIGNMENT, "=")
_QUERY = _tok(TokenType.QUERY, "?")
_COLON = _tok(TokenType.COLON, ":")
_COMMA = _tok(TokenType.COMMA, ",")
_STATIC = _tok(TokenType.STATIC, "static")
_EXTERN = _tok(TokenType.EXTERN, "extern")
_AUTO = _tok(TokenType.AUTO, "auto")
_REGISTER = _tok(TokenType.REGISTER, "register")
_CONST = _tok(TokenType.CONST, "const")
_LONG = _tok(TokenType.LONG, "long")
_CHAR = _tok(TokenType.CHAR, "char")
_SIGNED = _tok(TokenType.SIGNED, "signed")
_UNSIGNED = _tok(TokenType.UNSIGNED, "unsigned")
_LBRACKET = _tok(TokenType.LBRACKET, "[")
_RBRACKET = _tok(TokenType.RBRACKET, "]")


def _ident(name: str) -> Token:
    """Return an IDENTIFIER token with the given name."""
    return _tok(TokenType.IDENTIFIER, name)


def _number(value: str) -> Token:
    """Return a NUMBER token with the given literal text."""
    return _tok(TokenType.NUMBER, value)


def _longint(value: str) -> Token:
    """Return a LONGINT token with the given literal text (e.g. ``"42L"``)."""
    return _tok(TokenType.LONGINT, value)


def _charconst(value: str) -> Token:
    """Return a CONSTCHAR token with the given literal text (e.g. ``"'a'"``).

    Args:
        value: Raw source text of the character constant, including surrounding single quotes.

    Returns:
        A ``Token`` of type ``CONSTCHAR``.
    """
    return _tok(TokenType.CONSTCHAR, value)


def _string_tok(value: str) -> Token:
    """Return a STRING token with the given literal text (e.g. ``'"hello"'``).

    Args:
        value: Raw source text of the string literal, including surrounding double quotes.

    Returns:
        A ``Token`` of type ``STRING``.
    """
    return _tok(TokenType.STRING, value)


def _wchar_tok(value: str) -> Token:
    """Return a WCHAR token (e.g. ``"L'A'"``).

    Args:
        value: Raw source text including the ``L`` prefix and surrounding single quotes.

    Returns:
        A ``Token`` of type ``WCHAR``.
    """
    return _tok(TokenType.WCHAR, value)


def _u16char_tok(value: str) -> Token:
    """Return a U16CHAR token (e.g. ``"u'A'"``).

    Args:
        value: Raw source text including the ``u`` prefix and surrounding single quotes.

    Returns:
        A ``Token`` of type ``U16CHAR``.
    """
    return _tok(TokenType.U16CHAR, value)


def _u32char_tok(value: str) -> Token:
    """Return a U32CHAR token (e.g. ``"U'A'"``).

    Args:
        value: Raw source text including the ``U`` prefix and surrounding single quotes.

    Returns:
        A ``Token`` of type ``U32CHAR``.
    """
    return _tok(TokenType.U32CHAR, value)


def _wstring_tok(value: str) -> Token:
    """Return a WSTRING token (e.g. ``'L"hello"'``).

    Args:
        value: Raw source text including the ``L`` prefix and surrounding double quotes.

    Returns:
        A ``Token`` of type ``WSTRING``.
    """
    return _tok(TokenType.WSTRING, value)


def _u16string_tok(value: str) -> Token:
    """Return a U16STRING token (e.g. ``'u"hello"'``).

    Args:
        value: Raw source text including the ``u`` prefix and surrounding double quotes.

    Returns:
        A ``Token`` of type ``U16STRING``.
    """
    return _tok(TokenType.U16STRING, value)


def _u32string_tok(value: str) -> Token:
    """Return a U32STRING token (e.g. ``'U"hello"'``).

    Args:
        value: Raw source text including the ``U`` prefix and surrounding double quotes.

    Returns:
        A ``Token`` of type ``U32STRING``.
    """
    return _tok(TokenType.U32STRING, value)


def _u8string_tok(value: str) -> Token:
    """Return a U8STRING token (e.g. ``'u8"hello"'``).

    Args:
        value: Raw source text including the ``u8`` prefix and surrounding double quotes.

    Returns:
        A ``Token`` of type ``U8STRING``.
    """
    return _tok(TokenType.U8STRING, value)


def _parsed_body(*body_tokens: Token) -> tuple[BlockItem, ...]:
    """Parse a minimal function from the given body tokens and return the body tuple.

    Args:
        body_tokens: Token sequence that forms the function body (between braces).

    Returns:
        The ``body`` tuple of ``BlockItem`` nodes from the parsed function.
    """
    tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, *body_tokens, _RBRACE]
    decl = parse(tokens).declarations[0]
    assert isinstance(decl, FunDecl)
    body = decl.function_declaration.body
    assert body is not None
    return body


def _parsed_stmt(*body_tokens: Token) -> Statement:
    """Parse a function body and return the first statement node.

    Args:
        body_tokens: Token sequence forming the first (and only expected) statement.

    Returns:
        The ``Statement`` node from the first block item of the parsed body.
    """
    body = _parsed_body(*body_tokens)
    item = body[0]
    assert isinstance(item, S)
    return item.statement


def _parsed_exp(*exp_tokens: Token) -> Exp:
    """Parse a minimal ``return <exp>;`` function and return the expression node.

    Args:
        exp_tokens: The tokens that make up the expression being returned.

    Returns:
        The ``Exp`` node from the parsed ``Return`` statement.
    """
    tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, *exp_tokens, _SEMICOLON, _RBRACE]
    result = parse(tokens)
    decl = result.declarations[0]
    assert isinstance(decl, FunDecl)
    body = decl.function_declaration.body
    assert body is not None
    item = body[0]
    assert isinstance(item, S)
    assert isinstance(item.statement, Return)
    return item.statement.exp
