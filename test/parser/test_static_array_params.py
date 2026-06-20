"""Parser tests for static array parameters (C99/C11 §6.7.6.3).

``int foo(int a[static 10])`` — ``static`` inside an array parameter declarator
is a minimum-size hint; parsed and discarded.  Type qualifiers (const, restrict,
volatile) in the same position are also consumed without effect.
"""

from compiler.parser import parse
from compiler.syntax.ast import (
    Array,
    FunDecl,
    FunType,
    IntType,
)
from compiler.types.tokens import TokenType
from test.parser.helpers import (
    _COMMA,
    _CONST,
    _INT,
    _LBRACKET,
    _LPAREN,
    _RBRACKET,
    _RPAREN,
    _SEMICOLON,
    _STATIC,
    _ident,
    _number,
    _tok,
)

_RESTRICT = _tok(TokenType.RESTRICT, "restrict")
_VOID = _tok(TokenType.VOID, "void")


def _parse_fn(*param_tokens: object) -> FunDecl:
    """Parse ``int foo(<param_tokens>) ;`` and return the FunDecl."""
    tokens = [_INT, _ident("foo"), _LPAREN, *param_tokens, _RPAREN, _SEMICOLON]
    prog = parse(list(tokens))  # type: ignore[arg-type]
    decl = prog.declarations[0]
    assert isinstance(decl, FunDecl)
    return decl


def _param_type(decl: FunDecl, idx: int = 0) -> object:
    """Return the type of the idx-th parameter from the FunDecl."""
    fn = decl.function_declaration
    ft = fn.fun_type
    assert isinstance(ft, FunType)
    return ft.params[idx]


# ---------------------------------------------------------------------------
# Basic static keyword in array parameter
# ---------------------------------------------------------------------------


def test_static_array_param_basic() -> None:
    """``int a[static 10]`` — static consumed, Array(IntType, 10) produced."""
    # int foo(int a[static 10]);
    tokens = [_INT, _ident("a"), _LBRACKET, _STATIC, _number("10"), _RBRACKET]
    decl = _parse_fn(_INT, *tokens[1:])
    assert _param_type(decl) == Array(IntType(), 10)


def test_static_array_param_qualifier_before_static() -> None:
    """``int a[const static 10]`` — qualifier + static both consumed."""
    tokens = [_INT, _ident("a"), _LBRACKET, _CONST, _STATIC, _number("10"), _RBRACKET]
    decl = _parse_fn(_INT, *tokens[1:])
    assert _param_type(decl) == Array(IntType(), 10)


def test_static_array_param_static_before_qualifier() -> None:
    """``int a[static const 10]`` — static + qualifier both consumed."""
    tokens = [_INT, _ident("a"), _LBRACKET, _STATIC, _CONST, _number("10"), _RBRACKET]
    decl = _parse_fn(_INT, *tokens[1:])
    assert _param_type(decl) == Array(IntType(), 10)


def test_static_array_param_restrict() -> None:
    """``int a[static restrict 10]`` — static + restrict both consumed."""
    tokens = [_INT, _ident("a"), _LBRACKET, _STATIC, _RESTRICT, _number("10"), _RBRACKET]
    decl = _parse_fn(_INT, *tokens[1:])
    assert _param_type(decl) == Array(IntType(), 10)


def test_static_array_param_qualifier_only() -> None:
    """``int a[const 10]`` — qualifier without static also consumed."""
    tokens = [_INT, _ident("a"), _LBRACKET, _CONST, _number("10"), _RBRACKET]
    decl = _parse_fn(_INT, *tokens[1:])
    assert _param_type(decl) == Array(IntType(), 10)


# ---------------------------------------------------------------------------
# Multiple parameters, some with static
# ---------------------------------------------------------------------------


def test_static_array_multiple_params() -> None:
    """Two static array params in the same function."""
    # int foo(int a[static 10], int b[static 5]);
    tokens = [
        _INT,
        _ident("a"),
        _LBRACKET,
        _STATIC,
        _number("10"),
        _RBRACKET,
        _COMMA,
        _INT,
        _ident("b"),
        _LBRACKET,
        _STATIC,
        _number("5"),
        _RBRACKET,
    ]
    decl = _parse_fn(*tokens)
    assert _param_type(decl, 0) == Array(IntType(), 10)
    assert _param_type(decl, 1) == Array(IntType(), 5)


# ---------------------------------------------------------------------------
# Regular (non-static) array params still work
# ---------------------------------------------------------------------------


def test_plain_array_param_unaffected() -> None:
    """``int a[10]`` — ordinary array param still parses correctly."""
    tokens = [_INT, _ident("a"), _LBRACKET, _number("10"), _RBRACKET]
    decl = _parse_fn(_INT, *tokens[1:])
    assert _param_type(decl) == Array(IntType(), 10)


def test_incomplete_array_param_unaffected() -> None:
    """``int a[]`` — incomplete array param still parses correctly."""
    tokens = [_INT, _ident("a"), _LBRACKET, _RBRACKET]
    decl = _parse_fn(_INT, *tokens[1:])
    assert _param_type(decl) == Array(IntType(), None)
