"""Parser tests for ``_Static_assert`` declarations."""

import pytest

from compiler.syntax.ast import (
    Binary,
    Constant,
    ConstInt,
    D,
    Equal,
    FunDecl,
    IntType,
    SizeOfT,
    StaticAssert,
)
from compiler.types.tokens import TokenType
from compiler.utils.exceptions import ParseError
from test.parser.helpers import (
    _COMMA,
    _EQ,
    _INT,
    _LBRACE,
    _LPAREN,
    _RBRACE,
    _RPAREN,
    _SEMICOLON,
    _VOID,
    _ident,
    _number,
    _tok,
    parse,
)


def _STATIC_ASSERT() -> object:
    """Return a STATIC_ASSERT keyword token."""
    return _tok(TokenType.STATIC_ASSERT, "_Static_assert")


def _SIZEOF() -> object:
    """Return a SIZEOF keyword token."""
    return _tok(TokenType.SIZEOF, "sizeof")


def _string(value: str) -> object:
    """Return a STRING token whose source text is the quoted literal."""
    return _tok(TokenType.STRING, value)


# ---------------------------------------------------------------------------
# File-scope _Static_assert
# ---------------------------------------------------------------------------


def test_static_assert_at_file_scope_produces_static_assert_node() -> None:
    """``_Static_assert(1, "ok");`` at file scope produces a ``StaticAssert`` declaration."""
    # _Static_assert(1, "ok");
    tokens = [_STATIC_ASSERT(), _LPAREN, _number("1"), _COMMA, _string('"ok"'), _RPAREN, _SEMICOLON]
    program = parse(tokens)
    assert len(program.declarations) == 1
    decl = program.declarations[0]
    assert isinstance(decl, StaticAssert)
    assert isinstance(decl.condition, Constant)
    assert decl.condition.value == ConstInt(1)
    assert decl.message == "ok"


def test_static_assert_message_strips_quotes() -> None:
    """The ``message`` field holds the decoded string, not the quoted token text."""
    tokens = [_STATIC_ASSERT(), _LPAREN, _number("1"), _COMMA, _string('"hello world"'), _RPAREN, _SEMICOLON]
    program = parse(tokens)
    decl = program.declarations[0]
    assert isinstance(decl, StaticAssert)
    assert decl.message == "hello world"


def test_static_assert_with_zero_condition() -> None:
    """A zero condition is parsed correctly (semantic error is raised later, not by the parser)."""
    tokens = [_STATIC_ASSERT(), _LPAREN, _number("0"), _COMMA, _string('"fail"'), _RPAREN, _SEMICOLON]
    program = parse(tokens)
    decl = program.declarations[0]
    assert isinstance(decl, StaticAssert)
    assert isinstance(decl.condition, Constant)
    assert decl.condition.value == ConstInt(0)
    assert decl.message == "fail"


def test_static_assert_with_sizeof_expression() -> None:
    """``_Static_assert(sizeof(int) == 4, "msg");`` parses to a Binary(Equal, SizeOfT, Constant) condition."""
    # _Static_assert(sizeof(int) == 4, "msg");
    tokens = [
        _STATIC_ASSERT(),
        _LPAREN,
        _SIZEOF(),
        _LPAREN,
        _INT,
        _RPAREN,
        _EQ,
        _number("4"),
        _COMMA,
        _string('"msg"'),
        _RPAREN,
        _SEMICOLON,
    ]
    program = parse(tokens)
    decl = program.declarations[0]
    assert isinstance(decl, StaticAssert)
    assert isinstance(decl.condition, Binary)
    assert isinstance(decl.condition.operator, Equal)
    assert isinstance(decl.condition.left, SizeOfT)
    assert decl.condition.left.t == IntType()


def test_static_assert_followed_by_other_declarations() -> None:
    """``_Static_assert`` at file scope can be followed by further declarations."""
    tokens = [
        _STATIC_ASSERT(),
        _LPAREN,
        _number("1"),
        _COMMA,
        _string('"ok"'),
        _RPAREN,
        _SEMICOLON,
        _INT,
        _ident("x"),
        _SEMICOLON,
    ]
    program = parse(tokens)
    assert len(program.declarations) == 2
    assert isinstance(program.declarations[0], StaticAssert)


# ---------------------------------------------------------------------------
# Block-scope _Static_assert (inside a function body)
# ---------------------------------------------------------------------------


def _wrap_in_function(*inner_tokens: object) -> list[object]:
    """Wrap token sequence in ``int main(void) { ... }``."""
    return [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, *inner_tokens, _RBRACE]


def test_static_assert_inside_function_body() -> None:
    """``_Static_assert`` inside a function body is parsed as a block-item ``D``."""
    tokens = _wrap_in_function(
        _STATIC_ASSERT(),
        _LPAREN,
        _number("1"),
        _COMMA,
        _string('"ok"'),
        _RPAREN,
        _SEMICOLON,
    )
    program = parse(tokens)
    fn_decl = program.declarations[0]
    assert isinstance(fn_decl, FunDecl)
    body = fn_decl.function_declaration.body
    assert body is not None
    assert len(body) == 1
    item = body[0]
    assert isinstance(item, D)
    assert isinstance(item.declaration, StaticAssert)
    assert item.declaration.message == "ok"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_static_assert_missing_lparen_raises_parse_error() -> None:
    """``_Static_assert`` without an opening paren raises ``ParseError``."""
    tokens = [_STATIC_ASSERT(), _number("1"), _COMMA, _string('"ok"'), _RPAREN, _SEMICOLON]
    with pytest.raises(ParseError):
        parse(tokens)


def test_static_assert_missing_comma_raises_parse_error() -> None:
    """``_Static_assert`` without a comma between expr and message raises ``ParseError``."""
    tokens = [_STATIC_ASSERT(), _LPAREN, _number("1"), _string('"ok"'), _RPAREN, _SEMICOLON]
    with pytest.raises(ParseError):
        parse(tokens)


def test_static_assert_missing_message_raises_parse_error() -> None:
    """``_Static_assert`` without a string message raises ``ParseError``."""
    tokens = [_STATIC_ASSERT(), _LPAREN, _number("1"), _COMMA, _RPAREN, _SEMICOLON]
    with pytest.raises(ParseError):
        parse(tokens)


def test_static_assert_missing_rparen_raises_parse_error() -> None:
    """``_Static_assert`` without a closing paren raises ``ParseError``."""
    tokens = [_STATIC_ASSERT(), _LPAREN, _number("1"), _COMMA, _string('"ok"'), _SEMICOLON]
    with pytest.raises(ParseError):
        parse(tokens)


def test_static_assert_missing_semicolon_raises_parse_error() -> None:
    """``_Static_assert`` without a trailing semicolon raises ``ParseError``."""
    tokens = [_STATIC_ASSERT(), _LPAREN, _number("1"), _COMMA, _string('"ok"'), _RPAREN]
    with pytest.raises(ParseError):
        parse(tokens)
