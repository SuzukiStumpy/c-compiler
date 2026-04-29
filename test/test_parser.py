"""Tests for the parsing stage.

Covers the full range of constructs supported by the current grammar:

    program             = Program(function_definition+)
    function_definition = Function(type, identifier, parameter*, statement+)
    parameter           = Parameter(type, identifier)
    statement           = Return(exp)
    exp                 = Constant(int)
    type                = IntType

Each test builds a token stream directly (without invoking the lexer) so that
parser behaviour is tested in isolation from lexer behaviour.
"""

import pytest

from compiler.parser import parse
from compiler.syntax.ast import Constant, Function, IntType, Parameter, Program, Return
from compiler.types.tokens import Token, TokenType
from compiler.utils.exceptions import ParseError


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


def _ident(name: str) -> Token:
    """Return an IDENTIFIER token with the given name."""
    return _tok(TokenType.IDENTIFIER, name)


def _number(value: str) -> Token:
    """Return a NUMBER token with the given literal text."""
    return _tok(TokenType.NUMBER, value)


# ---------------------------------------------------------------------------
# Valid programs
# ---------------------------------------------------------------------------


class TestValidPrograms:
    """Parser accepts well-formed programs and produces the correct AST."""

    def test_simple_function_void_params(self) -> None:
        """``int main(void) { return 0; }`` produces the expected AST."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        result = parse(tokens)
        expected = Program(
            functions=(
                Function(
                    return_type=IntType(),
                    name="main",
                    params=(),
                    body=(Return(exp=Constant(value=0)),),
                ),
            )
        )
        assert result == expected

    def test_simple_function_empty_params(self) -> None:
        """``int main() { return 0; }`` — empty parens produce no parameters."""
        tokens = [_INT, _ident("main"), _LPAREN, _RPAREN, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        result = parse(tokens)
        assert result.functions[0].params == ()

    def test_nonzero_return_value(self) -> None:
        """``return 42;`` produces a ``Constant`` with the correct integer value."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("42"), _SEMICOLON, _RBRACE]
        result = parse(tokens)
        assert result.functions[0].body == (Return(exp=Constant(value=42)),)

    def test_single_int_parameter(self) -> None:
        """``int foo(int x) { return x; }`` — wait, no variable references yet.

        Tests that a single typed parameter is parsed into a ``Parameter`` node.
        """
        tokens = [
            _INT,
            _ident("foo"),
            _LPAREN,
            _INT,
            _ident("x"),
            _RPAREN,
            _LBRACE,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        ]
        result = parse(tokens)
        assert result.functions[0].params == (Parameter(param_type=IntType(), name="x"),)

    def test_multiple_functions(self) -> None:
        """Two consecutive function definitions both appear in ``Program.functions``."""

        def _simple_fn(name: str) -> list[Token]:
            return [_INT, _ident(name), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]

        tokens = _simple_fn("foo") + _simple_fn("bar")
        result = parse(tokens)
        assert len(result.functions) == 2
        assert result.functions[0].name == "foo"
        assert result.functions[1].name == "bar"

    def test_multiple_statements_in_body(self) -> None:
        """Multiple ``return`` statements are all captured in the function body."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _RETURN,
            _number("1"),
            _SEMICOLON,
            _RETURN,
            _number("2"),
            _SEMICOLON,
            _RBRACE,
        ]
        result = parse(tokens)
        assert result.functions[0].body == (
            Return(exp=Constant(value=1)),
            Return(exp=Constant(value=2)),
        )


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestParseErrors:
    """Parser raises ``ParseError`` for malformed token streams."""

    def test_empty_token_stream(self) -> None:
        """An empty token stream raises ``ParseError``."""
        with pytest.raises(ParseError):
            parse([])

    def test_missing_function_name(self) -> None:
        """``int (void) { ... }`` — LPAREN where IDENTIFIER expected."""
        tokens = [_INT, _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        with pytest.raises(ParseError, match="expected IDENTIFIER"):
            parse(tokens)

    def test_missing_opening_paren(self) -> None:
        """``int main void) { ... }`` — VOID where LPAREN expected."""
        tokens = [_INT, _ident("main"), _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        with pytest.raises(ParseError, match="expected LPAREN"):
            parse(tokens)

    def test_missing_closing_paren(self) -> None:
        """``int main(void { ... }`` — LBRACE where RPAREN expected."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        with pytest.raises(ParseError, match="expected RPAREN"):
            parse(tokens)

    def test_missing_opening_brace(self) -> None:
        """``int main(void) return 0; }`` — RETURN where LBRACE expected."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        with pytest.raises(ParseError, match="expected LBRACE"):
            parse(tokens)

    def test_empty_function_body(self) -> None:
        """``int main(void) {}`` — body with no statements is rejected."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RBRACE]
        with pytest.raises(ParseError, match="at least one statement"):
            parse(tokens)

    def test_missing_semicolon(self) -> None:
        """``return 0 }`` — missing semicolon after expression."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _RBRACE]
        with pytest.raises(ParseError, match="expected SEMICOLON"):
            parse(tokens)

    def test_non_int_return_type(self) -> None:
        """``void main(void) { ... }`` — VOID is not a valid return type in the grammar."""
        tokens = [_VOID, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        with pytest.raises(ParseError, match="expected a type"):
            parse(tokens)

    def test_non_number_expression(self) -> None:
        """``return main;`` — IDENTIFIER is not a valid expression."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _ident("main"), _SEMICOLON, _RBRACE]
        with pytest.raises(ParseError, match="Malformed expression."):
            parse(tokens)

    def test_unexpected_eof_in_body(self) -> None:
        """Token stream ending mid-body raises ``ParseError`` for unexpected EOF."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("0")]
        with pytest.raises(ParseError):
            parse(tokens)

    def test_parse_error_carries_location(self) -> None:
        """``ParseError`` records the filename, line, and column of the offending token."""
        bad_token = _tok(TokenType.LPAREN, "(", line=5, col=12, filename="src/foo.c")
        tokens = [_INT, bad_token]
        with pytest.raises(ParseError) as exc_info:
            parse(tokens)
        err = exc_info.value
        assert err.filename == "src/foo.c"
        assert err.line == 5
        assert err.col == 12

    def test_parse_error_str_contains_location_header(self) -> None:
        """``str(ParseError)`` begins with the clang-style location header."""
        bad_token = _tok(TokenType.LPAREN, "(", line=3, col=7, filename="main.c")
        tokens = [_INT, bad_token]
        with pytest.raises(ParseError) as exc_info:
            parse(tokens)
        text = str(exc_info.value)
        assert text.startswith("ParseError at main.c:3:7:")

    def test_parse_error_str_header_only_when_no_source_lines(self) -> None:
        """When the source file cannot be read, ``str()`` is just the header line."""
        bad_token = _tok(TokenType.LPAREN, "(", line=1, col=1, filename="nonexistent.c")
        tokens = [_INT, bad_token]
        with pytest.raises(ParseError) as exc_info:
            parse(tokens)
        text = str(exc_info.value)
        assert "\n" not in text
