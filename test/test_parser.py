"""Tests for the parsing stage.

Covers the full range of constructs supported by the current grammar:

    program             ::= function
    function            ::= "int" identifier "(" "void" ")" "{" statement "}"
    statement           ::= "return" exp ";"
    exp                 ::= factor | exp binop exp
    factor              ::= int | unop factor | "(" exp ")"
    unop                ::= "-" | "~"
    binop               ::= "+" | "-" | "*" | "/" | "%" | "<<" | ">>" | "&" | "^" | "|"
    identifier          ::= ? An identifier token ?
    int                 ::= ? A constant token ?

Each test builds a token stream directly (without invoking the lexer) so that
parser behaviour is tested in isolation from lexer behaviour.
"""

import pytest

from compiler.parser import parse
from compiler.syntax.ast import Constant, Function, IntType, Parameter, Program, Return
from compiler.syntax.ast.expressions import Binary, Exp, Unary
from compiler.syntax.ast.operators import (
    Add,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Complement,
    Divide,
    LeftShift,
    Multiply,
    Negate,
    Remainder,
    RightShift,
    Subtract,
)
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
_BITWISE_NOT = _tok(TokenType.BITWISE_NOT, "~")
_MINUS = _tok(TokenType.MINUS, "-")
_PLUS = _tok(TokenType.PLUS, "+")
_TIMES = _tok(TokenType.TIMES, "*")
_DIVIDE = _tok(TokenType.DIVIDE, "/")
_PERCENT = _tok(TokenType.PERCENT, "%")
_LSHIFT = _tok(TokenType.LSHIFT, "<<")
_RSHIFT = _tok(TokenType.RSHIFT, ">>")
_AMPERSAND = _tok(TokenType.AMPERSAND, "&")
_CARET = _tok(TokenType.CARET, "^")
_PIPE = _tok(TokenType.PIPE, "|")


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


# ---------------------------------------------------------------------------
# Unary expressions (_parse_factor)
# ---------------------------------------------------------------------------


class TestUnaryExpressions:
    """Parser correctly builds Unary nodes for all unary operators."""

    def _parsed_exp(self, *exp_tokens: Token) -> Exp:
        """Parse a minimal function returning the given expression tokens and return the expression."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, *exp_tokens, _SEMICOLON, _RBRACE]
        result = parse(tokens)
        stmt = result.functions[0].body[0]
        assert isinstance(stmt, Return)
        return stmt.exp

    def test_negate(self) -> None:
        """``return -1;`` → Unary(Negate(), Constant(1))."""
        assert self._parsed_exp(_MINUS, _number("1")) == Unary(Negate(), Constant(1))

    def test_complement(self) -> None:
        """``return ~1;`` → Unary(Complement(), Constant(1))."""
        assert self._parsed_exp(_BITWISE_NOT, _number("1")) == Unary(Complement(), Constant(1))

    def test_double_negate(self) -> None:
        """``return --1;`` — wait, DECREMENT is a separate token; ``- -1`` → Unary(Negate(), Unary(Negate(), Constant(1)))."""  # noqa: E501
        assert self._parsed_exp(_MINUS, _MINUS, _number("1")) == Unary(Negate(), Unary(Negate(), Constant(1)))

    def test_negate_complement(self) -> None:
        """``return -~1;`` → Unary(Negate(), Unary(Complement(), Constant(1)))."""
        assert self._parsed_exp(_MINUS, _BITWISE_NOT, _number("1")) == Unary(Negate(), Unary(Complement(), Constant(1)))

    def test_parenthesised_constant(self) -> None:
        """``return (42);`` — parentheses around a constant are transparent."""
        assert self._parsed_exp(_LPAREN, _number("42"), _RPAREN) == Constant(42)


# ---------------------------------------------------------------------------
# Binary expressions (_parse_exp with precedence climbing)
# ---------------------------------------------------------------------------


class TestBinaryExpressions:
    """Parser correctly builds Binary nodes for all binary operators, respecting precedence and associativity."""

    def _parsed_exp(self, *exp_tokens: Token) -> Exp:
        """Parse a minimal function returning the given expression tokens and return the expression."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, *exp_tokens, _SEMICOLON, _RBRACE]
        result = parse(tokens)
        stmt = result.functions[0].body[0]
        assert isinstance(stmt, Return)
        return stmt.exp

    def test_addition(self) -> None:
        """``return 1 + 2;`` → Binary(Add(), Constant(1), Constant(2))."""
        assert self._parsed_exp(_number("1"), _PLUS, _number("2")) == Binary(Add(), Constant(1), Constant(2))

    def test_subtraction(self) -> None:
        """``return 3 - 1;`` → Binary(Subtract(), Constant(3), Constant(1))."""
        assert self._parsed_exp(_number("3"), _MINUS, _number("1")) == Binary(Subtract(), Constant(3), Constant(1))

    def test_multiplication(self) -> None:
        """``return 2 * 3;`` → Binary(Multiply(), Constant(2), Constant(3))."""
        assert self._parsed_exp(_number("2"), _TIMES, _number("3")) == Binary(Multiply(), Constant(2), Constant(3))

    def test_division(self) -> None:
        """``return 6 / 2;`` → Binary(Divide(), Constant(6), Constant(2))."""
        assert self._parsed_exp(_number("6"), _DIVIDE, _number("2")) == Binary(Divide(), Constant(6), Constant(2))

    def test_remainder(self) -> None:
        """``return 7 % 3;`` → Binary(Remainder(), Constant(7), Constant(3))."""
        assert self._parsed_exp(_number("7"), _PERCENT, _number("3")) == Binary(Remainder(), Constant(7), Constant(3))

    def test_multiply_binds_tighter_than_add(self) -> None:
        """``return 1 + 2 * 3;`` → Add(1, Multiply(2, 3)) by precedence."""
        assert self._parsed_exp(_number("1"), _PLUS, _number("2"), _TIMES, _number("3")) == Binary(
            Add(), Constant(1), Binary(Multiply(), Constant(2), Constant(3))
        )

    def test_multiply_binds_tighter_than_subtract(self) -> None:
        """``return 4 - 2 * 3;`` → Subtract(4, Multiply(2, 3)) by precedence."""
        assert self._parsed_exp(_number("4"), _MINUS, _number("2"), _TIMES, _number("3")) == Binary(
            Subtract(), Constant(4), Binary(Multiply(), Constant(2), Constant(3))
        )

    def test_add_is_left_associative(self) -> None:
        """``return 1 + 2 + 3;`` → Add(Add(1, 2), 3) by left-associativity."""
        assert self._parsed_exp(_number("1"), _PLUS, _number("2"), _PLUS, _number("3")) == Binary(
            Add(), Binary(Add(), Constant(1), Constant(2)), Constant(3)
        )

    def test_subtract_is_left_associative(self) -> None:
        """``return 5 - 3 - 1;`` → Subtract(Subtract(5, 3), 1) by left-associativity."""
        assert self._parsed_exp(_number("5"), _MINUS, _number("3"), _MINUS, _number("1")) == Binary(
            Subtract(), Binary(Subtract(), Constant(5), Constant(3)), Constant(1)
        )

    def test_divide_is_left_associative(self) -> None:
        """``return 12 / 4 / 3;`` → Divide(Divide(12, 4), 3) by left-associativity."""
        assert self._parsed_exp(_number("12"), _DIVIDE, _number("4"), _DIVIDE, _number("3")) == Binary(
            Divide(), Binary(Divide(), Constant(12), Constant(4)), Constant(3)
        )

    def test_parentheses_override_precedence(self) -> None:
        """``return (1 + 2) * 3;`` → Multiply(Add(1, 2), 3) via grouping."""
        assert self._parsed_exp(_LPAREN, _number("1"), _PLUS, _number("2"), _RPAREN, _TIMES, _number("3")) == Binary(
            Multiply(), Binary(Add(), Constant(1), Constant(2)), Constant(3)
        )

    def test_unary_negation_as_left_operand(self) -> None:
        """``return -1 + 2;`` → Add(Negate(1), 2)."""
        assert self._parsed_exp(_MINUS, _number("1"), _PLUS, _number("2")) == Binary(
            Add(), Unary(Negate(), Constant(1)), Constant(2)
        )

    def test_unary_negation_as_right_operand(self) -> None:
        """``return 2 + -1;`` → Add(2, Negate(1))."""
        assert self._parsed_exp(_number("2"), _PLUS, _MINUS, _number("1")) == Binary(
            Add(), Constant(2), Unary(Negate(), Constant(1))
        )


class TestBitwiseExpressions:
    """Parser correctly builds Binary nodes for bitwise and shift operators with correct precedence."""

    def _parsed_exp(self, *exp_tokens: Token) -> Exp:
        """Parse a minimal function returning the given expression tokens and return the expression."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, *exp_tokens, _SEMICOLON, _RBRACE]
        result = parse(tokens)
        stmt = result.functions[0].body[0]
        assert isinstance(stmt, Return)
        return stmt.exp

    def test_left_shift(self) -> None:
        """``return 1 << 2;`` → Binary(LeftShift(), Constant(1), Constant(2))."""
        assert self._parsed_exp(_number("1"), _LSHIFT, _number("2")) == Binary(LeftShift(), Constant(1), Constant(2))

    def test_right_shift(self) -> None:
        """``return 8 >> 2;`` → Binary(RightShift(), Constant(8), Constant(2))."""
        assert self._parsed_exp(_number("8"), _RSHIFT, _number("2")) == Binary(RightShift(), Constant(8), Constant(2))

    def test_bitwise_and(self) -> None:
        """``return 5 & 3;`` → Binary(BitwiseAnd(), Constant(5), Constant(3))."""
        assert self._parsed_exp(_number("5"), _AMPERSAND, _number("3")) == Binary(
            BitwiseAnd(), Constant(5), Constant(3)
        )

    def test_bitwise_xor(self) -> None:
        """``return 6 ^ 3;`` → Binary(BitwiseXor(), Constant(6), Constant(3))."""
        assert self._parsed_exp(_number("6"), _CARET, _number("3")) == Binary(BitwiseXor(), Constant(6), Constant(3))

    def test_bitwise_or(self) -> None:
        """``return 4 | 2;`` → Binary(BitwiseOr(), Constant(4), Constant(2))."""
        assert self._parsed_exp(_number("4"), _PIPE, _number("2")) == Binary(BitwiseOr(), Constant(4), Constant(2))

    def test_shift_binds_tighter_than_bitwise_and(self) -> None:
        """``return 1 & 2 << 3;`` → And(1, LeftShift(2, 3)): shift prec 40 > AND prec 25."""
        assert self._parsed_exp(_number("1"), _AMPERSAND, _number("2"), _LSHIFT, _number("3")) == Binary(
            BitwiseAnd(), Constant(1), Binary(LeftShift(), Constant(2), Constant(3))
        )

    def test_shift_binds_tighter_than_bitwise_or(self) -> None:
        """``return 1 | 2 << 3;`` → Or(1, LeftShift(2, 3)): shift prec 40 > OR prec 15."""
        assert self._parsed_exp(_number("1"), _PIPE, _number("2"), _LSHIFT, _number("3")) == Binary(
            BitwiseOr(), Constant(1), Binary(LeftShift(), Constant(2), Constant(3))
        )

    def test_add_binds_tighter_than_shift(self) -> None:
        """``return 1 + 2 << 3;`` → LeftShift(Add(1, 2), 3): additive prec 45 > shift prec 40."""
        assert self._parsed_exp(_number("1"), _PLUS, _number("2"), _LSHIFT, _number("3")) == Binary(
            LeftShift(), Binary(Add(), Constant(1), Constant(2)), Constant(3)
        )

    def test_bitwise_and_binds_tighter_than_xor(self) -> None:
        """``return 1 ^ 2 & 3;`` → Xor(1, And(2, 3)): AND prec 25 > XOR prec 20."""
        assert self._parsed_exp(_number("1"), _CARET, _number("2"), _AMPERSAND, _number("3")) == Binary(
            BitwiseXor(), Constant(1), Binary(BitwiseAnd(), Constant(2), Constant(3))
        )

    def test_bitwise_xor_binds_tighter_than_or(self) -> None:
        """``return 1 | 2 ^ 3;`` → Or(1, Xor(2, 3)): XOR prec 20 > OR prec 15."""
        assert self._parsed_exp(_number("1"), _PIPE, _number("2"), _CARET, _number("3")) == Binary(
            BitwiseOr(), Constant(1), Binary(BitwiseXor(), Constant(2), Constant(3))
        )

    def test_shift_is_left_associative(self) -> None:
        """``return 1 << 2 << 3;`` → LeftShift(LeftShift(1, 2), 3)."""
        assert self._parsed_exp(_number("1"), _LSHIFT, _number("2"), _LSHIFT, _number("3")) == Binary(
            LeftShift(), Binary(LeftShift(), Constant(1), Constant(2)), Constant(3)
        )

    def test_bitwise_or_is_left_associative(self) -> None:
        """``return 1 | 2 | 3;`` → Or(Or(1, 2), 3)."""
        assert self._parsed_exp(_number("1"), _PIPE, _number("2"), _PIPE, _number("3")) == Binary(
            BitwiseOr(), Binary(BitwiseOr(), Constant(1), Constant(2)), Constant(3)
        )

    def test_parentheses_override_shift_precedence(self) -> None:
        """``return 1 << (2 + 3);`` → LeftShift(1, Add(2, 3)): grouping overrides normal precedence."""
        assert self._parsed_exp(_number("1"), _LSHIFT, _LPAREN, _number("2"), _PLUS, _number("3"), _RPAREN) == Binary(
            LeftShift(), Constant(1), Binary(Add(), Constant(2), Constant(3))
        )
