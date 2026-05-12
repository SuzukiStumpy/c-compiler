"""Tests for the parsing stage.

Covers the full range of constructs supported by the current grammar:

    program             ::= function
    function            ::= "int" identifier "(" "void" ")" "{" block_item* "}"
    block_item          ::= statement | declaration
    block               ::= "{" {block_item} "}"
    declaration         ::= "int" identifier [ "=" exp ] ";"
    for_init            ::= declaration | [exp] ";"
    statement           ::= "return" exp ";"
                          | exp ";"
                          | "if" "(" exp ")" statement ["else" statement]
                          | "goto" identifier ";"
                          | identifier ":" statement
                          | block
                          | "break" ";"
                          | "continue" ";"
                          | "while" "(" exp ")" statement
                          | "do" statement "while" "(" exp ")" ";"
                          | "for" "(" for_init [exp] ";" [exp] ")" statement
                          | "switch" "(" exp ")" statement
                          | "case" exp ":" statement
                          | "default" ":" statement
                          | ";"
    exp                 ::= factor | exp binop exp | exp "=" exp | exp assignop exp
    factor              ::= prefix_unop factor | primary postfix_unop?
    primary             ::= int | identifier | "(" exp ")"
    prefix_unop         ::= "-" | "~" | "!" | "++" | "--"
    postfix_unop        ::= "++" | "--"
    binop               ::= "+" | "-" | "*" | "/" | "%" | "<<" | ">>" | "&" | "^" | "|"
                          | "&&" | "||" | "==" | "!=" | "<" | "<=" | ">" | ">="
    assignop            ::= "+=" | "-=" | "*=" | "/=" | "%="
                          | "&=" | "|=" | "^=" | "<<=" | ">>="
    identifier          ::= ? An identifier token ?
    int                 ::= ? A constant token ?

Each test builds a token stream directly (without invoking the lexer) so that
parser behaviour is tested in isolation from lexer behaviour.
"""

import pytest

from compiler.parser import parse
from compiler.syntax.ast import (
    Block,
    BlockItem,
    Break,
    Case,
    Compound,
    Constant,
    Continue,
    D,
    Declaration,
    Default,
    DoWhile,
    Expression,
    For,
    Function,
    Goto,
    If,
    InitDecl,
    InitExp,
    IntType,
    Label,
    Null,
    Parameter,
    Program,
    Return,
    S,
    Statement,
    Switch,
    While,
)
from compiler.syntax.ast.expressions import Assignment, Binary, Conditional, Exp, Unary, Var
from compiler.syntax.ast.operators import (
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


def _ident(name: str) -> Token:
    """Return an IDENTIFIER token with the given name."""
    return _tok(TokenType.IDENTIFIER, name)


def _number(value: str) -> Token:
    """Return a NUMBER token with the given literal text."""
    return _tok(TokenType.NUMBER, value)


def _parsed_body(*body_tokens: Token) -> tuple[BlockItem, ...]:
    """Parse a minimal function from the given body tokens and return the body tuple.

    Args:
        body_tokens: Token sequence that forms the function body (between braces).

    Returns:
        The ``body`` tuple of ``BlockItem`` nodes from the parsed function.
    """
    tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, *body_tokens, _RBRACE]
    return parse(tokens).functions[0].body


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
    item = result.functions[0].body[0]
    assert isinstance(item, S)
    assert isinstance(item.statement, Return)
    return item.statement.exp


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
                    body=(S(statement=Return(exp=Constant(value=0))),),
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
        assert result.functions[0].body == (S(statement=Return(exp=Constant(value=42))),)

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
            S(statement=Return(exp=Constant(value=1))),
            S(statement=Return(exp=Constant(value=2))),
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
        """``int main(void) {}`` — empty body is valid; produces an empty block-item tuple."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RBRACE]
        result = parse(tokens)
        assert result.functions[0].body == ()

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
        """``return );`` — RPAREN is not a valid expression start."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _RPAREN, _SEMICOLON, _RBRACE]
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

    def test_negate(self) -> None:
        """``return -1;`` → Unary(Negate(), Constant(1))."""
        assert _parsed_exp(_MINUS, _number("1")) == Unary(Negate(), Constant(1))

    def test_complement(self) -> None:
        """``return ~1;`` → Unary(Complement(), Constant(1))."""
        assert _parsed_exp(_BITWISE_NOT, _number("1")) == Unary(Complement(), Constant(1))

    def test_double_negate(self) -> None:
        """``return --1;`` — wait, DECREMENT is a separate token; ``- -1`` → Unary(Negate(), Unary(Negate(), Constant(1)))."""  # noqa: E501
        assert _parsed_exp(_MINUS, _MINUS, _number("1")) == Unary(Negate(), Unary(Negate(), Constant(1)))

    def test_negate_complement(self) -> None:
        """``return -~1;`` → Unary(Negate(), Unary(Complement(), Constant(1)))."""
        assert _parsed_exp(_MINUS, _BITWISE_NOT, _number("1")) == Unary(Negate(), Unary(Complement(), Constant(1)))

    def test_logical_not(self) -> None:
        """``return !1;`` → Unary(Not(), Constant(1))."""
        assert _parsed_exp(_NOT, _number("1")) == Unary(Not(), Constant(1))

    def test_parenthesised_constant(self) -> None:
        """``return (42);`` — parentheses around a constant are transparent."""
        assert _parsed_exp(_LPAREN, _number("42"), _RPAREN) == Constant(42)


# ---------------------------------------------------------------------------
# Binary expressions (_parse_exp with precedence climbing)
# ---------------------------------------------------------------------------


class TestBinaryExpressions:
    """Parser correctly builds Binary nodes for all binary operators, respecting precedence and associativity."""

    @pytest.mark.parametrize(
        ("tok", "op"),
        [
            (_PLUS, Add()),
            (_MINUS, Subtract()),
            (_TIMES, Multiply()),
            (_DIVIDE, Divide()),
            (_PERCENT, Remainder()),
        ],
        ids=["add", "subtract", "multiply", "divide", "remainder"],
    )
    def test_operator_parses(self, tok: Token, op: object) -> None:
        """``return 1 <op> 2;`` → Binary(<op>, Constant(1), Constant(2))."""
        assert _parsed_exp(_number("1"), tok, _number("2")) == Binary(op, Constant(1), Constant(2))

    def test_multiply_binds_tighter_than_add(self) -> None:
        """``return 1 + 2 * 3;`` → Add(1, Multiply(2, 3)) by precedence."""
        assert _parsed_exp(_number("1"), _PLUS, _number("2"), _TIMES, _number("3")) == Binary(
            Add(), Constant(1), Binary(Multiply(), Constant(2), Constant(3))
        )

    def test_multiply_binds_tighter_than_subtract(self) -> None:
        """``return 4 - 2 * 3;`` → Subtract(4, Multiply(2, 3)) by precedence."""
        assert _parsed_exp(_number("4"), _MINUS, _number("2"), _TIMES, _number("3")) == Binary(
            Subtract(), Constant(4), Binary(Multiply(), Constant(2), Constant(3))
        )

    def test_add_is_left_associative(self) -> None:
        """``return 1 + 2 + 3;`` → Add(Add(1, 2), 3) by left-associativity."""
        assert _parsed_exp(_number("1"), _PLUS, _number("2"), _PLUS, _number("3")) == Binary(
            Add(), Binary(Add(), Constant(1), Constant(2)), Constant(3)
        )

    def test_subtract_is_left_associative(self) -> None:
        """``return 5 - 3 - 1;`` → Subtract(Subtract(5, 3), 1) by left-associativity."""
        assert _parsed_exp(_number("5"), _MINUS, _number("3"), _MINUS, _number("1")) == Binary(
            Subtract(), Binary(Subtract(), Constant(5), Constant(3)), Constant(1)
        )

    def test_divide_is_left_associative(self) -> None:
        """``return 12 / 4 / 3;`` → Divide(Divide(12, 4), 3) by left-associativity."""
        assert _parsed_exp(_number("12"), _DIVIDE, _number("4"), _DIVIDE, _number("3")) == Binary(
            Divide(), Binary(Divide(), Constant(12), Constant(4)), Constant(3)
        )

    def test_parentheses_override_precedence(self) -> None:
        """``return (1 + 2) * 3;`` → Multiply(Add(1, 2), 3) via grouping."""
        assert _parsed_exp(_LPAREN, _number("1"), _PLUS, _number("2"), _RPAREN, _TIMES, _number("3")) == Binary(
            Multiply(), Binary(Add(), Constant(1), Constant(2)), Constant(3)
        )

    def test_unary_negation_as_left_operand(self) -> None:
        """``return -1 + 2;`` → Add(Negate(1), 2)."""
        assert _parsed_exp(_MINUS, _number("1"), _PLUS, _number("2")) == Binary(
            Add(), Unary(Negate(), Constant(1)), Constant(2)
        )

    def test_unary_negation_as_right_operand(self) -> None:
        """``return 2 + -1;`` → Add(2, Negate(1))."""
        assert _parsed_exp(_number("2"), _PLUS, _MINUS, _number("1")) == Binary(
            Add(), Constant(2), Unary(Negate(), Constant(1))
        )


class TestBitwiseExpressions:
    """Parser correctly builds Binary nodes for bitwise and shift operators with correct precedence."""

    @pytest.mark.parametrize(
        ("tok", "op"),
        [
            (_LSHIFT, LeftShift()),
            (_RSHIFT, RightShift()),
            (_AMPERSAND, BitwiseAnd()),
            (_CARET, BitwiseXor()),
            (_PIPE, BitwiseOr()),
        ],
        ids=["left_shift", "right_shift", "bitwise_and", "bitwise_xor", "bitwise_or"],
    )
    def test_operator_parses(self, tok: Token, op: object) -> None:
        """``return 1 <op> 2;`` → Binary(<op>, Constant(1), Constant(2))."""
        assert _parsed_exp(_number("1"), tok, _number("2")) == Binary(op, Constant(1), Constant(2))

    def test_shift_binds_tighter_than_bitwise_and(self) -> None:
        """``return 1 & 2 << 3;`` → And(1, LeftShift(2, 3)): shift prec 40 > AND prec 25."""
        assert _parsed_exp(_number("1"), _AMPERSAND, _number("2"), _LSHIFT, _number("3")) == Binary(
            BitwiseAnd(), Constant(1), Binary(LeftShift(), Constant(2), Constant(3))
        )

    def test_shift_binds_tighter_than_bitwise_or(self) -> None:
        """``return 1 | 2 << 3;`` → Or(1, LeftShift(2, 3)): shift prec 40 > OR prec 15."""
        assert _parsed_exp(_number("1"), _PIPE, _number("2"), _LSHIFT, _number("3")) == Binary(
            BitwiseOr(), Constant(1), Binary(LeftShift(), Constant(2), Constant(3))
        )

    def test_add_binds_tighter_than_shift(self) -> None:
        """``return 1 + 2 << 3;`` → LeftShift(Add(1, 2), 3): additive prec 45 > shift prec 40."""
        assert _parsed_exp(_number("1"), _PLUS, _number("2"), _LSHIFT, _number("3")) == Binary(
            LeftShift(), Binary(Add(), Constant(1), Constant(2)), Constant(3)
        )

    def test_bitwise_and_binds_tighter_than_xor(self) -> None:
        """``return 1 ^ 2 & 3;`` → Xor(1, And(2, 3)): AND prec 25 > XOR prec 20."""
        assert _parsed_exp(_number("1"), _CARET, _number("2"), _AMPERSAND, _number("3")) == Binary(
            BitwiseXor(), Constant(1), Binary(BitwiseAnd(), Constant(2), Constant(3))
        )

    def test_bitwise_xor_binds_tighter_than_or(self) -> None:
        """``return 1 | 2 ^ 3;`` → Or(1, Xor(2, 3)): XOR prec 20 > OR prec 15."""
        assert _parsed_exp(_number("1"), _PIPE, _number("2"), _CARET, _number("3")) == Binary(
            BitwiseOr(), Constant(1), Binary(BitwiseXor(), Constant(2), Constant(3))
        )

    def test_shift_is_left_associative(self) -> None:
        """``return 1 << 2 << 3;`` → LeftShift(LeftShift(1, 2), 3)."""
        assert _parsed_exp(_number("1"), _LSHIFT, _number("2"), _LSHIFT, _number("3")) == Binary(
            LeftShift(), Binary(LeftShift(), Constant(1), Constant(2)), Constant(3)
        )

    def test_bitwise_or_is_left_associative(self) -> None:
        """``return 1 | 2 | 3;`` → Or(Or(1, 2), 3)."""
        assert _parsed_exp(_number("1"), _PIPE, _number("2"), _PIPE, _number("3")) == Binary(
            BitwiseOr(), Binary(BitwiseOr(), Constant(1), Constant(2)), Constant(3)
        )

    def test_parentheses_override_shift_precedence(self) -> None:
        """``return 1 << (2 + 3);`` → LeftShift(1, Add(2, 3)): grouping overrides normal precedence."""
        assert _parsed_exp(_number("1"), _LSHIFT, _LPAREN, _number("2"), _PLUS, _number("3"), _RPAREN) == Binary(
            LeftShift(), Constant(1), Binary(Add(), Constant(2), Constant(3))
        )


# ---------------------------------------------------------------------------
# Comparison expressions
# ---------------------------------------------------------------------------


class TestComparisonExpressions:
    """Parser correctly builds Binary nodes for all comparison operators."""

    @pytest.mark.parametrize(
        ("tok", "op"),
        [
            (_EQ, Equal()),
            (_NEQ, NotEqual()),
            (_LT, LessThan()),
            (_LEQ, LessOrEqual()),
            (_GT, GreaterThan()),
            (_GEQ, GreaterOrEqual()),
        ],
    )
    def test_operator_parses(self, tok: Token, op: object) -> None:
        """Each comparison token produces the correct AST operator node."""
        assert _parsed_exp(_number("1"), tok, _number("2")) == Binary(op, Constant(1), Constant(2))

    def test_comparison_binds_tighter_than_logical_and(self) -> None:
        """``1 && 2 == 3`` → ``And(1, Equal(2, 3))``: comparison prec 30 > logical AND prec 10."""
        assert _parsed_exp(_number("1"), _AND, _number("2"), _EQ, _number("3")) == Binary(
            And(), Constant(1), Binary(Equal(), Constant(2), Constant(3))
        )

    def test_comparison_binds_tighter_than_bitwise_or(self) -> None:
        """``1 == 2 | 3`` → ``BitwiseOr(Equal(1, 2), 3)``: comparison prec 30 > bitwise OR prec 15."""
        assert _parsed_exp(_number("1"), _EQ, _number("2"), _PIPE, _number("3")) == Binary(
            BitwiseOr(), Binary(Equal(), Constant(1), Constant(2)), Constant(3)
        )

    def test_comparison_is_left_associative(self) -> None:
        """``1 < 2 < 3`` → ``LessThan(LessThan(1, 2), 3)``."""
        assert _parsed_exp(_number("1"), _LT, _number("2"), _LT, _number("3")) == Binary(
            LessThan(), Binary(LessThan(), Constant(1), Constant(2)), Constant(3)
        )


# ---------------------------------------------------------------------------
# Logical AND / OR expressions
# ---------------------------------------------------------------------------


class TestLogicalExpressions:
    """Parser correctly builds Binary nodes for ``&&`` and ``||``."""

    def test_logical_and_parses(self) -> None:
        """``return 1 && 2;`` → ``Binary(And(), Constant(1), Constant(2))``."""
        assert _parsed_exp(_number("1"), _AND, _number("2")) == Binary(And(), Constant(1), Constant(2))

    def test_logical_or_parses(self) -> None:
        """``return 1 || 2;`` → ``Binary(Or(), Constant(1), Constant(2))``."""
        assert _parsed_exp(_number("1"), _OR, _number("2")) == Binary(Or(), Constant(1), Constant(2))

    def test_logical_and_binds_tighter_than_or(self) -> None:
        """``1 || 2 && 3`` → ``Or(1, And(2, 3))``: AND prec 10 > OR prec 5."""
        assert _parsed_exp(_number("1"), _OR, _number("2"), _AND, _number("3")) == Binary(
            Or(), Constant(1), Binary(And(), Constant(2), Constant(3))
        )

    def test_bitwise_or_binds_tighter_than_logical_and(self) -> None:
        """``1 && 2 | 3`` → ``And(1, BitwiseOr(2, 3))``: bitwise OR prec 15 > logical AND prec 10."""
        assert _parsed_exp(_number("1"), _AND, _number("2"), _PIPE, _number("3")) == Binary(
            And(), Constant(1), Binary(BitwiseOr(), Constant(2), Constant(3))
        )

    def test_logical_and_is_left_associative(self) -> None:
        """``1 && 2 && 3`` → ``And(And(1, 2), 3)``."""
        assert _parsed_exp(_number("1"), _AND, _number("2"), _AND, _number("3")) == Binary(
            And(), Binary(And(), Constant(1), Constant(2)), Constant(3)
        )

    def test_logical_or_is_left_associative(self) -> None:
        """``1 || 2 || 3`` → ``Or(Or(1, 2), 3)``."""
        assert _parsed_exp(_number("1"), _OR, _number("2"), _OR, _number("3")) == Binary(
            Or(), Binary(Or(), Constant(1), Constant(2)), Constant(3)
        )


# ---------------------------------------------------------------------------
# Prefix and postfix increment / decrement (_parse_factor)
# ---------------------------------------------------------------------------


class TestIncDecExpressions:
    """Parser correctly builds Unary nodes for prefix and postfix ++/-- operators."""

    def test_prefix_increment(self) -> None:
        """``return ++a;`` → ``Unary(PreIncrement(), Var("a"))``."""
        assert _parsed_exp(_INCREMENT, _ident("a")) == Unary(PreIncrement(), Var("a"))

    def test_prefix_decrement(self) -> None:
        """``return --a;`` → ``Unary(PreDecrement(), Var("a"))``."""
        assert _parsed_exp(_DECREMENT, _ident("a")) == Unary(PreDecrement(), Var("a"))

    def test_postfix_increment(self) -> None:
        """``return a++;`` → ``Unary(PostIncrement(), Var("a"))``."""
        assert _parsed_exp(_ident("a"), _INCREMENT) == Unary(PostIncrement(), Var("a"))

    def test_postfix_decrement(self) -> None:
        """``return a--;`` → ``Unary(PostDecrement(), Var("a"))``."""
        assert _parsed_exp(_ident("a"), _DECREMENT) == Unary(PostDecrement(), Var("a"))

    def test_chained_postfix_increment_then_decrement(self) -> None:
        """``return a++--;`` → ``Unary(PostDecrement(), Unary(PostIncrement(), Var("a")))``."""
        assert _parsed_exp(_ident("a"), _INCREMENT, _DECREMENT) == Unary(
            PostDecrement(), Unary(PostIncrement(), Var("a"))
        )

    def test_postfix_binds_tighter_than_prefix(self) -> None:
        """``return ++a++;`` → ``Unary(PreIncrement(), Unary(PostIncrement(), Var("a")))``.

        Postfix is parsed as part of the primary before prefix wraps it.
        """
        assert _parsed_exp(_INCREMENT, _ident("a"), _INCREMENT) == Unary(
            PreIncrement(), Unary(PostIncrement(), Var("a"))
        )


# ---------------------------------------------------------------------------
# If / else statements (_parse_statement → IF case)
# ---------------------------------------------------------------------------


class TestIfStatements:
    """Parser correctly builds ``If`` nodes for if/else statements."""

    def test_if_without_else(self) -> None:
        """``if (1) return 0;`` → ``If(condition=Constant(1), then=Return(Constant(0)), else_=None)``."""
        result = _parsed_stmt(_IF, _LPAREN, _number("1"), _RPAREN, _RETURN, _number("0"), _SEMICOLON)
        assert result == If(condition=Constant(1), then=Return(exp=Constant(0)), else_=None)

    def test_if_with_else(self) -> None:
        """``if (1) return 1; else return 0;`` → ``If`` with both branches populated."""
        result = _parsed_stmt(
            _IF,
            _LPAREN,
            _number("1"),
            _RPAREN,
            _RETURN,
            _number("1"),
            _SEMICOLON,
            _ELSE,
            _RETURN,
            _number("0"),
            _SEMICOLON,
        )
        assert result == If(
            condition=Constant(1),
            then=Return(exp=Constant(1)),
            else_=Return(exp=Constant(0)),
        )

    def test_if_null_then_clause(self) -> None:
        """``if (1);`` → ``If`` whose ``then`` is a ``Null`` statement."""
        result = _parsed_stmt(_IF, _LPAREN, _number("1"), _RPAREN, _SEMICOLON)
        assert result == If(condition=Constant(1), then=Null(), else_=None)

    def test_if_null_else_clause(self) -> None:
        """``if (1) return 1; else;`` → ``If`` whose ``else_`` is a ``Null`` statement."""
        result = _parsed_stmt(
            _IF,
            _LPAREN,
            _number("1"),
            _RPAREN,
            _RETURN,
            _number("1"),
            _SEMICOLON,
            _ELSE,
            _SEMICOLON,
        )
        assert result == If(
            condition=Constant(1),
            then=Return(exp=Constant(1)),
            else_=Null(),
        )

    def test_if_without_else_does_not_consume_next_statement(self) -> None:
        """``if (1) return 1; return 0;`` — the ``return 0`` is not swallowed by the if."""
        body = _parsed_body(
            _IF,
            _LPAREN,
            _number("1"),
            _RPAREN,
            _RETURN,
            _number("1"),
            _SEMICOLON,
            _RETURN,
            _number("0"),
            _SEMICOLON,
        )
        assert len(body) == 2
        item = body[1]
        assert isinstance(item, S)
        assert item.statement == Return(exp=Constant(0))

    def test_dangling_else_binds_to_nearest_if(self) -> None:
        """``if (1) if (2) return 2; else return 1;`` — else belongs to the inner if (C rule)."""
        result = _parsed_stmt(
            _IF,
            _LPAREN,
            _number("1"),
            _RPAREN,
            _IF,
            _LPAREN,
            _number("2"),
            _RPAREN,
            _RETURN,
            _number("2"),
            _SEMICOLON,
            _ELSE,
            _RETURN,
            _number("1"),
            _SEMICOLON,
        )
        assert result == If(
            condition=Constant(1),
            then=If(
                condition=Constant(2),
                then=Return(exp=Constant(2)),
                else_=Return(exp=Constant(1)),
            ),
            else_=None,
        )

    def test_if_condition_is_arbitrary_expression(self) -> None:
        """``if (a > 1) return a;`` — comparison expression as condition."""
        result = _parsed_stmt(
            _IF,
            _LPAREN,
            _ident("a"),
            _GT,
            _number("1"),
            _RPAREN,
            _RETURN,
            _ident("a"),
            _SEMICOLON,
        )
        assert result == If(
            condition=Binary(GreaterThan(), Var("a"), Constant(1)),
            then=Return(exp=Var("a")),
            else_=None,
        )

    def test_if_then_is_expression_statement(self) -> None:
        """``if (1) a;`` — expression statement (not return) in the then branch."""
        result = _parsed_stmt(_IF, _LPAREN, _number("1"), _RPAREN, _ident("a"), _SEMICOLON)
        assert result == If(condition=Constant(1), then=Expression(exp=Var("a")), else_=None)

    def test_if_missing_lparen(self) -> None:
        """``if 1) return 0;`` — no opening parenthesis raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _IF,
            _number("1"),
            _RPAREN,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected LPAREN"):
            parse(tokens)

    def test_if_missing_rparen(self) -> None:
        """``if (1 return 0;`` — no closing parenthesis raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _IF,
            _LPAREN,
            _number("1"),
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected RPAREN"):
            parse(tokens)


# ---------------------------------------------------------------------------
# Ternary / conditional expressions (_parse_exp → QUERY case)
# ---------------------------------------------------------------------------


class TestTernaryExpressions:
    """Parser correctly builds ``Conditional`` nodes for ternary ``?:`` expressions."""

    def test_basic_ternary(self) -> None:
        """``return 1 ? 2 : 3;`` → ``Conditional(Constant(1), Constant(2), Constant(3))``."""
        assert _parsed_exp(_number("1"), _QUERY, _number("2"), _COLON, _number("3")) == Conditional(
            condition=Constant(1), exp_true=Constant(2), exp_false=Constant(3)
        )

    def test_condition_is_arbitrary_expression(self) -> None:
        """``return a > 0 ? 1 : 0;`` — comparison expression as condition."""
        assert _parsed_exp(_ident("a"), _GT, _number("0"), _QUERY, _number("1"), _COLON, _number("0")) == Conditional(
            condition=Binary(GreaterThan(), Var("a"), Constant(0)),
            exp_true=Constant(1),
            exp_false=Constant(0),
        )

    def test_ternary_is_right_associative(self) -> None:
        """``return 1 ? 2 : 3 ? 4 : 5;`` — else-branch contains a nested ternary, not the reverse."""
        assert _parsed_exp(
            _number("1"),
            _QUERY,
            _number("2"),
            _COLON,
            _number("3"),
            _QUERY,
            _number("4"),
            _COLON,
            _number("5"),
        ) == Conditional(
            condition=Constant(1),
            exp_true=Constant(2),
            exp_false=Conditional(condition=Constant(3), exp_true=Constant(4), exp_false=Constant(5)),
        )

    def test_ternary_is_rhs_of_assignment(self) -> None:
        """``return a = 1 ? 2 : 3;`` — ternary (prec 3) binds tighter than assignment (prec 1)."""
        assert _parsed_exp(
            _ident("a"), _ASSIGNMENT, _number("1"), _QUERY, _number("2"), _COLON, _number("3")
        ) == Assignment(
            lvalue=Var("a"),
            rvalue=Conditional(condition=Constant(1), exp_true=Constant(2), exp_false=Constant(3)),
        )

    def test_ternary_binds_looser_than_logical_or(self) -> None:
        """``return 1 || 2 ? 3 : 4;`` — ``||`` (prec 5) groups before ``?:`` (prec 3)."""
        assert _parsed_exp(_number("1"), _OR, _number("2"), _QUERY, _number("3"), _COLON, _number("4")) == Conditional(
            condition=Binary(Or(), Constant(1), Constant(2)),
            exp_true=Constant(3),
            exp_false=Constant(4),
        )

    def test_true_branch_allows_assignment(self) -> None:
        """``return 1 ? a = 2 : 3;`` — middle is parsed at prec 0, so assignment is valid."""
        assert _parsed_exp(
            _number("1"), _QUERY, _ident("a"), _ASSIGNMENT, _number("2"), _COLON, _number("3")
        ) == Conditional(
            condition=Constant(1),
            exp_true=Assignment(lvalue=Var("a"), rvalue=Constant(2)),
            exp_false=Constant(3),
        )

    def test_missing_colon(self) -> None:
        """``return 1 ? 2;`` — absent ``:`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _RETURN,
            _number("1"),
            _QUERY,
            _number("2"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected COLON"):
            parse(tokens)


# ---------------------------------------------------------------------------
# Goto statements (_parse_statement → GOTO case)
# ---------------------------------------------------------------------------


class TestGotoStatements:
    """Parser correctly builds ``Goto`` nodes for goto statements."""

    def test_goto_produces_goto_node(self) -> None:
        """``goto foo;`` → ``Goto(target="foo")``."""
        assert _parsed_stmt(_GOTO, _ident("foo"), _SEMICOLON) == Goto(target="foo")

    def test_goto_target_name_is_preserved(self) -> None:
        """The label name is stored verbatim in ``Goto.target``."""
        result = _parsed_stmt(_GOTO, _ident("my_label"), _SEMICOLON)
        assert isinstance(result, Goto)
        assert result.target == "my_label"

    def test_goto_missing_semicolon_raises_parse_error(self) -> None:
        """``goto foo`` without a trailing semicolon raises ``ParseError``."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _GOTO, _ident("foo"), _RBRACE]
        with pytest.raises(ParseError, match="expected SEMICOLON"):
            parse(tokens)

    def test_goto_missing_identifier_raises_parse_error(self) -> None:
        """``goto;`` without a label name raises ``ParseError``."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _GOTO, _SEMICOLON, _RBRACE]
        with pytest.raises(ParseError, match="expected IDENTIFIER"):
            parse(tokens)


# ---------------------------------------------------------------------------
# Labeled statements (_parse_statement → label lookahead)
# ---------------------------------------------------------------------------


class TestLabeledStatements:
    """Parser correctly builds ``Label`` nodes for labeled statements."""

    def test_label_wraps_return_statement(self) -> None:
        """``foo: return 0;`` → ``Label(name="foo", statement=Return(Constant(0)))``."""
        assert _parsed_stmt(_ident("foo"), _COLON, _RETURN, _number("0"), _SEMICOLON) == Label(
            name="foo", statement=Return(exp=Constant(0))
        )

    def test_label_wraps_null_statement(self) -> None:
        """``foo:;`` → ``Label(name="foo", statement=Null())``."""
        assert _parsed_stmt(_ident("foo"), _COLON, _SEMICOLON) == Label(name="foo", statement=Null())

    def test_label_wraps_expression_statement(self) -> None:
        """``foo: a;`` → ``Label(name="foo", statement=Expression(Var("a")))``."""
        assert _parsed_stmt(_ident("foo"), _COLON, _ident("a"), _SEMICOLON) == Label(
            name="foo", statement=Expression(exp=Var("a"))
        )

    def test_nested_labels(self) -> None:
        """``foo: bar: return 0;`` → ``Label("foo", Label("bar", Return(Constant(0))))``."""
        assert _parsed_stmt(_ident("foo"), _COLON, _ident("bar"), _COLON, _RETURN, _number("0"), _SEMICOLON) == Label(
            name="foo", statement=Label(name="bar", statement=Return(exp=Constant(0)))
        )

    def test_label_target_name_is_preserved(self) -> None:
        """The label identifier is stored verbatim in ``Label.name``."""
        result = _parsed_stmt(_ident("my_label"), _COLON, _SEMICOLON)
        assert isinstance(result, Label)
        assert result.name == "my_label"

    def test_identifier_without_colon_is_expression_statement(self) -> None:
        """``foo;`` — a bare identifier not followed by ``:`` is an expression statement."""
        assert _parsed_stmt(_ident("foo"), _SEMICOLON) == Expression(exp=Var("foo"))


# ---------------------------------------------------------------------------
# Compound (block) statements (_parse_statement → LBRACE case)
# ---------------------------------------------------------------------------


class TestCompoundStatements:
    """Parser correctly builds ``Compound`` nodes for ``{ block_item* }`` statements."""

    def test_empty_compound(self) -> None:
        """``{}`` as a statement → ``Compound(body=())``."""
        assert _parsed_stmt(_LBRACE, _RBRACE) == Compound(body=())

    def test_compound_with_single_return(self) -> None:
        """``{ return 0; }`` → ``Compound`` wrapping a single ``Return``."""
        assert _parsed_stmt(_LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE) == Compound(
            body=(S(statement=Return(exp=Constant(0))),)
        )

    def test_compound_with_declaration(self) -> None:
        """``{ int x; }`` → ``Compound`` whose body contains a single ``D`` item."""
        assert _parsed_stmt(_LBRACE, _INT, _ident("x"), _SEMICOLON, _RBRACE) == Compound(
            body=(D(declaration=Declaration(name="x", init=None)),)
        )

    def test_compound_with_initialised_declaration(self) -> None:
        """``{ int x = 1; }`` → ``Compound`` with a ``D`` whose declaration carries an initialiser."""
        assert _parsed_stmt(_LBRACE, _INT, _ident("x"), _ASSIGNMENT, _number("1"), _SEMICOLON, _RBRACE) == Compound(
            body=(D(declaration=Declaration(name="x", init=Constant(1))),)
        )

    def test_compound_with_multiple_items(self) -> None:
        """``{ int x = 1; return x; }`` → body contains a declaration then a statement."""
        assert _parsed_stmt(
            _LBRACE,
            _INT,
            _ident("x"),
            _ASSIGNMENT,
            _number("1"),
            _SEMICOLON,
            _RETURN,
            _ident("x"),
            _SEMICOLON,
            _RBRACE,
        ) == Compound(
            body=(
                D(declaration=Declaration(name="x", init=Constant(1))),
                S(statement=Return(exp=Var("x"))),
            )
        )

    def test_nested_compound(self) -> None:
        """``{ { return 0; } }`` → outer ``Compound`` wrapping an inner ``Compound``."""
        assert _parsed_stmt(_LBRACE, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE, _RBRACE) == Compound(
            body=(S(statement=Compound(body=(S(statement=Return(exp=Constant(0))),))),)
        )

    def test_compound_does_not_consume_subsequent_block_items(self) -> None:
        """``{ {} return 0; }`` — the compound's ``}`` does not swallow the following ``return``."""
        body = _parsed_body(_LBRACE, _RBRACE, _RETURN, _number("0"), _SEMICOLON)
        assert body == (S(statement=Compound(body=())), S(statement=Return(exp=Constant(0))))

    def test_compound_as_if_then_branch(self) -> None:
        """``if (1) { return 0; }`` → ``If`` with a ``Compound`` as the then-branch."""
        result = _parsed_stmt(
            _IF,
            _LPAREN,
            _number("1"),
            _RPAREN,
            _LBRACE,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        )
        assert result == If(
            condition=Constant(1),
            then=Compound(body=(S(statement=Return(exp=Constant(0))),)),
            else_=None,
        )

    def test_compound_as_else_branch(self) -> None:
        """``if (1) return 1; else { return 0; }`` → ``If`` with a ``Compound`` else-branch."""
        result = _parsed_stmt(
            _IF,
            _LPAREN,
            _number("1"),
            _RPAREN,
            _RETURN,
            _number("1"),
            _SEMICOLON,
            _ELSE,
            _LBRACE,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        )
        assert result == If(
            condition=Constant(1),
            then=Return(exp=Constant(1)),
            else_=Compound(body=(S(statement=Return(exp=Constant(0))),)),
        )

    def test_label_wraps_compound(self) -> None:
        """``foo: { return 0; }`` → ``Label`` whose governed statement is a ``Compound``."""
        result = _parsed_stmt(_ident("foo"), _COLON, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE)
        assert result == Label(
            name="foo",
            statement=Compound(body=(S(statement=Return(exp=Constant(0))),)),
        )

    def test_compound_missing_closing_brace_raises_parse_error(self) -> None:
        """``int main(void) { { return 0; }`` — missing outer ``}`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _LBRACE,
            _RETURN,
            _number("0"),
            _SEMICOLON,
        ]
        with pytest.raises(ParseError):
            parse(tokens)

    def test_compound_body_is_block_type(self) -> None:
        """``Compound.body`` is an instance of ``Block`` (i.e., ``tuple[BlockItem, ...]``)."""
        result = _parsed_stmt(_LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE)
        assert isinstance(result, Compound)
        assert isinstance(result.body, tuple)
        # Block is a type alias — verify the runtime type satisfies it
        b: Block = result.body
        assert b == (S(statement=Return(exp=Constant(0))),)


# ---------------------------------------------------------------------------
# Break and continue statements (_parse_statement → BREAK / CONTINUE cases)
# ---------------------------------------------------------------------------


class TestBreakContinueStatements:
    """Parser correctly builds ``Break`` and ``Continue`` nodes."""

    def test_break_produces_break_node(self) -> None:
        """``break;`` → ``Break()``."""
        assert _parsed_stmt(_BREAK, _SEMICOLON) == Break()

    def test_continue_produces_continue_node(self) -> None:
        """``continue;`` → ``Continue()``."""
        assert _parsed_stmt(_CONTINUE, _SEMICOLON) == Continue()

    def test_break_missing_semicolon_raises_parse_error(self) -> None:
        """``break`` without a trailing semicolon raises ``ParseError``."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _BREAK, _RBRACE]
        with pytest.raises(ParseError, match="expected SEMICOLON"):
            parse(tokens)

    def test_continue_missing_semicolon_raises_parse_error(self) -> None:
        """``continue`` without a trailing semicolon raises ``ParseError``."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _CONTINUE, _RBRACE]
        with pytest.raises(ParseError, match="expected SEMICOLON"):
            parse(tokens)


# ---------------------------------------------------------------------------
# While statements (_parse_statement → WHILE case)
# ---------------------------------------------------------------------------


class TestWhileStatements:
    """Parser correctly builds ``While`` nodes for while loops."""

    def test_while_with_constant_condition(self) -> None:
        """``while (1) return 0;`` → ``While(condition=Constant(1), body=Return(Constant(0)))``."""
        result = _parsed_stmt(_WHILE, _LPAREN, _number("1"), _RPAREN, _RETURN, _number("0"), _SEMICOLON)
        assert result == While(condition=Constant(1), body=Return(exp=Constant(0)))

    def test_while_with_compound_body(self) -> None:
        """``while (1) {}`` → ``While`` with an empty ``Compound`` body."""
        result = _parsed_stmt(_WHILE, _LPAREN, _number("1"), _RPAREN, _LBRACE, _RBRACE)
        assert result == While(condition=Constant(1), body=Compound(body=()))

    def test_while_body_is_null_statement(self) -> None:
        """``while (1) ;`` → ``While`` whose body is a ``Null`` statement."""
        result = _parsed_stmt(_WHILE, _LPAREN, _number("1"), _RPAREN, _SEMICOLON)
        assert result == While(condition=Constant(1), body=Null())

    def test_while_condition_is_arbitrary_expression(self) -> None:
        """``while (a < 10) ;`` — comparison expression as condition."""
        result = _parsed_stmt(_WHILE, _LPAREN, _ident("a"), _LT, _number("10"), _RPAREN, _SEMICOLON)
        assert result == While(condition=Binary(LessThan(), Var("a"), Constant(10)), body=Null())

    def test_while_missing_lparen_raises_parse_error(self) -> None:
        """``while 1) ;`` — missing opening parenthesis raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _WHILE,
            _number("1"),
            _RPAREN,
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected LPAREN"):
            parse(tokens)

    def test_while_missing_rparen_raises_parse_error(self) -> None:
        """``while (1 ;`` — missing closing parenthesis raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _WHILE,
            _LPAREN,
            _number("1"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected RPAREN"):
            parse(tokens)


# ---------------------------------------------------------------------------
# Do/while statements (_parse_statement → DO case)
# ---------------------------------------------------------------------------


class TestDoWhileStatements:
    """Parser correctly builds ``DoWhile`` nodes for do/while loops."""

    def test_do_while_basic(self) -> None:
        """``do return 0; while (1);`` → ``DoWhile(body=Return(Constant(0)), condition=Constant(1))``."""
        result = _parsed_stmt(
            _DO, _RETURN, _number("0"), _SEMICOLON, _WHILE, _LPAREN, _number("1"), _RPAREN, _SEMICOLON
        )
        assert result == DoWhile(body=Return(exp=Constant(0)), condition=Constant(1))

    def test_do_while_with_compound_body(self) -> None:
        """``do {} while (1);`` → ``DoWhile`` with an empty ``Compound`` body."""
        result = _parsed_stmt(_DO, _LBRACE, _RBRACE, _WHILE, _LPAREN, _number("1"), _RPAREN, _SEMICOLON)
        assert result == DoWhile(body=Compound(body=()), condition=Constant(1))

    def test_do_while_condition_is_arbitrary_expression(self) -> None:
        """``do ; while (a < 10);`` — comparison expression as condition."""
        result = _parsed_stmt(_DO, _SEMICOLON, _WHILE, _LPAREN, _ident("a"), _LT, _number("10"), _RPAREN, _SEMICOLON)
        assert result == DoWhile(body=Null(), condition=Binary(LessThan(), Var("a"), Constant(10)))

    def test_do_while_missing_while_keyword_raises_parse_error(self) -> None:
        """``do ; (1);`` — absent ``while`` keyword raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _DO,
            _SEMICOLON,
            _LPAREN,
            _number("1"),
            _RPAREN,
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected WHILE"):
            parse(tokens)

    def test_do_while_missing_closing_paren_raises_parse_error(self) -> None:
        """``do ; while (1;`` — missing ``)`` after condition raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _DO,
            _SEMICOLON,
            _WHILE,
            _LPAREN,
            _number("1"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected RPAREN"):
            parse(tokens)

    def test_do_while_missing_trailing_semicolon_raises_parse_error(self) -> None:
        """``do ; while (1)`` — missing trailing ``;`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _DO,
            _SEMICOLON,
            _WHILE,
            _LPAREN,
            _number("1"),
            _RPAREN,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected SEMICOLON"):
            parse(tokens)


# ---------------------------------------------------------------------------
# For statements (_parse_statement → FOR case)
# ---------------------------------------------------------------------------


class TestForStatements:
    """Parser correctly builds ``For`` nodes for all for-loop clause combinations."""

    def test_for_with_declaration_init(self) -> None:
        """``for (int i = 0; i < 10; i++) return 0;`` — declaration init clause."""
        result = _parsed_stmt(
            _FOR,
            _LPAREN,
            _INT,
            _ident("i"),
            _ASSIGNMENT,
            _number("0"),
            _SEMICOLON,
            _ident("i"),
            _LT,
            _number("10"),
            _SEMICOLON,
            _ident("i"),
            _INCREMENT,
            _RPAREN,
            _RETURN,
            _number("0"),
            _SEMICOLON,
        )
        assert result == For(
            init=InitDecl(declaration=Declaration(name="i", init=Constant(0))),
            condition=Binary(LessThan(), Var("i"), Constant(10)),
            post=Unary(PostIncrement(), Var("i")),
            body=Return(exp=Constant(0)),
        )

    def test_for_with_expression_init(self) -> None:
        """``for (i = 0; i < 10; i++) ;`` — expression init clause."""
        result = _parsed_stmt(
            _FOR,
            _LPAREN,
            _ident("i"),
            _ASSIGNMENT,
            _number("0"),
            _SEMICOLON,
            _ident("i"),
            _LT,
            _number("10"),
            _SEMICOLON,
            _ident("i"),
            _INCREMENT,
            _RPAREN,
            _SEMICOLON,
        )
        assert result == For(
            init=InitExp(exp=Assignment(lvalue=Var("i"), rvalue=Constant(0))),
            condition=Binary(LessThan(), Var("i"), Constant(10)),
            post=Unary(PostIncrement(), Var("i")),
            body=Null(),
        )

    def test_for_with_empty_init(self) -> None:
        """``for (; i < 10; i++) ;`` — absent init clause yields ``InitExp(exp=None)``."""
        result = _parsed_stmt(
            _FOR,
            _LPAREN,
            _SEMICOLON,
            _ident("i"),
            _LT,
            _number("10"),
            _SEMICOLON,
            _ident("i"),
            _INCREMENT,
            _RPAREN,
            _SEMICOLON,
        )
        assert result == For(
            init=InitExp(exp=None),
            condition=Binary(LessThan(), Var("i"), Constant(10)),
            post=Unary(PostIncrement(), Var("i")),
            body=Null(),
        )

    def test_for_with_no_condition(self) -> None:
        """``for (int i = 0;; i++) ;`` — absent condition yields ``condition=None``."""
        result = _parsed_stmt(
            _FOR,
            _LPAREN,
            _INT,
            _ident("i"),
            _ASSIGNMENT,
            _number("0"),
            _SEMICOLON,
            _SEMICOLON,
            _ident("i"),
            _INCREMENT,
            _RPAREN,
            _SEMICOLON,
        )
        assert result == For(
            init=InitDecl(declaration=Declaration(name="i", init=Constant(0))),
            condition=None,
            post=Unary(PostIncrement(), Var("i")),
            body=Null(),
        )

    def test_for_with_no_post(self) -> None:
        """``for (int i = 0; i < 10;) ;`` — absent post clause yields ``post=None``."""
        result = _parsed_stmt(
            _FOR,
            _LPAREN,
            _INT,
            _ident("i"),
            _ASSIGNMENT,
            _number("0"),
            _SEMICOLON,
            _ident("i"),
            _LT,
            _number("10"),
            _SEMICOLON,
            _RPAREN,
            _SEMICOLON,
        )
        assert result == For(
            init=InitDecl(declaration=Declaration(name="i", init=Constant(0))),
            condition=Binary(LessThan(), Var("i"), Constant(10)),
            post=None,
            body=Null(),
        )

    def test_for_all_clauses_absent(self) -> None:
        """``for (;;) ;`` — all optional clauses absent."""
        result = _parsed_stmt(_FOR, _LPAREN, _SEMICOLON, _SEMICOLON, _RPAREN, _SEMICOLON)
        assert result == For(
            init=InitExp(exp=None),
            condition=None,
            post=None,
            body=Null(),
        )

    def test_for_body_is_compound(self) -> None:
        """``for (;;) { return 0; }`` — compound statement as the loop body."""
        result = _parsed_stmt(
            _FOR,
            _LPAREN,
            _SEMICOLON,
            _SEMICOLON,
            _RPAREN,
            _LBRACE,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        )
        assert result == For(
            init=InitExp(exp=None),
            condition=None,
            post=None,
            body=Compound(body=(S(statement=Return(exp=Constant(0))),)),
        )

    def test_for_missing_lparen_raises_parse_error(self) -> None:
        """``for ;; ) ;`` — missing ``(`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _FOR,
            _SEMICOLON,
            _SEMICOLON,
            _RPAREN,
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected LPAREN"):
            parse(tokens)

    def test_for_missing_first_semicolon_raises_parse_error(self) -> None:
        """``for (i) ;`` — init clause expression not terminated with ``;`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _FOR,
            _LPAREN,
            _ident("i"),
            _RPAREN,
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected SEMICOLON"):
            parse(tokens)

    def test_for_missing_second_semicolon_raises_parse_error(self) -> None:
        """``for (; i) ;`` — condition expression not terminated with ``;`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _FOR,
            _LPAREN,
            _SEMICOLON,
            _ident("i"),
            _RPAREN,
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected SEMICOLON"):
            parse(tokens)

    def test_for_missing_rparen_raises_parse_error(self) -> None:
        """``for (;;`` — missing ``)`` after post expression raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _FOR,
            _LPAREN,
            _SEMICOLON,
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError):
            parse(tokens)


# ---------------------------------------------------------------------------
# Switch / case / default statements
# ---------------------------------------------------------------------------


class TestSwitchStatements:
    """Parser correctly builds ``Switch``, ``Case``, and ``Default`` nodes."""

    def test_switch_with_null_body(self) -> None:
        """``switch (1) ;`` → ``Switch(control=Constant(1), body=Null())``."""
        result = _parsed_stmt(_SWITCH, _LPAREN, _number("1"), _RPAREN, _SEMICOLON)
        assert result == Switch(control=Constant(1), body=Null())

    def test_switch_with_empty_compound_body(self) -> None:
        """``switch (1) {}`` → ``Switch`` whose body is an empty ``Compound``."""
        result = _parsed_stmt(_SWITCH, _LPAREN, _number("1"), _RPAREN, _LBRACE, _RBRACE)
        assert result == Switch(control=Constant(1), body=Compound(body=()))

    def test_switch_control_is_arbitrary_expression(self) -> None:
        """``switch (a + 1) ;`` — binary expression as the control value."""
        result = _parsed_stmt(_SWITCH, _LPAREN, _ident("a"), _PLUS, _number("1"), _RPAREN, _SEMICOLON)
        assert result == Switch(control=Binary(Add(), Var("a"), Constant(1)), body=Null())

    def test_switch_identifier_is_empty_before_semantic_analysis(self) -> None:
        """``Switch.identifier`` is the empty string until semantic analysis assigns one."""
        result = _parsed_stmt(_SWITCH, _LPAREN, _number("1"), _RPAREN, _SEMICOLON)
        assert isinstance(result, Switch)
        assert result.identifier == ""

    def test_case_governing_null_statement(self) -> None:
        """``case 1: ;`` → ``Case(value=Constant(1), statement=Null())``."""
        result = _parsed_stmt(_CASE, _number("1"), _COLON, _SEMICOLON)
        assert result == Case(value=Constant(1), statement=Null())

    def test_case_governing_return_statement(self) -> None:
        """``case 2: return 0;`` → ``Case(value=Constant(2), statement=Return(Constant(0)))``."""
        result = _parsed_stmt(_CASE, _number("2"), _COLON, _RETURN, _number("0"), _SEMICOLON)
        assert result == Case(value=Constant(2), statement=Return(exp=Constant(0)))

    def test_case_value_must_be_constant_expression_raises_parse_error(self) -> None:
        """``case a: ;`` — a non-constant case value raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _CASE,
            _ident("a"),
            _COLON,
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError):
            parse(tokens)

    def test_case_identifier_is_empty_before_semantic_analysis(self) -> None:
        """``Case.identifier`` is the empty string until semantic analysis assigns one."""
        result = _parsed_stmt(_CASE, _number("1"), _COLON, _SEMICOLON)
        assert isinstance(result, Case)
        assert result.identifier == ""

    def test_default_governing_null_statement(self) -> None:
        """``default: ;`` → ``Default(statement=Null())``."""
        result = _parsed_stmt(_DEFAULT, _COLON, _SEMICOLON)
        assert result == Default(statement=Null())

    def test_default_governing_return_statement(self) -> None:
        """``default: return 0;`` → ``Default(statement=Return(Constant(0)))``."""
        result = _parsed_stmt(_DEFAULT, _COLON, _RETURN, _number("0"), _SEMICOLON)
        assert result == Default(statement=Return(exp=Constant(0)))

    def test_default_identifier_is_empty_before_semantic_analysis(self) -> None:
        """``Default.identifier`` is the empty string until semantic analysis assigns one."""
        result = _parsed_stmt(_DEFAULT, _COLON, _SEMICOLON)
        assert isinstance(result, Default)
        assert result.identifier == ""

    def test_fall_through_via_nested_case(self) -> None:
        """``case 1: case 2: return 0;`` — the inner ``Case`` is the governed statement of the outer.

        This is the canonical fall-through representation: ``Case(1, Case(2, Return(0)))``.
        """
        result = _parsed_stmt(
            _CASE, _number("1"), _COLON, _CASE, _number("2"), _COLON, _RETURN, _number("0"), _SEMICOLON
        )
        assert result == Case(
            value=Constant(1),
            statement=Case(value=Constant(2), statement=Return(exp=Constant(0))),
        )

    def test_switch_with_case_in_compound_body(self) -> None:
        """``switch (x) { case 1: return 1; }`` — single case inside a compound body."""
        result = _parsed_stmt(
            _SWITCH,
            _LPAREN,
            _ident("x"),
            _RPAREN,
            _LBRACE,
            _CASE,
            _number("1"),
            _COLON,
            _RETURN,
            _number("1"),
            _SEMICOLON,
            _RBRACE,
        )
        assert result == Switch(
            control=Var("x"),
            body=Compound(body=(S(statement=Case(value=Constant(1), statement=Return(exp=Constant(1)))),)),
        )

    def test_switch_with_case_and_default_in_compound_body(self) -> None:
        """``switch (x) { case 1: return 1; default: return 0; }`` — case and default as siblings."""
        result = _parsed_stmt(
            _SWITCH,
            _LPAREN,
            _ident("x"),
            _RPAREN,
            _LBRACE,
            _CASE,
            _number("1"),
            _COLON,
            _RETURN,
            _number("1"),
            _SEMICOLON,
            _DEFAULT,
            _COLON,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        )
        assert result == Switch(
            control=Var("x"),
            body=Compound(
                body=(
                    S(statement=Case(value=Constant(1), statement=Return(exp=Constant(1)))),
                    S(statement=Default(statement=Return(exp=Constant(0)))),
                )
            ),
        )

    def test_switch_missing_lparen_raises_parse_error(self) -> None:
        """``switch 1) ;`` — missing ``(`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _SWITCH,
            _number("1"),
            _RPAREN,
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected LPAREN"):
            parse(tokens)

    def test_switch_missing_rparen_raises_parse_error(self) -> None:
        """``switch (1 ;`` — missing ``)`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _SWITCH,
            _LPAREN,
            _number("1"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected RPAREN"):
            parse(tokens)

    def test_case_missing_colon_raises_parse_error(self) -> None:
        """``case 1 return 0;`` — missing ``:`` after the case value raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _CASE,
            _number("1"),
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected COLON"):
            parse(tokens)

    def test_default_missing_colon_raises_parse_error(self) -> None:
        """``default return 0;`` — missing ``:`` after ``default`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _DEFAULT,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected COLON"):
            parse(tokens)
