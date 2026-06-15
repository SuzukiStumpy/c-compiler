"""Parser tests for expression constructs."""

import pytest

from compiler.parser import parse
from compiler.syntax.ast import (
    AddrOf,
    Cast,
    Constant,
    ConstInt,
    ConstLong,
    Dereference,
    IntType,
    LongType,
    String,
    Unary,
)
from compiler.syntax.ast.expressions import Assignment, Binary, Conditional, Var
from compiler.syntax.ast.operators import (
    Add,
    And,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
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
from compiler.types.tokens import Token
from compiler.utils.exceptions import ParseError
from test.parser.helpers import (
    _AMPERSAND,
    _AND,
    _ASSIGNMENT,
    _ASTERISK,
    _BITWISE_NOT,
    _CARET,
    _COLON,
    _DECREMENT,
    _DIVIDE,
    _EQ,
    _GEQ,
    _GT,
    _INCREMENT,
    _INT,
    _LBRACE,
    _LEQ,
    _LPAREN,
    _LSHIFT,
    _LT,
    _MINUS,
    _NEQ,
    _NOT,
    _OR,
    _PERCENT,
    _PIPE,
    _PLUS,
    _QUERY,
    _RBRACE,
    _RETURN,
    _RPAREN,
    _RSHIFT,
    _SEMICOLON,
    _VOID,
    _charconst,
    _ident,
    _number,
    _parsed_exp,
    _string_tok,
)

# ---------------------------------------------------------------------------
# Unary expressions (_parse_factor)
# ---------------------------------------------------------------------------


class TestUnaryExpressions:
    """Parser correctly builds Unary nodes for all unary operators."""

    def test_negate(self) -> None:
        """``return -1;`` -> Unary(Negate(), Constant(ConstInt(1)))."""
        assert _parsed_exp(_MINUS, _number("1")) == Unary(Negate(), Constant(ConstInt(1)))

    def test_complement(self) -> None:
        """``return ~1;`` -> Unary(Complement(), Constant(ConstInt(1)))."""
        from compiler.syntax.ast.operators import Complement

        assert _parsed_exp(_BITWISE_NOT, _number("1")) == Unary(Complement(), Constant(ConstInt(1)))

    def test_double_negate(self) -> None:
        """``return --1;`` - wait, DECREMENT is a separate token; ``- -1`` -> Unary(Negate(), Unary(Negate(), Constant(ConstInt(1))))."""  # noqa: E501
        assert _parsed_exp(_MINUS, _MINUS, _number("1")) == Unary(Negate(), Unary(Negate(), Constant(ConstInt(1))))

    def test_negate_complement(self) -> None:
        """``return -~1;`` -> Unary(Negate(), Unary(Complement(), Constant(ConstInt(1))))."""
        from compiler.syntax.ast.operators import Complement

        assert _parsed_exp(_MINUS, _BITWISE_NOT, _number("1")) == Unary(
            Negate(), Unary(Complement(), Constant(ConstInt(1)))
        )

    def test_logical_not(self) -> None:
        """``return !1;`` -> Unary(Not(), Constant(ConstInt(1)))."""
        assert _parsed_exp(_NOT, _number("1")) == Unary(Not(), Constant(ConstInt(1)))

    def test_parenthesised_constant(self) -> None:
        """``return (42);`` - parentheses around a constant are transparent."""
        assert _parsed_exp(_LPAREN, _number("42"), _RPAREN) == Constant(ConstInt(42))


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
            (_ASTERISK, Multiply()),
            (_DIVIDE, Divide()),
            (_PERCENT, Remainder()),
        ],
        ids=["add", "subtract", "multiply", "divide", "remainder"],
    )
    def test_operator_parses(self, tok: Token, op: object) -> None:
        """``return 1 <op> 2;`` -> Binary(<op>, Constant(ConstInt(1)), Constant(ConstInt(2)))."""
        assert _parsed_exp(_number("1"), tok, _number("2")) == Binary(op, Constant(ConstInt(1)), Constant(ConstInt(2)))

    def test_multiply_binds_tighter_than_add(self) -> None:
        """``return 1 + 2 * 3;`` -> Add(1, Multiply(2, 3)) by precedence."""
        assert _parsed_exp(_number("1"), _PLUS, _number("2"), _ASTERISK, _number("3")) == Binary(
            Add(), Constant(ConstInt(1)), Binary(Multiply(), Constant(ConstInt(2)), Constant(ConstInt(3)))
        )

    def test_multiply_binds_tighter_than_subtract(self) -> None:
        """``return 4 - 2 * 3;`` -> Subtract(4, Multiply(2, 3)) by precedence."""
        assert _parsed_exp(_number("4"), _MINUS, _number("2"), _ASTERISK, _number("3")) == Binary(
            Subtract(), Constant(ConstInt(4)), Binary(Multiply(), Constant(ConstInt(2)), Constant(ConstInt(3)))
        )

    def test_add_is_left_associative(self) -> None:
        """``return 1 + 2 + 3;`` -> Add(Add(1, 2), 3) by left-associativity."""
        assert _parsed_exp(_number("1"), _PLUS, _number("2"), _PLUS, _number("3")) == Binary(
            Add(), Binary(Add(), Constant(ConstInt(1)), Constant(ConstInt(2))), Constant(ConstInt(3))
        )

    def test_subtract_is_left_associative(self) -> None:
        """``return 5 - 3 - 1;`` -> Subtract(Subtract(5, 3), 1) by left-associativity."""
        assert _parsed_exp(_number("5"), _MINUS, _number("3"), _MINUS, _number("1")) == Binary(
            Subtract(), Binary(Subtract(), Constant(ConstInt(5)), Constant(ConstInt(3))), Constant(ConstInt(1))
        )

    def test_divide_is_left_associative(self) -> None:
        """``return 12 / 4 / 3;`` -> Divide(Divide(12, 4), 3) by left-associativity."""
        assert _parsed_exp(_number("12"), _DIVIDE, _number("4"), _DIVIDE, _number("3")) == Binary(
            Divide(), Binary(Divide(), Constant(ConstInt(12)), Constant(ConstInt(4))), Constant(ConstInt(3))
        )

    def test_parentheses_override_precedence(self) -> None:
        """``return (1 + 2) * 3;`` -> Multiply(Add(1, 2), 3) via grouping."""
        assert _parsed_exp(_LPAREN, _number("1"), _PLUS, _number("2"), _RPAREN, _ASTERISK, _number("3")) == Binary(
            Multiply(), Binary(Add(), Constant(ConstInt(1)), Constant(ConstInt(2))), Constant(ConstInt(3))
        )

    def test_unary_negation_as_left_operand(self) -> None:
        """``return -1 + 2;`` -> Add(Negate(1), 2)."""
        assert _parsed_exp(_MINUS, _number("1"), _PLUS, _number("2")) == Binary(
            Add(), Unary(Negate(), Constant(ConstInt(1))), Constant(ConstInt(2))
        )

    def test_unary_negation_as_right_operand(self) -> None:
        """``return 2 + -1;`` -> Add(2, Negate(1))."""
        assert _parsed_exp(_number("2"), _PLUS, _MINUS, _number("1")) == Binary(
            Add(), Constant(ConstInt(2)), Unary(Negate(), Constant(ConstInt(1)))
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
        """``return 1 <op> 2;`` -> Binary(<op>, Constant(ConstInt(1)), Constant(ConstInt(2)))."""
        assert _parsed_exp(_number("1"), tok, _number("2")) == Binary(op, Constant(ConstInt(1)), Constant(ConstInt(2)))

    def test_shift_binds_tighter_than_bitwise_and(self) -> None:
        """``return 1 & 2 << 3;`` -> And(1, LeftShift(2, 3)): shift prec 40 > AND prec 25."""
        assert _parsed_exp(_number("1"), _AMPERSAND, _number("2"), _LSHIFT, _number("3")) == Binary(
            BitwiseAnd(), Constant(ConstInt(1)), Binary(LeftShift(), Constant(ConstInt(2)), Constant(ConstInt(3)))
        )

    def test_shift_binds_tighter_than_bitwise_or(self) -> None:
        """``return 1 | 2 << 3;`` -> Or(1, LeftShift(2, 3)): shift prec 40 > OR prec 15."""
        assert _parsed_exp(_number("1"), _PIPE, _number("2"), _LSHIFT, _number("3")) == Binary(
            BitwiseOr(), Constant(ConstInt(1)), Binary(LeftShift(), Constant(ConstInt(2)), Constant(ConstInt(3)))
        )

    def test_add_binds_tighter_than_shift(self) -> None:
        """``return 1 + 2 << 3;`` -> LeftShift(Add(1, 2), 3): additive prec 45 > shift prec 40."""
        assert _parsed_exp(_number("1"), _PLUS, _number("2"), _LSHIFT, _number("3")) == Binary(
            LeftShift(), Binary(Add(), Constant(ConstInt(1)), Constant(ConstInt(2))), Constant(ConstInt(3))
        )

    def test_bitwise_and_binds_tighter_than_xor(self) -> None:
        """``return 1 ^ 2 & 3;`` -> Xor(1, And(2, 3)): AND prec 25 > XOR prec 20."""
        assert _parsed_exp(_number("1"), _CARET, _number("2"), _AMPERSAND, _number("3")) == Binary(
            BitwiseXor(), Constant(ConstInt(1)), Binary(BitwiseAnd(), Constant(ConstInt(2)), Constant(ConstInt(3)))
        )

    def test_bitwise_xor_binds_tighter_than_or(self) -> None:
        """``return 1 | 2 ^ 3;`` -> Or(1, Xor(2, 3)): XOR prec 20 > OR prec 15."""
        assert _parsed_exp(_number("1"), _PIPE, _number("2"), _CARET, _number("3")) == Binary(
            BitwiseOr(), Constant(ConstInt(1)), Binary(BitwiseXor(), Constant(ConstInt(2)), Constant(ConstInt(3)))
        )

    def test_shift_is_left_associative(self) -> None:
        """``return 1 << 2 << 3;`` -> LeftShift(LeftShift(1, 2), 3)."""
        assert _parsed_exp(_number("1"), _LSHIFT, _number("2"), _LSHIFT, _number("3")) == Binary(
            LeftShift(), Binary(LeftShift(), Constant(ConstInt(1)), Constant(ConstInt(2))), Constant(ConstInt(3))
        )

    def test_bitwise_or_is_left_associative(self) -> None:
        """``return 1 | 2 | 3;`` -> Or(Or(1, 2), 3)."""
        assert _parsed_exp(_number("1"), _PIPE, _number("2"), _PIPE, _number("3")) == Binary(
            BitwiseOr(), Binary(BitwiseOr(), Constant(ConstInt(1)), Constant(ConstInt(2))), Constant(ConstInt(3))
        )

    def test_parentheses_override_shift_precedence(self) -> None:
        """``return 1 << (2 + 3);`` -> LeftShift(1, Add(2, 3)): grouping overrides normal precedence."""
        assert _parsed_exp(_number("1"), _LSHIFT, _LPAREN, _number("2"), _PLUS, _number("3"), _RPAREN) == Binary(
            LeftShift(), Constant(ConstInt(1)), Binary(Add(), Constant(ConstInt(2)), Constant(ConstInt(3)))
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
        assert _parsed_exp(_number("1"), tok, _number("2")) == Binary(op, Constant(ConstInt(1)), Constant(ConstInt(2)))

    def test_comparison_binds_tighter_than_logical_and(self) -> None:
        """``1 && 2 == 3`` -> ``And(1, Equal(2, 3))``: comparison prec 30 > logical AND prec 10."""
        assert _parsed_exp(_number("1"), _AND, _number("2"), _EQ, _number("3")) == Binary(
            And(), Constant(ConstInt(1)), Binary(Equal(), Constant(ConstInt(2)), Constant(ConstInt(3)))
        )

    def test_comparison_binds_tighter_than_bitwise_or(self) -> None:
        """``1 == 2 | 3`` -> ``BitwiseOr(Equal(1, 2), 3)``: comparison prec 30 > bitwise OR prec 15."""
        assert _parsed_exp(_number("1"), _EQ, _number("2"), _PIPE, _number("3")) == Binary(
            BitwiseOr(), Binary(Equal(), Constant(ConstInt(1)), Constant(ConstInt(2))), Constant(ConstInt(3))
        )

    def test_comparison_is_left_associative(self) -> None:
        """``1 < 2 < 3`` -> ``LessThan(LessThan(1, 2), 3)``."""
        assert _parsed_exp(_number("1"), _LT, _number("2"), _LT, _number("3")) == Binary(
            LessThan(), Binary(LessThan(), Constant(ConstInt(1)), Constant(ConstInt(2))), Constant(ConstInt(3))
        )


# ---------------------------------------------------------------------------
# Logical AND / OR expressions
# ---------------------------------------------------------------------------


class TestLogicalExpressions:
    """Parser correctly builds Binary nodes for ``&&`` and ``||``."""

    def test_logical_and_parses(self) -> None:
        """``return 1 && 2;`` -> ``Binary(And(), Constant(ConstInt(1)), Constant(ConstInt(2)))``."""
        assert _parsed_exp(_number("1"), _AND, _number("2")) == Binary(
            And(), Constant(ConstInt(1)), Constant(ConstInt(2))
        )

    def test_logical_or_parses(self) -> None:
        """``return 1 || 2;`` -> ``Binary(Or(), Constant(ConstInt(1)), Constant(ConstInt(2)))``."""
        assert _parsed_exp(_number("1"), _OR, _number("2")) == Binary(
            Or(), Constant(ConstInt(1)), Constant(ConstInt(2))
        )

    def test_logical_and_binds_tighter_than_or(self) -> None:
        """``1 || 2 && 3`` -> ``Or(1, And(2, 3))``: AND prec 10 > OR prec 5."""
        assert _parsed_exp(_number("1"), _OR, _number("2"), _AND, _number("3")) == Binary(
            Or(), Constant(ConstInt(1)), Binary(And(), Constant(ConstInt(2)), Constant(ConstInt(3)))
        )

    def test_bitwise_or_binds_tighter_than_logical_and(self) -> None:
        """``1 && 2 | 3`` -> ``And(1, BitwiseOr(2, 3))``: bitwise OR prec 15 > logical AND prec 10."""
        assert _parsed_exp(_number("1"), _AND, _number("2"), _PIPE, _number("3")) == Binary(
            And(), Constant(ConstInt(1)), Binary(BitwiseOr(), Constant(ConstInt(2)), Constant(ConstInt(3)))
        )

    def test_logical_and_is_left_associative(self) -> None:
        """``1 && 2 && 3`` -> ``And(And(1, 2), 3)``."""
        assert _parsed_exp(_number("1"), _AND, _number("2"), _AND, _number("3")) == Binary(
            And(), Binary(And(), Constant(ConstInt(1)), Constant(ConstInt(2))), Constant(ConstInt(3))
        )

    def test_logical_or_is_left_associative(self) -> None:
        """``1 || 2 || 3`` -> ``Or(Or(1, 2), 3)``."""
        assert _parsed_exp(_number("1"), _OR, _number("2"), _OR, _number("3")) == Binary(
            Or(), Binary(Or(), Constant(ConstInt(1)), Constant(ConstInt(2))), Constant(ConstInt(3))
        )


# ---------------------------------------------------------------------------
# Prefix and postfix increment / decrement (_parse_factor)
# ---------------------------------------------------------------------------


class TestIncDecExpressions:
    """Parser correctly builds Unary nodes for prefix and postfix ++/-- operators."""

    def test_prefix_increment(self) -> None:
        """``return ++a;`` -> ``Unary(PreIncrement(), Var("a"))``."""
        assert _parsed_exp(_INCREMENT, _ident("a")) == Unary(PreIncrement(), Var("a"))

    def test_prefix_decrement(self) -> None:
        """``return --a;`` -> ``Unary(PreDecrement(), Var("a"))``."""
        assert _parsed_exp(_DECREMENT, _ident("a")) == Unary(PreDecrement(), Var("a"))

    def test_postfix_increment(self) -> None:
        """``return a++;`` -> ``Unary(PostIncrement(), Var("a"))``."""
        assert _parsed_exp(_ident("a"), _INCREMENT) == Unary(PostIncrement(), Var("a"))

    def test_postfix_decrement(self) -> None:
        """``return a--;`` -> ``Unary(PostDecrement(), Var("a"))``."""
        assert _parsed_exp(_ident("a"), _DECREMENT) == Unary(PostDecrement(), Var("a"))

    def test_chained_postfix_increment_then_decrement(self) -> None:
        """``return a++--;`` -> ``Unary(PostDecrement(), Unary(PostIncrement(), Var("a")))``."""
        assert _parsed_exp(_ident("a"), _INCREMENT, _DECREMENT) == Unary(
            PostDecrement(), Unary(PostIncrement(), Var("a"))
        )

    def test_postfix_binds_tighter_than_prefix(self) -> None:
        """``return ++a++;`` -> ``Unary(PreIncrement(), Unary(PostIncrement(), Var("a")))``.

        Postfix is parsed as part of the primary before prefix wraps it.
        """
        assert _parsed_exp(_INCREMENT, _ident("a"), _INCREMENT) == Unary(
            PreIncrement(), Unary(PostIncrement(), Var("a"))
        )


# ---------------------------------------------------------------------------
# Ternary / conditional expressions (_parse_exp -> QUERY case)
# ---------------------------------------------------------------------------


class TestTernaryExpressions:
    """Parser correctly builds ``Conditional`` nodes for ternary ``?:`` expressions."""

    def test_basic_ternary(self) -> None:
        """
        ``return 1 ? 2 : 3;`` -> ``Conditional(Constant(ConstInt(1)), Constant(ConstInt(2)), Constant(ConstInt(3)))``.
        """
        assert _parsed_exp(_number("1"), _QUERY, _number("2"), _COLON, _number("3")) == Conditional(
            condition=Constant(ConstInt(1)), exp_true=Constant(ConstInt(2)), exp_false=Constant(ConstInt(3))
        )

    def test_condition_is_arbitrary_expression(self) -> None:
        """``return a > 0 ? 1 : 0;`` - comparison expression as condition."""
        assert _parsed_exp(_ident("a"), _GT, _number("0"), _QUERY, _number("1"), _COLON, _number("0")) == Conditional(
            condition=Binary(GreaterThan(), Var("a"), Constant(ConstInt(0))),
            exp_true=Constant(ConstInt(1)),
            exp_false=Constant(ConstInt(0)),
        )

    def test_ternary_is_right_associative(self) -> None:
        """``return 1 ? 2 : 3 ? 4 : 5;`` - else-branch contains a nested ternary, not the reverse."""
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
            condition=Constant(ConstInt(1)),
            exp_true=Constant(ConstInt(2)),
            exp_false=Conditional(
                condition=Constant(ConstInt(3)), exp_true=Constant(ConstInt(4)), exp_false=Constant(ConstInt(5))
            ),
        )

    def test_ternary_is_rhs_of_assignment(self) -> None:
        """``return a = 1 ? 2 : 3;`` - ternary (prec 3) binds tighter than assignment (prec 1)."""
        assert _parsed_exp(
            _ident("a"), _ASSIGNMENT, _number("1"), _QUERY, _number("2"), _COLON, _number("3")
        ) == Assignment(
            lvalue=Var("a"),
            rvalue=Conditional(
                condition=Constant(ConstInt(1)), exp_true=Constant(ConstInt(2)), exp_false=Constant(ConstInt(3))
            ),
        )

    def test_ternary_binds_looser_than_logical_or(self) -> None:
        """``return 1 || 2 ? 3 : 4;`` - ``||`` (prec 5) groups before ``?:`` (prec 3)."""
        assert _parsed_exp(_number("1"), _OR, _number("2"), _QUERY, _number("3"), _COLON, _number("4")) == Conditional(
            condition=Binary(Or(), Constant(ConstInt(1)), Constant(ConstInt(2))),
            exp_true=Constant(ConstInt(3)),
            exp_false=Constant(ConstInt(4)),
        )

    def test_true_branch_allows_assignment(self) -> None:
        """``return 1 ? a = 2 : 3;`` - middle is parsed at prec 0, so assignment is valid."""
        assert _parsed_exp(
            _number("1"), _QUERY, _ident("a"), _ASSIGNMENT, _number("2"), _COLON, _number("3")
        ) == Conditional(
            condition=Constant(ConstInt(1)),
            exp_true=Assignment(lvalue=Var("a"), rvalue=Constant(ConstInt(2))),
            exp_false=Constant(ConstInt(3)),
        )

    def test_missing_colon(self) -> None:
        """``return 1 ? 2;`` - absent ``:`` raises ``ParseError``."""
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
# Integer constant literals (type promotion rules)
# ---------------------------------------------------------------------------


class TestConstantLiterals:
    """Parser correctly classifies integer constants as ConstInt or ConstLong."""

    def test_small_number_produces_const_int(self) -> None:
        """A decimal literal within the 32-bit signed range produces ``ConstInt``."""
        assert _parsed_exp(_number("42")) == Constant(ConstInt(42))

    def test_int_max_produces_const_int(self) -> None:
        """2,147,483,647 (INT_MAX) is the largest value that produces ``ConstInt``."""
        assert _parsed_exp(_number("2147483647")) == Constant(ConstInt(2147483647))

    def test_int_max_plus_one_promotes_to_const_long(self) -> None:
        """2,147,483,648 exceeds INT_MAX and is silently promoted to ``ConstLong``."""
        assert _parsed_exp(_number("2147483648")) == Constant(ConstLong(2147483648))

    def test_large_number_within_long_range_produces_const_long(self) -> None:
        """A decimal literal above INT_MAX but within LONG_MAX produces ``ConstLong``."""
        assert _parsed_exp(_number("9223372036854775807")) == Constant(ConstLong(9223372036854775807))

    def test_number_exceeding_long_max_raises_parse_error(self) -> None:
        """A decimal literal exceeding LONG_MAX raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _RETURN,
            _number("9223372036854775808"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="too large"):
            parse(tokens)

    def test_long_suffix_small_value_produces_const_long(self) -> None:
        """``42L`` always produces ``ConstLong`` regardless of value magnitude."""
        from test.parser.helpers import _longint

        assert _parsed_exp(_longint("42L")) == Constant(ConstLong(42))

    def test_long_suffix_exceeding_long_max_raises_parse_error(self) -> None:
        """``9223372036854775808L`` exceeds LONG_MAX and raises ``ParseError``."""
        from test.parser.helpers import _longint

        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _RETURN,
            _longint("9223372036854775808L"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="too large"):
            parse(tokens)


# ---------------------------------------------------------------------------
# Cast expressions
# ---------------------------------------------------------------------------


class TestCastExpressions:
    """Parser correctly builds ``Cast`` nodes for type-cast primaries."""

    def test_cast_to_long(self) -> None:
        """``(long)1`` -> ``Cast(LongType(), Constant(ConstInt(1)))``."""
        from test.parser.helpers import _LONG

        assert _parsed_exp(_LPAREN, _LONG, _RPAREN, _number("1")) == Cast(
            target_type=LongType(), exp=Constant(ConstInt(1))
        )

    def test_cast_to_int(self) -> None:
        """``(int)x`` -> ``Cast(IntType(), Var("x"))``."""
        assert _parsed_exp(_LPAREN, _INT, _RPAREN, _ident("x")) == Cast(target_type=IntType(), exp=Var("x"))

    def test_cast_long_int(self) -> None:
        """``(long int)1`` -> ``Cast(LongType(), ...)`` - combined specifiers."""
        from test.parser.helpers import _LONG

        assert _parsed_exp(_LPAREN, _LONG, _INT, _RPAREN, _number("1")) == Cast(
            target_type=LongType(), exp=Constant(ConstInt(1))
        )

    def test_cast_int_long(self) -> None:
        """``(int long)1`` -> ``Cast(LongType(), ...)`` - specifiers in either order."""
        from test.parser.helpers import _LONG

        assert _parsed_exp(_LPAREN, _INT, _LONG, _RPAREN, _number("1")) == Cast(
            target_type=LongType(), exp=Constant(ConstInt(1))
        )

    def test_cast_result_used_in_binary(self) -> None:
        """``(long)1 + 2`` - cast binds tighter than addition."""
        from test.parser.helpers import _LONG

        assert _parsed_exp(_LPAREN, _LONG, _RPAREN, _number("1"), _PLUS, _number("2")) == Binary(
            Add(), Cast(target_type=LongType(), exp=Constant(ConstInt(1))), Constant(ConstInt(2))
        )

    def test_nested_cast(self) -> None:
        """``(int)(long)x`` - inner cast is the operand of the outer cast."""
        from test.parser.helpers import _LONG

        assert _parsed_exp(_LPAREN, _INT, _RPAREN, _LPAREN, _LONG, _RPAREN, _ident("x")) == Cast(
            target_type=IntType(), exp=Cast(target_type=LongType(), exp=Var("x"))
        )

    def test_cast_to_pointer(self) -> None:
        """``(int *)x`` -> ``Cast(Pointer(IntType()), Var("x"))``."""
        from compiler.syntax.ast import Pointer
        from test.parser.helpers import _LONG  # noqa: F401

        assert _parsed_exp(_LPAREN, _INT, _ASTERISK, _RPAREN, _ident("x")) == Cast(
            target_type=Pointer(IntType()), exp=Var("x")
        )

    def test_cast_to_double_pointer(self) -> None:
        """``(int **)x`` -> ``Cast(Pointer(Pointer(IntType())), Var("x"))``."""
        from compiler.syntax.ast import Pointer

        assert _parsed_exp(_LPAREN, _INT, _ASTERISK, _ASTERISK, _RPAREN, _ident("x")) == Cast(
            target_type=Pointer(Pointer(IntType())), exp=Var("x")
        )


# ---------------------------------------------------------------------------
# Dereference and address-of (_parse_factor prefix operators)
# ---------------------------------------------------------------------------


class TestDereferenceAndAddrOf:
    """Parser correctly builds ``Dereference`` and ``AddrOf`` nodes."""

    def test_dereference_var(self) -> None:
        """``return *p;`` -> ``Dereference(Var("p"))``."""
        assert _parsed_exp(_ASTERISK, _ident("p")) == Dereference(Var("p"))

    def test_addr_of_var(self) -> None:
        """``return &x;`` -> ``AddrOf(Var("x"))``."""
        assert _parsed_exp(_AMPERSAND, _ident("x")) == AddrOf(Var("x"))

    def test_double_dereference(self) -> None:
        """``return **pp;`` -> ``Dereference(Dereference(Var("pp")))``."""
        assert _parsed_exp(_ASTERISK, _ASTERISK, _ident("pp")) == Dereference(Dereference(Var("pp")))

    def test_addr_of_dereference(self) -> None:
        """``return &*p;`` -> ``AddrOf(Dereference(Var("p")))``."""
        assert _parsed_exp(_AMPERSAND, _ASTERISK, _ident("p")) == AddrOf(Dereference(Var("p")))

    def test_dereference_of_addr_of(self) -> None:
        """``return *&x;`` -> ``Dereference(AddrOf(Var("x")))``."""
        assert _parsed_exp(_ASTERISK, _AMPERSAND, _ident("x")) == Dereference(AddrOf(Var("x")))

    def test_dereference_used_in_binary(self) -> None:
        """``return *p + 1;`` - dereference binds tighter than addition."""
        assert _parsed_exp(_ASTERISK, _ident("p"), _PLUS, _number("1")) == Binary(
            Add(), Dereference(Var("p")), Constant(ConstInt(1))
        )

    def test_addr_of_used_in_assignment(self) -> None:
        """``return q = &x;`` - address-of on the right-hand side of assignment."""
        assert _parsed_exp(_ident("q"), _ASSIGNMENT, _AMPERSAND, _ident("x")) == Assignment(
            lvalue=Var("q"), rvalue=AddrOf(Var("x"))
        )


# ---------------------------------------------------------------------------
# Character constants and string literals
# ---------------------------------------------------------------------------


class TestCharConstantAndStringLiterals:
    """Parser correctly converts ``CONSTCHAR`` tokens and adjacent ``STRING`` tokens to AST nodes."""

    def test_char_constant_ascii_letter_parses_to_constint(self) -> None:
        """``'a'`` -> ``Constant(ConstInt(97))``."""
        assert _parsed_exp(_charconst("'a'")) == Constant(ConstInt(97))

    def test_char_constant_newline_escape_parses_to_value_10(self) -> None:
        """``'\\n'`` -> ``Constant(ConstInt(10))``."""
        assert _parsed_exp(_charconst("'\\n'")) == Constant(ConstInt(10))

    def test_char_constant_backslash_escape_parses_to_value_92(self) -> None:
        """``'\\\\'`` (backslash escape) -> ``Constant(ConstInt(92))``."""
        assert _parsed_exp(_charconst("'\\\\'")) == Constant(ConstInt(92))

    def test_char_constant_tab_escape_parses_to_value_9(self) -> None:
        """``'\\t'`` -> ``Constant(ConstInt(9))``."""
        assert _parsed_exp(_charconst("'\\t'")) == Constant(ConstInt(9))

    def test_char_constant_single_quote_escape_parses_to_value_39(self) -> None:
        """``'\\''`` -> ``Constant(ConstInt(39))``."""
        assert _parsed_exp(_charconst("'\\''")) == Constant(ConstInt(39))

    def test_string_literal_parses_to_string_node(self) -> None:
        """``"hello"`` -> ``String(b"hello")``."""
        assert _parsed_exp(_string_tok('"hello"')) == String(b"hello")

    def test_empty_string_literal_parses_to_empty_bytes(self) -> None:
        """``""`` -> ``String(b"")``."""
        assert _parsed_exp(_string_tok('""')) == String(b"")

    def test_string_with_newline_escape_is_unescaped(self) -> None:
        """``"hi\\n"`` -> ``String(b"hi\\n")`` (NUL not included)."""
        assert _parsed_exp(_string_tok('"hi\\n"')) == String(b"hi\n")

    def test_adjacent_string_literals_are_concatenated(self) -> None:
        """``"foo" "bar"`` -> ``String(b"foobar")``."""
        assert _parsed_exp(_string_tok('"foo"'), _string_tok('"bar"')) == String(b"foobar")
