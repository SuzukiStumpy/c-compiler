"""Parser tests for ``_Alignof`` expressions."""

from compiler.syntax.ast import AlignOf, AlignOfT, DoubleType, IntType, LongType, Pointer
from compiler.types.tokens import TokenType
from test.parser.helpers import (
    _INT,
    _LONG,
    _LPAREN,
    _RPAREN,
    _ident,
    _parsed_exp,
    _tok,
)

_ALIGNOF = _tok(TokenType.ALIGNOF, "_Alignof")
_DOUBLE = _tok(TokenType.DOUBLE, "double")


def test_alignof_type_int_produces_alignof_t_node() -> None:
    """``_Alignof(int)`` parses to ``AlignOfT(IntType())``."""
    exp = _parsed_exp(_ALIGNOF, _LPAREN, _INT, _RPAREN)
    assert isinstance(exp, AlignOfT)
    assert exp.t == IntType()


def test_alignof_type_double_produces_alignof_t_node() -> None:
    """``_Alignof(double)`` parses to ``AlignOfT(DoubleType())``."""
    exp = _parsed_exp(_ALIGNOF, _LPAREN, _DOUBLE, _RPAREN)
    assert isinstance(exp, AlignOfT)
    assert exp.t == DoubleType()


def test_alignof_type_long_produces_alignof_t_node() -> None:
    """``_Alignof(long)`` parses to ``AlignOfT(LongType())``."""
    exp = _parsed_exp(_ALIGNOF, _LPAREN, _LONG, _RPAREN)
    assert isinstance(exp, AlignOfT)
    assert exp.t == LongType()


def test_alignof_expression_produces_alignof_node() -> None:
    """``_Alignof x`` (without parens) parses to ``AlignOf(Var("x"))``."""

    exp = _parsed_exp(_ALIGNOF, _ident("x"))
    assert isinstance(exp, AlignOf)


def test_alignof_type_pointer_produces_alignof_t_node() -> None:
    """``_Alignof(int *)`` parses to ``AlignOfT(Pointer(IntType()))``."""
    _ASTERISK = _tok(TokenType.ASTERISK, "*")
    exp = _parsed_exp(_ALIGNOF, _LPAREN, _INT, _ASTERISK, _RPAREN)
    assert isinstance(exp, AlignOfT)
    assert exp.t == Pointer(IntType())
