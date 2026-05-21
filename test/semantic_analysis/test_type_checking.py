"""Tests for type-checking rules enforced by ``SemanticAnalyser``.

Verifies that operations illegal on ``double`` operands are rejected at the
semantic analysis stage with a ``SemanticError``, rather than silently
propagating to later pipeline stages.
"""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import _program


def _double_const() -> ast.Constant:
    """Return a typed ``double`` constant (5.0)."""
    return ast.Constant(value=ast.ConstDouble(5.0))


def _int_const() -> ast.Constant:
    """Return a typed ``int`` constant (3)."""
    return ast.Constant(value=ast.ConstInt(3))


def _return_exp(exp: ast.Exp) -> tuple[ast.BlockItem, ...]:
    """Wrap *exp* in a ``Return`` statement body suitable for ``_program``."""
    return (ast.S(ast.Return(exp)),)


class TestDoubleOperatorRestrictions:
    """Operators that are undefined for ``double`` must raise ``SemanticError``."""

    def test_bitwise_complement_of_double_raises(self) -> None:
        """Applying ``~`` to a ``double`` operand raises ``SemanticError``."""
        prog = _program(_return_exp(ast.Unary(ast.Complement(), _double_const())))
        with pytest.raises(SemanticError, match="bitwise complement"):
            resolve(prog)

    def test_remainder_with_double_left_raises(self) -> None:
        """Applying ``%`` with a ``double`` left operand raises ``SemanticError``."""
        prog = _program(_return_exp(ast.Binary(ast.Remainder(), _double_const(), _int_const())))
        with pytest.raises(SemanticError, match="modulus"):
            resolve(prog)

    def test_remainder_with_double_right_raises(self) -> None:
        """Applying ``%`` with a ``double`` right operand raises ``SemanticError``."""
        prog = _program(_return_exp(ast.Binary(ast.Remainder(), _int_const(), _double_const())))
        with pytest.raises(SemanticError, match="modulus"):
            resolve(prog)

    def test_left_shift_with_double_left_raises(self) -> None:
        """Applying ``<<`` with a ``double`` left operand raises ``SemanticError``."""
        prog = _program(_return_exp(ast.Binary(ast.LeftShift(), _double_const(), _int_const())))
        with pytest.raises(SemanticError, match="shift"):
            resolve(prog)

    def test_left_shift_with_double_right_raises(self) -> None:
        """Applying ``<<`` with a ``double`` right operand raises ``SemanticError``."""
        prog = _program(_return_exp(ast.Binary(ast.LeftShift(), _int_const(), _double_const())))
        with pytest.raises(SemanticError, match="shift"):
            resolve(prog)

    def test_right_shift_with_double_left_raises(self) -> None:
        """Applying ``>>`` with a ``double`` left operand raises ``SemanticError``."""
        prog = _program(_return_exp(ast.Binary(ast.RightShift(), _double_const(), _int_const())))
        with pytest.raises(SemanticError, match="shift"):
            resolve(prog)

    def test_right_shift_with_double_right_raises(self) -> None:
        """Applying ``>>`` with a ``double`` right operand raises ``SemanticError``."""
        prog = _program(_return_exp(ast.Binary(ast.RightShift(), _int_const(), _double_const())))
        with pytest.raises(SemanticError, match="shift"):
            resolve(prog)


class TestSwitchTypeRestrictions:
    """The controlling expression of a ``switch`` must be an integer type."""

    def test_switch_with_double_control_raises(self) -> None:
        """A ``switch`` whose control expression is ``double`` raises ``SemanticError``."""
        switch = ast.Switch(control=_double_const(), body=ast.Null())
        prog = _program((ast.S(statement=switch),))
        with pytest.raises(SemanticError, match="integer"):
            resolve(prog)
