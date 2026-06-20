"""Semantic analysis tests for ``_Generic`` selection expressions (C11 §6.5.1.1).

``_Generic`` is folded entirely in the ``resolve`` pass: the controlling
expression is typed, the matching association is selected, and the selected
expression is returned in place of the ``Generic`` node.  No ``Generic``
node ever appears in the resolved AST.
"""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import _program, _resolved_body, _vd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _return_generic(
    ctrl: ast.Exp,
    associations: tuple[tuple[ast.Type | None, ast.Exp], ...],
) -> tuple[ast.BlockItem, ...]:
    """Return block items: ``return _Generic(ctrl, associations);``

    Args:
        ctrl: The controlling expression.
        associations: Sequence of ``(type, expr)`` pairs; ``None`` type = ``default``.

    Returns:
        A single-element body tuple containing a ``Return`` of a ``Generic`` node.
    """
    return (ast.S(ast.Return(ast.Generic(controlling_exp=ctrl, associations=associations))),)


def _int_var(name: str) -> ast.Exp:
    """Return ``Var(name)`` to use as an int controlling expression.

    Args:
        name: Variable name.

    Returns:
        An ``ast.Var`` with the given name.
    """
    return ast.Var(name)


def _int_const(n: int) -> ast.Exp:
    """Return ``Constant(ConstInt(n))``.

    Args:
        n: Integer value.

    Returns:
        An ``ast.Constant`` wrapping ``ConstInt``.
    """
    return ast.Constant(ast.ConstInt(n))


def _double_const(v: float) -> ast.Exp:
    """Return ``Constant(ConstDouble(v))``.

    Args:
        v: Float value.

    Returns:
        An ``ast.Constant`` wrapping ``ConstDouble``.
    """
    return ast.Constant(ast.ConstDouble(v))


# ---------------------------------------------------------------------------
# Basic type selection
# ---------------------------------------------------------------------------


class TestGenericTypeSelection:
    """Controlling expression type selects the matching association."""

    def test_int_matches_int_association(self) -> None:
        """``_Generic(x, int: 7, default: 0)`` with int ``x`` folds to ``7``."""
        body = (
            _vd("x"),
            *_return_generic(
                _int_var("x"),
                ((ast.IntType(), _int_const(7)), (None, _int_const(0))),
            ),
        )
        resolved = _resolved_body(body)
        ret = resolved[1]
        assert isinstance(ret, ast.S)
        assert isinstance(ret.statement, ast.Return)
        exp = ret.statement.exp
        assert exp is not None
        assert isinstance(exp, ast.Constant)
        assert exp.value == ast.ConstInt(7)

    def test_double_matches_default_when_only_int_association(self) -> None:
        """``_Generic(d, int: 7, default: 0)`` with double ``d`` folds to ``0`` (default)."""
        body = (
            _vd("d", var_type=ast.DoubleType()),
            *_return_generic(
                _int_var("d"),
                ((ast.IntType(), _int_const(7)), (None, _int_const(0))),
            ),
        )
        resolved = _resolved_body(body)
        ret = resolved[1]
        assert isinstance(ret, ast.S)
        assert isinstance(ret.statement, ast.Return)
        exp = ret.statement.exp
        assert exp is not None
        assert isinstance(exp, ast.Constant)
        assert exp.value == ast.ConstInt(0)

    def test_double_matches_double_association(self) -> None:
        """``_Generic(d, int: 1, double: 2, default: 0)`` with double ``d`` folds to ``2``."""
        body = (
            _vd("d", var_type=ast.DoubleType()),
            *_return_generic(
                _int_var("d"),
                (
                    (ast.IntType(), _int_const(1)),
                    (ast.DoubleType(), _int_const(2)),
                    (None, _int_const(0)),
                ),
            ),
        )
        resolved = _resolved_body(body)
        ret = resolved[1]
        assert isinstance(ret, ast.S)
        assert isinstance(ret.statement, ast.Return)
        exp = ret.statement.exp
        assert exp is not None
        assert isinstance(exp, ast.Constant)
        assert exp.value == ast.ConstInt(2)

    def test_long_matches_long_association(self) -> None:
        """``_Generic(n, long: 5, default: 0)`` with long ``n`` folds to ``5``."""
        body = (
            _vd("n", var_type=ast.LongType()),
            *_return_generic(
                _int_var("n"),
                ((ast.LongType(), _int_const(5)), (None, _int_const(0))),
            ),
        )
        resolved = _resolved_body(body)
        ret = resolved[1]
        assert isinstance(ret, ast.S)
        assert isinstance(ret.statement, ast.Return)
        exp = ret.statement.exp
        assert exp is not None
        assert isinstance(exp, ast.Constant)
        assert exp.value == ast.ConstInt(5)

    def test_selected_expression_is_type_checked(self) -> None:
        """The selected expression receives the correct ``exp_type`` annotation."""
        body = (
            _vd("x"),
            *_return_generic(
                _int_var("x"),
                ((ast.IntType(), _int_const(42)), (None, _int_const(0))),
            ),
        )
        resolved = _resolved_body(body)
        ret = resolved[1]
        assert isinstance(ret, ast.S)
        assert isinstance(ret.statement, ast.Return)
        exp = ret.statement.exp
        assert exp is not None
        assert exp.exp_type is not None


# ---------------------------------------------------------------------------
# Array controlling expression (array-to-pointer decay)
# ---------------------------------------------------------------------------


class TestGenericArrayDecay:
    """Array controlling expression decays to pointer for type matching."""

    def test_array_matches_pointer_association(self) -> None:
        """``_Generic(arr, int *: 1, default: 0)`` with ``int arr[3]`` folds to ``1``."""
        body = (
            _vd("arr", var_type=ast.Array(ast.IntType(), 3)),
            *_return_generic(
                _int_var("arr"),
                ((ast.Pointer(ast.IntType()), _int_const(1)), (None, _int_const(0))),
            ),
        )
        resolved = _resolved_body(body)
        ret = resolved[1]
        assert isinstance(ret, ast.S)
        assert isinstance(ret.statement, ast.Return)
        exp = ret.statement.exp
        assert exp is not None
        assert isinstance(exp, ast.Constant)
        assert exp.value == ast.ConstInt(1)


# ---------------------------------------------------------------------------
# Qualifier stripping
# ---------------------------------------------------------------------------


class TestGenericQualifierStripping:
    """Qualified controlling expression types are stripped before matching."""

    def test_const_int_matches_int_association(self) -> None:
        """``const int x`` controlling expression matches ``int`` association."""
        body = (
            _vd("x", var_type=ast.QualifiedType(base=ast.IntType(), const=True)),
            *_return_generic(
                _int_var("x"),
                ((ast.IntType(), _int_const(99)), (None, _int_const(0))),
            ),
        )
        resolved = _resolved_body(body)
        ret = resolved[1]
        assert isinstance(ret, ast.S)
        assert isinstance(ret.statement, ast.Return)
        exp = ret.statement.exp
        assert exp is not None
        assert isinstance(exp, ast.Constant)
        assert exp.value == ast.ConstInt(99)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestGenericErrors:
    """SA raises ``SemanticError`` for invalid ``_Generic`` expressions."""

    def test_no_match_no_default_raises(self) -> None:
        """No matching type and no default → ``SemanticError``."""
        body = (
            _vd("x"),
            *_return_generic(
                _int_var("x"),
                ((ast.DoubleType(), _int_const(1)),),
            ),
        )
        with pytest.raises(SemanticError):
            resolve(_program(body))

    def test_duplicate_type_in_associations_raises(self) -> None:
        """Two associations with the same type → ``SemanticError``."""
        body = (
            _vd("x"),
            *_return_generic(
                _int_var("x"),
                (
                    (ast.IntType(), _int_const(1)),
                    (ast.IntType(), _int_const(2)),
                    (None, _int_const(0)),
                ),
            ),
        )
        with pytest.raises(SemanticError):
            resolve(_program(body))

    def test_multiple_defaults_raises(self) -> None:
        """Two ``default`` associations → ``SemanticError``."""
        body = (
            _vd("x"),
            *_return_generic(
                _int_var("x"),
                (
                    (None, _int_const(1)),
                    (None, _int_const(2)),
                ),
            ),
        )
        with pytest.raises(SemanticError):
            resolve(_program(body))
