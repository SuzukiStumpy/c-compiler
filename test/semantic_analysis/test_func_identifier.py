"""Tests for the ``__func__`` predefined identifier (C99 §6.4.2.2).

Verifies that ``SemanticAnalyser`` replaces every ``Var(name="__func__")``
with a ``String`` node whose content is the name of the enclosing function,
and that using ``__func__`` outside a function body raises ``SemanticError``.
"""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import (
    _get_fn,
    _make_prog,
    _prog_with_decls,
    _resolved_body,
)

# Helper: a block item that evaluates __func__ as an expression statement,
# compatible with any function return type.
_FUNC_EXPR: tuple[ast.BlockItem, ...] = (ast.S(ast.Expression(ast.Var(name="__func__"))),)


def _unwrap_string(exp: ast.Exp) -> ast.String:
    """Extract the inner ``String`` from a value-context expression.

    In a value context, array-to-pointer decay wraps the ``String`` in an
    ``AddrOf`` node.  This helper unwraps that outer node so tests can check
    the string content directly.

    Args:
        exp: The resolved expression, either a bare ``String`` or ``AddrOf(String(...))``.

    Returns:
        The inner ``ast.String`` node.

    Raises:
        AssertionError: If ``exp`` is neither ``String`` nor ``AddrOf(String(…))``.
    """
    if isinstance(exp, ast.String):
        return exp
    assert isinstance(exp, ast.AddrOf) and isinstance(exp.exp, ast.String)
    return exp.exp


class TestFuncIdentifier:
    """``__func__`` is expanded to the enclosing function's name as a ``String``."""

    def test_func_in_main_resolves_to_string_main(self) -> None:
        """``__func__`` inside ``main`` resolves to ``String(b"main")``."""
        resolved = _resolved_body(_FUNC_EXPR)
        item = resolved[0]
        assert isinstance(item, ast.S)
        assert isinstance(item.statement, ast.Expression)
        s = _unwrap_string(item.statement.exp)
        assert s.string == b"main"

    def test_func_reflects_function_name(self) -> None:
        """``__func__`` inside ``calculate`` resolves to ``String(b"calculate")``."""
        fn = ast.Function(
            name="calculate",
            params=(),
            body=_FUNC_EXPR,
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        resolved = _get_fn(resolve(_make_prog(fn)))
        item = resolved.body[0]
        assert isinstance(item, ast.S)
        assert isinstance(item.statement, ast.Expression)
        s = _unwrap_string(item.statement.exp)
        assert s.string == b"calculate"

    def test_func_has_correct_type(self) -> None:
        """``__func__`` resolves to ``Array(Char(), len + 1)`` covering the NUL terminator."""
        resolved = _resolved_body(_FUNC_EXPR)
        item = resolved[0]
        assert isinstance(item, ast.S)
        assert isinstance(item.statement, ast.Expression)
        s = _unwrap_string(item.statement.exp)
        assert s.exp_type == ast.Array(element=ast.Char(), size=len(b"main") + 1)

    def test_func_each_function_gets_its_own_name(self) -> None:
        """Separate functions each see their own ``__func__``."""
        foo = ast.Function(
            name="foo",
            params=(),
            body=_FUNC_EXPR,
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        bar = ast.Function(
            name="bar",
            params=(),
            body=_FUNC_EXPR,
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        prog = resolve(_make_prog(foo, bar))
        foo_body = _get_fn(prog, 0).body
        bar_body = _get_fn(prog, 1).body

        foo_item = foo_body[0]
        assert isinstance(foo_item, ast.S)
        assert isinstance(foo_item.statement, ast.Expression)
        assert _unwrap_string(foo_item.statement.exp).string == b"foo"

        bar_item = bar_body[0]
        assert isinstance(bar_item, ast.S)
        assert isinstance(bar_item.statement, ast.Expression)
        assert _unwrap_string(bar_item.statement.exp).string == b"bar"

    def test_func_outside_function_raises_semantic_error(self) -> None:
        """Using ``__func__`` in a file-scope initialiser is a semantic error.

        File-scope initialiser validation rejects ``__func__`` before name resolution
        (it is neither a constant nor an address-of-variable); any SemanticError is acceptable.
        """
        file_var = ast.VarDecl(
            ast.VariableDeclaration(
                name="x",
                init=ast.SingleInit(ast.Var(name="__func__")),
                var_type=ast.Pointer(ast.Char()),
            )
        )
        with pytest.raises(SemanticError):
            resolve(_prog_with_decls(file_var))
