"""Semantic-analysis tests for C11 §6.7.4 p8 _Noreturn contract enforcement.

A function declared with ``_Noreturn`` must not return to its caller.  The
semantic analyser should raise ``SemanticError`` whenever a ``return``
statement (with or without a value) appears inside such a function body.
"""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve
from compiler.utils.exceptions import SemanticError

_INT_FUN_TYPE = ast.FunType(params=(), ret=ast.IntType())
_VOID_FUN_TYPE = ast.FunType(params=(), ret=ast.VoidType())
_MAIN = ast.FunDecl(
    ast.Function(
        name="main",
        params=(),
        body=(ast.S(ast.Return(ast.Constant(0))),),
        fun_type=_INT_FUN_TYPE,
    )
)


def _noreturn_fn(
    body: tuple[ast.BlockItem, ...] | None,
    fun_type: ast.FunType = _INT_FUN_TYPE,
) -> ast.FunDecl:
    """Build a ``_Noreturn`` ``FunDecl`` with the given body.

    Args:
        body: Block items for the definition, or ``None`` for a forward declaration.
        fun_type: The function type; defaults to ``int(void)``.

    Returns:
        An ``ast.FunDecl`` wrapping a ``Function`` with ``is_noreturn=True``.
    """
    return ast.FunDecl(
        ast.Function(
            name="die",
            params=(),
            body=body,
            fun_type=fun_type,
            is_noreturn=True,
        )
    )


class TestNoreturnForwardDeclaration:
    """A ``_Noreturn`` prototype (no body) must not raise."""

    def test_noreturn_forward_decl_accepted(self) -> None:
        """``_Noreturn void die(void);`` with no body is valid — no error."""
        prog = ast.Program(declarations=(_noreturn_fn(body=None, fun_type=_VOID_FUN_TYPE), _MAIN))
        resolve(prog)  # must not raise


class TestNoreturnBodyWithoutReturn:
    """A ``_Noreturn`` function body that contains no ``return`` is valid."""

    def test_noreturn_body_calling_exit_accepted(self) -> None:
        """A ``_Noreturn`` body with no ``return`` statement is accepted."""
        body: tuple[ast.BlockItem, ...] = (ast.S(ast.Null()),)
        prog = ast.Program(declarations=(_noreturn_fn(body=body, fun_type=_VOID_FUN_TYPE), _MAIN))
        resolve(prog)  # must not raise


class TestNoreturnReturnValueRejected:
    """A ``return <expr>;`` inside a ``_Noreturn`` function must be rejected."""

    def test_noreturn_int_return_value_raises(self) -> None:
        """``_Noreturn int die(void) { return 1; }`` must raise ``SemanticError``."""
        body: tuple[ast.BlockItem, ...] = (ast.S(ast.Return(ast.Constant(1))),)
        prog = ast.Program(declarations=(_noreturn_fn(body=body), _MAIN))
        with pytest.raises(SemanticError, match="_Noreturn"):
            resolve(prog)

    def test_noreturn_void_bare_return_raises(self) -> None:
        """``_Noreturn void die(void) { return; }`` must raise ``SemanticError``."""
        body: tuple[ast.BlockItem, ...] = (ast.S(ast.Return(None)),)
        prog = ast.Program(declarations=(_noreturn_fn(body=body, fun_type=_VOID_FUN_TYPE), _MAIN))
        with pytest.raises(SemanticError, match="_Noreturn"):
            resolve(prog)

    def test_noreturn_return_inside_if_raises(self) -> None:
        """``_Noreturn`` function with ``return`` inside an ``if`` branch must raise."""
        body: tuple[ast.BlockItem, ...] = (ast.S(ast.If(ast.Constant(1), ast.Return(ast.Constant(0)), None)),)
        prog = ast.Program(declarations=(_noreturn_fn(body=body), _MAIN))
        with pytest.raises(SemanticError, match="_Noreturn"):
            resolve(prog)


class TestNonNoreturnFunctionReturnNotAffected:
    """An ordinary function with a ``return`` must continue to work normally."""

    def test_normal_int_function_return_accepted(self) -> None:
        """A plain ``int foo(void) { return 1; }`` is not affected by the noreturn check."""
        body: tuple[ast.BlockItem, ...] = (ast.S(ast.Return(ast.Constant(1))),)
        fn = ast.FunDecl(
            ast.Function(
                name="foo",
                params=(),
                body=body,
                fun_type=_INT_FUN_TYPE,
                is_noreturn=False,
            )
        )
        prog = ast.Program(declarations=(fn, _MAIN))
        resolve(prog)  # must not raise
