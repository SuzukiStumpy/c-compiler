"""Semantic analysis tests for ``_Static_assert`` declarations."""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import _program

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _static_assert(condition: ast.Exp, message: str = "assertion failed") -> ast.StaticAssert:
    """Return a ``StaticAssert`` declaration node with the given condition and message."""
    return ast.StaticAssert(condition=condition, message=message)


def _int_const(v: int) -> ast.Exp:
    """Return a ``Constant(ConstInt(v))`` expression."""
    return ast.Constant(ast.ConstInt(v))


def _sizeof_int() -> ast.Exp:
    """Return a ``SizeOfT(IntType())`` expression."""
    return ast.SizeOfT(t=ast.IntType())


def _sizeof_eq(n: int) -> ast.Exp:
    """Return ``sizeof(int) == n`` as an AST Binary expression."""
    return ast.Binary(operator=ast.Equal(), left=_sizeof_int(), right=_int_const(n))


# ---------------------------------------------------------------------------
# File-scope assertions that pass
# ---------------------------------------------------------------------------


def test_static_assert_with_nonzero_constant_passes() -> None:
    """``_Static_assert(1, "msg")`` at file scope does not raise."""
    program = ast.Program(declarations=(_static_assert(_int_const(1)),))
    resolve(program)  # should not raise


def test_static_assert_with_nonzero_constant_absent_from_resolved_program() -> None:
    """``StaticAssert`` nodes are NOT included in the resolved program (no runtime code)."""
    program = ast.Program(declarations=(_static_assert(_int_const(1)),))
    resolved = resolve(program)
    assert len(resolved.declarations) == 0


def test_static_assert_sizeof_int_eq_4_passes() -> None:
    """``_Static_assert(sizeof(int) == 4, ...)`` passes on x86-64."""
    program = ast.Program(declarations=(_static_assert(_sizeof_eq(4)),))
    resolve(program)  # should not raise


def test_static_assert_with_negative_nonzero_passes() -> None:
    """``_Static_assert(-1, ...)`` passes (non-zero)."""
    condition = ast.Unary(operator=ast.Negate(), exp=_int_const(1))
    program = ast.Program(declarations=(_static_assert(condition),))
    resolve(program)


# ---------------------------------------------------------------------------
# File-scope assertions that fail
# ---------------------------------------------------------------------------


def test_static_assert_with_zero_constant_raises_semantic_error() -> None:
    """``_Static_assert(0, "msg")`` raises ``SemanticError`` containing the message."""
    program = ast.Program(declarations=(_static_assert(_int_const(0), "expected nonzero"),))
    with pytest.raises(SemanticError, match="expected nonzero"):
        resolve(program)


def test_static_assert_sizeof_int_neq_4_fails() -> None:
    """``_Static_assert(sizeof(int) == 8, "wrong size")`` fails on x86-64 (int is 4 bytes)."""
    program = ast.Program(declarations=(_static_assert(_sizeof_eq(8), "wrong size"),))
    with pytest.raises(SemanticError, match="wrong size"):
        resolve(program)


def test_static_assert_false_error_contains_message() -> None:
    """The ``SemanticError`` for a failing assertion includes the user-supplied message."""
    program = ast.Program(declarations=(_static_assert(_int_const(0), "custom diagnostic"),))
    with pytest.raises(SemanticError, match="custom diagnostic"):
        resolve(program)


# ---------------------------------------------------------------------------
# Block-scope assertions (inside a function body)
# ---------------------------------------------------------------------------


def test_static_assert_inside_function_passes() -> None:
    """``_Static_assert(1, ...)`` inside a function body does not raise."""
    body = (ast.D(ast.StaticAssert(condition=_int_const(1), message="ok")),)
    program = _program(body)
    resolve(program)


def test_static_assert_inside_function_failure_raises() -> None:
    """``_Static_assert(0, "msg")`` inside a function body raises ``SemanticError``."""
    body = (ast.D(ast.StaticAssert(condition=_int_const(0), message="inner fail")),)
    program = _program(body)
    with pytest.raises(SemanticError, match="inner fail"):
        resolve(program)


def test_static_assert_sizeof_inside_function_passes() -> None:
    """``_Static_assert(sizeof(int) == 4, ...)`` inside a function passes."""
    body = (ast.D(ast.StaticAssert(condition=_sizeof_eq(4), message="size mismatch")),)
    program = _program(body)
    resolve(program)


# ---------------------------------------------------------------------------
# Mixed programs: assertions alongside real declarations
# ---------------------------------------------------------------------------


def test_static_assert_at_file_scope_before_function() -> None:
    """A passing file-scope assert followed by a function declaration resolves correctly."""
    fn = ast.Function(
        name="main",
        params=(),
        body=(),
        fun_type=ast.FunType(params=(), ret=ast.IntType()),
    )
    program = ast.Program(
        declarations=(
            _static_assert(_int_const(1)),
            ast.FunDecl(fn),
        )
    )
    resolved = resolve(program)
    # StaticAssert is dropped; only the FunDecl remains
    assert len(resolved.declarations) == 1
    assert isinstance(resolved.declarations[0], ast.FunDecl)


def test_static_assert_with_bitwise_expression() -> None:
    """``_Static_assert(1 & 1, ...)`` (non-zero) passes."""
    condition = ast.Binary(operator=ast.BitwiseAnd(), left=_int_const(1), right=_int_const(1))
    program = ast.Program(declarations=(_static_assert(condition),))
    resolve(program)


def test_static_assert_with_bitwise_expression_zero_fails() -> None:
    """``_Static_assert(1 & 0, "msg")`` (zero) fails."""
    condition = ast.Binary(operator=ast.BitwiseAnd(), left=_int_const(1), right=_int_const(0))
    program = ast.Program(declarations=(_static_assert(condition, "bitwise zero"),))
    with pytest.raises(SemanticError, match="bitwise zero"):
        resolve(program)
