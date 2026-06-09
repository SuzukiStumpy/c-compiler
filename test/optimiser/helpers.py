"""Shared helpers for optimiser unit tests."""

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac


def _function(body: tuple[tac.Instruction, ...], name: str = "f") -> tac.Function:
    """Build a minimal ``tac.Function`` with the given instruction body.

    Args:
        body: The instruction sequence to use as the function body.
        name: The function name (defaults to ``"f"``).

    Returns:
        A ``tac.Function`` with ``globl=False``, no params, and no source info.
    """
    return tac.Function(name=name, body=body)


def _ci(val: int) -> tac.Constant:
    """Return a ``tac.Constant`` wrapping a ``ConstInt``.

    Args:
        val: The integer value.

    Returns:
        A ``tac.Constant`` with ``ast.ConstInt(val)``.
    """
    return tac.Constant(value=ast.ConstInt(val))


def _cl(val: int) -> tac.Constant:
    """Return a ``tac.Constant`` wrapping a ``ConstLong``.

    Args:
        val: The integer value.

    Returns:
        A ``tac.Constant`` with ``ast.ConstLong(val)``.
    """
    return tac.Constant(value=ast.ConstLong(val))


def _cui(val: int) -> tac.Constant:
    """Return a ``tac.Constant`` wrapping a ``ConstUInt``.

    Args:
        val: The integer value.

    Returns:
        A ``tac.Constant`` with ``ast.ConstUInt(val)``.
    """
    return tac.Constant(value=ast.ConstUInt(val))


def _cul(val: int) -> tac.Constant:
    """Return a ``tac.Constant`` wrapping a ``ConstULong``.

    Args:
        val: The integer value.

    Returns:
        A ``tac.Constant`` with ``ast.ConstULong(val)``.
    """
    return tac.Constant(value=ast.ConstULong(val))


def _cd(val: float) -> tac.Constant:
    """Return a ``tac.Constant`` wrapping a ``ConstDouble``.

    Args:
        val: The floating-point value.

    Returns:
        A ``tac.Constant`` with ``ast.ConstDouble(val)``.
    """
    return tac.Constant(value=ast.ConstDouble(val))


def _var(name: str) -> tac.Var:
    """Return a ``tac.Var`` with the given name.

    Args:
        name: The variable name.

    Returns:
        A ``tac.Var`` wrapping ``name``.
    """
    return tac.Var(value=name)
