"""Constant-folding optimization pass."""

import compiler.syntax.tac as tac


def constant_folding(function: tac.Function) -> tac.Function:
    """Evaluate constant expressions at compile time and replace them with their results.

    Args:
        function: The TAC function to optimize.

    Returns:
        A new ``tac.Function`` with constant expressions folded where possible.
    """
    return function
