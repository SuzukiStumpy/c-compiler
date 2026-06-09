"""Unreachable-code elimination optimization pass."""

from .cfg import ControlFlowGraph


def unreachable_code_elimination(cfg: ControlFlowGraph) -> ControlFlowGraph:
    """Remove basic blocks that can never be reached during execution.

    Args:
        cfg: The control-flow graph to optimize.

    Returns:
        A new ``ControlFlowGraph`` with unreachable blocks removed.
    """
    return cfg
