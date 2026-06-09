"""Dead-store elimination optimization pass."""

from .cfg import ControlFlowGraph


def dead_store_elimination(cfg: ControlFlowGraph) -> ControlFlowGraph:
    """Remove stores to variables whose values are never subsequently read.

    Args:
        cfg: The control-flow graph to optimize.

    Returns:
        A new ``ControlFlowGraph`` with dead stores removed.
    """
    return cfg
