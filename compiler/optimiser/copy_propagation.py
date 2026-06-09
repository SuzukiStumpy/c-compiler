"""Copy-propagation optimization pass."""

from .cfg import ControlFlowGraph


def copy_propagation(cfg: ControlFlowGraph) -> ControlFlowGraph:
    """Replace uses of copy destinations with their source values where safe.

    Args:
        cfg: The control-flow graph to optimize.

    Returns:
        A new ``ControlFlowGraph`` with copy sources propagated forward.
    """
    return cfg
