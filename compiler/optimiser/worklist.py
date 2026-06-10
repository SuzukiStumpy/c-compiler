"""Generic backward dataflow worklist algorithm.

Provides ``run_backward``, a fully generic iterative solver for backward
dataflow analyses over a ``ControlFlowGraph[_I]``.  The lattice element type
``_F`` and the instruction type ``_I`` are both free type parameters so the
same engine can drive liveness analysis over TAC and, later, over assembly
without modification.

Analysis-specific details (meet semantics, transfer semantics, initial lattice
value) are supplied as plain callables, typically closures that capture any
extra parameters they need.
"""

from collections import deque
from collections.abc import Callable
from typing import Any, TypeVar

from .cfg import BasicBlock, BlockId, ControlFlowGraph

_I = TypeVar("_I")
_F = TypeVar("_F")


def _build_predecessor_map(cfg: ControlFlowGraph[Any]) -> dict[int, list[int]]:
    """Build an inverted edge map from block index to predecessor block indices.

    Only concrete block-to-block edges are recorded; ``ENTRY`` and ``EXIT``
    boundary nodes are ignored because the generic worklist deals exclusively
    with concrete block indices.

    Args:
        cfg: The control-flow graph whose successor edges are already wired.

    Returns:
        A dict mapping each block index to the (possibly empty) list of block
        indices that have a direct edge *into* that block.
    """
    pred_map: dict[int, list[int]] = {i: [] for i in range(len(cfg.blocks))}
    for src_idx, block in enumerate(cfg.blocks):
        for succ_id in block.successors:
            if isinstance(succ_id, BlockId):
                pred_map[succ_id.num].append(src_idx)
    return pred_map


def run_backward(
    cfg: ControlFlowGraph[_I],
    initial: _F,
    meet_fn: Callable[[BasicBlock[_I], dict[int, _F]], _F],
    transfer_fn: Callable[[BasicBlock[_I], _F], tuple[_F, list[_F]]],
) -> tuple[dict[int, _F], dict[int, list[_F]]]:
    """Run a generic backward iterative dataflow analysis over *cfg*.

    Initialises every block's entry fact to *initial*, then iterates a worklist
    until all entry facts stabilise.  On each iteration:

    1. ``meet_fn`` derives the exit fact for the current block from the entry
       facts of its successors (as stored in *block_in*).
    2. ``transfer_fn`` propagates the exit fact backward through the block,
       returning the new entry fact and a per-instruction annotation list.
    3. When the entry fact changes, all predecessor blocks are re-queued.

    Analysis-specific concerns — boundary-node handling, lattice identity
    elements, kill/gen rules — live entirely inside the two callables.  The
    engine itself is instruction-set-agnostic.

    Args:
        cfg: The control-flow graph to analyse.  Successor edges must already
            be wired (i.e. ``add_all_edges`` must have been called).
        initial: The starting lattice value assigned to every block's entry fact
            before the first iteration.  Typically the bottom element for
            union-based analyses (e.g. ``frozenset()`` for liveness) or the top
            element for intersection-based ones.
        meet_fn: ``meet_fn(block, block_in) -> exit_fact`` — computes the fact
            at the exit of *block* by combining the entry facts of its
            successors.  *block_in* is the current mapping from block index to
            entry fact.
        transfer_fn: ``transfer_fn(block, exit_fact) -> (entry_fact, annotations)``
            — propagates *exit_fact* backward through *block*, returning the
            resulting entry fact and a per-instruction list of annotations (one
            per instruction, in forward order).

    Returns:
        A 2-tuple ``(block_in, instr_annotations)`` where *block_in* maps each
        block index to its converged entry fact and *instr_annotations* maps
        each block index to the annotation list produced by the final
        ``transfer_fn`` call for that block.
    """
    if not cfg.blocks:
        return {}, {}

    block_in: dict[int, _F] = dict.fromkeys(range(len(cfg.blocks)), initial)
    instr_annotations: dict[int, list[_F]] = {}
    pred_map = _build_predecessor_map(cfg)

    worklist: deque[int] = deque(range(len(cfg.blocks)))
    in_worklist: set[int] = set(range(len(cfg.blocks)))

    while worklist:
        block_idx = worklist.popleft()
        in_worklist.discard(block_idx)

        block = cfg.blocks[block_idx]
        old_in = block_in[block_idx]

        exit_fact = meet_fn(block, block_in)
        new_in, annots = transfer_fn(block, exit_fact)

        block_in[block_idx] = new_in
        instr_annotations[block_idx] = annots

        if new_in != old_in:
            for pred_idx in pred_map[block_idx]:
                if pred_idx not in in_worklist:
                    worklist.append(pred_idx)
                    in_worklist.add(pred_idx)

    return block_in, instr_annotations
