"""Unreachable-code elimination optimization pass."""

import compiler.syntax.tac as tac

from .cfg import TAC_ADAPTER, BlockId, ControlFlowGraph, add_all_edges


def _truncate_at_first_terminator(cfg: ControlFlowGraph[tac.Instruction]) -> None:
    """Remove instructions after the first unconditional terminator in each block.

    A ``Return`` or unconditional ``Jump`` is an unconditional terminator.
    Instructions following one within the same block are unreachable: the
    terminator always redirects control before they can execute.

    This situation arises after constant folding converts a conditional jump
    (e.g. ``JumpIfZero(Constant(0), target)``) into an unconditional one,
    stranding the instructions from the "not-taken" path in the same block.

    Args:
        cfg: The control-flow graph to mutate in place.
    """
    for block in cfg.blocks:
        for i, instr in enumerate(block.instructions):
            is_unconditional_terminator = TAC_ADAPTER.is_return(instr) or TAC_ADAPTER.jump_target(instr) is not None
            if is_unconditional_terminator and i + 1 < len(block.instructions):
                del block.instructions[i + 1 :]
                break


def _find_reachable_blocks(cfg: ControlFlowGraph[tac.Instruction]) -> set[int]:
    """Return the set of block indices reachable from the entry block (index 0).

    Traverses the CFG from ``blocks[0]`` following both explicit jump targets
    (scanning *every* instruction in each block, not just the last) and
    fall-through edges derived from each block's last instruction.

    Scanning all instructions is necessary because the short-circuit ``&&``/``||``
    TAC pattern places multiple conditional jumps inside a single block with no
    intervening labels; those mid-block targets are invisible to a scan that
    only examines the final instruction.

    Args:
        cfg: The control-flow graph to analyse.

    Returns:
        A set of block indices that are reachable from ``blocks[0]``.
    """
    if not cfg.blocks:
        return set()

    label_index: dict[str, int] = {block.label: i for i, block in enumerate(cfg.blocks) if block.label is not None}

    reachable: set[int] = {0}
    worklist = [0]

    while worklist:
        i = worklist.pop()
        block = cfg.blocks[i]

        for instr in block.instructions:
            for target_label in (TAC_ADAPTER.jump_target(instr), TAC_ADAPTER.conditional_jump_target(instr)):
                if target_label is not None and target_label in label_index:
                    j = label_index[target_label]
                    if j not in reachable:
                        reachable.add(j)
                        worklist.append(j)

        if i + 1 < len(cfg.blocks):
            last = block.instructions[-1] if block.instructions else None
            falls_through = last is None or (not TAC_ADAPTER.is_return(last) and TAC_ADAPTER.jump_target(last) is None)
            if falls_through and (i + 1) not in reachable:
                reachable.add(i + 1)
                worklist.append(i + 1)

    return reachable


def _remove_redundant_jumps(cfg: ControlFlowGraph[tac.Instruction]) -> None:
    """Remove trailing jump instructions that transfer control to the default fall-through block.

    For every block except the last, if the final instruction is a jump (unconditional
    or conditional) and every successor of that block is the immediately following block,
    the jump is redundant: execution would fall through to the same destination without it.
    The instruction is removed and ``add_all_edges`` is called to rebuild successor lists.

    Args:
        cfg: The control-flow graph to mutate in place.
    """
    changed = False
    for i in range(len(cfg.blocks) - 1):
        block = cfg.blocks[i]
        if not block.instructions:
            continue

        last = block.instructions[-1]
        is_jump = TAC_ADAPTER.jump_target(last) is not None or TAC_ADAPTER.conditional_jump_target(last) is not None
        if not is_jump:
            continue

        default_succ = BlockId(i + 1)
        keep_jump = any(succ != default_succ for succ in block.successors)
        if not keep_jump:
            block.instructions.pop()
            changed = True

    if changed:
        add_all_edges(cfg, TAC_ADAPTER)


def _remove_useless_labels(cfg: ControlFlowGraph[tac.Instruction]) -> None:
    """Strip label instructions that are never the target of any jump.

    A label is useless when no ``Jump``, ``JumpIfZero``, or ``JumpIfNotZero``
    instruction in the CFG names it as a target.  Such a block is only entered
    by falling through from its immediate predecessor, so the label serves no
    purpose.  The leading ``tac.Label`` instruction is removed from the block's
    instruction list and ``block.label`` is cleared to ``None``.

    **Guard**: when the predecessor's last instruction is a jump *and* the block
    contains non-label instructions, the label is kept.  Removing it would merge
    those instructions into the predecessor in the flat output, burying the
    predecessor's jump mid-block where ``add_all_edges`` (which only inspects
    the final instruction) could not see it.  If the block contains *only* the
    label instruction (it would become empty after removal), the guard is skipped:
    an empty block carries no instructions and is safely removed by
    ``_remove_empty_blocks`` without any mid-block merging.

    Args:
        cfg: The control-flow graph to mutate in place.
    """
    jump_targets: set[str] = set()
    for block in cfg.blocks:
        for instr in block.instructions:
            target = TAC_ADAPTER.jump_target(instr)
            if target is not None:
                jump_targets.add(target)
            target = TAC_ADAPTER.conditional_jump_target(instr)
            if target is not None:
                jump_targets.add(target)

    for i, block in enumerate(cfg.blocks):
        if block.label is None or block.label in jump_targets:
            continue
        # Apply the predecessor-jump guard only when the block has instructions
        # beyond the label itself.  A label-only block becomes empty after
        # removal and is handled safely by _remove_empty_blocks.
        if i > 0 and len(block.instructions) > 1:
            pred = cfg.blocks[i - 1]
            if pred.instructions:
                last_pred = pred.instructions[-1]
                pred_jumps = TAC_ADAPTER.jump_target(last_pred) or TAC_ADAPTER.conditional_jump_target(last_pred)
                if pred_jumps is not None:
                    continue
        if block.instructions and TAC_ADAPTER.label_name(block.instructions[0]) == block.label:
            block.instructions.pop(0)
        block.label = None


def _remove_empty_blocks(cfg: ControlFlowGraph[tac.Instruction]) -> None:
    """Remove basic blocks that contain no instructions.

    After label and jump cleanup, a block may be left with an empty instruction
    list.  Because phase 3 only removes labels that are not jump targets, no
    remaining jump instruction names an empty block's label.  Removing the block
    and calling ``add_all_edges`` is therefore sufficient to rewire all edges.

    The entry block (``blocks[0]``) is kept even when empty so that the implicit
    ``ENTRY`` edge always has a valid target.

    Args:
        cfg: The control-flow graph to mutate in place.
    """
    empty = {i for i, block in enumerate(cfg.blocks) if not block.instructions}
    empty.discard(0)  # never remove the entry block
    if not empty:
        return
    cfg.blocks = [block for i, block in enumerate(cfg.blocks) if i not in empty]
    add_all_edges(cfg, TAC_ADAPTER)


def unreachable_code_elimination(cfg: ControlFlowGraph[tac.Instruction]) -> ControlFlowGraph[tac.Instruction]:
    """Remove basic blocks that can never be reached during execution, then strip redundant jumps and labels.

    **Phase 0 — intra-block dead code removal**: Each block is truncated at its
    first unconditional terminator (``Return`` or unconditional ``Jump``).
    Instructions after a terminator are unreachable within the block and are
    deleted before any block-level analysis is performed.

    **Phase 1 — dead block removal**: Reachability is computed by a BFS/DFS
    from ``blocks[0]``, following both explicit jump targets (scanning *every*
    instruction in each reachable block) and fall-through edges.  Blocks not
    reached by this traversal are removed.  Unlike a simple inbound-edge count,
    the traversal correctly identifies dead cycles: a group of blocks that refer
    to each other but have no path from ``ENTRY`` all appear unreachable.

    **Phase 2 — redundant jump removal**: Any trailing jump whose every successor
    is the immediately following block is removed because the fall-through
    behaviour is identical.

    **Phase 3 — useless label removal**: Any label that is not the target of a
    remaining jump instruction is stripped from the start of its block, provided
    either (a) the predecessor does not end with a jump, or (b) the block contains
    only the label instruction (so removal produces an empty block rather than
    merging instructions into the predecessor mid-block).

    **Phase 4 — empty block removal**: Blocks left empty by phase 3 are removed
    and edges are rewired via ``add_all_edges``.

    ``add_all_edges`` is called after phase 1 (and again inside phases 2 and 4
    when they make changes) so that successor edges are always consistent with
    the current block list.

    Args:
        cfg: The control-flow graph to optimize, mutated in place.

    Returns:
        The same ``ControlFlowGraph`` with unreachable blocks removed, redundant
        jumps stripped, useless labels removed, empty blocks removed, and
        successor edges rewired.
    """
    if not cfg.blocks:
        return cfg

    # Phase 0: truncate each block at its first unconditional terminator.
    _truncate_at_first_terminator(cfg)

    # Phase 1: remove blocks unreachable from ENTRY via reachability traversal.
    # Reachability handles dead cycles (every block in the cycle has a predecessor
    # but none is reachable from blocks[0]) that an inbound-edge count misses.
    reachable = _find_reachable_blocks(cfg)
    if len(reachable) < len(cfg.blocks):
        cfg.blocks = [block for i, block in enumerate(cfg.blocks) if i in reachable]

    # Rebuild successor edges after phases 0 and 1; required by phase 2.
    add_all_edges(cfg, TAC_ADAPTER)

    _remove_redundant_jumps(cfg)
    _remove_useless_labels(cfg)
    _remove_empty_blocks(cfg)
    return cfg
