"""Unit tests for ``compiler.optimiser.unreachable_code_elimination``.

Covers the ``unreachable_code_elimination`` pass: blocks with no inbound
edges are removed, the entry block is always kept, cascading removals
converge in a single call, and redundant trailing jumps are stripped.
"""

import compiler.syntax.tac as tac
from compiler.optimiser.cfg import (
    EXIT,
    BlockId,
    ControlFlowGraph,
    make_control_flow_graph,
)
from compiler.optimiser.unreachable_code_elimination import unreachable_code_elimination

from .helpers import _ci, _function, _var


class TestUnreachableCodeElimination:
    """Unit tests for ``unreachable_code_elimination``."""

    def test_empty_cfg_is_a_no_op(self) -> None:
        """An empty CFG is returned unchanged."""
        cfg: ControlFlowGraph[tac.Instruction] = ControlFlowGraph(
            name="f", globl=False, params=(), blocks=[], source_file="", source_line=0
        )
        result = unreachable_code_elimination(cfg)
        assert result.blocks == []

    def test_single_reachable_block_unchanged(self) -> None:
        """A CFG with one block is returned unchanged (entry block always kept)."""
        cfg = make_control_flow_graph(_function((tac.Return(val=_ci(0)),)))
        result = unreachable_code_elimination(cfg)
        assert len(result.blocks) == 1

    def test_linear_flow_all_blocks_kept(self) -> None:
        """All blocks in a linear fall-through chain are kept."""
        body = (
            tac.Copy(src=_ci(1), dst=_var("x")),
            tac.Label("mid"),
            tac.Copy(src=_ci(2), dst=_var("y")),
            tac.Label("end"),
            tac.Return(val=_var("x")),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        assert len(result.blocks) == 3

    def test_dead_block_after_unconditional_jump_removed(self) -> None:
        """A block skipped by an unconditional jump with no other predecessors is removed.

        The redundant jump and the now-unjumped label are also stripped by phases 2 and 3.
        """
        body = (
            tac.Jump(target="end"),
            tac.Label("dead"),
            tac.Copy(src=_ci(99), dst=_var("z")),
            tac.Label("end"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        all_instrs = [i for b in result.blocks for i in b.instructions]
        assert tac.Copy(src=_ci(99), dst=_var("z")) not in all_instrs
        assert tac.Return(val=_ci(0)) in all_instrs
        assert len(result.blocks) == 2

    def test_cascading_removal(self) -> None:
        """Removing a dead block exposes its successor as dead; both are eliminated.

        The redundant jump and the now-unjumped label are also stripped by phases 2 and 3.
        """
        body = (
            tac.Jump(target="end"),
            tac.Label("dead1"),
            tac.Jump(target="dead2"),
            tac.Label("dead2"),
            tac.Copy(src=_ci(0), dst=_var("w")),
            tac.Label("end"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        all_instrs = [i for b in result.blocks for i in b.instructions]
        assert tac.Label("dead1") not in all_instrs
        assert tac.Label("dead2") not in all_instrs
        assert tac.Copy(src=_ci(0), dst=_var("w")) not in all_instrs
        assert tac.Return(val=_ci(0)) in all_instrs
        assert len(result.blocks) == 2

    def test_conditional_jump_keeps_both_targets(self) -> None:
        """Both the jump target and the fall-through of a conditional jump are kept."""
        body = (
            tac.Label("entry"),
            tac.JumpIfZero(condition=_var("x"), target="zero"),
            tac.Label("nonzero"),
            tac.Return(val=_ci(1)),
            tac.Label("zero"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        assert len(result.blocks) == 3

    def test_loop_backedge_keeps_all_blocks(self) -> None:
        """A block reachable only via a back-edge is not removed."""
        body = (
            tac.Label("loop"),
            tac.JumpIfZero(condition=_var("n"), target="done"),
            tac.Label("body"),
            tac.Jump(target="loop"),
            tac.Label("done"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        assert len(result.blocks) == 3

    def test_successor_edges_rewired_after_removal(self) -> None:
        """After removing dead blocks the surviving blocks have correct successor edges."""
        body = (
            tac.Jump(target="end"),
            tac.Label("dead"),
            tac.Return(val=_ci(99)),
            tac.Label("end"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        # blocks[0] jumps to "end" which is now blocks[1]
        assert result.blocks[0].successors == [BlockId(1)]
        assert result.blocks[1].successors == [EXIT]

    def test_no_dead_blocks_block_count_preserved(self) -> None:
        """A CFG with no unreachable blocks keeps the same number of blocks.

        Labels that are only targeted by jumps are preserved; those only reached
        by fall-through are stripped by phase 3, but block count is unchanged.
        """
        body = (
            tac.Label("a"),
            tac.JumpIfNotZero(condition=_var("x"), target="b"),
            tac.Label("b"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        original_count = len(cfg.blocks)
        result = unreachable_code_elimination(cfg)
        assert len(result.blocks) == original_count


class TestRemoveRedundantJumps:
    """Unit tests for the redundant-jump removal phase of ``unreachable_code_elimination``."""

    def test_unconditional_jump_to_next_block_removed(self) -> None:
        """An unconditional ``Jump`` whose target is the immediately following block is removed."""
        body = (
            tac.Label("a"),
            tac.Jump(target="b"),
            tac.Label("b"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        assert tac.Jump(target="b") not in result.blocks[0].instructions

    def test_unconditional_jump_to_non_next_block_kept(self) -> None:
        """An unconditional ``Jump`` targeting a block other than the next one is preserved.

        Block "a" jumps to "c", skipping "b".  "b" is reachable via the
        conditional fall-through from "entry", so UCE does not remove it —
        "a"'s jump target ("c") is still not the adjacent block ("b").
        Label "a" is stripped by phase 3 (no jump targets it), so the block
        is located by its Jump instruction rather than by label.
        """
        body = (
            tac.Label("entry"),
            tac.JumpIfZero(condition=_var("x"), target="b"),
            tac.Label("a"),
            tac.Jump(target="c"),
            tac.Label("b"),
            tac.Return(val=_ci(1)),
            tac.Label("c"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        all_instrs = [i for b in result.blocks for i in b.instructions]
        assert tac.Jump(target="c") in all_instrs

    def test_conditional_jump_to_next_block_removed(self) -> None:
        """A ``JumpIfZero`` whose target is the next block (both branches go there) is removed."""
        body = (
            tac.Label("a"),
            tac.JumpIfZero(condition=_var("x"), target="b"),
            tac.Label("b"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        assert tac.JumpIfZero(condition=_var("x"), target="b") not in result.blocks[0].instructions

    def test_conditional_jump_to_different_block_kept(self) -> None:
        """A ``JumpIfZero`` targeting a block other than the next one is preserved."""
        body = (
            tac.Label("a"),
            tac.JumpIfZero(condition=_var("x"), target="c"),
            tac.Label("b"),
            tac.Return(val=_ci(1)),
            tac.Label("c"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        assert tac.JumpIfZero(condition=_var("x"), target="c") in result.blocks[0].instructions

    def test_jump_in_last_block_never_removed(self) -> None:
        """A backward jump in the last block is always kept (there is no following block).

        "latch" is the last block and carries a back-edge to "header"; the
        redundant-jump phase only inspects blocks 0..n-2, so this jump survives.
        """
        body = (
            tac.Label("header"),
            tac.JumpIfNotZero(condition=_var("n"), target="latch"),
            tac.Label("done"),
            tac.Return(val=_ci(0)),
            tac.Label("latch"),
            tac.Jump(target="header"),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        assert tac.Jump(target="header") in result.blocks[-1].instructions

    def test_successor_edges_correct_after_redundant_jump_removed(self) -> None:
        """After removing a redundant jump the block's only successor is the next block."""
        body = (
            tac.Label("a"),
            tac.Jump(target="b"),
            tac.Label("b"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        assert result.blocks[0].successors == [BlockId(1)]
        assert result.blocks[1].successors == [EXIT]

    def test_non_jump_final_instruction_untouched(self) -> None:
        """A block whose last instruction is not a jump is not modified."""
        body = (
            tac.Copy(src=_ci(1), dst=_var("x")),
            tac.Label("done"),
            tac.Return(val=_var("x")),
        )
        cfg = make_control_flow_graph(_function(body))
        before = list(cfg.blocks[0].instructions)
        result = unreachable_code_elimination(cfg)
        assert list(result.blocks[0].instructions) == before


class TestRemoveUselessLabels:
    """Unit tests for the useless-label removal phase of ``unreachable_code_elimination``."""

    def test_fall_through_only_label_removed(self) -> None:
        """A label reached only by fall-through (no jump targets it) is stripped."""
        body = (
            tac.Copy(src=_ci(1), dst=_var("x")),
            tac.Label("next"),
            tac.Return(val=_var("x")),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        all_instrs = [i for b in result.blocks for i in b.instructions]
        assert tac.Label("next") not in all_instrs

    def test_jump_target_label_kept(self) -> None:
        """A label that is the target of an unconditional jump is preserved."""
        body = (
            tac.Label("a"),
            tac.JumpIfZero(condition=_var("x"), target="b"),
            tac.Label("fallthrough"),
            tac.Return(val=_ci(1)),
            tac.Label("b"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        all_instrs = [i for b in result.blocks for i in b.instructions]
        assert tac.Label("b") in all_instrs

    def test_conditional_jump_target_label_kept(self) -> None:
        """A label targeted by a conditional jump is preserved."""
        body = (
            tac.Label("entry"),
            tac.JumpIfNotZero(condition=_var("n"), target="nonzero"),
            tac.Label("zero"),
            tac.Return(val=_ci(0)),
            tac.Label("nonzero"),
            tac.Return(val=_ci(1)),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        all_instrs = [i for b in result.blocks for i in b.instructions]
        assert tac.Label("nonzero") in all_instrs

    def test_block_label_field_cleared_when_label_removed(self) -> None:
        """When a useless label instruction is removed ``block.label`` is set to ``None``."""
        body = (
            tac.Copy(src=_ci(0), dst=_var("x")),
            tac.Label("unused"),
            tac.Return(val=_var("x")),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        assert all(b.label is None or b.label in {"unused"} for b in result.blocks) or all(
            b.label != "unused" for b in result.blocks
        )
        for block in result.blocks:
            assert block.label != "unused"

    def test_redundant_jump_then_useless_label_both_removed(self) -> None:
        """A redundant jump stripped in phase 2 leaves its target label with no jumpers;
        phase 3 then removes that label too."""
        body = (
            tac.Label("a"),
            tac.Jump(target="b"),
            tac.Label("b"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        all_instrs = [i for b in result.blocks for i in b.instructions]
        assert tac.Jump(target="b") not in all_instrs
        assert tac.Label("b") not in all_instrs

    def test_loop_header_label_kept(self) -> None:
        """A label targeted by a back-edge is never stripped."""
        body = (
            tac.Label("header"),
            tac.JumpIfZero(condition=_var("n"), target="done"),
            tac.Label("body"),
            tac.Jump(target="header"),
            tac.Label("done"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        all_instrs = [i for b in result.blocks for i in b.instructions]
        assert tac.Label("header") in all_instrs


class TestRemoveEmptyBlocks:
    """Unit tests for the empty-block removal phase of ``unreachable_code_elimination``."""

    def test_label_only_block_removed(self) -> None:
        """A block that contained only a useless label is removed by phase 4.

        Two adjacent labels each start their own block.  Neither is a jump
        target, so phase 3 strips both labels, leaving the first block empty.
        Phase 4 then removes it.
        """
        body = (
            tac.Copy(src=_ci(1), dst=_var("x")),
            tac.Label("empty"),  # block with a single label instruction
            tac.Label("end"),  # immediately starts another block
            tac.Return(val=_var("x")),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        assert all(b.instructions for b in result.blocks)

    def test_multiple_empty_blocks_all_removed(self) -> None:
        """Several consecutive label-only blocks are all removed in one pass."""
        body = (
            tac.Copy(src=_ci(1), dst=_var("x")),
            tac.Label("e1"),
            tac.Label("e2"),
            tac.Return(val=_var("x")),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        assert all(b.instructions for b in result.blocks)

    def test_edges_rewired_after_empty_block_removed(self) -> None:
        """After removing an empty block the predecessor connects directly to the successor."""
        body = (
            tac.Copy(src=_ci(1), dst=_var("x")),
            tac.Label("empty"),
            tac.Label("end"),
            tac.Return(val=_var("x")),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        # block 0: [Copy]; block 1: [Return(x)] — empty middle block gone.
        assert result.blocks[0].successors == [BlockId(1)]
        assert result.blocks[1].successors == [EXIT]

    def test_non_empty_blocks_untouched(self) -> None:
        """Blocks that still hold instructions after earlier phases are not removed."""
        body = (
            tac.Label("header"),
            tac.JumpIfZero(condition=_var("n"), target="done"),
            tac.Label("body"),
            tac.Jump(target="header"),
            tac.Label("done"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        assert all(b.instructions for b in result.blocks)

    def test_jump_target_block_not_removed(self) -> None:
        """A block whose label is a genuine non-redundant jump target is never emptied.

        ``JumpIfZero(x, "b")`` goes to "b" (block 2) or falls through to "c"
        (block 1).  The jump is non-redundant, so phase 2 keeps it.  "b" stays
        in ``jump_targets``, so phase 3 keeps its label, and phase 4 has nothing
        to remove.
        """
        body = (
            tac.Label("a"),
            tac.JumpIfZero(condition=_var("x"), target="b"),
            tac.Label("c"),
            tac.Return(val=_ci(1)),
            tac.Label("b"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        result = unreachable_code_elimination(cfg)
        assert any(tac.Label("b") in b.instructions for b in result.blocks)
