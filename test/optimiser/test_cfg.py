"""Unit tests for ``compiler.optimiser.cfg``.

Covers ``make_control_flow_graph``, ``add_all_edges``, and the successor-edge
relationships produced for every terminator and fall-through case.
"""

import compiler.syntax.tac as tac
from compiler.optimiser.cfg import (
    EXIT,
    TAC_ADAPTER,
    BasicBlock,
    BlockId,
    ControlFlowGraph,
    add_all_edges,
    make_control_flow_graph,
)

from .helpers import _ci, _function, _var

# ---------------------------------------------------------------------------
# add_all_edges — edge cases
# ---------------------------------------------------------------------------


class TestAddAllEdges:
    """Unit tests for ``add_all_edges``."""

    def test_empty_cfg_is_a_no_op(self) -> None:
        """Calling ``add_all_edges`` on a CFG with no blocks does not raise."""
        cfg = ControlFlowGraph(name="f", globl=False, params=(), blocks=[], source_file="", source_line=0)
        add_all_edges(cfg, TAC_ADAPTER)
        assert cfg.blocks == []

    def test_single_block_return_successor_is_exit(self) -> None:
        """A block whose last instruction is ``Return`` has ``EXIT`` as its sole successor."""
        cfg = make_control_flow_graph(_function((tac.Return(val=_ci(0)),)))
        assert cfg.blocks[0].successors == [EXIT]

    def test_single_block_plain_instruction_successor_is_exit(self) -> None:
        """A non-terminator instruction in the last (only) block falls through to ``EXIT``."""
        cfg = make_control_flow_graph(_function((tac.Copy(src=_ci(1), dst=_var("x")),)))
        assert cfg.blocks[0].successors == [EXIT]

    def test_unconditional_jump_points_to_target_block(self) -> None:
        """``Jump(target)`` produces a single successor pointing at the target block."""
        body = (
            tac.Label("a"),
            tac.Jump(target="b"),
            tac.Label("b"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        # Block 0 ("a"): Jump("b") → BlockId(1)
        # Block 1 ("b"): Return → EXIT
        assert cfg.blocks[0].successors == [BlockId(1)]
        assert cfg.blocks[1].successors == [EXIT]

    def test_jump_if_zero_produces_target_and_fallthrough(self) -> None:
        """``JumpIfZero`` produces two successors: the jump target then the fall-through block."""
        body = (
            tac.Label("entry"),
            tac.JumpIfZero(condition=_var("x"), target="done"),
            tac.Label("body"),
            tac.Return(val=_ci(1)),
            tac.Label("done"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        # Block 0 ("entry"): JumpIfZero("done") → [BlockId(2), BlockId(1)]
        assert cfg.blocks[0].successors == [BlockId(2), BlockId(1)]

    def test_jump_if_not_zero_produces_target_and_fallthrough(self) -> None:
        """``JumpIfNotZero`` produces two successors: the jump target then the fall-through block."""
        body = (
            tac.Label("entry"),
            tac.JumpIfNotZero(condition=_var("x"), target="done"),
            tac.Label("body"),
            tac.Return(val=_ci(1)),
            tac.Label("done"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        assert cfg.blocks[0].successors == [BlockId(2), BlockId(1)]

    def test_non_terminator_in_non_last_block_falls_through(self) -> None:
        """A non-terminator as the last instruction of a non-final block falls through to the next block."""
        body = (
            tac.Copy(src=_ci(1), dst=_var("x")),
            tac.Label("done"),
            tac.Return(val=_var("x")),
        )
        cfg = make_control_flow_graph(_function(body))
        # Block 0 (unlabelled): Copy → falls through to Block 1
        assert cfg.blocks[0].successors == [BlockId(1)]
        assert cfg.blocks[1].successors == [EXIT]

    def test_conditional_jump_in_last_block_fallthrough_is_exit(self) -> None:
        """A conditional jump in the last block uses ``EXIT`` as the fall-through successor."""
        body = (
            tac.Label("loop"),
            tac.Return(val=_ci(0)),
            tac.Label("check"),
            tac.JumpIfZero(condition=_var("n"), target="loop"),
        )
        cfg = make_control_flow_graph(_function(body))
        # Block 1 ("check") is the last block; its fallthrough is EXIT
        assert cfg.blocks[1].successors == [BlockId(0), EXIT]

    def test_backward_jump_resolves_to_earlier_block(self) -> None:
        """A backward ``Jump`` resolves to a block with a lower index than the jumping block."""
        body = (
            tac.Label("loop"),
            tac.JumpIfZero(condition=_var("n"), target="done"),
            tac.Label("body"),
            tac.Jump(target="loop"),
            tac.Label("done"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        # Block 0 ("loop"): JumpIfZero("done") → [BlockId(2), BlockId(1)]
        # Block 1 ("body"): Jump("loop") → [BlockId(0)]  — backward edge
        # Block 2 ("done"): Return → [EXIT]
        assert cfg.blocks[0].successors == [BlockId(2), BlockId(1)]
        assert cfg.blocks[1].successors == [BlockId(0)]
        assert cfg.blocks[2].successors == [EXIT]

    def test_add_all_edges_is_idempotent(self) -> None:
        """Calling ``add_all_edges`` twice produces the same successor lists as calling it once."""
        body = (
            tac.Label("a"),
            tac.Jump(target="b"),
            tac.Label("b"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        first_pass = [list(b.successors) for b in cfg.blocks]
        add_all_edges(cfg, TAC_ADAPTER)
        second_pass = [list(b.successors) for b in cfg.blocks]
        assert first_pass == second_pass

    def test_empty_block_instructions_fallthrough(self) -> None:
        """A block with no instructions (manually constructed) falls through to the next node."""
        block0 = BasicBlock(label=None, instructions=[])
        block1 = BasicBlock(label="b", instructions=[tac.Return(val=_ci(0))])
        cfg = ControlFlowGraph(name="f", globl=False, params=(), blocks=[block0, block1], source_file="", source_line=0)
        add_all_edges(cfg, TAC_ADAPTER)
        assert block0.successors == [BlockId(1)]
        assert block1.successors == [EXIT]

    def test_make_control_flow_graph_wires_edges_automatically(self) -> None:
        """``make_control_flow_graph`` calls ``add_all_edges`` so successors are non-empty."""
        body = (
            tac.Label("entry"),
            tac.JumpIfZero(condition=_var("x"), target="exit"),
            tac.Label("exit"),
            tac.Return(val=_ci(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        assert all(len(b.successors) > 0 for b in cfg.blocks)
