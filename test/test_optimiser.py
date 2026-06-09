"""Tests for the optimization pipeline.

Verifies ``make_control_flow_graph``, ``cfg_to_instructions``, ``optimize_function``,
and ``do_optimization`` in isolation, building TAC nodes directly without invoking
the lexer, parser, or TAC generator.
"""

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from compiler.optimiser import do_optimization
from compiler.optimiser.cfg import ControlFlowGraph, cfg_to_instructions, make_control_flow_graph
from compiler.optimiser.main import optimize_function
from compiler.types.optimizations import Optimizations

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NO_OPT = Optimizations(
    fold_constants=False,
    propagate_copies=False,
    eliminate_unreachable_code=False,
    eliminate_dead_stores=False,
)

_ALL_OPT = Optimizations(
    fold_constants=True,
    propagate_copies=True,
    eliminate_unreachable_code=True,
    eliminate_dead_stores=True,
)


def _function(body: tuple[tac.Instruction, ...], name: str = "f") -> tac.Function:
    """Build a minimal ``tac.Function`` with the given instruction body.

    Args:
        body: The instruction sequence to use as the function body.
        name: The function name (defaults to ``"f"``).

    Returns:
        A ``tac.Function`` with ``globl=False``, no params, and no source info.
    """
    return tac.Function(name=name, body=body)


def _const(val: int) -> tac.Constant:
    """Return a ``tac.Constant`` wrapping a ``ConstInt``.

    Args:
        val: The integer value.

    Returns:
        A ``tac.Constant`` with ``ConstInt(val)``.
    """
    return tac.Constant(value=ast.ConstInt(val))


def _var(name: str) -> tac.Var:
    """Return a ``tac.Var`` with the given name.

    Args:
        name: The variable name.

    Returns:
        A ``tac.Var`` wrapping ``name``.
    """
    return tac.Var(value=name)


# ---------------------------------------------------------------------------
# make_control_flow_graph
# ---------------------------------------------------------------------------


class TestMakeControlFlowGraph:
    """Unit tests for ``make_control_flow_graph``."""

    def test_empty_body_produces_no_blocks(self) -> None:
        """A function with an empty body produces a CFG with no blocks."""
        cfg = make_control_flow_graph(_function(()))
        assert cfg.blocks == []

    def test_no_labels_produces_single_unlabelled_block(self) -> None:
        """Instructions without any label land in a single block with ``label=None``."""
        body = (tac.Copy(src=_const(1), dst=_var("x")), tac.Return(val=_var("x")))
        cfg = make_control_flow_graph(_function(body))
        assert len(cfg.blocks) == 1
        assert cfg.blocks[0].label is None
        assert tuple(cfg.blocks[0].instructions) == body

    def test_single_label_at_start_produces_one_labelled_block(self) -> None:
        """A body that opens with a label produces one block whose label matches."""
        body = (tac.Label("entry"), tac.Return(val=_const(0)))
        cfg = make_control_flow_graph(_function(body))
        assert len(cfg.blocks) == 1
        assert cfg.blocks[0].label == "entry"

    def test_label_splits_into_entry_block_and_labelled_block(self) -> None:
        """Instructions before the first label go into an unlabelled entry block."""
        body = (
            tac.Copy(src=_const(1), dst=_var("x")),
            tac.Label("after"),
            tac.Return(val=_var("x")),
        )
        cfg = make_control_flow_graph(_function(body))
        assert len(cfg.blocks) == 2
        assert cfg.blocks[0].label is None
        assert cfg.blocks[0].instructions == [tac.Copy(src=_const(1), dst=_var("x"))]
        assert cfg.blocks[1].label == "after"
        assert cfg.blocks[1].instructions == [tac.Label("after"), tac.Return(val=_var("x"))]

    def test_multiple_labels_produce_multiple_blocks(self) -> None:
        """Each label starts a new block; block count equals the number of labels."""
        body = (
            tac.Label("a"),
            tac.Jump(target="b"),
            tac.Label("b"),
            tac.Return(val=_const(0)),
        )
        cfg = make_control_flow_graph(_function(body))
        assert len(cfg.blocks) == 2
        assert cfg.blocks[0].label == "a"
        assert cfg.blocks[1].label == "b"

    def test_label_instruction_is_first_in_its_block(self) -> None:
        """The ``Label`` instruction itself is stored as the first element of the block."""
        body = (tac.Label("loop"), tac.Return(val=_const(0)))
        cfg = make_control_flow_graph(_function(body))
        assert cfg.blocks[0].instructions[0] == tac.Label("loop")

    def test_function_metadata_is_carried_through(self) -> None:
        """Name, ``globl``, params, and source info are preserved in the CFG."""
        fn = tac.Function(
            name="my_fn",
            globl=True,
            params=("a", "b"),
            body=(tac.Return(val=_const(0)),),
            source_file="main.c",
            source_line=10,
        )
        cfg = make_control_flow_graph(fn)
        assert cfg.name == "my_fn"
        assert cfg.globl is True
        assert cfg.params == ("a", "b")
        assert cfg.source_file == "main.c"
        assert cfg.source_line == 10


# ---------------------------------------------------------------------------
# cfg_to_instructions — round-trip
# ---------------------------------------------------------------------------


class TestCfgToInstructions:
    """Unit tests for ``cfg_to_instructions`` and the CFG round-trip invariant."""

    def test_round_trip_empty_body(self) -> None:
        """Round-tripping an empty-body function returns an equal function."""
        fn = _function(())
        assert cfg_to_instructions(make_control_flow_graph(fn)) == fn

    def test_round_trip_no_labels(self) -> None:
        """Round-tripping a label-free body preserves instruction order."""
        body = (tac.Copy(src=_const(1), dst=_var("x")), tac.Return(val=_var("x")))
        fn = _function(body)
        assert cfg_to_instructions(make_control_flow_graph(fn)) == fn

    def test_round_trip_with_labels(self) -> None:
        """Round-tripping a body with labels preserves every instruction."""
        body = (
            tac.Copy(src=_const(1), dst=_var("x")),
            tac.Label("check"),
            tac.JumpIfZero(condition=_var("x"), target="done"),
            tac.Label("done"),
            tac.Return(val=_var("x")),
        )
        fn = _function(body)
        assert cfg_to_instructions(make_control_flow_graph(fn)) == fn

    def test_empty_cfg_produces_function_with_empty_body(self) -> None:
        """A CFG with no blocks flattens to a function with an empty body."""
        cfg = ControlFlowGraph(name="f", globl=False, params=(), blocks=[], source_file="", source_line=0)
        result = cfg_to_instructions(cfg)
        assert result.body == ()


# ---------------------------------------------------------------------------
# optimize_function
# ---------------------------------------------------------------------------


class TestOptimizeFunction:
    """Unit tests for the ``optimize_function`` fixed-point loop."""

    def test_empty_body_returns_unchanged(self) -> None:
        """The early-out for empty bodies returns the original function object."""
        fn = _function(())
        assert optimize_function(fn, _ALL_OPT) is fn

    def test_no_passes_enabled_returns_equal_function(self) -> None:
        """With all passes disabled the loop converges immediately."""
        body = (tac.Return(val=_const(42)),)
        fn = _function(body)
        result = optimize_function(fn, _NO_OPT)
        assert result == fn

    def test_all_passes_enabled_converges(self) -> None:
        """With all passes enabled the optimization loop converges to a fixed point."""
        body = (tac.Copy(src=_const(1), dst=_var("x")), tac.Return(val=_var("x")))
        fn = _function(body)
        result = optimize_function(fn, _ALL_OPT)
        expected = (tac.Copy(src=_const(1), dst=_var("x")), tac.Return(val=_const(1)))
        assert result.body == expected

    def test_result_body_matches_input_body(self) -> None:
        """The returned function has the same instruction body as the input (no-op passes)."""
        body = (tac.Return(val=_const(0)),)
        fn = _function(body)
        assert optimize_function(fn, _NO_OPT).body == body


# ---------------------------------------------------------------------------
# optimize_function — unreachable-code elimination
# ---------------------------------------------------------------------------

_UCE_ONLY = Optimizations(
    fold_constants=False,
    propagate_copies=False,
    eliminate_unreachable_code=True,
    eliminate_dead_stores=False,
)

_CF_UCE = Optimizations(
    fold_constants=True,
    propagate_copies=False,
    eliminate_unreachable_code=True,
    eliminate_dead_stores=False,
)


class TestOptimizeFunctionUCE:
    """Integration tests for the unreachable-code elimination pass via ``optimize_function``."""

    def test_dead_block_after_unconditional_jump_removed_from_body(self) -> None:
        """Instructions in a dead block are absent from the optimized function body."""
        body = (
            tac.Jump(target="end"),
            tac.Label("dead"),
            tac.Copy(src=_const(99), dst=_var("z")),
            tac.Label("end"),
            tac.Return(val=_const(0)),
        )
        result = optimize_function(_function(body), _UCE_ONLY)
        instrs = result.body
        assert tac.Copy(src=_const(99), dst=_var("z")) not in instrs
        assert tac.Return(val=_const(0)) in instrs

    def test_cascading_dead_blocks_removed_from_body(self) -> None:
        """Blocks dead only because their sole predecessor is also dead are both removed."""
        body = (
            tac.Jump(target="end"),
            tac.Label("dead1"),
            tac.Jump(target="dead2"),
            tac.Label("dead2"),
            tac.Copy(src=_const(7), dst=_var("w")),
            tac.Label("end"),
            tac.Return(val=_const(0)),
        )
        result = optimize_function(_function(body), _UCE_ONLY)
        instrs = result.body
        assert tac.Label("dead1") not in instrs
        assert tac.Label("dead2") not in instrs
        assert tac.Copy(src=_const(7), dst=_var("w")) not in instrs

    def test_both_conditional_branches_kept(self) -> None:
        """A conditional jump keeps instructions from both the true and false branches."""
        body = (
            tac.Label("entry"),
            tac.JumpIfZero(condition=_var("x"), target="zero"),
            tac.Label("nonzero"),
            tac.Return(val=_const(1)),
            tac.Label("zero"),
            tac.Return(val=_const(0)),
        )
        result = optimize_function(_function(body), _UCE_ONLY)
        instrs = result.body
        assert tac.Return(val=_const(1)) in instrs
        assert tac.Return(val=_const(0)) in instrs

    def test_no_dead_code_all_live_instructions_survive(self) -> None:
        """A function with no unreachable blocks keeps all non-label instructions.

        The fall-through-only label is stripped by phase 3, but the Copy and
        Return survive unchanged.
        """
        body = (
            tac.Copy(src=_const(1), dst=_var("x")),
            tac.Label("done"),
            tac.Return(val=_var("x")),
        )
        result = optimize_function(_function(body), _UCE_ONLY)
        assert tac.Copy(src=_const(1), dst=_var("x")) in result.body
        assert tac.Return(val=_var("x")) in result.body
        assert tac.Label("done") not in result.body

    def test_constant_folding_then_uce_removes_dead_branch(self) -> None:
        """Constant folding resolves a conditional to unconditional; UCE removes the dead branch.

        ``JumpIfNotZero(1, target="nonzero")`` folds to ``Jump("nonzero")``;
        the fall-through block then has no predecessors and is eliminated.
        """
        body = (
            tac.Label("entry"),
            tac.JumpIfNotZero(condition=_const(1), target="nonzero"),
            tac.Label("dead_branch"),
            tac.Return(val=_const(0)),
            tac.Label("nonzero"),
            tac.Return(val=_const(1)),
        )
        result = optimize_function(_function(body), _CF_UCE)
        instrs = result.body
        assert tac.Label("dead_branch") not in instrs
        assert tac.Return(val=_const(0)) not in instrs
        assert tac.Return(val=_const(1)) in instrs


# ---------------------------------------------------------------------------
# do_optimization
# ---------------------------------------------------------------------------


class TestDoOptimization:
    """Unit tests for the ``do_optimization`` program-level entry point."""

    def test_function_is_optimized(self) -> None:
        """Function entries are passed through ``optimize_function`` (no-op stubs)."""
        body = (tac.Return(val=_const(0)),)
        fn = _function(body)
        program = tac.Program(top_level=(fn,))
        result = do_optimization(program, _NO_OPT)
        assert result.top_level == (fn,)

    def test_static_variable_passes_through_unchanged(self) -> None:
        """``StaticVariable`` entries are not passed to ``optimize_function``."""
        from compiler.syntax.ast.types import IntType

        sv = tac.StaticVariable(name="x", globl=False, t=IntType(), init=(tac.IntInit(0),))
        program = tac.Program(top_level=(sv,))
        result = do_optimization(program, _ALL_OPT)
        assert result.top_level == (sv,)

    def test_static_constant_passes_through_unchanged(self) -> None:
        """``StaticConstant`` entries are not passed to ``optimize_function``."""
        from compiler.syntax.ast.types import IntType

        sc = tac.StaticConstant(name="k", t=IntType(), init=tac.IntInit(42))
        program = tac.Program(top_level=(sc,))
        result = do_optimization(program, _ALL_OPT)
        assert result.top_level == (sc,)

    def test_mixed_program_preserves_order(self) -> None:
        """A program mixing functions and static variables keeps its top-level order."""
        from compiler.syntax.ast.types import IntType

        fn = _function((tac.Return(val=_const(1)),))
        sv = tac.StaticVariable(name="g", globl=True, t=IntType(), init=(tac.IntInit(0),))
        program = tac.Program(top_level=(fn, sv))
        result = do_optimization(program, _NO_OPT)
        assert len(result.top_level) == 2
        assert isinstance(result.top_level[0], tac.Function)
        assert isinstance(result.top_level[1], tac.StaticVariable)
