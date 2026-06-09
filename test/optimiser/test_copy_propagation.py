"""Unit tests for the copy-propagation optimisation pass.

Tests ``_replace_operand``, ``_rewrite_instruction``, and ``copy_propagation``
in isolation, building TAC nodes directly without invoking the lexer, parser,
or TAC generator.
"""

import compiler.syntax.tac as tac
from compiler.optimiser.cfg import make_control_flow_graph
from compiler.optimiser.copy_propagation import (
    _replace_operand,
    _rewrite_instruction,
    _types_compatible,
    copy_propagation,
)
from compiler.syntax.ast.types import IntType, LongType, Pointer, ULongType
from compiler.syntax.tac import operators as to
from compiler.types.symbols import LocalAttr, StaticAttr, SymbolTable, SymbolTableEntry, Tentative

from .helpers import _ci, _function, _var

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_X = _var("x")
_Y = _var("y")
_Z = _var("z")
_DST = _var("dst")

_COPY_Y_TO_X: tac.Copy = tac.Copy(src=_Y, dst=_X)
_COPY_X_TO_Y: tac.Copy = tac.Copy(src=_X, dst=_Y)
_REACHING: frozenset[tac.Copy] = frozenset({_COPY_Y_TO_X})


def _static_sym(name: str) -> SymbolTable:
    """Return a one-entry symbol table marking *name* as a static variable.

    Args:
        name: The variable name to register as static.

    Returns:
        A ``SymbolTable`` with a ``StaticAttr`` entry for *name*.
    """
    return {name: SymbolTableEntry(t=IntType(), attrs=StaticAttr(init=Tentative(), globl=False))}


def _local_sym(name: str) -> SymbolTable:
    """Return a one-entry symbol table marking *name* as a local variable.

    Args:
        name: The variable name to register as local.

    Returns:
        A ``SymbolTable`` with a ``LocalAttr`` entry for *name*.
    """
    return {name: SymbolTableEntry(t=IntType(), attrs=LocalAttr())}


# ---------------------------------------------------------------------------
# _replace_operand
# ---------------------------------------------------------------------------


class TestReplaceOperand:
    """Unit tests for ``_replace_operand``."""

    def test_constant_is_returned_unchanged(self) -> None:
        """A constant operand is never replaced regardless of the reaching set."""
        const = _ci(5)
        assert _replace_operand(const, _REACHING) is const

    def test_var_with_no_matching_copy_is_unchanged(self) -> None:
        """A variable absent from every copy's dst is returned unchanged."""
        assert _replace_operand(_Z, _REACHING) is _Z

    def test_var_matching_copy_dst_is_replaced_with_copy_src(self) -> None:
        """A variable that is the dst of a reaching copy is replaced with that copy's src."""
        assert _replace_operand(_X, _REACHING) == _Y

    def test_replacement_works_when_copy_src_is_constant(self) -> None:
        """Replacement propagates a constant source through the copy."""
        reaching = frozenset({tac.Copy(src=_ci(42), dst=_X)})
        assert _replace_operand(_X, reaching) == _ci(42)

    def test_transitive_chain_is_followed_to_constant(self) -> None:
        """A two-link chain X → Y → const is collapsed to the constant in one call."""
        reaching = frozenset(
            {
                tac.Copy(src=_Y, dst=_X),
                tac.Copy(src=_ci(99), dst=_Y),
            }
        )
        assert _replace_operand(_X, reaching) == _ci(99)

    def test_transitive_chain_stops_at_unresolved_var(self) -> None:
        """A chain X → Y stops at Y when Y has no reaching copy (Y is not a dst)."""
        reaching = frozenset({tac.Copy(src=_Y, dst=_X)})
        assert _replace_operand(_X, reaching) == _Y

    def test_cycle_does_not_loop_forever(self) -> None:
        """A cyclic reaching set (X → Y, Y → X) is handled without infinite recursion."""
        reaching = frozenset(
            {
                tac.Copy(src=_Y, dst=_X),
                tac.Copy(src=_X, dst=_Y),
            }
        )
        result = _replace_operand(_X, reaching)
        assert result in (_X, _Y)


# ---------------------------------------------------------------------------
# _types_compatible
# ---------------------------------------------------------------------------


def _sym(*pairs: tuple[str, type]) -> SymbolTable:
    """Build a symbol table with one ``LocalAttr`` entry per (name, type-class) pair.

    Args:
        *pairs: Variable-length sequence of ``(name, TypeClass)`` tuples.

    Returns:
        A ``SymbolTable`` with a ``LocalAttr`` entry for each pair.
    """
    return {name: SymbolTableEntry(t=t(), attrs=LocalAttr()) for name, t in pairs}


class TestTypesCompatible:
    """Unit tests for ``_types_compatible``."""

    def test_same_type_is_compatible(self) -> None:
        """Two variables of identical type are always compatible."""
        st = _sym(("a", LongType), ("b", LongType))
        assert _types_compatible(_var("a"), _var("b"), st)

    def test_long_and_ulong_are_not_compatible(self) -> None:
        """``LongType`` and ``ULongType`` must NOT be considered compatible.

        Allowing this copy would let CP substitute a Long source into a ULong
        destination (or vice-versa), causing the code generator to emit the
        wrong signed/unsigned comparison instruction.
        """
        st = _sym(("a", LongType), ("b", ULongType))
        assert not _types_compatible(_var("a"), _var("b"), st)
        assert not _types_compatible(_var("b"), _var("a"), st)

    def test_pointer_and_long_are_compatible(self) -> None:
        """A ``Pointer`` and a ``LongType`` are compatible (null-pointer propagation)."""
        ptr_st: SymbolTable = {
            "p": SymbolTableEntry(t=Pointer(referenced=IntType()), attrs=LocalAttr()),
            "l": SymbolTableEntry(t=LongType(), attrs=LocalAttr()),
        }
        assert _types_compatible(_var("p"), _var("l"), ptr_st)
        assert _types_compatible(_var("l"), _var("p"), ptr_st)

    def test_pointer_and_ulong_are_compatible(self) -> None:
        """A ``Pointer`` and a ``ULongType`` are compatible (null-pointer propagation)."""
        ptr_st: SymbolTable = {
            "p": SymbolTableEntry(t=Pointer(referenced=IntType()), attrs=LocalAttr()),
            "u": SymbolTableEntry(t=ULongType(), attrs=LocalAttr()),
        }
        assert _types_compatible(_var("p"), _var("u"), ptr_st)
        assert _types_compatible(_var("u"), _var("p"), ptr_st)

    def test_unknown_type_is_compatible(self) -> None:
        """A variable absent from the symbol table is treated as compatible (conservative)."""
        assert _types_compatible(_var("unknown"), _var("also_unknown"), {})


# ---------------------------------------------------------------------------
# _rewrite_instruction — Copy
# ---------------------------------------------------------------------------


class TestRewriteCopy:
    """Rewriting behaviour for ``Copy`` instructions."""

    def test_copy_exact_match_in_reaching_set_is_deleted(self) -> None:
        """A ``Copy`` already present in the reaching set is redundant and returns ``None``."""
        assert _rewrite_instruction(_COPY_Y_TO_X, _REACHING) is None

    def test_copy_reverse_reaching_copy_is_deleted(self) -> None:
        """A ``Copy(x, y)`` when ``Copy(y, x)`` reaches is redundant (x == y already)."""
        assert _rewrite_instruction(_COPY_X_TO_Y, _REACHING) is None

    def test_copy_src_is_substituted(self) -> None:
        """The src operand of a ``Copy`` is replaced when a matching reaching copy exists."""
        # Copy(x, dst): x is dst of Copy(y, x) → new Copy(y, dst)
        instr = tac.Copy(src=_X, dst=_DST)
        result = _rewrite_instruction(instr, _REACHING)
        assert result == tac.Copy(src=_Y, dst=_DST)

    def test_copy_with_no_matching_src_is_unchanged(self) -> None:
        """A ``Copy`` whose src has no matching reaching copy is returned unchanged."""
        instr = tac.Copy(src=_Z, dst=_DST)
        result = _rewrite_instruction(instr, _REACHING)
        assert result == instr


# ---------------------------------------------------------------------------
# _rewrite_instruction — arithmetic / comparison instructions
# ---------------------------------------------------------------------------


class TestRewriteArithmetic:
    """Rewriting behaviour for ``Unary`` and ``Binary``."""

    def test_unary_src_is_substituted(self) -> None:
        """The src of a ``Unary`` is replaced by the matching copy's source."""
        instr = tac.Unary(operator=to.Negate(), src=_X, dst=_DST)
        result = _rewrite_instruction(instr, _REACHING)
        assert result == tac.Unary(operator=to.Negate(), src=_Y, dst=_DST)

    def test_binary_both_srcs_are_substituted(self) -> None:
        """Both src operands of a ``Binary`` are replaced independently."""
        reaching = frozenset(
            {
                tac.Copy(src=_ci(1), dst=_X),
                tac.Copy(src=_ci(2), dst=_Y),
            }
        )
        instr = tac.Binary(operator=to.Add(), src1=_X, src2=_Y, dst=_DST)
        result = _rewrite_instruction(instr, reaching)
        assert result == tac.Binary(operator=to.Add(), src1=_ci(1), src2=_ci(2), dst=_DST)

    def test_binary_unmatched_src_is_unchanged(self) -> None:
        """A ``Binary`` src with no matching copy is left alone."""
        instr = tac.Binary(operator=to.Add(), src1=_X, src2=_Z, dst=_DST)
        result = _rewrite_instruction(instr, _REACHING)
        assert result == tac.Binary(operator=to.Add(), src1=_Y, src2=_Z, dst=_DST)


# ---------------------------------------------------------------------------
# _rewrite_instruction — control-flow instructions
# ---------------------------------------------------------------------------


class TestRewriteControlFlow:
    """Rewriting behaviour for ``Return``, ``JumpIfZero``, and ``JumpIfNotZero``."""

    def test_return_val_is_substituted(self) -> None:
        """The return value is replaced by the matching copy's source."""
        result = _rewrite_instruction(tac.Return(val=_X), _REACHING)
        assert result == tac.Return(val=_Y)

    def test_void_return_is_unchanged(self) -> None:
        """A void ``Return`` (no value) is returned unchanged."""
        instr = tac.Return()
        assert _rewrite_instruction(instr, _REACHING) is instr

    def test_jump_if_zero_condition_is_substituted(self) -> None:
        """The condition of ``JumpIfZero`` is replaced."""
        instr = tac.JumpIfZero(condition=_X, target="label")
        result = _rewrite_instruction(instr, _REACHING)
        assert result == tac.JumpIfZero(condition=_Y, target="label")

    def test_jump_if_not_zero_condition_is_substituted(self) -> None:
        """The condition of ``JumpIfNotZero`` is replaced."""
        instr = tac.JumpIfNotZero(condition=_X, target="label")
        result = _rewrite_instruction(instr, _REACHING)
        assert result == tac.JumpIfNotZero(condition=_Y, target="label")

    def test_label_is_returned_unchanged(self) -> None:
        """``Label`` has no Val operands and is returned unchanged."""
        instr = tac.Label(name="lbl")
        assert _rewrite_instruction(instr, _REACHING) is instr

    def test_jump_is_returned_unchanged(self) -> None:
        """``Jump`` has no Val operands and is returned unchanged."""
        instr = tac.Jump(target="lbl")
        assert _rewrite_instruction(instr, _REACHING) is instr


# ---------------------------------------------------------------------------
# _rewrite_instruction — type-cast and memory instructions
# ---------------------------------------------------------------------------


class TestRewriteSecondary:
    """Rewriting behaviour for type-cast and memory instructions."""

    def test_sign_extend_src_is_substituted(self) -> None:
        """The src of ``SignExtend`` is replaced."""
        instr = tac.SignExtend(src=_X, dst=_DST)
        result = _rewrite_instruction(instr, _REACHING)
        assert result == tac.SignExtend(src=_Y, dst=_DST)

    def test_truncate_src_is_substituted(self) -> None:
        """The src of ``Truncate`` is replaced."""
        instr = tac.Truncate(src=_X, dst=_DST)
        result = _rewrite_instruction(instr, _REACHING)
        assert result == tac.Truncate(src=_Y, dst=_DST)

    def test_funcall_args_are_substituted(self) -> None:
        """Each argument in a ``FunCall`` is replaced independently."""
        reaching = frozenset(
            {
                tac.Copy(src=_ci(10), dst=_X),
                tac.Copy(src=_ci(20), dst=_Y),
            }
        )
        instr = tac.FunCall(fun_val=_var("f"), args=(_X, _Y), dst=_DST)
        result = _rewrite_instruction(instr, reaching)
        assert result == tac.FunCall(fun_val=_var("f"), args=(_ci(10), _ci(20)), dst=_DST)

    def test_load_src_ptr_is_substituted(self) -> None:
        """The pointer operand of ``Load`` is replaced."""
        ptr_a, ptr_b = _var("pa"), _var("pb")
        reaching = frozenset({tac.Copy(src=ptr_a, dst=ptr_b)})
        instr = tac.Load(src_ptr=ptr_b, dst=_DST)
        result = _rewrite_instruction(instr, reaching)
        assert result == tac.Load(src_ptr=ptr_a, dst=_DST)

    def test_store_src_and_dst_ptr_are_substituted(self) -> None:
        """Both operands of ``Store`` are replaced."""
        ptr = _var("p")
        reaching = frozenset(
            {
                tac.Copy(src=_ci(99), dst=_X),
                tac.Copy(src=ptr, dst=_var("q")),
            }
        )
        instr = tac.Store(src=_X, dst_ptr=_var("q"))
        result = _rewrite_instruction(instr, reaching)
        assert result == tac.Store(src=_ci(99), dst_ptr=ptr)

    def test_add_ptr_operands_are_substituted(self) -> None:
        """Both ptr and index operands of ``AddPtr`` are replaced."""
        base = _var("base")
        idx = _var("idx")
        reaching = frozenset(
            {
                tac.Copy(src=base, dst=_X),
                tac.Copy(src=idx, dst=_Y),
            }
        )
        instr = tac.AddPtr(ptr=_X, index=_Y, scale=4, dst=_DST)
        result = _rewrite_instruction(instr, reaching)
        assert result == tac.AddPtr(ptr=base, index=idx, scale=4, dst=_DST)

    def test_copy_to_offset_src_is_substituted(self) -> None:
        """The src of ``CopyToOffset`` is replaced (dst is a str, not a Val)."""
        instr = tac.CopyToOffset(src=_X, dst="agg", offset=0)
        result = _rewrite_instruction(instr, _REACHING)
        assert result == tac.CopyToOffset(src=_Y, dst="agg", offset=0)

    def test_get_address_src_is_not_substituted(self) -> None:
        """``GetAddress`` src is intentionally excluded: value equality does not imply address equality."""
        instr = tac.GetAddress(src=_X, dst=_DST)
        result = _rewrite_instruction(instr, _REACHING)
        assert result is instr


# ---------------------------------------------------------------------------
# copy_propagation — end-to-end
# ---------------------------------------------------------------------------


class TestCopyPropagation:
    """Integration tests for the full ``copy_propagation`` pass."""

    def test_empty_cfg_is_a_no_op(self) -> None:
        """An empty CFG is returned unchanged without error."""
        from compiler.optimiser.cfg import ControlFlowGraph

        cfg: ControlFlowGraph[tac.Instruction] = ControlFlowGraph(
            name="f", globl=False, params=(), blocks=[], source_file="", source_line=0
        )
        result = copy_propagation(cfg)
        assert result.blocks == []

    def test_use_replaced_by_copy_source(self) -> None:
        """A variable that is the destination of a reaching copy is replaced at use sites."""
        body = (
            tac.Copy(src=_Y, dst=_X),
            tac.Return(val=_X),
        )
        cfg = make_control_flow_graph(_function(body))
        copy_propagation(cfg)
        instrs = [i for b in cfg.blocks for i in b.instructions]
        assert tac.Return(val=_Y) in instrs

    def test_redundant_copy_is_eliminated(self) -> None:
        """A ``Copy`` that restates an already-reaching copy is deleted."""
        body = (
            tac.Copy(src=_Y, dst=_X),
            tac.Copy(src=_Y, dst=_X),
            tac.Return(val=_X),
        )
        cfg = make_control_flow_graph(_function(body))
        copy_propagation(cfg)
        instrs = [i for b in cfg.blocks for i in b.instructions]
        assert instrs.count(tac.Copy(src=_Y, dst=_X)) == 1

    def test_copy_killed_by_reassignment(self) -> None:
        """A copy is no longer reaching after the destination is reassigned."""
        body = (
            tac.Copy(src=_Y, dst=_X),
            tac.Copy(src=_ci(99), dst=_X),
            tac.Return(val=_X),
        )
        cfg = make_control_flow_graph(_function(body))
        copy_propagation(cfg)
        instrs = [i for b in cfg.blocks for i in b.instructions]
        # After reassignment x=99, Return should propagate 99, not y.
        assert tac.Return(val=_ci(99)) in instrs
        assert tac.Return(val=_Y) not in instrs

    def test_copy_propagates_through_conditional_branch(self) -> None:
        """A copy that reaches all predecessors of a join point is propagated there."""
        # Both branches copy y→x, so x==y is known at the return.
        body = (
            tac.JumpIfZero(condition=_Z, target="else"),
            tac.Copy(src=_Y, dst=_X),
            tac.Jump(target="end"),
            tac.Label("else"),
            tac.Copy(src=_Y, dst=_X),
            tac.Label("end"),
            tac.Return(val=_X),
        )
        cfg = make_control_flow_graph(_function(body))
        copy_propagation(cfg)
        instrs = [i for b in cfg.blocks for i in b.instructions]
        assert tac.Return(val=_Y) in instrs

    def test_copy_not_propagated_when_branch_redefines_dst(self) -> None:
        """When one branch redefines the copy destination, the copy does not reach the join."""
        # One branch: x = y.  Other: x = 0.  At join, x is NOT known to equal y.
        body = (
            tac.JumpIfZero(condition=_Z, target="else"),
            tac.Copy(src=_Y, dst=_X),
            tac.Jump(target="end"),
            tac.Label("else"),
            tac.Copy(src=_ci(0), dst=_X),
            tac.Label("end"),
            tac.Return(val=_X),
        )
        cfg = make_control_flow_graph(_function(body))
        copy_propagation(cfg)
        instrs = [i for b in cfg.blocks for i in b.instructions]
        assert tac.Return(val=_Y) not in instrs
        assert tac.Return(val=_ci(0)) not in instrs
        assert tac.Return(val=_X) in instrs

    def test_copy_involving_static_var_killed_by_funcall(self) -> None:
        """A copy whose operand is a static variable is killed by a ``FunCall``."""
        sym: SymbolTable = {
            **_static_sym("g"),
            **_local_sym("tmp"),
        }
        body = (
            tac.Copy(src=_var("g"), dst=_var("tmp")),
            tac.FunCall(fun_val=_var("foo"), args=(), dst=None),
            tac.Return(val=_var("tmp")),
        )
        cfg = make_control_flow_graph(_function(body))
        copy_propagation(cfg, sym)
        instrs = [i for b in cfg.blocks for i in b.instructions]
        # The FunCall may have modified g, so tmp must not be replaced with g.
        assert tac.Return(val=_var("g")) not in instrs
        assert tac.Return(val=_var("tmp")) in instrs
