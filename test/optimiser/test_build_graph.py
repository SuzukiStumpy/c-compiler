"""Unit tests for the interference-graph construction passes in ``asm_main``.

Covers AsmAdapter CFG dispatch, pseudo-register classification, hardware
register pre-population, and interference edge generation for both the GP
and FP passes.
"""

import compiler.syntax.asm as asm
import compiler.syntax.ast as ast
from compiler.optimiser.asm_main import (
    _EXIT_LIVE,
    _classify_pseudos,
    _make_asm_cfg,
    _meet,
    _operands_to_nodes,
    _transfer,
    add_spill_costs,
    build_fp_graph,
    build_gp_graph,
    color_graph,
    create_register_map,
    find_used_and_updated,
    replace_pseudoregs,
)
from compiler.optimiser.cfg import ASM_ADAPTER, EXIT, BlockId, BoundaryNode, ControlFlowGraph
from compiler.optimiser.interference_graph import InterferenceGraph
from compiler.syntax.asm.operands import (
    ALLOCATABLE_FP,
    ALLOCATABLE_GP,
    FPRegisters,
    GPRegisters,
    Width,
)
from compiler.syntax.ast.types import DoubleType, FunType, IntType, LongType
from compiler.types.symbols import (
    FunAttr,
    LocalAttr,
    NoInitializer,
    StaticAttr,
    SymbolTable,
    SymbolTableEntry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AX = asm.Register(GPRegisters.AX, Width.QWORD)
_XMM0 = asm.Register(FPRegisters.XMM0, Width.QWORD)
_XMM1 = asm.Register(FPRegisters.XMM1, Width.QWORD)


def _pseudo(name: str) -> asm.Pseudo:
    return asm.Pseudo(name)


def _mov_gp(src: asm.Operand, dst: asm.Operand) -> asm.Mov:
    return asm.Mov(src=src, dst=dst, assembly_type=asm.Longword())


def _mov_fp(src: asm.Operand, dst: asm.Operand) -> asm.Mov:
    return asm.Mov(src=src, dst=dst, assembly_type=asm.Double())


def _gp_sym(*names: str) -> SymbolTable:
    """Minimal SymbolTable: all names are IntType local variables (classified GP)."""
    return {n: SymbolTableEntry(t=IntType(), attrs=LocalAttr()) for n in names}


def _fp_sym(*names: str) -> SymbolTable:
    """Minimal SymbolTable: all names are DoubleType local variables (classified FP)."""
    return {n: SymbolTableEntry(t=DoubleType(), attrs=LocalAttr()) for n in names}


# ---------------------------------------------------------------------------
# AsmAdapter - CFG construction dispatch
# ---------------------------------------------------------------------------


class TestAsmAdapterCfg:
    """Verify AsmAdapter drives make_cfg correctly for asm instructions."""

    def test_label_starts_new_block(self) -> None:
        """A Label instruction causes make_cfg to start a new basic block."""
        instrs = [asm.Label("then"), asm.Ret()]
        cfg = _make_asm_cfg(instrs)
        assert len(cfg.blocks) == 1
        assert cfg.blocks[0].label == "then"

    def test_ret_successor_is_exit(self) -> None:
        """A block ending with Ret has EXIT as its sole successor."""
        cfg = _make_asm_cfg([asm.Ret()])
        assert cfg.blocks[0].successors == [EXIT]

    def test_jmp_successor_is_target_block(self) -> None:
        """Jmp produces a successor pointing at the target label block."""
        instrs = [asm.Label("a"), asm.Jmp("b"), asm.Label("b"), asm.Ret()]
        cfg = _make_asm_cfg(instrs)
        assert cfg.blocks[0].successors == [BlockId(1)]

    def test_jmpcc_produces_two_successors(self) -> None:
        """JmpCC produces two successors: the jump target then the fall-through."""
        instrs = [
            asm.Label("entry"),
            asm.JmpCC(cond_code=asm.CondCodes.E, label="done"),
            asm.Label("body"),
            asm.Ret(),
            asm.Label("done"),
            asm.Ret(),
        ]
        cfg = _make_asm_cfg(instrs)
        assert cfg.blocks[0].successors == [BlockId(2), BlockId(1)]


# ---------------------------------------------------------------------------
# AsmAdapter - defs / uses
# ---------------------------------------------------------------------------


class TestAsmAdapterDefs:
    """Verify AsmAdapter.defs() returns the correct defined nodes."""

    def test_mov_defines_pseudo_dst(self) -> None:
        assert ASM_ADAPTER.defs(_mov_gp(_pseudo("a"), _pseudo("b"))) == {"b"}

    def test_mov_defines_register_dst(self) -> None:
        assert ASM_ADAPTER.defs(_mov_gp(_pseudo("a"), _AX)) == {GPRegisters.AX}

    def test_mov_memory_dst_defines_nothing(self) -> None:
        bp = asm.Register(GPRegisters.BP, Width.QWORD)
        assert ASM_ADAPTER.defs(_mov_gp(_pseudo("a"), asm.Memory(reg=bp, offset=-4))) == set()

    def test_call_defines_caller_saved_regs(self) -> None:
        d = ASM_ADAPTER.defs(asm.Call(name="foo"))
        assert GPRegisters.AX in d
        assert FPRegisters.XMM0 in d
        assert GPRegisters.SP not in d
        assert GPRegisters.BP not in d
        assert GPRegisters.R10 not in d
        assert GPRegisters.R11 not in d

    def test_idiv_defines_ax_and_dx(self) -> None:
        assert ASM_ADAPTER.defs(asm.Idiv(operand=_pseudo("x"))) == {GPRegisters.AX, GPRegisters.DX}

    def test_cdq_defines_dx(self) -> None:
        assert ASM_ADAPTER.defs(asm.Cdq()) == {GPRegisters.DX}

    def test_label_defines_nothing(self) -> None:
        assert ASM_ADAPTER.defs(asm.Label("lbl")) == set()

    def test_cvttss2si_defines_gp_dst(self) -> None:
        """``Cvttss2si`` writes to its destination GP pseudo/register."""
        instr = asm.Cvttss2si(src=_pseudo("f"), dst=_pseudo("n"), assembly_type=asm.Longword())
        assert ASM_ADAPTER.defs(instr) == {"n"}

    def test_cvtsi2ss_defines_fp_dst(self) -> None:
        """``Cvtsi2ss`` writes to its destination FP pseudo/register."""
        instr = asm.Cvtsi2ss(src=_pseudo("n"), dst=_pseudo("f"), assembly_type=asm.Longword())
        assert ASM_ADAPTER.defs(instr) == {"f"}

    def test_cvtss2sd_defines_fp_dst(self) -> None:
        """``Cvtss2sd`` writes to its destination FP pseudo/register."""
        instr = asm.Cvtss2sd(src=_pseudo("f"), dst=_pseudo("d"))
        assert ASM_ADAPTER.defs(instr) == {"d"}

    def test_cvtsd2ss_defines_fp_dst(self) -> None:
        """``Cvtsd2ss`` writes to its destination FP pseudo/register."""
        instr = asm.Cvtsd2ss(src=_pseudo("d"), dst=_pseudo("f"))
        assert ASM_ADAPTER.defs(instr) == {"f"}


class TestAsmAdapterUses:
    """Verify AsmAdapter.uses() returns the correct used nodes."""

    def test_mov_uses_pseudo_src(self) -> None:
        assert ASM_ADAPTER.uses(_mov_gp(_pseudo("a"), _pseudo("b"))) == {"a"}

    def test_mov_uses_register_src(self) -> None:
        assert ASM_ADAPTER.uses(_mov_gp(_AX, _pseudo("b"))) == {GPRegisters.AX}

    def test_mov_memory_src_non_allocatable_base_not_used(self) -> None:
        bp = asm.Register(GPRegisters.BP, Width.QWORD)
        assert ASM_ADAPTER.uses(_mov_gp(asm.Memory(reg=bp, offset=-4), _pseudo("b"))) == set()

    def test_cdq_uses_ax(self) -> None:
        assert GPRegisters.AX in ASM_ADAPTER.uses(asm.Cdq())

    def test_ret_uses_nothing(self) -> None:
        # Ret has no operands; exit-liveness is handled by _meet's exit_live injection.
        assert ASM_ADAPTER.uses(asm.Ret()) == set()

    def test_shift_uses_cx(self) -> None:
        assert GPRegisters.CX in ASM_ADAPTER.uses(asm.Shl(operand=_pseudo("x")))

    def test_label_uses_nothing(self) -> None:
        assert ASM_ADAPTER.uses(asm.Label("lbl")) == set()

    def test_cvttss2si_uses_fp_src(self) -> None:
        """``Cvttss2si`` reads from its FP source pseudo/register."""
        instr = asm.Cvttss2si(src=_pseudo("f"), dst=_pseudo("n"), assembly_type=asm.Longword())
        assert ASM_ADAPTER.uses(instr) == {"f"}

    def test_cvtsi2ss_uses_gp_src(self) -> None:
        """``Cvtsi2ss`` reads from its GP source pseudo/register."""
        instr = asm.Cvtsi2ss(src=_pseudo("n"), dst=_pseudo("f"), assembly_type=asm.Longword())
        assert ASM_ADAPTER.uses(instr) == {"n"}

    def test_cvtss2sd_uses_fp_src(self) -> None:
        """``Cvtss2sd`` reads from its FP source pseudo/register."""
        instr = asm.Cvtss2sd(src=_pseudo("f"), dst=_pseudo("d"))
        assert ASM_ADAPTER.uses(instr) == {"f"}

    def test_cvtsd2ss_uses_fp_src(self) -> None:
        """``Cvtsd2ss`` reads from its FP source pseudo/register."""
        instr = asm.Cvtsd2ss(src=_pseudo("d"), dst=_pseudo("f"))
        assert ASM_ADAPTER.uses(instr) == {"d"}


# ---------------------------------------------------------------------------
# _classify_pseudos
# ---------------------------------------------------------------------------


class TestClassifyPseudos:
    """Verify _classify_pseudos correctly separates GP and FP pseudo-registers."""

    def test_int_type_classified_as_gp(self) -> None:
        sym = _gp_sym("x", "y")
        gp, fp = _classify_pseudos([_mov_gp(_pseudo("x"), _pseudo("y"))], sym)
        assert "x" in gp and "y" in gp
        assert "x" not in fp and "y" not in fp

    def test_double_type_classified_as_fp(self) -> None:
        sym = _fp_sym("a", "b")
        gp, fp = _classify_pseudos([_mov_fp(_pseudo("a"), _pseudo("b"))], sym)
        assert "a" in fp and "b" in fp
        assert "a" not in gp and "b" not in gp

    def test_long_type_classified_as_gp(self) -> None:
        sym = {n: SymbolTableEntry(t=LongType(), attrs=LocalAttr()) for n in ("p",)}
        gp, fp = _classify_pseudos([_mov_gp(_pseudo("p"), _AX)], sym)
        assert "p" in gp and "p" not in fp

    def test_cvttsd2si_src_fp_dst_gp(self) -> None:
        sym = {**_fp_sym("s"), **_gp_sym("d")}
        instr = asm.Cvttsd2si(src=_pseudo("s"), dst=_pseudo("d"))
        gp, fp = _classify_pseudos([instr], sym)
        assert "s" in fp and "d" in gp

    def test_cvtsi2sd_src_gp_dst_fp(self) -> None:
        sym = {**_gp_sym("s"), **_fp_sym("d")}
        instr = asm.Cvtsi2sd(src=_pseudo("s"), dst=_pseudo("d"))
        gp, fp = _classify_pseudos([instr], sym)
        assert "s" in gp and "d" in fp

    def test_no_pseudos_returns_empty_sets(self) -> None:
        gp, fp = _classify_pseudos([asm.Ret()], {})
        assert gp == set() and fp == set()

    def test_no_symbol_table_returns_empty_sets(self) -> None:
        """Without a symbol table no pseudos can be classified."""
        gp, fp = _classify_pseudos([_mov_gp(_pseudo("x"), _pseudo("y"))])
        assert gp == set() and fp == set()

    def test_address_taken_pseudo_excluded(self) -> None:
        """A pseudo that appears as the Lea source (address taken) is excluded from both graphs."""
        sym = {**_gp_sym("x", "y")}
        instrs = [
            asm.Lea(src=_pseudo("x"), dst=_AX),
            _mov_gp(_pseudo("y"), _AX),
        ]
        gp, fp = _classify_pseudos(instrs, sym)
        assert "x" not in gp and "x" not in fp
        assert "y" in gp

    def test_non_scalar_type_excluded_from_both(self) -> None:
        """Pseudos whose type has no register-class mapping are not added to either graph."""
        sym = {"f": SymbolTableEntry(t=FunType(params=(), ret=IntType()), attrs=LocalAttr())}
        gp, fp = _classify_pseudos([_mov_gp(_pseudo("f"), _AX)], sym)
        assert "f" not in gp and "f" not in fp


# ---------------------------------------------------------------------------
# build_gp_graph - node population and edges
# ---------------------------------------------------------------------------


class TestBuildGpGraph:
    """Verify build_gp_graph populates nodes and edges correctly."""

    def test_allocatable_gp_regs_pre_populated(self) -> None:
        graph = build_gp_graph([asm.Ret()])
        for reg in ALLOCATABLE_GP:
            assert reg in graph.nodes

    def test_fp_regs_not_in_gp_graph(self) -> None:
        graph = build_gp_graph([asm.Ret()])
        for reg in ALLOCATABLE_FP:
            assert reg not in graph.nodes

    def test_excluded_gp_regs_absent(self) -> None:
        graph = build_gp_graph([asm.Ret()])
        assert GPRegisters.SP not in graph.nodes
        assert GPRegisters.BP not in graph.nodes
        assert GPRegisters.R10 not in graph.nodes
        assert GPRegisters.R11 not in graph.nodes

    def test_gp_pseudo_nodes_added(self) -> None:
        graph = build_gp_graph([_mov_gp(_pseudo("x"), _pseudo("y")), asm.Ret()], _gp_sym("x", "y"))
        assert "x" in graph.nodes
        assert "y" in graph.nodes

    def test_fp_pseudo_not_in_gp_graph(self) -> None:
        graph = build_gp_graph([_mov_fp(_pseudo("f"), _XMM0), asm.Ret()], _fp_sym("f"))
        assert "f" not in graph.nodes

    def test_simultaneously_live_gp_pseudos_interfere(self) -> None:
        # x defined, then y defined while x still live (x used later).
        instrs = [
            _mov_gp(asm.Imm(1), _pseudo("x")),
            _mov_gp(asm.Imm(2), _pseudo("y")),
            _mov_gp(_pseudo("x"), _AX),
            asm.Ret(),
        ]
        graph = build_gp_graph(instrs, _gp_sym("x", "y"))
        assert frozenset({"x", "y"}) in graph.edges

    def test_non_interfering_gp_pseudos_have_no_edge(self) -> None:
        # x used (dead) before y is defined.
        instrs = [
            _mov_gp(asm.Imm(1), _pseudo("x")),
            _mov_gp(_pseudo("x"), _AX),
            _mov_gp(asm.Imm(2), _pseudo("y")),
            _mov_gp(_pseudo("y"), _AX),
            asm.Ret(),
        ]
        graph = build_gp_graph(instrs, _gp_sym("x", "y"))
        assert frozenset({"x", "y"}) not in graph.edges

    def test_mov_src_does_not_interfere_with_dst(self) -> None:
        """Mov source gets no interference edge with the destination (coalescing hint)."""
        # x is live after the Mov (used later), so without the optimisation x and y
        # would interfere.  The Mov optimisation must suppress that edge.
        instrs = [
            _mov_gp(asm.Imm(1), _pseudo("x")),
            _mov_gp(asm.Imm(2), _pseudo("z")),
            _mov_gp(_pseudo("x"), _pseudo("y")),  # Mov: src=x, dst=y
            _mov_gp(_pseudo("x"), _AX),  # x still live after the Mov
            _mov_gp(_pseudo("z"), _AX),
            _mov_gp(_pseudo("y"), _AX),
            asm.Ret(),
        ]
        graph = build_gp_graph(instrs, _gp_sym("x", "y", "z"))
        assert frozenset({"x", "y"}) not in graph.edges

    def test_non_src_live_value_still_interferes_with_mov_dst(self) -> None:
        """Values live after a Mov that are NOT the source DO interfere with the destination."""
        instrs = [
            _mov_gp(asm.Imm(1), _pseudo("x")),
            _mov_gp(asm.Imm(2), _pseudo("z")),
            _mov_gp(_pseudo("x"), _pseudo("y")),  # Mov: src=x, dst=y
            _mov_gp(_pseudo("x"), _AX),
            _mov_gp(_pseudo("z"), _AX),  # z live after the Mov
            _mov_gp(_pseudo("y"), _AX),
            asm.Ret(),
        ]
        graph = build_gp_graph(instrs, _gp_sym("x", "y", "z"))
        assert frozenset({"z", "y"}) in graph.edges


# ---------------------------------------------------------------------------
# build_fp_graph - node population and edges
# ---------------------------------------------------------------------------


class TestBuildFpGraph:
    """Verify build_fp_graph populates nodes and edges correctly."""

    def test_allocatable_fp_regs_pre_populated(self) -> None:
        graph = build_fp_graph([asm.Ret()])
        for reg in ALLOCATABLE_FP:
            assert reg in graph.nodes

    def test_gp_regs_not_in_fp_graph(self) -> None:
        graph = build_fp_graph([asm.Ret()])
        for reg in ALLOCATABLE_GP:
            assert reg not in graph.nodes

    def test_fp_pseudo_nodes_added(self) -> None:
        graph = build_fp_graph([_mov_fp(_pseudo("f"), _XMM0), asm.Ret()], _fp_sym("f"))
        assert "f" in graph.nodes

    def test_gp_pseudo_not_in_fp_graph(self) -> None:
        graph = build_fp_graph([_mov_gp(_pseudo("g"), _AX), asm.Ret()], _gp_sym("g"))
        assert "g" not in graph.nodes

    def test_simultaneously_live_fp_pseudos_interfere(self) -> None:
        instrs = [
            _mov_fp(asm.Imm(0), _pseudo("a")),
            _mov_fp(asm.Imm(0), _pseudo("b")),
            _mov_fp(_pseudo("a"), _XMM0),
            asm.Ret(),
        ]
        graph = build_fp_graph(instrs, _fp_sym("a", "b"))
        assert frozenset({"a", "b"}) in graph.edges

    def test_non_interfering_fp_pseudos_have_no_edge(self) -> None:
        instrs = [
            _mov_fp(asm.Imm(0), _pseudo("a")),
            _mov_fp(_pseudo("a"), _XMM0),
            _mov_fp(asm.Imm(0), _pseudo("b")),
            _mov_fp(_pseudo("b"), _XMM0),
            asm.Ret(),
        ]
        graph = build_fp_graph(instrs, _fp_sym("a", "b"))
        assert frozenset({"a", "b"}) not in graph.edges


# ---------------------------------------------------------------------------
# find_used_and_updated
# ---------------------------------------------------------------------------


def _fun_sym(params: list[type]) -> SymbolTable:
    """Build a SymbolTable with a single 'fn' entry whose type is FunType(params)."""
    t = FunType(params=tuple(p() for p in params), ret=IntType())
    return {"fn": SymbolTableEntry(t=t, attrs=FunAttr(defined=True, globl=False))}


class TestFindUsedAndUpdated:
    """Verify find_used_and_updated returns the correct used/updated operand lists."""

    def _operand_regs(self, ops: list) -> set[GPRegisters | FPRegisters]:
        """Extract hardware register enums from a list of operands."""
        result: set[GPRegisters | FPRegisters] = set()
        for op in ops:
            if isinstance(op, asm.Register):
                result.add(op.reg)
        return result

    def _operand_pseudos(self, ops: list) -> set[str]:
        """Extract pseudo names from a list of operands."""
        return {op.name for op in ops if isinstance(op, asm.Pseudo)}

    # --- Mov ---

    def test_mov_used_is_src(self) -> None:
        used, _ = find_used_and_updated(_mov_gp(_pseudo("a"), _pseudo("b")))
        assert self._operand_pseudos(used) == {"a"}

    def test_mov_updated_is_dst(self) -> None:
        _, updated = find_used_and_updated(_mov_gp(_pseudo("a"), _pseudo("b")))
        assert self._operand_pseudos(updated) == {"b"}

    # --- Binary ---

    def test_binary_used_is_both_sources(self) -> None:
        instr = asm.Binary(
            operator=asm.Add(),
            src1=_pseudo("x"),
            src2=_pseudo("y"),
            dst=asm.Register(GPRegisters.AX, Width.QWORD),
        )
        used, _ = find_used_and_updated(instr)
        assert self._operand_pseudos(used) == {"x", "y"}

    def test_binary_updated_is_dst_only(self) -> None:
        instr = asm.Binary(
            operator=asm.Add(),
            src1=_pseudo("x"),
            src2=_pseudo("y"),
            dst=asm.Register(GPRegisters.AX, Width.QWORD),
        )
        _, updated = find_used_and_updated(instr)
        assert self._operand_regs(updated) == {GPRegisters.AX}

    # --- Unary ---

    def test_unary_operand_in_both_used_and_updated(self) -> None:
        instr = asm.Unary(operator=asm.Neg(), operand=_pseudo("x"))
        used, updated = find_used_and_updated(instr)
        assert self._operand_pseudos(used) == {"x"}
        assert self._operand_pseudos(updated) == {"x"}

    # --- Cmp ---

    def test_cmp_used_both_updated_empty(self) -> None:
        instr = asm.Cmp(op1=_pseudo("a"), op2=_pseudo("b"))
        used, updated = find_used_and_updated(instr)
        assert self._operand_pseudos(used) == {"a", "b"}
        assert updated == []

    # --- SetCC ---

    def test_setcc_used_empty_updated_dst(self) -> None:
        instr = asm.SetCC(cond_code=asm.CondCodes.E, operand=_pseudo("x"))
        used, updated = find_used_and_updated(instr)
        assert used == []
        assert self._operand_pseudos(updated) == {"x"}

    # --- Push ---

    def test_push_used_operand_updated_empty(self) -> None:
        used, updated = find_used_and_updated(asm.Push(operand=_pseudo("x")))
        assert self._operand_pseudos(used) == {"x"}
        assert updated == []

    # --- Pop ---

    def test_pop_used_empty_updated_target(self) -> None:
        used, updated = find_used_and_updated(asm.Pop(target=_AX))
        assert used == []
        assert self._operand_regs(updated) == {GPRegisters.AX}

    # --- Idiv / Div ---

    def test_idiv_uses_operand_and_ax_dx(self) -> None:
        used, updated = find_used_and_updated(asm.Idiv(operand=_pseudo("d")))
        assert self._operand_pseudos(used) == {"d"}
        assert {GPRegisters.AX, GPRegisters.DX} <= self._operand_regs(used)

    def test_idiv_updates_ax_and_dx(self) -> None:
        _, updated = find_used_and_updated(asm.Idiv(operand=_pseudo("d")))
        assert self._operand_regs(updated) == {GPRegisters.AX, GPRegisters.DX}

    def test_div_same_as_idiv(self) -> None:
        used_i, upd_i = find_used_and_updated(asm.Idiv(operand=_pseudo("d")))
        used_u, upd_u = find_used_and_updated(asm.Div(operand=_pseudo("d")))
        assert self._operand_regs(used_i) == self._operand_regs(used_u)
        assert self._operand_regs(upd_i) == self._operand_regs(upd_u)

    # --- Cdq ---

    def test_cdq_uses_ax_updates_dx(self) -> None:
        used, updated = find_used_and_updated(asm.Cdq())
        assert self._operand_regs(used) == {GPRegisters.AX}
        assert self._operand_regs(updated) == {GPRegisters.DX}

    # --- Shifts ---

    def test_shl_uses_operand_and_cx(self) -> None:
        used, updated = find_used_and_updated(asm.Shl(operand=_pseudo("x")))
        assert self._operand_pseudos(used) == {"x"}
        assert GPRegisters.CX in self._operand_regs(used)

    def test_shl_updates_operand(self) -> None:
        _, updated = find_used_and_updated(asm.Shl(operand=_pseudo("x")))
        assert self._operand_pseudos(updated) == {"x"}

    def test_sar_same_shape_as_shl(self) -> None:
        used_sar, upd_sar = find_used_and_updated(asm.Sar(operand=_pseudo("x")))
        used_shr, upd_shr = find_used_and_updated(asm.Shr(operand=_pseudo("x")))
        assert self._operand_pseudos(used_sar) == self._operand_pseudos(used_shr)
        assert self._operand_pseudos(upd_sar) == self._operand_pseudos(upd_shr)

    # --- Cvt instructions ---

    def test_cvttsd2si_used_src_updated_dst(self) -> None:
        used, updated = find_used_and_updated(asm.Cvttsd2si(src=_pseudo("s"), dst=_pseudo("d")))
        assert self._operand_pseudos(used) == {"s"}
        assert self._operand_pseudos(updated) == {"d"}

    def test_cvtsi2sd_used_src_updated_dst(self) -> None:
        used, updated = find_used_and_updated(asm.Cvtsi2sd(src=_pseudo("s"), dst=_pseudo("d")))
        assert self._operand_pseudos(used) == {"s"}
        assert self._operand_pseudos(updated) == {"d"}

    def test_cvttss2si_used_src_updated_dst(self) -> None:
        instr = asm.Cvttss2si(src=_pseudo("f"), dst=_pseudo("n"), assembly_type=asm.Longword())
        used, updated = find_used_and_updated(instr)
        assert self._operand_pseudos(used) == {"f"}
        assert self._operand_pseudos(updated) == {"n"}

    def test_cvtsi2ss_used_src_updated_dst(self) -> None:
        instr = asm.Cvtsi2ss(src=_pseudo("n"), dst=_pseudo("f"), assembly_type=asm.Longword())
        used, updated = find_used_and_updated(instr)
        assert self._operand_pseudos(used) == {"n"}
        assert self._operand_pseudos(updated) == {"f"}

    def test_cvtss2sd_used_src_updated_dst(self) -> None:
        used, updated = find_used_and_updated(asm.Cvtss2sd(src=_pseudo("f"), dst=_pseudo("d")))
        assert self._operand_pseudos(used) == {"f"}
        assert self._operand_pseudos(updated) == {"d"}

    def test_cvtsd2ss_used_src_updated_dst(self) -> None:
        used, updated = find_used_and_updated(asm.Cvtsd2ss(src=_pseudo("d"), dst=_pseudo("f")))
        assert self._operand_pseudos(used) == {"d"}
        assert self._operand_pseudos(updated) == {"f"}

    # --- Call ---

    def test_call_no_table_uses_all_gp_arg_regs(self) -> None:
        """Without a symbol table, all 6 GP arg regs are conservatively marked used."""
        used, _ = find_used_and_updated(asm.Call(name="fn"))
        regs = self._operand_regs(used)
        for r in (GPRegisters.DI, GPRegisters.SI, GPRegisters.DX, GPRegisters.CX, GPRegisters.R8, GPRegisters.R9):
            assert r in regs

    def test_call_updates_gp_caller_saved(self) -> None:
        _, updated = find_used_and_updated(asm.Call(name="fn"))
        regs = self._operand_regs(updated)
        assert GPRegisters.AX in regs
        assert GPRegisters.DI in regs

    def test_call_int_param_uses_di(self) -> None:
        """One int param -> only DI is used (not SI, DX, etc.)."""
        sym = _fun_sym([IntType])
        used, _ = find_used_and_updated(asm.Call(name="fn"), sym)
        assert self._operand_regs(used) == {GPRegisters.DI}

    def test_call_double_param_uses_xmm0(self) -> None:
        """One double param -> only XMM0 is used."""
        sym = _fun_sym([DoubleType])
        used, _ = find_used_and_updated(asm.Call(name="fn"), sym)
        assert self._operand_regs(used) == {FPRegisters.XMM0}

    def test_call_mixed_params_correct_regs(self) -> None:
        """int, double, int -> DI, XMM0, SI."""
        sym = _fun_sym([IntType, DoubleType, IntType])
        used, _ = find_used_and_updated(asm.Call(name="fn"), sym)
        assert self._operand_regs(used) == {GPRegisters.DI, FPRegisters.XMM0, GPRegisters.SI}

    def test_call_no_params_used_empty(self) -> None:
        """Zero-param function -> no argument registers used."""
        sym = _fun_sym([])
        used, _ = find_used_and_updated(asm.Call(name="fn"), sym)
        assert used == []

    # --- Fallthrough / no-op instructions ---

    def test_ret_used_and_updated_empty(self) -> None:
        used, updated = find_used_and_updated(asm.Ret())
        assert used == [] and updated == []

    def test_label_used_and_updated_empty(self) -> None:
        used, updated = find_used_and_updated(asm.Label("lbl"))
        assert used == [] and updated == []

    def test_jmp_used_and_updated_empty(self) -> None:
        used, updated = find_used_and_updated(asm.Jmp("lbl"))
        assert used == [] and updated == []


# ---------------------------------------------------------------------------
# _operands_to_nodes
# ---------------------------------------------------------------------------


class TestOperandsToNodes:
    """Verify _operands_to_nodes correctly extracts GraphNode values."""

    def test_pseudo_yields_name(self) -> None:
        assert _operands_to_nodes([asm.Pseudo("x")]) == {"x"}

    def test_allocatable_register_yields_enum(self) -> None:
        assert _operands_to_nodes([_AX]) == {GPRegisters.AX}

    def test_non_allocatable_register_yields_nothing(self) -> None:
        bp = asm.Register(GPRegisters.BP, Width.QWORD)
        assert _operands_to_nodes([bp]) == set()

    def test_memory_with_allocatable_base_yields_base(self) -> None:
        # In practice memory bases are BP (non-alloc), but test the general case.
        base = asm.Register(GPRegisters.AX, Width.QWORD)
        assert _operands_to_nodes([asm.Memory(reg=base, offset=-4)]) == {GPRegisters.AX}

    def test_memory_with_non_allocatable_base_yields_nothing(self) -> None:
        bp = asm.Register(GPRegisters.BP, Width.QWORD)
        assert _operands_to_nodes([asm.Memory(reg=bp, offset=-4)]) == set()

    def test_indexed_yields_allocatable_base_and_index(self) -> None:
        base = asm.Register(GPRegisters.R8, Width.QWORD)
        idx = asm.Register(GPRegisters.R9, Width.QWORD)
        assert _operands_to_nodes([asm.Indexed(base=base, index=idx, scale=4)]) == {
            GPRegisters.R8,
            GPRegisters.R9,
        }

    def test_imm_yields_nothing(self) -> None:
        assert _operands_to_nodes([asm.Imm(42)]) == set()

    def test_multiple_operands_combined(self) -> None:
        assert _operands_to_nodes([_pseudo("a"), _AX, asm.Imm(0)]) == {"a", GPRegisters.AX}


# ---------------------------------------------------------------------------
# _transfer
# ---------------------------------------------------------------------------


class TestTransfer:
    """Verify _transfer propagates liveness correctly and returns live_after lists."""

    def _gp_tracked(self) -> set:
        return set(ALLOCATABLE_GP)

    def test_empty_block_live_in_equals_end_live(self) -> None:
        """A block with no instructions passes end_live straight through as live_in."""
        from compiler.optimiser.cfg import BasicBlock as _BB

        block = _BB(label=None, instructions=[])
        block.successors = []
        tracked = self._gp_tracked()
        live_in, live_after = _transfer(block, {GPRegisters.AX}, tracked)
        assert live_in == {GPRegisters.AX}
        assert live_after == []

    def test_live_after_has_one_entry_per_instruction(self) -> None:
        instrs = [_mov_gp(asm.Imm(1), _pseudo("x")), asm.Ret()]
        cfg = _make_asm_cfg(instrs)
        tracked = self._gp_tracked() | {"x"}
        live_in, live_after = _transfer(cfg.blocks[0], set(), tracked)
        assert len(live_after) == len(cfg.blocks[0].instructions)

    def test_use_propagates_backward(self) -> None:
        """A pseudo used near the end must appear live in the live_after of earlier instrs."""
        instrs = [_mov_gp(asm.Imm(1), _pseudo("x")), _mov_gp(_pseudo("x"), _AX), asm.Ret()]
        cfg = _make_asm_cfg(instrs)
        tracked: set = self._gp_tracked() | {"x"}
        _, live_after = _transfer(cfg.blocks[0], set(), tracked)
        # live_after[0] = live set after instr[0] (mov imm->x); x is defined here and used later -> live
        # live_after[1] = live set after instr[1] (mov x->AX); x consumed here -> no longer live
        assert "x" in live_after[0]
        assert "x" not in live_after[1]

    def test_def_kills_live(self) -> None:
        """A definition removes the pseudo from subsequent (earlier) live sets."""
        instrs = [_mov_gp(asm.Imm(1), _pseudo("x")), asm.Ret()]
        cfg = _make_asm_cfg(instrs)
        tracked: set = self._gp_tracked() | {"x"}
        live_in, live_after = _transfer(cfg.blocks[0], set(), tracked)
        # After the Ret, x is not live (end_live={}); after the mov it's also not live
        # (x is defined here but not used after -> killed immediately)
        assert "x" not in live_in

    def test_tracked_nodes_filters_out_fp_regs(self) -> None:
        """FP registers should not appear in live_in when using GP tracked_nodes."""
        instrs = [_mov_fp(_pseudo("f"), _XMM0), asm.Ret()]
        cfg = _make_asm_cfg(instrs)
        tracked = self._gp_tracked()  # GP only
        live_in, _ = _transfer(cfg.blocks[0], set(), tracked)
        assert FPRegisters.XMM0 not in live_in
        assert "f" not in live_in

    def test_live_in_correct_for_multiple_instrs(self) -> None:
        """live_in reflects only values not yet defined at the block entry."""
        instrs = [
            _mov_gp(asm.Imm(1), _pseudo("x")),  # x defined
            _mov_gp(_pseudo("x"), _AX),  # x used, AX defined
            asm.Ret(),
        ]
        cfg = _make_asm_cfg(instrs)
        tracked: set = self._gp_tracked() | {"x"}
        live_in, _ = _transfer(cfg.blocks[0], set(), tracked)
        # x is defined before it's used; AX is "defined" by the mov but not used after
        assert "x" not in live_in


# ---------------------------------------------------------------------------
# Static storage duration exclusion
# ---------------------------------------------------------------------------

_INT_TYPE = ast.IntType()


def _make_symbol_table(static_names: list[str], local_names: list[str]) -> SymbolTable:
    """Build a minimal SymbolTable with the given static and local entries."""
    table: SymbolTable = {}
    for name in static_names:
        table[name] = SymbolTableEntry(t=_INT_TYPE, attrs=StaticAttr(init=NoInitializer(), globl=False))
    for name in local_names:
        table[name] = SymbolTableEntry(t=_INT_TYPE, attrs=LocalAttr())
    return table


class TestStaticExclusion:
    """Static-duration pseudos must not appear as nodes in the interference graph."""

    def test_static_gp_pseudo_excluded_from_gp_graph(self) -> None:
        """A GP pseudo with StaticAttr is not added to the GP graph."""
        sym = _make_symbol_table(static_names=["s"], local_names=["x"])
        instrs = [_mov_gp(_pseudo("s"), _AX), _mov_gp(_pseudo("x"), _AX), asm.Ret()]
        graph = build_gp_graph(instrs, sym)
        assert "s" not in graph.nodes
        assert "x" in graph.nodes

    def test_static_fp_pseudo_excluded_from_fp_graph(self) -> None:
        """A FP pseudo with StaticAttr is not added to the FP graph."""
        sym: SymbolTable = {
            "sf": SymbolTableEntry(t=DoubleType(), attrs=StaticAttr(init=NoInitializer(), globl=False)),
            "xf": SymbolTableEntry(t=DoubleType(), attrs=LocalAttr()),
        }
        instrs = [_mov_fp(_pseudo("sf"), _XMM0), _mov_fp(_pseudo("xf"), _XMM0), asm.Ret()]
        graph = build_fp_graph(instrs, sym)
        assert "sf" not in graph.nodes
        assert "xf" in graph.nodes

    def test_no_symbol_table_no_pseudos_classified(self) -> None:
        """When symbol_table is None no pseudos are classified or added to the graph."""
        instrs = [_mov_gp(_pseudo("x"), _AX), asm.Ret()]
        graph = build_gp_graph(instrs, symbol_table=None)
        assert "x" not in graph.nodes

    def test_classify_pseudos_skips_static(self) -> None:
        """_classify_pseudos excludes pseudos marked StaticAttr."""
        sym = _make_symbol_table(static_names=["s"], local_names=["x"])
        gp, _ = _classify_pseudos([_mov_gp(_pseudo("s"), _pseudo("x"))], sym)
        assert "s" not in gp
        assert "x" in gp

    def test_aliased_pseudo_excluded_from_gp_graph(self) -> None:
        """A pseudo whose address is taken (Lea source) is not added to the GP graph."""
        sym = _gp_sym("x", "y")
        instrs = [asm.Lea(src=_pseudo("x"), dst=_AX), _mov_gp(_pseudo("y"), _AX), asm.Ret()]
        graph = build_gp_graph(instrs, sym)
        assert "x" not in graph.nodes
        assert "y" in graph.nodes


# ---------------------------------------------------------------------------
# InterferenceGraph helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _meet - liveness propagation at successors
# ---------------------------------------------------------------------------


class TestMeet:
    """Verify _meet injects return-value regs at EXIT and errors on ENTRY."""

    def _cfg_with_exit(self) -> ControlFlowGraph[asm.Instruction]:
        """Build a single-block CFG whose sole successor is EXIT."""
        return _make_asm_cfg([asm.Ret()])

    def test_exit_contributes_ax_to_gp_pass(self) -> None:
        """EXIT successor injects AX into live_out for the GP pass."""
        cfg = self._cfg_with_exit()
        tracked: set = set(ALLOCATABLE_GP)
        live_in: list[set] = [set()]
        result = _meet(0, cfg, live_in, tracked, _EXIT_LIVE)
        assert GPRegisters.AX in result

    def test_exit_does_not_contribute_xmm0_to_gp_pass(self) -> None:
        """EXIT does not inject XMM0 when tracked_nodes contains only GP regs."""
        cfg = self._cfg_with_exit()
        tracked: set = set(ALLOCATABLE_GP)
        live_in: list[set] = [set()]
        result = _meet(0, cfg, live_in, tracked, _EXIT_LIVE)
        assert FPRegisters.XMM0 not in result

    def test_exit_contributes_xmm0_to_fp_pass(self) -> None:
        """EXIT successor injects XMM0 into live_out for the FP pass."""
        cfg = self._cfg_with_exit()
        tracked: set = set(ALLOCATABLE_FP)
        live_in: list[set] = [set()]
        result = _meet(0, cfg, live_in, tracked, _EXIT_LIVE)
        assert FPRegisters.XMM0 in result

    def test_exit_does_not_contribute_ax_to_fp_pass(self) -> None:
        """EXIT does not inject AX when tracked_nodes contains only FP regs."""
        cfg = self._cfg_with_exit()
        tracked: set = set(ALLOCATABLE_FP)
        live_in: list[set] = [set()]
        result = _meet(0, cfg, live_in, tracked, _EXIT_LIVE)
        assert GPRegisters.AX not in result

    def test_blockid_successor_merges_live_in(self) -> None:
        """A BlockId successor contributes that block's live_in to the result."""
        instrs = [asm.Label("a"), asm.Jmp("b"), asm.Label("b"), asm.Ret()]
        cfg = _make_asm_cfg(instrs)
        tracked: set = set(ALLOCATABLE_GP)
        live_in: list[set] = [set(), {GPRegisters.CX}]
        result = _meet(0, cfg, live_in, tracked, _EXIT_LIVE)
        assert GPRegisters.CX in result

    def test_entry_as_successor_raises(self) -> None:
        """ENTRY appearing as a block successor must raise ValueError."""
        import pytest

        cfg = _make_asm_cfg([asm.Ret()])
        # Manually inject ENTRY as a successor to simulate a malformed CFG.
        cfg.blocks[0].successors.append(BoundaryNode.ENTRY)
        tracked: set = set(ALLOCATABLE_GP)
        live_in: list[set] = [set()]
        with pytest.raises(ValueError, match="Malformed"):
            _meet(0, cfg, live_in, tracked, _EXIT_LIVE)


# ---------------------------------------------------------------------------
# InterferenceGraph helpers
# ---------------------------------------------------------------------------


class TestInterferenceGraph:
    """Unit tests for InterferenceGraph.add_edge and neighbours."""

    def test_add_edge_ignores_self_loop(self) -> None:
        g = InterferenceGraph(nodes={"a"})
        g.add_edge("a", "a")
        assert g.edges == set()

    def test_neighbours_reflects_edges(self) -> None:
        g = InterferenceGraph(nodes={"a", "b", "c"})
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        assert g.neighbours("a") == {"b", "c"}
        assert g.neighbours("b") == {"a"}


# ---------------------------------------------------------------------------
# add_spill_costs
# ---------------------------------------------------------------------------


class TestAddSpillCosts:
    """Verify add_spill_costs annotates nodes with correct spill costs."""

    def _graph(self, *nodes: str) -> InterferenceGraph:
        """Build a graph pre-populated with the given pseudo nodes."""
        g: InterferenceGraph = InterferenceGraph()
        for n in nodes:
            g.nodes.add(n)
        return g

    def test_hw_register_gets_infinite_cost(self) -> None:
        """Hardware-register nodes receive math.inf as their spill cost."""
        import math

        g = build_gp_graph([asm.Ret()])
        add_spill_costs(g, [asm.Ret()])
        for reg in ALLOCATABLE_GP:
            assert g.spill_costs[reg] == math.inf

    def test_pseudo_cost_equals_occurrence_count(self) -> None:
        """Pseudo spill cost equals the number of times it appears in instructions."""
        g = self._graph("x")
        instrs = [
            _mov_gp(asm.Imm(1), _pseudo("x")),
            _mov_gp(_pseudo("x"), _AX),
            _mov_gp(_pseudo("x"), _AX),
        ]
        add_spill_costs(g, instrs)
        assert g.spill_costs["x"] == 3

    def test_pseudo_not_in_graph_not_counted(self) -> None:
        """Occurrences of a pseudo absent from the graph do not appear in spill_costs."""
        g = self._graph("x")
        instrs = [_mov_gp(_pseudo("y"), _AX)]
        add_spill_costs(g, instrs)
        assert "y" not in g.spill_costs

    def test_pseudo_with_no_occurrences_gets_zero(self) -> None:
        """A pseudo in the graph that never appears in instructions gets cost 0."""
        g = self._graph("x")
        add_spill_costs(g, [asm.Ret()])
        assert g.spill_costs["x"] == 0

    def test_all_nodes_assigned_a_cost(self) -> None:
        """Every node in the graph has an entry in spill_costs after the call."""
        instrs = [_mov_gp(_pseudo("x"), _pseudo("y")), asm.Ret()]
        g = build_gp_graph(instrs, _gp_sym("x", "y"))
        add_spill_costs(g, instrs)
        for node in g.nodes:
            assert node in g.spill_costs


# ---------------------------------------------------------------------------
# color_graph
# ---------------------------------------------------------------------------


def _ready_graph(
    instrs: list[asm.Instruction],
    sym: SymbolTable,
    *,
    fp: bool = False,
) -> InterferenceGraph:
    """Build + annotate a graph so color_graph can be called on it."""
    g = build_fp_graph(instrs, sym) if fp else build_gp_graph(instrs, sym)
    add_spill_costs(g, instrs)
    return g


class TestColorGraph:
    """Verify color_graph assigns hardware registers and spills correctly."""

    def test_hw_registers_colored_with_themselves(self) -> None:
        """After coloring, every hardware register node maps to itself."""
        g = _ready_graph([asm.Ret()], {})
        color_graph(g)
        for node in g.nodes:
            if isinstance(node, (GPRegisters, FPRegisters)):
                assert g.colors.get(node) == node

    def test_colorable_pseudo_gets_a_register(self) -> None:
        """A pseudo with no interference edges should receive a caller-saved register."""
        instrs = [_mov_gp(asm.Imm(1), _pseudo("x")), _mov_gp(_pseudo("x"), _AX), asm.Ret()]
        g = _ready_graph(instrs, _gp_sym("x"))
        color_graph(g)
        assert "x" in g.colors
        assert isinstance(g.colors["x"], GPRegisters)

    def test_pseudo_not_colored_to_interfering_hw_reg(self) -> None:
        """A pseudo live at the same time as AX must not receive AX as its color."""
        # x and AX are simultaneously live: x defined, then x and AX both used in a call.
        instrs = [
            _mov_gp(asm.Imm(1), _pseudo("x")),
            asm.Call(name="fn"),  # clobbers AX; x is still live after
            _mov_gp(_pseudo("x"), _AX),
            asm.Ret(),
        ]
        g = _ready_graph(instrs, _gp_sym("x"))
        color_graph(g)
        assert g.colors.get("x") != GPRegisters.AX

    def test_two_non_interfering_pseudos_may_share_register(self) -> None:
        """Two pseudos whose live ranges do not overlap can receive the same register."""
        instrs = [
            _mov_gp(asm.Imm(1), _pseudo("x")),
            _mov_gp(_pseudo("x"), _AX),
            _mov_gp(asm.Imm(2), _pseudo("y")),
            _mov_gp(_pseudo("y"), _AX),
            asm.Ret(),
        ]
        g = _ready_graph(instrs, _gp_sym("x", "y"))
        color_graph(g)
        assert g.colors.get("x") == g.colors.get("y")

    def test_interfering_pseudos_get_different_registers(self) -> None:
        """Two simultaneously live pseudos must receive distinct registers."""
        instrs = [
            _mov_gp(asm.Imm(1), _pseudo("x")),
            _mov_gp(asm.Imm(2), _pseudo("y")),
            _mov_gp(_pseudo("x"), _AX),  # x used (and kept live with y)
            _mov_gp(_pseudo("y"), _AX),
            asm.Ret(),
        ]
        g = _ready_graph(instrs, _gp_sym("x", "y"))
        color_graph(g)
        assert "x" in g.colors and "y" in g.colors
        assert g.colors["x"] != g.colors["y"]

    def test_spilled_pseudo_has_no_color_and_stays_pruned(self) -> None:
        """A pseudo that cannot be colored has no entry in colors and stays in pruned."""
        # Build a graph where k pseudos all mutually interfere (k = len(ALLOCATABLE_GP)).
        # Add one more pseudo that interferes with all of them -> it must be spilled.
        k = len(ALLOCATABLE_GP)
        # Use the real build pipeline on a carefully constructed instruction sequence.
        # Simpler: construct the graph directly.
        g = build_gp_graph([], {})  # gets all hw-reg nodes; k allocatable colors
        # Add k+1 mutually-interfering pseudos
        names = [f"p{i}" for i in range(k + 1)]
        for name in names:
            g.nodes.add(name)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                g.add_edge(a, b)
        # Give each pseudo a non-zero spill cost so the metric is defined.
        for name in names:
            g.spill_costs[name] = 1.0
        color_graph(g)
        spilled = [n for n in names if n not in g.colors]
        assert len(spilled) == 1
        assert spilled[0] in g.pruned

    def test_caller_saved_preferred_over_callee_saved(self) -> None:
        """A pseudo with no special constraints should receive a caller-saved register."""
        instrs = [_mov_gp(asm.Imm(1), _pseudo("x")), _mov_gp(_pseudo("x"), _AX), asm.Ret()]
        g = _ready_graph(instrs, _gp_sym("x"))
        color_graph(g)
        callee_saved = {GPRegisters.BX, GPRegisters.R12, GPRegisters.R13, GPRegisters.R14, GPRegisters.R15}
        assert g.colors.get("x") not in callee_saved


# ---------------------------------------------------------------------------
# create_register_map
# ---------------------------------------------------------------------------


class TestCreateRegisterMap:
    """Verify create_register_map extracts only successfully coloured pseudos."""

    def test_empty_colors_returns_empty_dict(self) -> None:
        """An uncolored graph produces an empty map."""
        g = InterferenceGraph()
        assert create_register_map(g) == {}

    def test_hw_register_nodes_excluded(self) -> None:
        """Hardware-register keys in colors are not included in the result."""
        g = InterferenceGraph()
        g.colors[GPRegisters.AX] = GPRegisters.AX
        g.colors[GPRegisters.CX] = GPRegisters.CX
        assert create_register_map(g) == {}

    def test_colored_pseudo_included(self) -> None:
        """A successfully colored pseudo appears in the result with its assigned register."""
        g = InterferenceGraph()
        g.colors["x"] = GPRegisters.BX
        result = create_register_map(g)
        assert result == {"x": GPRegisters.BX}

    def test_spilled_pseudo_excluded(self) -> None:
        """A spilled pseudo (absent from colors) does not appear in the result."""
        g = InterferenceGraph()
        g.colors["x"] = GPRegisters.AX
        g.pruned.add("y")  # "y" never colored
        result = create_register_map(g)
        assert "x" in result
        assert "y" not in result

    def test_mixed_nodes_only_pseudos_returned(self) -> None:
        """Mixed colors dict: only str (pseudo) keys reach the output."""
        g = InterferenceGraph()
        g.colors["x"] = GPRegisters.AX
        g.colors["y"] = GPRegisters.CX
        g.colors[GPRegisters.DX] = GPRegisters.DX
        result = create_register_map(g)
        assert result == {"x": GPRegisters.AX, "y": GPRegisters.CX}


# ---------------------------------------------------------------------------
# replace_pseudoregs
# ---------------------------------------------------------------------------


class TestReplacePseudoregs:
    """Verify replace_pseudoregs rewrites Pseudo operands and prunes identity Movs."""

    def test_empty_instructions_returns_empty(self) -> None:
        """Empty input produces empty output."""
        assert replace_pseudoregs([], {}) == []

    def test_pseudo_in_map_replaced(self) -> None:
        """A pseudo present in register_map is replaced with the assigned Register."""
        reg_map: dict[str, GPRegisters | FPRegisters] = {"x": GPRegisters.CX}
        instrs = [_mov_gp(_pseudo("x"), _AX)]
        result = replace_pseudoregs(instrs, reg_map)
        assert len(result) == 1
        assert isinstance(result[0], asm.Mov)
        assert result[0].src == asm.Register(GPRegisters.CX, Width.DWORD)

    def test_pseudo_not_in_map_unchanged(self) -> None:
        """A pseudo absent from the map (spilled) is left as a Pseudo operand."""
        instrs = [_mov_gp(_pseudo("x"), _AX)]
        result = replace_pseudoregs(instrs, {})
        assert len(result) == 1
        assert result[0].src == _pseudo("x")  # type: ignore[union-attr]

    def test_mov_same_register_dropped(self) -> None:
        """A Mov whose source and destination both resolve to the same register is removed."""
        reg_map: dict[str, GPRegisters | FPRegisters] = {"x": GPRegisters.AX, "y": GPRegisters.AX}
        instrs = [_mov_gp(_pseudo("x"), _pseudo("y"))]
        assert replace_pseudoregs(instrs, reg_map) == []

    def test_mov_different_registers_kept(self) -> None:
        """A Mov whose source and destination resolve to distinct registers is kept."""
        reg_map: dict[str, GPRegisters | FPRegisters] = {"x": GPRegisters.AX, "y": GPRegisters.CX}
        instrs = [_mov_gp(_pseudo("x"), _pseudo("y"))]
        result = replace_pseudoregs(instrs, reg_map)
        assert len(result) == 1

    def test_non_mov_identity_not_dropped(self) -> None:
        """The identity-elimination rule applies only to Mov; other instructions are kept."""
        reg_map: dict[str, GPRegisters | FPRegisters] = {"x": GPRegisters.AX, "y": GPRegisters.AX}
        instrs = [asm.Cmp(op1=_pseudo("x"), op2=_pseudo("y"))]
        result = replace_pseudoregs(instrs, reg_map)
        assert len(result) == 1

    def test_quadword_mov_same_register_dropped(self) -> None:
        """A quadword Mov whose pseudo source resolves to the existing QWORD destination is removed."""
        reg_map: dict[str, GPRegisters | FPRegisters] = {"x": GPRegisters.AX}
        ax_q = asm.Register(GPRegisters.AX, Width.QWORD)
        instrs = [asm.Mov(src=_pseudo("x"), dst=ax_q, assembly_type=asm.Quadword())]
        assert replace_pseudoregs(instrs, reg_map) == []

    def test_fp_pseudo_replaced_with_xmm(self) -> None:
        """An FP pseudo is replaced with the assigned XMM register at QWORD width."""
        reg_map: dict[str, GPRegisters | FPRegisters] = {"d": FPRegisters.XMM2}
        instrs = [_mov_fp(_pseudo("d"), _XMM0)]
        result = replace_pseudoregs(instrs, reg_map)
        assert len(result) == 1
        assert result[0].src == asm.Register(FPRegisters.XMM2, Width.QWORD)  # type: ignore[union-attr]

    def test_non_pseudo_operands_untouched(self) -> None:
        """Immediate and hardware-register operands pass through unchanged."""
        reg_map: dict[str, GPRegisters | FPRegisters] = {"x": GPRegisters.CX}
        instrs = [_mov_gp(asm.Imm(42), _AX)]
        result = replace_pseudoregs(instrs, reg_map)
        assert result == instrs

    def test_cvttss2si_pseudo_replaced(self) -> None:
        """``Cvttss2si`` pseudo src and dst are replaced from the register map."""
        reg_map: dict[str, GPRegisters | FPRegisters] = {
            "f": FPRegisters.XMM1,
            "n": GPRegisters.AX,
        }
        instr = asm.Cvttss2si(src=_pseudo("f"), dst=_pseudo("n"), assembly_type=asm.Longword())
        result = replace_pseudoregs([instr], reg_map)
        assert len(result) == 1
        assert isinstance(result[0], asm.Cvttss2si)
        assert result[0].src == asm.Register(FPRegisters.XMM1, Width.DWORD)
        assert result[0].dst == asm.Register(GPRegisters.AX, Width.DWORD)

    def test_cvtsi2ss_pseudo_replaced(self) -> None:
        """``Cvtsi2ss`` pseudo src and dst are replaced from the register map."""
        reg_map: dict[str, GPRegisters | FPRegisters] = {
            "n": GPRegisters.CX,
            "f": FPRegisters.XMM3,
        }
        instr = asm.Cvtsi2ss(src=_pseudo("n"), dst=_pseudo("f"), assembly_type=asm.Longword())
        result = replace_pseudoregs([instr], reg_map)
        assert len(result) == 1
        assert isinstance(result[0], asm.Cvtsi2ss)
        assert result[0].src == asm.Register(GPRegisters.CX, Width.DWORD)
        assert result[0].dst == asm.Register(FPRegisters.XMM3, Width.DWORD)

    def test_cvtss2sd_pseudo_replaced(self) -> None:
        """``Cvtss2sd`` pseudo src and dst are replaced from the register map."""
        reg_map: dict[str, GPRegisters | FPRegisters] = {
            "f": FPRegisters.XMM1,
            "d": FPRegisters.XMM2,
        }
        instr = asm.Cvtss2sd(src=_pseudo("f"), dst=_pseudo("d"))
        result = replace_pseudoregs([instr], reg_map)
        assert len(result) == 1
        assert isinstance(result[0], asm.Cvtss2sd)
        assert result[0].src == asm.Register(FPRegisters.XMM1, Width.DWORD)
        assert result[0].dst == asm.Register(FPRegisters.XMM2, Width.QWORD)

    def test_cvtsd2ss_pseudo_replaced(self) -> None:
        """``Cvtsd2ss`` pseudo src and dst are replaced from the register map."""
        reg_map: dict[str, GPRegisters | FPRegisters] = {
            "d": FPRegisters.XMM0,
            "f": FPRegisters.XMM1,
        }
        instr = asm.Cvtsd2ss(src=_pseudo("d"), dst=_pseudo("f"))
        result = replace_pseudoregs([instr], reg_map)
        assert len(result) == 1
        assert isinstance(result[0], asm.Cvtsd2ss)
        assert result[0].src == asm.Register(FPRegisters.XMM0, Width.QWORD)
        assert result[0].dst == asm.Register(FPRegisters.XMM1, Width.DWORD)
