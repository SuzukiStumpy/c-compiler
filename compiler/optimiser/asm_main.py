"""Top-level register allocation pipeline for the assembly representation.

Implements graph-colouring register allocation over a function's instruction
list in two independent passes — one for general-purpose registers and one for
SIMD floating-point registers — then rewrites pseudo-register operands with the
allocated hardware registers.
"""

import dataclasses
import math
from typing import cast

import compiler.syntax.asm as asm
from compiler.optimiser.cfg import (
    ASM_ADAPTER,
    BasicBlock,
    BlockId,
    BoundaryNode,
    ControlFlowGraph,
    GraphNode,
    make_cfg,
)
from compiler.optimiser.interference_graph import InterferenceGraph
from compiler.syntax.asm.operands import ALLOCATABLE_FP, ALLOCATABLE_GP, FPRegisters, GPRegisters, Width
from compiler.syntax.asm.values import AssemblyType, Byte, Longword
from compiler.syntax.ast.types import (
    Char,
    DoubleType,
    FunType,
    IntType,
    LongType,
    Pointer,
    SChar,
    Structure,
    UChar,
    UIntType,
    ULongType,
    Union,
)
from compiler.types.symbols import ConstantAttr, LocalAttr, StaticAttr, StructTypeTable, SymbolTable

# ---------------------------------------------------------------------------
# Pseudo-register classification
# ---------------------------------------------------------------------------


def _find_aliased_vars(
    instructions: list[asm.Instruction],
    symbol_table: SymbolTable | None = None,
) -> frozenset[str]:
    """Return pseudo-register names that are ineligible for register allocation.

    A pseudo is ineligible when its address is taken (it appears as the source
    operand of a ``Lea`` instruction) or it has static / constant storage
    duration.

    Args:
        instructions: Flat instruction list to scan for ``Lea`` sources.
        symbol_table: Program symbol table used to identify static-duration
            variables.  Pass ``None`` to skip that check.

    Returns:
        A ``frozenset`` of pseudo names that must not be assigned to hardware
        registers.
    """
    address_taken: frozenset[str] = frozenset(
        instr.src.name for instr in instructions if isinstance(instr, asm.Lea) and isinstance(instr.src, asm.Pseudo)
    )
    if symbol_table is None:
        return address_taken
    statics: frozenset[str] = frozenset(
        name for name, entry in symbol_table.items() if isinstance(entry.attrs, (StaticAttr, ConstantAttr))
    )
    return address_taken | statics


def _all_pseudo_operands(instr: asm.Instruction) -> list[asm.Pseudo]:
    """Return every ``Pseudo`` operand present in ``instr``, from any field.

    Args:
        instr: The assembly instruction to inspect.

    Returns:
        A list of all ``Pseudo`` operands found in the instruction's fields.
        Returns an empty list for instructions with no operand fields or no
        ``Pseudo`` operands.
    """
    match instr:
        case asm.Mov(src=s, dst=d) | asm.Movsx(src=s, dst=d) | asm.MoveZeroExtend(src=s, dst=d):
            ops: list[asm.Operand] = [s, d]
        case asm.Binary(src1=s1, src2=s2, dst=d):
            ops = [s1, s2, d]
        case asm.Unary(operand=op) | asm.SetCC(operand=op) | asm.Push(operand=op):
            ops = [op]
        case asm.Idiv(operand=op) | asm.Div(operand=op):
            ops = [op]
        case asm.Shl(operand=op) | asm.Sar(operand=op) | asm.Shr(operand=op):
            ops = [op]
        case asm.Cmp(op1=o1, op2=o2):
            ops = [o1, o2]
        case asm.Pop(target=t) | asm.IndirectCall(target=t):
            ops = [t]
        case asm.Lea(src=s, dst=d):
            ops = [s, d]
        case asm.Cvttsd2si(src=s, dst=d) | asm.Cvtsi2sd(src=s, dst=d):
            ops = [s, d]
        case _:
            ops = []
    return [op for op in ops if isinstance(op, asm.Pseudo)]


def _classify_pseudos(
    instructions: list[asm.Instruction],
    symbol_table: SymbolTable | None = None,
) -> tuple[set[str], set[str]]:
    """Classify every allocatable pseudo-register in ``instructions`` as GP or FP.

    Uses the symbol table C type to assign each pseudo to a register class:

    * ``DoubleType`` → FP (XMM registers).
    * ``LongType``, ``ULongType``, ``Pointer`` → GP (Quadword).
    * ``IntType``, ``UIntType`` → GP (Longword).
    * ``Char``, ``SChar``, ``UChar`` → GP (Byte).
    * All other types (arrays, structs, functions) → not allocatable.

    Pseudos excluded from both sets:

    * Address-taken pseudos (appear as the source of a ``Lea`` instruction).
    * Static / constant-duration pseudos (``StaticAttr`` / ``ConstantAttr``).
    * Pseudos absent from ``symbol_table`` (unresolvable; skipped).

    Args:
        instructions: The flat instruction list to scan.
        symbol_table: The program symbol table.  Pass ``None`` to skip
            classification entirely (useful when the symbol table is
            unavailable, e.g. in isolated unit tests).

    Returns:
        A ``(gp_pseudos, fp_pseudos)`` tuple of pseudo-register name sets.
    """
    if symbol_table is None:
        return set(), set()

    aliased = _find_aliased_vars(instructions, symbol_table)
    gp: set[str] = set()
    fp: set[str] = set()

    for instr in instructions:
        for op in _all_pseudo_operands(instr):
            name = op.name
            if name in aliased:
                continue
            entry = symbol_table.get(name)
            if entry is None or not isinstance(entry.attrs, LocalAttr):
                continue
            match entry.t:
                case DoubleType():
                    fp.add(name)
                case LongType() | ULongType() | Pointer() | IntType() | UIntType() | Char() | SChar() | UChar():
                    gp.add(name)

    return gp, fp


# ---------------------------------------------------------------------------
# Instruction operand analysis
# ---------------------------------------------------------------------------

# SysV ABI argument-passing registers in order.
_GP_ARG_REGS: tuple[GPRegisters, ...] = (
    GPRegisters.DI,
    GPRegisters.SI,
    GPRegisters.DX,
    GPRegisters.CX,
    GPRegisters.R8,
    GPRegisters.R9,
)
_FP_ARG_REGS: tuple[FPRegisters, ...] = (
    FPRegisters.XMM0,
    FPRegisters.XMM1,
    FPRegisters.XMM2,
    FPRegisters.XMM3,
    FPRegisters.XMM4,
    FPRegisters.XMM5,
    FPRegisters.XMM6,
    FPRegisters.XMM7,
)

# GP caller-saved registers that a call instruction overwrites.
_GP_CALL_CLOBBERS: tuple[GPRegisters, ...] = (*_GP_ARG_REGS, GPRegisters.AX)
# All allocatable XMM registers are caller-saved on Linux SysV — none are callee-saved.
_FP_CALL_CLOBBERS: tuple[FPRegisters, ...] = (
    FPRegisters.XMM0,
    FPRegisters.XMM1,
    FPRegisters.XMM2,
    FPRegisters.XMM3,
    FPRegisters.XMM4,
    FPRegisters.XMM5,
    FPRegisters.XMM6,
    FPRegisters.XMM7,
    FPRegisters.XMM8,
    FPRegisters.XMM9,
    FPRegisters.XMM10,
    FPRegisters.XMM11,
    FPRegisters.XMM12,
    FPRegisters.XMM13,
)

# SysV callee-saved registers (must be preserved across calls).
_GP_CALLEE_SAVED: frozenset[GPRegisters] = frozenset(
    {
        GPRegisters.BX,
        GPRegisters.R12,
        GPRegisters.R13,
        GPRegisters.R14,
        GPRegisters.R15,
    }
)
# XMM registers are all caller-saved on Linux SysV ABI — no FP callee-saved registers.
_FP_CALLEE_SAVED: frozenset[FPRegisters] = frozenset()
_CALLEE_SAVED: frozenset[GPRegisters | FPRegisters] = _GP_CALLEE_SAVED | _FP_CALLEE_SAVED


def _hw(r: GPRegisters | FPRegisters) -> asm.Register:
    """Wrap a hardware register enum as a ``Register`` operand (QWORD width).

    Width is QWORD throughout; the exact value does not affect GraphNode
    extraction — only the register identity matters for interference analysis.

    Args:
        r: The hardware register enum member to wrap.

    Returns:
        A ``Register`` operand with ``Width.QWORD``.
    """
    return asm.Register(r, Width.QWORD)


# Pre-built synthesised operands for registers with implicit roles.
_R_AX: asm.Register = _hw(GPRegisters.AX)
_R_DX: asm.Register = _hw(GPRegisters.DX)
_R_CX: asm.Register = _hw(GPRegisters.CX)


def _call_param_regs(
    name: str,
    symbol_table: SymbolTable | None,
    struct_type_table: StructTypeTable | None = None,
) -> list[asm.Operand]:
    """Return the parameter-passing ``Register`` operands for a call to ``name``.

    Looks up the callee's ``FunType`` in ``symbol_table`` to determine how many
    GP and FP argument registers are consumed by its parameter list.  A
    ``DoubleType`` parameter uses the next available XMM register; struct
    parameters are classified per the SysV ABI (each eightbyte uses one register
    according to its class); all other scalar types use the next available GP
    argument register.  Structs > 16 bytes (MEMORY class) consume no registers.

    Falls back to all six GP argument registers when ``symbol_table`` is
    ``None``, when ``name`` is absent from the table, or when the entry's type
    is not a ``FunType`` (e.g. a function-pointer variable).

    Args:
        name: The callee symbol name as it appears in the ``Call`` instruction.
        symbol_table: The program symbol table, or ``None`` for a conservative
            fallback that treats all GP argument registers as used.
        struct_type_table: Struct layout table used to classify struct parameters.
            Pass ``None`` to fall back to treating each struct as a single GP arg.

    Returns:
        An ordered list of ``Register`` operands — one per eightbyte register slot —
        for the registers that carry the call's arguments.
    """
    if symbol_table is None:
        return [_hw(r) for r in _GP_ARG_REGS]
    entry = symbol_table.get(name)
    if entry is None or not isinstance(entry.t, FunType):
        return [_hw(r) for r in _GP_ARG_REGS]
    result: list[asm.Operand] = []
    gp_idx = 0
    fp_idx = 0
    for param_t in entry.t.params:
        if isinstance(param_t, (Structure, Union)) and struct_type_table is not None:
            # Lazy import avoids a circular module dependency (code_generator imports us).
            from compiler.code_generator import EightbyteClass, classify_struct  # type: ignore[attr-defined]

            classes = classify_struct(param_t.tag, struct_type_table)
            for cls in classes:
                if cls == EightbyteClass.SSE:
                    if fp_idx < len(_FP_ARG_REGS):
                        result.append(_hw(_FP_ARG_REGS[fp_idx]))
                        fp_idx += 1
                elif cls == EightbyteClass.INTEGER:
                    if gp_idx < len(_GP_ARG_REGS):
                        result.append(_hw(_GP_ARG_REGS[gp_idx]))
                        gp_idx += 1
                # EightbyteClass.MEMORY / NO_CLASS: no register allocated
        elif isinstance(param_t, DoubleType):
            if fp_idx < len(_FP_ARG_REGS):
                result.append(_hw(_FP_ARG_REGS[fp_idx]))
                fp_idx += 1
        else:
            if gp_idx < len(_GP_ARG_REGS):
                result.append(_hw(_GP_ARG_REGS[gp_idx]))
                gp_idx += 1
    return result


def find_used_and_updated(
    instr: asm.Instruction,
    symbol_table: SymbolTable | None = None,
    struct_type_table: StructTypeTable | None = None,
) -> tuple[list[asm.Operand], list[asm.Operand]]:
    """Return the operands read and written by a single assembly instruction.

    ``used`` is the set of operands whose values are read before (or instead
    of) being overwritten; ``updated`` is the set of operands that receive a
    new value.  Operands that are both read and written (e.g. the destination
    of a shift) appear in both lists.

    Implicit hardware-register effects are represented as synthesised
    ``Register`` operands with ``Width.QWORD``:

    * ``Idiv``/``Div`` — reads and overwrites ``AX`` and ``DX``
    * ``Cdq`` — reads ``AX``, overwrites ``DX``
    * ``Shl``/``Sar``/``Shr`` — reads ``CX`` (shift count)
    * ``Call``/``IndirectCall`` — reads the argument-passing registers,
      overwrites all GP caller-saved registers

    Instructions that have no allocatable operands (``Ret``, ``Label``, ``Jmp``,
    ``JmpCC``, ``AllocateStack``, ``DeallocateStack``) return empty lists.

    Args:
        instr: The assembly instruction to analyse.
        symbol_table: Used by the ``Call`` case to look up the callee's
            parameter types so that only the registers actually consumed by the
            argument list are included in ``used``.  Pass ``None`` for a
            conservative fallback.

    Returns:
        A ``(used, updated)`` tuple, each element being a list of
        ``asm.Operand`` objects.
    """
    match instr:
        case asm.Mov(src=src, dst=dst):
            if isinstance(dst, asm.Memory):
                # Store through pointer: reads src and the base-register address; writes no register.
                return ([src, dst], [])
            return ([src], [dst])
        case asm.Movsx(src=src, dst=dst) | asm.MoveZeroExtend(src=src, dst=dst):
            return ([src], [dst])
        case asm.Binary(src1=src1, src2=src2, dst=dst):
            return ([src1, src2], [dst])
        case asm.Unary(operand=op):
            return ([op], [op])
        case asm.Cmp(op1=op1, op2=op2):
            return ([op1, op2], [])
        case asm.SetCC(operand=dst):
            return ([], [dst])
        case asm.Push(operand=op):
            return ([op], [])
        case asm.Pop(target=target):
            return ([], [target])
        case asm.Idiv(operand=op) | asm.Div(operand=op):
            return ([op, _R_AX, _R_DX], [_R_AX, _R_DX])
        case asm.Cdq():
            return ([_R_AX], [_R_DX])
        case asm.Shl(operand=op) | asm.Sar(operand=op) | asm.Shr(operand=op):
            return ([op, _R_CX], [op])
        case asm.Lea(src=src, dst=dst):
            return ([src], [dst])
        case asm.Cvttsd2si(src=src, dst=dst) | asm.Cvtsi2sd(src=src, dst=dst):
            return ([src], [dst])
        case asm.Call(name=name):
            clobbers = [_hw(r) for r in (*_GP_CALL_CLOBBERS, *_FP_CALL_CLOBBERS)]
            return (_call_param_regs(name, symbol_table, struct_type_table), clobbers)
        case asm.IndirectCall(target=target):
            # No name → conservative: all GP and FP arg regs used; target (fn ptr) also used.
            used: list[asm.Operand] = [target, *(_hw(r) for r in _GP_ARG_REGS), *(_hw(r) for r in _FP_ARG_REGS)]
            clobbers = [_hw(r) for r in (*_GP_CALL_CLOBBERS, *_FP_CALL_CLOBBERS)]
            return (used, clobbers)
        case _:
            return ([], [])


# ---------------------------------------------------------------------------
# CFG construction
# ---------------------------------------------------------------------------


def _make_asm_cfg(instructions: list[asm.Instruction]) -> ControlFlowGraph[asm.Instruction]:
    """Build a CFG from a flat assembly instruction list using dummy function metadata.

    Args:
        instructions: The instruction list to decompose into basic blocks.

    Returns:
        A ``ControlFlowGraph[asm.Instruction]`` with successor edges populated.
    """
    return make_cfg(
        name="",
        globl=False,
        params=(),
        body=tuple(instructions),
        source_file="",
        source_line=0,
        adapter=ASM_ADAPTER,
    )


# ---------------------------------------------------------------------------
# Liveness analysis
# ---------------------------------------------------------------------------

_LiveMap = dict[int, tuple[set[GraphNode], set[GraphNode]]]
"""Liveness map: block index → (live_in, live_out)."""

# Conservative fallback: both integer and FP return-value registers live at exit.
# Overridden per-function by _compute_exit_live when the return type is known.
_EXIT_LIVE: frozenset[GraphNode] = frozenset({GPRegisters.AX, FPRegisters.XMM0})


def _compute_exit_live(fn_name: str | None, symbol_table: SymbolTable | None) -> frozenset[GraphNode]:
    """Return the set of hardware registers live at function exit.

    Inspects the function's return type so that only the register(s) that
    actually carry the return value are treated as live.  This prevents
    spurious interference edges — e.g. a double-returning function does not
    need AX live at exit, and an int-returning function does not need XMM0.

    Falls back to ``_EXIT_LIVE`` (both AX and XMM0) when the return type
    cannot be determined.

    Args:
        fn_name: Symbol-table name of the function being compiled.
        symbol_table: The program symbol table.

    Returns:
        A frozenset of ``GraphNode`` values (hardware registers) live at exit.
    """
    if fn_name is None or symbol_table is None:
        return _EXIT_LIVE
    entry = symbol_table.get(fn_name)
    if entry is None or not isinstance(entry.t, FunType):
        return _EXIT_LIVE
    ret = entry.t.ret
    if isinstance(ret, DoubleType):
        return frozenset({FPRegisters.XMM0})
    if isinstance(ret, (IntType, UIntType, LongType, ULongType, Pointer, Char, SChar, UChar)):
        return frozenset({GPRegisters.AX})
    # VoidType, Structure, Union — no return-value register is implicitly live at exit.
    return frozenset()


def _meet(
    block_idx: int,
    cfg: ControlFlowGraph[asm.Instruction],
    live_in: list[set[GraphNode]],
    tracked_nodes: set[GraphNode],
    exit_live: frozenset[GraphNode],
) -> set[GraphNode]:
    """Compute ``live_out`` for a block by merging liveness from all successors.

    The assembly meet function differs from the TAC version: when a successor
    is the ``EXIT`` sentinel the return-value registers are injected as live
    (filtered by ``exit_live`` and ``tracked_nodes``).  Encountering ``ENTRY``
    as a successor indicates a malformed CFG and raises an error.

    Args:
        block_idx: Index of the block whose ``live_out`` is being computed.
        cfg: The control-flow graph containing block and successor information.
        live_in: Current ``live_in`` sets indexed by block number.
        tracked_nodes: The set of nodes relevant to this pass; used to filter
            which exit-live registers are injected.
        exit_live: The set of registers live at function exit for this function,
            as produced by ``_compute_exit_live``.

    Returns:
        The merged ``live_out`` set for the block.

    Raises:
        ValueError: If ``ENTRY`` appears as a successor of any block.
    """
    result: set[GraphNode] = set()
    for succ in cfg.blocks[block_idx].successors:
        match succ:
            case BoundaryNode.EXIT:
                result |= exit_live & tracked_nodes
            case BoundaryNode.ENTRY:
                raise ValueError("Malformed control-flow graph: ENTRY found as a block successor")
            case BlockId(num=idx):
                result |= live_in[idx]
    return result


def _analyze_liveness(
    cfg: ControlFlowGraph[asm.Instruction],
    tracked_nodes: set[GraphNode],
    exit_live: frozenset[GraphNode],
) -> _LiveMap:
    """Compute live-in and live-out sets for every basic block via backward dataflow.

    Only nodes present in ``tracked_nodes`` are propagated; nodes outside it are
    filtered from every def and use set before the fixed-point loop runs.  This
    lets the GP and FP passes share the same algorithm while keeping their
    liveness information independent.

    ``live_out`` is computed via ``_meet``, which injects the return-value
    register(s) when a successor is the ``EXIT`` sentinel.

    For each block *b*:

        live_out[b] = _meet(b, live_in, tracked_nodes, exit_live)
        live_in[b]  = uses[b] ∪ (live_out[b] − defs[b])

    where ``uses[b]`` / ``defs[b]`` are the upward-exposed uses and definitions
    of the block as a whole, intersected with ``tracked_nodes``.

    Args:
        cfg: The assembly CFG whose blocks carry ``successors`` edges.
        tracked_nodes: The set of nodes to include in the analysis.  Nodes not
            in this set are ignored during propagation.
        exit_live: Registers live at function exit, as produced by
            ``_compute_exit_live``.

    Returns:
        A dict mapping each block's 0-based index to a ``(live_in, live_out)``
        tuple of ``GraphNode`` sets.
    """
    n = len(cfg.blocks)

    block_uses: list[set[GraphNode]] = []
    block_defs: list[set[GraphNode]] = []
    for block in cfg.blocks:
        b_uses: set[GraphNode] = set()
        b_defs: set[GraphNode] = set()
        for instr in block.instructions:
            b_uses |= (ASM_ADAPTER.uses(instr) & tracked_nodes) - b_defs
            b_defs |= ASM_ADAPTER.defs(instr) & tracked_nodes
        block_uses.append(b_uses)
        block_defs.append(b_defs)

    live_in: list[set[GraphNode]] = [set() for _ in range(n)]
    live_out: list[set[GraphNode]] = [set() for _ in range(n)]

    changed = True
    while changed:
        changed = False
        for i in range(n - 1, -1, -1):
            new_out = _meet(i, cfg, live_in, tracked_nodes, exit_live)
            new_in = block_uses[i] | (new_out - block_defs[i])
            if new_in != live_in[i] or new_out != live_out[i]:
                live_in[i] = new_in
                live_out[i] = new_out
                changed = True

    return {i: (live_in[i], live_out[i]) for i in range(n)}


# ---------------------------------------------------------------------------
# Transfer function and edge generation
# ---------------------------------------------------------------------------

# Union of all allocatable hardware registers — used to extract GraphNodes
# from operands without re-instantiating the set on every call.
_ALLOCATABLE: frozenset[GPRegisters | FPRegisters] = ALLOCATABLE_GP | ALLOCATABLE_FP


def _operands_to_nodes(operands: list[asm.Operand]) -> set[GraphNode]:
    """Extract allocatable ``GraphNode`` values from a list of assembly operands.

    Only operands that can reside in a hardware register contribute to the
    result:

    * ``Pseudo`` — the pseudo-register name (``str``)
    * ``Register`` with an allocatable enum — the register enum member
    * ``Memory`` with an allocatable base register — the base register enum
    * ``Indexed`` with allocatable base/index registers — both enum members

    ``Imm``, ``Data``, ``PseudoMem``, and ``Memory``/``Indexed`` whose base
    registers are non-allocatable (e.g. ``BP``, ``SP``, ``R10``, ``R11``)
    contribute nothing.

    Args:
        operands: The list of operands to process.

    Returns:
        The set of ``GraphNode`` values (pseudo names and/or register enums)
        extracted from ``operands``.
    """
    result: set[GraphNode] = set()
    for op in operands:
        match op:
            case asm.Pseudo(name=name):
                result.add(name)
            case asm.Register(reg=r) if r in _ALLOCATABLE:
                result.add(r)
            case asm.Memory(reg=base) if base.reg in _ALLOCATABLE:
                result.add(base.reg)
            case asm.Indexed(base=base, index=idx):
                if base.reg in _ALLOCATABLE:
                    result.add(base.reg)
                if idx.reg in _ALLOCATABLE:
                    result.add(idx.reg)
    return result


def _transfer(
    block: BasicBlock[asm.Instruction],
    end_live: set[GraphNode],
    tracked_nodes: set[GraphNode],
    symbol_table: SymbolTable | None = None,
    struct_type_table: StructTypeTable | None = None,
) -> tuple[set[GraphNode], list[set[GraphNode]]]:
    """Apply backward per-instruction liveness transfer to a basic block.

    Iterates through ``block.instructions`` in reverse, maintaining a
    ``current_live`` set that represents the registers live *after* each
    instruction.  At each instruction:

    * ❶ The current live set (= live_after for that instruction) is recorded.
    * ❷ The transfer is applied: nodes in ``updated`` are removed from
      ``current_live``; nodes in ``used`` are added.

    After all instructions the remaining ``current_live`` is the block's
    live_in (❸).

    Only nodes present in ``tracked_nodes`` are propagated; this allows
    the GP and FP passes to share the same implementation.

    Args:
        block: The basic block to analyse.
        end_live: The live set at the block exit (live_out), already filtered
            to ``tracked_nodes``.
        tracked_nodes: Nodes relevant to this pass; extracted nodes outside
            this set are discarded.
        symbol_table: Forwarded to ``find_used_and_updated`` for ``Call``
            parameter-register lookup.  Pass ``None`` for a conservative
            fallback.

    Returns:
        A ``(live_in, live_after)`` pair where ``live_in`` is the live set at
        the block entry and ``live_after[i]`` is the live set immediately
        after ``block.instructions[i]``.
    """
    current_live: set[GraphNode] = set(end_live)
    live_after_rev: list[set[GraphNode]] = []

    for instr in reversed(block.instructions):
        live_after_rev.append(set(current_live))  # ❶ annotate_instruction
        used, updated = find_used_and_updated(instr, symbol_table, struct_type_table)  # ❷
        current_live -= _operands_to_nodes(updated) & tracked_nodes
        current_live |= _operands_to_nodes(used) & tracked_nodes

    return current_live, list(reversed(live_after_rev))  # ❸ annotate_block


def _add_edges(
    cfg: ControlFlowGraph[asm.Instruction],
    live_map: _LiveMap,
    graph: InterferenceGraph,
    tracked_nodes: set[GraphNode],
    symbol_table: SymbolTable | None = None,
    struct_type_table: StructTypeTable | None = None,
) -> None:
    """Add interference edges to ``graph`` using per-instruction liveness.

    For each instruction, the live set recorded immediately after it (❶) is
    iterated.  For each live value ``l`` and each updated (defined) value ``u``:

    * ❷ If the instruction is a ``Mov`` and ``l`` is the source operand, the
      edge is skipped.  This is the *copy-coalescing* hint: the source and
      destination of a Mov may be assigned the same register, eliminating the
      Mov entirely, provided no other interference prevents it.
    * ❸ Otherwise, ``add_edge(l, u)`` is called when both ``l`` and ``u`` are
      nodes in the interference graph (guaranteed by ``tracked_nodes``
      filtering) and ``l ≠ u`` (enforced inside ``add_edge``).

    Args:
        cfg: The assembly CFG.
        live_map: Block-level liveness produced by ``_analyze_liveness``.
        graph: The interference graph to populate in place.
        tracked_nodes: The set of nodes relevant to this pass (equals
            ``graph.nodes`` for GP/FP passes).
        symbol_table: Forwarded to ``_transfer`` / ``find_used_and_updated``
            for ``Call`` parameter-register lookup.
        struct_type_table: Forwarded to ``_transfer`` / ``_call_param_regs``
            for struct parameter classification.
    """
    for i, block in enumerate(cfg.blocks):
        _, live_out = live_map[i]
        _, live_after_list = _transfer(block, live_out, tracked_nodes, symbol_table, struct_type_table)

        for instr, live_after in zip(block.instructions, live_after_list):
            _, updated = find_used_and_updated(instr, symbol_table, struct_type_table)
            updated_nodes = _operands_to_nodes(updated) & tracked_nodes

            # Pre-compute Mov source nodes once per instruction (❷ optimisation).
            mov_src_nodes: set[GraphNode] = _operands_to_nodes([instr.src]) if isinstance(instr, asm.Mov) else set()

            for live in live_after:  # ❶ live registers after this instruction
                if live in mov_src_nodes:  # ❷ skip Mov source → no edge with dst
                    continue
                for u in updated_nodes:  # ❸ add edge between live and each defined value
                    graph.add_edge(live, u)


# ---------------------------------------------------------------------------
# Graph construction — two independent passes
# ---------------------------------------------------------------------------


def build_gp_graph(
    instructions: list[asm.Instruction],
    symbol_table: SymbolTable | None = None,
    exit_live: frozenset[GraphNode] = _EXIT_LIVE,
    struct_type_table: StructTypeTable | None = None,
) -> InterferenceGraph:
    """Build the interference graph for the general-purpose register pass.

    Pre-populates nodes with all allocatable GP hardware registers
    (``ALLOCATABLE_GP``), then adds a node for every GP pseudo-register
    discovered by ``_classify_pseudos``, excluding any with static storage
    duration.  Liveness analysis and edge generation are restricted to GP
    nodes only.

    Args:
        instructions: The flat list of assembly instructions for a single function.
        symbol_table: Used to exclude static-duration pseudo-registers.  Pass
            ``None`` to treat all pseudos as allocatable.
        exit_live: Registers live at function exit, as produced by
            ``_compute_exit_live``.  Defaults to the conservative fallback.

    Returns:
        A populated ``InterferenceGraph`` for the GP allocation pass.
    """
    gp_pseudos, _ = _classify_pseudos(instructions, symbol_table)
    tracked = cast(set[GraphNode], set(ALLOCATABLE_GP) | gp_pseudos)
    graph = InterferenceGraph(nodes=set(tracked))
    cfg = _make_asm_cfg(instructions)
    live_map = _analyze_liveness(cfg, tracked, exit_live)
    _add_edges(cfg, live_map, graph, tracked, symbol_table, struct_type_table)
    return graph


def build_fp_graph(
    instructions: list[asm.Instruction],
    symbol_table: SymbolTable | None = None,
    exit_live: frozenset[GraphNode] = _EXIT_LIVE,
    struct_type_table: StructTypeTable | None = None,
) -> InterferenceGraph:
    """Build the interference graph for the floating-point register pass.

    Pre-populates nodes with all allocatable FP hardware registers
    (``ALLOCATABLE_FP``), then adds a node for every FP pseudo-register
    discovered by ``_classify_pseudos``, excluding any with static storage
    duration.  Liveness analysis and edge generation are restricted to FP
    nodes only.

    Args:
        instructions: The flat list of assembly instructions for a single function.
        symbol_table: Used to exclude static-duration pseudo-registers.  Pass
            ``None`` to treat all pseudos as allocatable.
        exit_live: Registers live at function exit, as produced by
            ``_compute_exit_live``.  Defaults to the conservative fallback.

    Returns:
        A populated ``InterferenceGraph`` for the FP allocation pass.
    """
    _, fp_pseudos = _classify_pseudos(instructions, symbol_table)
    tracked = cast(set[GraphNode], set(ALLOCATABLE_FP) | fp_pseudos)
    graph = InterferenceGraph(nodes=set(tracked))
    cfg = _make_asm_cfg(instructions)
    live_map = _analyze_liveness(cfg, tracked, exit_live)
    _add_edges(cfg, live_map, graph, tracked, symbol_table, struct_type_table)
    return graph


# ---------------------------------------------------------------------------
# Remaining pipeline stubs
# ---------------------------------------------------------------------------


def add_spill_costs(interference_graph: InterferenceGraph, instructions: list[asm.Instruction]) -> None:
    """Annotate each node in the interference graph with a spill cost.

    Hardware-register nodes receive ``math.inf`` — they are never spilled.
    Pseudo-register nodes receive a cost equal to the number of times the
    pseudo appears across all instruction operands; higher-frequency pseudos
    are preferred for colouring over lower-frequency ones.

    Modifies ``interference_graph.spill_costs`` in place.

    Args:
        interference_graph: The graph produced by ``build_gp_graph`` or
            ``build_fp_graph``.  Its ``nodes`` set determines which pseudos
            are counted.
        instructions: The flat instruction list used to count occurrences.
    """
    counts: dict[str, int] = {}
    for instr in instructions:
        for op in _all_pseudo_operands(instr):
            if op.name in interference_graph.nodes:
                counts[op.name] = counts.get(op.name, 0) + 1

    for node in interference_graph.nodes:
        if isinstance(node, str):
            interference_graph.spill_costs[node] = float(counts.get(node, 0))
        else:
            interference_graph.spill_costs[node] = math.inf


def color_graph(interference_graph: InterferenceGraph) -> None:
    """Assign a colour (hardware register) to each node via graph colouring.

    Uses a Chaitin-style simplification/select approach.  Hardware-register
    nodes are pre-coloured with themselves and excluded from simplification.
    Only pseudo-register nodes are pruned and re-coloured recursively.

    At each step the pseudo with the lowest unpruned degree (if < k) is pruned
    first (guaranteed colorable).  When no such pseudo exists the one with the
    lowest ``spill_cost / degree`` metric is chosen as a potential spill
    candidate.  After the recursion bottoms out, nodes are re-added and
    coloured greedily — preferring caller-saved registers (``min``) to avoid
    unnecessary callee-saved register pressure.  A pseudo that cannot be
    coloured stays pruned (spilled to memory by a subsequent pass).

    Modifies ``interference_graph.colors`` and ``interference_graph.pruned``
    in place.

    Args:
        interference_graph: The annotated graph produced by ``build_gp_graph`` or
            ``build_fp_graph`` followed by ``add_spill_costs``.
    """
    hw_nodes = [n for n in interference_graph.nodes if isinstance(n, (GPRegisters, FPRegisters))]
    sorted_colors: list[GPRegisters | FPRegisters] = sorted(
        hw_nodes,
        key=lambda r: (r in _CALLEE_SAVED, list(type(r)).index(r)),
    )
    k = len(sorted_colors)

    # Pre-color every hardware-register node with itself.
    for reg in hw_nodes:
        interference_graph.colors[reg] = reg

    _color_recursive(interference_graph, k, sorted_colors)


def _color_recursive(
    graph: InterferenceGraph,
    k: int,
    sorted_colors: list[GPRegisters | FPRegisters],
) -> None:
    """Recursive Chaitin-style simplification/select step for pseudo nodes.

    Hardware-register nodes are never pruned here; they are pre-coloured by
    ``color_graph`` and serve only as fixed constraints when computing
    available colours.

    Args:
        graph: The interference graph being coloured (mutated in place).
        k: Number of available colours (allocatable hardware registers).
        sorted_colors: Hardware registers in preference order — caller-saved
            first, callee-saved last.
    """
    remaining = {n for n in graph.nodes if isinstance(n, str) and n not in graph.pruned}

    # ❶ Base case: all pseudo nodes have been processed.
    if not remaining:
        return

    # ❷ Find any pseudo whose current unpruned degree is below k.
    chosen_node: GraphNode | None = None
    for node in sorted(remaining, key=str):
        if len(graph.neighbours(node) - graph.pruned) < k:
            chosen_node = node
            break

    # ❸ No easy pseudo — pick the one with the lowest spill_cost / degree.
    if chosen_node is None:
        best_metric = math.inf
        for node in sorted(remaining, key=str):
            degree = len(graph.neighbours(node) - graph.pruned)
            metric = graph.spill_costs.get(node, 0.0) / degree
            if metric < best_metric:
                chosen_node = node
                best_metric = metric

    assert chosen_node is not None, "_color_recursive: remaining is non-empty but no node chosen"
    graph.pruned.add(chosen_node)

    # ❹ Colour the rest of the graph first.
    _color_recursive(graph, k, sorted_colors)

    # ❺ Colour this pseudo.  Hardware-register neighbours always block their own
    #    register (they are pre-coloured with themselves); pseudo neighbours block
    #    whichever register was assigned to them.
    taken: set[GPRegisters | FPRegisters] = set()
    for neighbor in graph.neighbours(chosen_node):
        if isinstance(neighbor, (GPRegisters, FPRegisters)):
            taken.add(neighbor)
        elif neighbor in graph.colors:
            taken.add(graph.colors[neighbor])  # type: ignore[arg-type]

    available = [c for c in sorted_colors if c not in taken]

    if available:
        graph.colors[chosen_node] = available[0]  # min — prefer caller-saved
        graph.pruned.discard(chosen_node)


def create_register_map(interference_graph: InterferenceGraph) -> dict[str, GPRegisters | FPRegisters]:
    """Build a mapping from pseudo-register names to hardware registers.

    Reads the colour assignments produced by ``color_graph`` and returns a dict
    that ``replace_pseudoregs`` can use for the substitution pass.
    Pseudo-registers marked for spilling (absent from ``colors``) are omitted.

    Args:
        interference_graph: The coloured graph produced by ``color_graph``.

    Returns:
        A dict mapping each coloured pseudo-register name to its assigned
        ``GPRegisters`` or ``FPRegisters`` member.  Hardware-register nodes in
        ``colors`` are excluded; only ``str`` keys are returned.
    """
    return {node: color for node, color in interference_graph.colors.items() if isinstance(node, str)}


def _asm_type_to_width(assembly_type: AssemblyType) -> Width:
    """Map an ``AssemblyType`` to the ``Width`` used for ``Register`` operands.

    Args:
        assembly_type: The instruction's assembly type.

    Returns:
        ``Width.BYTE`` for ``Byte``, ``Width.DWORD`` for ``Longword``,
        and ``Width.QWORD`` for ``Quadword`` or ``Double``.
    """
    if isinstance(assembly_type, Byte):
        return Width.BYTE
    if isinstance(assembly_type, Longword):
        return Width.DWORD
    return Width.QWORD


def _resolve_op(
    op: asm.Operand,
    reg_map: dict[str, GPRegisters | FPRegisters],
    width: Width,
) -> asm.Operand:
    """Replace a ``Pseudo`` operand with its allocated ``Register``, if available.

    Pseudos absent from ``reg_map`` (spilled) are returned unchanged.

    Args:
        op: The operand to resolve.
        reg_map: Pseudo name → hardware register mapping.
        width: The ``Width`` to use when constructing the replacement ``Register``.

    Returns:
        A ``Register`` operand when ``op`` is a ``Pseudo`` present in ``reg_map``;
        ``op`` itself otherwise.
    """
    if isinstance(op, asm.Pseudo) and op.name in reg_map:
        return asm.Register(reg_map[op.name], width)
    return op


def _replace_in_instr(
    instr: asm.Instruction,
    reg_map: dict[str, GPRegisters | FPRegisters],
) -> asm.Instruction:
    """Rewrite the ``Pseudo`` operands of a single instruction using ``reg_map``.

    Args:
        instr: The instruction to transform.
        reg_map: Pseudo name → hardware register mapping.

    Returns:
        A new instruction with all mapped ``Pseudo`` operands replaced by
        ``Register`` operands; ``instr`` itself when no operand changes.
    """
    match instr:
        case asm.Mov(src=src, dst=dst, assembly_type=at):
            w = _asm_type_to_width(at)
            return dataclasses.replace(instr, src=_resolve_op(src, reg_map, w), dst=_resolve_op(dst, reg_map, w))
        case asm.Movsx(src_type=st, dst_type=dt, src=src, dst=dst):
            return dataclasses.replace(
                instr,
                src=_resolve_op(src, reg_map, _asm_type_to_width(st)),
                dst=_resolve_op(dst, reg_map, _asm_type_to_width(dt)),
            )
        case asm.MoveZeroExtend(src_type=st, dst_type=dt, src=src, dst=dst):
            src_w = _asm_type_to_width(st)
            # Longword→Quadword emits `movl src32, dst32` (CPU auto-zero-extends upper 32 bits),
            # so both operands must use DWORD width.  Byte variants use the actual dst_type width.
            dst_w = src_w if not isinstance(st, Byte) else _asm_type_to_width(dt)
            return dataclasses.replace(
                instr, src=_resolve_op(src, reg_map, src_w), dst=_resolve_op(dst, reg_map, dst_w)
            )
        case asm.Binary(src1=s1, src2=s2, dst=dst, assembly_type=at):
            w = _asm_type_to_width(at)
            return dataclasses.replace(  # type: ignore[arg-type]
                instr,
                src1=_resolve_op(s1, reg_map, w),
                src2=_resolve_op(s2, reg_map, w),
                dst=_resolve_op(dst, reg_map, w),
            )
        case asm.Unary(operand=op, assembly_type=at):
            return dataclasses.replace(instr, operand=_resolve_op(op, reg_map, _asm_type_to_width(at)))
        case asm.Idiv(operand=op, assembly_type=at):
            return dataclasses.replace(instr, operand=_resolve_op(op, reg_map, _asm_type_to_width(at)))
        case asm.Div(operand=op, assembly_type=at):
            return dataclasses.replace(instr, operand=_resolve_op(op, reg_map, _asm_type_to_width(at)))
        case asm.Shl(operand=op, assembly_type=at):
            return dataclasses.replace(instr, operand=_resolve_op(op, reg_map, _asm_type_to_width(at)))
        case asm.Sar(operand=op, assembly_type=at):
            return dataclasses.replace(instr, operand=_resolve_op(op, reg_map, _asm_type_to_width(at)))
        case asm.Shr(operand=op, assembly_type=at):
            return dataclasses.replace(instr, operand=_resolve_op(op, reg_map, _asm_type_to_width(at)))
        case asm.Cmp(op1=o1, op2=o2, assembly_type=at):
            w = _asm_type_to_width(at)
            return dataclasses.replace(instr, op1=_resolve_op(o1, reg_map, w), op2=_resolve_op(o2, reg_map, w))
        case asm.SetCC(operand=op):
            return dataclasses.replace(instr, operand=_resolve_op(op, reg_map, Width.BYTE))
        case asm.Push(operand=op, assembly_type=at):
            # pushq always encodes a 64-bit operand; assembly_type here is the value type, not the reg width.
            return dataclasses.replace(instr, operand=_resolve_op(op, reg_map, Width.QWORD))
        case asm.IndirectCall(target=t):
            return dataclasses.replace(instr, target=_resolve_op(t, reg_map, Width.QWORD))
        case asm.Lea(src=src, dst=dst):
            return dataclasses.replace(
                instr,
                src=_resolve_op(src, reg_map, Width.QWORD),
                dst=_resolve_op(dst, reg_map, Width.QWORD),
            )
        case asm.Cvttsd2si(src=src, dst=dst, assembly_type=at):
            # src is an XMM register (Double width = QWORD); dst is the integer result
            return dataclasses.replace(
                instr,
                src=_resolve_op(src, reg_map, Width.QWORD),
                dst=_resolve_op(dst, reg_map, _asm_type_to_width(at)),
            )
        case asm.Cvtsi2sd(src=src, dst=dst, assembly_type=at):
            # src is the integer source; dst is an XMM register (QWORD)
            return dataclasses.replace(
                instr,
                src=_resolve_op(src, reg_map, _asm_type_to_width(at)),
                dst=_resolve_op(dst, reg_map, Width.QWORD),
            )
        case _:
            return instr


def replace_pseudoregs(
    instructions: list[asm.Instruction],
    register_map: dict[str, GPRegisters | FPRegisters],
) -> list[asm.Instruction]:
    """Substitute pseudo-register operands with their allocated hardware registers.

    For each instruction, replaces any ``Pseudo`` operand whose name appears in
    ``register_map`` with the corresponding ``Register`` operand.  Pseudos absent
    from the map (spilled) are left unchanged.  After substitution, any ``Mov``
    instruction whose source and destination have become identical ``Register``
    operands is discarded — it is a no-op produced by copy coalescing.

    Args:
        instructions: The flat instruction list to transform.
        register_map: Mapping from pseudo-register name to hardware register,
            as produced by ``create_register_map``.

    Returns:
        A new instruction list with allocated pseudo-register operands replaced
        and redundant self-moves removed.
    """
    result: list[asm.Instruction] = []
    for instr in instructions:
        new_instr = _replace_in_instr(instr, register_map)
        if isinstance(new_instr, asm.Mov) and new_instr.src == new_instr.dst:
            continue
        # MoveZeroExtend(Longword→Quadword) emits movl, which auto-zero-extends the upper 32 bits
        # of its destination register.  When src and dst coalesce to the same register the write
        # is a no-op (writing %eax already zeroes the upper 32 bits of %rax).
        if (
            isinstance(new_instr, asm.MoveZeroExtend)
            and isinstance(new_instr.src_type, Longword)
            and new_instr.src == new_instr.dst
        ):
            continue
        result.append(new_instr)
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def allocate_registers(
    instructions: list[asm.Instruction],
    symbol_table: SymbolTable | None = None,
    fn_name: str | None = None,
    struct_type_table: StructTypeTable | None = None,
) -> tuple[list[asm.Instruction], list[GPRegisters], list[FPRegisters]]:
    """Run the full register allocation pipeline on a function's instruction list.

    Executes two independent passes — one for GP registers, one for FP
    registers — then combines the resulting maps and rewrites the instruction
    list.  Pseudo-registers with static storage duration are excluded from
    allocation and left untouched in the output.

    Args:
        instructions: The flat list of assembly instructions for a single function,
            containing ``Pseudo`` operands to be replaced.
        symbol_table: Used to exclude static-duration pseudo-registers.  Pass
            ``None`` to treat all pseudos as allocatable.
        fn_name: The name of the function being compiled, used to determine
            which registers are live at function exit based on the return type.
            Pass ``None`` for the conservative fallback (both AX and XMM0 live).

    Returns:
        A three-tuple ``(rewritten_instructions, gp_callee_saved, fp_callee_saved)``
        where ``rewritten_instructions`` has pseudo-register operands replaced by
        hardware ``Register`` operands, ``gp_callee_saved`` is the sorted list of
        callee-saved GP registers that were assigned to pseudos (must be saved and
        restored by the caller), and ``fp_callee_saved`` is the equivalent list for
        FP (XMM) registers.
    """
    exit_live = _compute_exit_live(fn_name, symbol_table)

    gp_graph = build_gp_graph(instructions, symbol_table, exit_live, struct_type_table)
    add_spill_costs(gp_graph, instructions)
    color_graph(gp_graph)
    gp_map = create_register_map(gp_graph)

    fp_graph = build_fp_graph(instructions, symbol_table, exit_live, struct_type_table)
    add_spill_costs(fp_graph, instructions)
    color_graph(fp_graph)
    fp_map = create_register_map(fp_graph)

    rewritten = replace_pseudoregs(instructions, {**gp_map, **fp_map})

    gp_callee_used: list[GPRegisters] = sorted(
        {reg for reg in gp_map.values() if reg in _GP_CALLEE_SAVED},
        key=lambda r: list(GPRegisters).index(r),
    )
    fp_callee_used: list[FPRegisters] = sorted(
        {reg for reg in fp_map.values() if reg in _FP_CALLEE_SAVED},
        key=lambda r: list(FPRegisters).index(r),
    )
    return rewritten, gp_callee_used, fp_callee_used
