"""Control-flow graph types and the TAC <-> CFG conversion functions.

A ``ControlFlowGraph[I]`` decomposes the flat instruction stream of a function
into ``BasicBlock[I]`` objects parameterised by the instruction type ``I``.
The decomposition is lossless: ``cfg_to_instructions`` always reconstructs the
original TAC function exactly.

Instruction-set-specific knowledge (how to detect labels, returns, and jumps)
is isolated behind the ``CfgAdapter[I]`` protocol.  ``TacAdapter`` implements
it for TAC instructions; ``AsmAdapter`` implements it for the assembly AST.

Each ``BasicBlock`` carries a ``successors`` list populated by ``add_all_edges``,
which is called automatically by ``make_cfg`` and ``make_control_flow_graph``.
Passes that mutate the block list should call ``add_all_edges`` again before
relying on successor data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Final, Generic, Protocol, TypeAlias, TypeVar

import compiler.syntax.asm as asm
import compiler.syntax.tac as tac
from compiler.syntax.asm.operands import (
    ALLOCATABLE_FP,
    ALLOCATABLE_GP,
    FPRegisters,
    GPRegisters,
    Indexed,
    Memory,
    Pseudo,
    Register,
)

# ---------------------------------------------------------------------------
# Graph node identifiers
# ---------------------------------------------------------------------------


class BoundaryNode(Enum):
    """Synthetic boundary nodes that bracket every control-flow graph.

    These are not real basic blocks; they model the implicit transfer of
    control into the function (``ENTRY``) and out of it (``EXIT``).
    """

    ENTRY = auto()
    EXIT = auto()


ENTRY: Final = BoundaryNode.ENTRY
"""Sentinel for the implicit control-flow entry point of a function."""

EXIT: Final = BoundaryNode.EXIT
"""Sentinel for the implicit control-flow exit point of a function."""


@dataclass(frozen=True, slots=True)
class BlockId:
    """Identifies a basic block by its 0-based index in ``ControlFlowGraph.blocks``.

    Attributes:
        num: 0-based index into ``ControlFlowGraph.blocks``.
    """

    num: int


NodeId: TypeAlias = BlockId | BoundaryNode
"""A node identifier in a control-flow graph: a basic-block index or a boundary sentinel."""


# ---------------------------------------------------------------------------
# Generic CFG data structures
# ---------------------------------------------------------------------------

_I = TypeVar("_I")
_I_contra = TypeVar("_I_contra", contravariant=True)


@dataclass
class BasicBlock(Generic[_I]):
    """A maximal, linear sequence of instructions with a single entry and exit.

    A block starts at the beginning of a function body or at a label
    instruction.  It ends immediately before the next label, at a
    terminator (return, unconditional jump, or conditional jump),
    or at the end of the function body.

    The leading label instruction (if any) is stored as the first element of
    ``instructions`` so that reconstruction functions can reproduce the original
    stream exactly without special-casing.

    Attributes:
        label: The name carried by the leading label instruction, or ``None``
            for an implicit entry block that precedes the first label.
        instructions: Ordered sequence of instructions in this block, including
            the leading label (if any) and the trailing terminator (if any).
        successors: Ordered list of ``NodeId`` values to which control may
            transfer from this block.  Populated by ``add_all_edges``; empty
            until that function runs.
    """

    label: str | None
    instructions: list[_I]
    successors: list[NodeId] = field(default_factory=list)


@dataclass
class ControlFlowGraph(Generic[_I]):
    """A control-flow graph (CFG) derived from a single function.

    The CFG decomposes a flat instruction stream into ``BasicBlock`` objects.
    Block order is preserved so that reconstruction functions faithfully
    reproduce the original function.  ``ENTRY`` implicitly connects to
    ``blocks[0]`` (when ``blocks`` is non-empty).

    Attributes:
        name: The function name, carried through for reconstruction.
        globl: Whether the function bears a global-visibility directive.
        params: The ordered sequence of parameter descriptors.  The element
            type depends on the instruction set: ``str`` for TAC, ``Operand``
            for assembly.
        blocks: Ordered list of basic blocks parameterised by instruction type.
        source_file: Source file path propagated from the original function.
        source_line: 1-indexed source line propagated from the original function.
    """

    name: str
    globl: bool
    params: tuple[Any, ...]
    blocks: list[BasicBlock[_I]]
    source_file: str
    source_line: int


# ---------------------------------------------------------------------------
# CfgAdapter protocol
# ---------------------------------------------------------------------------


class CfgAdapter(Protocol[_I_contra]):
    """Instruction-set-specific predicates required to build and annotate a CFG.

    Implementations isolate knowledge about a particular instruction set (TAC,
    x86 assembly, ...) so that the generic ``make_cfg`` and ``add_all_edges``
    functions remain instruction-set-agnostic.

    The type parameter is contravariant because all four methods consume
    instructions as inputs and produce only primitive results.
    """

    def label_name(self, instr: _I_contra) -> str | None:
        """Return the label name when *instr* is a label instruction, else ``None``.

        Args:
            instr: The instruction to inspect.

        Returns:
            The label identifier string, or ``None`` if *instr* is not a label.
        """
        ...

    def is_return(self, instr: _I_contra) -> bool:
        """Return ``True`` when *instr* unconditionally exits the function.

        Args:
            instr: The instruction to inspect.

        Returns:
            ``True`` if *instr* is a return, ``False`` otherwise.
        """
        ...

    def jump_target(self, instr: _I_contra) -> str | None:
        """Return the target label when *instr* is an unconditional jump, else ``None``.

        Args:
            instr: The instruction to inspect.

        Returns:
            The target label name, or ``None`` if *instr* is not an unconditional jump.
        """
        ...

    def conditional_jump_target(self, instr: _I_contra) -> str | None:
        """Return the target label when *instr* is a conditional jump, else ``None``.

        Args:
            instr: The instruction to inspect.

        Returns:
            The target label name, or ``None`` if *instr* is not a conditional jump.
        """
        ...


# ---------------------------------------------------------------------------
# TAC adapter
# ---------------------------------------------------------------------------


class TacAdapter:
    """``CfgAdapter`` implementation for TAC instructions.

    Recognises ``tac.Label``, ``tac.Return``, ``tac.Jump``,
    ``tac.JumpIfZero``, and ``tac.JumpIfNotZero``.
    """

    def label_name(self, instr: tac.Instruction) -> str | None:
        """Return the label name when *instr* is a ``tac.Label``, else ``None``.

        Args:
            instr: The TAC instruction to inspect.

        Returns:
            The label name string, or ``None``.
        """
        return instr.name if isinstance(instr, tac.Label) else None

    def is_return(self, instr: tac.Instruction) -> bool:
        """Return ``True`` when *instr* is a ``tac.Return``.

        Args:
            instr: The TAC instruction to inspect.

        Returns:
            ``True`` if *instr* is a ``tac.Return``.
        """
        return isinstance(instr, tac.Return)

    def jump_target(self, instr: tac.Instruction) -> str | None:
        """Return the target label when *instr* is a ``tac.Jump``, else ``None``.

        Args:
            instr: The TAC instruction to inspect.

        Returns:
            The ``target`` field of a ``tac.Jump``, or ``None``.
        """
        return instr.target if isinstance(instr, tac.Jump) else None

    def conditional_jump_target(self, instr: tac.Instruction) -> str | None:
        """Return the target label when *instr* is a ``tac.JumpIfZero`` or
        ``tac.JumpIfNotZero``, else ``None``.

        Args:
            instr: The TAC instruction to inspect.

        Returns:
            The ``target`` field of the conditional jump, or ``None``.
        """
        if isinstance(instr, (tac.JumpIfZero, tac.JumpIfNotZero)):
            return instr.target
        return None


TAC_ADAPTER: Final = TacAdapter()
"""Singleton ``TacAdapter`` for use with TAC-based CFGs."""


# ---------------------------------------------------------------------------
# Generic CFG construction
# ---------------------------------------------------------------------------


def make_cfg(
    name: str,
    globl: bool,
    params: tuple[Any, ...],
    body: tuple[_I, ...],
    source_file: str,
    source_line: int,
    adapter: CfgAdapter[_I],
) -> ControlFlowGraph[_I]:
    """Build a control-flow graph from a flat instruction sequence.

    A new basic block begins at every label instruction and after every
    conditional jump.  Splitting on conditional jumps produces true basic
    blocks - exactly one exit per block - which is required for correct
    dataflow analysis: without the split, instructions that follow a
    mid-block conditional jump would appear to execute on *both* the
    fall-through and the jump-taken paths, causing the exit reaching-copy
    set to be over-approximate and leading to incorrect propagation.

    The label instruction is retained as the first instruction of its block
    so that ``cfg_to_instructions`` can reproduce the original flat stream
    exactly.

    ``add_all_edges`` is called before returning so that every block's
    ``successors`` list is fully populated.

    Args:
        name: Function name forwarded to the resulting ``ControlFlowGraph``.
        globl: Global-visibility flag forwarded to the resulting ``ControlFlowGraph``.
        params: Parameter descriptors forwarded to the resulting ``ControlFlowGraph``.
        body: The flat instruction sequence to partition.
        source_file: Source file path forwarded to the resulting ``ControlFlowGraph``.
        source_line: Source line forwarded to the resulting ``ControlFlowGraph``.
        adapter: Instruction-set adapter used to identify labels and terminators.

    Returns:
        A ``ControlFlowGraph[_I]`` whose blocks, flattened in order, reproduce
        *body* exactly, with successor edges wired up.
    """
    blocks: list[BasicBlock[_I]] = []
    current_label: str | None = None
    current_instrs: list[_I] = []

    for instr in body:
        lbl = adapter.label_name(instr)
        if lbl is not None:
            if current_instrs:
                blocks.append(BasicBlock(label=current_label, instructions=current_instrs))
            current_label = lbl
            current_instrs = [instr]
        else:
            current_instrs.append(instr)
            # Close the block after any terminator (return, unconditional jump,
            # or conditional jump) so that each block has exactly one exit and
            # dead instructions after a return/jump are isolated in their own
            # unreachable blocks.  This prevents spurious liveness propagation
            # from dead code (e.g. Ret(); Jmp(end)) that would otherwise poison
            # the live sets of all predecessor blocks.
            if (
                adapter.is_return(instr)
                or adapter.jump_target(instr) is not None
                or adapter.conditional_jump_target(instr) is not None
            ):
                blocks.append(BasicBlock(label=current_label, instructions=current_instrs))
                current_label = None
                current_instrs = []

    if current_instrs:
        blocks.append(BasicBlock(label=current_label, instructions=current_instrs))

    cfg: ControlFlowGraph[_I] = ControlFlowGraph(
        name=name,
        globl=globl,
        params=params,
        blocks=blocks,
        source_file=source_file,
        source_line=source_line,
    )
    add_all_edges(cfg, adapter)
    return cfg


def add_all_edges(cfg: ControlFlowGraph[_I], adapter: CfgAdapter[_I]) -> None:
    """Populate the ``successors`` list of every basic block in the CFG.

    Edges are derived from each block's last instruction using *adapter*:

    * Return -> ``EXIT``
    * Unconditional jump -> the block whose leading label matches the target
    * Conditional jump -> both the target block and the fall-through successor
      (the next block in order, or ``EXIT`` for the last block)
    * Any other instruction -> fall-through successor only

    An implicit ``ENTRY -> blocks[0]`` edge is not stored; callers may treat
    ``blocks[0]`` as the entry point directly.

    Existing ``successors`` lists are cleared before re-computation, so this
    function is safe to call multiple times after the block list is mutated.

    Args:
        cfg: The control-flow graph to annotate in place.
        adapter: Instruction-set adapter used to classify the last instruction
            of each block.
    """
    for block in cfg.blocks:
        block.successors.clear()

    if not cfg.blocks:
        return

    label_index: dict[str, BlockId] = {
        block.label: BlockId(i) for i, block in enumerate(cfg.blocks) if block.label is not None
    }

    max_idx = len(cfg.blocks) - 1

    for i, block in enumerate(cfg.blocks):
        fallthrough: NodeId = EXIT if i == max_idx else BlockId(i + 1)

        if not block.instructions:
            block.successors.append(fallthrough)
            continue

        last = block.instructions[-1]

        if adapter.is_return(last):
            block.successors.append(EXIT)
        elif (target := adapter.jump_target(last)) is not None:
            dest = label_index.get(target)
            block.successors.append(dest if dest is not None else EXIT)
        elif (target := adapter.conditional_jump_target(last)) is not None:
            dest = label_index.get(target)
            block.successors.append(dest if dest is not None else EXIT)
            block.successors.append(fallthrough)
        else:
            block.successors.append(fallthrough)


# ---------------------------------------------------------------------------
# TAC-specific convenience wrappers
# ---------------------------------------------------------------------------


def make_control_flow_graph(function: tac.Function) -> ControlFlowGraph[tac.Instruction]:
    """Decompose a TAC function into a control-flow graph of basic blocks.

    Convenience wrapper around ``make_cfg`` that uses ``TAC_ADAPTER`` and
    extracts the relevant fields from *function*.  Successor edges are wired
    up before returning.

    Args:
        function: The TAC function to decompose.

    Returns:
        A ``ControlFlowGraph[tac.Instruction]`` whose blocks, flattened in
        order, reproduce the original instruction stream exactly.
    """
    return make_cfg(
        name=function.name,
        globl=function.globl,
        params=function.params,
        body=function.body,
        source_file=function.source_file,
        source_line=function.source_line,
        adapter=TAC_ADAPTER,
    )


def cfg_to_instructions(cfg: ControlFlowGraph[tac.Instruction]) -> tac.Function:
    """Flatten a TAC control-flow graph back into a ``tac.Function``.

    Each block's instruction list is appended in order to produce the flat
    body of the returned function.

    Args:
        cfg: The TAC control-flow graph to flatten.

    Returns:
        A ``tac.Function`` whose body is the concatenation of all block
        instruction lists in order.
    """
    instructions: list[tac.Instruction] = []
    for block in cfg.blocks:
        instructions.extend(block.instructions)
    return tac.Function(
        name=cfg.name,
        globl=cfg.globl,
        params=cfg.params,  # type: ignore[arg-type]
        body=tuple(instructions),
        source_file=cfg.source_file,
        source_line=cfg.source_line,
    )


# ---------------------------------------------------------------------------
# Assembly CFG support
# ---------------------------------------------------------------------------

GraphNode: TypeAlias = str | GPRegisters | FPRegisters
"""A node in an interference graph: a pseudo-register name or a hardware register."""

# Caller-saved registers that are clobbered by any call instruction.
# SP, BP, R10, R11 are excluded (not in the allocatable sets).
_CALLER_SAVED_GP: frozenset[GPRegisters] = frozenset(
    {
        GPRegisters.AX,
        GPRegisters.CX,
        GPRegisters.DX,
        GPRegisters.DI,
        GPRegisters.SI,
        GPRegisters.R8,
        GPRegisters.R9,
    }
)
_CALLER_SAVED_FP: frozenset[FPRegisters] = ALLOCATABLE_FP

# Argument-passing registers (x86-64 System V ABI).
_GP_ARG_REGS: frozenset[GPRegisters] = frozenset(
    {GPRegisters.DI, GPRegisters.SI, GPRegisters.DX, GPRegisters.CX, GPRegisters.R8, GPRegisters.R9}
)
_FP_ARG_REGS: frozenset[FPRegisters] = frozenset(
    {
        FPRegisters.XMM0,
        FPRegisters.XMM1,
        FPRegisters.XMM2,
        FPRegisters.XMM3,
        FPRegisters.XMM4,
        FPRegisters.XMM5,
        FPRegisters.XMM6,
        FPRegisters.XMM7,
    }
)


def _dst_base_nodes(operand: asm.Operand) -> set[GraphNode]:
    """Return register nodes *used* when writing to a ``Memory`` or ``Indexed`` destination.

    A ``Memory`` or ``Indexed`` destination requires reading the base/index
    register(s) to compute the effective address before writing.  ``Pseudo`` and
    ``Register`` destinations do not; they are pure writes.

    Args:
        operand: The destination operand of a write instruction.

    Returns:
        The base/index register nodes read by the address calculation, or the
        empty set for non-memory destinations.
    """
    if isinstance(operand, Memory):
        return _use_nodes(operand.reg)
    if isinstance(operand, Indexed):
        return _use_nodes(operand.base) | _use_nodes(operand.index)
    return set()


def _def_nodes(operand: asm.Operand) -> set[GraphNode]:
    """Return the interference-graph node(s) *defined* by writing to ``operand``.

    Only direct register destinations count as definitions; a ``Memory``
    destination writes to memory through a register but does not redefine
    that register.

    Args:
        operand: The destination operand of an assembly instruction.

    Returns:
        A set containing the single pseudo name (for ``Pseudo``) or hardware
        register (for an allocatable ``Register``), or the empty set otherwise.
    """
    if isinstance(operand, Pseudo):
        return {operand.name}
    if isinstance(operand, Register) and (operand.reg in ALLOCATABLE_GP or operand.reg in ALLOCATABLE_FP):
        return {operand.reg}
    return set()


def _use_nodes(operand: asm.Operand) -> set[GraphNode]:
    """Return the interference-graph node(s) *used* by reading ``operand``.

    Unlike ``_def_nodes``, ``Memory`` and ``Indexed`` operands contribute the
    base/index register(s) as uses, since those registers must be live to
    perform the address calculation.

    Args:
        operand: The source operand of an assembly instruction.

    Returns:
        A set of pseudo names and/or hardware registers read by this operand.
    """
    if isinstance(operand, Pseudo):
        return {operand.name}
    if isinstance(operand, Register) and (operand.reg in ALLOCATABLE_GP or operand.reg in ALLOCATABLE_FP):
        return {operand.reg}
    if isinstance(operand, Memory):
        return _use_nodes(operand.reg)
    if isinstance(operand, Indexed):
        return _use_nodes(operand.base) | _use_nodes(operand.index)
    return set()


class AsmAdapter:
    """``CfgAdapter`` implementation for assembly instructions.

    Recognises ``asm.Label``, ``asm.Ret``, ``asm.Jmp``, and ``asm.JmpCC``
    for CFG construction, and additionally provides ``defs`` / ``uses``
    methods that return interference-graph nodes for liveness analysis.
    """

    def label_name(self, instr: asm.Instruction) -> str | None:
        """Return the label name when *instr* is an ``asm.Label``, else ``None``.

        Args:
            instr: The assembly instruction to inspect.

        Returns:
            The label name string, or ``None``.
        """
        return instr.name if isinstance(instr, asm.Label) else None

    def is_return(self, instr: asm.Instruction) -> bool:
        """Return ``True`` when *instr* is an ``asm.Ret``.

        Args:
            instr: The assembly instruction to inspect.

        Returns:
            ``True`` if *instr* is a ``Ret``.
        """
        return isinstance(instr, asm.Ret)

    def jump_target(self, instr: asm.Instruction) -> str | None:
        """Return the target label when *instr* is an unconditional ``asm.Jmp``, else ``None``.

        Args:
            instr: The assembly instruction to inspect.

        Returns:
            The ``label`` field of a ``Jmp``, or ``None``.
        """
        return instr.label if isinstance(instr, asm.Jmp) else None

    def conditional_jump_target(self, instr: asm.Instruction) -> str | None:
        """Return the target label when *instr* is an ``asm.JmpCC``, else ``None``.

        Args:
            instr: The assembly instruction to inspect.

        Returns:
            The ``label`` field of a ``JmpCC``, or ``None``.
        """
        return instr.label if isinstance(instr, asm.JmpCC) else None

    def defs(self, instr: asm.Instruction) -> set[GraphNode]:
        """Return the set of interference-graph nodes *defined* by ``instr``.

        Covers all instruction forms that write to a register or pseudo-register,
        including the implicit clobbers of ``Call``/``IndirectCall`` (caller-saved
        registers) and ``Idiv``/``Div``/``Cdq`` (AX/DX).

        Args:
            instr: The assembly instruction to analyse.

        Returns:
            A set of ``GraphNode`` values written by the instruction.
        """
        match instr:
            case asm.Mov(dst=dst):
                return _def_nodes(dst)
            case asm.Movsx(dst=dst) | asm.MoveZeroExtend(dst=dst):
                return _def_nodes(dst)
            case asm.Unary(operand=op):
                return _def_nodes(op)
            case asm.Binary(dst=dst):
                return _def_nodes(dst)
            case asm.SetCC(operand=op):
                return _def_nodes(op)
            case asm.Shl(operand=op) | asm.Sar(operand=op) | asm.Shr(operand=op):
                return _def_nodes(op)
            case asm.Idiv() | asm.Div():
                return {GPRegisters.AX, GPRegisters.DX}
            case asm.Cdq():
                return {GPRegisters.DX}
            case asm.Pop(target=reg):
                return _def_nodes(reg)
            case asm.Call() | asm.IndirectCall():
                gp_clobbers: set[GraphNode] = set(_CALLER_SAVED_GP)
                fp_clobbers: set[GraphNode] = set(_CALLER_SAVED_FP)
                return gp_clobbers | fp_clobbers
            case asm.Cvttsd2si(dst=dst) | asm.Cvtsi2sd(dst=dst):
                return _def_nodes(dst)
            case asm.Cvttss2si(dst=dst) | asm.Cvtsi2ss(dst=dst) | asm.Cvtss2sd(dst=dst) | asm.Cvtsd2ss(dst=dst):
                return _def_nodes(dst)
            case asm.Lea(dst=dst):
                return _def_nodes(dst)
            case _:
                return set()

    def uses(self, instr: asm.Instruction) -> set[GraphNode]:
        """Return the set of interference-graph nodes *used* by ``instr``.

        Covers all instruction forms that read from a register or pseudo-register,
        including implicit uses such as ``Cdq`` (reads AX) and shift instructions
        (read CX for the count).

        ``Ret`` returns an empty set here - the return-value register(s) that are
        live at exit are injected by ``_meet`` via ``exit_live``, not by treating
        ``Ret`` itself as a user of AX/XMM0.  This avoids spuriously propagating
        the conservative ``{AX, XMM0}`` set through all predecessors of every
        return block when the function's actual return type only needs one of them.

        Args:
            instr: The assembly instruction to analyse.

        Returns:
            A set of ``GraphNode`` values read by the instruction.
        """
        match instr:
            case asm.Mov(src=src, dst=dst):
                return _use_nodes(src) | _dst_base_nodes(dst)
            case asm.Movsx(src=src, dst=dst) | asm.MoveZeroExtend(src=src, dst=dst):
                return _use_nodes(src) | _dst_base_nodes(dst)
            case asm.Unary(operand=op):
                return _use_nodes(op)
            case asm.Binary(src1=s1, src2=s2, dst=dst):
                return _use_nodes(s1) | _use_nodes(s2) | _dst_base_nodes(dst)
            case asm.Cmp(op1=o1, op2=o2):
                return _use_nodes(o1) | _use_nodes(o2)
            case asm.Idiv(operand=op) | asm.Div(operand=op):
                return _use_nodes(op) | {GPRegisters.AX, GPRegisters.DX}
            case asm.Cdq():
                return {GPRegisters.AX}
            case asm.Shl(operand=op) | asm.Sar(operand=op) | asm.Shr(operand=op):
                return _use_nodes(op) | {GPRegisters.CX}
            case asm.Push(operand=op):
                return _use_nodes(op)
            case asm.Call():
                # Conservative: all GP and FP argument registers may carry arguments.
                # Without the callee's type, we cannot know which are actually used.
                return set(_GP_ARG_REGS) | set(_FP_ARG_REGS)
            case asm.IndirectCall(target=tgt):
                return _use_nodes(tgt) | set(_GP_ARG_REGS) | set(_FP_ARG_REGS)
            case asm.Cvttsd2si(src=src) | asm.Cvtsi2sd(src=src):
                return _use_nodes(src)
            case asm.Cvttss2si(src=src) | asm.Cvtsi2ss(src=src) | asm.Cvtss2sd(src=src) | asm.Cvtsd2ss(src=src):
                return _use_nodes(src)
            case asm.Lea(src=src):
                return _use_nodes(src)
            case _:
                return set()


ASM_ADAPTER: Final = AsmAdapter()
"""Singleton ``AsmAdapter`` for use with assembly-based CFGs."""


def make_asm_control_flow_graph(function: asm.Function) -> ControlFlowGraph[asm.Instruction]:
    """Decompose an assembly function into a control-flow graph of basic blocks.

    Convenience wrapper around ``make_cfg`` that uses ``ASM_ADAPTER`` and
    extracts the relevant fields from *function*.  Successor edges are wired
    up before returning.

    Args:
        function: The assembly function to decompose.

    Returns:
        A ``ControlFlowGraph[asm.Instruction]`` whose blocks, flattened in
        order, reproduce the original instruction stream exactly.
    """
    return make_cfg(
        name=function.name,
        globl=function.globl,
        params=function.params,
        body=function.instructions,
        source_file=function.source_file,
        source_line=function.source_line,
        adapter=ASM_ADAPTER,
    )
