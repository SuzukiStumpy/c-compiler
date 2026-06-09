"""Control-flow graph types and the TAC ↔ CFG conversion functions.

A ``ControlFlowGraph[I]`` decomposes the flat instruction stream of a function
into ``BasicBlock[I]`` objects parameterised by the instruction type ``I``.
The decomposition is lossless: ``cfg_to_instructions`` always reconstructs the
original TAC function exactly.

Instruction-set-specific knowledge (how to detect labels, returns, and jumps)
is isolated behind the ``CfgAdapter[I]`` protocol.  ``TacAdapter`` implements
it for TAC instructions; a future ``AsmAdapter`` would implement it for the
assembly AST without touching this module.

Each ``BasicBlock`` carries a ``successors`` list populated by ``add_all_edges``,
which is called automatically by ``make_cfg`` and ``make_control_flow_graph``.
Passes that mutate the block list should call ``add_all_edges`` again before
relying on successor data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Final, Generic, Protocol, TypeAlias, TypeVar

import compiler.syntax.tac as tac

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
    x86 assembly, …) so that the generic ``make_cfg`` and ``add_all_edges``
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

    A new basic block begins at every instruction identified as a label by
    *adapter* and at the implicit start of the body.  The preceding block is
    closed before the new one is opened.  The label instruction is kept as the
    first instruction of its block so that the original stream can be
    reconstructed exactly.

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

    * Return → ``EXIT``
    * Unconditional jump → the block whose leading label matches the target
    * Conditional jump → both the target block and the fall-through successor
      (the next block in order, or ``EXIT`` for the last block)
    * Any other instruction → fall-through successor only

    An implicit ``ENTRY → blocks[0]`` edge is not stored; callers may treat
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
            block.successors.append(label_index[target])
        elif (target := adapter.conditional_jump_target(last)) is not None:
            block.successors.append(label_index[target])
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
