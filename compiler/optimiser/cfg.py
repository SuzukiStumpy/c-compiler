"""Control-flow graph types and the TAC ↔ CFG conversion functions.

A ``ControlFlowGraph`` decomposes the flat instruction stream of a ``tac.Function``
into ``BasicBlock`` objects.  The decomposition is lossless: ``cfg_to_instructions``
always reconstructs the original function exactly.
"""

from dataclasses import dataclass

import compiler.syntax.tac as tac


@dataclass
class BasicBlock:
    """A maximal, linear sequence of instructions with a single entry and exit.

    A block starts at the beginning of a function body or at a ``Label``
    instruction.  It ends immediately before the next ``Label``, at a
    terminator (``Return``, ``Jump``, ``JumpIfZero``, ``JumpIfNotZero``),
    or at the end of the function body.

    The leading ``Label`` instruction (if any) is stored as the first element
    of ``instructions`` so that ``cfg_to_instructions`` can reconstruct the
    original stream exactly without any special-casing.

    Attributes:
        label: The name of the leading ``Label`` instruction, or ``None`` for
            the implicit entry block that precedes the first label.
        instructions: Ordered sequence of instructions in this block, including
            the leading ``Label`` (if any) and the trailing terminator (if any).
    """

    label: str | None
    instructions: list[tac.Instruction]


@dataclass
class ControlFlowGraph:
    """A control-flow graph (CFG) derived from a single TAC function.

    The CFG decomposes a flat instruction stream into ``BasicBlock`` objects.
    Block order is preserved so that ``cfg_to_instructions`` faithfully
    reconstructs the original function.

    Attributes:
        name: The function name, carried through for reconstruction.
        globl: Whether the function bears the ``.globl`` directive.
        params: The ordered sequence of parameter names.
        blocks: Ordered list of basic blocks.
        source_file: Source file path propagated from the original ``Function``.
        source_line: 1-indexed source line propagated from the original ``Function``.
    """

    name: str
    globl: bool
    params: tuple[str, ...]
    blocks: list[BasicBlock]
    source_file: str
    source_line: int


def make_control_flow_graph(function: tac.Function) -> ControlFlowGraph:
    """Decompose a TAC function into a control-flow graph of basic blocks.

    A new basic block begins at every ``Label`` instruction and at the implicit
    start of the function body.  The preceding block is closed before the new
    one is opened.  The ``Label`` instruction is kept as the first instruction
    of its block so that the original stream can be reconstructed exactly by
    ``cfg_to_instructions``.

    Args:
        function: The TAC function to decompose.

    Returns:
        A ``ControlFlowGraph`` whose blocks, flattened in order, reproduce the
        original instruction stream exactly.
    """
    blocks: list[BasicBlock] = []
    current_label: str | None = None
    current_instrs: list[tac.Instruction] = []

    for instr in function.body:
        if isinstance(instr, tac.Label):
            if current_instrs:
                blocks.append(BasicBlock(label=current_label, instructions=current_instrs))
            current_label = instr.name
            current_instrs = [instr]
        else:
            current_instrs.append(instr)

    if current_instrs:
        blocks.append(BasicBlock(label=current_label, instructions=current_instrs))

    return ControlFlowGraph(
        name=function.name,
        globl=function.globl,
        params=function.params,
        blocks=blocks,
        source_file=function.source_file,
        source_line=function.source_line,
    )


def cfg_to_instructions(cfg: ControlFlowGraph) -> tac.Function:
    """Flatten a control-flow graph back into a TAC function.

    Each block's instruction list is appended in order to produce the flat
    body of the returned function.

    Args:
        cfg: The control-flow graph to flatten.

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
        params=cfg.params,
        body=tuple(instructions),
        source_file=cfg.source_file,
        source_line=cfg.source_line,
    )
