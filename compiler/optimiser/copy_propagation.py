"""Copy-propagation optimization pass."""

from collections import deque
from typing import TypeAlias

import compiler.syntax.tac as tac
from compiler.syntax.ast.constants import ConstChar, ConstDouble, ConstInt, ConstLong, ConstUChar, ConstUInt, ConstULong
from compiler.syntax.ast.types import (
    Char,
    DoubleType,
    IntType,
    LongType,
    Pointer,
    SChar,
    Type,
    UChar,
    UIntType,
    ULongType,
)
from compiler.types.symbols import StaticAttr, SymbolTable

from .cfg import ENTRY, BasicBlock, BlockId, BoundaryNode, ControlFlowGraph, NodeId

CopySet: TypeAlias = frozenset[tac.Copy]
"""The set of TAC Copy instructions whose values reach a specific program point."""


def _val_type(val: tac.Val, symbol_table: SymbolTable) -> Type | None:
    """Return the C type of *val*, or ``None`` when it cannot be determined.

    Args:
        val: The TAC value to classify.
        symbol_table: The program symbol table used to look up variable types.

    Returns:
        The ``Type`` of *val*, or ``None`` when the type cannot be determined
        (e.g. a variable absent from the symbol table).
    """
    match val:
        case tac.Constant(value=ConstInt()):
            return IntType()
        case tac.Constant(value=ConstLong()):
            return LongType()
        case tac.Constant(value=ConstUInt()):
            return UIntType()
        case tac.Constant(value=ConstULong()):
            return ULongType()
        case tac.Constant(value=ConstDouble()):
            return DoubleType()
        case tac.Constant(value=ConstChar()):
            return Char()
        case tac.Constant(value=ConstUChar()):
            return UChar()
        case tac.Var(value=name):
            entry = symbol_table.get(name)
            return entry.t if entry else None
        case _:
            return None


def _signedness(t: Type) -> bool | None:
    """Return ``True`` for signed integer types, ``False`` for unsigned, ``None`` otherwise.

    Args:
        t: The C type to classify.

    Returns:
        ``True`` for ``IntType``, ``LongType``, ``Char``, ``SChar``;
        ``False`` for ``UIntType``, ``ULongType``, ``UChar``; ``None`` for all
        other types (``DoubleType``, ``Pointer``, etc.).
    """
    match t:
        case IntType() | LongType() | Char() | SChar():
            return True
        case UIntType() | ULongType() | UChar():
            return False
        case _:
            return None


def _types_compatible(src: tac.Val, dst: tac.Val, symbol_table: SymbolTable) -> bool:
    """Return ``True`` if *src* and *dst* have types compatible for copy tracking.

    A copy is safe to track when both operands have the exact same type, or when
    both have the same signedness (both signed integers or both unsigned integers).
    When either type cannot be determined the copy is assumed compatible so that
    callers without a full symbol table are not penalised.

    Args:
        src: The copy source operand.
        dst: The copy destination operand.
        symbol_table: The program symbol table used to look up variable types.

    Returns:
        ``True`` when the types are compatible for copy propagation.
    """
    src_t = _val_type(src, symbol_table)
    dst_t = _val_type(dst, symbol_table)
    if src_t is None or dst_t is None:
        return True
    if src_t == dst_t:
        return True
    # Pointer types share the same 64-bit machine representation and may be
    # copied between differently-typed pointer variables.  Long/ULong may also
    # be copied to/from a Pointer (null-pointer constant propagation), but we
    # must NOT allow Long ↔ ULong here: their signedness differs, and the code
    # generator selects signed vs unsigned comparison instructions based on the
    # type recorded in the symbol table.  Propagating a Long source into a slot
    # typed ULong (or vice-versa) would silently change the comparison operator.
    _ptr_or_64 = (Pointer, LongType, ULongType)
    if isinstance(src_t, _ptr_or_64) and isinstance(dst_t, _ptr_or_64):
        if isinstance(src_t, Pointer) or isinstance(dst_t, Pointer):
            return True
    src_s = _signedness(src_t)
    dst_s = _signedness(dst_t)
    return src_s is not None and src_s == dst_s


def _find_aliased_vars(
    cfg: ControlFlowGraph[tac.Instruction],
    symbol_table: SymbolTable,
) -> frozenset[str]:
    """Return the set of variable names that may be accessed through pointers.

    Includes variables explicitly aliased by a ``GetAddress`` instruction and
    all static-duration variables (which are globally accessible and may be
    modified through any pointer the compiler cannot track).

    Args:
        cfg: The control-flow graph to scan for ``GetAddress`` instructions.
        symbol_table: The program symbol table used to identify static variables.

    Returns:
        A ``frozenset`` of variable name strings for all potentially aliased
        variables.
    """
    address_taken: frozenset[str] = frozenset(
        instr.src.value
        for block in cfg.blocks
        for instr in block.instructions
        if isinstance(instr, tac.GetAddress) and isinstance(instr.src, tac.Var)
    )
    statics: frozenset[str] = frozenset(
        name for name, entry in symbol_table.items() if isinstance(entry.attrs, StaticAttr)
    )
    return address_taken | statics


def _collect_all_copies(cfg: ControlFlowGraph[tac.Instruction]) -> CopySet:
    """Return the set of every ``Copy`` instruction that appears in the CFG.

    This is the *top* element of the reaching-copies lattice: the largest set
    that can ever be the initial value before any intersection is applied.

    Args:
        cfg: The control-flow graph to scan.

    Returns:
        A ``CopySet`` containing every distinct ``tac.Copy`` instruction found
        in any block of *cfg*.
    """
    return frozenset(instr for block in cfg.blocks for instr in block.instructions if isinstance(instr, tac.Copy))


def _build_predecessor_map(cfg: ControlFlowGraph[tac.Instruction]) -> dict[int, list[NodeId]]:
    """Build an inverted edge map from block index to predecessor ``NodeId`` list.

    ``blocks[0]`` always receives ``ENTRY`` as a predecessor.  For every other
    block, predecessors are derived by inverting the ``successors`` lists that
    ``add_all_edges`` has already populated.

    Args:
        cfg: The control-flow graph whose successor edges are already wired.

    Returns:
        A dict mapping each block index to its list of predecessor ``NodeId``
        values (``BlockId`` or ``BoundaryNode.ENTRY``).
    """
    pred_map: dict[int, list[NodeId]] = {i: [] for i in range(len(cfg.blocks))}

    if cfg.blocks:
        pred_map[0].append(ENTRY)

    for i, block in enumerate(cfg.blocks):
        for succ in block.successors:
            if isinstance(succ, BlockId):
                pred_map[succ.num].append(BlockId(i))

    return pred_map


def meet(
    predecessors: list[NodeId],
    block_out: dict[int, CopySet],
    all_copies: CopySet,
) -> CopySet:
    """Compute the meet (intersection) of all predecessor exit reaching-copy sets.

    This is the standard forward-dataflow meet for copy propagation: a copy
    reaches the entry of a block only when it reaches the exit of *every*
    predecessor.

    The *top* element ``all_copies`` is used as the identity for intersection:
    starting with all copies and intersecting with each predecessor's exit set
    converges to the correct answer for every block that has at least one
    listed predecessor.

    When ``predecessors`` is empty the function returns ``frozenset()`` (bottom).
    This is the conservative safe choice: a block with no listed predecessors is
    either unreachable (UCE should have pruned it) or reachable only via a
    mid-block conditional jump invisible to ``add_all_edges``, which only
    inspects the last instruction of each block.  Using ``all_copies`` (top)
    instead would make every copy appear to reach such a block, causing
    downstream rewriting to falsely eliminate non-redundant ``Copy``
    instructions.

    Args:
        predecessors: Predecessor ``NodeId`` list for the block being analysed,
            as produced by ``_build_predecessor_map``.
        block_out: Map from block index to the exit ``CopySet`` computed by
            ``transfer`` for that block in the current iteration.
        all_copies: The top element of the lattice — the set of every ``Copy``
            instruction in the function.

    Returns:
        The ``CopySet`` that reaches the entry of the block: the intersection
        of all predecessor exit sets.

    Raises:
        ValueError: If ``EXIT`` appears as a predecessor, indicating a
            malformed control-flow graph.
    """
    # A block with no listed predecessors is either unreachable (UCE missed it)
    # or reachable only via a mid-block conditional jump whose target is invisible
    # to add_all_edges (which only inspects the last instruction).  In both cases
    # the conservative safe choice is the empty set rather than all_copies.
    if not predecessors:
        return frozenset()

    incoming: CopySet = all_copies

    for pred_id in predecessors:
        match pred_id:
            case BoundaryNode.ENTRY:
                return frozenset()
            case BlockId(num=num):
                incoming = incoming & block_out[num]
            case BoundaryNode.EXIT:
                raise ValueError("Malformed control-flow graph: EXIT appears as a predecessor")

    return incoming


def transfer(
    block: BasicBlock[tac.Instruction],
    initial: CopySet,
    symbol_table: SymbolTable,
    aliased_vars: frozenset[str],
) -> tuple[CopySet, list[CopySet]]:
    """Compute the transfer function for one basic block.

    Walks ``block.instructions`` in order, maintaining the set of copies that
    still reach the current program point.  At each step the *pre*-instruction
    reaching set is recorded as an annotation so the rewriting pass can later
    substitute operands.

    The function models each instruction class as follows:

    * ``Copy(src, dst)`` — if the reverse copy ``Copy(src=dst, dst=src)``
      already reaches, the instruction is a no-op (src and dst are already
      known equal).  Otherwise all copies that mention ``dst`` are killed; the
      new copy is added only when *src* and *dst* have compatible types (same
      type, or same signedness).
    * ``FunCall(dst=dst)`` — kills every copy whose ``src`` or ``dst`` is an
      aliased variable (address-taken or static, either of which may be
      clobbered by the callee) and every copy that mentions the return-value
      destination ``dst`` (if any).
    * ``Store`` — kills every copy whose ``src`` or ``dst`` is an aliased
      variable, since a store through a pointer may overwrite any aliased
      location.
    * ``Unary``, ``Binary``, and all type-conversion / addressing instructions
      that write to a ``dst: Val`` field — kill every copy that mentions
      ``dst``.
    * Everything else (``Label``, ``Jump``, ``JumpIfZero``, ``JumpIfNotZero``,
      ``Return``, ``CopyToOffset``) — leaves the reaching set unchanged.

    Args:
        block: The basic block to analyse.
        initial: Reaching copies on entry to the block, as computed by the
            meet function for all predecessor exits.
        symbol_table: The program symbol table, used to determine operand types
            and to look up static variables via ``aliased_vars``.
        aliased_vars: Set of variable names that may be accessed through
            pointers (address-taken variables and static-duration variables).

    Returns:
        A 2-tuple ``(exit_copies, instr_annotations)`` where *exit_copies* is
        the ``CopySet`` reaching the end of the block and *instr_annotations*
        is a list — one ``CopySet`` per instruction in ``block.instructions``
        — recording the reaching copies *before* that instruction executes.
    """
    current: set[tac.Copy] = set(initial)
    instr_annotations: list[CopySet] = []

    for instr in block.instructions:
        instr_annotations.append(frozenset(current))

        match instr:
            case tac.Copy(src=src, dst=dst):
                # If the reverse copy already reaches, src == dst is already
                # known — this instruction changes nothing; skip kill/add.
                if tac.Copy(src=dst, dst=src) in current:
                    continue
                # Kill all copies that mention dst (it is being overwritten).
                current -= {c for c in current if c.src == dst or c.dst == dst}
                if _types_compatible(src, dst, symbol_table):
                    current.add(instr)

            case tac.FunCall(dst=dst):
                # A call may clobber any aliased variable and the return dst.
                current = {
                    c
                    for c in current
                    if not (isinstance(c.src, tac.Var) and c.src.value in aliased_vars)
                    and not (isinstance(c.dst, tac.Var) and c.dst.value in aliased_vars)
                    and (dst is None or (c.src != dst and c.dst != dst))
                }

            case tac.Store():
                # A store through a pointer may overwrite any aliased variable.
                current = {
                    c
                    for c in current
                    if not (isinstance(c.src, tac.Var) and c.src.value in aliased_vars)
                    and not (isinstance(c.dst, tac.Var) and c.dst.value in aliased_vars)
                }

            case (
                tac.Unary(dst=dst)
                | tac.Binary(dst=dst)
                | tac.SignExtend(dst=dst)
                | tac.Truncate(dst=dst)
                | tac.ZeroExtend(dst=dst)
                | tac.DoubleToInt(dst=dst)
                | tac.DoubleToUInt(dst=dst)
                | tac.IntToDouble(dst=dst)
                | tac.UIntToDouble(dst=dst)
                | tac.GetAddress(dst=dst)
                | tac.Load(dst=dst)
                | tac.AddPtr(dst=dst)
                | tac.CopyFromOffset(dst=dst)
            ):
                current -= {c for c in current if c.src == dst or c.dst == dst}

            case _:
                pass

    return frozenset(current), instr_annotations


def _replace_operand(op: tac.Val, reaching_copies: CopySet) -> tac.Val:
    """Return the canonical source for *op* given the current reaching copies.

    Chases the copy chain transitively: if ``Copy(b, a)`` and ``Copy(c, b)``
    both reach the current point, ``a`` is replaced by ``c`` in one call.
    The chain is followed until a ``Constant``, a variable with no reaching
    copy, or a cycle is detected.

    Args:
        op: The operand to (potentially) replace.
        reaching_copies: Copies that reach the current program point.

    Returns:
        The deepest reachable ``Val`` in the copy chain, or *op* unchanged
        when no reaching copy has *op* as its destination.
    """
    if isinstance(op, tac.Constant):
        return op
    lookup: dict[tac.Val, tac.Val] = {c.dst: c.src for c in reaching_copies}
    current: tac.Val = op
    seen: set[tac.Val] = set()
    while isinstance(current, tac.Var) and current not in seen:
        seen.add(current)
        nxt = lookup.get(current)
        if nxt is None:
            break
        current = nxt
    return current


def _rewrite_secondary(instr: tac.Instruction, r: CopySet) -> tac.Instruction | None:
    """Rewrite type-cast and memory instructions; return ``None`` if unrecognised.

    Handles ``SignExtend``, ``Truncate``, ``ZeroExtend``, ``DoubleToInt``,
    ``DoubleToUInt``, ``IntToDouble``, ``UIntToDouble``, ``Load``, ``Store``,
    ``AddPtr``, and ``CopyToOffset``.  All other instruction types cause
    ``None`` to be returned, signalling to the caller that the instruction
    should be returned unchanged.

    Args:
        instr: The instruction to attempt to rewrite.
        r: Reaching-copy set at this program point.

    Returns:
        A rewritten instruction, or ``None`` if *instr* is not handled here.
    """
    match instr:
        case tac.SignExtend(src=src, dst=dst):
            return tac.SignExtend(src=_replace_operand(src, r), dst=dst)
        case tac.Truncate(src=src, dst=dst):
            return tac.Truncate(src=_replace_operand(src, r), dst=dst)
        case tac.ZeroExtend(src=src, dst=dst):
            return tac.ZeroExtend(src=_replace_operand(src, r), dst=dst)
        case tac.DoubleToInt(src=src, dst=dst):
            return tac.DoubleToInt(src=_replace_operand(src, r), dst=dst)
        case tac.DoubleToUInt(src=src, dst=dst):
            return tac.DoubleToUInt(src=_replace_operand(src, r), dst=dst)
        case tac.IntToDouble(src=src, dst=dst):
            return tac.IntToDouble(src=_replace_operand(src, r), dst=dst)
        case tac.UIntToDouble(src=src, dst=dst):
            return tac.UIntToDouble(src=_replace_operand(src, r), dst=dst)
        case tac.Load(src_ptr=src_ptr, dst=dst):
            return tac.Load(src_ptr=_replace_operand(src_ptr, r), dst=dst)
        case tac.Store(src=src, dst_ptr=dst_ptr):
            return tac.Store(
                src=_replace_operand(src, r),
                dst_ptr=_replace_operand(dst_ptr, r),
            )
        case tac.AddPtr(ptr=ptr, index=index, scale=scale, dst=dst):
            return tac.AddPtr(
                ptr=_replace_operand(ptr, r),
                index=_replace_operand(index, r),
                scale=scale,
                dst=dst,
            )
        case tac.CopyToOffset(src=src, dst=dst, offset=offset):
            return tac.CopyToOffset(src=_replace_operand(src, r), dst=dst, offset=offset)
        case _:
            return None


def _rewrite_instruction(
    instr: tac.Instruction,
    reaching_copies: CopySet,
) -> tac.Instruction | None:
    """Rewrite one instruction by substituting operands from reaching copies.

    Every ``Val`` operand that is the destination of a reaching copy is
    replaced with that copy's source.  A ``Copy`` instruction that is
    already redundant — either because it is itself a member of the reaching
    set, or because the reverse copy is reaching (operands are already known
    equal) — is eliminated by returning ``None``.

    ``GetAddress`` is deliberately excluded from operand replacement: copy
    propagation establishes value equality but not address equality, so
    substituting the source variable would yield a different memory address.

    Secondary instruction types (type-cast and memory instructions) are
    delegated to ``_rewrite_secondary``.

    Args:
        instr: The instruction to rewrite.
        reaching_copies: The reaching-copy set immediately *before* this
            instruction, as annotated by ``transfer``.

    Returns:
        A rewritten ``tac.Instruction``, or ``None`` if the instruction is
        redundant and should be deleted.
    """
    r = reaching_copies

    match instr:
        case tac.Copy(src=src, dst=dst):
            for copy in r:
                if copy == instr or (copy.src == dst and copy.dst == src):
                    return None
            return tac.Copy(src=_replace_operand(src, r), dst=dst)

        case tac.Unary(operator=op, src=src, dst=dst):
            return tac.Unary(operator=op, src=_replace_operand(src, r), dst=dst)

        case tac.Binary(operator=op, src1=src1, src2=src2, dst=dst):
            return tac.Binary(
                operator=op,
                src1=_replace_operand(src1, r),
                src2=_replace_operand(src2, r),
                dst=dst,
            )

        case tac.Return(val=val):
            if val is None:
                return instr
            return tac.Return(val=_replace_operand(val, r))

        case tac.JumpIfZero(condition=cond, target=target):
            return tac.JumpIfZero(condition=_replace_operand(cond, r), target=target)

        case tac.JumpIfNotZero(condition=cond, target=target):
            return tac.JumpIfNotZero(condition=_replace_operand(cond, r), target=target)

        case tac.FunCall(fun_val=fun_val, args=args, dst=dst):
            return tac.FunCall(
                fun_val=_replace_operand(fun_val, r),
                args=tuple(_replace_operand(a, r) for a in args),
                dst=dst,
            )

        case _:
            # Delegate type-cast and memory instructions to _rewrite_secondary.
            # GetAddress.src is intentionally not substituted: address identity
            # differs from value equality. CopyFromOffset.src is a str. Label
            # and Jump have no Val operands.
            result = _rewrite_secondary(instr, r)
            return result if result is not None else instr


def _find_reaching_copies(
    cfg: ControlFlowGraph[tac.Instruction],
    symbol_table: SymbolTable,
) -> tuple[dict[int, CopySet], dict[int, list[CopySet]]]:
    """Run the reaching-copies forward dataflow analysis over the CFG.

    Initialises every block's exit set to ``all_copies`` (the lattice top),
    then iterates a worklist until the exit sets stabilise.  On each iteration
    the meet function computes the entry set from predecessor exits, and the
    transfer function propagates it through the block to produce a new exit set.
    When the exit set changes, the block's successors are re-queued.

    Args:
        cfg: The control-flow graph to analyse.  Successor edges must already
            be wired (i.e. ``add_all_edges`` must have been called).
        symbol_table: The program symbol table, forwarded to ``transfer`` for
            static-variable kill decisions.

    Returns:
        A 2-tuple ``(block_out, instr_annotations)`` where *block_out* maps
        each block index to its converged exit ``CopySet`` and
        *instr_annotations* maps each block index to the list of per-instruction
        pre-state ``CopySet`` values produced by the final ``transfer`` call
        for that block.
    """
    all_copies = _collect_all_copies(cfg)
    pred_map = _build_predecessor_map(cfg)
    aliased_vars = _find_aliased_vars(cfg, symbol_table)

    block_out: dict[int, CopySet] = dict.fromkeys(range(len(cfg.blocks)), all_copies)
    instr_annotations: dict[int, list[CopySet]] = {}

    worklist: deque[int] = deque(range(len(cfg.blocks)))
    in_worklist: set[int] = set(range(len(cfg.blocks)))

    while worklist:
        block_idx = worklist.popleft()
        in_worklist.discard(block_idx)

        block = cfg.blocks[block_idx]
        old_out = block_out[block_idx]

        incoming = meet(pred_map[block_idx], block_out, all_copies)
        new_out, annots = transfer(block, incoming, symbol_table, aliased_vars)

        block_out[block_idx] = new_out
        instr_annotations[block_idx] = annots

        if new_out != old_out:
            for succ_id in block.successors:
                match succ_id:
                    case BoundaryNode.EXIT:
                        continue
                    case BoundaryNode.ENTRY:
                        raise ValueError("Malformed control-flow graph: ENTRY appears as a successor")
                    case BlockId(num=num):
                        if num not in in_worklist:
                            worklist.append(num)
                            in_worklist.add(num)

    return block_out, instr_annotations


def copy_propagation(
    cfg: ControlFlowGraph[tac.Instruction],
    symbol_table: SymbolTable | None = None,
) -> ControlFlowGraph[tac.Instruction]:
    """Replace uses of copy destinations with their source values where safe.

    Args:
        cfg: The control-flow graph to optimize.
        symbol_table: The program symbol table.  Required for correct
            ``FunCall`` kill behaviour; pass ``None`` only in tests that do
            not exercise calls to functions.

    Returns:
        A new ``ControlFlowGraph`` with copy sources propagated forward.
    """
    if not cfg.blocks:
        return cfg

    _, instr_annotations = _find_reaching_copies(cfg, symbol_table or {})

    for block_idx, block in enumerate(cfg.blocks):
        new_instrs: list[tac.Instruction] = []
        for instr, reaching in zip(block.instructions, instr_annotations.get(block_idx, [])):
            rewritten = _rewrite_instruction(instr, reaching)
            if rewritten is not None:
                new_instrs.append(rewritten)
        block.instructions = new_instrs

    return cfg
