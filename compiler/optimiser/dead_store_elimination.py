"""Dead-store elimination optimization pass."""

from typing import TypeAlias

import compiler.syntax.tac as tac
from compiler.types.symbols import StaticAttr, SymbolTable

from .cfg import BasicBlock, BlockId, BoundaryNode, ControlFlowGraph
from .worklist import run_backward

LiveSet: TypeAlias = frozenset[str]
"""Set of variable names that are live at a specific program point."""


def _var_name(val: tac.Val | None) -> str | None:
    """Return the variable name if *val* is a ``Var``, else ``None``.

    Args:
        val: A TAC value, or ``None``.

    Returns:
        The variable name string, or ``None`` when *val* is a constant or ``None``.
    """
    return val.value if isinstance(val, tac.Var) else None


def _kill(current: set[str], dst: tac.Val) -> None:
    """Remove *dst* from *current* if it is a variable.

    Args:
        current: The mutable live-variable set to update.
        dst: The destination operand to kill.
    """
    if (name := _var_name(dst)) is not None:
        current.discard(name)


def _use(current: set[str], *operands: tac.Val | None) -> None:
    """Add each ``Var`` in *operands* to *current*.

    Args:
        current: The mutable live-variable set to update.
        *operands: Zero or more TAC values; non-``Var`` values and ``None`` are skipped.
    """
    for op in operands:
        if (name := _var_name(op)) is not None:
            current.add(name)


def _update_live(
    instr: tac.Instruction,
    current: set[str],
    aliased_vars: frozenset[str],
) -> None:
    """Update *current* live-variable set in place for one instruction.

    This is the per-instruction kernel of the backward liveness analysis: given
    the live set immediately *after* *instr*, it produces the live set immediately
    *before* *instr*.

    Args:
        instr: The instruction whose liveness effect is to be applied.
        current: Mutable live-variable set, updated in place.
        aliased_vars: Variables whose addresses have been taken or that have
            static storage duration.  Added to live at ``FunCall`` sites (a
            callee may read any aliased variable through a pointer it received)
            and at ``Load`` sites (a load through a pointer may read any aliased
            variable).
    """
    match instr:
        case tac.Return(val=val):
            _use(current, val)

        case tac.Unary(src=src, dst=dst):
            _kill(current, dst)
            _use(current, src)

        case tac.Binary(src1=src1, src2=src2, dst=dst):
            _kill(current, dst)
            _use(current, src1, src2)

        case tac.Copy(src=src, dst=dst):
            _kill(current, dst)
            _use(current, src)

        case tac.JumpIfZero(condition=cond) | tac.JumpIfNotZero(condition=cond):
            _use(current, cond)

        case tac.FunCall(fun_val=fun_val, args=args, dst=dst):
            if dst is not None:
                _kill(current, dst)
            for arg in args:
                _use(current, arg)
            # fun_val may be a local variable when making an indirect call through
            # a function pointer.
            _use(current, fun_val)
            current.update(aliased_vars)

        case (
            tac.SignExtend(src=src, dst=dst)
            | tac.Truncate(src=src, dst=dst)
            | tac.ZeroExtend(src=src, dst=dst)
            | tac.DoubleToInt(src=src, dst=dst)
            | tac.DoubleToUInt(src=src, dst=dst)
            | tac.IntToDouble(src=src, dst=dst)
            | tac.UIntToDouble(src=src, dst=dst)
        ):
            _kill(current, dst)
            _use(current, src)

        case tac.GetAddress(src=src, dst=dst):
            # GetAddress reads the *address* of src, not its value.  Adding src
            # to live here would prevent elimination of dead initializations
            # before a GetAddress whose pointer is never dereferenced.
            _kill(current, dst)

        case tac.Load(src_ptr=src_ptr, dst=dst):
            # A load through a pointer may read any aliased variable, so add
            # them all conservatively.
            _kill(current, dst)
            _use(current, src_ptr)
            current.update(aliased_vars)

        case tac.Store(src=src, dst_ptr=dst_ptr):
            _use(current, src, dst_ptr)

        case tac.AddPtr(ptr=ptr, index=index, dst=dst):
            _kill(current, dst)
            _use(current, ptr, index)

        case tac.CopyToOffset(src=src, dst=_dst):
            # CopyToOffset writes *into* an aggregate - it does not read the
            # aggregate's value, so the aggregate name is not added to live.
            # DSE may later eliminate this instruction if the aggregate is dead.
            _use(current, src)

        case tac.CopyFromOffset(src=src, dst=dst):
            _kill(current, dst)
            current.add(src)

        case _:
            # Jump, Label - no operands that affect liveness.
            pass


def transfer(
    block: BasicBlock[tac.Instruction],
    end_live: LiveSet,
    aliased_vars: frozenset[str],
) -> tuple[LiveSet, list[LiveSet]]:
    """Compute the liveness transfer function for one basic block (backward analysis).

    Walks ``block.instructions`` in reverse, maintaining the set of variable names
    that are live at the current program point.  Before processing each instruction,
    the current live set (= the set live *after* that instruction) is recorded as
    an annotation so the dead-store elimination pass can decide whether a write is
    dead.

    Args:
        block: The basic block to analyse.
        end_live: Variable names live on exit from the block, as computed by the
            meet function over all successor entry sets.
        aliased_vars: Variables whose addresses have been taken or that have
            static storage duration, forwarded to ``_update_live``.

    Returns:
        A 2-tuple ``(entry_live, instr_annotations)`` where *entry_live* is the
        ``LiveSet`` on entry to the block and *instr_annotations* is a list -
        one ``LiveSet`` per instruction in ``block.instructions`` - recording the
        live set *after* that instruction executes (i.e. the annotation captured
        before the backward update for that instruction).
    """
    current: set[str] = set(end_live)
    reversed_annotations: list[LiveSet] = []

    for instr in reversed(block.instructions):
        reversed_annotations.append(frozenset(current))
        _update_live(instr, current, aliased_vars)

    reversed_annotations.reverse()
    return frozenset(current), reversed_annotations


def meet(
    block: BasicBlock[tac.Instruction],
    all_static_variables: frozenset[str],
    block_in: dict[int, LiveSet],
) -> LiveSet:
    """Compute the meet (union) of all successor entry live-variable sets.

    Liveness is a backward analysis: a variable is live at the exit of a block
    when it is live at the entry of *at least one* successor.  The meet operator
    is therefore set union.

    ``EXIT`` is treated conservatively: any static-duration variable may be
    observed by the caller after the function returns, so all statics are added
    to the live set.  ``ENTRY`` appearing as a successor indicates a malformed
    graph and raises ``ValueError``.

    Args:
        block: The basic block whose exit live set is being computed.
        all_static_variables: Names of all static-duration variables in the
            function.  Added to the live set when ``EXIT`` is a successor.
        block_in: Map from block index to that block's entry ``LiveSet``, as
            produced by ``transfer`` for each block during the current iteration.

    Returns:
        The union of all successor entry live sets.

    Raises:
        ValueError: If ``ENTRY`` appears as a successor, indicating a malformed
            control-flow graph.
    """
    live: set[str] = set()

    for succ_id in block.successors:
        match succ_id:
            case BoundaryNode.EXIT:
                live.update(all_static_variables)
            case BoundaryNode.ENTRY:
                raise ValueError("Malformed control-flow graph: ENTRY appears as a successor")
            case BlockId(num=num):
                live.update(block_in[num])

    return frozenset(live)


def is_dead_store(instr: tac.Instruction, annotation: LiveSet) -> bool:
    """Return ``True`` when *instr* writes a result that is never subsequently read.

    A store is dead when the following conditions hold:

    1. The instruction is not a ``FunCall`` (calls may have arbitrary side effects
       regardless of whether their return value is used).
    2. For ``CopyToOffset``: the aggregate name is absent from *annotation*.
       Liveness conservatively adds aggregate names via ``CopyFromOffset`` and
       ``Load`` (through aliased-var tracking), so this is safe to check directly.
    3. For all other instructions: the instruction has a ``dst`` field that is a
       ``tac.Var``, and that variable is absent from *annotation*.  Instructions
       without a TAC variable destination - ``Store``, jumps, labels, ``Return``
       - are never dead stores.

    Aliased variables (address-taken or static-duration) are *not* exempted here;
    the liveness analysis itself is made conservative for aliased vars by adding
    them to the live set at ``Load`` and ``FunCall`` sites.

    Args:
        instr: The TAC instruction to test.
        annotation: The ``LiveSet`` immediately *after* this instruction, as
            produced by the liveness transfer function.

    Returns:
        ``True`` when the instruction's result is unused and the instruction
        may safely be deleted.
    """
    if isinstance(instr, tac.FunCall):
        return False
    if isinstance(instr, tac.CopyToOffset):
        return instr.dst not in annotation
    dst = getattr(instr, "dst", None)
    if isinstance(dst, tac.Var):
        return dst.value not in annotation
    return False


def _collect_all_static_variables(symbol_table: SymbolTable) -> frozenset[str]:
    """Return the names of all static-duration variables in *symbol_table*.

    Static variables may be read by any called function, so they are
    conservatively treated as live at every call site.

    Args:
        symbol_table: The program symbol table to scan.

    Returns:
        A ``frozenset`` of variable name strings for every symbol whose
        storage duration is static.
    """
    return frozenset(name for name, entry in symbol_table.items() if isinstance(entry.attrs, StaticAttr))


def _find_aliased_vars(
    cfg: ControlFlowGraph[tac.Instruction],
    symbol_table: SymbolTable,
) -> frozenset[str]:
    """Return variables that may be accessed through a pointer.

    Includes every variable whose address is explicitly taken by a
    ``GetAddress`` instruction in *cfg*, plus every static-duration variable
    (which is globally visible and may be modified through any pointer the
    compiler cannot track).

    Writes to aliased variables must never be eliminated by DSE: even if the
    variable name does not appear in the live set at that program point, the
    value may be observed through a pointer held in some other live variable.

    Args:
        cfg: The control-flow graph to scan for ``GetAddress`` instructions.
        symbol_table: The program symbol table used to identify static-duration
            variables.

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
    return address_taken | _collect_all_static_variables(symbol_table)


def _find_liveness(
    cfg: ControlFlowGraph[tac.Instruction],
    symbol_table: SymbolTable,
) -> tuple[dict[int, LiveSet], dict[int, list[LiveSet]]]:
    """Run the backward liveness dataflow analysis over *cfg*.

    Builds closures that capture *all_static_variables*, then delegates to
    ``run_backward`` with those closures as the meet and transfer functions.
    The result is the converged per-block entry live sets and the per-instruction
    post-state annotations produced by the final transfer call for each block.

    Args:
        cfg: The control-flow graph to analyse.  Successor edges must already
            be wired.
        symbol_table: The program symbol table used to identify static-duration
            variables.

    Returns:
        A 2-tuple ``(block_in, instr_annotations)`` where *block_in* maps each
        block index to its converged entry ``LiveSet`` and *instr_annotations*
        maps each block index to the list of per-instruction post-state
        ``LiveSet`` values from the final transfer pass over that block.
    """
    all_statics = _collect_all_static_variables(symbol_table)
    aliased_vars = _find_aliased_vars(cfg, symbol_table)

    def _meet(block: BasicBlock[tac.Instruction], block_in: dict[int, LiveSet]) -> LiveSet:
        # EXIT boundary: only static-duration vars survive function return.
        return meet(block, all_statics, block_in)

    def _transfer(block: BasicBlock[tac.Instruction], end_live: LiveSet) -> tuple[LiveSet, list[LiveSet]]:
        # aliased_vars (address-taken + statics) used at FunCall and Load sites.
        return transfer(block, end_live, aliased_vars)

    return run_backward(cfg, frozenset(), _meet, _transfer)


def dead_store_elimination(
    cfg: ControlFlowGraph[tac.Instruction],
    symbol_table: SymbolTable | None = None,
) -> ControlFlowGraph[tac.Instruction]:
    """Remove stores to variables whose values are never subsequently read.

    Args:
        cfg: The control-flow graph to optimize.
        symbol_table: The program symbol table used to identify static-duration
            variables.  Pass ``None`` only in tests that do not exercise calls
            to functions or static variables.

    Returns:
        A new ``ControlFlowGraph`` with dead stores removed.
    """
    if not cfg.blocks:
        return cfg

    _, instr_annotations = _find_liveness(cfg, symbol_table or {})

    for block_idx, block in enumerate(cfg.blocks):
        block.instructions = [
            instr
            for instr, annotation in zip(block.instructions, instr_annotations.get(block_idx, []))
            if not is_dead_store(instr, annotation)
        ]

    return cfg
