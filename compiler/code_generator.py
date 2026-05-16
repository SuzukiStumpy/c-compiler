"""Code generation stage: translates a TAC AST into an assembly AST.

Performs a multi-pass structural translation from ``compiler.syntax.tac`` nodes
to ``compiler.syntax.asm`` nodes.  The passes are applied in order:

1. **TAC → pseudo-asm**: each TAC instruction is lowered to one or more assembly
   instructions, using ``Pseudo`` operands for named temporaries.

2. **Pseudo replacement**: every ``Pseudo`` operand is replaced with a ``Stack``
   operand at a unique frame offset; an ``AllocateStack`` instruction is
   prepended when any temporaries were spilled.

3. **Instruction fixup**: invalid two-memory-operand ``Mov`` instructions are
   rewritten to route the value through ``%eax``.

The resulting ``asm.Program`` is a complete assembly AST ready for the emission
stage.

TAC → pseudo-asm mapping:

    tac.Return(val)                           → Mov(val, Register(AX)), Ret()
    tac.Unary(op, src, dst)                   → Mov(src, dst), Unary(op, dst)
    tac.Binary(Add|Sub|Mul, src1, src2, dst)         → Binary(op, src1, src2, Register(R11)),
                                                        Mov(Register(R11), dst)
    tac.Binary(And|Or|Xor, src1, src2, dst)          → Binary(op, src1, src2, Register(R11)),
                                                        Mov(Register(R11), dst)
    tac.Binary(Divide, src1, src2, dst)              → Mov(src1, Register(AX)), Cdq,
                                                        Idiv(src2), Mov(Register(AX), dst)
    tac.Binary(Remainder, src1, src2, dst)           → Mov(src1, Register(AX)), Cdq,
                                                        Idiv(src2), Mov(Register(DX), dst)
    tac.Binary(LeftShift, src1, src2, dst)           → Mov(src1, Register(R11)),
                                                        Mov(src2, Register(CX)),
                                                        Shl(Register(R11)), Mov(Register(R11), dst)
    tac.Binary(RightShift, src1, src2, dst)          → Mov(src1, Register(R11)),
                                                        Mov(src2, Register(CX)),
                                                        Sar(Register(R11)), Mov(Register(R11), dst)
    tac.Constant(v)                           → Imm(v)
    tac.Var(name)                             → Pseudo(name)
"""

import compiler.syntax.asm as asm
import compiler.syntax.tac as tac
from compiler.types.symbols import StaticAttr, SymbolTable
from compiler.utils.exceptions import CodeGenerationError

# System V AMD64 ABI integer argument registers, in order.
_ABI_REGS = [
    asm.Registers.DI,
    asm.Registers.SI,
    asm.Registers.DX,
    asm.Registers.CX,
    asm.Registers.R8,
    asm.Registers.R9,
]


def generate(program: tac.Program, symbol_table: SymbolTable | None = None) -> asm.Program:
    """Translate a TAC ``Program`` into an assembly ``Program``.

    Applies all three translation passes (TAC lowering, pseudo replacement,
    and instruction fixup) and returns a fully resolved assembly AST.

    The ``symbol_table`` is used to identify every name with static storage
    duration, including ``extern``-only (``NoInitializer``) declarations that
    produce no ``tac.StaticVariable`` node.  When ``symbol_table`` is ``None``
    (e.g. in unit tests that construct TAC programs directly), the set is
    derived solely from ``program.top_level``.

    Args:
        program: The ``tac.Program`` node produced by the TAC generation stage.
        symbol_table: The program-wide symbol table produced by semantic
            analysis.  All names carrying a ``StaticAttr`` entry are treated as
            static variables and addressed via RIP-relative ``Data`` operands.

    Returns:
        An ``asm.Program`` node representing the equivalent assembly program.

    Raises:
        CodeGenerationError: If any TAC construct cannot be translated.
    """
    # Names from tac.StaticVariable nodes (Initial / Tentative variables).
    tac_static_names: frozenset[str] = frozenset(
        item.name for item in program.top_level if isinstance(item, tac.StaticVariable)
    )
    # Names from the symbol table for NoInitializer (extern-only) declarations that
    # produce no tac.StaticVariable but must still be addressed via Data operands.
    sym_static_names: frozenset[str] = (
        frozenset(name for name, entry in symbol_table.items() if isinstance(entry.attrs, StaticAttr))
        if symbol_table is not None
        else frozenset()
    )
    static_names = tac_static_names | sym_static_names
    definitions: list[asm.TopLevel] = []
    for item in program.top_level:
        match item:
            case tac.Function():
                raw_fn = _translate_function(item)
                resolved_fn, stack_size = _replace_pseudos(raw_fn, static_names)
                fixed_fn = _fix_instructions(resolved_fn)
                definitions.append(_add_stack_allocation(fixed_fn, stack_size))
            case tac.StaticVariable(name=name, globl=globl, init=init):
                definitions.append(asm.StaticVariable(identifier=name, globl=globl, init=init))
    return asm.Program(definitions=tuple(definitions))


# ==============================================================================
# Pass 1 — TAC lowering
# ==============================================================================


def _translate_function(fn: tac.Function) -> asm.Function:
    """Lower a TAC function to an assembly function with pseudo operands.

    For each parameter, a ``Mov`` from the corresponding ABI register into a
    ``Pseudo`` operand is prepended to the instruction list.  Parameters
    beyond the sixth (which arrive on the stack) are handled by positive-offset
    ``Stack`` operands rather than pseudo-registers.

    Args:
        fn: The TAC function to translate.

    Returns:
        An ``asm.Function`` with the same name and lowered instructions.

    Raises:
        CodeGenerationError: If any instruction cannot be translated.
    """
    instructions: list[asm.Instruction] = []

    # Copy register-passed params into pseudo-registers at function entry.
    for i, param in enumerate(fn.params[:6]):
        instructions.append(asm.Mov(asm.Register(_ABI_REGS[i]), asm.Pseudo(param)))

    # Parameters beyond the sixth arrive on the caller's stack frame at
    # 16(%rbp), 24(%rbp), … (above the saved %rbp and return address).
    for i, param in enumerate(fn.params[6:]):
        instructions.append(asm.Mov(asm.Stack(16 + 8 * i), asm.Pseudo(param)))

    for instr in fn.body:
        instructions.extend(_translate_instruction(instr))

    params_asm = tuple(asm.Register(_ABI_REGS[i]) for i in range(min(len(fn.params), 6)))
    return asm.Function(name=fn.name, globl=fn.globl, params=params_asm, instructions=tuple(instructions))


def _translate_instruction(instr: tac.Instruction) -> list[asm.Instruction]:
    """Lower a single TAC instruction to a sequence of assembly instructions.

    Args:
        instr: The TAC instruction to translate.

    Returns:
        A list of ``asm.Instruction`` nodes equivalent to the TAC instruction.

    Raises:
        CodeGenerationError: If the instruction variant is not supported.
    """
    match instr:
        case tac.Return(val=val):
            return [asm.Mov(src=_translate_val(val), dst=asm.Register(asm.Registers.AX)), asm.Ret()]
        case tac.Jump(target=target):
            return [asm.Jmp(target)]
        case tac.JumpIfZero(condition=condition, target=target):
            return [
                asm.Cmp(asm.Imm(0), _translate_val(condition)),
                asm.JmpCC(asm.CondCodes.E, target),
            ]
        case tac.JumpIfNotZero(condition=condition, target=target):
            return [
                asm.Cmp(asm.Imm(0), _translate_val(condition)),
                asm.JmpCC(asm.CondCodes.NE, target),
            ]
        case tac.Copy(src=src, dst=dst):
            return [asm.Mov(_translate_val(src), _translate_val(dst))]
        case tac.Label(name=name):
            return [asm.Label(name)]
        case tac.Unary(operator=op, src=src, dst=dst):
            src_asm = _translate_val(src)
            dst_asm = _translate_val(dst)
            match op:
                case tac.Not():
                    return [
                        asm.Cmp(asm.Imm(0), src_asm),
                        asm.Mov(asm.Imm(0), dst_asm),
                        asm.SetCC(asm.CondCodes.E, dst_asm),
                    ]
                case _:
                    return [
                        asm.Mov(src_asm, dst_asm),
                        asm.Unary(operator=_translate_unary_op(op), operand=dst_asm),
                    ]
        case tac.Binary(operator=op, src1=src1, src2=src2, dst=dst):
            src1_asm = _translate_val(src1)
            src2_asm = _translate_val(src2)
            dst_asm = _translate_val(dst)
            match op:
                case (
                    tac.Equal()
                    | tac.NotEqual()
                    | tac.LessThan()
                    | tac.LessOrEqual()
                    | tac.GreaterThan()
                    | tac.GreaterOrEqual()
                ):
                    return [
                        asm.Cmp(src2_asm, src1_asm),
                        asm.Mov(asm.Imm(0), dst_asm),
                        asm.SetCC(_translate_relational_op(op), dst_asm),
                    ]
                case (
                    tac.Add() | tac.Subtract() | tac.Multiply() | tac.BitwiseAnd() | tac.BitwiseOr() | tac.BitwiseXor()
                ):
                    return [
                        asm.Binary(
                            operator=_translate_binary_op(op),
                            src1=src1_asm,
                            src2=src2_asm,
                            dst=asm.Register(asm.Registers.R11),
                        ),
                        asm.Mov(src=asm.Register(asm.Registers.R11), dst=dst_asm),
                    ]
                case tac.LeftShift():
                    return [
                        asm.Mov(src=src1_asm, dst=asm.Register(asm.Registers.R11)),
                        asm.Mov(src=src2_asm, dst=asm.Register(asm.Registers.CX)),
                        asm.Shl(operand=asm.Register(asm.Registers.R11)),
                        asm.Mov(src=asm.Register(asm.Registers.R11), dst=dst_asm),
                    ]
                case tac.RightShift():
                    return [
                        asm.Mov(src=src1_asm, dst=asm.Register(asm.Registers.R11)),
                        asm.Mov(src=src2_asm, dst=asm.Register(asm.Registers.CX)),
                        asm.Sar(operand=asm.Register(asm.Registers.R11)),
                        asm.Mov(src=asm.Register(asm.Registers.R11), dst=dst_asm),
                    ]
                case tac.Divide():
                    return [
                        asm.Mov(src=src1_asm, dst=asm.Register(asm.Registers.AX)),
                        asm.Cdq(),
                        asm.Idiv(operand=src2_asm),
                        asm.Mov(src=asm.Register(asm.Registers.AX), dst=dst_asm),
                    ]
                case tac.Remainder():
                    return [
                        asm.Mov(src=src1_asm, dst=asm.Register(asm.Registers.AX)),
                        asm.Cdq(),
                        asm.Idiv(operand=src2_asm),
                        asm.Mov(src=asm.Register(asm.Registers.DX), dst=dst_asm),
                    ]
                case _:
                    raise CodeGenerationError(f"unsupported TAC binary operator: {type(op).__name__}")
        case tac.FunCall(name=name, args=args, dst=dst):
            return _translate_funcall(name, list(args), _translate_val(dst))
        case _:
            raise CodeGenerationError(f"unsupported TAC instruction: {type(instr).__name__}")


def _translate_funcall(name: str, args: list[tac.Val], dst: asm.Operand) -> list[asm.Instruction]:
    """Lower a ``tac.FunCall`` to the System V AMD64 ABI calling sequence.

    Up to six integer arguments are placed in the ABI registers (DI, SI, DX,
    CX, R8, R9).  Additional arguments are pushed onto the stack right-to-left.
    An 8-byte padding ``AllocateStack`` is inserted before the stack pushes
    when there is an odd number of stack-passed arguments, so that the stack
    pointer remains 16-byte aligned at the ``call`` instruction.

    After the call, any stack space consumed by stack-passed arguments (and
    the optional padding) is reclaimed with ``DeallocateStack``.  The return
    value is then moved from ``%eax`` into ``dst``.

    Args:
        name: The symbol name of the function to call.
        args: The ordered list of TAC argument values.
        dst: The assembly operand that receives the return value.

    Returns:
        A list of ``asm.Instruction`` nodes implementing the call site.
    """
    instrs: list[asm.Instruction] = []

    reg_args = args[:6]
    stack_args = args[6:]

    # Odd number of stack args would misalign the stack before `call`.
    needs_padding = len(stack_args) % 2 != 0
    if needs_padding:
        instrs.append(asm.AllocateStack(8))

    # Push stack args right-to-left so the first stack arg is closest to %rbp
    # in the callee's frame (at 16(%rbp)).
    for arg in reversed(stack_args):
        instrs.append(asm.Push(_translate_val(arg)))

    # Load register args into ABI registers.
    for i, arg in enumerate(reg_args):
        instrs.append(asm.Mov(_translate_val(arg), asm.Register(_ABI_REGS[i])))

    instrs.append(asm.Call(name))

    # Reclaim stack space used for stack-passed args and any alignment padding.
    bytes_to_free = 8 * len(stack_args) + (8 if needs_padding else 0)
    if bytes_to_free:
        instrs.append(asm.DeallocateStack(bytes_to_free))

    # Return value is in %eax / %rax.
    instrs.append(asm.Mov(asm.Register(asm.Registers.AX), dst))

    return instrs


def _translate_val(val: tac.Val) -> asm.Operand:
    """Translate a TAC value to an assembly operand.

    Constants become immediate values; variables become pseudo-register operands.

    Args:
        val: The TAC value to translate.

    Returns:
        An ``asm.Operand`` equivalent to the TAC value.

    Raises:
        CodeGenerationError: If the value variant is not supported.
    """
    match val:
        case tac.Constant(value=v):
            return asm.Imm(value=v)
        case tac.Var(value=name):
            return asm.Pseudo(name=name)
        case _:
            raise CodeGenerationError(f"unsupported TAC value: {type(val).__name__}")


def _translate_binary_op(op: tac.BinaryOperator) -> asm.BinaryOperator:
    """Translate a TAC binary operator (Add/Sub/Mul) to its assembly equivalent.

    Divide and Remainder are intentionally absent — they are handled at the
    instruction level by emitting a ``Cdq`` + ``Idiv`` sequence rather than
    an ``asm.Binary`` node.

    Args:
        op: The TAC binary operator node.

    Returns:
        The corresponding ``asm.BinaryOperator``.

    Raises:
        CodeGenerationError: If the operator variant is not supported.
    """
    match op:
        case tac.Add():
            return asm.Add()
        case tac.Subtract():
            return asm.Sub()
        case tac.Multiply():
            return asm.Mult()
        case tac.BitwiseAnd():
            return asm.And()
        case tac.BitwiseOr():
            return asm.Or()
        case tac.BitwiseXor():
            return asm.Xor()
        case _:
            raise CodeGenerationError(f"unsupported TAC binary operator for asm.Binary: {type(op).__name__}")


def _translate_unary_op(op: tac.UnaryOperator) -> asm.UnaryOperator:
    """Translate a TAC unary operator to its assembly equivalent.

    Args:
        op: The TAC unary operator node.

    Returns:
        The corresponding ``asm.UnaryOperator``.

    Raises:
        CodeGenerationError: If the operator variant is not supported.
    """
    match op:
        case tac.Complement():
            return asm.Not()
        case tac.Negate():
            return asm.Neg()
        case _:
            raise CodeGenerationError(f"unsupported TAC unary operator: {type(op).__name__}")


def _translate_relational_op(op: tac.BinaryOperator) -> asm.CondCodes:
    """Translate a TAC relational operator (Equal/NotEqual/LessThan/etc.) to its assembly equivalent.

    Args:
        op: The TAC binary operator node.

    Returns:
        The corresponding ``asm.CondCode``.

    Raises:
        CodeGenerationError: If the operator variant is not supported.
    """
    match op:
        case tac.Equal():
            return asm.CondCodes.E
        case tac.NotEqual():
            return asm.CondCodes.NE
        case tac.LessThan():
            return asm.CondCodes.L
        case tac.LessOrEqual():
            return asm.CondCodes.LE
        case tac.GreaterThan():
            return asm.CondCodes.G
        case tac.GreaterOrEqual():
            return asm.CondCodes.GE
        case _:
            raise CodeGenerationError(f"unsupported TAC relational operator for asm.Binary: {type(op).__name__}")


# ==============================================================================
# Pass 2 — Pseudo-register replacement
# ==============================================================================


def _replace_pseudos(fn: asm.Function, static_names: frozenset[str]) -> tuple[asm.Function, int]:
    """Replace every ``Pseudo`` operand with a concrete operand.

    Pseudo operands whose names appear in ``static_names`` become RIP-relative
    ``Data`` operands.  All other pseudos are assigned a unique 4-byte slot in
    the stack frame, starting at ``-4(%rbp)`` and growing downward.  The total
    number of stack bytes used is returned so the caller can emit an
    ``AllocateStack``.

    Args:
        fn: The assembly function whose pseudo operands are to be resolved.
        static_names: Names of all static/global variables in the program;
            pseudos with these names are mapped to ``asm.Data`` rather than
            a local stack slot.

    Returns:
        A tuple ``(resolved_function, stack_bytes)`` where ``stack_bytes`` is
        the total frame space required (positive, pre-alignment).

    Raises:
        CodeGenerationError: If the function contains an unrecognised
            instruction shape that cannot be rewritten.
    """
    pseudo_offsets: dict[str, int] = {}
    next_offset: list[int] = [0]

    def resolve(operand: asm.Operand) -> asm.Operand:
        if not isinstance(operand, asm.Pseudo):
            return operand
        if operand.name in static_names:
            return asm.Data(name=operand.name)
        if operand.name not in pseudo_offsets:
            next_offset[0] -= 4
            pseudo_offsets[operand.name] = next_offset[0]
        return asm.Stack(offset=pseudo_offsets[operand.name])

    new_instructions: list[asm.Instruction] = []
    for instr in fn.instructions:
        match instr:
            case asm.Mov(src=src, dst=dst):
                new_instructions.append(asm.Mov(src=resolve(src), dst=resolve(dst)))
            case asm.Unary(operator=op, operand=operand):
                new_instructions.append(asm.Unary(operator=op, operand=resolve(operand)))
            case asm.Binary(operator=op, src1=src1, src2=src2, dst=dst):
                new_instructions.append(asm.Binary(operator=op, src1=resolve(src1), src2=resolve(src2), dst=dst))
            case asm.Idiv(operand=operand):
                new_instructions.append(asm.Idiv(operand=resolve(operand)))
            case asm.Shl(operand=operand) | asm.Sar(operand=operand):
                new_instructions.append(type(instr)(operand=resolve(operand)))
            case asm.Cmp(op1=op1, op2=op2):
                new_instructions.append(asm.Cmp(op1=resolve(op1), op2=resolve(op2)))
            case asm.SetCC(cond_code=cc, operand=operand):
                new_instructions.append(asm.SetCC(cond_code=cc, operand=resolve(operand)))
            case asm.Push(operand=operand):
                new_instructions.append(asm.Push(operand=resolve(operand)))
            case _:
                new_instructions.append(instr)

    stack_size = -next_offset[0]
    return asm.Function(
        name=fn.name, globl=fn.globl, params=fn.params, instructions=tuple(new_instructions)
    ), stack_size


# ==============================================================================
# Pass 3 — Instruction fixup
# ==============================================================================


def _fix_instructions(fn: asm.Function) -> asm.Function:
    """Rewrite instructions that are invalid in x86-64 assembly.

    The x86 ``movl`` instruction does not allow both operands to be memory
    addresses (``Stack`` or ``Data``).  Any such ``Mov`` is replaced by two
    moves that route the value through ``%r10d``.  The same constraint applies
    to ``Cmp`` when op1 is a memory address.  ``Push`` of a memory operand is
    expanded through ``%eax``/``%rax``.

    Args:
        fn: The assembly function to fix up.

    Returns:
        An ``asm.Function`` with all invalid instructions rewritten.
    """

    def _is_mem(op: asm.Operand) -> bool:
        return isinstance(op, (asm.Stack, asm.Data))

    fixed: list[asm.Instruction] = []
    for instr in fn.instructions:
        match instr:
            case asm.Mov(src=src, dst=dst) if _is_mem(src) and _is_mem(dst):
                fixed.append(asm.Mov(src=src, dst=asm.Register(asm.Registers.R10)))
                fixed.append(asm.Mov(src=asm.Register(asm.Registers.R10), dst=dst))
            case asm.Idiv(operand=asm.Imm() as imm):
                # idivl cannot take an immediate operand; route through %r11d
                fixed.append(asm.Mov(src=imm, dst=asm.Register(asm.Registers.R11)))
                fixed.append(asm.Idiv(operand=asm.Register(asm.Registers.R11)))
            # op2 cannot be an immediate - so route it through %r11d
            case asm.Cmp(op1=op1, op2=asm.Imm() as op2):
                fixed.append(asm.Mov(src=op2, dst=asm.Register(asm.Registers.R11)))
                fixed.append(asm.Cmp(op1=op1, op2=asm.Register(asm.Registers.R11)))
            # op1 cannot be a memory address when op2 is also memory - route op1 through %r10d
            case asm.Cmp(op1=op1, op2=op2) if _is_mem(op1):
                fixed.append(asm.Mov(src=op1, dst=asm.Register(asm.Registers.R10)))
                fixed.append(asm.Cmp(op1=asm.Register(asm.Registers.R10), op2=op2))
            # pushq cannot address a 32-bit memory slot directly; load into %eax first.
            # movl zero-extends to %rax in x86-64, so pushq %rax carries the correct value.
            case asm.Push(operand=operand) if _is_mem(operand):
                fixed.append(asm.Mov(src=operand, dst=asm.Register(asm.Registers.AX)))
                fixed.append(asm.Push(operand=asm.Register(asm.Registers.AX, asm.Width.QWORD)))
            case _:
                fixed.append(instr)
    return asm.Function(name=fn.name, globl=fn.globl, params=fn.params, instructions=tuple(fixed))


# ==============================================================================
# Stack allocation
# ==============================================================================


def _add_stack_allocation(fn: asm.Function, stack_size: int) -> asm.Function:
    """Prepend an ``AllocateStack`` instruction when the function uses the stack.

    The size is rounded up to the nearest multiple of 16 bytes to preserve
    16-byte stack alignment required by the System V AMD64 ABI.

    Args:
        fn: The assembly function to modify.
        stack_size: The raw number of stack bytes needed (0 if none).

    Returns:
        An ``asm.Function`` with an ``AllocateStack`` prepended, or the
        original function unchanged when ``stack_size`` is zero.
    """
    if stack_size == 0:
        return fn
    aligned = (stack_size + 15) & ~15
    return asm.Function(
        name=fn.name,
        globl=fn.globl,
        params=fn.params,
        instructions=(asm.AllocateStack(size=aligned), *fn.instructions),
    )
