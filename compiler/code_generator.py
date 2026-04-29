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

    tac.Return(val)              → Mov(val, Register(Registers.AX)), Ret()
    tac.Unary(op, src, dst)      → Mov(src, dst), Unary(op, dst)
    tac.Constant(v)              → Imm(v)
    tac.Var(name)                → Pseudo(name)
"""

import compiler.syntax.asm as asm
import compiler.syntax.tac as tac
from compiler.utils.exceptions import CodeGenerationError


def generate(program: tac.Program) -> asm.Program:
    """Translate a TAC ``Program`` into an assembly ``Program``.

    Applies all three translation passes (TAC lowering, pseudo replacement,
    and instruction fixup) and returns a fully resolved assembly AST.

    Args:
        program: The ``tac.Program`` node produced by the TAC generation stage.

    Returns:
        An ``asm.Program`` node representing the equivalent assembly program.

    Raises:
        CodeGenerationError: If any TAC construct cannot be translated.
    """
    raw_fn = _translate_function(program.function)
    resolved_fn, stack_size = _replace_pseudos(raw_fn)
    fixed_fn = _fix_instructions(resolved_fn)
    final_fn = _add_stack_allocation(fixed_fn, stack_size)
    return asm.Program(function=final_fn)


# ==============================================================================
# Pass 1 — TAC lowering
# ==============================================================================


def _translate_function(fn: tac.Function) -> asm.Function:
    """Lower a TAC function to an assembly function with pseudo operands.

    Args:
        fn: The TAC function to translate.

    Returns:
        An ``asm.Function`` with the same name and lowered instructions.

    Raises:
        CodeGenerationError: If any instruction cannot be translated.
    """
    instructions: list[asm.Instruction] = []
    for instr in fn.body:
        instructions.extend(_translate_instruction(instr))
    return asm.Function(name=fn.name, instructions=tuple(instructions))


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
        case tac.Unary(operator=op, src=src, dst=dst):
            return [
                asm.Mov(src=_translate_val(src), dst=_translate_val(dst)),
                asm.Unary(operator=_translate_unary_op(op), operand=_translate_val(dst)),
            ]
        case _:
            raise CodeGenerationError(f"unsupported TAC instruction: {type(instr).__name__}")


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


# ==============================================================================
# Pass 2 — Pseudo-register replacement
# ==============================================================================


def _replace_pseudos(fn: asm.Function) -> tuple[asm.Function, int]:
    """Replace every ``Pseudo`` operand with a frame-relative ``Stack`` operand.

    Each distinct pseudo name is assigned a unique 4-byte slot in the stack
    frame, starting at ``-4(%rbp)`` and growing downward.  The total number of
    bytes used is returned so the caller can emit an ``AllocateStack``.

    Args:
        fn: The assembly function whose pseudo operands are to be resolved.

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
            case _:
                new_instructions.append(instr)

    stack_size = -next_offset[0]
    return asm.Function(name=fn.name, instructions=tuple(new_instructions)), stack_size


# ==============================================================================
# Pass 3 — Instruction fixup
# ==============================================================================


def _fix_instructions(fn: asm.Function) -> asm.Function:
    """Rewrite instructions that are invalid in x86-64 assembly.

    The x86 ``movl`` instruction does not allow both operands to be memory
    addresses.  Any such ``Mov`` is replaced by two moves that route the value
    through ``%eax``.

    Args:
        fn: The assembly function to fix up.

    Returns:
        An ``asm.Function`` with all invalid instructions rewritten.
    """
    fixed: list[asm.Instruction] = []
    for instr in fn.instructions:
        match instr:
            case asm.Mov(src=asm.Stack() as src, dst=asm.Stack() as dst):
                fixed.append(asm.Mov(src=src, dst=asm.Register(asm.Registers.R10)))
                fixed.append(asm.Mov(src=asm.Register(asm.Registers.R10), dst=dst))
            case _:
                fixed.append(instr)
    return asm.Function(name=fn.name, instructions=tuple(fixed))


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
    return asm.Function(name=fn.name, instructions=(asm.AllocateStack(size=aligned), *fn.instructions))
