"""Code generation stage: translates a TAC AST into an assembly AST.

Performs a multi-pass structural translation from ``compiler.syntax.tac`` nodes
to ``compiler.syntax.asm`` nodes.  The passes are applied in order:

1. **TAC → pseudo-asm**: each TAC instruction is lowered to one or more assembly
   instructions, using ``Pseudo`` operands for named temporaries.

2. **Pseudo replacement**: every ``Pseudo`` operand is replaced with a ``Stack``
   operand at a unique frame offset; an ``AllocateStack`` instruction is
   prepended when any temporaries were spilled.

3. **Instruction fixup**: invalid two-memory-operand ``Mov`` instructions are
   rewritten to route the value through a scratch register.

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
    tac.SignExtend(src, dst)                  → Movsx(src_type, dst_type, src, dst)
    tac.ZeroExtend(src, dst)                  → MoveZeroExtend(src_type, dst_type, src, dst)
    tac.Truncate(src, dst)                    → Mov(dst_type, src, dst)
    tac.Constant(v)                           → Imm(v)
    tac.Var(name)                             → Pseudo(name)

Unsigned variants (selected when the source operand has an unsigned C type):

    tac.Binary(Divide, unsigned, src1, src2, dst)    → Mov(src1, AX), Mov(Imm(0), DX),
                                                        Div(src2), Mov(AX, dst)
    tac.Binary(Remainder, unsigned, src1, src2, dst) → Mov(src1, AX), Mov(Imm(0), DX),
                                                        Div(src2), Mov(DX, dst)
    tac.Binary(RightShift, unsigned, src1, src2, dst) → Mov(src1, R11), Mov(src2, CX),
                                                         Shr(R11), Mov(R11, dst)
    tac.Binary(relational, unsigned, ...)             → Cmp, Mov(Imm(0), dst), SetCC(A/AE/B/BE)
"""

import itertools

import compiler.syntax.asm as asm
import compiler.syntax.tac as tac
from compiler.syntax.ast.constants import ConstChar, ConstDouble, ConstInt, ConstLong, ConstUChar, ConstUInt, ConstULong
from compiler.syntax.ast.types import Array as ArrayType
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
from compiler.types.symbols import FunAttr, StaticAttr, SymbolTable
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

# System V AMD64 ABI floating-point argument registers, in order.
_XMM_REGS = [
    asm.Registers.XMM0,
    asm.Registers.XMM1,
    asm.Registers.XMM2,
    asm.Registers.XMM3,
    asm.Registers.XMM4,
    asm.Registers.XMM5,
    asm.Registers.XMM6,
    asm.Registers.XMM7,
]

# 2^63 as a double — the threshold used by DoubleToUInt for unsigned-long destinations.
# Values in [2^63, 2^64) exceed the signed int64 range and need special handling.
_DOUBLE_TO_ULONG_THRESHOLD = 9_223_372_036_854_775_808.0

_uid_counter = itertools.count()


def _fresh_uid() -> int:
    """Return a unique integer, used to generate distinct branch-target labels.

    Returns:
        A monotonically increasing integer unique to this process run.
    """
    return next(_uid_counter)


def _asm_type(val: tac.Val, symbol_table: SymbolTable | None = None) -> asm.AssemblyType:
    """Derive the assembly type for a TAC value.

    Constants are typed by their concrete ``Const`` subclass; variables are
    looked up in the symbol table.  Unresolvable variables default to
    ``Longword``.

    Args:
        val: The TAC value whose assembly type is needed.
        symbol_table: The program-wide symbol table.

    Returns:
        ``Byte`` for 8-bit char values, ``Longword`` for 32-bit values,
        ``Quadword`` for 64-bit values, ``Double`` for floating-point values.
    """
    match val:
        case tac.Constant(value=ConstChar() | ConstUChar()):
            return asm.Byte()
        case tac.Constant(value=ConstInt() | ConstUInt()):
            return asm.Longword()
        case tac.Constant(value=ConstLong() | ConstULong()):
            return asm.Quadword()
        case tac.Constant(value=ConstDouble()):
            return asm.Double()
        case tac.Var(value=name) if symbol_table is not None and name in symbol_table:
            match symbol_table[name].t:
                case LongType() | ULongType() | Pointer():
                    return asm.Quadword()
                case DoubleType():
                    return asm.Double()
                case Char() | SChar() | UChar():
                    return asm.Byte()
                case _:
                    return asm.Longword()
        case _:
            return asm.Longword()


def _reg_width(assembly_type: asm.AssemblyType) -> asm.Width:
    """Return the register width corresponding to an assembly type.

    Args:
        assembly_type: The assembly operand size.

    Returns:
        ``Width.BYTE`` for Byte, ``Width.DWORD`` for Longword, ``Width.QWORD`` for Quadword.
    """
    if assembly_type == asm.Byte():
        return asm.Width.BYTE
    return asm.Width.DWORD if assembly_type == asm.Longword() else asm.Width.QWORD


def _is_unsigned(val: tac.Val, symbol_table: SymbolTable | None = None) -> bool:
    """Return ``True`` when ``val`` has an unsigned C type.

    Used to select between signed (``Idiv``/``Sar``/``L``/``G`` etc.) and unsigned
    (``Div``/``Shr``/``B``/``A`` etc.) instruction variants.

    Args:
        val: The TAC value to classify.
        symbol_table: The program-wide symbol table; required for ``tac.Var`` lookups.

    Returns:
        ``True`` for ``unsigned int`` and ``unsigned long`` values; ``False`` otherwise.
    """
    match val:
        case tac.Constant(value=ConstUChar() | ConstUInt() | ConstULong()):
            return True
        case tac.Var(value=name) if symbol_table is not None and name in symbol_table:
            return isinstance(symbol_table[name].t, (UChar, UIntType, ULongType, Pointer))
        case _:
            return False


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
        frozenset(name for name, entry in symbol_table.items() if isinstance(entry.attrs, (StaticAttr, FunAttr)))
        if symbol_table is not None
        else frozenset()
    )
    static_names = tac_static_names | sym_static_names
    # Accumulates StaticConstant entries emitted by conversion instructions (e.g. the
    # 2^63 threshold for double → unsigned-long).  These are prepended to the output
    # so that they appear before the first .text section.
    constants: list[asm.TopLevel] = []
    definitions: list[asm.TopLevel] = []
    for item in program.top_level:
        match item:
            case tac.Function():
                raw_fn = _translate_function(item, symbol_table, constants)
                resolved_fn, stack_size = _replace_pseudos(raw_fn, static_names, symbol_table)
                fixed_fn = _fix_instructions(resolved_fn)
                definitions.append(_add_stack_allocation(fixed_fn, stack_size))
            case tac.StaticVariable(name=name, globl=globl, t=t, init=init):
                definitions.append(
                    asm.StaticVariable(
                        identifier=name,
                        globl=globl,
                        alignment=_static_alignment(t),
                        init=tuple(_translate_static_init(i) for i in init),
                    )
                )
    return asm.Program(definitions=tuple(constants + definitions))


def _type_size(t: Type) -> int:
    """Return the size in bytes of a C type.

    Args:
        t: The C type whose size is required.

    Returns:
        4 for ``int``/``unsigned int``; 8 for ``long``/``unsigned long``/``double``/pointer;
        element_size * count for arrays.
    """
    if isinstance(t, (Char, SChar, UChar)):
        return 1
    if isinstance(t, (IntType, UIntType)):
        return 4
    if isinstance(t, (LongType, ULongType, DoubleType, Pointer)):
        return 8
    if isinstance(t, ArrayType):
        return t.size * _type_size(t.element)
    return 8


def _static_alignment(t: Type) -> int:
    """Return the data-segment alignment in bytes for a static variable of type ``t``.

    Arrays >= 16 bytes are aligned to 16 bytes so SSE instructions can address them
    directly.  Arrays smaller than 16 bytes use the alignment of their element type.
    The rule applies only to the outermost array; element alignment within the array
    is not affected.

    Args:
        t: The C type of the static variable.

    Returns:
        16 for arrays whose total size is >= 16 bytes; the element-type alignment
        for smaller arrays; 8 for ``long``, ``unsigned long``, ``double``, or any
        pointer; 4 otherwise.
    """
    if isinstance(t, (LongType, ULongType, DoubleType, Pointer)):
        return 8
    if isinstance(t, ArrayType):
        if _type_size(t) >= 16:
            return 16
        return _static_alignment(t.element)
    return 4


def _translate_static_init(init: tac.StaticInit) -> asm.StaticInit:
    """Convert a single TAC static initialiser element to its assembly counterpart.

    Args:
        init: One element of a ``tac.StaticVariable.init`` tuple.

    Returns:
        The corresponding ``asm.StaticInit`` node.

    Raises:
        CodeGenerationError: If the initialiser variant is not supported.
    """
    match init:
        case tac.IntInit(value=v):
            return asm.IntInit(v)
        case tac.LongInit(value=v):
            return asm.LongInit(v)
        case tac.DoubleInit(value=v):
            return asm.DoubleInit(v)
        case tac.ZeroInit(bytes=n):
            return asm.ZeroInit(n)
        case _:
            raise CodeGenerationError(f"unsupported static initialiser: {type(init).__name__}")


# ==============================================================================
# Pass 1 — TAC lowering
# ==============================================================================


def _translate_function(
    fn: tac.Function,
    symbol_table: SymbolTable | None = None,
    constants: list[asm.TopLevel] | None = None,
) -> asm.Function:
    """Lower a TAC function to an assembly function with pseudo operands.

    For each parameter, a ``Mov`` from the corresponding ABI register into a
    ``Pseudo`` operand is prepended to the instruction list.  Parameters
    beyond the sixth (which arrive on the stack) are handled by positive-offset
    ``Stack`` operands rather than pseudo-registers.

    Args:
        fn: The TAC function to translate.
        symbol_table: Used to derive the assembly type for each parameter.
        constants: Mutable list into which any required ``StaticConstant`` nodes
            (e.g. the ``DoubleToUInt`` threshold) are appended.

    Returns:
        An ``asm.Function`` with the same name and lowered instructions.

    Raises:
        CodeGenerationError: If any instruction cannot be translated.
    """
    instructions: list[asm.Instruction] = []

    # Assign each parameter to an ABI register or a caller stack slot.
    # Integer and double parameters each have their own register class (GP vs XMM)
    # and are counted independently; each class spills to the stack when exhausted.
    # Stack-passed parameters of both classes share one stack index, advancing in
    # parameter-declaration order.
    int_reg_idx = 0
    xmm_reg_idx = 0
    stack_param_idx = 0

    for param in fn.params:
        at = _asm_type(tac.Var(param), symbol_table)
        if at == asm.Double():
            if xmm_reg_idx < len(_XMM_REGS):
                src: asm.Operand = asm.Register(_XMM_REGS[xmm_reg_idx], asm.Width.QWORD)
                xmm_reg_idx += 1
            else:
                src = asm.Memory(reg=asm.Register(asm.Registers.BP, asm.Width.QWORD), offset=16 + 8 * stack_param_idx)
                stack_param_idx += 1
        else:
            if int_reg_idx < len(_ABI_REGS):
                src = asm.Register(_ABI_REGS[int_reg_idx], _reg_width(at))
                int_reg_idx += 1
            else:
                src = asm.Memory(reg=asm.Register(asm.Registers.BP, asm.Width.QWORD), offset=16 + 8 * stack_param_idx)
                stack_param_idx += 1
        instructions.append(asm.Mov(src=src, dst=asm.Pseudo(param), assembly_type=at))

    for instr in fn.body:
        instructions.extend(_translate_instruction(instr, symbol_table, constants))

    return asm.Function(name=fn.name, globl=fn.globl, params=(), instructions=tuple(instructions))


def _translate_binary(
    op: tac.BinaryOperator,
    src1: tac.Val,
    src2: tac.Val,
    dst: tac.Val,
    symbol_table: SymbolTable | None = None,
    constants: list[asm.TopLevel] | None = None,
) -> list[asm.Instruction]:
    """Lower a single TAC Binary instruction to a sequence of assembly instructions.

    Handles arithmetic, bitwise, shift, division, and relational operators,
    selecting signed or unsigned variants based on the type of ``src1``.
    Double-precision operations route through XMM15 using SSE scalar mnemonics.

    Args:
        op: The TAC binary operator.
        src1: The left source operand.
        src2: The right source operand.
        dst: The destination.
        symbol_table: Used to derive assembly types and signedness for variable operands.
        constants: Accumulator for ``StaticConstant`` entries emitted for double literals.

    Returns:
        A list of ``asm.Instruction`` nodes equivalent to the TAC binary instruction.

    Raises:
        CodeGenerationError: If the operator variant is not supported.
    """
    src1_asm = _translate_val(src1, constants, symbol_table)
    src2_asm = _translate_val(src2, constants, symbol_table)
    dst_asm = _translate_val(dst, constants, symbol_table)
    match op:
        case (
            tac.Equal() | tac.NotEqual() | tac.LessThan() | tac.LessOrEqual() | tac.GreaterThan() | tac.GreaterOrEqual()
        ):
            cmp_at = _asm_type(src1, symbol_table)
            if cmp_at == asm.Double():
                return _translate_double_comparison(op, src1_asm, src2_asm, dst_asm)
            unsigned = _is_unsigned(src1, symbol_table)
            return [
                asm.Cmp(op1=src2_asm, op2=src1_asm, assembly_type=cmp_at),
                asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                asm.SetCC(_translate_relational_op(op, unsigned), dst_asm),
            ]
        case tac.Add() | tac.Subtract() | tac.Multiply() | tac.BitwiseAnd() | tac.BitwiseOr() | tac.BitwiseXor():
            at = _asm_type(dst, symbol_table)
            if at == asm.Double():
                xmm15 = asm.Register(asm.Registers.XMM15, asm.Width.QWORD)
                return [
                    asm.Binary(
                        operator=_translate_binary_op_double(op),
                        src1=src1_asm,
                        src2=src2_asm,
                        dst=xmm15,
                        assembly_type=at,
                    ),
                    asm.Mov(src=xmm15, dst=dst_asm, assembly_type=at),
                ]
            r11 = asm.Register(asm.Registers.R11, _reg_width(at))
            return [
                asm.Binary(
                    operator=_translate_binary_op(op),
                    src1=src1_asm,
                    src2=src2_asm,
                    dst=r11,
                    assembly_type=at,
                ),
                asm.Mov(src=r11, dst=dst_asm, assembly_type=at),
            ]
        case tac.LeftShift():
            at = _asm_type(dst, symbol_table)
            r11 = asm.Register(asm.Registers.R11, _reg_width(at))
            cx = asm.Register(asm.Registers.CX, asm.Width.DWORD)
            return [
                asm.Mov(src=src1_asm, dst=r11, assembly_type=at),
                asm.Mov(src=src2_asm, dst=cx, assembly_type=asm.Longword()),
                asm.Shl(operand=r11, assembly_type=at),
                asm.Mov(src=r11, dst=dst_asm, assembly_type=at),
            ]
        case tac.RightShift():
            at = _asm_type(dst, symbol_table)
            r11 = asm.Register(asm.Registers.R11, _reg_width(at))
            cx = asm.Register(asm.Registers.CX, asm.Width.DWORD)
            # Unsigned right shift is logical (shr); signed is arithmetic (sar).
            shift: asm.Instruction = (
                asm.Shr(operand=r11, assembly_type=at)
                if _is_unsigned(src1, symbol_table)
                else asm.Sar(operand=r11, assembly_type=at)
            )
            return [
                asm.Mov(src=src1_asm, dst=r11, assembly_type=at),
                asm.Mov(src=src2_asm, dst=cx, assembly_type=asm.Longword()),
                shift,
                asm.Mov(src=r11, dst=dst_asm, assembly_type=at),
            ]
        case tac.Divide():
            at = _asm_type(dst, symbol_table)
            if at == asm.Double():
                xmm15 = asm.Register(asm.Registers.XMM15, asm.Width.QWORD)
                return [
                    asm.Binary(
                        operator=asm.DivDouble(),
                        src1=src1_asm,
                        src2=src2_asm,
                        dst=xmm15,
                        assembly_type=at,
                    ),
                    asm.Mov(src=xmm15, dst=dst_asm, assembly_type=at),
                ]
            ax = asm.Register(asm.Registers.AX, _reg_width(at))
            dx = asm.Register(asm.Registers.DX, _reg_width(at))
            if _is_unsigned(src1, symbol_table):
                return [
                    asm.Mov(src=src1_asm, dst=ax, assembly_type=at),
                    asm.Mov(src=asm.Imm(0), dst=dx, assembly_type=at),
                    asm.Div(operand=src2_asm, assembly_type=at),
                    asm.Mov(src=ax, dst=dst_asm, assembly_type=at),
                ]
            return [
                asm.Mov(src=src1_asm, dst=ax, assembly_type=at),
                asm.Cdq(assembly_type=at),
                asm.Idiv(operand=src2_asm, assembly_type=at),
                asm.Mov(src=ax, dst=dst_asm, assembly_type=at),
            ]
        case tac.Remainder():
            at = _asm_type(dst, symbol_table)
            ax = asm.Register(asm.Registers.AX, _reg_width(at))
            dx = asm.Register(asm.Registers.DX, _reg_width(at))
            if _is_unsigned(src1, symbol_table):
                return [
                    asm.Mov(src=src1_asm, dst=ax, assembly_type=at),
                    asm.Mov(src=asm.Imm(0), dst=dx, assembly_type=at),
                    asm.Div(operand=src2_asm, assembly_type=at),
                    asm.Mov(src=dx, dst=dst_asm, assembly_type=at),
                ]
            return [
                asm.Mov(src=src1_asm, dst=ax, assembly_type=at),
                asm.Cdq(assembly_type=at),
                asm.Idiv(operand=src2_asm, assembly_type=at),
                asm.Mov(src=dx, dst=dst_asm, assembly_type=at),
            ]
        case _:
            raise CodeGenerationError(f"unsupported TAC binary operator: {type(op).__name__}")


def _zero_operand_for(at: asm.AssemblyType, constants: list[asm.TopLevel] | None) -> asm.Operand:
    """Return a zero operand appropriate for the given assembly type.

    For integer types returns ``Imm(0)``.  For ``Double``, integer immediates
    are not valid SSE operands, so a 0.0 ``StaticConstant`` is allocated in
    ``.rodata`` and a ``Data`` reference is returned.

    Args:
        at: The assembly type of the zero operand needed.
        constants: Accumulator for ``StaticConstant`` entries; the 0.0 entry
            is appended here when ``at`` is ``Double`` and it is not already
            present.

    Returns:
        An ``Imm(0)`` for integer types, or a ``Data`` reference to 0.0 for
        ``Double``.
    """
    if at == asm.Double():
        zero_bits = asm.DoubleInit(0.0).as_bits()
        label = f"double_const.{zero_bits}"
        if constants is not None and not any(
            isinstance(c, asm.StaticConstant) and c.identifier == label for c in constants
        ):
            constants.append(asm.StaticConstant(identifier=label, alignment=8, init=asm.DoubleInit(0.0)))
        return asm.Data(label)
    return asm.Imm(0)


def _translate_instruction(  # noqa: C901 - ignore complexity warning
    instr: tac.Instruction,
    symbol_table: SymbolTable | None = None,
    constants: list[asm.TopLevel] | None = None,
) -> list[asm.Instruction]:
    """Lower a single TAC instruction to a sequence of assembly instructions.

    Args:
        instr: The TAC instruction to translate.
        symbol_table: Used to derive assembly types for variable operands.
        constants: Mutable list into which any required ``StaticConstant`` nodes
            are appended (used by ``DoubleToUInt`` for the 2^63 threshold).

    Returns:
        A list of ``asm.Instruction`` nodes equivalent to the TAC instruction.

    Raises:
        CodeGenerationError: If the instruction variant is not supported.
    """
    match instr:
        case tac.Return(val=val):
            at = _asm_type(val, symbol_table)
            val_asm = _translate_val(val, constants, symbol_table)
            if at == asm.Double():
                xmm0 = asm.Register(asm.Registers.XMM0, asm.Width.QWORD)
                return [asm.Mov(src=val_asm, dst=xmm0, assembly_type=at), asm.Ret()]
            ax = asm.Register(asm.Registers.AX, _reg_width(at))
            return [asm.Mov(src=val_asm, dst=ax, assembly_type=at), asm.Ret()]
        case tac.Jump(target=target):
            return [asm.Jmp(target)]
        case tac.JumpIfZero(condition=condition, target=target):
            at = _asm_type(condition, symbol_table)
            op1 = _zero_operand_for(at, constants)
            cond_asm = _translate_val(condition, constants, symbol_table)
            if at == asm.Double():
                # ucomisd sets PF=1 for NaN; je fires on ZF=1 which is also set by NaN.
                # Skip the je when PF=1 so NaN is never treated as zero.
                skip_label = f"nan_jz.{_fresh_uid()}"
                return [
                    asm.Cmp(op1=op1, op2=cond_asm, assembly_type=at),
                    asm.JmpCC(asm.CondCodes.P, skip_label),
                    asm.JmpCC(asm.CondCodes.E, target),
                    asm.Label(skip_label),
                ]
            return [
                asm.Cmp(op1=op1, op2=cond_asm, assembly_type=at),
                asm.JmpCC(asm.CondCodes.E, target),
            ]
        case tac.JumpIfNotZero(condition=condition, target=target):
            at = _asm_type(condition, symbol_table)
            op1 = _zero_operand_for(at, constants)
            cond_asm = _translate_val(condition, constants, symbol_table)
            if at == asm.Double():
                # ucomisd sets PF=1 and ZF=1 for NaN; jne checks ZF=0 so misses NaN.
                # Jump on PF first so NaN is treated as nonzero.
                return [
                    asm.Cmp(op1=op1, op2=cond_asm, assembly_type=at),
                    asm.JmpCC(asm.CondCodes.P, target),
                    asm.JmpCC(asm.CondCodes.NE, target),
                ]
            return [
                asm.Cmp(op1=op1, op2=cond_asm, assembly_type=at),
                asm.JmpCC(asm.CondCodes.NE, target),
            ]
        case tac.Copy(src=src, dst=dst):
            src_asm = _translate_val(src, constants, symbol_table)
            dst_asm = _translate_val(dst, constants, symbol_table)
            # A function name used as a value carries its address, not a dereferenceable word;
            # use Lea so the emitter produces leaq <fn>(%rip), <dst> rather than a movq.
            if (
                isinstance(src, tac.Var)
                and symbol_table is not None
                and src.value in symbol_table
                and isinstance(symbol_table[src.value].attrs, FunAttr)
            ):
                return [asm.Lea(src=src_asm, dst=dst_asm)]
            at = _asm_type(src, symbol_table)
            return [asm.Mov(src=src_asm, dst=dst_asm, assembly_type=at)]
        case tac.Label(name=name):
            return [asm.Label(name)]
        case tac.Unary(operator=op, src=src, dst=dst):
            at = _asm_type(dst, symbol_table)
            src_asm = _translate_val(src, constants, symbol_table)
            dst_asm = _translate_val(dst, constants, symbol_table)
            match op:
                case tac.Not():
                    src_at = _asm_type(src, symbol_table)
                    zero = _zero_operand_for(src_at, constants)
                    if src_at == asm.Double():
                        # ucomisd sets PF=1 and ZF=1 for NaN; sete fires on ZF=1 so would
                        # return 1 for NaN.  Jump over sete when PF=1 so !NaN stays 0.
                        nan_label = f"nan_not.{_fresh_uid()}"
                        return [
                            asm.Cmp(op1=zero, op2=src_asm, assembly_type=src_at),
                            asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                            asm.JmpCC(asm.CondCodes.P, nan_label),
                            asm.SetCC(asm.CondCodes.E, dst_asm),
                            asm.Label(nan_label),
                        ]
                    return [
                        asm.Cmp(op1=zero, op2=src_asm, assembly_type=src_at),
                        asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                        asm.SetCC(asm.CondCodes.E, dst_asm),
                    ]
                case tac.Negate() if at == asm.Double():
                    # IEEE 754 negation flips only the sign bit; use xorpd with -0.0 (sign bit set).
                    sign_mask_bits = asm.DoubleInit(-0.0).as_bits()
                    sign_mask_label = f"double_const.{sign_mask_bits}"
                    if constants is not None and not any(
                        isinstance(c, asm.StaticConstant) and c.identifier == sign_mask_label for c in constants
                    ):
                        constants.append(
                            asm.StaticConstant(identifier=sign_mask_label, alignment=16, init=asm.DoubleInit(-0.0))
                        )
                    xmm14 = asm.Register(asm.Registers.XMM14, asm.Width.QWORD)
                    return [
                        asm.Binary(
                            operator=asm.XorDouble(),
                            src1=src_asm,
                            src2=asm.Data(sign_mask_label),
                            dst=xmm14,
                            assembly_type=at,
                        ),
                        asm.Mov(src=xmm14, dst=dst_asm, assembly_type=at),
                    ]
                case _:
                    return [
                        asm.Mov(src=src_asm, dst=dst_asm, assembly_type=at),
                        asm.Unary(operator=_translate_unary_op(op), operand=dst_asm, assembly_type=at),
                    ]
        case tac.Binary(operator=op, src1=src1, src2=src2, dst=dst):
            return _translate_binary(op, src1, src2, dst, symbol_table, constants)
        case tac.FunCall(fun_val=fun_val, args=args, dst=dst):
            dst_asm = _translate_val(dst, None, symbol_table)
            return _translate_funcall(fun_val, list(args), dst_asm, symbol_table, constants)
        case tac.SignExtend(src=src, dst=dst):
            return [
                asm.Movsx(
                    src_type=_asm_type(src, symbol_table),
                    dst_type=_asm_type(dst, symbol_table),
                    src=_translate_val(src, None, symbol_table),
                    dst=_translate_val(dst, None, symbol_table),
                )
            ]
        case tac.ZeroExtend(src=src, dst=dst):
            return [
                asm.MoveZeroExtend(
                    src_type=_asm_type(src, symbol_table),
                    dst_type=_asm_type(dst, symbol_table),
                    src=_translate_val(src, None, symbol_table),
                    dst=_translate_val(dst, None, symbol_table),
                )
            ]
        case tac.Truncate(src=src, dst=dst):
            # movb/movl truncates by discarding upper bits; assembly_type matches the destination width.
            return [
                asm.Mov(
                    src=_translate_val(src, None, symbol_table),
                    dst=_translate_val(dst, None, symbol_table),
                    assembly_type=_asm_type(dst, symbol_table),
                )
            ]
        case tac.DoubleToInt(src=src, dst=dst):
            at = _asm_type(dst, symbol_table)
            if at == asm.Byte():
                # cvttsd2si has no byte form; convert to int32 then truncate the low byte.
                r10d = asm.Register(asm.Registers.R10, asm.Width.DWORD)
                r10b = asm.Register(asm.Registers.R10, asm.Width.BYTE)
                return [
                    asm.Cvttsd2si(
                        src=_translate_val(src, constants, symbol_table), dst=r10d, assembly_type=asm.Longword()
                    ),
                    asm.Mov(src=r10b, dst=_translate_val(dst, None, symbol_table), assembly_type=asm.Byte()),
                ]
            # assembly_type controls the destination integer width (l = int, q = long).
            return [
                asm.Cvttsd2si(
                    src=_translate_val(src, constants, symbol_table),
                    dst=_translate_val(dst, None, symbol_table),
                    assembly_type=at,
                )
            ]
        case tac.IntToDouble(src=src, dst=dst):
            at = _asm_type(src, symbol_table)
            if at == asm.Byte():
                # cvtsi2sd has no byte form; sign-extend to int32 first then convert.
                r10d = asm.Register(asm.Registers.R10, asm.Width.DWORD)
                xmm14 = asm.Register(asm.Registers.XMM14, asm.Width.QWORD)
                return [
                    asm.Movsx(
                        src_type=asm.Byte(),
                        dst_type=asm.Longword(),
                        src=_translate_val(src, None, symbol_table),
                        dst=r10d,
                    ),
                    asm.Cvtsi2sd(src=r10d, dst=xmm14, assembly_type=asm.Longword()),
                    asm.Mov(src=xmm14, dst=_translate_val(dst, None, symbol_table), assembly_type=asm.Double()),
                ]
            # assembly_type controls the source integer width (l = int, q = long).
            return [
                asm.Cvtsi2sd(
                    src=_translate_val(src, constants, symbol_table),
                    dst=_translate_val(dst, None, symbol_table),
                    assembly_type=at,
                )
            ]
        case tac.UIntToDouble(src=src, dst=dst):
            return _translate_uint_to_double(src, dst, symbol_table)
        case tac.DoubleToUInt(src=src, dst=dst):
            return _translate_double_to_uint(src, dst, symbol_table, constants)
        case tac.Load(src_ptr=src_ptr, dst=dst):
            at = _asm_type(dst, symbol_table)
            ax = asm.Register(asm.Registers.AX, asm.Width.QWORD)
            dst_asm = _translate_val(dst, constants, symbol_table)
            return [
                asm.Mov(src=_translate_val(src_ptr, constants, symbol_table), dst=ax, assembly_type=asm.Quadword()),
                asm.Mov(src=asm.Memory(reg=ax, offset=0), dst=dst_asm, assembly_type=at),
            ]
        case tac.Store(src=src, dst_ptr=dst_ptr):
            at = _asm_type(src, symbol_table)
            ax = asm.Register(asm.Registers.AX, asm.Width.QWORD)
            src_asm = _translate_val(src, constants, symbol_table)
            return [
                asm.Mov(src=_translate_val(dst_ptr, constants, symbol_table), dst=ax, assembly_type=asm.Quadword()),
                asm.Mov(src=src_asm, dst=asm.Memory(reg=ax, offset=0), assembly_type=at),
            ]
        case tac.GetAddress(src=src, dst=dst):
            src_asm = _translate_val(src, constants, symbol_table)
            if isinstance(src_asm, (asm.Imm, asm.Register)):
                raise CodeGenerationError(
                    f"GetAddress requires an addressable memory operand, got {type(src_asm).__name__}"
                )
            return [asm.Lea(src=src_asm, dst=_translate_val(dst, constants, symbol_table))]
        case tac.CopyToOffset(src=src, dst=dst_name, offset=offset):
            at = _asm_type(src, symbol_table)
            src_asm = _translate_val(src, constants, symbol_table)
            return [asm.Mov(src=src_asm, dst=asm.PseudoMem(dst_name, offset), assembly_type=at)]
        case tac.AddPtr(ptr=ptr, index=index, scale=scale, dst=dst):
            r10 = asm.Register(asm.Registers.R10, asm.Width.QWORD)
            r11 = asm.Register(asm.Registers.R11, asm.Width.QWORD)
            ptr_asm = _translate_val(ptr, constants, symbol_table)
            index_asm = _translate_val(index, constants, symbol_table)
            dst_asm = _translate_val(dst, constants, symbol_table)
            instrs: list[asm.Instruction] = [
                asm.Mov(src=ptr_asm, dst=r11, assembly_type=asm.Quadword()),
                asm.Mov(src=index_asm, dst=r10, assembly_type=asm.Quadword()),
            ]
            if scale in {1, 2, 4, 8}:
                instrs.append(asm.Lea(src=asm.Indexed(r11, r10, scale), dst=dst_asm))
            else:
                instrs.append(
                    asm.Binary(
                        operator=asm.Mult(),
                        src1=r10,
                        src2=asm.Imm(scale),
                        dst=r10,
                        assembly_type=asm.Quadword(),
                    )
                )
                instrs.append(asm.Lea(src=asm.Indexed(r11, r10, 1), dst=dst_asm))
            return instrs
        case _:
            raise CodeGenerationError(f"unsupported TAC instruction: {type(instr).__name__}")


def _translate_uint_to_double(
    src: tac.Val,
    dst: tac.Val,
    symbol_table: SymbolTable | None = None,
) -> list[asm.Instruction]:
    """Lower a ``UIntToDouble`` TAC instruction to assembly.

    For an 8-bit unsigned source (``unsigned char``): all values (0..255) fit
    in a signed ``int``; zero-extend to ``Longword`` then convert with ``cvtsi2sdl``.

    For a 32-bit unsigned source: a plain ``movl`` zero-extends to 64 bits,
    after which ``cvtsi2sdq`` converts the non-negative 64-bit value exactly.

    For a 64-bit unsigned source: values < 2^63 are converted directly via
    ``cvtsi2sdq``; values >= 2^63 are halved (preserving the LSB to avoid
    double-rounding), converted, then doubled back.

    Args:
        src: The unsigned-integer TAC source value.
        dst: The double TAC destination value.
        symbol_table: Used to derive the assembly type of the source.

    Returns:
        A list of ``asm.Instruction`` nodes implementing the conversion.
    """
    src_at = _asm_type(src, symbol_table)
    src_asm = _translate_val(src)
    dst_asm = _translate_val(dst)

    r10d = asm.Register(asm.Registers.R10, asm.Width.DWORD)
    r10q = asm.Register(asm.Registers.R10, asm.Width.QWORD)
    r11q = asm.Register(asm.Registers.R11, asm.Width.QWORD)
    rcxq = asm.Register(asm.Registers.CX, asm.Width.QWORD)
    xmm14 = asm.Register(asm.Registers.XMM14, asm.Width.QWORD)

    if src_at == asm.Byte():
        # All unsigned char values (0..255) fit in a signed int32; zero-extend to Longword
        # so cvtsi2sdl sees a non-negative 32-bit value and produces the exact double.
        return [
            asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=asm.Longword(), src=src_asm, dst=r10d),
            asm.Cvtsi2sd(src=r10d, dst=xmm14, assembly_type=asm.Longword()),
            asm.Mov(src=xmm14, dst=dst_asm, assembly_type=asm.Double()),
        ]

    if src_at == asm.Longword():
        # Zero-extend 32-bit to 64-bit (movl clears the upper 32 bits on x86-64),
        # then convert the non-negative 64-bit value with cvtsi2sdq.
        return [
            asm.Mov(src=src_asm, dst=r10d, assembly_type=asm.Longword()),
            asm.Cvtsi2sd(src=r10q, dst=xmm14, assembly_type=asm.Quadword()),
            asm.Mov(src=xmm14, dst=dst_asm, assembly_type=asm.Double()),
        ]

    # 64-bit unsigned long: branch on whether the high bit is set.
    uid = _fresh_uid()
    simple_label = f"uint_to_dbl_simple.{uid}"
    done_label = f"uint_to_dbl_done.{uid}"

    return [
        asm.Mov(src=src_asm, dst=r11q, assembly_type=asm.Quadword()),
        # jge (SF == OF after cmpq $0) ↔ bit 63 clear ↔ value < 2^63
        asm.Cmp(op1=asm.Imm(0), op2=r11q, assembly_type=asm.Quadword()),
        asm.JmpCC(asm.CondCodes.GE, simple_label),
        # Complex path: src >= 2^63 — halve, convert, double back.
        asm.Mov(src=r11q, dst=r10q, assembly_type=asm.Quadword()),
        asm.Mov(src=asm.Imm(1), dst=rcxq, assembly_type=asm.Quadword()),
        asm.Shr(operand=r10q, assembly_type=asm.Quadword()),
        # r11 &= 1  — preserve the LSB to avoid double-rounding
        asm.Binary(operator=asm.And(), src1=r11q, src2=asm.Imm(1), dst=r11q, assembly_type=asm.Quadword()),
        # r10 |= r11  — (src >> 1) | (src & 1)
        asm.Binary(operator=asm.Or(), src1=r10q, src2=r11q, dst=r10q, assembly_type=asm.Quadword()),
        asm.Cvtsi2sd(src=r10q, dst=xmm14, assembly_type=asm.Quadword()),
        asm.Binary(operator=asm.AddDouble(), src1=xmm14, src2=xmm14, dst=xmm14, assembly_type=asm.Double()),
        asm.Mov(src=xmm14, dst=dst_asm, assembly_type=asm.Double()),
        asm.Jmp(done_label),
        # Simple path: 0 <= src < 2^63, value fits in signed int64.
        asm.Label(simple_label),
        asm.Cvtsi2sd(src=r11q, dst=xmm14, assembly_type=asm.Quadword()),
        asm.Mov(src=xmm14, dst=dst_asm, assembly_type=asm.Double()),
        asm.Label(done_label),
    ]


def _translate_double_to_uint(
    src: tac.Val,
    dst: tac.Val,
    symbol_table: SymbolTable | None = None,
    constants: list[asm.TopLevel] | None = None,
) -> list[asm.Instruction]:
    """Lower a ``DoubleToUInt`` TAC instruction to assembly.

    For a 32-bit unsigned destination: convert via ``cvttsd2siq`` to a 64-bit
    signed integer (which can hold all unsigned 32-bit values), then truncate
    with ``movl`` to produce the correct 32-bit unsigned bit pattern.

    For a 64-bit unsigned destination: values < 2^63 are converted directly;
    values >= 2^63 subtract the threshold (2^63 as a double, stored in
    ``.rodata``), convert, then add 2^63 back as an unsigned 64-bit addition.

    Args:
        src: The double TAC source value.
        dst: The unsigned-integer TAC destination value.
        symbol_table: Used to derive the assembly type of the destination.
        constants: Mutable list to which the 2^63 threshold ``StaticConstant``
            is appended (deduplicated by name) when needed.

    Returns:
        A list of ``asm.Instruction`` nodes implementing the conversion.
    """
    dst_at = _asm_type(dst, symbol_table)
    src_asm = _translate_val(src, constants)
    dst_asm = _translate_val(dst, constants)

    r10q = asm.Register(asm.Registers.R10, asm.Width.QWORD)
    r10d = asm.Register(asm.Registers.R10, asm.Width.DWORD)
    r10b = asm.Register(asm.Registers.R10, asm.Width.BYTE)

    if dst_at == asm.Byte():
        # All unsigned char values (0..255) fit in a signed int32; convert via cvttsd2sil
        # then truncate to the low byte with movb.
        return [
            asm.Cvttsd2si(src=src_asm, dst=r10d, assembly_type=asm.Longword()),
            asm.Mov(src=r10b, dst=dst_asm, assembly_type=asm.Byte()),
        ]

    if dst_at == asm.Longword():
        # All unsigned int values (0..2^32-1) fit in a signed int64, so convert via
        # cvttsd2siq then truncate the lower 32 bits with movl.
        return [
            asm.Cvttsd2si(src=src_asm, dst=r10q, assembly_type=asm.Quadword()),
            asm.Mov(src=r10d, dst=dst_asm, assembly_type=asm.Longword()),
        ]

    # 64-bit unsigned long destination.
    # Ensure the threshold constant is present in .rodata (emitted once per program).
    _THRESHOLD_LABEL = "double_to_ulong_threshold"
    if constants is not None and not any(
        isinstance(c, asm.StaticConstant) and c.identifier == _THRESHOLD_LABEL for c in constants
    ):
        constants.append(
            asm.StaticConstant(
                identifier=_THRESHOLD_LABEL, alignment=8, init=asm.DoubleInit(_DOUBLE_TO_ULONG_THRESHOLD)
            )
        )

    uid = _fresh_uid()
    large_label = f"dbl_to_ulong_large.{uid}"
    done_label = f"dbl_to_ulong_done.{uid}"

    threshold_op = asm.Data(_THRESHOLD_LABEL)
    r11q = asm.Register(asm.Registers.R11, asm.Width.QWORD)
    xmm14 = asm.Register(asm.Registers.XMM14, asm.Width.QWORD)

    return [
        # Load src into xmm14; ucomisd requires op2 to be an XMM register.
        asm.Mov(src=src_asm, dst=xmm14, assembly_type=asm.Double()),
        # ucomisd threshold(%rip), xmm14 — AE if xmm14 >= threshold (= 2^63)
        asm.Cmp(op1=threshold_op, op2=xmm14, assembly_type=asm.Double()),
        asm.JmpCC(asm.CondCodes.AE, large_label),
        # Simple path: src < 2^63, fits in signed int64.
        asm.Cvttsd2si(src=xmm14, dst=dst_asm, assembly_type=asm.Quadword()),
        asm.Jmp(done_label),
        # Large path: src in [2^63, 2^64).
        asm.Label(large_label),
        # xmm14 = src - 2^63  (now in [0, 2^63), fits in signed int64)
        asm.Binary(operator=asm.SubDouble(), src1=xmm14, src2=threshold_op, dst=xmm14, assembly_type=asm.Double()),
        asm.Cvttsd2si(src=xmm14, dst=r10q, assembly_type=asm.Quadword()),
        # Add 2^63 back as an unsigned 64-bit operation.
        # INT64_MIN == -2^63 in two's complement == +2^63 when treated as unsigned.
        asm.Mov(src=asm.Imm(-(1 << 63)), dst=r11q, assembly_type=asm.Quadword()),
        asm.Binary(operator=asm.Add(), src1=r10q, src2=r11q, dst=r10q, assembly_type=asm.Quadword()),
        asm.Mov(src=r10q, dst=dst_asm, assembly_type=asm.Quadword()),
        asm.Label(done_label),
    ]


def _translate_funcall(
    fun_val: tac.Val,
    args: list[tac.Val],
    dst: asm.Operand,
    symbol_table: SymbolTable | None = None,
    constants: list[asm.TopLevel] | None = None,
) -> list[asm.Instruction]:
    """Lower a ``tac.FunCall`` to the System V AMD64 ABI calling sequence.

    Integer and double arguments are allocated to separate register classes:
    up to six integers go to DI/SI/DX/CX/R8/R9, and up to eight doubles go to
    XMM0-XMM7.  Arguments beyond each class's limit are pushed right-to-left.
    An 8-byte padding ``AllocateStack`` is inserted when the total number of
    stack-passed arguments is odd, to keep the stack 16-byte aligned at ``call``.

    After the call the return value is read from ``%xmm0`` (double) or
    ``%eax``/``%rax`` (integer) into ``dst``.

    Args:
        fun_val: The TAC value identifying the callable.  Must be a ``Var``.  When its
            symbol table entry carries ``FunAttr`` a direct ``call`` is emitted; otherwise
            the pointer value is loaded into ``%r10`` and an indirect ``call *%r10`` is used.
        args: The ordered list of TAC argument values.
        dst: The assembly operand that receives the return value.
        symbol_table: Used to derive assembly types for variable arguments.
        constants: Accumulator for ``StaticConstant`` entries emitted for double literals.

    Returns:
        A list of ``asm.Instruction`` nodes implementing the call site.

    Raises:
        CodeGenerationError: If ``fun_val`` is not a ``Var``.
    """
    if not isinstance(fun_val, tac.Var):
        raise CodeGenerationError("fun_val must be a Var.")
    name = fun_val.value
    is_direct = symbol_table is not None and name in symbol_table and isinstance(symbol_table[name].attrs, FunAttr)
    instrs: list[asm.Instruction] = []

    # Classify each argument and assign it to a register or the stack.
    int_reg_idx = 0
    xmm_reg_idx = 0
    int_reg_args: list[tuple[tac.Val, int]] = []
    dbl_reg_args: list[tuple[tac.Val, int]] = []
    stack_args: list[tac.Val] = []

    for arg in args:
        at = _asm_type(arg, symbol_table)
        if at == asm.Double():
            if xmm_reg_idx < len(_XMM_REGS):
                dbl_reg_args.append((arg, xmm_reg_idx))
                xmm_reg_idx += 1
            else:
                stack_args.append(arg)
        else:
            if int_reg_idx < len(_ABI_REGS):
                int_reg_args.append((arg, int_reg_idx))
                int_reg_idx += 1
            else:
                stack_args.append(arg)

    # Odd number of stack args would misalign the stack before `call`.
    needs_padding = len(stack_args) % 2 != 0
    if needs_padding:
        instrs.append(asm.AllocateStack(8))

    # Push stack args right-to-left so the first stack arg is closest to %rbp
    # in the callee's frame (at 16(%rbp)).
    for arg in reversed(stack_args):
        at = _asm_type(arg, symbol_table)
        if at == asm.Byte():
            # `pushq` has no 1-byte form; extend the char into %r10d (zero-extends %r10 to
            # 64 bits) then push the full 8-byte register so the ABI slot is well-formed.
            r10d = asm.Register(asm.Registers.R10, asm.Width.DWORD)
            r10q = asm.Register(asm.Registers.R10, asm.Width.QWORD)
            src_asm = _translate_val(arg, constants)
            if _is_unsigned(arg, symbol_table):
                instrs.append(asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=asm.Longword(), src=src_asm, dst=r10d))
            else:
                instrs.append(asm.Movsx(src_type=asm.Byte(), dst_type=asm.Longword(), src=src_asm, dst=r10d))
            instrs.append(asm.Push(operand=r10q, assembly_type=asm.Quadword()))
        else:
            instrs.append(asm.Push(operand=_translate_val(arg, constants), assembly_type=at))

    # Load integer args into GP registers.
    # The System V AMD64 ABI requires sub-32-bit integer arguments to be sign- or zero-extended
    # to at least 32 bits before placement in the argument register.  Char arguments are extended
    # to Longword so that Clang-compiled callees can rely on the upper bytes being well-defined.
    for arg, i in int_reg_args:
        at = _asm_type(arg, symbol_table)
        if at == asm.Byte():
            reg_dword = asm.Register(_ABI_REGS[i], asm.Width.DWORD)
            src_asm = _translate_val(arg, constants)
            if _is_unsigned(arg, symbol_table):
                instrs.append(
                    asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=asm.Longword(), src=src_asm, dst=reg_dword)
                )
            else:
                instrs.append(asm.Movsx(src_type=asm.Byte(), dst_type=asm.Longword(), src=src_asm, dst=reg_dword))
        else:
            instrs.append(
                asm.Mov(
                    src=_translate_val(arg, constants),
                    dst=asm.Register(_ABI_REGS[i], _reg_width(at)),
                    assembly_type=at,
                )
            )

    # Load double args into XMM registers.
    for arg, i in dbl_reg_args:
        instrs.append(
            asm.Mov(
                src=_translate_val(arg, constants),
                dst=asm.Register(_XMM_REGS[i], asm.Width.QWORD),
                assembly_type=asm.Double(),
            )
        )

    if is_direct:
        instrs.append(asm.Call(name))
    else:
        r10q = asm.Register(asm.Registers.R10, asm.Width.QWORD)
        instrs.append(asm.Mov(src=_translate_val(fun_val, constants), dst=r10q, assembly_type=asm.Quadword()))
        instrs.append(asm.IndirectCall(target=r10q))

    # Reclaim stack space used for stack-passed args and any alignment padding.
    bytes_to_free = 8 * len(stack_args) + (8 if needs_padding else 0)
    if bytes_to_free:
        instrs.append(asm.DeallocateStack(bytes_to_free))

    # Return value: doubles in %xmm0, integers in %eax/%rax.
    at = _asm_type(tac.Var(dst.name) if isinstance(dst, asm.Pseudo) else tac.Constant(ConstInt(0)), symbol_table)
    if at == asm.Double():
        instrs.append(asm.Mov(src=asm.Register(asm.Registers.XMM0, asm.Width.QWORD), dst=dst, assembly_type=at))
    else:
        instrs.append(asm.Mov(src=asm.Register(asm.Registers.AX, _reg_width(at)), dst=dst, assembly_type=at))

    return instrs


def _translate_val(
    val: tac.Val,
    constants: list[asm.TopLevel] | None = None,
    symbol_table: SymbolTable | None = None,
) -> asm.Operand:
    """Translate a TAC value to an assembly operand.

    Integer and long constants become ``Imm`` values.  Double constants are
    allocated as read-only ``StaticConstant`` entries in ``.rodata`` and
    referenced via a ``Data`` (RIP-relative) operand.  Scalar variables become
    ``Pseudo`` operands; array-typed variables become ``PseudoMem`` at offset 0
    so that the aggregate's full memory region is designated.

    Args:
        val: The TAC value to translate.
        constants: Accumulator for ``StaticConstant`` entries; required when
            ``val`` may be a ``ConstDouble``.
        symbol_table: The program-wide symbol table; used to detect array-typed
            variables.  When ``None`` all variables are treated as scalars.

    Returns:
        An ``asm.Operand`` equivalent to the TAC value.

    Raises:
        CodeGenerationError: If the value variant is not supported.
    """
    match val:
        case tac.Constant(value=ConstDouble(val=v)):
            # Double constants cannot be encoded as immediates in SSE instructions.
            # Allocate each unique value once in .rodata, keyed by its IEEE 754 bits.
            bits = asm.DoubleInit(v).as_bits()
            label = f"double_const.{bits}"
            if constants is not None and not any(
                isinstance(c, asm.StaticConstant) and c.identifier == label for c in constants
            ):
                constants.append(asm.StaticConstant(identifier=label, alignment=8, init=asm.DoubleInit(v)))
            return asm.Data(label)
        case tac.Constant(value=v):
            return asm.Imm(value=v.val)
        case tac.Var(value=name):
            if symbol_table is not None and name in symbol_table and isinstance(symbol_table[name].t, ArrayType):
                return asm.PseudoMem(name, 0)
            return asm.Pseudo(name=name)
        case _:
            raise CodeGenerationError(f"unsupported TAC value: {type(val).__name__}")


def _translate_binary_op(op: tac.BinaryOperator) -> asm.BinaryOperator:
    """Translate a TAC binary operator (Add/Sub/Mul/bitwise) to its assembly equivalent.

    Divide and Remainder are intentionally absent — they are handled at the
    instruction level by emitting a ``Cdq``/``Cqo`` + ``Idiv`` sequence.

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


def _translate_binary_op_double(op: tac.BinaryOperator) -> asm.BinaryOperator:
    """Translate a TAC arithmetic operator to its SSE scalar-double equivalent.

    Args:
        op: The TAC binary operator node (Add, Subtract, or Multiply).

    Returns:
        The corresponding scalar-double ``asm.BinaryOperator``.

    Raises:
        CodeGenerationError: If the operator has no scalar-double equivalent.
    """
    match op:
        case tac.Add():
            return asm.AddDouble()
        case tac.Subtract():
            return asm.SubDouble()
        case tac.Multiply():
            return asm.MultDouble()
        case _:
            raise CodeGenerationError(f"unsupported TAC double binary operator for asm.Binary: {type(op).__name__}")


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


def _translate_relational_op(op: tac.BinaryOperator, unsigned: bool = False) -> asm.CondCodes:
    """Translate a TAC relational operator to its assembly condition code.

    Signed comparisons map to ``L``/``LE``/``G``/``GE``; unsigned comparisons map to
    ``B``/``BE``/``A``/``AE``.  ``Equal`` and ``NotEqual`` are the same for both.

    Args:
        op: The TAC binary operator node.
        unsigned: When ``True``, select the unsigned (above/below) condition codes instead of the
            signed (less/greater) ones.

    Returns:
        The corresponding ``asm.CondCodes`` value.

    Raises:
        CodeGenerationError: If the operator variant is not supported.
    """
    match op:
        case tac.Equal():
            return asm.CondCodes.E
        case tac.NotEqual():
            return asm.CondCodes.NE
        case tac.LessThan():
            return asm.CondCodes.B if unsigned else asm.CondCodes.L
        case tac.LessOrEqual():
            return asm.CondCodes.BE if unsigned else asm.CondCodes.LE
        case tac.GreaterThan():
            return asm.CondCodes.A if unsigned else asm.CondCodes.G
        case tac.GreaterOrEqual():
            return asm.CondCodes.AE if unsigned else asm.CondCodes.GE
        case _:
            raise CodeGenerationError(f"unsupported TAC relational operator for asm.Binary: {type(op).__name__}")


def _translate_double_comparison(
    op: tac.BinaryOperator,
    src1_asm: asm.Operand,
    src2_asm: asm.Operand,
    dst_asm: asm.Operand,
) -> list[asm.Instruction]:
    """Lower a double-precision comparison to assembly, with correct NaN handling.

    ``ucomisd`` sets ZF=1, CF=1, PF=1 when either operand is NaN (unordered result).
    For ``==``, ``<``, and ``<=`` the naïve ``SetCC`` would incorrectly return 1 for NaN because
    ``sete`` reads ZF and ``setb``/``setbe`` read CF, all of which are 1.  For ``!=`` the
    naïve ``setne`` would incorrectly return 0 because ZF=1.  ``>`` and ``>=`` are naturally
    correct: ``seta`` (CF=0 AND ZF=0) and ``setae`` (CF=0) both return 0 when CF=1.

    The fix uses a conditional jump on PF (the parity flag, set to 1 only for NaN):
    * For ``==``, ``<``, ``<=``: pre-load 0 into ``dst``, jump over ``SetCC`` if PF=1
      so the result stays 0 (all orderings vs. NaN are false).
    * For ``!=``: pre-load 1 into ``dst``, jump over ``setne`` if PF=1
      so the result stays 1 (NaN != anything is always true).

    Args:
        op: The TAC relational operator (Equal, NotEqual, LessThan, LessOrEqual, GreaterThan,
            or GreaterOrEqual).
        src1_asm: The left operand in assembly form.
        src2_asm: The right operand in assembly form.
        dst_asm: The destination operand in assembly form.

    Returns:
        A list of ``asm.Instruction`` nodes implementing the NaN-aware comparison.

    Raises:
        CodeGenerationError: If ``op`` is not a supported relational operator.
    """
    uid = _fresh_uid()
    nan_label = f"nan_cmp.{uid}"
    cmp = asm.Cmp(op1=src2_asm, op2=src1_asm, assembly_type=asm.Double())
    match op:
        case tac.Equal():
            # sete returns 1 for NaN (ZF=1); jump over it so dst stays 0.
            return [
                cmp,
                asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                asm.JmpCC(asm.CondCodes.P, nan_label),
                asm.SetCC(asm.CondCodes.E, dst_asm),
                asm.Label(nan_label),
            ]
        case tac.NotEqual():
            # setne returns 0 for NaN (ZF=1); pre-load 1 and jump over setne so dst stays 1.
            return [
                cmp,
                asm.Mov(src=asm.Imm(1), dst=dst_asm, assembly_type=asm.Longword()),
                asm.JmpCC(asm.CondCodes.P, nan_label),
                asm.SetCC(asm.CondCodes.NE, dst_asm),
                asm.Label(nan_label),
            ]
        case tac.LessThan():
            # setb returns 1 for NaN (CF=1); jump over it so dst stays 0.
            return [
                cmp,
                asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                asm.JmpCC(asm.CondCodes.P, nan_label),
                asm.SetCC(asm.CondCodes.B, dst_asm),
                asm.Label(nan_label),
            ]
        case tac.LessOrEqual():
            # setbe returns 1 for NaN (CF=1 or ZF=1); jump over it so dst stays 0.
            return [
                cmp,
                asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                asm.JmpCC(asm.CondCodes.P, nan_label),
                asm.SetCC(asm.CondCodes.BE, dst_asm),
                asm.Label(nan_label),
            ]
        case tac.GreaterThan():
            # seta (CF=0 AND ZF=0) naturally returns 0 for NaN (CF=1); no extra jump needed.
            return [
                cmp,
                asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                asm.SetCC(asm.CondCodes.A, dst_asm),
            ]
        case tac.GreaterOrEqual():
            # setae (CF=0) naturally returns 0 for NaN (CF=1); no extra jump needed.
            return [
                cmp,
                asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                asm.SetCC(asm.CondCodes.AE, dst_asm),
            ]
        case _:
            raise CodeGenerationError(f"unsupported TAC relational operator for double comparison: {type(op).__name__}")


# ==============================================================================
# Pass 2 — Pseudo-register replacement
# ==============================================================================


def _replace_pseudos(  # noqa: C901 - ignore complexity warning
    fn: asm.Function, static_names: frozenset[str], symbol_table: SymbolTable | None = None
) -> tuple[asm.Function, int]:
    """Replace every ``Pseudo`` operand with a concrete operand.

    Pseudo operands whose names appear in ``static_names`` become RIP-relative
    ``Data`` operands.  All other pseudos are assigned a unique slot in the
    stack frame (1 byte for ``char``/``schar``/``uchar``, 4 bytes for ``int``,
    8 bytes for ``long``), starting just below ``%rbp`` and growing downward.
    The total number of stack bytes used is returned so the caller can emit an
    ``AllocateStack``.

    Args:
        fn: The assembly function whose pseudo operands are to be resolved.
        static_names: Names of all static/global variables in the program;
            pseudos with these names are mapped to ``asm.Data`` rather than
            a local stack slot.
        symbol_table: Used to determine the storage size (4 or 8 bytes) for
            each pseudo variable.

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
        """Map Pseudo/PseudoMem to concrete operands; pass through everything else."""
        if isinstance(operand, asm.PseudoMem):
            name, elem_offset = operand.name, operand.offset
            if name in static_names:
                return asm.Data(name=name)
            if name not in pseudo_offsets:
                if symbol_table is not None and name in symbol_table:
                    t = symbol_table[name].t
                    size = _type_size(t)
                    alignment = _static_alignment(t)
                else:
                    size, alignment = 8, 8
                next_offset[0] -= size
                next_offset[0] = (next_offset[0] // alignment) * alignment
                pseudo_offsets[name] = next_offset[0]
            return asm.Memory(
                reg=asm.Register(asm.Registers.BP, asm.Width.QWORD),
                offset=pseudo_offsets[name] + elem_offset,
            )
        if not isinstance(operand, asm.Pseudo):
            return operand
        if operand.name in static_names:
            return asm.Data(name=operand.name)
        if operand.name not in pseudo_offsets:
            size = 4
            if symbol_table is not None and operand.name in symbol_table:
                t = symbol_table[operand.name].t
                if isinstance(t, (LongType, ULongType, DoubleType, Pointer)):
                    size = 8
                elif isinstance(t, (Char, UChar, SChar)):
                    size = 1
            next_offset[0] -= size
            pseudo_offsets[operand.name] = next_offset[0]
        return asm.Memory(reg=asm.Register(asm.Registers.BP, asm.Width.QWORD), offset=pseudo_offsets[operand.name])

    new_instructions: list[asm.Instruction] = []
    for instr in fn.instructions:
        match instr:
            case asm.Mov(assembly_type=at, src=src, dst=dst):
                new_instructions.append(asm.Mov(src=resolve(src), dst=resolve(dst), assembly_type=at))
            case asm.Movsx(src_type=st, dst_type=dt, src=src, dst=dst):
                new_instructions.append(asm.Movsx(src_type=st, dst_type=dt, src=resolve(src), dst=resolve(dst)))
            case asm.MoveZeroExtend(src_type=st, dst_type=dt, src=src, dst=dst):
                new_instructions.append(
                    asm.MoveZeroExtend(src_type=st, dst_type=dt, src=resolve(src), dst=resolve(dst))
                )
            case asm.Unary(assembly_type=at, operator=op, operand=operand):
                new_instructions.append(asm.Unary(operator=op, operand=resolve(operand), assembly_type=at))
            case asm.Binary(assembly_type=at, operator=op, src1=src1, src2=src2, dst=dst):
                new_instructions.append(
                    asm.Binary(operator=op, src1=resolve(src1), src2=resolve(src2), dst=dst, assembly_type=at)
                )
            case asm.Idiv(assembly_type=at, operand=operand):
                new_instructions.append(asm.Idiv(operand=resolve(operand), assembly_type=at))
            case asm.Div(assembly_type=at, operand=operand):
                new_instructions.append(asm.Div(operand=resolve(operand), assembly_type=at))
            case (
                asm.Shl(assembly_type=at, operand=operand)
                | asm.Sar(assembly_type=at, operand=operand)
                | asm.Shr(assembly_type=at, operand=operand)
            ):
                new_instructions.append(type(instr)(operand=resolve(operand), assembly_type=at))
            case asm.Cmp(assembly_type=at, op1=op1, op2=op2):
                new_instructions.append(asm.Cmp(op1=resolve(op1), op2=resolve(op2), assembly_type=at))
            case asm.SetCC(cond_code=cc, operand=operand):
                new_instructions.append(asm.SetCC(cond_code=cc, operand=resolve(operand)))
            case asm.Push(assembly_type=at, operand=operand):
                new_instructions.append(asm.Push(operand=resolve(operand), assembly_type=at))
            case asm.Cvttsd2si(assembly_type=at, src=src, dst=dst) | asm.Cvtsi2sd(assembly_type=at, src=src, dst=dst):
                new_instructions.append(type(instr)(src=resolve(src), dst=resolve(dst), assembly_type=at))
            case asm.Lea(src=src, dst=dst):
                new_instructions.append(asm.Lea(src=resolve(src), dst=resolve(dst)))
            case _:
                new_instructions.append(instr)

    stack_size = -next_offset[0]
    return asm.Function(
        name=fn.name, globl=fn.globl, params=fn.params, instructions=tuple(new_instructions)
    ), stack_size


# ==============================================================================
# Pass 3 — Instruction fixup
# ==============================================================================


def _fix_movsx(
    src_type: asm.AssemblyType,
    dst_type: asm.AssemblyType,
    src: asm.Operand,
    dst: asm.Operand,
    fixed: list[asm.Instruction],
    is_mem: object,
) -> None:
    """Emit a legal sign-extending move sequence into ``fixed``, routing around illegal operand combinations.

    ``movslq``/``movsbl``/``movsbq`` cannot accept an immediate source and their destination must be
    a register.  This helper handles both constraints:

    * If ``src`` is an immediate: load it into a scratch register of the correct source width first.
    * If ``dst`` is memory: sign-extend into the appropriately-sized ``%r11``, then store.

    Args:
        src_type: Assembly type of the source operand (``Byte`` or ``Longword``).
        dst_type: Assembly type of the destination operand (``Longword`` or ``Quadword``).
        src: The raw source operand (may be ``Imm``, register, or memory).
        dst: The raw destination operand (may be a register or memory).
        fixed: The output instruction list to append into.
        is_mem: Callable that returns ``True`` for ``Stack``/``Data`` operands.
    """
    r10_src = asm.Register(asm.Registers.R10, _reg_width(src_type))
    r11_dst = asm.Register(asm.Registers.R11, _reg_width(dst_type))
    if isinstance(src, asm.Imm):
        fixed.append(asm.Mov(src=src, dst=r10_src, assembly_type=src_type))
        if is_mem(dst):  # type: ignore[operator]
            fixed.append(asm.Movsx(src_type=src_type, dst_type=dst_type, src=r10_src, dst=r11_dst))
            fixed.append(asm.Mov(src=r11_dst, dst=dst, assembly_type=dst_type))
        else:
            fixed.append(asm.Movsx(src_type=src_type, dst_type=dst_type, src=r10_src, dst=dst))
    elif is_mem(dst):  # type: ignore[operator]
        fixed.append(asm.Movsx(src_type=src_type, dst_type=dst_type, src=src, dst=r11_dst))
        fixed.append(asm.Mov(src=r11_dst, dst=dst, assembly_type=dst_type))
    else:
        fixed.append(asm.Movsx(src_type=src_type, dst_type=dst_type, src=src, dst=dst))


def _fix_instructions(fn: asm.Function) -> asm.Function:  # noqa: C901 - ignore complexity warning
    """Rewrite instructions that are invalid in x86-64 assembly.

    Handles the following cases:

    * ``Mov(Quadword, Imm(v), mem)`` where ``v`` exceeds the signed 32-bit range —
      x86-64 has no encoding for a 64-bit immediate to memory; load into ``%r10``
      then store.
    * ``Mov(Longword, Imm(v), any)`` where ``v`` exceeds the signed 32-bit range —
      arises from ``Truncate``; the immediate is pre-masked to its lower 32 bits so
      the assembler does not warn about truncation.
    * ``Mov(mem, mem)`` — both operands cannot be memory; route ``src`` through
      ``%r10``/``%r10d`` of the appropriate width.
    * ``Idiv(Imm)`` — the divisor cannot be an immediate; route through
      ``%r11``/``%r11d``.
    * ``Cmp(op1, Imm)`` — ``op2`` (the second printed operand) cannot be an
      immediate; route through ``%r11``/``%r11d``.
    * ``Cmp(Quadword, Imm(v), op2)`` where ``v`` exceeds the signed 32-bit range —
      x86-64 ``cmpq`` can only encode 32-bit sign-extended immediates; load into
      ``%r10`` then compare.
    * ``Cmp(mem, op2)`` — ``op1`` cannot be memory when ``op2`` is also memory;
      route ``op1`` through ``%r10``/``%r10d``.
    * ``Movsx(src, mem)`` — the destination of ``movslq`` must be a register;
      route through ``%r11`` then move to ``dst``.
    * ``MoveZeroExtend(Longword, …)`` — ``movl`` to a 32-bit register already
      zero-extends; replaced with ``Mov(Longword)``, with a ``%r10d`` scratch hop
      when both operands are memory.
    * ``MoveZeroExtend(Byte, …, Imm, …)`` — ``movzbl``/``movzbq`` cannot encode
      an immediate source; load into ``%r10b`` first, then zero-extend (with an
      additional ``%r11`` store hop when ``dst`` is also memory).
    * ``MoveZeroExtend(Byte, …, src, mem)`` — destination must be a register;
      route through ``%r11`` of the destination width then store.
    * ``Mov(Byte, Imm(v), …)`` where ``v`` does not fit in a single byte — GAS
      rejects byte immediates outside −128…255; reduce modulo 256.
    * ``Binary(Quadword, op, src1, Imm(v), dst)`` where ``v`` exceeds the signed
      32-bit range — ``addq``/``subq``/``imulq`` cannot encode a 64-bit
      immediate; load ``src2`` into ``%r10`` first.
    * ``Cmp(Quadword, Imm(v), Imm)`` where ``v`` exceeds the signed 32-bit
      range and ``op2`` is also an immediate — the op2-is-Imm fixup would fire
      first and produce a new ``Cmp(Imm(large), reg)`` that is never
      re-processed; handle both operands upfront instead.
    * ``Push(Quadword, Imm(v))`` where ``v`` exceeds the signed 32-bit range —
      ``pushq`` only accepts 8-bit or 32-bit sign-extended immediates; load
      into ``%r10`` first.
    * ``Push(mem)`` — ``pushq`` cannot address a memory slot directly; load
      into ``%eax``/``%rax`` first.
    * ``Push(Double, Register)`` — ``pushq`` cannot encode an XMM register; expand
      to ``subq $8, %rsp`` followed by ``movsd xmm, (%rsp)``.
    * ``Lea(src, mem)`` — ``leaq`` destination must be a register; route through
      ``%r11`` then store.

    Args:
        fn: The assembly function to fix up.

    Returns:
        An ``asm.Function`` with all invalid instructions rewritten.
    """

    def _is_mem(op: asm.Operand) -> bool:
        """Return True when ``op`` is a memory address (Memory or Data)."""
        return isinstance(op, (asm.Memory, asm.Data))

    def _fits_in_int32(v: int) -> bool:
        """Return True when ``v`` is representable as a signed 32-bit integer."""
        return -(1 << 31) <= v <= (1 << 31) - 1

    def _fits_in_byte(v: int) -> bool:
        """Return True when ``v`` fits in a single byte (−128…255)."""
        return -128 <= v <= 255

    fixed: list[asm.Instruction] = []
    for instr in fn.instructions:
        match instr:
            case asm.Mov(assembly_type=asm.Quadword(), src=asm.Imm(value=v), dst=dst) if _is_mem(
                dst
            ) and not _fits_in_int32(v):
                # movq cannot encode a 64-bit immediate to a memory operand.
                r10q = asm.Register(asm.Registers.R10, asm.Width.QWORD)
                fixed.append(asm.Mov(src=asm.Imm(v), dst=r10q, assembly_type=asm.Quadword()))
                fixed.append(asm.Mov(src=r10q, dst=dst, assembly_type=asm.Quadword()))
            case asm.Mov(assembly_type=asm.Longword(), src=asm.Imm(value=v), dst=dst) if not _fits_in_int32(v):
                # movl immediate must fit in 32 bits; mask to the lower 32 bits (as signed int32).
                truncated = v & 0xFFFFFFFF
                if truncated >= (1 << 31):
                    truncated -= 1 << 32
                fixed.append(asm.Mov(src=asm.Imm(truncated), dst=dst, assembly_type=asm.Longword()))
            case asm.Mov(assembly_type=asm.Byte(), src=asm.Imm(value=v), dst=dst) if not _fits_in_byte(v):
                # movb immediate must fit in 8 bits; reduce modulo 256 to the signed byte range.
                reduced = v & 0xFF
                if reduced > 127:
                    reduced -= 256
                fixed.append(asm.Mov(src=asm.Imm(reduced), dst=dst, assembly_type=asm.Byte()))
            case asm.Mov(assembly_type=asm.Double(), src=src, dst=dst) if _is_mem(src) and _is_mem(dst):
                # movsd cannot route through a GP register; use %xmm14 as the scratch.
                xmm14 = asm.Register(asm.Registers.XMM14, asm.Width.QWORD)
                fixed.append(asm.Mov(src=src, dst=xmm14, assembly_type=asm.Double()))
                fixed.append(asm.Mov(src=xmm14, dst=dst, assembly_type=asm.Double()))
            case asm.Mov(assembly_type=at, src=src, dst=dst) if _is_mem(src) and _is_mem(dst):
                r10 = asm.Register(asm.Registers.R10, _reg_width(at))
                fixed.append(asm.Mov(src=src, dst=r10, assembly_type=at))
                fixed.append(asm.Mov(src=r10, dst=dst, assembly_type=at))
            case asm.Idiv(assembly_type=at, operand=asm.Imm() as imm):
                r11 = asm.Register(asm.Registers.R11, _reg_width(at))
                fixed.append(asm.Mov(src=imm, dst=r11, assembly_type=at))
                fixed.append(asm.Idiv(operand=r11, assembly_type=at))
            case asm.Div(assembly_type=at, operand=asm.Imm() as imm):
                r11 = asm.Register(asm.Registers.R11, _reg_width(at))
                fixed.append(asm.Mov(src=imm, dst=r11, assembly_type=at))
                fixed.append(asm.Div(operand=r11, assembly_type=at))
            case asm.MoveZeroExtend(src_type=asm.Longword(), dst_type=dt, src=src, dst=dst):
                # movl to a 32-bit register implicitly zero-extends to 64 bits; replace with Mov.
                # If both operands are memory, route src through %r10d (single-pass; no re-processing).
                if _is_mem(src) and _is_mem(dst):
                    r10d = asm.Register(asm.Registers.R10, asm.Width.DWORD)
                    fixed.append(asm.Mov(src=src, dst=r10d, assembly_type=asm.Longword()))
                    fixed.append(asm.Mov(src=r10d, dst=dst, assembly_type=asm.Longword()))
                else:
                    fixed.append(asm.Mov(src=src, dst=dst, assembly_type=asm.Longword()))
            case asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=dt, src=asm.Imm() as imm, dst=dst):
                # movzbl/movzbq src must not be an immediate; load into %r10b first.
                r10b = asm.Register(asm.Registers.R10, asm.Width.BYTE)
                if _is_mem(dst):
                    r11 = asm.Register(asm.Registers.R11, _reg_width(dt))
                    fixed.append(asm.Mov(src=imm, dst=r10b, assembly_type=asm.Byte()))
                    fixed.append(asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=dt, src=r10b, dst=r11))
                    fixed.append(asm.Mov(src=r11, dst=dst, assembly_type=dt))
                else:
                    fixed.append(asm.Mov(src=imm, dst=r10b, assembly_type=asm.Byte()))
                    fixed.append(asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=dt, src=r10b, dst=dst))
            case asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=dt, src=src, dst=dst) if _is_mem(dst):
                # movzbl/movzbq dst must be a register; route through %r11 of the destination width.
                r11 = asm.Register(asm.Registers.R11, _reg_width(dt))
                fixed.append(asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=dt, src=src, dst=r11))
                fixed.append(asm.Mov(src=r11, dst=dst, assembly_type=dt))
            case asm.Binary(assembly_type=asm.Quadword(), operator=op, src1=src1, src2=asm.Imm(value=v), dst=dst) if (
                not _fits_in_int32(v)
            ):
                # addq/subq/imulq cannot encode a 64-bit immediate; route src2 through %r10.
                r10q = asm.Register(asm.Registers.R10, asm.Width.QWORD)
                fixed.append(asm.Mov(src=asm.Imm(v), dst=r10q, assembly_type=asm.Quadword()))
                fixed.append(asm.Binary(operator=op, src1=src1, src2=r10q, dst=dst, assembly_type=asm.Quadword()))
            case asm.Cmp(assembly_type=asm.Quadword(), op1=asm.Imm(value=v), op2=asm.Imm() as op2_imm) if (
                not _fits_in_int32(v)
            ):
                # Both operands are immediates and op1 is large: the generic op2-is-Imm case would fire
                # first and leave Cmp(Imm(large), reg) unprocessed. Handle both upfront.
                r10q = asm.Register(asm.Registers.R10, asm.Width.QWORD)
                r11q = asm.Register(asm.Registers.R11, asm.Width.QWORD)
                fixed.append(asm.Mov(src=asm.Imm(v), dst=r10q, assembly_type=asm.Quadword()))
                fixed.append(asm.Mov(src=op2_imm, dst=r11q, assembly_type=asm.Quadword()))
                fixed.append(asm.Cmp(op1=r10q, op2=r11q, assembly_type=asm.Quadword()))
            case asm.Cmp(assembly_type=at, op1=op1, op2=asm.Imm() as op2):
                r11 = asm.Register(asm.Registers.R11, _reg_width(at))
                fixed.append(asm.Mov(src=op2, dst=r11, assembly_type=at))
                fixed.append(asm.Cmp(op1=op1, op2=r11, assembly_type=at))
            case asm.Cmp(assembly_type=asm.Quadword(), op1=asm.Imm(value=v), op2=op2) if not _fits_in_int32(v):
                r10q = asm.Register(asm.Registers.R10, asm.Width.QWORD)
                fixed.append(asm.Mov(src=asm.Imm(v), dst=r10q, assembly_type=asm.Quadword()))
                fixed.append(asm.Cmp(op1=r10q, op2=op2, assembly_type=asm.Quadword()))
            case asm.Cmp(assembly_type=asm.Double(), op1=op1, op2=op2) if _is_mem(op2):
                # ucomisd requires op2 (the XMM "destination" in Intel terms) to be a
                # hardware register. Route op2 through %xmm15 when it is a memory slot.
                xmm15 = asm.Register(asm.Registers.XMM15, asm.Width.QWORD)
                fixed.append(asm.Mov(src=op2, dst=xmm15, assembly_type=asm.Double()))
                fixed.append(asm.Cmp(op1=op1, op2=xmm15, assembly_type=asm.Double()))
            case asm.Cmp(assembly_type=at, op1=op1, op2=op2) if _is_mem(op1) and at != asm.Double():
                # For integer cmpl/cmpq both operands cannot be memory; route op1 through
                # %r10/%r10d.  (ucomisd CAN accept op1 as m64, so doubles are excluded.)
                r10 = asm.Register(asm.Registers.R10, _reg_width(at))
                fixed.append(asm.Mov(src=op1, dst=r10, assembly_type=at))
                fixed.append(asm.Cmp(op1=r10, op2=op2, assembly_type=at))
            case asm.Movsx(src_type=st, dst_type=dt, src=src, dst=dst):
                _fix_movsx(st, dt, src, dst, fixed, _is_mem)
            case asm.Cvttsd2si(assembly_type=at, src=src, dst=dst) if _is_mem(dst):
                # cvttsd2si dst must be a register; route through %r11d/%r11 then store with
                # the correct width so we do not clobber adjacent bytes on the stack.
                r11 = asm.Register(asm.Registers.R11, _reg_width(at))
                fixed.append(asm.Cvttsd2si(src=src, dst=r11, assembly_type=at))
                fixed.append(asm.Mov(src=r11, dst=dst, assembly_type=at))
            case asm.Cvtsi2sd(assembly_type=at, src=asm.Imm() as imm, dst=dst):
                # cvtsi2sd src must be a register or memory, not an immediate; load into %r10/%r10d first.
                r10 = asm.Register(asm.Registers.R10, _reg_width(at))
                fixed.append(asm.Mov(src=imm, dst=r10, assembly_type=at))
                if _is_mem(dst):
                    xmm14 = asm.Register(asm.Registers.XMM14, asm.Width.QWORD)
                    fixed.append(asm.Cvtsi2sd(src=r10, dst=xmm14, assembly_type=at))
                    fixed.append(asm.Mov(src=xmm14, dst=dst, assembly_type=asm.Double()))
                else:
                    fixed.append(asm.Cvtsi2sd(src=r10, dst=dst, assembly_type=at))
            case asm.Cvtsi2sd(assembly_type=at, src=src, dst=dst) if _is_mem(dst):
                # cvtsi2sd dst must be an XMM register; route through %xmm14 then movsd to memory.
                xmm14 = asm.Register(asm.Registers.XMM14, asm.Width.QWORD)
                fixed.append(asm.Cvtsi2sd(src=src, dst=xmm14, assembly_type=at))
                fixed.append(asm.Mov(src=xmm14, dst=dst, assembly_type=asm.Double()))
            case asm.Push(assembly_type=asm.Longword(), operand=asm.Imm(value=v)) if not _fits_in_int32(v):
                # pushq with a sign-extended 32-bit immediate cannot represent an unsigned int
                # value with bit 31 set. Reinterpret as signed int32 for the movl (GAS accepts any
                # 32-bit immediate there), load into %r10d — movl zero-extends so %r10 holds the
                # correct 64-bit unsigned value — then pushq the 64-bit register.
                truncated = v & 0xFFFFFFFF
                if truncated >= (1 << 31):
                    truncated -= 1 << 32
                r10d = asm.Register(asm.Registers.R10, asm.Width.DWORD)
                r10q = asm.Register(asm.Registers.R10, asm.Width.QWORD)
                fixed.append(asm.Mov(src=asm.Imm(truncated), dst=r10d, assembly_type=asm.Longword()))
                fixed.append(asm.Push(operand=r10q, assembly_type=asm.Quadword()))
            case asm.Push(assembly_type=asm.Quadword(), operand=asm.Imm(value=v)) if not _fits_in_int32(v):
                # pushq only accepts 8-bit or 32-bit sign-extended immediates; a 64-bit long
                # constant that doesn't fit in 32 bits must be routed through %r10.
                r10q = asm.Register(asm.Registers.R10, asm.Width.QWORD)
                fixed.append(asm.Mov(src=asm.Imm(v), dst=r10q, assembly_type=asm.Quadword()))
                fixed.append(asm.Push(operand=r10q, assembly_type=asm.Quadword()))
            case asm.Push(assembly_type=asm.Double(), operand=asm.Register() as xmm_reg):
                # pushq %xmmN is not encodable; write explicitly through the stack pointer.
                rsp = asm.Register(asm.Registers.SP, asm.Width.QWORD)
                fixed.append(asm.AllocateStack(8))
                fixed.append(asm.Mov(src=xmm_reg, dst=asm.Memory(reg=rsp, offset=0), assembly_type=asm.Double()))
            case asm.Push(assembly_type=asm.Double(), operand=operand) if _is_mem(operand):
                # movsd cannot move to a GP register, but the double's bits can be
                # loaded as a raw quadword with movq and then pushed normally.
                axq = asm.Register(asm.Registers.AX, asm.Width.QWORD)
                fixed.append(asm.Mov(src=operand, dst=axq, assembly_type=asm.Quadword()))
                fixed.append(asm.Push(operand=axq, assembly_type=asm.Quadword()))
            case asm.Push(assembly_type=at, operand=operand) if _is_mem(operand):
                # pushq cannot address a memory slot directly.
                # Load via movl (zero-extends) for ints, movq for longs, then push the 64-bit register.
                ax = asm.Register(asm.Registers.AX, _reg_width(at))
                fixed.append(asm.Mov(src=operand, dst=ax, assembly_type=at))
                fixed.append(asm.Push(operand=asm.Register(asm.Registers.AX, asm.Width.QWORD), assembly_type=at))
            case asm.Lea(src=src, dst=dst) if _is_mem(dst):
                # leaq destination must be a register; route through %r11 then store.
                r11q = asm.Register(asm.Registers.R11, asm.Width.QWORD)
                fixed.append(asm.Lea(src=src, dst=r11q))
                fixed.append(asm.Mov(src=r11q, dst=dst, assembly_type=asm.Quadword()))
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
