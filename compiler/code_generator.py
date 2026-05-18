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
    tac.SignExtend(src, dst)                  → Movsx(src, dst)
    tac.Truncate(src, dst)                    → Mov(Longword, src, dst)
    tac.Constant(v)                           → Imm(v)
    tac.Var(name)                             → Pseudo(name)
"""

import compiler.syntax.asm as asm
import compiler.syntax.tac as tac
from compiler.syntax.ast.constants import ConstInt, ConstLong
from compiler.syntax.ast.types import LongType
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


def _asm_type(val: tac.Val, symbol_table: SymbolTable | None = None) -> asm.AssemblyType:
    """Derive the assembly type (Longword or Quadword) for a TAC value.

    Constants are typed by their concrete ``Const`` subclass; variables are
    looked up in the symbol table.  Unresolvable variables default to
    ``Longword``.

    Args:
        val: The TAC value whose assembly type is needed.
        symbol_table: The program-wide symbol table.

    Returns:
        ``AssemblyType.Longword`` for 32-bit values, ``AssemblyType.Quadword``
        for 64-bit values.
    """
    match val:
        case tac.Constant(value=ConstInt()):
            return asm.AssemblyType.Longword
        case tac.Constant(value=ConstLong()):
            return asm.AssemblyType.Quadword
        case tac.Var(value=name) if symbol_table is not None and name in symbol_table:
            match symbol_table[name].t:
                case LongType():
                    return asm.AssemblyType.Quadword
                case _:
                    return asm.AssemblyType.Longword
        case _:
            return asm.AssemblyType.Longword


def _reg_width(assembly_type: asm.AssemblyType) -> asm.Width:
    """Return the register width corresponding to an assembly type.

    Args:
        assembly_type: The assembly operand size.

    Returns:
        ``Width.DWORD`` for Longword, ``Width.QWORD`` for Quadword.
    """
    return asm.Width.DWORD if assembly_type == asm.AssemblyType.Longword else asm.Width.QWORD


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
                raw_fn = _translate_function(item, symbol_table)
                resolved_fn, stack_size = _replace_pseudos(raw_fn, static_names, symbol_table)
                fixed_fn = _fix_instructions(resolved_fn)
                definitions.append(_add_stack_allocation(fixed_fn, stack_size))
            case tac.StaticVariable(name=name, globl=globl, t=t, init=init):
                alignment = 8 if isinstance(t, LongType) else 4
                definitions.append(
                    asm.StaticVariable(
                        identifier=name,
                        globl=globl,
                        alignment=alignment,
                        init=_translate_static_init(init),
                    )
                )
    return asm.Program(definitions=tuple(definitions))


def _translate_static_init(init: tac.StaticInit) -> asm.StaticInit:
    """Convert a TAC static initialiser to its assembly counterpart.

    Args:
        init: The TAC ``StaticInit`` node (``IntInit`` or ``LongInit``).

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
        case _:
            raise CodeGenerationError(f"unsupported static initialiser: {type(init).__name__}")


# ==============================================================================
# Pass 1 — TAC lowering
# ==============================================================================


def _translate_function(fn: tac.Function, symbol_table: SymbolTable | None = None) -> asm.Function:
    """Lower a TAC function to an assembly function with pseudo operands.

    For each parameter, a ``Mov`` from the corresponding ABI register into a
    ``Pseudo`` operand is prepended to the instruction list.  Parameters
    beyond the sixth (which arrive on the stack) are handled by positive-offset
    ``Stack`` operands rather than pseudo-registers.

    Args:
        fn: The TAC function to translate.
        symbol_table: Used to derive the assembly type for each parameter.

    Returns:
        An ``asm.Function`` with the same name and lowered instructions.

    Raises:
        CodeGenerationError: If any instruction cannot be translated.
    """
    instructions: list[asm.Instruction] = []

    # Copy register-passed params into pseudo-registers at function entry.
    for i, param in enumerate(fn.params[:6]):
        at = _asm_type(tac.Var(param), symbol_table)
        instructions.append(
            asm.Mov(
                src=asm.Register(_ABI_REGS[i], _reg_width(at)),
                dst=asm.Pseudo(param),
                assembly_type=at,
            )
        )

    # Parameters beyond the sixth arrive on the caller's stack frame at
    # 16(%rbp), 24(%rbp), … (above the saved %rbp and return address).
    for i, param in enumerate(fn.params[6:]):
        at = _asm_type(tac.Var(param), symbol_table)
        instructions.append(
            asm.Mov(
                src=asm.Stack(16 + 8 * i),
                dst=asm.Pseudo(param),
                assembly_type=at,
            )
        )

    for instr in fn.body:
        instructions.extend(_translate_instruction(instr, symbol_table))

    params_asm = tuple(asm.Register(_ABI_REGS[i]) for i in range(min(len(fn.params), 6)))
    return asm.Function(name=fn.name, globl=fn.globl, params=params_asm, instructions=tuple(instructions))


def _translate_instruction(instr: tac.Instruction, symbol_table: SymbolTable | None = None) -> list[asm.Instruction]:
    """Lower a single TAC instruction to a sequence of assembly instructions.

    Args:
        instr: The TAC instruction to translate.
        symbol_table: Used to derive assembly types for variable operands.

    Returns:
        A list of ``asm.Instruction`` nodes equivalent to the TAC instruction.

    Raises:
        CodeGenerationError: If the instruction variant is not supported.
    """
    match instr:
        case tac.Return(val=val):
            at = _asm_type(val, symbol_table)
            ax = asm.Register(asm.Registers.AX, _reg_width(at))
            return [asm.Mov(src=_translate_val(val), dst=ax, assembly_type=at), asm.Ret()]
        case tac.Jump(target=target):
            return [asm.Jmp(target)]
        case tac.JumpIfZero(condition=condition, target=target):
            at = _asm_type(condition, symbol_table)
            return [
                asm.Cmp(op1=asm.Imm(0), op2=_translate_val(condition), assembly_type=at),
                asm.JmpCC(asm.CondCodes.E, target),
            ]
        case tac.JumpIfNotZero(condition=condition, target=target):
            at = _asm_type(condition, symbol_table)
            return [
                asm.Cmp(op1=asm.Imm(0), op2=_translate_val(condition), assembly_type=at),
                asm.JmpCC(asm.CondCodes.NE, target),
            ]
        case tac.Copy(src=src, dst=dst):
            at = _asm_type(src, symbol_table)
            return [asm.Mov(src=_translate_val(src), dst=_translate_val(dst), assembly_type=at)]
        case tac.Label(name=name):
            return [asm.Label(name)]
        case tac.Unary(operator=op, src=src, dst=dst):
            at = _asm_type(dst, symbol_table)
            src_asm = _translate_val(src)
            dst_asm = _translate_val(dst)
            match op:
                case tac.Not():
                    return [
                        asm.Cmp(op1=asm.Imm(0), op2=src_asm, assembly_type=at),
                        asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.AssemblyType.Longword),
                        asm.SetCC(asm.CondCodes.E, dst_asm),
                    ]
                case _:
                    return [
                        asm.Mov(src=src_asm, dst=dst_asm, assembly_type=at),
                        asm.Unary(operator=_translate_unary_op(op), operand=dst_asm, assembly_type=at),
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
                    # Cmp uses the type of the values being compared (src operands).
                    cmp_at = _asm_type(src1, symbol_table)
                    return [
                        asm.Cmp(op1=src2_asm, op2=src1_asm, assembly_type=cmp_at),
                        asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.AssemblyType.Longword),
                        asm.SetCC(_translate_relational_op(op), dst_asm),
                    ]
                case (
                    tac.Add() | tac.Subtract() | tac.Multiply() | tac.BitwiseAnd() | tac.BitwiseOr() | tac.BitwiseXor()
                ):
                    at = _asm_type(dst, symbol_table)
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
                        asm.Mov(src=src2_asm, dst=cx, assembly_type=asm.AssemblyType.Longword),
                        asm.Shl(operand=r11, assembly_type=at),
                        asm.Mov(src=r11, dst=dst_asm, assembly_type=at),
                    ]
                case tac.RightShift():
                    at = _asm_type(dst, symbol_table)
                    r11 = asm.Register(asm.Registers.R11, _reg_width(at))
                    cx = asm.Register(asm.Registers.CX, asm.Width.DWORD)
                    return [
                        asm.Mov(src=src1_asm, dst=r11, assembly_type=at),
                        asm.Mov(src=src2_asm, dst=cx, assembly_type=asm.AssemblyType.Longword),
                        asm.Sar(operand=r11, assembly_type=at),
                        asm.Mov(src=r11, dst=dst_asm, assembly_type=at),
                    ]
                case tac.Divide():
                    at = _asm_type(dst, symbol_table)
                    ax = asm.Register(asm.Registers.AX, _reg_width(at))
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
                    return [
                        asm.Mov(src=src1_asm, dst=ax, assembly_type=at),
                        asm.Cdq(assembly_type=at),
                        asm.Idiv(operand=src2_asm, assembly_type=at),
                        asm.Mov(src=dx, dst=dst_asm, assembly_type=at),
                    ]
                case _:
                    raise CodeGenerationError(f"unsupported TAC binary operator: {type(op).__name__}")
        case tac.FunCall(name=name, args=args, dst=dst):
            return _translate_funcall(name, list(args), _translate_val(dst), symbol_table)
        case tac.SignExtend(src=src, dst=dst):
            return [asm.Movsx(src=_translate_val(src), dst=_translate_val(dst))]
        case tac.Truncate(src=src, dst=dst):
            # Copying only the lower 32 bits via movl achieves truncation from long → int.
            return [asm.Mov(src=_translate_val(src), dst=_translate_val(dst), assembly_type=asm.AssemblyType.Longword)]
        case _:
            raise CodeGenerationError(f"unsupported TAC instruction: {type(instr).__name__}")


def _translate_funcall(
    name: str, args: list[tac.Val], dst: asm.Operand, symbol_table: SymbolTable | None = None
) -> list[asm.Instruction]:
    """Lower a ``tac.FunCall`` to the System V AMD64 ABI calling sequence.

    Up to six integer arguments are placed in the ABI registers (DI, SI, DX,
    CX, R8, R9).  Additional arguments are pushed onto the stack right-to-left.
    An 8-byte padding ``AllocateStack`` is inserted before the stack pushes
    when there is an odd number of stack-passed arguments, so that the stack
    pointer remains 16-byte aligned at the ``call`` instruction.

    After the call, any stack space consumed by stack-passed arguments (and
    the optional padding) is reclaimed with ``DeallocateStack``.  The return
    value is then moved from ``%eax``/``%rax`` into ``dst``.

    Args:
        name: The symbol name of the function to call.
        args: The ordered list of TAC argument values.
        dst: The assembly operand that receives the return value.
        symbol_table: Used to derive assembly types for variable arguments.

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
        at = _asm_type(arg, symbol_table)
        instrs.append(asm.Push(operand=_translate_val(arg), assembly_type=at))

    # Load register args into ABI registers.
    for i, arg in enumerate(reg_args):
        at = _asm_type(arg, symbol_table)
        instrs.append(
            asm.Mov(src=_translate_val(arg), dst=asm.Register(_ABI_REGS[i], _reg_width(at)), assembly_type=at)
        )

    instrs.append(asm.Call(name))

    # Reclaim stack space used for stack-passed args and any alignment padding.
    bytes_to_free = 8 * len(stack_args) + (8 if needs_padding else 0)
    if bytes_to_free:
        instrs.append(asm.DeallocateStack(bytes_to_free))

    # Return value is in %eax / %rax; use dst's type to select the right register.
    at = _asm_type(tac.Var(dst.name) if isinstance(dst, asm.Pseudo) else tac.Constant(ConstInt(0)), symbol_table)
    instrs.append(asm.Mov(src=asm.Register(asm.Registers.AX, _reg_width(at)), dst=dst, assembly_type=at))

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
            return asm.Imm(value=v.val)
        case tac.Var(value=name):
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
    """Translate a TAC relational operator to its assembly condition code.

    Args:
        op: The TAC binary operator node.

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


def _replace_pseudos(
    fn: asm.Function, static_names: frozenset[str], symbol_table: SymbolTable | None = None
) -> tuple[asm.Function, int]:
    """Replace every ``Pseudo`` operand with a concrete operand.

    Pseudo operands whose names appear in ``static_names`` become RIP-relative
    ``Data`` operands.  All other pseudos are assigned a unique slot in the
    stack frame (4 bytes for ``int``, 8 bytes for ``long``), starting just
    below ``%rbp`` and growing downward.  The total number of stack bytes used
    is returned so the caller can emit an ``AllocateStack``.

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
        """Map a single operand: pass through non-Pseudo, map Pseudo to Data or Stack."""
        if not isinstance(operand, asm.Pseudo):
            return operand
        if operand.name in static_names:
            return asm.Data(name=operand.name)
        if operand.name not in pseudo_offsets:
            size = 4
            if symbol_table is not None and operand.name in symbol_table:
                if isinstance(symbol_table[operand.name].t, LongType):
                    size = 8
            next_offset[0] -= size
            pseudo_offsets[operand.name] = next_offset[0]
        return asm.Stack(offset=pseudo_offsets[operand.name])

    new_instructions: list[asm.Instruction] = []
    for instr in fn.instructions:
        match instr:
            case asm.Mov(assembly_type=at, src=src, dst=dst):
                new_instructions.append(asm.Mov(src=resolve(src), dst=resolve(dst), assembly_type=at))
            case asm.Movsx(src=src, dst=dst):
                new_instructions.append(asm.Movsx(src=resolve(src), dst=resolve(dst)))
            case asm.Unary(assembly_type=at, operator=op, operand=operand):
                new_instructions.append(asm.Unary(operator=op, operand=resolve(operand), assembly_type=at))
            case asm.Binary(assembly_type=at, operator=op, src1=src1, src2=src2, dst=dst):
                new_instructions.append(
                    asm.Binary(operator=op, src1=resolve(src1), src2=resolve(src2), dst=dst, assembly_type=at)
                )
            case asm.Idiv(assembly_type=at, operand=operand):
                new_instructions.append(asm.Idiv(operand=resolve(operand), assembly_type=at))
            case asm.Shl(assembly_type=at, operand=operand) | asm.Sar(assembly_type=at, operand=operand):
                new_instructions.append(type(instr)(operand=resolve(operand), assembly_type=at))
            case asm.Cmp(assembly_type=at, op1=op1, op2=op2):
                new_instructions.append(asm.Cmp(op1=resolve(op1), op2=resolve(op2), assembly_type=at))
            case asm.SetCC(cond_code=cc, operand=operand):
                new_instructions.append(asm.SetCC(cond_code=cc, operand=resolve(operand)))
            case asm.Push(assembly_type=at, operand=operand):
                new_instructions.append(asm.Push(operand=resolve(operand), assembly_type=at))
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

    Handles the following cases:

    * ``Mov(mem, mem)`` — both operands cannot be memory; route ``src`` through
      ``%r10``/``%r10d`` of the appropriate width.
    * ``Idiv(Imm)`` — the divisor cannot be an immediate; route through
      ``%r11``/``%r11d``.
    * ``Cmp(op1, Imm)`` — ``op2`` (the second printed operand) cannot be an
      immediate; route through ``%r11``/``%r11d``.
    * ``Cmp(mem, op2)`` — ``op1`` cannot be memory when ``op2`` is also memory;
      route ``op1`` through ``%r10``/``%r10d``.
    * ``Movsx(src, mem)`` — the destination of ``movslq`` must be a register;
      route through ``%r11`` then move to ``dst``.
    * ``Push(mem)`` — ``pushq`` cannot address a memory slot directly; load
      into ``%eax``/``%rax`` first.

    Args:
        fn: The assembly function to fix up.

    Returns:
        An ``asm.Function`` with all invalid instructions rewritten.
    """

    def _is_mem(op: asm.Operand) -> bool:
        """Return True when ``op`` is a memory address (Stack or Data)."""
        return isinstance(op, (asm.Stack, asm.Data))

    fixed: list[asm.Instruction] = []
    for instr in fn.instructions:
        match instr:
            case asm.Mov(assembly_type=at, src=src, dst=dst) if _is_mem(src) and _is_mem(dst):
                r10 = asm.Register(asm.Registers.R10, _reg_width(at))
                fixed.append(asm.Mov(src=src, dst=r10, assembly_type=at))
                fixed.append(asm.Mov(src=r10, dst=dst, assembly_type=at))
            case asm.Idiv(assembly_type=at, operand=asm.Imm() as imm):
                r11 = asm.Register(asm.Registers.R11, _reg_width(at))
                fixed.append(asm.Mov(src=imm, dst=r11, assembly_type=at))
                fixed.append(asm.Idiv(operand=r11, assembly_type=at))
            case asm.Cmp(assembly_type=at, op1=op1, op2=asm.Imm() as op2):
                r11 = asm.Register(asm.Registers.R11, _reg_width(at))
                fixed.append(asm.Mov(src=op2, dst=r11, assembly_type=at))
                fixed.append(asm.Cmp(op1=op1, op2=r11, assembly_type=at))
            case asm.Cmp(assembly_type=at, op1=op1, op2=op2) if _is_mem(op1):
                r10 = asm.Register(asm.Registers.R10, _reg_width(at))
                fixed.append(asm.Mov(src=op1, dst=r10, assembly_type=at))
                fixed.append(asm.Cmp(op1=r10, op2=op2, assembly_type=at))
            case asm.Movsx(src=src, dst=dst) if _is_mem(dst):
                # movslq dst must be a register; route through %r11 then store.
                r11q = asm.Register(asm.Registers.R11, asm.Width.QWORD)
                fixed.append(asm.Movsx(src=src, dst=r11q))
                fixed.append(asm.Mov(src=r11q, dst=dst, assembly_type=asm.AssemblyType.Quadword))
            case asm.Push(assembly_type=at, operand=operand) if _is_mem(operand):
                # pushq cannot address a memory slot directly.
                # Load via movl (zero-extends) for ints, movq for longs, then push the 64-bit register.
                ax = asm.Register(asm.Registers.AX, _reg_width(at))
                fixed.append(asm.Mov(src=operand, dst=ax, assembly_type=at))
                fixed.append(asm.Push(operand=asm.Register(asm.Registers.AX, asm.Width.QWORD), assembly_type=at))
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
