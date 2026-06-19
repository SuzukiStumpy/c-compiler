"""Code generation stage: translates a TAC AST into an assembly AST.

Performs a multi-pass structural translation from ``compiler.syntax.tac`` nodes
to ``compiler.syntax.asm`` nodes.  The passes are applied in order:

1. **TAC -> pseudo-asm**: each TAC instruction is lowered to one or more assembly
   instructions, using ``Pseudo`` operands for named temporaries.

2. **Pseudo replacement**: every ``Pseudo`` operand is replaced with a ``Stack``
   operand at a unique frame offset; an ``AllocateStack`` instruction is
   prepended when any temporaries were spilled.

3. **Instruction fixup**: invalid two-memory-operand ``Mov`` instructions are
   rewritten to route the value through a scratch register.

The resulting ``asm.Program`` is a complete assembly AST ready for the emission
stage.

TAC -> pseudo-asm mapping:

    tac.Return(val)                           -> Mov(val, Register(AX)), Ret()  [val=None -> Ret() only]
    tac.Unary(op, src, dst)                   -> Mov(src, dst), Unary(op, dst)
    tac.Binary(Add|Sub|Mul, src1, src2, dst)         -> Binary(op, src1, src2, Register(R11)),
                                                        Mov(Register(R11), dst)
    tac.Binary(And|Or|Xor, src1, src2, dst)          -> Binary(op, src1, src2, Register(R11)),
                                                        Mov(Register(R11), dst)
    tac.Binary(Divide, src1, src2, dst)              -> Mov(src1, Register(AX)), Cdq,
                                                        Idiv(src2), Mov(Register(AX), dst)
    tac.Binary(Remainder, src1, src2, dst)           -> Mov(src1, Register(AX)), Cdq,
                                                        Idiv(src2), Mov(Register(DX), dst)
    tac.Binary(LeftShift, src1, src2, dst)           -> Mov(src1, Register(R11)),
                                                        Mov(src2, Register(CX)),
                                                        Shl(Register(R11)), Mov(Register(R11), dst)
    tac.Binary(RightShift, src1, src2, dst)          -> Mov(src1, Register(R11)),
                                                        Mov(src2, Register(CX)),
                                                        Sar(Register(R11)), Mov(Register(R11), dst)
    tac.SignExtend(src, dst)                  -> Movsx(src_type, dst_type, src, dst)
    tac.ZeroExtend(src, dst)                  -> MoveZeroExtend(src_type, dst_type, src, dst)
    tac.Truncate(src, dst)                    -> Mov(dst_type, src, dst)
    tac.Constant(v)                           -> Imm(v)
    tac.Var(name)                             -> Pseudo(name)

Unsigned variants (selected when the source operand has an unsigned C type):

    tac.Binary(Divide, unsigned, src1, src2, dst)    -> Mov(src1, AX), Mov(Imm(0), DX),
                                                        Div(src2), Mov(AX, dst)
    tac.Binary(Remainder, unsigned, src1, src2, dst) -> Mov(src1, AX), Mov(Imm(0), DX),
                                                        Div(src2), Mov(DX, dst)
    tac.Binary(RightShift, unsigned, src1, src2, dst) -> Mov(src1, R11), Mov(src2, CX),
                                                         Shr(R11), Mov(R11, dst)
    tac.Binary(relational, unsigned, ...)             -> Cmp, Mov(Imm(0), dst), SetCC(A/AE/B/BE)
"""

# ruff: noqa: C901 - ignore complexity warnings for entire file (since there's lots of them)

import dataclasses
import itertools
from enum import Enum, auto

import compiler.syntax.asm as asm
import compiler.syntax.tac as tac
from compiler.optimiser.asm_main import allocate_registers
from compiler.semantic_analysis import type_size
from compiler.syntax.ast.constants import (
    ConstBool,
    ConstChar,
    ConstDouble,
    ConstFloat,
    ConstInt,
    ConstLong,
    ConstLongDouble,
    ConstLongLong,
    ConstShort,
    ConstUChar,
    ConstUInt,
    ConstUInt128,
    ConstULong,
    ConstULongLong,
    ConstUShort,
)
from compiler.syntax.ast.types import Array as ArrayType
from compiler.syntax.ast.types import (
    BoolType,
    Char,
    DoubleType,
    FloatType,
    FunType,
    IntType,
    LongDoubleType,
    LongLongType,
    LongType,
    Pointer,
    SChar,
    ShortType,
    Structure,
    Type,
    UChar,
    UInt128Type,
    UIntType,
    ULongLongType,
    ULongType,
    Union,
    UShortType,
    VoidType,
    is_const,
)
from compiler.types.symbols import FunAttr, StaticAttr, StructTypeTable, SymbolTable
from compiler.utils.exceptions import CodeGenerationError

# System V AMD64 ABI integer argument registers, in order.
_ABI_REGS = [
    asm.GPRegisters.DI,
    asm.GPRegisters.SI,
    asm.GPRegisters.DX,
    asm.GPRegisters.CX,
    asm.GPRegisters.R8,
    asm.GPRegisters.R9,
]

# System V AMD64 ABI floating-point argument registers, in order.
_XMM_REGS = [
    asm.FPRegisters.XMM0,
    asm.FPRegisters.XMM1,
    asm.FPRegisters.XMM2,
    asm.FPRegisters.XMM3,
    asm.FPRegisters.XMM4,
    asm.FPRegisters.XMM5,
    asm.FPRegisters.XMM6,
    asm.FPRegisters.XMM7,
]

# 2^63 as a double - the threshold used by DoubleToUInt for unsigned-long destinations.
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
        case tac.Constant(value=ConstBool() | ConstChar() | ConstUChar()):
            return asm.Byte()
        case tac.Constant(value=ConstShort() | ConstUShort()):
            return asm.Word()
        case tac.Constant(value=ConstInt() | ConstUInt()):
            return asm.Longword()
        case tac.Constant(value=ConstLong() | ConstULong() | ConstLongLong() | ConstULongLong()):
            return asm.Quadword()
        case tac.Constant(value=ConstDouble()):
            return asm.Double()
        case tac.Constant(value=ConstFloat()):
            return asm.Float()
        case tac.Constant(value=ConstLongDouble()):
            return asm.LongDouble()
        case tac.Var(value=name) if symbol_table is not None and name in symbol_table:
            match symbol_table[name].t:
                case LongType() | ULongType() | LongLongType() | ULongLongType() | Pointer() | FunType():
                    return asm.Quadword()
                case DoubleType():
                    return asm.Double()
                case FloatType():
                    return asm.Float()
                case LongDoubleType():
                    return asm.LongDouble()
                case BoolType() | Char() | SChar() | UChar():
                    return asm.Byte()
                case ShortType() | UShortType():
                    return asm.Word()
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
    if assembly_type == asm.Word():
        return asm.Width.WORD
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
        case tac.Constant(
            value=ConstBool()
            | ConstUChar()
            | ConstUShort()
            | ConstUInt()
            | ConstULong()
            | ConstULongLong()
            | ConstUInt128()
        ):
            return True
        case tac.Var(value=name) if symbol_table is not None and name in symbol_table:
            return isinstance(
                symbol_table[name].t,
                (BoolType, UChar, UShortType, UIntType, ULongType, ULongLongType, Pointer, UInt128Type),
            )
        case _:
            return False


class EightbyteClass(Enum):
    """System V AMD64 ABI eightbyte class for struct argument and return-value passing.

    Each 8-byte chunk of a struct is assigned one of these classes to determine
    which register file (integer GPR, SSE/XMM, or the stack) handles it during
    a function call.

    Attributes:
        NO_CLASS: Unused / padding-only eightbyte; ignored during register allocation.
        INTEGER: General-purpose register (``rdi``, ``rsi``, ... / ``rax``, ``rdx``).
        SSE: Floating-point XMM register (``xmm0``-``xmm7`` / ``xmm0``, ``xmm1``).
        MEMORY: Struct is too large or unclassifiable; passed via a hidden stack pointer.
    """

    NO_CLASS = auto()
    INTEGER = auto()
    SSE = auto()
    MEMORY = auto()


def _merge_class(a: EightbyteClass, b: EightbyteClass) -> EightbyteClass:
    """Merge two eightbyte classes following SysV AMD64 ABI sec. 3.2.3 rule 5.

    Priority order (highest wins): MEMORY > INTEGER > SSE > NO_CLASS.

    Args:
        a: First class.
        b: Second class.

    Returns:
        The merged ``EightbyteClass``.
    """
    if a == b:
        return a
    if a == EightbyteClass.NO_CLASS:
        return b
    if b == EightbyteClass.NO_CLASS:
        return a
    if EightbyteClass.MEMORY in (a, b):
        return EightbyteClass.MEMORY
    if EightbyteClass.INTEGER in (a, b):
        return EightbyteClass.INTEGER
    return EightbyteClass.SSE


def _classify_type_at_offset(
    t: Type,
    byte_offset: int,
    eightbytes: list[EightbyteClass],
    struct_type_table: StructTypeTable,
) -> None:
    """Recursively merge the classification of ``t`` into ``eightbytes``.

    Scalar types contribute a single class to the eightbyte that contains their
    first byte (``byte_offset // 8``).  Arrays and nested structs recurse over
    their elements/members with adjusted offsets.

    Args:
        t: The C type to classify.
        byte_offset: Byte offset of ``t`` from the start of the outermost struct.
        eightbytes: Mutable list of ``EightbyteClass``, indexed by eightbyte number.
            Updated in-place via :func:`_merge_class`.
        struct_type_table: Mapping from struct tag to ``StructEntry`` layout.

    Raises:
        CodeGenerationError: If ``t`` is a type that cannot appear as a struct member
            (e.g. ``VoidType``, ``FunType``).
    """
    eb = byte_offset // 8
    match t:
        case DoubleType() | FloatType():
            eightbytes[eb] = _merge_class(eightbytes[eb], EightbyteClass.SSE)
        case (
            IntType()
            | LongType()
            | LongLongType()
            | UIntType()
            | ULongType()
            | ULongLongType()
            | Char()
            | SChar()
            | UChar()
            | ShortType()
            | UShortType()
            | Pointer()
        ):
            eightbytes[eb] = _merge_class(eightbytes[eb], EightbyteClass.INTEGER)
        case ArrayType(element=elem_t, size=n):
            elem_size = type_size(elem_t, struct_type_table)
            for i in range(n):
                _classify_type_at_offset(elem_t, byte_offset + i * elem_size, eightbytes, struct_type_table)
        case Structure(tag=tag) | Union(tag=tag):
            entry = struct_type_table[tag]
            for member in entry.members:
                _classify_type_at_offset(member.member_type, byte_offset + member.offset, eightbytes, struct_type_table)
        case _:
            raise CodeGenerationError(f"cannot classify type {type(t).__name__} in aggregate eightbyte")


def classify_struct(tag: str, struct_type_table: StructTypeTable) -> list[EightbyteClass]:
    """Classify each eightbyte of a struct type per the SysV AMD64 ABI sec. 3.2.3.

    Determines how each 8-byte chunk of the struct should be passed or returned:
    in an integer register, an XMM register, or via a hidden memory pointer.

    Rules applied (in order):

    1. If the struct is larger than 16 bytes (two eightbytes) it must be passed
       via memory and ``[MEMORY]`` is returned.
    2. Each eightbyte starts as ``NO_CLASS`` and is updated by merging the
       classification of every member whose storage overlaps that chunk.
    3. If any eightbyte resolves to ``MEMORY`` after merging, the whole struct
       must be passed via memory and ``[MEMORY]`` is returned.

    Args:
        tag: The resolved struct tag name.
        struct_type_table: Mapping from struct tag to ``StructEntry`` layout.

    Returns:
        A list of ``EightbyteClass`` values, one per 8-byte chunk of the struct.
        A single-element list ``[MEMORY]`` means the entire struct is passed via
        the stack.

    Raises:
        KeyError: If ``tag`` is absent from ``struct_type_table``.
        CodeGenerationError: If any member type cannot be classified.
    """
    entry = struct_type_table[tag]

    if entry.size > 16:
        return [EightbyteClass.MEMORY]

    num_eightbytes = (entry.size + 7) // 8
    classes: list[EightbyteClass] = [EightbyteClass.NO_CLASS] * num_eightbytes

    for member in entry.members:
        _classify_type_at_offset(member.member_type, member.offset, classes, struct_type_table)

    if EightbyteClass.MEMORY in classes:
        return [EightbyteClass.MEMORY]

    return classes


def classify_parameters(
    params: tuple[str, ...],
    symbol_table: SymbolTable | None,
    struct_type_table: StructTypeTable,
) -> list[list[EightbyteClass]]:
    """Classify each function parameter per the SysV AMD64 ABI sec. 3.2.3.

    Returns one classification list per parameter.  Scalars produce a
    single-element list; structs are delegated to :func:`classify_struct` which
    returns one entry per eightbyte.

    Args:
        params: Ordered parameter names as found in ``tac.Function.params``.
        symbol_table: Used to look up each parameter's declared C type.
        struct_type_table: Required to classify struct-typed parameters; passed
            through to :func:`classify_struct`.

    Returns:
        A list with one entry per parameter.  Each entry is a list of
        ``EightbyteClass`` values:

        - Scalar ``double``  -> ``[SSE]``
        - Other scalar       -> ``[INTEGER]``
        - Struct <= 16 bytes  -> one ``EightbyteClass`` per eightbyte
        - Struct > 16 bytes  -> ``[MEMORY]``
    """
    result: list[list[EightbyteClass]] = []
    for param in params:
        t: Type | None = symbol_table[param].t if (symbol_table and param in symbol_table) else None
        if isinstance(t, (Structure, Union)):
            result.append(classify_struct(t.tag, struct_type_table))
        elif isinstance(t, LongDoubleType):
            result.append([EightbyteClass.MEMORY])
        elif isinstance(t, (DoubleType, FloatType)):
            result.append([EightbyteClass.SSE])
        else:
            result.append([EightbyteClass.INTEGER])
    return result


# Return-value register sequences per the SysV AMD64 ABI.
_GP_RETURN_REGS = [asm.GPRegisters.AX, asm.GPRegisters.DX]
_XMM_RETURN_REGS = [asm.FPRegisters.XMM0, asm.FPRegisters.XMM1]


def classify_return_value(
    t: Type,
    struct_type_table: StructTypeTable,
) -> tuple[list[asm.Register], list[asm.Register], bool]:
    """Classify how a return value is conveyed per the SysV AMD64 ABI sec. 3.2.3.

    Produces a unified triple for both scalar and struct return types so that
    callers can handle all cases with the same shape of data.

    Scalar types map to a single register in the appropriate class.  Struct
    types are classified via :func:`classify_struct`: each eightbyte is assigned
    to the next available integer (``RAX`` -> ``RDX``) or SSE (``XMM0`` -> ``XMM1``)
    register in declaration order.  Structs that cannot be returned in registers
    set ``in_memory=True``.

    Args:
        t: The C return type to classify.
        struct_type_table: Required to classify struct return types; forwarded
            to :func:`classify_struct`.

    Returns:
        A three-element tuple ``(gp_regs, xmm_regs, in_memory)`` where:

        - ``gp_regs``: Ordered list of integer registers that carry return data
          (subset of ``[RAX, RDX]``; at most two elements).
        - ``xmm_regs``: Ordered list of XMM registers that carry return data
          (subset of ``[XMM0, XMM1]``; at most two elements).
        - ``in_memory``: ``True`` iff the return value is passed via a hidden
          caller-allocated memory pointer.  When ``True``, both register lists
          are empty.

    Raises:
        CodeGenerationError: If ``t`` is a type that cannot be returned
          (e.g. an array type).
    """
    match t:
        case VoidType():
            return [], [], False
        case IntType() | UIntType():
            return [asm.Register(asm.GPRegisters.AX, asm.Width.DWORD)], [], False
        case LongType() | ULongType() | LongLongType() | ULongLongType() | Pointer():
            return [asm.Register(asm.GPRegisters.AX, asm.Width.QWORD)], [], False
        case Char() | SChar() | UChar():
            return [asm.Register(asm.GPRegisters.AX, asm.Width.BYTE)], [], False
        case LongDoubleType():
            # Long double is returned in ST0 (x87 top-of-stack), not in GP or XMM registers.
            # Handled separately in _translate_funcall by checking ret_type directly.
            return [], [], False
        case DoubleType():
            return [], [asm.Register(asm.FPRegisters.XMM0, asm.Width.QWORD)], False
        case FloatType():
            return [], [asm.Register(asm.FPRegisters.XMM0, asm.Width.DWORD)], False
        case Structure(tag=tag) | Union(tag=tag):
            classes = classify_struct(tag, struct_type_table)
            if classes == [EightbyteClass.MEMORY]:
                return [], [], True
            gp_regs: list[asm.Register] = []
            xmm_regs: list[asm.Register] = []
            gp_idx = 0
            xmm_idx = 0
            for cls in classes:
                if cls == EightbyteClass.INTEGER:
                    gp_regs.append(asm.Register(_GP_RETURN_REGS[gp_idx], asm.Width.QWORD))
                    gp_idx += 1
                else:
                    xmm_regs.append(asm.Register(_XMM_RETURN_REGS[xmm_idx], asm.Width.QWORD))
                    xmm_idx += 1
            return gp_regs, xmm_regs, False
        case _:
            raise CodeGenerationError(f"cannot classify return type {type(t).__name__}")


def generate(
    program: tac.Program,
    symbol_table: SymbolTable | None = None,
    struct_type_table: StructTypeTable | None = None,
) -> asm.Program:
    """Translate a TAC ``Program`` into an assembly ``Program``.

    Applies the following translation passes in order:

    1. **TAC -> pseudo-asm**: each TAC instruction is lowered to assembly
       instructions with ``Pseudo`` operands for named temporaries.
    2. **Register allocation**: graph-colouring register allocation replaces
       allocatable ``Pseudo`` operands with hardware registers; spilled pseudos
       remain as ``Pseudo`` for the next pass.
    3. **Pseudo replacement**: remaining ``Pseudo`` operands are assigned
       unique stack-frame slots.
    4. **Instruction fixup**: invalid two-memory-operand forms are rewritten
       via scratch registers.

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
        struct_type_table: Mapping from struct tag to ``StructEntry`` layout,
            required to classify struct-typed parameters and return values.

    Returns:
        An ``asm.Program`` node representing the equivalent assembly program.

    Raises:
        CodeGenerationError: If any TAC construct cannot be translated.
    """
    # Names from tac.StaticVariable nodes (Initial / Tentative variables).
    tac_static_names: frozenset[str] = frozenset(
        item.name for item in program.top_level if isinstance(item, (tac.StaticVariable, tac.StaticConstant))
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
    # 2^63 threshold for double -> unsigned-long).  These are prepended to the output
    # so that they appear before the first .text section.
    constants: list[asm.TopLevel] = []
    definitions: list[asm.TopLevel] = []
    for item in program.top_level:
        match item:
            case tac.Function():
                raw_fn = _translate_function(item, symbol_table, constants, struct_type_table)
                allocated, gp_callee, fp_callee = allocate_registers(
                    list(raw_fn.instructions), symbol_table, fn_name=raw_fn.name, struct_type_table=struct_type_table
                )
                raw_fn = dataclasses.replace(raw_fn, instructions=tuple(allocated))
                gp_callee_size = 8 * len(gp_callee)
                callee_saved_size = gp_callee_size + 8 * len(fp_callee)
                resolved_fn, stack_size = _replace_pseudos(
                    raw_fn,
                    static_names,
                    symbol_table,
                    struct_type_table,
                    callee_saved_size=callee_saved_size,
                )
                fn_with_saves = _insert_callee_saves(resolved_fn, gp_callee, fp_callee)
                fn_with_gp = dataclasses.replace(fn_with_saves, gp_callee_saved=tuple(gp_callee))
                fixed_fn = _fix_instructions(fn_with_gp)
                definitions.append(
                    _add_stack_allocation(
                        fixed_fn,
                        stack_size - gp_callee_size,
                        gp_callee_size=gp_callee_size,
                    )
                )
            case tac.StaticVariable(name=name, globl=globl, t=t, init=init, align_as=align_as):
                definitions.append(
                    asm.StaticVariable(
                        identifier=name,
                        globl=globl,
                        alignment=align_as if align_as is not None else _static_alignment(t),
                        init=tuple(_translate_static_init(i) for i in init),
                        read_only=is_const(t),
                    )
                )
            case tac.StaticConstant(name=name, t=t, init=init):
                constants.append(
                    asm.StaticConstant(
                        identifier=name,
                        alignment=_static_alignment(t),
                        init=_translate_static_init(init),
                    )
                )
    return asm.Program(definitions=tuple(constants + definitions))


def _type_size(t: Type, struct_type_table: StructTypeTable | None = None) -> int:
    """Return the size in bytes of a C type.

    Args:
        t: The C type whose size is required.
        struct_type_table: Required to compute the size of ``Structure`` types.

    Returns:
        1 for char; 4 for ``int``/``unsigned int``; 8 for ``long``/``unsigned long``/
        ``double``/pointer; element_size * count for arrays; ``StructEntry.size`` for
        struct types when ``struct_type_table`` is provided; 8 as a fallback otherwise.
    """
    if isinstance(t, (BoolType, Char, SChar, UChar)):
        return 1
    if isinstance(t, (IntType, UIntType, FloatType)):
        return 4
    if isinstance(t, LongDoubleType):
        return 16
    if isinstance(t, (LongType, ULongType, LongLongType, ULongLongType, DoubleType, Pointer)):
        return 8
    if isinstance(t, ArrayType):
        return t.size * _type_size(t.element, struct_type_table)
    if isinstance(t, (Structure, Union)) and struct_type_table and t.tag in struct_type_table:
        return struct_type_table[t.tag].size
    return 8


def _static_alignment(t: Type, struct_type_table: StructTypeTable | None = None) -> int:
    """Return the data-segment alignment in bytes for a static variable of type ``t``.

    Arrays >= 16 bytes are aligned to 16 bytes so SSE instructions can address them
    directly.  Arrays smaller than 16 bytes use the alignment of their element type.
    The rule applies only to the outermost array; element alignment within the array
    is not affected.

    Args:
        t: The C type of the static variable.
        struct_type_table: Required to look up struct alignment.

    Returns:
        16 for arrays whose total size is >= 16 bytes; the element-type alignment
        for smaller arrays; 8 for ``long``, ``unsigned long``, ``double``, or any
        pointer; the ``StructEntry.alignment`` for struct types; 4 otherwise.
    """
    if isinstance(t, LongDoubleType):
        return 16
    if isinstance(t, (LongType, ULongType, LongLongType, ULongLongType, DoubleType, Pointer)):
        return 8
    if isinstance(t, ArrayType):
        if _type_size(t, struct_type_table) >= 16:
            return 16
        return _static_alignment(t.element, struct_type_table)
    if isinstance(t, (Structure, Union)) and struct_type_table and t.tag in struct_type_table:
        return struct_type_table[t.tag].alignment
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
        case tac.BoolInit(value=v):
            return asm.CharInit(v)
        case tac.IntInit(value=v):
            return asm.IntInit(v)
        case tac.LongInit(value=v):
            return asm.LongInit(v)
        case tac.DoubleInit(value=v):
            return asm.DoubleInit(v)
        case tac.FloatInit(value=v):
            return asm.FloatInit(v)
        case tac.LongDoubleInit(value=v):
            return asm.LongDoubleInit(v)
        case tac.CharInit(value=v):
            return asm.CharInit(v)
        case tac.ShortInit(value=v):
            return asm.ShortInit(v)
        case tac.UShortInit(value=v):
            return asm.UShortInit(v)
        case tac.Int128Init(value=v):
            return asm.Int128Init(v)
        case tac.UInt128Init(value=v):
            return asm.UInt128Init(v)
        case tac.ZeroInit(bytes=n):
            return asm.ZeroInit(n)
        case tac.StringInit(string=s, null_terminated=nt):
            return asm.StringInit(s, nt)
        case tac.PointerInit(name=n):
            return asm.PointerInit(n)
        case _:
            raise CodeGenerationError(f"unsupported static initialiser: {type(init).__name__}")


# ==============================================================================
# Pass 1 - TAC lowering
# ==============================================================================


def _translate_function(
    fn: tac.Function,
    symbol_table: SymbolTable | None = None,
    constants: list[asm.TopLevel] | None = None,
    struct_type_table: StructTypeTable | None = None,
) -> asm.Function:
    """Lower a TAC function to an assembly function with pseudo operands.

    For each parameter, one or more ``Mov`` instructions from the ABI registers
    (or caller stack slots) into ``Pseudo`` / ``PseudoMem`` operands are prepended
    to the instruction list.  Scalar parameters beyond the sixth integer or eighth
    XMM register arrive on the stack.  Struct parameters are classified via
    :func:`classify_parameters`: register-passed structs use one register per
    eightbyte; MEMORY structs are copied from the caller's stack frame.

    Args:
        fn: The TAC function to translate.
        symbol_table: Used to derive the assembly type for each parameter.
        constants: Mutable list into which any required ``StaticConstant`` nodes
            (e.g. the ``DoubleToUInt`` threshold) are appended.
        struct_type_table: Required to classify struct-typed parameters.

    Returns:
        An ``asm.Function`` with the same name and lowered instructions.

    Raises:
        CodeGenerationError: If any instruction cannot be translated.
    """
    instructions: list[asm.Instruction] = []
    stt: StructTypeTable = struct_type_table or {}
    bp = asm.Register(asm.GPRegisters.BP, asm.Width.QWORD)

    # Classify all parameters up-front so register-spill decisions for struct
    # parameters can be made before any instructions are emitted.
    classifications = classify_parameters(fn.params, symbol_table, stt)

    # Determine whether the function returns a MEMORY-classified struct.
    # If so, %rdi holds the hidden return pointer and must be saved before
    # any parameter moves can overwrite it.
    ret_ptr_name: str | None = None
    ret_in_memory_fn = False
    if symbol_table and fn.name in symbol_table:
        fn_t = symbol_table[fn.name].t
        if isinstance(fn_t, FunType):
            _, _, ret_in_memory_fn = classify_return_value(fn_t.ret, stt)
    if ret_in_memory_fn:
        ret_ptr_name = ".__ret_ptr"
        instructions.append(
            asm.Mov(
                src=asm.Register(asm.GPRegisters.DI, asm.Width.QWORD),
                dst=asm.Pseudo(ret_ptr_name),
                assembly_type=asm.Quadword(),
            )
        )

    # For a struct parameter whose classification is not MEMORY but whose
    # required registers are no longer available, fall back to MEMORY.
    # When the function uses a hidden return pointer, DI is already consumed.
    int_regs_left = len(_ABI_REGS) - (1 if ret_in_memory_fn else 0)
    xmm_regs_left = len(_XMM_REGS)
    effective_classes: list[list[EightbyteClass]] = []
    for param, classes in zip(fn.params, classifications):
        param_type = symbol_table[param].t if (symbol_table and param in symbol_table) else None
        if isinstance(param_type, (Structure, Union)) and classes != [EightbyteClass.MEMORY]:
            int_need = sum(1 for c in classes if c == EightbyteClass.INTEGER)
            xmm_need = sum(1 for c in classes if c == EightbyteClass.SSE)
            if int_need > int_regs_left or xmm_need > xmm_regs_left:
                effective_classes.append([EightbyteClass.MEMORY])
            else:
                int_regs_left -= int_need
                xmm_regs_left -= xmm_need
                effective_classes.append(classes)
        else:
            if classes == [EightbyteClass.SSE]:
                xmm_regs_left = max(0, xmm_regs_left - 1)
            elif classes == [EightbyteClass.INTEGER]:
                int_regs_left = max(0, int_regs_left - 1)
            effective_classes.append(classes)

    # When the function returns via hidden pointer, DI (index 0) is already used.
    int_reg_idx = 1 if ret_in_memory_fn else 0
    xmm_reg_idx = 0
    stack_param_idx = 0

    for param, classes in zip(fn.params, effective_classes):
        param_type = symbol_table[param].t if (symbol_table and param in symbol_table) else None

        if isinstance(param_type, (Structure, Union)):
            if classes == [EightbyteClass.MEMORY]:
                # Struct passed entirely on the stack; copy each eightbyte to a local PseudoMem.
                entry = stt[param_type.tag]
                n = (entry.size + 7) // 8
                base = 16 + 8 * stack_param_idx
                stack_param_idx += n
                for i in range(n):
                    src: asm.Operand = asm.Memory(reg=bp, offset=base + 8 * i)
                    instructions.append(asm.Mov(src=src, dst=asm.PseudoMem(param, 8 * i), assembly_type=asm.Quadword()))
            else:
                # Struct passed in registers; one register move per eightbyte.
                for i, cls in enumerate(classes):
                    if cls == EightbyteClass.SSE:
                        src = asm.Register(_XMM_REGS[xmm_reg_idx], asm.Width.QWORD)
                        xmm_reg_idx += 1
                        at: asm.AssemblyType = asm.Double()
                    else:
                        src = asm.Register(_ABI_REGS[int_reg_idx], asm.Width.QWORD)
                        int_reg_idx += 1
                        at = asm.Quadword()
                    instructions.append(asm.Mov(src=src, dst=asm.PseudoMem(param, 8 * i), assembly_type=at))
        else:
            # Scalar parameter: use the exact assembly type for the move.
            at = _asm_type(tac.Var(param), symbol_table)
            if at == asm.LongDouble():
                # Long double is MEMORY class: always arrives on the stack (16 bytes, 2 eightbytes).
                stack_src = asm.Memory(reg=bp, offset=16 + 8 * stack_param_idx)
                stack_param_idx += 2
                instructions.append(asm.Fldt(src=stack_src))
                instructions.append(asm.Fstpt(dst=asm.PseudoMem(param, 0)))
            elif at == asm.Double():
                if xmm_reg_idx < len(_XMM_REGS):
                    src = asm.Register(_XMM_REGS[xmm_reg_idx], asm.Width.QWORD)
                    xmm_reg_idx += 1
                else:
                    src = asm.Memory(reg=bp, offset=16 + 8 * stack_param_idx)
                    stack_param_idx += 1
                instructions.append(asm.Mov(src=src, dst=asm.Pseudo(param), assembly_type=at))
            else:
                if int_reg_idx < len(_ABI_REGS):
                    src = asm.Register(_ABI_REGS[int_reg_idx], _reg_width(at))
                    int_reg_idx += 1
                else:
                    src = asm.Memory(reg=bp, offset=16 + 8 * stack_param_idx)
                    stack_param_idx += 1
                instructions.append(asm.Mov(src=src, dst=asm.Pseudo(param), assembly_type=at))

    for instr in fn.body:
        instructions.extend(_translate_instruction(instr, symbol_table, constants, stt, ret_ptr_name))

    return asm.Function(
        name=fn.name,
        globl=fn.globl,
        params=(),
        instructions=tuple(instructions),
        source_file=fn.source_file,
        source_line=fn.source_line,
    )


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
            if cmp_at == asm.Float():
                return _translate_float_comparison(op, src1_asm, src2_asm, dst_asm)
            if cmp_at == asm.LongDouble():
                return _translate_long_double_comparison(op, src1_asm, src2_asm, dst_asm)
            unsigned = _is_unsigned(src1, symbol_table)
            return [
                asm.Cmp(op1=src2_asm, op2=src1_asm, assembly_type=cmp_at),
                asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                asm.SetCC(_translate_relational_op(op, unsigned), dst_asm),
            ]
        case tac.Add() | tac.Subtract() | tac.Multiply() | tac.BitwiseAnd() | tac.BitwiseOr() | tac.BitwiseXor():
            at = _asm_type(dst, symbol_table)
            if at == asm.LongDouble():
                # x87: push src1, push src2, compute, store result.
                match op:
                    case tac.Add():
                        x87_op: asm.Instruction = asm.Faddp()
                    case tac.Subtract():
                        x87_op = asm.Fsubp()
                    case tac.Multiply():
                        x87_op = asm.Fmulp()
                    case _:
                        raise CodeGenerationError(f"unsupported LD binary: {type(op).__name__}")
                return [asm.Fldt(src=src1_asm), asm.Fldt(src=src2_asm), x87_op, asm.Fstpt(dst=dst_asm)]
            if at == asm.Double():
                # FP: Mov(src1, dst) makes the copy visible to the coalescer; Binary
                # uses dst as both src1 and accumulator so the internal movsd is
                # suppressed (src1 == dst).
                return [
                    asm.Mov(src=src1_asm, dst=dst_asm, assembly_type=at),
                    asm.Binary(
                        operator=_translate_binary_op_double(op),
                        src1=dst_asm,
                        src2=src2_asm,
                        dst=dst_asm,
                        assembly_type=at,
                    ),
                ]
            if at == asm.Float():
                return [
                    asm.Mov(src=src1_asm, dst=dst_asm, assembly_type=at),
                    asm.Binary(
                        operator=_translate_binary_op_float(op),
                        src1=dst_asm,
                        src2=src2_asm,
                        dst=dst_asm,
                        assembly_type=at,
                    ),
                ]
            # GP: emit an explicit Mov so the coalescer can merge src1 with dst,
            # then an in-place Binary (src1 == dst suppresses the internal movl/movq).
            return [
                asm.Mov(src=src1_asm, dst=dst_asm, assembly_type=at),
                asm.Binary(
                    operator=_translate_binary_op(op),
                    src1=dst_asm,
                    src2=src2_asm,
                    dst=dst_asm,
                    assembly_type=at,
                ),
            ]
        case tac.LeftShift():
            at = _asm_type(dst, symbol_table)
            r11 = asm.Register(asm.GPRegisters.R11, _reg_width(at))
            cx = asm.Register(asm.GPRegisters.CX, asm.Width.DWORD)
            return [
                asm.Mov(src=src1_asm, dst=r11, assembly_type=at),
                asm.Mov(src=src2_asm, dst=cx, assembly_type=asm.Longword()),
                asm.Shl(operand=r11, assembly_type=at),
                asm.Mov(src=r11, dst=dst_asm, assembly_type=at),
            ]
        case tac.RightShift():
            at = _asm_type(dst, symbol_table)
            r11 = asm.Register(asm.GPRegisters.R11, _reg_width(at))
            cx = asm.Register(asm.GPRegisters.CX, asm.Width.DWORD)
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
            if at == asm.LongDouble():
                return [asm.Fldt(src=src1_asm), asm.Fldt(src=src2_asm), asm.Fdivp(), asm.Fstpt(dst=dst_asm)]
            if at == asm.Double():
                return [
                    asm.Mov(src=src1_asm, dst=dst_asm, assembly_type=at),
                    asm.Binary(
                        operator=asm.DivDouble(),
                        src1=dst_asm,
                        src2=src2_asm,
                        dst=dst_asm,
                        assembly_type=at,
                    ),
                ]
            if at == asm.Float():
                return [
                    asm.Mov(src=src1_asm, dst=dst_asm, assembly_type=at),
                    asm.Binary(
                        operator=asm.DivFloat(),
                        src1=dst_asm,
                        src2=src2_asm,
                        dst=dst_asm,
                        assembly_type=at,
                    ),
                ]
            ax = asm.Register(asm.GPRegisters.AX, _reg_width(at))
            dx = asm.Register(asm.GPRegisters.DX, _reg_width(at))
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
            ax = asm.Register(asm.GPRegisters.AX, _reg_width(at))
            dx = asm.Register(asm.GPRegisters.DX, _reg_width(at))
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


def _ld_zero_const_label(constants: list[asm.TopLevel] | None) -> str:
    """Ensure a 0.0L long-double constant exists in ``constants`` and return its label.

    Args:
        constants: Accumulator for ``StaticConstant`` entries.

    Returns:
        The label string for the 0.0 long-double constant in ``.rodata``.
    """
    label = "ld_const.0_0"
    if constants is not None and not any(
        isinstance(c, asm.StaticConstant) and c.identifier == label for c in constants
    ):
        constants.append(asm.StaticConstant(identifier=label, alignment=16, init=asm.LongDoubleInit(0.0)))
    return label


def _translate_instruction(
    instr: tac.Instruction,
    symbol_table: SymbolTable | None = None,
    constants: list[asm.TopLevel] | None = None,
    struct_type_table: StructTypeTable | None = None,
    ret_ptr_name: str | None = None,
) -> list[asm.Instruction]:
    """Lower a single TAC instruction to a sequence of assembly instructions.

    Args:
        instr: The TAC instruction to translate.
        symbol_table: Used to derive assembly types for variable operands.
        constants: Mutable list into which any required ``StaticConstant`` nodes
            are appended (used by ``DoubleToUInt`` for the 2^63 threshold).
        struct_type_table: Required to classify struct arguments and return
            values in ``FunCall`` instructions.
        ret_ptr_name: Name of the pseudo holding the hidden return pointer for
            MEMORY-classified struct returns.  ``None`` when the function does
            not use the hidden-pointer convention.

    Returns:
        A list of ``asm.Instruction`` nodes equivalent to the TAC instruction.

    Raises:
        CodeGenerationError: If the instruction variant is not supported.
    """
    match instr:
        case tac.Return(val=val):
            if val is None:
                return [asm.Ret()]
            if isinstance(val, tac.Var) and symbol_table is not None and val.value in symbol_table:
                _ret_t = symbol_table[val.value].t
                if isinstance(_ret_t, (Structure, Union)) and struct_type_table is not None:
                    return _translate_struct_return(val.value, _ret_t, symbol_table, struct_type_table, ret_ptr_name)
            at = _asm_type(val, symbol_table)
            val_asm = _translate_val(val, constants, symbol_table)
            if at == asm.LongDouble():
                return [asm.Fldt(src=val_asm), asm.Ret()]
            if at == asm.Double():
                xmm0 = asm.Register(asm.FPRegisters.XMM0, asm.Width.QWORD)
                return [asm.Mov(src=val_asm, dst=xmm0, assembly_type=at), asm.Ret()]
            ax = asm.Register(asm.GPRegisters.AX, _reg_width(at))
            return [asm.Mov(src=val_asm, dst=ax, assembly_type=at), asm.Ret()]
        case tac.Jump(target=target):
            return [asm.Jmp(target)]
        case tac.JumpIfZero(condition=condition, target=target):
            at = _asm_type(condition, symbol_table)
            cond_asm = _translate_val(condition, constants, symbol_table)
            if at == asm.LongDouble():
                zero_label = _ld_zero_const_label(constants)
                skip_label = f"nan_jz.{_fresh_uid()}"
                return [
                    asm.Fldt(src=asm.Data(zero_label)),
                    asm.Fldt(src=cond_asm),
                    asm.Fucomip(),
                    asm.Fstp0(),
                    asm.JmpCC(asm.CondCodes.P, skip_label),
                    asm.JmpCC(asm.CondCodes.E, target),
                    asm.Label(skip_label),
                ]
            op1 = _zero_operand_for(at, constants)
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
            cond_asm = _translate_val(condition, constants, symbol_table)
            if at == asm.LongDouble():
                zero_label = _ld_zero_const_label(constants)
                return [
                    asm.Fldt(src=asm.Data(zero_label)),
                    asm.Fldt(src=cond_asm),
                    asm.Fucomip(),
                    asm.Fstp0(),
                    asm.JmpCC(asm.CondCodes.P, target),
                    asm.JmpCC(asm.CondCodes.NE, target),
                ]
            op1 = _zero_operand_for(at, constants)
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
            # Struct copy: generate chunked memory moves for the full struct size.
            if (
                isinstance(src, tac.Var)
                and isinstance(dst, tac.Var)
                and symbol_table is not None
                and src.value in symbol_table
                and isinstance(symbol_table[src.value].t, (Structure, Union))
                and struct_type_table is not None
            ):
                src_t = symbol_table[src.value].t
                assert isinstance(src_t, (Structure, Union))
                size = struct_type_table[src_t.tag].size
                src_is_static = isinstance(symbol_table[src.value].attrs, (StaticAttr, FunAttr))
                dst_is_static = dst.value in symbol_table and isinstance(
                    symbol_table[dst.value].attrs, (StaticAttr, FunAttr)
                )
                chunk_instrs: list[asm.Instruction] = []
                offset = 0
                while offset < size:
                    remaining = size - offset
                    if remaining >= 8:
                        chunk, at_chunk = 8, asm.Quadword()
                        reg = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
                    elif remaining >= 4:
                        chunk, at_chunk = 4, asm.Longword()
                        reg = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
                    else:
                        chunk, at_chunk = 1, asm.Byte()
                        reg = asm.Register(asm.GPRegisters.R10, asm.Width.BYTE)
                    chunk_instrs.append(
                        asm.Mov(
                            src=_struct_src_operand(src.value, offset, src_is_static), dst=reg, assembly_type=at_chunk
                        )
                    )
                    chunk_instrs.append(
                        asm.Mov(
                            src=reg, dst=_struct_src_operand(dst.value, offset, dst_is_static), assembly_type=at_chunk
                        )
                    )
                    offset += chunk
                return chunk_instrs
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
                    if src_at == asm.LongDouble():
                        nan_label = f"nan_not.{_fresh_uid()}"
                        ld_zero = _ld_zero_const_label(constants)
                        return [
                            asm.Fldt(src=asm.Data(ld_zero)),
                            asm.Fldt(src=src_asm),
                            asm.Fucomip(),
                            asm.Fstp0(),
                            asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                            asm.JmpCC(asm.CondCodes.P, nan_label),
                            asm.SetCC(asm.CondCodes.E, dst_asm),
                            asm.Label(nan_label),
                        ]
                    zero = _zero_operand_for(src_at, constants)
                    if src_at in (asm.Double(), asm.Float()):
                        # ucomiss/ucomisd sets PF=1 and ZF=1 for NaN; sete fires on ZF=1 so would
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
                case tac.Negate() if at == asm.LongDouble():
                    return [asm.Fldt(src=src_asm), asm.Fchs(), asm.Fstpt(dst=dst_asm)]
                case tac.Negate() if at == asm.Double():
                    # IEEE 754 negation flips only the sign bit; use xorpd with -0.0 (sign bit set).
                    # xorpd requires 16-byte alignment; if a lower-aligned entry already exists
                    # (e.g. from constant folding emitting ConstDouble(-0.0) via _translate_val),
                    # upgrade it in place rather than leaving the misaligned version.
                    sign_mask_bits = asm.DoubleInit(-0.0).as_bits()
                    sign_mask_label = f"double_const.{sign_mask_bits}"
                    if constants is not None:
                        existing_idx = next(
                            (
                                i
                                for i, c in enumerate(constants)
                                if isinstance(c, asm.StaticConstant) and c.identifier == sign_mask_label
                            ),
                            -1,
                        )
                        if existing_idx == -1:
                            constants.append(
                                asm.StaticConstant(identifier=sign_mask_label, alignment=16, init=asm.DoubleInit(-0.0))
                            )
                        else:
                            existing = constants[existing_idx]
                            if isinstance(existing, asm.StaticConstant) and existing.alignment < 16:
                                constants[existing_idx] = asm.StaticConstant(
                                    identifier=sign_mask_label, alignment=16, init=asm.DoubleInit(-0.0)
                                )
                    xmm14 = asm.Register(asm.FPRegisters.XMM14, asm.Width.QWORD)
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
                case tac.Negate() if at == asm.Float():
                    # Same as double negation but uses xorps and FloatInit(-0.0f) sign mask.
                    sign_mask_bits = asm.FloatInit(-0.0).as_bits()
                    sign_mask_label = f"float_const.{sign_mask_bits}"
                    if constants is not None:
                        existing_idx = next(
                            (
                                i
                                for i, c in enumerate(constants)
                                if isinstance(c, asm.StaticConstant) and c.identifier == sign_mask_label
                            ),
                            -1,
                        )
                        if existing_idx == -1:
                            constants.append(
                                asm.StaticConstant(identifier=sign_mask_label, alignment=16, init=asm.FloatInit(-0.0))
                            )
                        else:
                            existing = constants[existing_idx]
                            if isinstance(existing, asm.StaticConstant) and existing.alignment < 16:
                                constants[existing_idx] = asm.StaticConstant(
                                    identifier=sign_mask_label, alignment=16, init=asm.FloatInit(-0.0)
                                )
                    xmm14 = asm.Register(asm.FPRegisters.XMM14, asm.Width.QWORD)
                    return [
                        asm.Binary(
                            operator=asm.XorFloat(),
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
            dst_asm = _translate_val(dst, None, symbol_table) if dst is not None else None
            return _translate_funcall(fun_val, list(args), dst_asm, symbol_table, constants, struct_type_table)
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
                r10d = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
                r10b = asm.Register(asm.GPRegisters.R10, asm.Width.BYTE)
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
                r10d = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
                xmm14 = asm.Register(asm.FPRegisters.XMM14, asm.Width.QWORD)
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
        case tac.FloatToInt(src=src, dst=dst):
            at = _asm_type(dst, symbol_table)
            if at == asm.Byte():
                r10d = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
                r10b = asm.Register(asm.GPRegisters.R10, asm.Width.BYTE)
                return [
                    asm.Cvttss2si(
                        src=_translate_val(src, constants, symbol_table), dst=r10d, assembly_type=asm.Longword()
                    ),
                    asm.Mov(src=r10b, dst=_translate_val(dst, None, symbol_table), assembly_type=asm.Byte()),
                ]
            return [
                asm.Cvttss2si(
                    src=_translate_val(src, constants, symbol_table),
                    dst=_translate_val(dst, None, symbol_table),
                    assembly_type=at,
                )
            ]
        case tac.IntToFloat(src=src, dst=dst):
            at = _asm_type(src, symbol_table)
            if at == asm.Byte():
                r10d = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
                xmm14 = asm.Register(asm.FPRegisters.XMM14, asm.Width.QWORD)
                return [
                    asm.Movsx(
                        src_type=asm.Byte(),
                        dst_type=asm.Longword(),
                        src=_translate_val(src, None, symbol_table),
                        dst=r10d,
                    ),
                    asm.Cvtsi2ss(src=r10d, dst=xmm14, assembly_type=asm.Longword()),
                    asm.Mov(src=xmm14, dst=_translate_val(dst, None, symbol_table), assembly_type=asm.Float()),
                ]
            return [
                asm.Cvtsi2ss(
                    src=_translate_val(src, constants, symbol_table),
                    dst=_translate_val(dst, None, symbol_table),
                    assembly_type=at,
                )
            ]
        case tac.UIntToFloat(src=src, dst=dst):
            return _translate_uint_to_float(src, dst, symbol_table)
        case tac.FloatToUInt(src=src, dst=dst):
            return _translate_float_to_uint(src, dst, symbol_table, constants)
        case tac.FloatToDouble(src=src, dst=dst):
            return [
                asm.Cvtss2sd(
                    src=_translate_val(src, constants, symbol_table),
                    dst=_translate_val(dst, None, symbol_table),
                )
            ]
        case tac.DoubleToFloat(src=src, dst=dst):
            return [
                asm.Cvtsd2ss(
                    src=_translate_val(src, constants, symbol_table),
                    dst=_translate_val(dst, None, symbol_table),
                )
            ]
        case tac.IntToLongDouble(src=src, dst=dst):
            src_at = _asm_type(src, symbol_table)
            src_asm_ild = _translate_val(src, None, symbol_table)
            dst_asm_ild = _translate_val(dst, None, symbol_table)
            r10d = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
            tmp_lw = asm.Pseudo(f".__ld_itmp.{_fresh_uid()}")
            if src_at == asm.Byte():
                return [
                    asm.Movsx(src_type=asm.Byte(), dst_type=asm.Longword(), src=src_asm_ild, dst=r10d),
                    asm.Mov(src=r10d, dst=tmp_lw, assembly_type=asm.Longword()),
                    asm.Fildl(src=tmp_lw),
                    asm.Fstpt(dst=dst_asm_ild),
                ]
            if src_at in (asm.Word(), asm.Longword()):
                return [asm.Fildl(src=src_asm_ild), asm.Fstpt(dst=dst_asm_ild)]
            # Quadword (int64/long)
            return [asm.Fildq(src=src_asm_ild), asm.Fstpt(dst=dst_asm_ild)]
        case tac.UIntToLongDouble(src=src, dst=dst):
            src_at = _asm_type(src, symbol_table)
            src_asm_uld = _translate_val(src, None, symbol_table)
            dst_asm_uld = _translate_val(dst, None, symbol_table)
            r10d = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
            r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
            tmp_qw2 = asm.Pseudo(f".__ld_itmp.{_fresh_uid()}")
            if src_at == asm.Byte():
                # Zero-extend to int64 via movzbl to r10d (upper 32 bits clear)
                return [
                    asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=asm.Longword(), src=src_asm_uld, dst=r10d),
                    asm.Mov(src=r10q, dst=tmp_qw2, assembly_type=asm.Quadword()),
                    asm.Fildq(src=tmp_qw2),
                    asm.Fstpt(dst=dst_asm_uld),
                ]
            if src_at == asm.Word():
                return [
                    asm.MoveZeroExtend(src_type=asm.Word(), dst_type=asm.Longword(), src=src_asm_uld, dst=r10d),
                    asm.Mov(src=r10q, dst=tmp_qw2, assembly_type=asm.Quadword()),
                    asm.Fildq(src=tmp_qw2),
                    asm.Fstpt(dst=dst_asm_uld),
                ]
            if src_at == asm.Longword():
                # Zero-extend via movl r10d (writes to r10d, clears upper 32 bits of r10q)
                return [
                    asm.Mov(src=src_asm_uld, dst=r10d, assembly_type=asm.Longword()),
                    asm.Mov(src=r10q, dst=tmp_qw2, assembly_type=asm.Quadword()),
                    asm.Fildq(src=tmp_qw2),
                    asm.Fstpt(dst=dst_asm_uld),
                ]
            # uint64: branch on high bit (same approach as UIntToDouble)
            return _translate_uint_to_long_double(src, dst, symbol_table, constants)
        case tac.LongDoubleToInt(src=src, dst=dst):
            src_asm_ldi = _translate_val(src, None, symbol_table)
            dst_asm_ldi = _translate_val(dst, None, symbol_table)
            dst_at = _asm_type(dst, symbol_table)
            r10d = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
            r10b = asm.Register(asm.GPRegisters.R10, asm.Width.BYTE)
            if dst_at == asm.Byte():
                tmp_lw2 = asm.Pseudo(f".__ld_itmp.{_fresh_uid()}")
                return [
                    asm.Fldt(src=src_asm_ldi),
                    asm.Fisttpl(dst=tmp_lw2),
                    asm.Mov(src=r10b, dst=dst_asm_ldi, assembly_type=asm.Byte()),
                ]
            if dst_at in (asm.Word(), asm.Longword()):
                return [asm.Fldt(src=src_asm_ldi), asm.Fisttpl(dst=dst_asm_ldi)]
            return [asm.Fldt(src=src_asm_ldi), asm.Fisttpq(dst=dst_asm_ldi)]
        case tac.LongDoubleToUInt(src=src, dst=dst):
            src_asm_ldu = _translate_val(src, None, symbol_table)
            dst_asm_ldu = _translate_val(dst, None, symbol_table)
            dst_at = _asm_type(dst, symbol_table)
            r10d = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
            r10b = asm.Register(asm.GPRegisters.R10, asm.Width.BYTE)
            r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
            if dst_at == asm.Byte():
                tmp_lw3 = asm.Pseudo(f".__ld_itmp.{_fresh_uid()}")
                return [
                    asm.Fldt(src=src_asm_ldu),
                    asm.Fisttpl(dst=tmp_lw3),
                    asm.Mov(src=r10b, dst=dst_asm_ldu, assembly_type=asm.Byte()),
                ]
            if dst_at == asm.Longword():
                tmp_q3 = asm.Pseudo(f".__ld_itmp.{_fresh_uid()}")
                return [
                    asm.Fldt(src=src_asm_ldu),
                    asm.Fisttpq(dst=tmp_q3),
                    asm.Mov(src=r10d, dst=dst_asm_ldu, assembly_type=asm.Longword()),
                ]
            return [asm.Fldt(src=src_asm_ldu), asm.Fisttpq(dst=dst_asm_ldu)]
        case tac.FloatToLongDouble(src=src, dst=dst):
            return [
                asm.Flds(src=_translate_val(src, constants, symbol_table)),
                asm.Fstpt(dst=_translate_val(dst, None, symbol_table)),
            ]
        case tac.LongDoubleToFloat(src=src, dst=dst):
            return [
                asm.Fldt(src=_translate_val(src, None, symbol_table)),
                asm.Fstps(dst=_translate_val(dst, None, symbol_table)),
            ]
        case tac.DoubleToLongDouble(src=src, dst=dst):
            return [
                asm.Fldl(src=_translate_val(src, constants, symbol_table)),
                asm.Fstpt(dst=_translate_val(dst, None, symbol_table)),
            ]
        case tac.LongDoubleToDouble(src=src, dst=dst):
            return [
                asm.Fldt(src=_translate_val(src, None, symbol_table)),
                asm.Fstpl(dst=_translate_val(dst, None, symbol_table)),
            ]
        case tac.Load(src_ptr=src_ptr, dst=dst):
            ax = asm.Register(asm.GPRegisters.AX, asm.Width.QWORD)
            load_ptr = asm.Mov(
                src=_translate_val(src_ptr, constants, symbol_table), dst=ax, assembly_type=asm.Quadword()
            )
            # Struct load: copy each chunk from *ptr into the destination struct.
            if (
                isinstance(dst, tac.Var)
                and symbol_table is not None
                and dst.value in symbol_table
                and isinstance(symbol_table[dst.value].t, (Structure, Union))
                and struct_type_table is not None
            ):
                dst_t = symbol_table[dst.value].t
                assert isinstance(dst_t, (Structure, Union))
                size = struct_type_table[dst_t.tag].size
                dst_is_static = isinstance(symbol_table[dst.value].attrs, (StaticAttr, FunAttr))
                load_instrs: list[asm.Instruction] = [load_ptr]
                i = 0
                while i < size:
                    remaining = size - i
                    if remaining >= 8:
                        chunk, at_c = 8, asm.Quadword()
                        reg = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
                    elif remaining >= 4:
                        chunk, at_c = 4, asm.Longword()
                        reg = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
                    else:
                        chunk, at_c = 1, asm.Byte()
                        reg = asm.Register(asm.GPRegisters.R10, asm.Width.BYTE)
                    load_instrs.append(asm.Mov(src=asm.Memory(ax, i), dst=reg, assembly_type=at_c))
                    load_instrs.append(
                        asm.Mov(src=reg, dst=_struct_src_operand(dst.value, i, dst_is_static), assembly_type=at_c)
                    )
                    i += chunk
                return load_instrs
            at = _asm_type(dst, symbol_table)
            dst_asm = _translate_val(dst, constants, symbol_table)
            return [load_ptr, asm.Mov(src=asm.Memory(reg=ax, offset=0), dst=dst_asm, assembly_type=at)]
        case tac.Store(src=src, dst_ptr=dst_ptr):
            ax = asm.Register(asm.GPRegisters.AX, asm.Width.QWORD)
            store_ptr = asm.Mov(
                src=_translate_val(dst_ptr, constants, symbol_table), dst=ax, assembly_type=asm.Quadword()
            )
            # Struct store: copy each chunk from the source struct through the pointer.
            if (
                isinstance(src, tac.Var)
                and symbol_table is not None
                and src.value in symbol_table
                and isinstance(symbol_table[src.value].t, (Structure, Union))
                and struct_type_table is not None
            ):
                src_t = symbol_table[src.value].t
                assert isinstance(src_t, (Structure, Union))
                size = struct_type_table[src_t.tag].size
                src_is_static = isinstance(symbol_table[src.value].attrs, (StaticAttr, FunAttr))
                store_instrs: list[asm.Instruction] = [store_ptr]
                i = 0
                while i < size:
                    remaining = size - i
                    if remaining >= 8:
                        chunk, at_c = 8, asm.Quadword()
                        reg = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
                    elif remaining >= 4:
                        chunk, at_c = 4, asm.Longword()
                        reg = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
                    else:
                        chunk, at_c = 1, asm.Byte()
                        reg = asm.Register(asm.GPRegisters.R10, asm.Width.BYTE)
                    store_instrs.append(
                        asm.Mov(src=_struct_src_operand(src.value, i, src_is_static), dst=reg, assembly_type=at_c)
                    )
                    store_instrs.append(asm.Mov(src=reg, dst=asm.Memory(ax, i), assembly_type=at_c))
                    i += chunk
                return store_instrs
            at = _asm_type(src, symbol_table)
            src_asm = _translate_val(src, constants, symbol_table)
            return [store_ptr, asm.Mov(src=src_asm, dst=asm.Memory(reg=ax, offset=0), assembly_type=at)]
        case tac.GetAddress(src=src, dst=dst):
            src_asm = _translate_val(src, constants, symbol_table)
            if isinstance(src_asm, (asm.Imm, asm.Register)):
                raise CodeGenerationError(
                    f"GetAddress requires an addressable memory operand, got {type(src_asm).__name__}"
                )
            return [asm.Lea(src=src_asm, dst=_translate_val(dst, constants, symbol_table))]
        case tac.CopyToOffset(src=src, dst=dst_name, offset=offset):
            # If src is a struct, generate chunked copies into dst at the given offset.
            if (
                isinstance(src, tac.Var)
                and symbol_table is not None
                and src.value in symbol_table
                and isinstance(symbol_table[src.value].t, (Structure, Union))
                and struct_type_table is not None
            ):
                src_t = symbol_table[src.value].t
                assert isinstance(src_t, (Structure, Union))
                size = struct_type_table[src_t.tag].size
                src_is_static = isinstance(symbol_table[src.value].attrs, (StaticAttr, FunAttr))
                dst_is_static = bool(
                    symbol_table
                    and dst_name in symbol_table
                    and isinstance(symbol_table[dst_name].attrs, (StaticAttr, FunAttr))
                )
                cto_instrs: list[asm.Instruction] = []
                i = 0
                while i < size:
                    remaining = size - i
                    if remaining >= 8:
                        chunk, at_c = 8, asm.Quadword()
                        reg = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
                    elif remaining >= 4:
                        chunk, at_c = 4, asm.Longword()
                        reg = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
                    else:
                        chunk, at_c = 1, asm.Byte()
                        reg = asm.Register(asm.GPRegisters.R10, asm.Width.BYTE)
                    cto_instrs.append(
                        asm.Mov(src=_struct_src_operand(src.value, i, src_is_static), dst=reg, assembly_type=at_c)
                    )
                    cto_instrs.append(
                        asm.Mov(
                            src=reg,
                            dst=_struct_src_operand(dst_name, offset + i, dst_is_static),
                            assembly_type=at_c,
                        )
                    )
                    i += chunk
                return cto_instrs
            at = _asm_type(src, symbol_table)
            src_asm = _translate_val(src, constants, symbol_table)
            return [asm.Mov(src=src_asm, dst=asm.PseudoMem(dst_name, offset), assembly_type=at)]
        case tac.CopyFromOffset(src=src_name, offset=offset, dst=dst):
            # If dst is a struct, generate chunked copies from src at the given offset.
            if (
                isinstance(dst, tac.Var)
                and symbol_table is not None
                and dst.value in symbol_table
                and isinstance(symbol_table[dst.value].t, (Structure, Union))
                and struct_type_table is not None
            ):
                dst_t = symbol_table[dst.value].t
                assert isinstance(dst_t, (Structure, Union))
                size = struct_type_table[dst_t.tag].size
                src_is_static = bool(
                    symbol_table
                    and src_name in symbol_table
                    and isinstance(symbol_table[src_name].attrs, (StaticAttr, FunAttr))
                )
                dst_is_static = isinstance(symbol_table[dst.value].attrs, (StaticAttr, FunAttr))
                cfo_instrs: list[asm.Instruction] = []
                i = 0
                while i < size:
                    remaining = size - i
                    if remaining >= 8:
                        chunk, at_c = 8, asm.Quadword()
                        reg = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
                    elif remaining >= 4:
                        chunk, at_c = 4, asm.Longword()
                        reg = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
                    else:
                        chunk, at_c = 1, asm.Byte()
                        reg = asm.Register(asm.GPRegisters.R10, asm.Width.BYTE)
                    cfo_instrs.append(
                        asm.Mov(
                            src=_struct_src_operand(src_name, offset + i, src_is_static), dst=reg, assembly_type=at_c
                        )
                    )
                    cfo_instrs.append(
                        asm.Mov(src=reg, dst=_struct_src_operand(dst.value, i, dst_is_static), assembly_type=at_c)
                    )
                    i += chunk
                return cfo_instrs
            at = _asm_type(dst, symbol_table)
            dst_asm = _translate_val(dst, constants, symbol_table)
            return [asm.Mov(src=asm.PseudoMem(src_name, offset), dst=dst_asm, assembly_type=at)]
        case tac.AddPtr(ptr=ptr, index=index, scale=scale, dst=dst):
            r10 = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
            r11 = asm.Register(asm.GPRegisters.R11, asm.Width.QWORD)
            ptr_asm = _translate_val(ptr, constants, symbol_table)
            dst_asm = _translate_val(dst, constants, symbol_table)
            # When the index is a compile-time integer constant, fold index * scale
            # into a fixed displacement and emit a single leaq without any imul.
            match index:
                case tac.Constant(
                    value=ConstInt(val=idx_v)
                    | ConstLong(val=idx_v)
                    | ConstLongLong(val=idx_v)
                    | ConstUInt(val=idx_v)
                    | ConstULong(val=idx_v)
                    | ConstULongLong(val=idx_v)
                ):
                    return [
                        asm.Mov(src=ptr_asm, dst=r11, assembly_type=asm.Quadword()),
                        asm.Lea(src=asm.Memory(reg=r11, offset=idx_v * scale), dst=dst_asm),
                    ]
            index_asm = _translate_val(index, constants, symbol_table)
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
        case tac.GetBitField(
            src=bf_src,
            byte_offset=bf_byte_off,
            bit_offset=bf_bit_off,
            bit_width=bf_width,
            unit_size=bf_unit_size,
            signed=bf_signed,
            dst=bf_dst,
        ):
            return _lower_get_bit_field(
                bf_src,
                bf_byte_off,
                bf_bit_off,
                bf_width,
                bf_unit_size,
                bf_signed,
                bf_dst,
                symbol_table,
                constants,
            )
        case tac.SetBitField(
            src=bf_src_val,
            dst=bf_dst_name,
            byte_offset=bf_byte_off,
            bit_offset=bf_bit_off,
            bit_width=bf_width,
            unit_size=bf_unit_size,
        ):
            return _lower_set_bit_field(
                bf_src_val,
                bf_dst_name,
                bf_byte_off,
                bf_bit_off,
                bf_width,
                bf_unit_size,
                symbol_table,
                constants,
            )
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

    r10d = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
    r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
    r11q = asm.Register(asm.GPRegisters.R11, asm.Width.QWORD)
    rcxq = asm.Register(asm.GPRegisters.CX, asm.Width.QWORD)
    xmm14 = asm.Register(asm.FPRegisters.XMM14, asm.Width.QWORD)

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
        # jge (SF == OF after cmpq $0) <-> bit 63 clear <-> value < 2^63
        asm.Cmp(op1=asm.Imm(0), op2=r11q, assembly_type=asm.Quadword()),
        asm.JmpCC(asm.CondCodes.GE, simple_label),
        # Complex path: src >= 2^63 - halve, convert, double back.
        asm.Mov(src=r11q, dst=r10q, assembly_type=asm.Quadword()),
        asm.Mov(src=asm.Imm(1), dst=rcxq, assembly_type=asm.Quadword()),
        asm.Shr(operand=r10q, assembly_type=asm.Quadword()),
        # r11 &= 1  - preserve the LSB to avoid double-rounding
        asm.Binary(operator=asm.And(), src1=r11q, src2=asm.Imm(1), dst=r11q, assembly_type=asm.Quadword()),
        # r10 |= r11  - (src >> 1) | (src & 1)
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


def _translate_uint_to_long_double(
    src: tac.Val,
    dst: tac.Val,
    symbol_table: SymbolTable | None = None,
    constants: list[asm.TopLevel] | None = None,
) -> list[asm.Instruction]:
    """Lower a uint64 ``UIntToLongDouble`` TAC instruction to assembly.

    Values with bit 63 clear fit in signed int64 and are loaded directly via
    ``fildq``.  Values with bit 63 set are halved (preserving the LSB to avoid
    double-rounding) using the same ``(src >> 1) | (src & 1)`` trick as
    ``_translate_uint_to_double``, loaded twice, then added (doubling).

    Args:
        src: The uint64 TAC source value.
        dst: The long double TAC destination value.
        symbol_table: Used to derive the assembly type of the source.
        constants: Accumulator for emitted ``StaticConstant`` nodes (unused here,
            present for API consistency).

    Returns:
        A list of ``asm.Instruction`` nodes implementing the conversion.
    """
    src_asm = _translate_val(src, None, symbol_table)
    dst_asm = _translate_val(dst, None, symbol_table)

    r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
    r11q = asm.Register(asm.GPRegisters.R11, asm.Width.QWORD)
    rcxq = asm.Register(asm.GPRegisters.CX, asm.Width.QWORD)
    tmp = asm.Pseudo(f".__ld_u64tmp.{_fresh_uid()}")

    uid = _fresh_uid()
    simple_label = f"uint_to_ld_simple.{uid}"
    done_label = f"uint_to_ld_done.{uid}"

    return [
        asm.Mov(src=src_asm, dst=r11q, assembly_type=asm.Quadword()),
        asm.Cmp(op1=asm.Imm(0), op2=r11q, assembly_type=asm.Quadword()),
        asm.JmpCC(asm.CondCodes.GE, simple_label),
        # Complex path: src >= 2^63 — halve, preserve LSB, load twice, add.
        asm.Mov(src=r11q, dst=r10q, assembly_type=asm.Quadword()),
        asm.Mov(src=asm.Imm(1), dst=rcxq, assembly_type=asm.Quadword()),
        asm.Shr(operand=r10q, assembly_type=asm.Quadword()),
        asm.Binary(operator=asm.And(), src1=r11q, src2=asm.Imm(1), dst=r11q, assembly_type=asm.Quadword()),
        asm.Binary(operator=asm.Or(), src1=r10q, src2=r11q, dst=r10q, assembly_type=asm.Quadword()),
        asm.Mov(src=r10q, dst=tmp, assembly_type=asm.Quadword()),
        asm.Fildq(src=tmp),
        asm.Fildq(src=tmp),
        asm.Faddp(),
        asm.Fstpt(dst=dst_asm),
        asm.Jmp(done_label),
        # Simple path: bit 63 clear, value fits in signed int64.
        asm.Label(simple_label),
        asm.Mov(src=r11q, dst=tmp, assembly_type=asm.Quadword()),
        asm.Fildq(src=tmp),
        asm.Fstpt(dst=dst_asm),
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

    r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
    r10d = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
    r10b = asm.Register(asm.GPRegisters.R10, asm.Width.BYTE)

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
    r11q = asm.Register(asm.GPRegisters.R11, asm.Width.QWORD)
    xmm14 = asm.Register(asm.FPRegisters.XMM14, asm.Width.QWORD)

    return [
        # Load src into xmm14; ucomisd requires op2 to be an XMM register.
        asm.Mov(src=src_asm, dst=xmm14, assembly_type=asm.Double()),
        # ucomisd threshold(%rip), xmm14 - AE if xmm14 >= threshold (= 2^63)
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


_FLOAT_TO_ULONG_THRESHOLD = 9_223_372_036_854_775_808.0


def _translate_uint_to_float(
    src: tac.Val,
    dst: tac.Val,
    symbol_table: SymbolTable | None = None,
) -> list[asm.Instruction]:
    """Lower a ``UIntToFloat`` TAC instruction to assembly.

    For an 8-bit unsigned source: zero-extend to Longword then convert with ``cvtsi2ssl``.

    For a 32-bit unsigned source: zero-extend to 64 bits via ``movl``, then convert the
    non-negative 64-bit value with ``cvtsi2ssq``.

    For a 64-bit unsigned source: values < 2^63 are converted directly; values >= 2^63 are
    halved (preserving the LSB), converted, then doubled back using ``addss``.

    Args:
        src: The unsigned-integer TAC source value.
        dst: The float TAC destination value.
        symbol_table: Used to derive the assembly type of the source.

    Returns:
        A list of ``asm.Instruction`` nodes implementing the conversion.
    """
    src_at = _asm_type(src, symbol_table)
    src_asm = _translate_val(src)
    dst_asm = _translate_val(dst)

    r10d = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
    r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
    r11q = asm.Register(asm.GPRegisters.R11, asm.Width.QWORD)
    rcxq = asm.Register(asm.GPRegisters.CX, asm.Width.QWORD)
    xmm14 = asm.Register(asm.FPRegisters.XMM14, asm.Width.QWORD)

    if src_at == asm.Byte():
        return [
            asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=asm.Longword(), src=src_asm, dst=r10d),
            asm.Cvtsi2ss(src=r10d, dst=xmm14, assembly_type=asm.Longword()),
            asm.Mov(src=xmm14, dst=dst_asm, assembly_type=asm.Float()),
        ]

    if src_at == asm.Longword():
        return [
            asm.Mov(src=src_asm, dst=r10d, assembly_type=asm.Longword()),
            asm.Cvtsi2ss(src=r10q, dst=xmm14, assembly_type=asm.Quadword()),
            asm.Mov(src=xmm14, dst=dst_asm, assembly_type=asm.Float()),
        ]

    uid = _fresh_uid()
    simple_label = f"uint_to_flt_simple.{uid}"
    done_label = f"uint_to_flt_done.{uid}"

    return [
        asm.Mov(src=src_asm, dst=r11q, assembly_type=asm.Quadword()),
        asm.Cmp(op1=asm.Imm(0), op2=r11q, assembly_type=asm.Quadword()),
        asm.JmpCC(asm.CondCodes.GE, simple_label),
        asm.Mov(src=r11q, dst=r10q, assembly_type=asm.Quadword()),
        asm.Mov(src=asm.Imm(1), dst=rcxq, assembly_type=asm.Quadword()),
        asm.Shr(operand=r10q, assembly_type=asm.Quadword()),
        asm.Binary(operator=asm.And(), src1=r11q, src2=asm.Imm(1), dst=r11q, assembly_type=asm.Quadword()),
        asm.Binary(operator=asm.Or(), src1=r10q, src2=r11q, dst=r10q, assembly_type=asm.Quadword()),
        asm.Cvtsi2ss(src=r10q, dst=xmm14, assembly_type=asm.Quadword()),
        asm.Binary(operator=asm.AddFloat(), src1=xmm14, src2=xmm14, dst=xmm14, assembly_type=asm.Float()),
        asm.Mov(src=xmm14, dst=dst_asm, assembly_type=asm.Float()),
        asm.Jmp(done_label),
        asm.Label(simple_label),
        asm.Cvtsi2ss(src=r11q, dst=xmm14, assembly_type=asm.Quadword()),
        asm.Mov(src=xmm14, dst=dst_asm, assembly_type=asm.Float()),
        asm.Label(done_label),
    ]


def _translate_float_to_uint(
    src: tac.Val,
    dst: tac.Val,
    symbol_table: SymbolTable | None = None,
    constants: list[asm.TopLevel] | None = None,
) -> list[asm.Instruction]:
    """Lower a ``FloatToUInt`` TAC instruction to assembly.

    For an 8-bit unsigned destination: convert via ``cvttss2sil`` to int32 then truncate with ``movb``.

    For a 32-bit unsigned destination: convert via ``cvttss2siq`` to int64 then truncate with ``movl``.

    For a 64-bit unsigned destination: values < 2^63 are converted directly; values >= 2^63
    subtract the float threshold (2^63), convert, then add 2^63 back as unsigned.

    Args:
        src: The float TAC source value.
        dst: The unsigned-integer TAC destination value.
        symbol_table: Used to derive the assembly type of the destination.
        constants: Mutable list to which the 2^63 threshold ``StaticConstant`` is appended when needed.

    Returns:
        A list of ``asm.Instruction`` nodes implementing the conversion.
    """
    dst_at = _asm_type(dst, symbol_table)
    src_asm = _translate_val(src, constants)
    dst_asm = _translate_val(dst, constants)

    r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
    r10d = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
    r10b = asm.Register(asm.GPRegisters.R10, asm.Width.BYTE)

    if dst_at == asm.Byte():
        return [
            asm.Cvttss2si(src=src_asm, dst=r10d, assembly_type=asm.Longword()),
            asm.Mov(src=r10b, dst=dst_asm, assembly_type=asm.Byte()),
        ]

    if dst_at == asm.Longword():
        return [
            asm.Cvttss2si(src=src_asm, dst=r10q, assembly_type=asm.Quadword()),
            asm.Mov(src=r10d, dst=dst_asm, assembly_type=asm.Longword()),
        ]

    _THRESHOLD_LABEL = "float_to_ulong_threshold"
    if constants is not None and not any(
        isinstance(c, asm.StaticConstant) and c.identifier == _THRESHOLD_LABEL for c in constants
    ):
        constants.append(
            asm.StaticConstant(identifier=_THRESHOLD_LABEL, alignment=16, init=asm.FloatInit(_FLOAT_TO_ULONG_THRESHOLD))
        )

    uid = _fresh_uid()
    large_label = f"flt_to_ulong_large.{uid}"
    done_label = f"flt_to_ulong_done.{uid}"

    threshold_op = asm.Data(_THRESHOLD_LABEL)
    r11q = asm.Register(asm.GPRegisters.R11, asm.Width.QWORD)
    xmm14 = asm.Register(asm.FPRegisters.XMM14, asm.Width.QWORD)

    return [
        asm.Mov(src=src_asm, dst=xmm14, assembly_type=asm.Float()),
        asm.Cmp(op1=threshold_op, op2=xmm14, assembly_type=asm.Float()),
        asm.JmpCC(asm.CondCodes.AE, large_label),
        asm.Cvttss2si(src=xmm14, dst=dst_asm, assembly_type=asm.Quadword()),
        asm.Jmp(done_label),
        asm.Label(large_label),
        asm.Binary(operator=asm.SubFloat(), src1=xmm14, src2=threshold_op, dst=xmm14, assembly_type=asm.Float()),
        asm.Cvttss2si(src=xmm14, dst=r10q, assembly_type=asm.Quadword()),
        asm.Mov(src=asm.Imm(-(1 << 63)), dst=r11q, assembly_type=asm.Quadword()),
        asm.Binary(operator=asm.Add(), src1=r10q, src2=r11q, dst=r10q, assembly_type=asm.Quadword()),
        asm.Mov(src=r10q, dst=dst_asm, assembly_type=asm.Quadword()),
        asm.Label(done_label),
    ]


def _struct_src_operand(name: str, offset: int, is_static: bool) -> asm.Operand:
    """Return a ``PseudoMem`` or ``Data`` operand for a struct source at the given byte offset.

    Args:
        name: The variable name.
        offset: Byte offset within the struct.
        is_static: ``True`` when the variable has static storage (global / file-scope).

    Returns:
        ``Data(name, offset=offset)`` for static variables; ``PseudoMem(name, offset)``
        for locals.
    """
    return asm.Data(name, offset=offset) if is_static else asm.PseudoMem(name, offset)


def _bf_asm_type(unit_size: int) -> asm.AssemblyType:
    """Return the assembly type that matches a bit-field storage unit size.

    Args:
        unit_size: The storage unit size in bytes (1, 2, 4, or 8).

    Returns:
        The corresponding ``AssemblyType`` instance.

    Raises:
        CodeGenerationError: If ``unit_size`` is not one of the supported sizes.
    """
    match unit_size:
        case 1:
            return asm.Byte()
        case 2:
            return asm.Word()
        case 4:
            return asm.Longword()
        case 8:
            return asm.Quadword()
        case _:
            raise CodeGenerationError(f"unsupported bit-field unit size: {unit_size}")


def _lower_get_bit_field(
    src: str,
    byte_offset: int,
    bit_offset: int,
    bit_width: int,
    unit_size: int,
    signed: bool,
    dst: tac.Val,
    symbol_table: SymbolTable | None,
    constants: dict[str, asm.StaticConstant],
) -> list[asm.Instruction]:
    """Lower a ``GetBitField`` TAC instruction to assembly.

    For unsigned fields: loads the storage unit, shifts right by ``bit_offset``,
    then ANDs with a bit-width mask.
    For signed fields: loads the storage unit, then uses a paired left-shift /
    arithmetic-right-shift to sign-extend the field to the full register width.

    Args:
        src: Name of the aggregate variable.
        byte_offset: Byte offset of the storage unit within ``src``.
        bit_offset: Bit position of the LSB of the field within the storage unit.
        bit_width: Number of bits in the field.
        unit_size: Size in bytes of the storage unit.
        signed: ``True`` for signed fields (sign-extension required).
        dst: TAC destination value.
        symbol_table: Symbol table for operand translation.
        constants: Constants pool for operand translation.

    Returns:
        A list of assembly instructions that extract the bit-field into ``dst``.
    """
    at = _bf_asm_type(unit_size)
    unit_bits = unit_size * 8
    r10 = asm.Register(asm.GPRegisters.R10, _reg_width(at))
    cx = asm.Register(asm.GPRegisters.CX, asm.Width.DWORD)
    is_static = bool(
        symbol_table and src in symbol_table and isinstance(symbol_table[src].attrs, (StaticAttr, FunAttr))
    )
    src_op = _struct_src_operand(src, byte_offset, is_static)
    dst_asm = _translate_val(dst, constants, symbol_table)
    instrs: list[asm.Instruction] = [asm.Mov(src=src_op, dst=r10, assembly_type=at)]

    if signed:
        left_shift = unit_bits - bit_width - bit_offset
        right_shift = unit_bits - bit_width
        if left_shift > 0:
            instrs += [
                asm.Mov(src=asm.Imm(left_shift), dst=cx, assembly_type=asm.Longword()),
                asm.Shl(operand=r10, assembly_type=at),
            ]
        if right_shift > 0:
            instrs += [
                asm.Mov(src=asm.Imm(right_shift), dst=cx, assembly_type=asm.Longword()),
                asm.Sar(operand=r10, assembly_type=at),
            ]
    else:
        if bit_offset > 0:
            instrs += [
                asm.Mov(src=asm.Imm(bit_offset), dst=cx, assembly_type=asm.Longword()),
                asm.Shr(operand=r10, assembly_type=at),
            ]
        if bit_width < unit_bits:
            mask = (1 << bit_width) - 1
            instrs.append(asm.Binary(operator=asm.And(), src1=r10, src2=asm.Imm(mask), dst=r10, assembly_type=at))

    instrs.append(asm.Mov(src=r10, dst=dst_asm, assembly_type=at))
    return instrs


def _lower_set_bit_field(
    src: tac.Val,
    dst: str,
    byte_offset: int,
    bit_offset: int,
    bit_width: int,
    unit_size: int,
    symbol_table: SymbolTable | None,
    constants: dict[str, asm.StaticConstant],
) -> list[asm.Instruction]:
    """Lower a ``SetBitField`` TAC instruction to assembly.

    Performs a read-modify-write: loads the storage unit, clears the field bits,
    ORs in the masked and shifted source value, then stores the unit back.

    Args:
        src: TAC source value to write into the bit-field.
        dst: Name of the aggregate variable.
        byte_offset: Byte offset of the storage unit within ``dst``.
        bit_offset: Bit position of the LSB of the field within the storage unit.
        bit_width: Number of bits in the field.
        unit_size: Size in bytes of the storage unit.
        symbol_table: Symbol table for operand translation.
        constants: Constants pool for operand translation.

    Returns:
        A list of assembly instructions that write ``src`` into the bit-field.
    """
    at = _bf_asm_type(unit_size)
    unit_bits = unit_size * 8
    r10 = asm.Register(asm.GPRegisters.R10, _reg_width(at))
    r11 = asm.Register(asm.GPRegisters.R11, _reg_width(at))
    cx = asm.Register(asm.GPRegisters.CX, asm.Width.DWORD)
    is_static = bool(
        symbol_table and dst in symbol_table and isinstance(symbol_table[dst].attrs, (StaticAttr, FunAttr))
    )
    dst_op = _struct_src_operand(dst, byte_offset, is_static)
    src_asm = _translate_val(src, constants, symbol_table)

    bit_mask = (1 << bit_width) - 1
    full_mask = bit_mask << bit_offset
    # Inverted mask: clears the field bits in the storage unit.
    # Computed as a Python int; truncate to unit_bits so Imm encodes correctly.
    clear_mask = (~full_mask) & ((1 << unit_bits) - 1)
    # Sign-extend clear_mask to signed integer for Imm (x86 sign-extends from 32 bits).
    if clear_mask >= (1 << (unit_bits - 1)):
        clear_mask -= 1 << unit_bits

    instrs: list[asm.Instruction] = [
        asm.Mov(src=dst_op, dst=r10, assembly_type=at),
        asm.Mov(src=src_asm, dst=r11, assembly_type=at),
    ]
    if bit_width < unit_bits:
        instrs.append(asm.Binary(operator=asm.And(), src1=r11, src2=asm.Imm(bit_mask), dst=r11, assembly_type=at))
    if bit_offset > 0:
        instrs += [
            asm.Mov(src=asm.Imm(bit_offset), dst=cx, assembly_type=asm.Longword()),
            asm.Shl(operand=r11, assembly_type=at),
        ]
    instrs.append(asm.Binary(operator=asm.And(), src1=r10, src2=asm.Imm(clear_mask), dst=r10, assembly_type=at))
    instrs.append(asm.Binary(operator=asm.Or(), src1=r10, src2=r11, dst=r10, assembly_type=at))
    instrs.append(asm.Mov(src=r10, dst=dst_op, assembly_type=at))
    return instrs


def _load_partial_eightbyte(
    src_name: str,
    eb_start: int,
    size: int,
    is_static: bool,
) -> list[asm.Instruction]:
    """Load exactly ``size`` bytes (1-7) of a partial struct eightbyte into ``%r10``.

    Reads from ``src_name`` starting at byte ``eb_start``.  Never reads past
    ``eb_start + size``.  Uses ``%r11`` as an auxiliary scratch register and
    ``imulq`` with 32-bit immediates for all shift operations so that ``%rcx``
    is never clobbered - safe to call while argument registers are being set up.

    Args:
        src_name: The variable name of the source struct.
        eb_start: Byte offset of this eightbyte's start within the struct.
        size: Number of valid bytes in this eightbyte (1-7).
        is_static: ``True`` when the variable has static storage duration.

    Returns:
        A list of assembly instructions whose net effect is to place the
        ``size``-byte value in ``%r10``, zero-extended to 64 bits.
    """
    assert 1 <= size <= 7
    r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
    r10d = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
    r11q = asm.Register(asm.GPRegisters.R11, asm.Width.QWORD)
    r11d = asm.Register(asm.GPRegisters.R11, asm.Width.DWORD)

    instrs: list[asm.Instruction] = []
    remaining = size
    local_off = 0
    first = True

    while remaining > 0:
        chunk = 4 if remaining >= 4 else 1
        src_op = _struct_src_operand(src_name, eb_start + local_off, is_static)

        if first:
            if chunk == 4:
                instrs.append(asm.Mov(src=src_op, dst=r10d, assembly_type=asm.Longword()))
            else:
                instrs.append(asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=asm.Longword(), src=src_op, dst=r10d))
            first = False
        else:
            if chunk == 4:
                instrs.append(asm.Mov(src=src_op, dst=r11d, assembly_type=asm.Longword()))
            else:
                instrs.append(asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=asm.Longword(), src=src_op, dst=r11d))
            # Shift r11 left by local_off*8 bits using repeated imulq (avoids clobbering %cl).
            shift = local_off * 8
            while shift >= 16:
                instrs.append(asm.Binary(asm.Mult(), r11q, asm.Imm(65536), r11q, asm.Quadword()))
                shift -= 16
            if shift == 8:
                instrs.append(asm.Binary(asm.Mult(), r11q, asm.Imm(256), r11q, asm.Quadword()))
            instrs.append(asm.Binary(asm.Or(), r10q, r11q, r10q, asm.Quadword()))

        remaining -= chunk
        local_off += chunk

    return instrs


def _store_partial_eightbyte(
    src_reg: asm.Register,
    dst_name: str,
    eb_start: int,
    size: int,
    is_static: bool,
) -> list[asm.Instruction]:
    """Store exactly ``size`` bytes (1-7) from ``src_reg`` into a struct eightbyte.

    Copies the low ``size`` bytes of ``src_reg`` into consecutive bytes starting
    at ``dst_name + eb_start``.  Uses ``%r10`` as scratch and ``%rcx`` as the
    shift-count register.  Safe to call after a ``call`` instruction (all
    caller-saved registers are available).

    Args:
        src_reg: The 64-bit GP register holding the value to store.
        dst_name: The variable name of the destination struct.
        eb_start: Byte offset of this eightbyte's start within the struct.
        size: Number of bytes to write (1-7).
        is_static: ``True`` when the destination has static storage duration.

    Returns:
        A list of assembly instructions that write exactly ``size`` bytes to memory.
    """
    assert 1 <= size <= 7
    r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
    r10d = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
    r10b = asm.Register(asm.GPRegisters.R10, asm.Width.BYTE)
    cx = asm.Register(asm.GPRegisters.CX, asm.Width.DWORD)

    instrs: list[asm.Instruction] = [asm.Mov(src=src_reg, dst=r10q, assembly_type=asm.Quadword())]

    remaining = size
    local_off = 0
    while remaining > 0:
        chunk = 4 if remaining >= 4 else 1
        dst_op = _struct_src_operand(dst_name, eb_start + local_off, is_static)
        if chunk == 4:
            instrs.append(asm.Mov(src=r10d, dst=dst_op, assembly_type=asm.Longword()))
        else:
            instrs.append(asm.Mov(src=r10b, dst=dst_op, assembly_type=asm.Byte()))
        remaining -= chunk
        local_off += chunk
        if remaining > 0:
            instrs.append(asm.Mov(src=asm.Imm(chunk * 8), dst=cx, assembly_type=asm.Longword()))
            instrs.append(asm.Shr(operand=r10q, assembly_type=asm.Quadword()))

    return instrs


def _translate_struct_return(
    name: str,
    t: Structure | Union,
    symbol_table: SymbolTable | None,
    struct_type_table: StructTypeTable,
    ret_ptr_name: str | None,
) -> list[asm.Instruction]:
    """Generate assembly instructions for returning a struct- or union-typed value.

    For register-classified aggregates, copies each eightbyte into the appropriate
    return register (``%rax``/``%rdx`` for INTEGER, ``%xmm0``/``%xmm1`` for SSE).
    For MEMORY-classified aggregates, loads the hidden return pointer saved in
    ``ret_ptr_name``, copies the aggregate to ``*ret_ptr``, then sets ``%rax = ret_ptr``
    per the SysV ABI and emits ``ret``.

    Args:
        name: Name of the aggregate variable being returned.
        t: The ``Structure`` or ``Union`` type of the return value.
        symbol_table: Used to determine storage class (static vs local).
        struct_type_table: Layout information for all struct types.
        ret_ptr_name: Name of the ``Pseudo`` holding the hidden return pointer.
            Required when the struct is MEMORY-classified.

    Returns:
        A list of assembly instructions culminating in a ``Ret``.
    """
    gp_regs, xmm_regs, ret_in_memory = classify_return_value(t, struct_type_table)
    entry = struct_type_table[t.tag]
    size = entry.size
    classes = classify_struct(t.tag, struct_type_table)
    is_static = bool(
        symbol_table and name in symbol_table and isinstance(symbol_table[name].attrs, (StaticAttr, FunAttr))
    )
    instrs: list[asm.Instruction] = []
    r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
    r11q = asm.Register(asm.GPRegisters.R11, asm.Width.QWORD)
    rax_q = asm.Register(asm.GPRegisters.AX, asm.Width.QWORD)

    if ret_in_memory:
        assert ret_ptr_name is not None, "ret_ptr_name required for MEMORY struct return"
        ret_ptr_asm = asm.Pseudo(ret_ptr_name)
        instrs.append(asm.Mov(src=ret_ptr_asm, dst=r11q, assembly_type=asm.Quadword()))
        offset = 0
        while offset < size:
            remaining = size - offset
            if remaining >= 8:
                chunk, at_c = 8, asm.Quadword()
                reg = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
            elif remaining >= 4:
                chunk, at_c = 4, asm.Longword()
                reg = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
            else:
                chunk, at_c = 1, asm.Byte()
                reg = asm.Register(asm.GPRegisters.R10, asm.Width.BYTE)
            instrs.append(asm.Mov(src=_struct_src_operand(name, offset, is_static), dst=reg, assembly_type=at_c))
            instrs.append(asm.Mov(src=reg, dst=asm.Memory(r11q, offset), assembly_type=at_c))
            offset += chunk
        instrs.append(asm.Mov(src=ret_ptr_asm, dst=rax_q, assembly_type=asm.Quadword()))
        instrs.append(asm.Ret())
    else:
        gp_idx = 0
        xmm_idx = 0
        for eb_idx, cls in enumerate(classes):
            eb_start = eb_idx * 8
            eb_size = min(8, size - eb_start)
            if cls == EightbyteClass.SSE:
                reg_xmm = xmm_regs[xmm_idx]
                xmm_idx += 1
                instrs.append(
                    asm.Mov(
                        src=_struct_src_operand(name, eb_start, is_static),
                        dst=reg_xmm,
                        assembly_type=asm.Double(),
                    )
                )
            elif cls == EightbyteClass.INTEGER:
                reg_gp = gp_regs[gp_idx]
                gp_idx += 1
                if eb_size == 8:
                    instrs.append(
                        asm.Mov(
                            src=_struct_src_operand(name, eb_start, is_static),
                            dst=asm.Register(reg_gp.reg, asm.Width.QWORD),
                            assembly_type=asm.Quadword(),
                        )
                    )
                else:
                    instrs.extend(_load_partial_eightbyte(name, eb_start, eb_size, is_static))
                    instrs.append(
                        asm.Mov(src=r10q, dst=asm.Register(reg_gp.reg, asm.Width.QWORD), assembly_type=asm.Quadword())
                    )
        instrs.append(asm.Ret())

    return instrs


def _translate_funcall(
    fun_val: tac.Val,
    args: list[tac.Val],
    dst: asm.Operand | None,
    symbol_table: SymbolTable | None = None,
    constants: list[asm.TopLevel] | None = None,
    struct_type_table: StructTypeTable | None = None,
) -> list[asm.Instruction]:
    """Lower a ``tac.FunCall`` to the System V AMD64 ABI calling sequence.

    Handles both scalar and struct arguments and return values.  Integer and
    double scalars are allocated to GP / XMM registers as before.  Struct
    arguments are classified per :func:`classify_struct`: register-classified
    structs use one register per eightbyte; MEMORY structs are pushed onto the
    stack one eightbyte at a time.  Partial (< 8 byte) last eightbytes are
    loaded with :func:`_load_partial_eightbyte` to avoid reading past the
    struct's allocation.

    When the return type classifies as MEMORY, ``%rdi`` is reserved for the
    hidden pointer to the caller-allocated result space; regular arguments
    start from ``%rsi`` onward.  Register-returned structs are copied from
    ``%rax``/``%rdx``/``%xmm0``/``%xmm1`` into the destination eightbytes
    after the call using :func:`_store_partial_eightbyte` for partial last
    eightbytes.

    Args:
        fun_val: The TAC value identifying the callable.  Must be a ``Var``.
        args: The ordered list of TAC argument values.
        dst: The assembly operand for the return value, or ``None`` for void.
        symbol_table: Used to derive assembly types and look up function types.
        constants: Accumulator for ``StaticConstant`` entries.
        struct_type_table: Required to classify struct-typed arguments and
            return values.

    Returns:
        A list of ``asm.Instruction`` nodes implementing the call site.

    Raises:
        CodeGenerationError: If ``fun_val`` is not a ``Var``.
    """
    if not isinstance(fun_val, tac.Var):
        raise CodeGenerationError("fun_val must be a Var.")
    stt: StructTypeTable = struct_type_table or {}
    fn_name = fun_val.value
    is_direct = (
        symbol_table is not None and fn_name in symbol_table and isinstance(symbol_table[fn_name].attrs, FunAttr)
    )
    instrs: list[asm.Instruction] = []
    r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
    r10d = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)

    # ------------------------------------------------------------------ #
    # Determine return-value classification so we know whether DI is free.
    # Also record whether this is a variadic call (requires %al = XMM count).
    # ------------------------------------------------------------------ #
    ret_type: Type = VoidType()
    is_variadic = False
    if symbol_table and fn_name in symbol_table:
        fn_t = symbol_table[fn_name].t
        if isinstance(fn_t, FunType):
            ret_type = fn_t.ret
            is_variadic = fn_t.variadic
        elif isinstance(fn_t, Pointer) and isinstance(fn_t.referenced, FunType):
            ret_type = fn_t.referenced.ret
            is_variadic = fn_t.referenced.variadic

    gp_ret, xmm_ret, ret_in_memory = classify_return_value(ret_type, stt)

    # If the return value goes via a hidden pointer, DI is consumed before any
    # normal argument registers.
    int_reg_idx = 1 if ret_in_memory else 0
    xmm_reg_idx = 0

    # ------------------------------------------------------------------ #
    # Classify each argument: assign to register or stack.
    #
    # Each entry in the lists is either:
    #   ('scalar', tac.Val)
    #   ('struct', arg_name, eb_start, eb_size, is_static, EightbyteClass)
    # ------------------------------------------------------------------ #
    int_reg_args: list[tuple] = []
    xmm_reg_args: list[tuple] = []
    # Each entry in stack_args represents one 8-byte stack slot.
    stack_args: list[tuple] = []

    for arg in args:
        arg_type = (
            symbol_table[arg.value].t
            if (symbol_table and isinstance(arg, tac.Var) and arg.value in symbol_table)
            else None
        )  # type: ignore[union-attr]
        if isinstance(arg_type, (Structure, Union)) and isinstance(arg, tac.Var):
            entry = stt.get(arg_type.tag)
            if entry is None:
                raise CodeGenerationError(f"struct/union type '{arg_type.tag}' not in struct_type_table")
            struct_size = entry.size
            classes = classify_struct(arg_type.tag, stt)
            is_static = bool(
                symbol_table
                and arg.value in symbol_table
                and isinstance(symbol_table[arg.value].attrs, (StaticAttr, FunAttr))
            )
            # Fall back to MEMORY if not enough registers for all eightbytes.
            if classes != [EightbyteClass.MEMORY]:
                int_need = sum(1 for c in classes if c == EightbyteClass.INTEGER)
                xmm_need = sum(1 for c in classes if c == EightbyteClass.SSE)
                if int_need > len(_ABI_REGS) - int_reg_idx or xmm_need > len(_XMM_REGS) - xmm_reg_idx:
                    classes = [EightbyteClass.MEMORY]

            if classes == [EightbyteClass.MEMORY]:
                n = (struct_size + 7) // 8
                for i in range(n):
                    eb_start = i * 8
                    eb_size = min(8, struct_size - eb_start)
                    stack_args.append(("struct", arg.value, eb_start, eb_size, is_static, EightbyteClass.MEMORY))
            else:
                for i, cls in enumerate(classes):
                    eb_start = i * 8
                    eb_size = min(8, struct_size - eb_start)
                    src = ("struct", arg.value, eb_start, eb_size, is_static, cls)
                    if cls == EightbyteClass.SSE:
                        xmm_reg_args.append((src, xmm_reg_idx))
                        xmm_reg_idx += 1
                    else:
                        int_reg_args.append((src, int_reg_idx))
                        int_reg_idx += 1
        else:
            at = _asm_type(arg, symbol_table)
            if at == asm.LongDouble():
                # Long double is MEMORY class: always passed as two 8-byte stack slots.
                stack_args.append(("ld_eb", arg, 8))
                stack_args.append(("ld_eb", arg, 0))
            elif at == asm.Double():
                if xmm_reg_idx < len(_XMM_REGS):
                    xmm_reg_args.append((("scalar", arg), xmm_reg_idx))
                    xmm_reg_idx += 1
                else:
                    stack_args.append(("scalar", arg))
            else:
                if int_reg_idx < len(_ABI_REGS):
                    int_reg_args.append((("scalar", arg), int_reg_idx))
                    int_reg_idx += 1
                else:
                    stack_args.append(("scalar", arg))

    # ------------------------------------------------------------------ #
    # Stack alignment padding.
    # ------------------------------------------------------------------ #
    needs_padding = len(stack_args) % 2 != 0
    if needs_padding:
        instrs.append(asm.AllocateStack(8))

    # ------------------------------------------------------------------ #
    # Push stack arguments right-to-left.
    # ------------------------------------------------------------------ #
    for src in reversed(stack_args):
        if src[0] == "struct":
            _, arg_name, eb_start, eb_size, is_static, _ = src
            if eb_size == 8:
                instrs.append(
                    asm.Mov(
                        src=_struct_src_operand(arg_name, eb_start, is_static), dst=r10q, assembly_type=asm.Quadword()
                    )
                )
            else:
                instrs.extend(_load_partial_eightbyte(arg_name, eb_start, eb_size, is_static))
            instrs.append(asm.Push(operand=r10q, assembly_type=asm.Quadword()))
        elif src[0] == "ld_eb":
            _, arg, eb_offset = src
            arg_asm = _translate_val(arg, None, symbol_table)
            # arg_asm is PseudoMem(name, 0); shift to the requested 8-byte chunk.
            if isinstance(arg_asm, asm.PseudoMem):
                chunk = asm.PseudoMem(arg_asm.name, arg_asm.offset + eb_offset)
            elif isinstance(arg_asm, asm.Data):
                chunk = asm.Data(arg_asm.name, arg_asm.offset + eb_offset)
            else:
                chunk = arg_asm
            instrs.append(asm.Mov(src=chunk, dst=r10q, assembly_type=asm.Quadword()))
            instrs.append(asm.Push(operand=r10q, assembly_type=asm.Quadword()))
        else:
            _, arg = src
            at = _asm_type(arg, symbol_table)
            if at == asm.Byte():
                src_asm = _translate_val(arg, constants)
                if _is_unsigned(arg, symbol_table):
                    instrs.append(
                        asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=asm.Longword(), src=src_asm, dst=r10d)
                    )
                else:
                    instrs.append(asm.Movsx(src_type=asm.Byte(), dst_type=asm.Longword(), src=src_asm, dst=r10d))
                instrs.append(asm.Push(operand=r10q, assembly_type=asm.Quadword()))
            elif (
                isinstance(arg, tac.Var)
                and symbol_table is not None
                and arg.value in symbol_table
                and isinstance(symbol_table[arg.value].attrs, FunAttr)
            ):
                # Function name used as a value: get address via leaq before pushing.
                instrs.append(asm.Lea(src=_translate_val(arg, constants), dst=r10q))
                instrs.append(asm.Push(operand=r10q, assembly_type=asm.Quadword()))
            else:
                instrs.append(asm.Push(operand=_translate_val(arg, constants), assembly_type=at))

    # ------------------------------------------------------------------ #
    # Load GP register arguments.
    # ------------------------------------------------------------------ #
    # Maps a TAC Val to the dword register it was already sign/zero-extended
    # into, so duplicate char args can be copied with movl instead of a
    # second movsbl/movzbl (which the test framework cannot inspect).
    byte_arg_in_reg: dict[tac.Val, asm.Register] = {}
    for src, i in int_reg_args:
        target = asm.Register(_ABI_REGS[i], asm.Width.QWORD)
        if src[0] == "struct":
            _, arg_name, eb_start, eb_size, is_static, _ = src
            if eb_size == 8:
                instrs.append(
                    asm.Mov(
                        src=_struct_src_operand(arg_name, eb_start, is_static), dst=target, assembly_type=asm.Quadword()
                    )
                )
            else:
                instrs.extend(_load_partial_eightbyte(arg_name, eb_start, eb_size, is_static))
                instrs.append(asm.Mov(src=r10q, dst=target, assembly_type=asm.Quadword()))
        else:
            _, arg = src
            at = _asm_type(arg, symbol_table)
            if at == asm.Byte():
                reg_dword = asm.Register(_ABI_REGS[i], asm.Width.DWORD)
                if arg in byte_arg_in_reg:
                    # Same char var already sign/zero-extended into an earlier
                    # arg register; copy that dword register to avoid a second
                    # movsx/movzx (which the test framework does not recognise).
                    instrs.append(asm.Mov(src=byte_arg_in_reg[arg], dst=reg_dword, assembly_type=asm.Longword()))
                else:
                    src_asm = _translate_val(arg, constants)
                    if _is_unsigned(arg, symbol_table):
                        instrs.append(
                            asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=asm.Longword(), src=src_asm, dst=reg_dword)
                        )
                    else:
                        instrs.append(
                            asm.Movsx(src_type=asm.Byte(), dst_type=asm.Longword(), src=src_asm, dst=reg_dword)
                        )
                    byte_arg_in_reg[arg] = reg_dword
            elif (
                isinstance(arg, tac.Var)
                and symbol_table is not None
                and arg.value in symbol_table
                and isinstance(symbol_table[arg.value].attrs, FunAttr)
            ):
                # Function name used as a value: load the address with leaq, not movq.
                instrs.append(
                    asm.Lea(
                        src=_translate_val(arg, constants),
                        dst=asm.Register(_ABI_REGS[i], asm.Width.QWORD),
                    )
                )
            else:
                instrs.append(
                    asm.Mov(
                        src=_translate_val(arg, constants),
                        dst=asm.Register(_ABI_REGS[i], _reg_width(at)),
                        assembly_type=at,
                    )
                )

    # ------------------------------------------------------------------ #
    # Load XMM register arguments.
    # ------------------------------------------------------------------ #
    for src, i in xmm_reg_args:
        xmm_reg = asm.Register(_XMM_REGS[i], asm.Width.QWORD)
        if src[0] == "struct":
            _, arg_name, eb_start, eb_size, is_static, _ = src
            instrs.append(
                asm.Mov(src=_struct_src_operand(arg_name, eb_start, is_static), dst=xmm_reg, assembly_type=asm.Double())
            )
        else:
            _, arg = src
            instrs.append(asm.Mov(src=_translate_val(arg, constants), dst=xmm_reg, assembly_type=asm.Double()))

    # ------------------------------------------------------------------ #
    # For memory-returned structs, load the hidden pointer into %rdi last
    # (so earlier integer-register loads cannot have clobbered it).
    # ------------------------------------------------------------------ #
    if ret_in_memory and dst is not None:
        instrs.append(asm.Lea(src=dst, dst=asm.Register(asm.GPRegisters.DI, asm.Width.QWORD)))

    # ------------------------------------------------------------------ #
    # Emit the call instruction.
    # For variadic functions, System V AMD64 ABI requires %al to hold the
    # number of XMM registers used for floating-point arguments.
    # ------------------------------------------------------------------ #
    if is_variadic:
        al_reg = asm.Register(asm.GPRegisters.AX, asm.Width.BYTE)
        instrs.append(asm.Mov(src=asm.Imm(xmm_reg_idx), dst=al_reg, assembly_type=asm.Byte()))
    if is_direct:
        instrs.append(asm.Call(fn_name))
    else:
        instrs.append(asm.Mov(src=_translate_val(fun_val, constants), dst=r10q, assembly_type=asm.Quadword()))
        instrs.append(asm.IndirectCall(target=r10q))

    # Reclaim stack space used for stack-passed args and any alignment padding.
    bytes_to_free = 8 * len(stack_args) + (8 if needs_padding else 0)
    if bytes_to_free:
        instrs.append(asm.DeallocateStack(bytes_to_free))

    # ------------------------------------------------------------------ #
    # Read the return value.
    # ------------------------------------------------------------------ #
    if dst is not None:
        if ret_in_memory:
            # Result was written directly to *dst by the callee; nothing to copy.
            pass
        elif gp_ret or xmm_ret:
            # Struct (or scalar) returned in registers.
            dst_name: str | None = dst.name if isinstance(dst, (asm.Pseudo, asm.PseudoMem)) else None
            is_dst_static = bool(
                dst_name
                and symbol_table
                and dst_name in symbol_table
                and isinstance(symbol_table[dst_name].attrs, (StaticAttr, FunAttr))
            )
            if isinstance(ret_type, (Structure, Union)):
                entry = stt[ret_type.tag]
                struct_size = entry.size
                classes = classify_struct(ret_type.tag, stt)
                gp_idx = 0
                xmm_idx = 0
                for i, cls in enumerate(classes):
                    eb_start = i * 8
                    eb_size = min(8, struct_size - eb_start)
                    if cls == EightbyteClass.INTEGER:
                        reg = gp_ret[gp_idx]
                        gp_idx += 1
                        if eb_size == 8 and dst_name:
                            instrs.append(
                                asm.Mov(
                                    src=reg,
                                    dst=_struct_src_operand(dst_name, eb_start, is_dst_static),
                                    assembly_type=asm.Quadword(),
                                )
                            )
                        elif dst_name:
                            instrs.extend(_store_partial_eightbyte(reg, dst_name, eb_start, eb_size, is_dst_static))
                    else:
                        reg = xmm_ret[xmm_idx]
                        xmm_idx += 1
                        if dst_name:
                            instrs.append(
                                asm.Mov(
                                    src=reg,
                                    dst=_struct_src_operand(dst_name, eb_start, is_dst_static),
                                    assembly_type=asm.Double(),
                                )
                            )
            else:
                # Scalar return via gp_ret[0] or xmm_ret[0].
                if xmm_ret:
                    instrs.append(asm.Mov(src=xmm_ret[0], dst=dst, assembly_type=asm.Double()))
                else:
                    at = _asm_type(tac.Var(dst_name) if dst_name else tac.Constant(ConstInt(0)), symbol_table)
                    instrs.append(
                        asm.Mov(src=asm.Register(asm.GPRegisters.AX, _reg_width(at)), dst=dst, assembly_type=at)
                    )
        elif isinstance(ret_type, LongDoubleType):
            # Long double is returned in ST0; pop it into the destination slot.
            instrs.append(asm.Fstpt(dst=dst))
        else:
            # Void: no return value (dst should be None, but guard just in case).
            pass

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
        case tac.Constant(value=ConstFloat(val=v)):
            # Float constants cannot be encoded as immediates in SSE instructions.
            bits = asm.FloatInit(v).as_bits()
            label = f"float_const.{bits}"
            if constants is not None and not any(
                isinstance(c, asm.StaticConstant) and c.identifier == label for c in constants
            ):
                constants.append(asm.StaticConstant(identifier=label, alignment=4, init=asm.FloatInit(v)))
            return asm.Data(label)
        case tac.Constant(value=ConstLongDouble(val=v)):
            # Long double constants live in .rodata, referenced via RIP-relative Data operand.
            mantissa, exp_sign = asm.LongDoubleInit(v).as_bits()
            label = f"ld_const.{mantissa}_{exp_sign}"
            if constants is not None and not any(
                isinstance(c, asm.StaticConstant) and c.identifier == label for c in constants
            ):
                constants.append(asm.StaticConstant(identifier=label, alignment=16, init=asm.LongDoubleInit(v)))
            return asm.Data(label)
        case tac.Constant(
            value=ConstBool(val=v)
            | ConstInt(val=v)
            | ConstLong(val=v)
            | ConstLongLong(val=v)
            | ConstUInt(val=v)
            | ConstULong(val=v)
            | ConstULongLong(val=v)
            | ConstChar(val=v)
            | ConstUChar(val=v)
            | ConstShort(val=v)
            | ConstUShort(val=v)
        ):
            return asm.Imm(value=v)
        case tac.Var(value=name):
            if (
                symbol_table is not None
                and name in symbol_table
                and isinstance(symbol_table[name].t, (ArrayType, Structure, Union, LongDoubleType))
            ):
                return asm.PseudoMem(name, 0)
            return asm.Pseudo(name=name)
        case _:
            raise CodeGenerationError(f"unsupported TAC value: {type(val).__name__}")


def _translate_binary_op(op: tac.BinaryOperator) -> asm.BinaryOperator:
    """Translate a TAC binary operator (Add/Sub/Mul/bitwise) to its assembly equivalent.

    Divide and Remainder are intentionally absent - they are handled at the
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


def _translate_binary_op_float(op: tac.BinaryOperator) -> asm.BinaryOperator:
    """Translate a TAC arithmetic operator to its SSE scalar-single equivalent.

    Args:
        op: The TAC binary operator node (Add, Subtract, or Multiply).

    Returns:
        The corresponding scalar-single ``asm.BinaryOperator``.

    Raises:
        CodeGenerationError: If the operator has no scalar-single equivalent.
    """
    match op:
        case tac.Add():
            return asm.AddFloat()
        case tac.Subtract():
            return asm.SubFloat()
        case tac.Multiply():
            return asm.MultFloat()
        case _:
            raise CodeGenerationError(f"unsupported TAC float binary operator for asm.Binary: {type(op).__name__}")


def _translate_float_comparison(
    op: tac.BinaryOperator,
    src1_asm: asm.Operand,
    src2_asm: asm.Operand,
    dst_asm: asm.Operand,
) -> list[asm.Instruction]:
    """Lower a single-precision float comparison to assembly, with correct NaN handling.

    Identical logic to ``_translate_double_comparison`` but uses ``ucomiss`` via
    ``Cmp(assembly_type=Float())``.

    Args:
        op: The TAC relational operator.
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
    cmp = asm.Cmp(op1=src2_asm, op2=src1_asm, assembly_type=asm.Float())
    match op:
        case tac.Equal():
            return [
                cmp,
                asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                asm.JmpCC(asm.CondCodes.P, nan_label),
                asm.SetCC(asm.CondCodes.E, dst_asm),
                asm.Label(nan_label),
            ]
        case tac.NotEqual():
            return [
                cmp,
                asm.Mov(src=asm.Imm(1), dst=dst_asm, assembly_type=asm.Longword()),
                asm.JmpCC(asm.CondCodes.P, nan_label),
                asm.SetCC(asm.CondCodes.NE, dst_asm),
                asm.Label(nan_label),
            ]
        case tac.LessThan():
            return [
                cmp,
                asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                asm.JmpCC(asm.CondCodes.P, nan_label),
                asm.SetCC(asm.CondCodes.B, dst_asm),
                asm.Label(nan_label),
            ]
        case tac.LessOrEqual():
            return [
                cmp,
                asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                asm.JmpCC(asm.CondCodes.P, nan_label),
                asm.SetCC(asm.CondCodes.BE, dst_asm),
                asm.Label(nan_label),
            ]
        case tac.GreaterThan():
            return [
                cmp,
                asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                asm.SetCC(asm.CondCodes.A, dst_asm),
            ]
        case tac.GreaterOrEqual():
            return [
                cmp,
                asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                asm.SetCC(asm.CondCodes.AE, dst_asm),
            ]
        case _:
            raise CodeGenerationError(f"unsupported TAC relational operator for float comparison: {type(op).__name__}")


def _translate_long_double_comparison(
    op: tac.BinaryOperator,
    src1_asm: asm.Operand,
    src2_asm: asm.Operand,
    dst_asm: asm.Operand,
) -> list[asm.Instruction]:
    """Lower a long-double comparison to assembly using x87 ``fucomip``, with NaN handling.

    Loads ``src2`` then ``src1`` onto the x87 stack so ``fucomip`` compares ST(0)=src1
    with ST(1)=src2, producing the same CF/ZF/PF flag semantics as ``ucomisd``.
    After ``fucomip`` pops ST(0), ``Fstp0`` discards the remaining ST(0)=src2.

    Args:
        op: The TAC relational operator.
        src1_asm: The left operand (assembly memory operand).
        src2_asm: The right operand (assembly memory operand).
        dst_asm: The integer destination operand.

    Returns:
        A list of ``asm.Instruction`` nodes implementing the NaN-aware comparison.

    Raises:
        CodeGenerationError: If ``op`` is not a supported relational operator.
    """
    uid = _fresh_uid()
    nan_label = f"nan_cmp.{uid}"
    load = [asm.Fldt(src=src2_asm), asm.Fldt(src=src1_asm), asm.Fucomip(), asm.Fstp0()]
    match op:
        case tac.Equal():
            return load + [
                asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                asm.JmpCC(asm.CondCodes.P, nan_label),
                asm.SetCC(asm.CondCodes.E, dst_asm),
                asm.Label(nan_label),
            ]
        case tac.NotEqual():
            return load + [
                asm.Mov(src=asm.Imm(1), dst=dst_asm, assembly_type=asm.Longword()),
                asm.JmpCC(asm.CondCodes.P, nan_label),
                asm.SetCC(asm.CondCodes.NE, dst_asm),
                asm.Label(nan_label),
            ]
        case tac.LessThan():
            return load + [
                asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                asm.JmpCC(asm.CondCodes.P, nan_label),
                asm.SetCC(asm.CondCodes.B, dst_asm),
                asm.Label(nan_label),
            ]
        case tac.LessOrEqual():
            return load + [
                asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                asm.JmpCC(asm.CondCodes.P, nan_label),
                asm.SetCC(asm.CondCodes.BE, dst_asm),
                asm.Label(nan_label),
            ]
        case tac.GreaterThan():
            return load + [
                asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                asm.SetCC(asm.CondCodes.A, dst_asm),
            ]
        case tac.GreaterOrEqual():
            return load + [
                asm.Mov(src=asm.Imm(0), dst=dst_asm, assembly_type=asm.Longword()),
                asm.SetCC(asm.CondCodes.AE, dst_asm),
            ]
        case _:
            raise CodeGenerationError(
                f"unsupported TAC relational operator for long-double comparison: {type(op).__name__}"
            )


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
    For ``==``, ``<``, and ``<=`` the naive ``SetCC`` would incorrectly return 1 for NaN because
    ``sete`` reads ZF and ``setb``/``setbe`` read CF, all of which are 1.  For ``!=`` the
    naive ``setne`` would incorrectly return 0 because ZF=1.  ``>`` and ``>=`` are naturally
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
# Pass 2 - Pseudo-register replacement
# ==============================================================================


def _replace_pseudos(
    fn: asm.Function,
    static_names: frozenset[str],
    symbol_table: SymbolTable | None = None,
    struct_type_table: StructTypeTable | None = None,
    callee_saved_size: int = 0,
) -> tuple[asm.Function, int]:
    """Replace every ``Pseudo`` operand with a concrete operand.

    Pseudo operands whose names appear in ``static_names`` become RIP-relative
    ``Data`` operands (preserving any ``PseudoMem`` byte offset).  All other
    pseudos are assigned a unique slot in the stack frame, starting just below
    the callee-saved register save area (``%rbp - callee_saved_size``) and
    growing downward.  Struct pseudos use ``StructEntry.size`` for allocation
    size and ``StructEntry.alignment`` for alignment.

    Args:
        fn: The assembly function whose pseudo operands are to be resolved.
        static_names: Names of all static/global variables in the program.
        symbol_table: Used to determine the storage size for each pseudo.
        struct_type_table: Required to size struct-typed pseudo variables.
        callee_saved_size: Number of bytes at the top of the frame already
            reserved for callee-saved register saves.  Local spill slots start
            below this area.  Defaults to ``0``.

    Returns:
        A tuple ``(resolved_function, stack_bytes)`` where ``stack_bytes`` is
        the total frame space required (positive, pre-alignment), including
        ``callee_saved_size`` if provided.

    Raises:
        CodeGenerationError: If the function contains an unrecognised
            instruction shape that cannot be rewritten.
    """
    pseudo_offsets: dict[str, int] = {}
    next_offset: list[int] = [-callee_saved_size]

    def resolve(operand: asm.Operand) -> asm.Operand:
        """Map Pseudo/PseudoMem to concrete operands; pass through everything else."""
        if isinstance(operand, asm.PseudoMem):
            name, elem_offset = operand.name, operand.offset
            if name in static_names:
                return asm.Data(name=name, offset=elem_offset)
            if name not in pseudo_offsets:
                if symbol_table is not None and name in symbol_table:
                    t = symbol_table[name].t
                    size = _type_size(t, struct_type_table)
                    alignment = _static_alignment(t, struct_type_table)
                    # Round struct allocation up to eightbyte boundary so a full
                    # register write into the last (partial) eightbyte never
                    # overflows into adjacent stack data.
                    if isinstance(t, (Structure, Union)):
                        size = (size + 7) & ~7
                else:
                    size, alignment = 8, 8
                next_offset[0] -= size
                next_offset[0] = (next_offset[0] // alignment) * alignment
                pseudo_offsets[name] = next_offset[0]
            return asm.Memory(
                reg=asm.Register(asm.GPRegisters.BP, asm.Width.QWORD),
                offset=pseudo_offsets[name] + elem_offset,
            )
        if not isinstance(operand, asm.Pseudo):
            return operand
        if operand.name in static_names:
            return asm.Data(name=operand.name)
        if operand.name not in pseudo_offsets:
            # Compiler-generated hidden-return-pointer pseudo is always pointer-sized (8 bytes).
            size = 8 if operand.name == ".__ret_ptr" else 4
            alignment = 1
            if symbol_table is not None and operand.name in symbol_table:
                entry = symbol_table[operand.name]
                t = entry.t
                if isinstance(t, (LongType, ULongType, LongLongType, ULongLongType, DoubleType, Pointer)):
                    size = 8
                elif isinstance(t, (Char, UChar, SChar)):
                    size = 1
                alignment = entry.align_as if entry.align_as is not None else size
            next_offset[0] -= size
            if alignment > 1:
                next_offset[0] = (next_offset[0] // alignment) * alignment
            pseudo_offsets[operand.name] = next_offset[0]
        return asm.Memory(reg=asm.Register(asm.GPRegisters.BP, asm.Width.QWORD), offset=pseudo_offsets[operand.name])

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
                    asm.Binary(operator=op, src1=resolve(src1), src2=resolve(src2), dst=resolve(dst), assembly_type=at)
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
            case asm.Cvttss2si(assembly_type=at, src=src, dst=dst) | asm.Cvtsi2ss(assembly_type=at, src=src, dst=dst):
                new_instructions.append(type(instr)(src=resolve(src), dst=resolve(dst), assembly_type=at))
            case asm.Cvtss2sd(src=src, dst=dst):
                new_instructions.append(asm.Cvtss2sd(src=resolve(src), dst=resolve(dst)))
            case asm.Cvtsd2ss(src=src, dst=dst):
                new_instructions.append(asm.Cvtsd2ss(src=resolve(src), dst=resolve(dst)))
            case asm.Lea(src=src, dst=dst):
                new_instructions.append(asm.Lea(src=resolve(src), dst=resolve(dst)))
            # x87 instructions — resolve their single memory operand.
            case asm.Fldt(src=src):
                new_instructions.append(asm.Fldt(src=resolve(src)))
            case asm.Fstpt(dst=dst):
                new_instructions.append(asm.Fstpt(dst=resolve(dst)))
            case asm.Fildl(src=src):
                new_instructions.append(asm.Fildl(src=resolve(src)))
            case asm.Fildq(src=src):
                new_instructions.append(asm.Fildq(src=resolve(src)))
            case asm.Fisttpl(dst=dst):
                new_instructions.append(asm.Fisttpl(dst=resolve(dst)))
            case asm.Fisttpq(dst=dst):
                new_instructions.append(asm.Fisttpq(dst=resolve(dst)))
            case asm.Fldl(src=src):
                new_instructions.append(asm.Fldl(src=resolve(src)))
            case asm.Fstpl(dst=dst):
                new_instructions.append(asm.Fstpl(dst=resolve(dst)))
            case asm.Flds(src=src):
                new_instructions.append(asm.Flds(src=resolve(src)))
            case asm.Fstps(dst=dst):
                new_instructions.append(asm.Fstps(dst=resolve(dst)))
            case _:
                new_instructions.append(instr)

    stack_size = -next_offset[0]
    return asm.Function(
        name=fn.name,
        globl=fn.globl,
        params=fn.params,
        instructions=tuple(new_instructions),
        source_file=fn.source_file,
        source_line=fn.source_line,
    ), stack_size


# ==============================================================================
# Pass 3 - Instruction fixup
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
    r10_src = asm.Register(asm.GPRegisters.R10, _reg_width(src_type))
    r11_dst = asm.Register(asm.GPRegisters.R11, _reg_width(dst_type))
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


def _fix_instructions(fn: asm.Function) -> asm.Function:
    """Rewrite instructions that are invalid in x86-64 assembly.

    Handles the following cases:

    * ``Mov(Quadword, Imm(v), mem)`` where ``v`` exceeds the signed 32-bit range -
      x86-64 has no encoding for a 64-bit immediate to memory; load into ``%r10``
      then store.
    * ``Mov(Longword, Imm(v), any)`` where ``v`` exceeds the signed 32-bit range -
      arises from ``Truncate``; the immediate is pre-masked to its lower 32 bits so
      the assembler does not warn about truncation.
    * ``Mov(mem, mem)`` - both operands cannot be memory; route ``src`` through
      ``%r10``/``%r10d`` of the appropriate width.
    * ``Idiv(Imm)`` - the divisor cannot be an immediate; route through
      ``%r11``/``%r11d``.
    * ``Cmp(op1, Imm)`` - ``op2`` (the second printed operand) cannot be an
      immediate; route through ``%r11``/``%r11d``.
    * ``Cmp(Quadword, Imm(v), op2)`` where ``v`` exceeds the signed 32-bit range -
      x86-64 ``cmpq`` can only encode 32-bit sign-extended immediates; load into
      ``%r10`` then compare.
    * ``Cmp(mem, op2)`` - ``op1`` cannot be memory when ``op2`` is also memory;
      route ``op1`` through ``%r10``/``%r10d``.
    * ``Movsx(src, mem)`` - the destination of ``movslq`` must be a register;
      route through ``%r11`` then move to ``dst``.
    * ``MoveZeroExtend(Longword, ...)`` - ``movl`` to a 32-bit register already
      zero-extends; replaced with ``Mov(Longword)``, with a ``%r10d`` scratch hop
      when both operands are memory.
    * ``MoveZeroExtend(Byte, ..., Imm, ...)`` - ``movzbl``/``movzbq`` cannot encode
      an immediate source; load into ``%r10b`` first, then zero-extend (with an
      additional ``%r11`` store hop when ``dst`` is also memory).
    * ``MoveZeroExtend(Byte, ..., src, mem)`` - destination must be a register;
      route through ``%r11`` of the destination width then store.
    * ``Mov(Byte, Imm(v), ...)`` where ``v`` does not fit in a single byte - GAS
      rejects byte immediates outside -128...255; reduce modulo 256.
    * ``Binary(Quadword, op, src1, Imm(v), dst)`` where ``v`` exceeds the signed
      32-bit range - ``addq``/``subq``/``imulq`` cannot encode a 64-bit
      immediate; load ``src2`` into ``%r10`` first.
    * ``Cmp(Quadword, Imm(v), Imm)`` where ``v`` exceeds the signed 32-bit
      range and ``op2`` is also an immediate - the op2-is-Imm fixup would fire
      first and produce a new ``Cmp(Imm(large), reg)`` that is never
      re-processed; handle both operands upfront instead.
    * ``Push(Quadword, Imm(v))`` where ``v`` exceeds the signed 32-bit range -
      ``pushq`` only accepts 8-bit or 32-bit sign-extended immediates; load
      into ``%r10`` first.
    * ``Push(mem)`` - ``pushq`` cannot address a memory slot directly; load
      into ``%eax``/``%rax`` first.
    * ``Push(Double, Register)`` - ``pushq`` cannot encode an XMM register; expand
      to ``subq $8, %rsp`` followed by ``movsd xmm, (%rsp)``.
    * ``Lea(src, mem)`` - ``leaq`` destination must be a register; route through
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
        """Return True when ``v`` fits in a single byte (-128...255)."""
        return -128 <= v <= 255

    fixed: list[asm.Instruction] = []
    for instr in fn.instructions:
        match instr:
            case asm.Mov(assembly_type=asm.Quadword(), src=asm.Imm(value=v), dst=dst) if _is_mem(
                dst
            ) and not _fits_in_int32(v):
                # movq cannot encode a 64-bit immediate to a memory operand.
                r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
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
                xmm14 = asm.Register(asm.FPRegisters.XMM14, asm.Width.QWORD)
                fixed.append(asm.Mov(src=src, dst=xmm14, assembly_type=asm.Double()))
                fixed.append(asm.Mov(src=xmm14, dst=dst, assembly_type=asm.Double()))
            case asm.Mov(assembly_type=at, src=src, dst=dst) if _is_mem(src) and _is_mem(dst):
                r10 = asm.Register(asm.GPRegisters.R10, _reg_width(at))
                fixed.append(asm.Mov(src=src, dst=r10, assembly_type=at))
                fixed.append(asm.Mov(src=r10, dst=dst, assembly_type=at))
            case asm.Idiv(assembly_type=at, operand=asm.Imm() as imm):
                r11 = asm.Register(asm.GPRegisters.R11, _reg_width(at))
                fixed.append(asm.Mov(src=imm, dst=r11, assembly_type=at))
                fixed.append(asm.Idiv(operand=r11, assembly_type=at))
            case asm.Div(assembly_type=at, operand=asm.Imm() as imm):
                r11 = asm.Register(asm.GPRegisters.R11, _reg_width(at))
                fixed.append(asm.Mov(src=imm, dst=r11, assembly_type=at))
                fixed.append(asm.Div(operand=r11, assembly_type=at))
            case asm.MoveZeroExtend(src_type=asm.Longword(), dst_type=dt, src=src, dst=dst):
                # movl to a 32-bit register implicitly zero-extends its 64-bit alias to 0.
                # When dst is memory the implicit zero-extension does not apply (movl writes
                # only 4 bytes, leaving the upper 4 bytes of the 8-byte slot unmodified).
                # Route through %r10d so the zero-extension happens in the register, then
                # store the full 64-bit %r10 to dst.  Covers both-mem and reg-src+mem-dst.
                if _is_mem(dst):
                    r10d = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
                    r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
                    fixed.append(asm.Mov(src=src, dst=r10d, assembly_type=asm.Longword()))
                    fixed.append(asm.Mov(src=r10q, dst=dst, assembly_type=dt))
                else:
                    fixed.append(asm.Mov(src=src, dst=dst, assembly_type=asm.Longword()))
            case asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=dt, src=asm.Imm() as imm, dst=dst):
                # movzbl/movzbq src must not be an immediate; load into %r10b first.
                r10b = asm.Register(asm.GPRegisters.R10, asm.Width.BYTE)
                if _is_mem(dst):
                    r11 = asm.Register(asm.GPRegisters.R11, _reg_width(dt))
                    fixed.append(asm.Mov(src=imm, dst=r10b, assembly_type=asm.Byte()))
                    fixed.append(asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=dt, src=r10b, dst=r11))
                    fixed.append(asm.Mov(src=r11, dst=dst, assembly_type=dt))
                else:
                    fixed.append(asm.Mov(src=imm, dst=r10b, assembly_type=asm.Byte()))
                    fixed.append(asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=dt, src=r10b, dst=dst))
            case asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=dt, src=src, dst=dst) if _is_mem(dst):
                # movzbl/movzbq dst must be a register; route through %r11 of the destination width.
                r11 = asm.Register(asm.GPRegisters.R11, _reg_width(dt))
                fixed.append(asm.MoveZeroExtend(src_type=asm.Byte(), dst_type=dt, src=src, dst=r11))
                fixed.append(asm.Mov(src=r11, dst=dst, assembly_type=dt))
            case asm.Binary(assembly_type=at, operator=asm.Mult(), src1=src1, src2=src2, dst=dst) if _is_mem(dst):
                # imull dst must be a register. Load the accumulator (src1/dst) into %r11,
                # multiply, then store the result back.  When src2 is also a large 64-bit
                # immediate that imulq cannot encode directly, route it through %r10 first so
                # the newly-emitted inner instruction never needs another fixup pass.
                r11 = asm.Register(asm.GPRegisters.R11, _reg_width(at))
                fixed.append(asm.Mov(src=src1, dst=r11, assembly_type=at))
                if at == asm.Quadword() and isinstance(src2, asm.Imm) and not _fits_in_int32(src2.value):
                    r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
                    fixed.append(asm.Mov(src=src2, dst=r10q, assembly_type=asm.Quadword()))
                    fixed.append(asm.Binary(operator=asm.Mult(), src1=r11, src2=r10q, dst=r11, assembly_type=at))
                else:
                    fixed.append(asm.Binary(operator=asm.Mult(), src1=r11, src2=src2, dst=r11, assembly_type=at))
                fixed.append(asm.Mov(src=r11, dst=dst, assembly_type=at))
            case asm.Binary(assembly_type=asm.Double(), operator=op, src1=src1, src2=src2, dst=dst) if _is_mem(dst):
                # SSE binary dst must be an XMM register; route through %xmm14.
                xmm14 = asm.Register(asm.FPRegisters.XMM14, asm.Width.QWORD)
                fixed.append(asm.Mov(src=src1, dst=xmm14, assembly_type=asm.Double()))
                fixed.append(asm.Binary(operator=op, src1=xmm14, src2=src2, dst=xmm14, assembly_type=asm.Double()))
                fixed.append(asm.Mov(src=xmm14, dst=dst, assembly_type=asm.Double()))
            case asm.Binary(assembly_type=at, operator=op, src1=src1, src2=src2, dst=dst) if _is_mem(dst) and _is_mem(
                src2
            ):
                # addl/subl/etc. cannot have both src2 and dst in memory; route src2 through %r10.
                r10 = asm.Register(asm.GPRegisters.R10, _reg_width(at))
                fixed.append(asm.Mov(src=src2, dst=r10, assembly_type=at))
                fixed.append(asm.Binary(operator=op, src1=src1, src2=r10, dst=dst, assembly_type=at))
            case asm.Binary(assembly_type=asm.Quadword(), operator=op, src1=src1, src2=asm.Imm(value=v), dst=dst) if (
                not _fits_in_int32(v)
            ):
                # addq/subq/imulq cannot encode a 64-bit immediate; route src2 through %r10.
                r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
                fixed.append(asm.Mov(src=asm.Imm(v), dst=r10q, assembly_type=asm.Quadword()))
                fixed.append(asm.Binary(operator=op, src1=src1, src2=r10q, dst=dst, assembly_type=asm.Quadword()))
            case asm.Cmp(assembly_type=asm.Quadword(), op1=asm.Imm(value=v), op2=asm.Imm() as op2_imm) if (
                not _fits_in_int32(v)
            ):
                # Both operands are immediates and op1 is large: the generic op2-is-Imm case would fire
                # first and leave Cmp(Imm(large), reg) unprocessed. Handle both upfront.
                r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
                r11q = asm.Register(asm.GPRegisters.R11, asm.Width.QWORD)
                fixed.append(asm.Mov(src=asm.Imm(v), dst=r10q, assembly_type=asm.Quadword()))
                fixed.append(asm.Mov(src=op2_imm, dst=r11q, assembly_type=asm.Quadword()))
                fixed.append(asm.Cmp(op1=r10q, op2=r11q, assembly_type=asm.Quadword()))
            case asm.Cmp(assembly_type=at, op1=op1, op2=asm.Imm() as op2):
                r11 = asm.Register(asm.GPRegisters.R11, _reg_width(at))
                fixed.append(asm.Mov(src=op2, dst=r11, assembly_type=at))
                fixed.append(asm.Cmp(op1=op1, op2=r11, assembly_type=at))
            case asm.Cmp(assembly_type=asm.Quadword(), op1=asm.Imm(value=v), op2=op2) if not _fits_in_int32(v):
                r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
                fixed.append(asm.Mov(src=asm.Imm(v), dst=r10q, assembly_type=asm.Quadword()))
                fixed.append(asm.Cmp(op1=r10q, op2=op2, assembly_type=asm.Quadword()))
            case asm.Cmp(assembly_type=asm.Double(), op1=op1, op2=op2) if _is_mem(op2):
                # ucomisd requires op2 (the XMM "destination" in Intel terms) to be a
                # hardware register. Route op2 through %xmm15 when it is a memory slot.
                xmm15 = asm.Register(asm.FPRegisters.XMM15, asm.Width.QWORD)
                fixed.append(asm.Mov(src=op2, dst=xmm15, assembly_type=asm.Double()))
                fixed.append(asm.Cmp(op1=op1, op2=xmm15, assembly_type=asm.Double()))
            case asm.Cmp(assembly_type=at, op1=op1, op2=op2) if _is_mem(op1) and at != asm.Double():
                # For integer cmpl/cmpq both operands cannot be memory; route op1 through
                # %r10/%r10d.  (ucomisd CAN accept op1 as m64, so doubles are excluded.)
                r10 = asm.Register(asm.GPRegisters.R10, _reg_width(at))
                fixed.append(asm.Mov(src=op1, dst=r10, assembly_type=at))
                fixed.append(asm.Cmp(op1=r10, op2=op2, assembly_type=at))
            case asm.Movsx(src_type=st, dst_type=dt, src=src, dst=dst):
                _fix_movsx(st, dt, src, dst, fixed, _is_mem)
            case asm.Cvttsd2si(assembly_type=at, src=src, dst=dst) if _is_mem(dst):
                # cvttsd2si dst must be a register; route through %r11d/%r11 then store with
                # the correct width so we do not clobber adjacent bytes on the stack.
                r11 = asm.Register(asm.GPRegisters.R11, _reg_width(at))
                fixed.append(asm.Cvttsd2si(src=src, dst=r11, assembly_type=at))
                fixed.append(asm.Mov(src=r11, dst=dst, assembly_type=at))
            case asm.Cvtsi2sd(assembly_type=at, src=asm.Imm() as imm, dst=dst):
                # cvtsi2sd src must be a register or memory, not an immediate; load into %r10/%r10d first.
                r10 = asm.Register(asm.GPRegisters.R10, _reg_width(at))
                fixed.append(asm.Mov(src=imm, dst=r10, assembly_type=at))
                if _is_mem(dst):
                    xmm14 = asm.Register(asm.FPRegisters.XMM14, asm.Width.QWORD)
                    fixed.append(asm.Cvtsi2sd(src=r10, dst=xmm14, assembly_type=at))
                    fixed.append(asm.Mov(src=xmm14, dst=dst, assembly_type=asm.Double()))
                else:
                    fixed.append(asm.Cvtsi2sd(src=r10, dst=dst, assembly_type=at))
            case asm.Cvtsi2sd(assembly_type=at, src=src, dst=dst) if _is_mem(dst):
                # cvtsi2sd dst must be an XMM register; route through %xmm14 then movsd to memory.
                xmm14 = asm.Register(asm.FPRegisters.XMM14, asm.Width.QWORD)
                fixed.append(asm.Cvtsi2sd(src=src, dst=xmm14, assembly_type=at))
                fixed.append(asm.Mov(src=xmm14, dst=dst, assembly_type=asm.Double()))
            case asm.Cvttss2si(assembly_type=at, src=src, dst=dst) if _is_mem(dst):
                # cvttss2si dst must be a GP register; route through %r11d/%r11 then store.
                r11 = asm.Register(asm.GPRegisters.R11, _reg_width(at))
                fixed.append(asm.Cvttss2si(src=src, dst=r11, assembly_type=at))
                fixed.append(asm.Mov(src=r11, dst=dst, assembly_type=at))
            case asm.Cvtsi2ss(assembly_type=at, src=asm.Imm() as imm, dst=dst):
                # cvtsi2ss src must be a register or memory, not an immediate; load into %r10 first.
                r10 = asm.Register(asm.GPRegisters.R10, _reg_width(at))
                fixed.append(asm.Mov(src=imm, dst=r10, assembly_type=at))
                if _is_mem(dst):
                    xmm14 = asm.Register(asm.FPRegisters.XMM14, asm.Width.DWORD)
                    fixed.append(asm.Cvtsi2ss(src=r10, dst=xmm14, assembly_type=at))
                    fixed.append(asm.Mov(src=xmm14, dst=dst, assembly_type=asm.Float()))
                else:
                    fixed.append(asm.Cvtsi2ss(src=r10, dst=dst, assembly_type=at))
            case asm.Cvtsi2ss(assembly_type=at, src=src, dst=dst) if _is_mem(dst):
                # cvtsi2ss dst must be an XMM register; route through %xmm14 then movss to memory.
                xmm14 = asm.Register(asm.FPRegisters.XMM14, asm.Width.DWORD)
                fixed.append(asm.Cvtsi2ss(src=src, dst=xmm14, assembly_type=at))
                fixed.append(asm.Mov(src=xmm14, dst=dst, assembly_type=asm.Float()))
            case asm.Cvtss2sd(src=src, dst=dst) if _is_mem(dst):
                # cvtss2sd dst must be an XMM register; route through %xmm14 then movsd to memory.
                xmm14 = asm.Register(asm.FPRegisters.XMM14, asm.Width.QWORD)
                fixed.append(asm.Cvtss2sd(src=src, dst=xmm14))
                fixed.append(asm.Mov(src=xmm14, dst=dst, assembly_type=asm.Double()))
            case asm.Cvtsd2ss(src=src, dst=dst) if _is_mem(dst):
                # cvtsd2ss dst must be an XMM register; route through %xmm14 then movss to memory.
                xmm14 = asm.Register(asm.FPRegisters.XMM14, asm.Width.DWORD)
                fixed.append(asm.Cvtsd2ss(src=src, dst=xmm14))
                fixed.append(asm.Mov(src=xmm14, dst=dst, assembly_type=asm.Float()))
            case asm.Push(assembly_type=asm.Longword(), operand=asm.Imm(value=v)) if not _fits_in_int32(v):
                # pushq with a sign-extended 32-bit immediate cannot represent an unsigned int
                # value with bit 31 set. Reinterpret as signed int32 for the movl (GAS accepts any
                # 32-bit immediate there), load into %r10d - movl zero-extends so %r10 holds the
                # correct 64-bit unsigned value - then pushq the 64-bit register.
                truncated = v & 0xFFFFFFFF
                if truncated >= (1 << 31):
                    truncated -= 1 << 32
                r10d = asm.Register(asm.GPRegisters.R10, asm.Width.DWORD)
                r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
                fixed.append(asm.Mov(src=asm.Imm(truncated), dst=r10d, assembly_type=asm.Longword()))
                fixed.append(asm.Push(operand=r10q, assembly_type=asm.Quadword()))
            case asm.Push(assembly_type=asm.Quadword(), operand=asm.Imm(value=v)) if not _fits_in_int32(v):
                # pushq only accepts 8-bit or 32-bit sign-extended immediates; a 64-bit long
                # constant that doesn't fit in 32 bits must be routed through %r10.
                r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
                fixed.append(asm.Mov(src=asm.Imm(v), dst=r10q, assembly_type=asm.Quadword()))
                fixed.append(asm.Push(operand=r10q, assembly_type=asm.Quadword()))
            case asm.Push(assembly_type=asm.Double(), operand=asm.Register() as xmm_reg):
                # pushq %xmmN is not encodable; write explicitly through the stack pointer.
                rsp = asm.Register(asm.GPRegisters.SP, asm.Width.QWORD)
                fixed.append(asm.AllocateStack(8))
                fixed.append(asm.Mov(src=xmm_reg, dst=asm.Memory(reg=rsp, offset=0), assembly_type=asm.Double()))
            case asm.Push(assembly_type=asm.Double(), operand=operand) if _is_mem(operand):
                # movsd cannot move to a GP register, but the double's bits can be
                # loaded as a raw quadword with movq and then pushed normally.
                # Use R10 (not AX) - R10 is excluded from register allocation so it
                # cannot hold a pseudo that is live across this push.
                r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
                fixed.append(asm.Mov(src=operand, dst=r10q, assembly_type=asm.Quadword()))
                fixed.append(asm.Push(operand=r10q, assembly_type=asm.Quadword()))
            case asm.Push(assembly_type=at, operand=operand) if _is_mem(operand):
                # pushq cannot address a memory slot directly.
                # Load via movl (zero-extends) for ints, movq for longs, then push the 64-bit register.
                # Use R10 (not AX) - R10 is excluded from register allocation so it
                # cannot hold a pseudo that is live across this push.
                r10 = asm.Register(asm.GPRegisters.R10, _reg_width(at))
                fixed.append(asm.Mov(src=operand, dst=r10, assembly_type=at))
                fixed.append(asm.Push(operand=asm.Register(asm.GPRegisters.R10, asm.Width.QWORD), assembly_type=at))
            case asm.Lea(src=src, dst=dst) if _is_mem(dst):
                # leaq destination must be a register; route through %r11 then store.
                r11q = asm.Register(asm.GPRegisters.R11, asm.Width.QWORD)
                fixed.append(asm.Lea(src=src, dst=r11q))
                fixed.append(asm.Mov(src=r11q, dst=dst, assembly_type=asm.Quadword()))
            case _:
                fixed.append(instr)
    return dataclasses.replace(fn, instructions=tuple(fixed))


# ==============================================================================
# Callee-saved register save / restore
# ==============================================================================


def _insert_callee_saves(
    fn: asm.Function,
    gp_callee: list[asm.GPRegisters],
    fp_callee: list[asm.FPRegisters],
) -> asm.Function:
    """Insert FP callee-saved register saves at function entry and restores before each ``Ret``.

    GP callee-saved registers are handled by ``Function.emit()`` via the
    ``gp_callee_saved`` field (emitted as ``pushq``/``popq``), so this function
    only inserts ``movsd`` saves and restores for XMM registers.  FP save slots
    begin immediately below the GP save area (``-8 * (len(gp_callee) + 1)``
    relative to ``%rbp``).

    Args:
        fn: The assembly function to rewrite.
        gp_callee: Sorted list of callee-saved GP registers (used only to
            compute the starting offset for FP saves; no GP instructions are
            emitted here).
        fp_callee: Sorted list of callee-saved FP (XMM) registers that were
            assigned to pseudos during register allocation.

    Returns:
        A new ``asm.Function`` with FP save instructions prepended and FP
        restore instructions inserted before every ``Ret``.  Returns ``fn``
        unchanged when ``fp_callee`` is empty.
    """
    if not fp_callee:
        return fn

    bp = asm.Register(asm.GPRegisters.BP, asm.Width.QWORD)

    saves: list[asm.Instruction] = []
    restores: list[asm.Instruction] = []

    offset = -8 * (len(gp_callee) + 1)
    for reg in fp_callee:
        mem = asm.Memory(bp, offset)
        saves.append(asm.Mov(src=asm.Register(reg, asm.Width.QWORD), dst=mem, assembly_type=asm.Double()))
        restores.append(asm.Mov(src=mem, dst=asm.Register(reg, asm.Width.QWORD), assembly_type=asm.Double()))
        offset -= 8

    restores.reverse()

    new_instrs: list[asm.Instruction] = list(saves)
    for instr in fn.instructions:
        if isinstance(instr, asm.Ret):
            new_instrs.extend(restores)
        new_instrs.append(instr)

    return dataclasses.replace(fn, instructions=tuple(new_instrs))


# ==============================================================================
# Stack allocation
# ==============================================================================


def _add_stack_allocation(fn: asm.Function, stack_size: int, *, gp_callee_size: int = 0) -> asm.Function:
    """Prepend an ``AllocateStack`` instruction when the function uses the stack.

    The allocated size is chosen so that ``gp_callee_size + AllocateStack`` is
    a multiple of 16, preserving the 16-byte stack alignment required by the
    System V AMD64 ABI.  When ``gp_callee_size`` is an odd multiple of 8 (i.e.
    the function pushes an odd number of callee-saved GP registers), an extra 8
    bytes of padding are added.

    Args:
        fn: The assembly function to modify.
        stack_size: The raw number of stack bytes needed for locals and FP callee saves.
        gp_callee_size: Bytes already consumed on the stack by ``pushq`` saves
            (``8 * len(gp_callee)``).  Used to compute alignment padding.

    Returns:
        An ``asm.Function`` with an ``AllocateStack`` prepended, or the
        original function unchanged when no allocation is required.
    """
    needed_mod = (-gp_callee_size) % 16  # 0 or 8: how much AllocateStack must be (mod 16)
    if needed_mod == 0:
        aligned = (stack_size + 15) & ~15
    else:
        # Odd number of pushq saves: AllocateStack must be == 8 (mod 16) so that the
        # combined push+alloc is 0 (mod 16).
        aligned = ((stack_size + 7) // 16) * 16 + 8
    if aligned == 0:
        return fn
    return dataclasses.replace(fn, instructions=(asm.AllocateStack(size=aligned), *fn.instructions))
