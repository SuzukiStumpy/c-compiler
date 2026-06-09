"""Constant-folding optimization pass."""

import math

import compiler.syntax.tac as tac
from compiler.syntax.ast.constants import (
    Const,
    ConstChar,
    ConstDouble,
    ConstInt,
    ConstLong,
    ConstUChar,
    ConstUInt,
    ConstULong,
)
from compiler.syntax.ast.types import (
    Char,
    DoubleType,
    IntType,
    LongType,
    SChar,
    Type,
    UChar,
    UIntType,
    ULongType,
)
from compiler.syntax.tac import instructions as ti
from compiler.syntax.tac import operators as to
from compiler.syntax.tac.values import Constant
from compiler.types.symbols import SymbolTable


def _val(c: Const) -> int | float:
    """Extract the numeric value from any ``Const`` subclass.

    Uses structural pattern matching so type checkers can verify the attribute
    access on each concrete type.

    Args:
        c: Any ``Const`` instance.

    Returns:
        The integer or floating-point value held by ``c``.

    Raises:
        TypeError: If ``c`` is an unrecognised ``Const`` subclass.
    """
    match c:
        case ConstInt(val=v) | ConstLong(val=v) | ConstUInt(val=v) | ConstULong(val=v):
            return v
        case ConstChar(val=v) | ConstUChar(val=v):
            return v
        case ConstDouble(val=v):
            return v
    raise TypeError(f"Unrecognised Const subclass: {type(c)}")


def _wrap_int(value: int, bits: int, *, signed: bool) -> int:
    """Reduce ``value`` to the representable range for a ``bits``-wide integer.

    Args:
        value: Arbitrary-precision Python integer to wrap.
        bits: Bit width of the target type.
        signed: If ``True``, treat the type as two's-complement signed.

    Returns:
        The wrapped integer in ``[-2^(bits-1), 2^(bits-1)-1]`` when signed,
        or ``[0, 2^bits-1]`` when unsigned.
    """
    mod = 1 << bits
    value = value % mod
    if signed and value >= mod // 2:
        value -= mod
    return value


def _wrap_const(proto: Const, value: int | float) -> Const:
    """Create a new ``Const`` of the same type as *proto*, wrapping *value* to its range.

    Args:
        proto: A ``Const`` instance whose concrete type determines the output type and range.
        value: The computed result to wrap.

    Returns:
        A new ``Const`` of the same subclass as ``proto`` with the wrapped value.

    Raises:
        TypeError: If ``proto`` is an unrecognised ``Const`` subclass.
    """
    match proto:
        case ConstInt():
            return ConstInt(_wrap_int(int(value), 32, signed=True))
        case ConstLong():
            return ConstLong(_wrap_int(int(value), 64, signed=True))
        case ConstUInt():
            return ConstUInt(_wrap_int(int(value), 32, signed=False))
        case ConstULong():
            return ConstULong(_wrap_int(int(value), 64, signed=False))
        case ConstDouble():
            return ConstDouble(float(value))
        case ConstChar():
            return ConstChar(_wrap_int(int(value), 8, signed=True))
        case ConstUChar():
            return ConstUChar(_wrap_int(int(value), 8, signed=False))
    raise TypeError(f"Unrecognised Const subclass: {type(proto)}")


def _bit_width(c: Const) -> int:
    """Return the bit width of an integer ``Const`` type.

    Args:
        c: A ``Const`` instance.

    Returns:
        Bit width (8, 32, or 64). Returns 0 for ``ConstDouble``.
    """
    match c:
        case ConstChar() | ConstUChar():
            return 8
        case ConstInt() | ConstUInt():
            return 32
        case ConstLong() | ConstULong():
            return 64
    return 0


def _c_int_div(a: int, b: int) -> int:
    """Integer division truncating toward zero, matching C semantics.

    Args:
        a: Dividend.
        b: Divisor (must be non-zero).

    Returns:
        ``a / b`` truncated toward zero.
    """
    q = a // b
    if (a % b) != 0 and (a < 0) != (b < 0):
        q += 1
    return q


def _float_div(a: float, b: float) -> float:
    """IEEE 754 floating-point division, including the division-by-zero cases.

    Python raises ``ZeroDivisionError`` for any ``float / 0.0``, which is
    incorrect for IEEE 754.  This function reproduces the standard results:

    * ``0.0 / 0.0`` (or ``-0.0`` in any combination) → NaN (invalid operation).
    * finite non-zero / ±0.0 → ±∞, sign determined by XOR of operand signs.
    * ±∞ / ±0.0 → ±∞ with the same sign rule.
    * Any NaN operand → NaN.

    Args:
        a: Dividend.
        b: Divisor.

    Returns:
        The IEEE 754 result of ``a / b``.
    """
    if b != 0.0:
        return a / b
    if a == 0.0 or math.isnan(a):
        return float("nan")
    return math.copysign(float("inf"), math.copysign(1.0, a) * math.copysign(1.0, b))


def _const_of_type(t: Type, val: int | float) -> Const | None:
    """Create a ``Const`` of the C type ``t``, wrapping ``val`` to the type's range.

    Args:
        t: The target C type.
        val: The numeric value to store.

    Returns:
        A concrete ``Const`` subclass with the value wrapped to the type's range,
        or ``None`` for non-scalar types (pointers, arrays, structs, void, functions)
        for which no ``Const`` representation exists.
    """
    match t:
        case IntType():
            return ConstInt(_wrap_int(int(val), 32, signed=True))
        case LongType():
            return ConstLong(_wrap_int(int(val), 64, signed=True))
        case UIntType():
            return ConstUInt(_wrap_int(int(val), 32, signed=False))
        case ULongType():
            return ConstULong(_wrap_int(int(val), 64, signed=False))
        case DoubleType():
            return ConstDouble(float(val))
        case Char() | SChar():
            return ConstChar(_wrap_int(int(val), 8, signed=True))
        case UChar():
            return ConstUChar(_wrap_int(int(val), 8, signed=False))
    return None


def _dst_type(dst: tac.Val, symbol_table: SymbolTable) -> Type | None:
    """Return the C type of a destination ``Val`` from the symbol table.

    Args:
        dst: The destination operand of a TAC instruction.
        symbol_table: The program's symbol table.

    Returns:
        The ``Type`` recorded for this variable, or ``None`` if ``dst`` is not
        a ``Var`` or is not found in the symbol table.
    """
    if isinstance(dst, tac.Var):
        entry = symbol_table.get(dst.value)
        if entry is not None:
            return entry.t
    return None


def _fold_unary(instr: ti.Unary) -> ti.Instruction:
    """Fold a unary instruction whose source is a constant.

    Args:
        instr: The ``Unary`` instruction to attempt to fold.

    Returns:
        A ``Copy`` instruction with the computed constant if folding succeeded,
        otherwise the original ``instr`` unchanged.
    """
    if not isinstance(instr.src, Constant):
        return instr
    c = instr.src.value
    v = _val(c)
    match instr.operator:
        case to.Negate():
            return ti.Copy(src=Constant(_wrap_const(c, -v)), dst=instr.dst)
        case to.Complement():
            if isinstance(c, ConstDouble):
                return instr
            return ti.Copy(src=Constant(_wrap_const(c, ~int(v))), dst=instr.dst)
        case to.Not():
            return ti.Copy(src=Constant(ConstInt(0 if v != 0 else 1)), dst=instr.dst)
    return instr


def _compute_arithmetic(op: to.BinaryOperator, c1: Const, c2: Const) -> Const | None:
    """Evaluate an arithmetic or bitwise binary operation on two constants.

    Args:
        op: The binary operator to apply.
        c1: Left-hand constant (determines result type).
        c2: Right-hand constant.

    Returns:
        The folded ``Const`` result, or ``None`` if the operation cannot be folded
        (division by zero, double bitwise op, out-of-range shift).
    """
    a, b = _val(c1), _val(c2)
    match op:
        case to.Add():
            return _wrap_const(c1, a + b)
        case to.Subtract():
            return _wrap_const(c1, a - b)
        case to.Multiply():
            return _wrap_const(c1, a * b)
        case to.Divide():
            if isinstance(c1, ConstDouble):
                result: int | float = _float_div(float(a), float(b))
            elif b == 0:
                return None  # integer division by zero is UB
            else:
                result = _c_int_div(int(a), int(b))
            return _wrap_const(c1, result)
        case to.Remainder():
            if b == 0 or isinstance(c1, ConstDouble):
                return None
            return _wrap_const(c1, int(a) - _c_int_div(int(a), int(b)) * int(b))
        case to.BitwiseAnd():
            return None if isinstance(c1, ConstDouble) else _wrap_const(c1, int(a) & int(b))
        case to.BitwiseOr():
            return None if isinstance(c1, ConstDouble) else _wrap_const(c1, int(a) | int(b))
        case to.BitwiseXor():
            return None if isinstance(c1, ConstDouble) else _wrap_const(c1, int(a) ^ int(b))
        case to.LeftShift():
            shift = int(b)
            if isinstance(c1, ConstDouble) or shift < 0 or shift >= _bit_width(c1):
                return None
            return _wrap_const(c1, int(a) << shift)
        case to.RightShift():
            shift = int(b)
            if isinstance(c1, ConstDouble) or shift < 0 or shift >= _bit_width(c1):
                return None
            return _wrap_const(c1, int(a) >> shift)
    return None


def _compute_comparison(op: to.BinaryOperator, a: int | float, b: int | float) -> ConstInt | None:
    """Evaluate a comparison binary operation on two constant values.

    Args:
        op: The comparison operator to apply.
        a: Left-hand value.
        b: Right-hand value.

    Returns:
        ``ConstInt(1)`` or ``ConstInt(0)`` for a recognised comparison operator,
        or ``None`` for non-comparison operators.
    """
    match op:
        case to.Equal():
            return ConstInt(1 if a == b else 0)
        case to.NotEqual():
            return ConstInt(1 if a != b else 0)
        case to.LessThan():
            return ConstInt(1 if a < b else 0)
        case to.LessOrEqual():
            return ConstInt(1 if a <= b else 0)
        case to.GreaterThan():
            return ConstInt(1 if a > b else 0)
        case to.GreaterOrEqual():
            return ConstInt(1 if a >= b else 0)
    return None


def _fold_binary(instr: ti.Binary) -> ti.Instruction:
    """Fold a binary instruction whose both sources are constants.

    Args:
        instr: The ``Binary`` instruction to attempt to fold.

    Returns:
        A ``Copy`` instruction with the computed constant if folding succeeded,
        otherwise the original ``instr`` unchanged.
    """
    if not isinstance(instr.src1, Constant) or not isinstance(instr.src2, Constant):
        return instr
    c1, c2 = instr.src1.value, instr.src2.value
    result: Const | None = _compute_comparison(instr.operator, _val(c1), _val(c2)) or _compute_arithmetic(
        instr.operator, c1, c2
    )
    if result is None:
        return instr
    return ti.Copy(src=Constant(result), dst=instr.dst)


def _fold_conditional_jump(instr: ti.JumpIfZero | ti.JumpIfNotZero) -> ti.Instruction | None:
    """Fold a conditional jump whose condition is a constant.

    Args:
        instr: A ``JumpIfZero`` or ``JumpIfNotZero`` instruction.

    Returns:
        An unconditional ``Jump`` if the condition is statically met, ``None`` if the
        condition is statically not met (instruction can be removed), or the original
        ``instr`` when the condition is not a constant.
    """
    if not isinstance(instr.condition, Constant):
        return instr
    taken = _val(instr.condition.value) == 0 if isinstance(instr, ti.JumpIfZero) else _val(instr.condition.value) != 0
    return ti.Jump(target=instr.target) if taken else None


def _fold_conversion(instr: ti.Instruction, symbol_table: SymbolTable) -> ti.Instruction:
    """Fold a type-conversion instruction whose source operand is a constant.

    Handles ``Truncate``, ``SignExtend``, ``ZeroExtend``, ``IntToDouble``,
    ``UIntToDouble``, ``DoubleToInt``, and ``DoubleToUInt``.  Conversions that
    require a destination type look it up from *symbol_table*; if the variable is
    not present the instruction is returned unchanged.

    The numeric semantics match the runtime behaviour:

    * Integer narrowing (``Truncate``) keeps the low-order bits and reinterprets
      them according to the destination type's signedness.
    * Integer widening (``SignExtend``, ``ZeroExtend``) preserves the source value
      in Python's arbitrary-precision representation, which is already equivalent
      to the extended value.
    * Floating-to-integer (``DoubleToInt``, ``DoubleToUInt``) truncates toward
      zero.  Non-finite sources (inf, nan) are not folded.
    * Integer-to-floating (``IntToDouble``, ``UIntToDouble``) converts exactly,
      subject to IEEE 754 rounding for large integers.

    Args:
        instr: Any TAC instruction; only conversion variants with a constant source
            are folded — others are returned unchanged.
        symbol_table: The program's symbol table, used to resolve destination types.

    Returns:
        A ``Copy`` with the folded constant, or the original ``instr`` unchanged.
    """
    match instr:
        case (
            ti.SignExtend(src=Constant(value=c), dst=dst)
            | ti.ZeroExtend(src=Constant(value=c), dst=dst)
            | ti.Truncate(src=Constant(value=c), dst=dst)
        ):
            dst_t = _dst_type(dst, symbol_table)
            if dst_t is None:
                return instr
            result = _const_of_type(dst_t, _val(c))
            return ti.Copy(src=Constant(result), dst=dst) if result is not None else instr

        case ti.IntToDouble(src=Constant(value=c), dst=dst) | ti.UIntToDouble(src=Constant(value=c), dst=dst):
            return ti.Copy(src=Constant(ConstDouble(float(_val(c)))), dst=dst)

        case ti.DoubleToInt(src=Constant(value=c), dst=dst) | ti.DoubleToUInt(src=Constant(value=c), dst=dst):
            v = _val(c)
            if not (isinstance(v, float) and math.isfinite(v)):
                return instr
            dst_t = _dst_type(dst, symbol_table)
            if dst_t is None:
                return instr
            result = _const_of_type(dst_t, int(v))
            return ti.Copy(src=Constant(result), dst=dst) if result is not None else instr

    return instr


def _fold_instruction(instr: ti.Instruction, symbol_table: SymbolTable) -> ti.Instruction | None:
    """Dispatch a single instruction to the appropriate folding handler.

    Args:
        instr: Any TAC instruction.
        symbol_table: The program's symbol table, forwarded to conversion folding.

    Returns:
        A folded replacement instruction, ``None`` if the instruction should be
        removed, or the original if no folding applied.
    """
    match instr:
        case ti.Unary():
            return _fold_unary(instr)
        case ti.Binary():
            return _fold_binary(instr)
        case ti.JumpIfZero() | ti.JumpIfNotZero():
            return _fold_conditional_jump(instr)
        case (
            ti.SignExtend()
            | ti.ZeroExtend()
            | ti.Truncate()
            | ti.IntToDouble()
            | ti.UIntToDouble()
            | ti.DoubleToInt()
            | ti.DoubleToUInt()
        ):
            return _fold_conversion(instr, symbol_table)
    return instr


def constant_folding(function: tac.Function, symbol_table: SymbolTable | None = None) -> tac.Function:
    """Evaluate constant expressions at compile time and replace them with their results.

    Folds ``Unary`` and ``Binary`` instructions whose source operands are all constants
    into ``Copy`` instructions carrying the pre-computed result.  ``JumpIfZero`` and
    ``JumpIfNotZero`` with constant conditions are replaced with an unconditional
    ``Jump`` when the condition is statically met, or removed entirely when it is not.
    Type-conversion instructions (``Truncate``, ``SignExtend``, ``ZeroExtend``,
    ``IntToDouble``, ``UIntToDouble``, ``DoubleToInt``, ``DoubleToUInt``) are folded
    when their source is a constant; conversions whose destination type cannot be
    determined without *symbol_table* are left unchanged when it is ``None``.

    Args:
        function: The TAC function to optimize.
        symbol_table: The program's symbol table.  Required to resolve the destination
            type of ``Truncate``, ``SignExtend``, ``ZeroExtend``, ``DoubleToInt``, and
            ``DoubleToUInt`` instructions; omitting it skips those folds.

    Returns:
        A new ``tac.Function`` with constant expressions folded where possible.
        Returns the original object when no instruction changed.
    """
    st: SymbolTable = symbol_table if symbol_table is not None else {}
    new_body = tuple(r for instr in function.body if (r := _fold_instruction(instr, st)) is not None)
    if new_body == function.body:
        return function
    return tac.Function(
        name=function.name,
        globl=function.globl,
        params=function.params,
        body=new_body,
        source_file=function.source_file,
        source_line=function.source_line,
    )
