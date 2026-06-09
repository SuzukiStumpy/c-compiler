"""Unit tests for the constant-folding optimisation pass.

Tests ``constant_folding`` in isolation, building TAC nodes directly without
invoking the lexer, parser, or TAC generator.
"""

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from compiler.optimiser.constant_folding import constant_folding
from compiler.syntax.ast.types import (
    Char,
    IntType,
    LongType,
    SChar,
    UChar,
    UIntType,
    ULongType,
)
from compiler.syntax.tac import operators as to
from compiler.types.symbols import LocalAttr, SymbolTable, SymbolTableEntry

from .helpers import _cd, _ci, _cl, _cui, _cul, _function, _var


def _sym(name: str, t: ast.Type) -> SymbolTable:
    """Build a one-entry symbol table mapping ``name`` to type ``t``.

    Args:
        name: The variable name to register.
        t: The C type to associate with the variable.

    Returns:
        A ``SymbolTable`` containing a single ``LocalAttr`` entry for ``name``.
    """
    return {name: SymbolTableEntry(t=t, attrs=LocalAttr())}


_DST = _var("dst")


# ---------------------------------------------------------------------------
# Identity / no-op cases
# ---------------------------------------------------------------------------


class TestConstantFoldingNoOp:
    """Pass-through cases where no folding should occur."""

    def test_non_constant_unary_is_unchanged(self) -> None:
        """A ``Unary`` with a variable source is not folded."""
        instr = tac.Unary(operator=to.Negate(), src=_var("x"), dst=_DST)
        fn = _function((instr,))
        assert constant_folding(fn) is fn

    def test_non_constant_binary_src1_is_unchanged(self) -> None:
        """A ``Binary`` where src1 is a variable is not folded."""
        instr = tac.Binary(operator=to.Add(), src1=_var("x"), src2=_ci(1), dst=_DST)
        fn = _function((instr,))
        assert constant_folding(fn) is fn

    def test_non_constant_binary_src2_is_unchanged(self) -> None:
        """A ``Binary`` where src2 is a variable is not folded."""
        instr = tac.Binary(operator=to.Add(), src1=_ci(1), src2=_var("y"), dst=_DST)
        fn = _function((instr,))
        assert constant_folding(fn) is fn

    def test_copy_instruction_passes_through(self) -> None:
        """Non-arithmetic instructions are left untouched."""
        instr = tac.Copy(src=_ci(5), dst=_DST)
        fn = _function((instr,))
        assert constant_folding(fn) is fn

    def test_unchanged_function_returns_same_object(self) -> None:
        """When no instruction changes, the original ``Function`` object is returned."""
        fn = _function((tac.Return(val=_ci(0)),))
        assert constant_folding(fn) is fn


# ---------------------------------------------------------------------------
# Unary folding
# ---------------------------------------------------------------------------


class TestFoldUnary:
    """Folding of ``Unary`` instructions with constant sources."""

    def test_negate_int(self) -> None:
        """``Negate`` folds to ``Copy(Constant(-val))`` for ConstInt."""
        instr = tac.Unary(operator=to.Negate(), src=_ci(5), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(-5), dst=_DST),)

    def test_negate_long(self) -> None:
        """``Negate`` works on ConstLong."""
        instr = tac.Unary(operator=to.Negate(), src=_cl(100), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_cl(-100), dst=_DST),)

    def test_negate_double(self) -> None:
        """``Negate`` folds to the negative double."""
        instr = tac.Unary(operator=to.Negate(), src=_cd(3.5), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_cd(-3.5), dst=_DST),)

    def test_negate_zero(self) -> None:
        """``Negate(0)`` folds to ``Copy(0)``."""
        instr = tac.Unary(operator=to.Negate(), src=_ci(0), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(0), dst=_DST),)

    def test_complement_int(self) -> None:
        """``Complement`` folds ~5 == -6 for ConstInt."""
        instr = tac.Unary(operator=to.Complement(), src=_ci(5), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(-6), dst=_DST),)

    def test_complement_zero(self) -> None:
        """``Complement`` folds ~0 == -1 for ConstInt."""
        instr = tac.Unary(operator=to.Complement(), src=_ci(0), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(-1), dst=_DST),)

    def test_complement_unsigned_wraps(self) -> None:
        """``Complement`` on ConstUInt wraps correctly: ~0u == 0xFFFFFFFF."""
        instr = tac.Unary(operator=to.Complement(), src=_cui(0), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_cui(0xFFFF_FFFF), dst=_DST),)

    def test_complement_double_not_folded(self) -> None:
        """``Complement`` on a double is not folded (bitwise ops undefined for float)."""
        instr = tac.Unary(operator=to.Complement(), src=_cd(1.0), dst=_DST)
        fn = _function((instr,))
        assert constant_folding(fn) is fn

    def test_not_nonzero_gives_zero(self) -> None:
        """``Not`` of a non-zero constant gives ``ConstInt(0)``."""
        instr = tac.Unary(operator=to.Not(), src=_ci(42), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(0), dst=_DST),)

    def test_not_zero_gives_one(self) -> None:
        """``Not`` of zero gives ``ConstInt(1)``."""
        instr = tac.Unary(operator=to.Not(), src=_ci(0), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(1), dst=_DST),)

    def test_not_double_nonzero(self) -> None:
        """``Not`` of a non-zero double gives ``ConstInt(0)``."""
        instr = tac.Unary(operator=to.Not(), src=_cd(1.5), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(0), dst=_DST),)


# ---------------------------------------------------------------------------
# Binary arithmetic folding
# ---------------------------------------------------------------------------


class TestFoldBinaryArithmetic:
    """Folding of arithmetic ``Binary`` instructions."""

    def test_add_int(self) -> None:
        """``Add(1, 2)`` folds to ``Copy(3)``."""
        instr = tac.Binary(operator=to.Add(), src1=_ci(1), src2=_ci(2), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(3), dst=_DST),)

    def test_subtract_int(self) -> None:
        """``Subtract(10, 3)`` folds to ``Copy(7)``."""
        instr = tac.Binary(operator=to.Subtract(), src1=_ci(10), src2=_ci(3), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(7), dst=_DST),)

    def test_multiply_int(self) -> None:
        """``Multiply(6, 7)`` folds to ``Copy(42)``."""
        instr = tac.Binary(operator=to.Multiply(), src1=_ci(6), src2=_ci(7), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(42), dst=_DST),)

    def test_divide_int_truncates_toward_zero(self) -> None:
        """Integer division truncates toward zero: ``-7 / 2 == -3`` (not ``-4``)."""
        instr = tac.Binary(operator=to.Divide(), src1=_ci(-7), src2=_ci(2), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(-3), dst=_DST),)

    def test_divide_by_zero_not_folded(self) -> None:
        """Integer division by zero is left unchanged (undefined behaviour)."""
        instr = tac.Binary(operator=to.Divide(), src1=_ci(5), src2=_ci(0), dst=_DST)
        fn = _function((instr,))
        assert constant_folding(fn) is fn

    def test_remainder_int(self) -> None:
        """``Remainder(7, 3)`` folds to ``Copy(1)``."""
        instr = tac.Binary(operator=to.Remainder(), src1=_ci(7), src2=_ci(3), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(1), dst=_DST),)

    def test_remainder_sign_follows_dividend(self) -> None:
        """``Remainder(-7, 3)`` is ``-1`` (sign follows dividend, matching C semantics)."""
        instr = tac.Binary(operator=to.Remainder(), src1=_ci(-7), src2=_ci(3), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(-1), dst=_DST),)

    def test_remainder_by_zero_not_folded(self) -> None:
        """Remainder by zero is left unchanged."""
        instr = tac.Binary(operator=to.Remainder(), src1=_ci(5), src2=_ci(0), dst=_DST)
        fn = _function((instr,))
        assert constant_folding(fn) is fn

    def test_add_double(self) -> None:
        """``Add`` on ConstDouble folds correctly."""
        instr = tac.Binary(operator=to.Add(), src1=_cd(1.5), src2=_cd(2.5), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_cd(4.0), dst=_DST),)

    def test_divide_double(self) -> None:
        """``Divide`` on ConstDouble folds correctly."""
        instr = tac.Binary(operator=to.Divide(), src1=_cd(7.0), src2=_cd(2.0), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_cd(3.5), dst=_DST),)


# ---------------------------------------------------------------------------
# Integer overflow wrapping
# ---------------------------------------------------------------------------


class TestIntegerOverflowWrapping:
    """Result values are wrapped to the target type's bit width."""

    def test_int_overflow_wraps_to_32_bit(self) -> None:
        """Adding past INT_MAX wraps to the correct 32-bit two's-complement value."""
        instr = tac.Binary(operator=to.Add(), src1=_ci(2**31 - 1), src2=_ci(1), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(-(2**31)), dst=_DST),)

    def test_long_overflow_wraps_to_64_bit(self) -> None:
        """Adding past LLONG_MAX wraps to the correct 64-bit value."""
        instr = tac.Binary(operator=to.Add(), src1=_cl(2**63 - 1), src2=_cl(1), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_cl(-(2**63)), dst=_DST),)

    def test_uint_overflow_wraps_to_32_bit(self) -> None:
        """Adding past UINT_MAX wraps to zero for unsigned 32-bit."""
        instr = tac.Binary(operator=to.Add(), src1=_cui(2**32 - 1), src2=_cui(1), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_cui(0), dst=_DST),)

    def test_ulong_overflow_wraps_to_64_bit(self) -> None:
        """Adding past ULLONG_MAX wraps to zero for unsigned 64-bit."""
        instr = tac.Binary(operator=to.Add(), src1=_cul(2**64 - 1), src2=_cul(1), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_cul(0), dst=_DST),)


# ---------------------------------------------------------------------------
# Binary bitwise folding
# ---------------------------------------------------------------------------


class TestFoldBinaryBitwise:
    """Folding of bitwise ``Binary`` instructions."""

    def test_bitwise_and(self) -> None:
        """``BitwiseAnd(0b1100, 0b1010)`` folds to ``0b1000``."""
        instr = tac.Binary(operator=to.BitwiseAnd(), src1=_ci(0b1100), src2=_ci(0b1010), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(0b1000), dst=_DST),)

    def test_bitwise_or(self) -> None:
        """``BitwiseOr(0b1100, 0b0011)`` folds to ``0b1111``."""
        instr = tac.Binary(operator=to.BitwiseOr(), src1=_ci(0b1100), src2=_ci(0b0011), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(0b1111), dst=_DST),)

    def test_bitwise_xor(self) -> None:
        """``BitwiseXor(0b1100, 0b1010)`` folds to ``0b0110``."""
        instr = tac.Binary(operator=to.BitwiseXor(), src1=_ci(0b1100), src2=_ci(0b1010), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(0b0110), dst=_DST),)

    def test_left_shift(self) -> None:
        """``LeftShift(1, 3)`` folds to ``8``."""
        instr = tac.Binary(operator=to.LeftShift(), src1=_ci(1), src2=_ci(3), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(8), dst=_DST),)

    def test_right_shift(self) -> None:
        """``RightShift(16, 2)`` folds to ``4``."""
        instr = tac.Binary(operator=to.RightShift(), src1=_ci(16), src2=_ci(2), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(4), dst=_DST),)

    def test_right_shift_negative_is_arithmetic(self) -> None:
        """``RightShift(-8, 1)`` gives ``-4`` (arithmetic shift)."""
        instr = tac.Binary(operator=to.RightShift(), src1=_ci(-8), src2=_ci(1), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(-4), dst=_DST),)

    def test_shift_by_negative_amount_not_folded(self) -> None:
        """A shift by a negative amount is undefined behaviour and not folded."""
        instr = tac.Binary(operator=to.LeftShift(), src1=_ci(1), src2=_ci(-1), dst=_DST)
        fn = _function((instr,))
        assert constant_folding(fn) is fn

    def test_shift_by_width_not_folded(self) -> None:
        """A shift by exactly the bit width (32 for int) is undefined and not folded."""
        instr = tac.Binary(operator=to.LeftShift(), src1=_ci(1), src2=_ci(32), dst=_DST)
        fn = _function((instr,))
        assert constant_folding(fn) is fn

    def test_bitwise_ops_on_double_not_folded(self) -> None:
        """Bitwise ops on double constants are not folded."""
        instr = tac.Binary(operator=to.BitwiseAnd(), src1=_cd(1.0), src2=_cd(1.0), dst=_DST)
        fn = _function((instr,))
        assert constant_folding(fn) is fn


# ---------------------------------------------------------------------------
# Binary comparison folding
# ---------------------------------------------------------------------------


class TestFoldBinaryComparison:
    """Folding of comparison ``Binary`` instructions (result is always ``ConstInt``)."""

    def test_equal_true(self) -> None:
        """``Equal(3, 3)`` folds to ``ConstInt(1)``."""
        instr = tac.Binary(operator=to.Equal(), src1=_ci(3), src2=_ci(3), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(1), dst=_DST),)

    def test_equal_false(self) -> None:
        """``Equal(3, 4)`` folds to ``ConstInt(0)``."""
        instr = tac.Binary(operator=to.Equal(), src1=_ci(3), src2=_ci(4), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(0), dst=_DST),)

    def test_not_equal(self) -> None:
        """``NotEqual(3, 4)`` folds to ``ConstInt(1)``."""
        instr = tac.Binary(operator=to.NotEqual(), src1=_ci(3), src2=_ci(4), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(1), dst=_DST),)

    def test_less_than_true(self) -> None:
        """``LessThan(2, 5)`` folds to ``ConstInt(1)``."""
        instr = tac.Binary(operator=to.LessThan(), src1=_ci(2), src2=_ci(5), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(1), dst=_DST),)

    def test_less_than_false(self) -> None:
        """``LessThan(5, 2)`` folds to ``ConstInt(0)``."""
        instr = tac.Binary(operator=to.LessThan(), src1=_ci(5), src2=_ci(2), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(0), dst=_DST),)

    def test_less_or_equal_equal_case(self) -> None:
        """``LessOrEqual(3, 3)`` folds to ``ConstInt(1)``."""
        instr = tac.Binary(operator=to.LessOrEqual(), src1=_ci(3), src2=_ci(3), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(1), dst=_DST),)

    def test_greater_than(self) -> None:
        """``GreaterThan(5, 2)`` folds to ``ConstInt(1)``."""
        instr = tac.Binary(operator=to.GreaterThan(), src1=_ci(5), src2=_ci(2), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(1), dst=_DST),)

    def test_greater_or_equal_false(self) -> None:
        """``GreaterOrEqual(1, 5)`` folds to ``ConstInt(0)``."""
        instr = tac.Binary(operator=to.GreaterOrEqual(), src1=_ci(1), src2=_ci(5), dst=_DST)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_ci(0), dst=_DST),)

    def test_comparison_result_is_always_const_int(self) -> None:
        """Comparison of ConstLong operands still produces a ``ConstInt`` result."""
        instr = tac.Binary(operator=to.Equal(), src1=_cl(1), src2=_cl(1), dst=_DST)
        result = constant_folding(_function((instr,)))
        folded = result.body[0]
        assert isinstance(folded, tac.Copy)
        assert folded.src == tac.Constant(value=ast.ConstInt(1))


# ---------------------------------------------------------------------------
# Conditional jump folding
# ---------------------------------------------------------------------------


class TestFoldConditionalJumps:
    """Folding of ``JumpIfZero`` and ``JumpIfNotZero`` with constant conditions."""

    def test_jump_if_zero_with_zero_becomes_jump(self) -> None:
        """``JumpIfZero(0, target)`` → ``Jump(target)``."""
        instr = tac.JumpIfZero(condition=_ci(0), target="done")
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Jump(target="done"),)

    def test_jump_if_zero_with_nonzero_is_removed(self) -> None:
        """``JumpIfZero(1, target)`` is removed — condition never met."""
        instr = tac.JumpIfZero(condition=_ci(1), target="done")
        result = constant_folding(_function((instr,)))
        assert result.body == ()

    def test_jump_if_zero_with_negative_nonzero_is_removed(self) -> None:
        """``JumpIfZero(-1, target)`` is removed — nonzero condition never jumps."""
        instr = tac.JumpIfZero(condition=_ci(-1), target="done")
        result = constant_folding(_function((instr,)))
        assert result.body == ()

    def test_jump_if_not_zero_with_nonzero_becomes_jump(self) -> None:
        """``JumpIfNotZero(5, target)`` → ``Jump(target)``."""
        instr = tac.JumpIfNotZero(condition=_ci(5), target="loop")
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Jump(target="loop"),)

    def test_jump_if_not_zero_with_zero_is_removed(self) -> None:
        """``JumpIfNotZero(0, target)`` is removed — condition never met."""
        instr = tac.JumpIfNotZero(condition=_ci(0), target="loop")
        result = constant_folding(_function((instr,)))
        assert result.body == ()

    def test_jump_if_zero_double_zero_becomes_jump(self) -> None:
        """``JumpIfZero(0.0, target)`` → ``Jump(target)`` for double constants."""
        instr = tac.JumpIfZero(condition=_cd(0.0), target="end")
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Jump(target="end"),)

    def test_jump_if_zero_double_nonzero_is_removed(self) -> None:
        """``JumpIfZero(1.5, target)`` is removed for double constants."""
        instr = tac.JumpIfZero(condition=_cd(1.5), target="end")
        result = constant_folding(_function((instr,)))
        assert result.body == ()

    def test_jump_if_zero_variable_condition_unchanged(self) -> None:
        """``JumpIfZero`` with a variable condition is not folded."""
        instr = tac.JumpIfZero(condition=_var("x"), target="done")
        fn = _function((instr,))
        assert constant_folding(fn) is fn

    def test_jump_if_not_zero_variable_condition_unchanged(self) -> None:
        """``JumpIfNotZero`` with a variable condition is not folded."""
        instr = tac.JumpIfNotZero(condition=_var("x"), target="loop")
        fn = _function((instr,))
        assert constant_folding(fn) is fn

    def test_removed_jump_does_not_affect_surrounding_instructions(self) -> None:
        """Removing a dead conditional jump leaves adjacent instructions intact."""
        body = (
            tac.Copy(src=_ci(1), dst=_var("a")),
            tac.JumpIfZero(condition=_ci(1), target="skip"),
            tac.Return(val=_var("a")),
        )
        result = constant_folding(_function(body))
        assert result.body == (
            tac.Copy(src=_ci(1), dst=_var("a")),
            tac.Return(val=_var("a")),
        )


# ---------------------------------------------------------------------------
# Multi-instruction function
# ---------------------------------------------------------------------------


class TestFoldingInContext:
    """Constant folding within a multi-instruction function body."""

    def test_only_foldable_instructions_change(self) -> None:
        """Non-foldable instructions are preserved alongside folded ones."""
        body = (
            tac.Unary(operator=to.Negate(), src=_ci(3), dst=_var("a")),
            tac.Copy(src=_var("a"), dst=_var("b")),
            tac.Binary(operator=to.Add(), src1=_var("b"), src2=_ci(1), dst=_var("c")),
            tac.Return(val=_var("c")),
        )
        result = constant_folding(_function(body))
        assert result.body == (
            tac.Copy(src=_ci(-3), dst=_var("a")),
            tac.Copy(src=_var("a"), dst=_var("b")),
            tac.Binary(operator=to.Add(), src1=_var("b"), src2=_ci(1), dst=_var("c")),
            tac.Return(val=_var("c")),
        )


# ---------------------------------------------------------------------------
# Type-conversion folding
# ---------------------------------------------------------------------------


class TestFoldSignExtend:
    """Folding of ``SignExtend`` with a constant source."""

    def test_int_to_long(self) -> None:
        """``SignExtend(ConstInt(5), dst)`` → ``Copy(ConstLong(5), dst)``."""
        dst = _var("tmp")
        instr = tac.SignExtend(src=_ci(5), dst=dst)
        result = constant_folding(_function((instr,)), _sym("tmp", LongType()))
        assert result.body == (tac.Copy(src=_cl(5), dst=dst),)

    def test_negative_int_to_long(self) -> None:
        """Negative ConstInt sign-extends to the same negative ConstLong."""
        dst = _var("tmp")
        instr = tac.SignExtend(src=_ci(-1), dst=dst)
        result = constant_folding(_function((instr,)), _sym("tmp", LongType()))
        assert result.body == (tac.Copy(src=_cl(-1), dst=dst),)

    def test_char_to_int(self) -> None:
        """``SignExtend(ConstChar(-1), dst:int)`` → ``Copy(ConstInt(-1), dst)``."""
        dst = _var("tmp")
        instr = tac.SignExtend(src=tac.Constant(ast.ConstChar(-1)), dst=dst)
        result = constant_folding(_function((instr,)), _sym("tmp", IntType()))
        assert result.body == (tac.Copy(src=_ci(-1), dst=dst),)

    def test_variable_src_not_folded(self) -> None:
        """``SignExtend`` with a variable source is not folded."""
        instr = tac.SignExtend(src=_var("x"), dst=_var("tmp"))
        fn = _function((instr,))
        assert constant_folding(fn, _sym("tmp", LongType())) is fn

    def test_missing_symbol_table_not_folded(self) -> None:
        """``SignExtend`` without a symbol table is not folded."""
        instr = tac.SignExtend(src=_ci(5), dst=_var("tmp"))
        fn = _function((instr,))
        assert constant_folding(fn) is fn


class TestFoldZeroExtend:
    """Folding of ``ZeroExtend`` with a constant source."""

    def test_uint_to_ulong(self) -> None:
        """``ZeroExtend(ConstUInt(0xFFFFFFFF), dst:ulong)`` → ``Copy(ConstULong(0xFFFFFFFF), dst)``."""
        dst = _var("tmp")
        instr = tac.ZeroExtend(src=_cui(0xFFFF_FFFF), dst=dst)
        result = constant_folding(_function((instr,)), _sym("tmp", ULongType()))
        assert result.body == (tac.Copy(src=_cul(0xFFFF_FFFF), dst=dst),)

    def test_uint_to_long(self) -> None:
        """``ZeroExtend(ConstUInt(v), dst:long)`` zero-extends to signed long (value fits)."""
        dst = _var("tmp")
        instr = tac.ZeroExtend(src=_cui(0xFFFF_FFFF), dst=dst)
        result = constant_folding(_function((instr,)), _sym("tmp", LongType()))
        assert result.body == (tac.Copy(src=_cl(0xFFFF_FFFF), dst=dst),)

    def test_uchar_to_uint(self) -> None:
        """``ZeroExtend(ConstUChar(200), dst:uint)`` zero-extends to ConstUInt(200)."""
        dst = _var("tmp")
        instr = tac.ZeroExtend(src=tac.Constant(ast.ConstUChar(200)), dst=dst)
        result = constant_folding(_function((instr,)), _sym("tmp", UIntType()))
        assert result.body == (tac.Copy(src=_cui(200), dst=dst),)

    def test_missing_symbol_table_not_folded(self) -> None:
        """``ZeroExtend`` without a symbol table is not folded."""
        instr = tac.ZeroExtend(src=_cui(1), dst=_var("tmp"))
        fn = _function((instr,))
        assert constant_folding(fn) is fn


class TestFoldTruncate:
    """Folding of ``Truncate`` with a constant source."""

    def test_long_to_int(self) -> None:
        """``Truncate(ConstLong(5), dst:int)`` → ``Copy(ConstInt(5), dst)``."""
        dst = _var("tmp")
        instr = tac.Truncate(src=_cl(5), dst=dst)
        result = constant_folding(_function((instr,)), _sym("tmp", IntType()))
        assert result.body == (tac.Copy(src=_ci(5), dst=dst),)

    def test_long_truncates_upper_bits(self) -> None:
        """``Truncate(ConstLong(2^32 + 7), dst:int)`` keeps only the lower 32 bits."""
        dst = _var("tmp")
        instr = tac.Truncate(src=_cl(2**32 + 7), dst=dst)
        result = constant_folding(_function((instr,)), _sym("tmp", IntType()))
        assert result.body == (tac.Copy(src=_ci(7), dst=dst),)

    def test_truncation_preserves_sign(self) -> None:
        """Truncating a long whose lower 32 bits represent a negative int is correct."""
        dst = _var("tmp")
        # 2^32 - 1 = 0xFFFFFFFF; as signed int = -1
        instr = tac.Truncate(src=_cl(2**32 - 1), dst=dst)
        result = constant_folding(_function((instr,)), _sym("tmp", IntType()))
        assert result.body == (tac.Copy(src=_ci(-1), dst=dst),)

    def test_int_to_char(self) -> None:
        """``Truncate(ConstInt(300), dst:char)`` wraps to the 8-bit signed value (44)."""
        dst = _var("tmp")
        instr = tac.Truncate(src=_ci(300), dst=dst)
        result = constant_folding(_function((instr,)), _sym("tmp", Char()))
        # 300 & 0xFF = 44
        assert result.body == (tac.Copy(src=tac.Constant(ast.ConstChar(44)), dst=dst),)

    def test_missing_symbol_table_not_folded(self) -> None:
        """``Truncate`` without a symbol table is not folded."""
        instr = tac.Truncate(src=_cl(5), dst=_var("tmp"))
        fn = _function((instr,))
        assert constant_folding(fn) is fn


class TestFoldIntToDouble:
    """Folding of ``IntToDouble`` — result is always ``ConstDouble``."""

    def test_int_to_double(self) -> None:
        """``IntToDouble(ConstInt(5), dst)`` → ``Copy(ConstDouble(5.0), dst)``."""
        dst = _var("tmp")
        instr = tac.IntToDouble(src=_ci(5), dst=dst)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_cd(5.0), dst=dst),)

    def test_long_to_double(self) -> None:
        """``IntToDouble(ConstLong(-3), dst)`` → ``Copy(ConstDouble(-3.0), dst)``."""
        dst = _var("tmp")
        instr = tac.IntToDouble(src=_cl(-3), dst=dst)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_cd(-3.0), dst=dst),)

    def test_char_to_double(self) -> None:
        """``IntToDouble(ConstChar(65), dst)`` → ``Copy(ConstDouble(65.0), dst)``."""
        dst = _var("tmp")
        instr = tac.IntToDouble(src=tac.Constant(ast.ConstChar(65)), dst=dst)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_cd(65.0), dst=dst),)

    def test_no_symbol_table_needed(self) -> None:
        """``IntToDouble`` folds without a symbol table (result type is always double)."""
        dst = _var("tmp")
        instr = tac.IntToDouble(src=_ci(7), dst=dst)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_cd(7.0), dst=dst),)

    def test_variable_src_not_folded(self) -> None:
        """``IntToDouble`` with a variable source is not folded."""
        instr = tac.IntToDouble(src=_var("x"), dst=_var("tmp"))
        fn = _function((instr,))
        assert constant_folding(fn) is fn


class TestFoldUIntToDouble:
    """Folding of ``UIntToDouble`` — result is always ``ConstDouble``."""

    def test_uint_to_double(self) -> None:
        """``UIntToDouble(ConstUInt(0xFFFFFFFF), dst)`` converts to the exact double."""
        dst = _var("tmp")
        instr = tac.UIntToDouble(src=_cui(0xFFFF_FFFF), dst=dst)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_cd(float(0xFFFF_FFFF)), dst=dst),)

    def test_ulong_to_double(self) -> None:
        """``UIntToDouble(ConstULong(2), dst)`` → ``Copy(ConstDouble(2.0), dst)``."""
        dst = _var("tmp")
        instr = tac.UIntToDouble(src=_cul(2), dst=dst)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_cd(2.0), dst=dst),)

    def test_no_symbol_table_needed(self) -> None:
        """``UIntToDouble`` folds without a symbol table."""
        dst = _var("tmp")
        instr = tac.UIntToDouble(src=_cui(10), dst=dst)
        result = constant_folding(_function((instr,)))
        assert result.body == (tac.Copy(src=_cd(10.0), dst=dst),)


class TestFoldDoubleToInt:
    """Folding of ``DoubleToInt`` — requires symbol table to determine destination type."""

    def test_double_to_int_truncates_toward_zero(self) -> None:
        """``DoubleToInt(2.9, dst:int)`` truncates toward zero → ``ConstInt(2)``."""
        dst = _var("tmp")
        instr = tac.DoubleToInt(src=_cd(2.9), dst=dst)
        result = constant_folding(_function((instr,)), _sym("tmp", IntType()))
        assert result.body == (tac.Copy(src=_ci(2), dst=dst),)

    def test_double_to_int_negative_truncates_toward_zero(self) -> None:
        """``DoubleToInt(-2.9, dst:int)`` truncates toward zero → ``ConstInt(-2)``."""
        dst = _var("tmp")
        instr = tac.DoubleToInt(src=_cd(-2.9), dst=dst)
        result = constant_folding(_function((instr,)), _sym("tmp", IntType()))
        assert result.body == (tac.Copy(src=_ci(-2), dst=dst),)

    def test_double_to_long(self) -> None:
        """``DoubleToInt(3.0e9, dst:long)`` → ``ConstLong(3000000000)``."""
        dst = _var("tmp")
        instr = tac.DoubleToInt(src=_cd(3.0e9), dst=dst)
        result = constant_folding(_function((instr,)), _sym("tmp", LongType()))
        assert result.body == (tac.Copy(src=_cl(3_000_000_000), dst=dst),)

    def test_double_to_schar(self) -> None:
        """``DoubleToInt(65.9, dst:signed char)`` → ``ConstChar(65)``."""
        dst = _var("tmp")
        instr = tac.DoubleToInt(src=_cd(65.9), dst=dst)
        result = constant_folding(_function((instr,)), _sym("tmp", SChar()))
        assert result.body == (tac.Copy(src=tac.Constant(ast.ConstChar(65)), dst=dst),)

    def test_inf_not_folded(self) -> None:
        """``DoubleToInt(inf, dst)`` is not folded — non-finite source."""
        import math

        instr = tac.DoubleToInt(src=_cd(math.inf), dst=_var("tmp"))
        fn = _function((instr,))
        assert constant_folding(fn, _sym("tmp", IntType())) is fn

    def test_nan_not_folded(self) -> None:
        """``DoubleToInt(nan, dst)`` is not folded — non-finite source."""
        import math

        instr = tac.DoubleToInt(src=_cd(math.nan), dst=_var("tmp"))
        fn = _function((instr,))
        assert constant_folding(fn, _sym("tmp", IntType())) is fn

    def test_missing_symbol_table_not_folded(self) -> None:
        """``DoubleToInt`` without a symbol table is not folded."""
        instr = tac.DoubleToInt(src=_cd(1.5), dst=_var("tmp"))
        fn = _function((instr,))
        assert constant_folding(fn) is fn

    def test_variable_src_not_folded(self) -> None:
        """``DoubleToInt`` with a variable source is not folded."""
        instr = tac.DoubleToInt(src=_var("x"), dst=_var("tmp"))
        fn = _function((instr,))
        assert constant_folding(fn, _sym("tmp", IntType())) is fn


class TestFoldDoubleToUInt:
    """Folding of ``DoubleToUInt`` — requires symbol table to determine destination type."""

    def test_double_to_uint_truncates_toward_zero(self) -> None:
        """``DoubleToUInt(3.9, dst:uint)`` truncates toward zero → ``ConstUInt(3)``."""
        dst = _var("tmp")
        instr = tac.DoubleToUInt(src=_cd(3.9), dst=dst)
        result = constant_folding(_function((instr,)), _sym("tmp", UIntType()))
        assert result.body == (tac.Copy(src=_cui(3), dst=dst),)

    def test_double_to_ulong(self) -> None:
        """``DoubleToUInt(4.0e18, dst:ulong)`` produces the expected ConstULong."""
        dst = _var("tmp")
        val = 4_000_000_000_000_000_000
        instr = tac.DoubleToUInt(src=_cd(float(val)), dst=dst)
        result = constant_folding(_function((instr,)), _sym("tmp", ULongType()))
        assert result.body == (tac.Copy(src=_cul(val), dst=dst),)

    def test_double_to_uchar(self) -> None:
        """``DoubleToUInt(200.9, dst:uchar)`` → ``ConstUChar(200)``."""
        dst = _var("tmp")
        instr = tac.DoubleToUInt(src=_cd(200.9), dst=dst)
        result = constant_folding(_function((instr,)), _sym("tmp", UChar()))
        assert result.body == (tac.Copy(src=tac.Constant(ast.ConstUChar(200)), dst=dst),)

    def test_inf_not_folded(self) -> None:
        """``DoubleToUInt(inf, dst)`` is not folded — non-finite source."""
        import math

        instr = tac.DoubleToUInt(src=_cd(math.inf), dst=_var("tmp"))
        fn = _function((instr,))
        assert constant_folding(fn, _sym("tmp", UIntType())) is fn

    def test_missing_symbol_table_not_folded(self) -> None:
        """``DoubleToUInt`` without a symbol table is not folded."""
        instr = tac.DoubleToUInt(src=_cd(1.0), dst=_var("tmp"))
        fn = _function((instr,))
        assert constant_folding(fn) is fn
