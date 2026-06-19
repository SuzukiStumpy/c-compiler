"""Tests for the code generation stage.

Verifies that ``generate`` correctly translates each TAC construct into the
equivalent assembly-AST nodes, and that the pseudo-replacement and instruction
fixup passes produce valid, fully resolved assembly.
"""

import pytest

import compiler.syntax.asm as asm
import compiler.syntax.tac as tac
from compiler.code_generator import (
    _asm_type,
    _is_unsigned,
    _reg_width,
    _static_alignment,
    _translate_static_init,
    _type_size,
    generate,
)
from compiler.syntax.ast.constants import (
    ConstChar,
    ConstDouble,
    ConstFloat,
    ConstInt,
    ConstLong,
    ConstLongDouble,
    ConstLongLong,
    ConstShort,
    ConstUChar,
    ConstULongLong,
    ConstUShort,
)
from compiler.syntax.ast.types import (
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
    UChar,
    UIntType,
    ULongLongType,
    ULongType,
    Union,
    UShortType,
)
from compiler.types.symbols import FunAttr, LocalAttr, MemberEntry, StructEntry, SymbolTable, SymbolTableEntry
from compiler.utils.exceptions import CodeGenerationError


def _simple_tac_program(value: int = 0, name: str = "main") -> tac.Program:
    """Build a minimal ``tac.Program`` whose sole function returns a constant.

    Args:
        value: The integer constant returned by the function.
        name: The function name.

    Returns:
        A ``tac.Program`` containing a single function that returns ``value``.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name=name,
                body=(tac.Return(val=tac.Constant(value=ConstInt(value))),),
            ),
        )
    )


# ---------------------------------------------------------------------------
# Program and function translation
# ---------------------------------------------------------------------------


def test_generates_asm_program() -> None:
    """``generate`` returns an ``asm.Program`` node."""
    result = generate(_simple_tac_program())
    assert isinstance(result, asm.Program)


def test_function_name_is_preserved() -> None:
    """The assembly function carries the same name as the source function."""
    result = generate(_simple_tac_program(name="foo"))
    assert result.definitions[0].name == "foo"


def test_single_return_produces_two_instructions() -> None:
    """``return <n>;`` with no temporaries translates to exactly ``Mov`` + ``Ret``."""
    result = generate(_simple_tac_program(value=0))
    assert len(result.definitions[0].instructions) == 2


# ---------------------------------------------------------------------------
# Return statement translation
# ---------------------------------------------------------------------------


def test_return_constant_produces_mov_imm_to_register() -> None:
    """``Return(Constant(42))`` produces ``Mov(src=Imm(42), dst=Register())``."""
    result = generate(_simple_tac_program(value=42))
    mov = result.definitions[0].instructions[0]
    assert mov == asm.Mov(src=asm.Imm(value=42), dst=asm.Register(asm.GPRegisters.AX))


def test_return_produces_ret_as_second_instruction() -> None:
    """The second instruction for a return statement is always ``Ret()``."""
    result = generate(_simple_tac_program(value=0))
    assert result.definitions[0].instructions[1] == asm.Ret()


def test_return_value_is_encoded_in_imm() -> None:
    """The immediate operand value matches the returned integer constant."""
    result = generate(_simple_tac_program(value=99))
    mov = result.definitions[0].instructions[0]
    assert isinstance(mov, asm.Mov)
    assert isinstance(mov.src, asm.Imm)
    assert mov.src.value == 99


def test_mov_destination_is_register() -> None:
    """The destination operand of the ``Mov`` is always ``Register()``."""
    result = generate(_simple_tac_program(value=0))
    mov = result.definitions[0].instructions[0]
    assert isinstance(mov, asm.Mov)
    assert mov.dst == asm.Register(asm.GPRegisters.AX)


# ---------------------------------------------------------------------------
# Multiple return statements
# ---------------------------------------------------------------------------


def test_multiple_return_statements_produce_instruction_pairs() -> None:
    """Each ``Return`` contributes a ``Mov``+``Ret`` pair; no stack allocation."""
    program = tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.Return(val=tac.Constant(value=ConstInt(1))),
                    tac.Return(val=tac.Constant(value=ConstInt(2))),
                ),
            ),
        )
    )
    result = generate(program)
    assert len(result.definitions[0].instructions) == 4
    assert result.definitions[0].instructions[0] == asm.Mov(src=asm.Imm(value=1), dst=asm.Register(asm.GPRegisters.AX))
    assert result.definitions[0].instructions[1] == asm.Ret()
    assert result.definitions[0].instructions[2] == asm.Mov(src=asm.Imm(value=2), dst=asm.Register(asm.GPRegisters.AX))
    assert result.definitions[0].instructions[3] == asm.Ret()


# ---------------------------------------------------------------------------
# Unary operations and stack allocation
# ---------------------------------------------------------------------------


def test_unary_negate_produces_allocate_mov_unary_mov_ret() -> None:
    """``Unary(Negate, Constant(5), Var('t'))`` + ``Return(Var('t'))`` produces the expected sequence."""
    program = tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.Unary(
                        operator=tac.Negate(),
                        src=tac.Constant(value=ConstInt(5)),
                        dst=tac.Var(value="tmp.0"),
                    ),
                    tac.Return(val=tac.Var(value="tmp.0")),
                ),
            ),
        )
    )
    result = generate(program)
    instrs = result.definitions[0].instructions
    # AllocateStack, Mov(Imm(5), Stack(-4)), Unary(Neg, Stack(-4)), Mov(Stack(-4), Register()), Ret()
    assert isinstance(instrs[0], asm.AllocateStack)
    assert instrs[1] == asm.Mov(
        src=asm.Imm(value=5), dst=asm.Memory(reg=asm.Register(asm.GPRegisters.BP, asm.Width.QWORD), offset=-4)
    )
    assert instrs[2] == asm.Unary(
        operator=asm.Neg(), operand=asm.Memory(reg=asm.Register(asm.GPRegisters.BP, asm.Width.QWORD), offset=-4)
    )
    assert instrs[3] == asm.Mov(
        src=asm.Memory(reg=asm.Register(asm.GPRegisters.BP, asm.Width.QWORD), offset=-4),
        dst=asm.Register(asm.GPRegisters.AX),
    )
    assert instrs[4] == asm.Ret()


def test_unary_complement_uses_not_operator() -> None:
    """``Unary(Complement, ...)`` translates to an ``asm.Unary`` with ``Not()``."""
    program = tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.Unary(
                        operator=tac.Complement(),
                        src=tac.Constant(value=ConstInt(1)),
                        dst=tac.Var(value="tmp.0"),
                    ),
                    tac.Return(val=tac.Var(value="tmp.0")),
                ),
            ),
        )
    )
    result = generate(program)
    unary_instr = result.definitions[0].instructions[2]
    assert isinstance(unary_instr, asm.Unary)
    assert unary_instr.operator == asm.Not()


def test_stack_allocation_is_multiple_of_16() -> None:
    """``AllocateStack`` size is always a multiple of 16 bytes."""
    program = tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.Unary(
                        operator=tac.Negate(),
                        src=tac.Constant(value=ConstInt(1)),
                        dst=tac.Var(value="tmp.0"),
                    ),
                    tac.Return(val=tac.Var(value="tmp.0")),
                ),
            ),
        )
    )
    result = generate(program)
    alloc = result.definitions[0].instructions[0]
    assert isinstance(alloc, asm.AllocateStack)
    assert alloc.size % 16 == 0


def test_no_pseudo_operands_in_output() -> None:
    """No ``Pseudo`` operands remain after code generation."""
    program = tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.Unary(
                        operator=tac.Negate(),
                        src=tac.Constant(value=ConstInt(3)),
                        dst=tac.Var(value="tmp.0"),
                    ),
                    tac.Return(val=tac.Var(value="tmp.0")),
                ),
            ),
        )
    )
    result = generate(program)
    for instr in result.definitions[0].instructions:
        match instr:
            case asm.Mov(src=src, dst=dst):
                assert not isinstance(src, asm.Pseudo)
                assert not isinstance(dst, asm.Pseudo)
            case asm.Unary(operand=operand):
                assert not isinstance(operand, asm.Pseudo)


# ---------------------------------------------------------------------------
# Instruction fixup - Mov(mem, mem)
# ---------------------------------------------------------------------------


def test_chained_temps_produce_no_mem_mem_mov() -> None:
    """Two chained unary ops produce no ``Mov(Stack, Stack)`` instructions."""
    program = tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.Unary(
                        operator=tac.Complement(),
                        src=tac.Constant(value=ConstInt(2)),
                        dst=tac.Var(value="tmp.0"),
                    ),
                    tac.Unary(
                        operator=tac.Negate(),
                        src=tac.Var(value="tmp.0"),
                        dst=tac.Var(value="tmp.1"),
                    ),
                    tac.Return(val=tac.Var(value="tmp.1")),
                ),
            ),
        )
    )
    result = generate(program)
    for instr in result.definitions[0].instructions:
        if isinstance(instr, asm.Mov):
            assert not (isinstance(instr.src, asm.Memory) and isinstance(instr.dst, asm.Memory)), (
                f"Invalid Mov(mem, mem) found: {instr}"
            )


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def _binary_tac_program(op: tac.BinaryOperator, src1: tac.Val, src2: tac.Val) -> tac.Program:
    """Build a ``tac.Program`` that evaluates ``src1 <op> src2`` and returns the result.

    Args:
        op: The binary operator to apply.
        src1: The left operand.
        src2: The right operand.

    Returns:
        A ``tac.Program`` whose sole function returns the result of the binary expression.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.Binary(operator=op, src1=src1, src2=src2, dst=tac.Var(value="tmp.0")),
                    tac.Return(val=tac.Var(value="tmp.0")),
                ),
            ),
        )
    )


# ---------------------------------------------------------------------------
# Binary operations - Add, Sub, Mul
# ---------------------------------------------------------------------------


def test_binary_add_src1_loaded_into_dst_slot() -> None:
    """``Binary(Add, ...)`` first copies src1 into the result slot via an explicit ``Mov``."""
    result = generate(_binary_tac_program(tac.Add(), tac.Constant(ConstInt(1)), tac.Constant(ConstInt(2))))
    # AllocateStack, Mov(src1, dst), Binary(Add, dst, src2, dst), Mov(dst, AX), Ret
    mov = result.definitions[0].instructions[1]
    assert isinstance(mov, asm.Mov)
    assert mov.src == asm.Imm(value=1)


def test_binary_add_operator_is_asm_add() -> None:
    """``tac.Add()`` maps to ``asm.Add()`` inside the ``Binary`` instruction."""
    result = generate(_binary_tac_program(tac.Add(), tac.Constant(ConstInt(1)), tac.Constant(ConstInt(2))))
    # AllocateStack, Mov(src1, dst), Binary(Add, dst, src2, dst), Mov(dst, AX), Ret
    binary = result.definitions[0].instructions[2]
    assert isinstance(binary, asm.Binary)
    assert binary.operator == asm.Add()


def test_binary_subtract_operator_is_asm_sub() -> None:
    """``tac.Subtract()`` maps to ``asm.Sub()`` inside the ``Binary`` instruction."""
    result = generate(_binary_tac_program(tac.Subtract(), tac.Constant(ConstInt(5)), tac.Constant(ConstInt(3))))
    binary = result.definitions[0].instructions[2]
    assert isinstance(binary, asm.Binary)
    assert binary.operator == asm.Sub()


def test_binary_multiply_operator_is_asm_mult() -> None:
    """``tac.Multiply()`` maps to ``asm.Mult()`` inside the ``Binary`` instruction."""
    result = generate(_binary_tac_program(tac.Multiply(), tac.Constant(ConstInt(4)), tac.Constant(ConstInt(3))))
    binary = next(i for i in result.definitions[0].instructions if isinstance(i, asm.Binary))
    assert binary.operator == asm.Mult()


def test_binary_add_binary_uses_dst_as_inplace_accumulator() -> None:
    """The ``Binary`` instruction uses ``dst`` as both src1 and the accumulator (in-place operation)."""
    result = generate(_binary_tac_program(tac.Add(), tac.Constant(ConstInt(1)), tac.Constant(ConstInt(2))))
    # AllocateStack, Mov(src1, dst), Binary(Add, dst, src2, dst), Mov(dst, AX), Ret
    binary = result.definitions[0].instructions[2]
    assert isinstance(binary, asm.Binary)
    assert binary.src1 == binary.dst


def test_binary_add_instruction_count() -> None:
    """``Binary(Add, c, c)`` + ``Return`` produces exactly 5 instructions."""
    # AllocateStack, Mov(src1, dst), Binary(Add, dst, src2, dst), Mov(dst, AX), Ret
    result = generate(_binary_tac_program(tac.Add(), tac.Constant(ConstInt(1)), tac.Constant(ConstInt(2))))
    assert len(result.definitions[0].instructions) == 5


# ---------------------------------------------------------------------------
# Division
# ---------------------------------------------------------------------------


def test_divide_loads_dividend_into_ax() -> None:
    """``Binary(Divide, ...)`` moves the dividend into ``%eax`` as the first instruction."""
    result = generate(_binary_tac_program(tac.Divide(), tac.Constant(ConstInt(10)), tac.Constant(ConstInt(2))))
    # AllocateStack, Mov(Imm(10), AX), Cdq, Mov(Imm(2), R11), Idiv(R11), Mov(AX, Stack), Mov(Stack, AX), Ret
    assert result.definitions[0].instructions[1] == asm.Mov(src=asm.Imm(value=10), dst=asm.Register(asm.GPRegisters.AX))


def test_divide_emits_cdq_after_dividend_load() -> None:
    """``Cdq`` follows the dividend ``Mov`` to sign-extend ``%eax`` into ``%edx:%eax``."""
    result = generate(_binary_tac_program(tac.Divide(), tac.Constant(ConstInt(10)), tac.Constant(ConstInt(2))))
    assert isinstance(result.definitions[0].instructions[2], asm.Cdq)


def test_divide_result_comes_from_ax() -> None:
    """The quotient ``Mov`` sources from ``%eax``."""
    result = generate(_binary_tac_program(tac.Divide(), tac.Constant(ConstInt(10)), tac.Constant(ConstInt(2))))
    # After fixup: AllocateStack, Mov(10, AX), Cdq, Mov(2, R11), Idiv(R11), Mov(AX, Stack), Mov(Stack, AX), Ret
    quotient_mov = result.definitions[0].instructions[5]
    assert isinstance(quotient_mov, asm.Mov)
    assert quotient_mov.src == asm.Register(asm.GPRegisters.AX)


# ---------------------------------------------------------------------------
# Remainder
# ---------------------------------------------------------------------------


def test_remainder_result_comes_from_dx() -> None:
    """The remainder ``Mov`` sources from ``%edx``, not ``%eax``."""
    result = generate(_binary_tac_program(tac.Remainder(), tac.Constant(ConstInt(10)), tac.Constant(ConstInt(3))))
    # AllocateStack, Mov(10, AX), Cdq, Mov(3, R11), Idiv(R11), Mov(DX, Stack), Mov(Stack, AX), Ret
    remainder_mov = result.definitions[0].instructions[5]
    assert isinstance(remainder_mov, asm.Mov)
    assert remainder_mov.src == asm.Register(asm.GPRegisters.DX)


def test_remainder_emits_cdq() -> None:
    """``Cdq`` is emitted for remainder just as for divide."""
    result = generate(_binary_tac_program(tac.Remainder(), tac.Constant(ConstInt(7)), tac.Constant(ConstInt(3))))
    assert isinstance(result.definitions[0].instructions[2], asm.Cdq)


# ---------------------------------------------------------------------------
# Instruction fixup - Idiv with immediate divisor
# ---------------------------------------------------------------------------


def test_idiv_immediate_divisor_routed_through_r11() -> None:
    """An immediate divisor is moved into ``%r11d`` before ``idivl``."""
    result = generate(_binary_tac_program(tac.Divide(), tac.Constant(ConstInt(10)), tac.Constant(ConstInt(2))))
    # AllocateStack, Mov(10, AX), Cdq, Mov(2, R11), Idiv(R11), ...
    mov_r11 = result.definitions[0].instructions[3]
    idiv = result.definitions[0].instructions[4]
    assert mov_r11 == asm.Mov(src=asm.Imm(value=2), dst=asm.Register(asm.GPRegisters.R11))
    assert idiv == asm.Idiv(operand=asm.Register(asm.GPRegisters.R11))


def test_idiv_memory_divisor_not_routed_through_r11() -> None:
    """A stack-slot divisor is passed directly to ``idivl`` without an extra ``Mov``."""
    program = tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.Binary(
                        operator=tac.Divide(),
                        src1=tac.Constant(ConstInt(10)),
                        src2=tac.Var("b"),
                        dst=tac.Var("tmp.0"),
                    ),
                    tac.Return(val=tac.Var("tmp.0")),
                ),
            ),
        )
    )
    result = generate(program)
    # Find the Idiv instruction and verify its operand is a Stack slot, not a register
    idiv_instrs = [i for i in result.definitions[0].instructions if isinstance(i, asm.Idiv)]
    assert len(idiv_instrs) == 1
    assert isinstance(idiv_instrs[0].operand, asm.Memory)


# ---------------------------------------------------------------------------
# Pseudo replacement for binary instructions
# ---------------------------------------------------------------------------


def test_no_pseudo_operands_after_binary_code_gen() -> None:
    """No ``Pseudo`` operands survive into the final binary instruction output."""
    result = generate(_binary_tac_program(tac.Add(), tac.Constant(ConstInt(1)), tac.Constant(ConstInt(2))))
    for instr in result.definitions[0].instructions:
        match instr:
            case asm.Binary(src1=src1, src2=src2):
                assert not isinstance(src1, asm.Pseudo)
                assert not isinstance(src2, asm.Pseudo)
            case asm.Idiv(operand=operand):
                assert not isinstance(operand, asm.Pseudo)


# ---------------------------------------------------------------------------
# Bitwise AND / OR / XOR code generation
# ---------------------------------------------------------------------------


class TestBitwiseAndOrXorCodeGen:
    """``tac.Binary`` with And/Or/Xor operators lowers to ``asm.Binary`` using in-place accumulation."""

    def _instructions(self, op: tac.BinaryOperator) -> tuple[asm.Instruction, ...]:
        """Generate instructions for ``return 3 <op> 5;`` with two constants."""
        return (
            generate(_binary_tac_program(op, tac.Constant(ConstInt(3)), tac.Constant(ConstInt(5))))
            .definitions[0]
            .instructions
        )

    @pytest.mark.parametrize(
        ("tac_op", "asm_op_type"),
        [
            (tac.BitwiseAnd(), asm.And),
            (tac.BitwiseOr(), asm.Or),
            (tac.BitwiseXor(), asm.Xor),
        ],
        ids=["and", "or", "xor"],
    )
    def test_operator_and_inplace_accumulation(self, tac_op: tac.BinaryOperator, asm_op_type: type) -> None:
        """Each bitwise TAC op lowers to ``asm.Binary`` with the correct operator and src1 == dst."""
        binary = next(i for i in self._instructions(tac_op) if isinstance(i, asm.Binary))
        assert isinstance(binary.operator, asm_op_type)
        assert binary.src1 == binary.dst

    @pytest.mark.parametrize(
        ("op", "src1", "src2", "expected"),
        [
            (asm.And(), 3, 5, "      movl  $3, %r11d\n      andl  $5, %r11d"),
            (asm.Or(), 4, 2, "      movl  $4, %r11d\n      orl   $2, %r11d"),
            (asm.Xor(), 6, 3, "      movl  $6, %r11d\n      xorl  $3, %r11d"),
        ],
        ids=["and", "or", "xor"],
    )
    def test_emit(self, op: asm.BinaryOperator, src1: int, src2: int, expected: str) -> None:
        """``asm.Binary`` with each bitwise operator emits the correct two-line sequence."""
        instr = asm.Binary(
            operator=op,
            src1=asm.Imm(src1),
            src2=asm.Imm(src2),
            dst=asm.Register(asm.GPRegisters.R11),
        )
        assert instr.emit() == expected


# ---------------------------------------------------------------------------
# Shift code generation
# ---------------------------------------------------------------------------


def _shift_tac_program(op: tac.BinaryOperator, value: int, count: int) -> tac.Program:
    """Build a ``tac.Program`` that shifts ``value`` by ``count`` using ``op``."""
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.Binary(
                        operator=op,
                        src1=tac.Constant(ConstInt(value)),
                        src2=tac.Constant(ConstInt(count)),
                        dst=tac.Var(value="tmp.0"),
                    ),
                    tac.Return(val=tac.Var(value="tmp.0")),
                ),
            ),
        )
    )


class TestShiftCodeGen:
    """``tac.LeftShift`` / ``tac.RightShift`` lower to a 4-instruction sequence via ``%ecx``."""

    def _instructions(self, op: tac.BinaryOperator) -> tuple[asm.Instruction, ...]:
        return generate(_shift_tac_program(op, value=8, count=2)).definitions[0].instructions

    @pytest.mark.parametrize(
        ("tac_op", "instr_type"),
        [
            (tac.LeftShift(), asm.Shl),
            (tac.RightShift(), asm.Sar),
        ],
        ids=["left", "right"],
    )
    def test_shift_emits_instruction_routed_through_r11(self, tac_op: tac.BinaryOperator, instr_type: type) -> None:
        """Each shift direction emits the correct instruction type with ``%r11d`` as its operand."""
        instrs = self._instructions(tac_op)
        shift = next(i for i in instrs if isinstance(i, instr_type))
        assert shift.operand == asm.Register(asm.GPRegisters.R11)

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            (asm.Shl(operand=asm.Register(asm.GPRegisters.R11)), "      shll  %cl, %r11d"),
            (asm.Sar(operand=asm.Register(asm.GPRegisters.R11)), "      sarl  %cl, %r11d"),
        ],
        ids=["shl", "sar"],
    )
    def test_shift_emit(self, instr: asm.Instruction, expected: str) -> None:
        """Shift instructions emit the correct mnemonic with the implicit ``%cl`` count."""
        assert instr.emit() == expected

    def test_shift_count_moved_to_cx(self) -> None:
        """A ``Mov`` to ``%ecx`` (``CX``) appears before ``Shl`` to load the shift count."""
        instrs = self._instructions(tac.LeftShift())
        cx_movs = [i for i in instrs if isinstance(i, asm.Mov) and i.dst == asm.Register(asm.GPRegisters.CX)]
        assert len(cx_movs) == 1

    def test_shl_emit_stack_operand(self) -> None:
        """``asm.Shl(Stack(-4))`` emits ``shll %cl, -4(%rbp)`` - memory operand is valid."""
        assert (
            asm.Shl(operand=asm.Memory(reg=asm.Register(asm.GPRegisters.BP, asm.Width.QWORD), offset=-4)).emit()
            == "      shll  %cl, -4(%rbp)"
        )

    def test_no_pseudo_operands_after_left_shift_code_gen(self) -> None:
        """No ``Pseudo`` operands survive pseudo-replacement for a shift instruction."""
        result = generate(_shift_tac_program(tac.LeftShift(), 4, 1))
        for instr in result.definitions[0].instructions:
            if isinstance(instr, asm.Shl):
                assert not isinstance(instr.operand, asm.Pseudo)


def test_unsupported_instruction_raises_codegen_error() -> None:
    """An unrecognised TAC instruction subclass raises ``CodeGenerationError``."""

    class _FakeInstruction(tac.Instruction):
        pass

    program = tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(_FakeInstruction(),),
            ),
        )
    )
    with pytest.raises(CodeGenerationError):
        generate(program)


# ---------------------------------------------------------------------------
# Helpers for relational / jump / label tests
# ---------------------------------------------------------------------------


def _relational_tac_program(
    op: tac.BinaryOperator,
    src1: tac.Val | None = None,
    src2: tac.Val | None = None,
) -> tac.Program:
    """Build a ``tac.Program`` that evaluates ``src1 <op> src2`` and returns the result.

    Args:
        op: A relational binary operator (Equal, NotEqual, LessThan, etc.).
        src1: Left operand; defaults to ``Constant(1)``.
        src2: Right operand; defaults to ``Constant(2)``.

    Returns:
        A ``tac.Program`` whose sole function returns 0 or 1 based on the comparison.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.Binary(
                        operator=op,
                        src1=src1 if src1 is not None else tac.Constant(value=ConstInt(1)),
                        src2=src2 if src2 is not None else tac.Constant(value=ConstInt(2)),
                        dst=tac.Var(value="tmp.0"),
                    ),
                    tac.Return(val=tac.Var(value="tmp.0")),
                ),
            ),
        )
    )


# ---------------------------------------------------------------------------
# Relational operator code generation
# ---------------------------------------------------------------------------


class TestRelationalCodeGen:
    """Relational TAC operators lower to ``Cmp + Mov(0, dst) + SetCC``."""

    @pytest.mark.parametrize(
        ("tac_op", "expected_cc"),
        [
            (tac.Equal(), asm.CondCodes.E),
            (tac.NotEqual(), asm.CondCodes.NE),
            (tac.LessThan(), asm.CondCodes.L),
            (tac.LessOrEqual(), asm.CondCodes.LE),
            (tac.GreaterThan(), asm.CondCodes.G),
            (tac.GreaterOrEqual(), asm.CondCodes.GE),
        ],
        ids=["eq", "ne", "lt", "le", "gt", "ge"],
    )
    def test_relational_op_maps_to_correct_cond_code(
        self, tac_op: tac.BinaryOperator, expected_cc: asm.CondCodes
    ) -> None:
        """Each relational TAC operator produces a ``SetCC`` with the matching condition code."""
        result = generate(_relational_tac_program(tac_op))
        setcc = next(i for i in result.definitions[0].instructions if isinstance(i, asm.SetCC))
        assert setcc.cond_code == expected_cc

    def test_relational_emits_cmp_instruction(self) -> None:
        """A relational binary op always produces a ``Cmp`` instruction."""
        result = generate(_relational_tac_program(tac.Equal()))
        assert any(isinstance(i, asm.Cmp) for i in result.definitions[0].instructions)

    def test_relational_zeroes_destination_before_setcc(self) -> None:
        """``Mov(Imm(0), dst)`` immediately precedes ``SetCC`` in the output."""
        result = generate(_relational_tac_program(tac.Equal()))
        instrs = result.definitions[0].instructions
        setcc_idx = next(i for i, instr in enumerate(instrs) if isinstance(instr, asm.SetCC))
        zero_mov = instrs[setcc_idx - 1]
        assert isinstance(zero_mov, asm.Mov)
        assert zero_mov.src == asm.Imm(value=0)

    def test_no_pseudo_operands_in_relational_output(self) -> None:
        """No ``Pseudo`` operands survive into the final output for a relational expression."""
        result = generate(_relational_tac_program(tac.Equal(), tac.Var("a"), tac.Var("b")))
        for instr in result.definitions[0].instructions:
            match instr:
                case asm.Cmp(op1=op1, op2=op2):
                    assert not isinstance(op1, asm.Pseudo)
                    assert not isinstance(op2, asm.Pseudo)
                case asm.SetCC(operand=operand):
                    assert not isinstance(operand, asm.Pseudo)
                case asm.Mov(src=src, dst=dst):
                    assert not isinstance(src, asm.Pseudo)
                    assert not isinstance(dst, asm.Pseudo)


# ---------------------------------------------------------------------------
# Double (NaN-aware) comparison code generation
# ---------------------------------------------------------------------------


def _double_relational_tac_program(op: tac.BinaryOperator) -> tac.Program:
    """Build a ``tac.Program`` comparing two ``double`` constants with ``op``.

    Args:
        op: A relational binary operator (Equal, NotEqual, LessThan, etc.).

    Returns:
        A ``tac.Program`` whose sole function returns the comparison result.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.Binary(
                        operator=op,
                        src1=tac.Constant(value=ConstDouble(1.0)),
                        src2=tac.Constant(value=ConstDouble(2.0)),
                        dst=tac.Var(value="tmp.0"),
                    ),
                    tac.Return(val=tac.Var(value="tmp.0")),
                ),
            ),
        )
    )


def _double_fn_instrs(tac_op: tac.BinaryOperator) -> tuple[asm.Instruction, ...]:
    """Generate a double comparison program and return the main function's instructions.

    Args:
        tac_op: The relational operator to compare with.

    Returns:
        The instruction sequence from the generated main function.
    """
    result = generate(_double_relational_tac_program(tac_op))
    fn = next(d for d in result.definitions if isinstance(d, asm.Function))
    return fn.instructions


class TestDoubleRelationalCodeGen:
    """Double-precision relational operators use NaN-aware code sequences.

    ``ucomisd`` sets ZF=CF=PF=1 for unordered (NaN) results.  Operations that would
    naively mis-read those flags (``==``, ``!=``, ``<``, ``<=``) must use a conditional
    jump on the parity flag to produce the correct 0/1 result.  ``>`` and ``>=`` are
    naturally correct because ``seta``/``setae`` both return 0 when CF=1.
    """

    @pytest.mark.parametrize(
        ("tac_op", "expected_cc"),
        [
            (tac.Equal(), asm.CondCodes.E),
            (tac.NotEqual(), asm.CondCodes.NE),
            (tac.LessThan(), asm.CondCodes.B),
            (tac.LessOrEqual(), asm.CondCodes.BE),
            (tac.GreaterThan(), asm.CondCodes.A),
            (tac.GreaterOrEqual(), asm.CondCodes.AE),
        ],
        ids=["eq", "ne", "lt", "le", "gt", "ge"],
    )
    def test_double_relational_uses_unsigned_cond_codes(
        self, tac_op: tac.BinaryOperator, expected_cc: asm.CondCodes
    ) -> None:
        """Double comparisons use above/below (unsigned) condition codes, not signed ones."""
        instrs = _double_fn_instrs(tac_op)
        setcc = next(i for i in instrs if isinstance(i, asm.SetCC))
        assert setcc.cond_code == expected_cc

    @pytest.mark.parametrize(
        "tac_op",
        [tac.Equal(), tac.NotEqual(), tac.LessThan(), tac.LessOrEqual()],
        ids=["eq", "ne", "lt", "le"],
    )
    def test_double_relational_emits_parity_jump(self, tac_op: tac.BinaryOperator) -> None:
        """``==``, ``!=``, ``<``, ``<=`` emit a ``JmpCC(P, ...)`` to guard against NaN."""
        instrs = _double_fn_instrs(tac_op)
        assert any(isinstance(i, asm.JmpCC) and i.cond_code == asm.CondCodes.P for i in instrs)

    @pytest.mark.parametrize(
        "tac_op",
        [tac.GreaterThan(), tac.GreaterOrEqual()],
        ids=["gt", "ge"],
    )
    def test_double_relational_gt_ge_no_parity_jump(self, tac_op: tac.BinaryOperator) -> None:
        """``>`` and ``>=`` do not need a parity-flag guard; no ``JmpCC(P)`` is emitted."""
        instrs = _double_fn_instrs(tac_op)
        assert not any(isinstance(i, asm.JmpCC) and i.cond_code == asm.CondCodes.P for i in instrs)

    def test_double_not_equal_preloads_one(self) -> None:
        """``!=`` pre-loads 1 into dst (so NaN returns 1 when the parity jump fires)."""
        instrs = _double_fn_instrs(tac.NotEqual())
        jmpcc_idx = next(i for i, instr in enumerate(instrs) if isinstance(instr, asm.JmpCC))
        preload = instrs[jmpcc_idx - 1]
        assert isinstance(preload, asm.Mov)
        assert preload.src == asm.Imm(1)

    def test_double_equal_preloads_zero(self) -> None:
        """``==`` pre-loads 0 into dst (so NaN returns 0 when the parity jump fires)."""
        instrs = _double_fn_instrs(tac.Equal())
        jmpcc_idx = next(i for i, instr in enumerate(instrs) if isinstance(instr, asm.JmpCC))
        preload = instrs[jmpcc_idx - 1]
        assert isinstance(preload, asm.Mov)
        assert preload.src == asm.Imm(0)

    def test_double_relational_emits_ucomisd(self) -> None:
        """Double comparisons emit a ``Cmp`` (``ucomisd``) instruction."""
        instrs = _double_fn_instrs(tac.Equal())
        assert any(isinstance(i, asm.Cmp) and i.assembly_type == asm.Double() for i in instrs)


# ---------------------------------------------------------------------------
# Float (NaN-aware) comparison and binary code generation
# ---------------------------------------------------------------------------


def _float_relational_tac_program(op: tac.BinaryOperator) -> tac.Program:
    """Build a ``tac.Program`` comparing two ``float`` constants with ``op``.

    Args:
        op: A relational binary operator (Equal, NotEqual, LessThan, etc.).

    Returns:
        A ``tac.Program`` whose sole function returns the comparison result.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.Binary(
                        operator=op,
                        src1=tac.Constant(value=ConstFloat(1.0)),
                        src2=tac.Constant(value=ConstFloat(2.0)),
                        dst=tac.Var(value="tmp.0"),
                    ),
                    tac.Return(val=tac.Var(value="tmp.0")),
                ),
            ),
        )
    )


def _float_fn_instrs(tac_op: tac.BinaryOperator) -> tuple[asm.Instruction, ...]:
    """Generate a float comparison program and return the main function's instructions.

    Args:
        tac_op: The relational operator to compare with.

    Returns:
        The instruction sequence from the generated main function.
    """
    result = generate(_float_relational_tac_program(tac_op))
    fn = next(d for d in result.definitions if isinstance(d, asm.Function))
    return fn.instructions


class TestFloatRelationalCodeGen:
    """Float-precision relational operators use NaN-aware code sequences with ``ucomiss``."""

    @pytest.mark.parametrize(
        ("tac_op", "expected_cc"),
        [
            (tac.Equal(), asm.CondCodes.E),
            (tac.NotEqual(), asm.CondCodes.NE),
            (tac.LessThan(), asm.CondCodes.B),
            (tac.LessOrEqual(), asm.CondCodes.BE),
            (tac.GreaterThan(), asm.CondCodes.A),
            (tac.GreaterOrEqual(), asm.CondCodes.AE),
        ],
        ids=["eq", "ne", "lt", "le", "gt", "ge"],
    )
    def test_float_relational_uses_unsigned_cond_codes(
        self, tac_op: tac.BinaryOperator, expected_cc: asm.CondCodes
    ) -> None:
        """Float comparisons use above/below (unsigned) condition codes."""
        instrs = _float_fn_instrs(tac_op)
        setcc = next(i for i in instrs if isinstance(i, asm.SetCC))
        assert setcc.cond_code == expected_cc

    @pytest.mark.parametrize(
        "tac_op",
        [tac.Equal(), tac.NotEqual(), tac.LessThan(), tac.LessOrEqual()],
        ids=["eq", "ne", "lt", "le"],
    )
    def test_float_relational_emits_parity_jump(self, tac_op: tac.BinaryOperator) -> None:
        """``==``, ``!=``, ``<``, ``<=`` emit a ``JmpCC(P, ...)`` to guard against NaN."""
        instrs = _float_fn_instrs(tac_op)
        assert any(isinstance(i, asm.JmpCC) and i.cond_code == asm.CondCodes.P for i in instrs)

    def test_float_relational_emits_ucomiss(self) -> None:
        """Float comparisons emit a ``Cmp`` with ``assembly_type=Float()`` (``ucomiss``)."""
        instrs = _float_fn_instrs(tac.Equal())
        assert any(isinstance(i, asm.Cmp) and i.assembly_type == asm.Float() for i in instrs)


def _float_binary_tac_program(op: tac.BinaryOperator) -> tac.Program:
    """Build a ``tac.Program`` applying a float arithmetic operation.

    Args:
        op: An arithmetic binary operator (Add, Subtract, Multiply, or Divide).

    Returns:
        A ``tac.Program`` whose sole function applies ``op`` to two float constants.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.Binary(
                        operator=op,
                        src1=tac.Constant(value=ConstFloat(1.0)),
                        src2=tac.Constant(value=ConstFloat(2.0)),
                        dst=tac.Var(value="tmp.0"),
                    ),
                    tac.Return(val=tac.Var(value="tmp.0")),
                ),
            ),
        )
    )


class TestFloatBinaryCodeGen:
    """Float arithmetic uses ``ss`` SSE instructions (``addss``, ``subss``, ``mulss``, ``divss``)."""

    @pytest.mark.parametrize(
        ("tac_op", "expected_op"),
        [
            (tac.Add(), asm.AddFloat()),
            (tac.Subtract(), asm.SubFloat()),
            (tac.Multiply(), asm.MultFloat()),
        ],
        ids=["add", "sub", "mul"],
    )
    def test_float_arithmetic_emits_ss_binary(
        self, tac_op: tac.BinaryOperator, expected_op: asm.BinaryOperator
    ) -> None:
        """Float add/sub/mul lower to the corresponding ``ss`` SSE binary instruction."""
        st = _make_symbol_table(**{"tmp.0": FloatType})
        instrs = _fn_instructions_with_table(_float_binary_tac_program(tac_op), st)
        assert any(
            isinstance(i, asm.Binary) and i.operator == expected_op and i.assembly_type == asm.Float() for i in instrs
        )

    def test_float_divide_emits_divss(self) -> None:
        """Float division lowers to ``divss``."""
        st = _make_symbol_table(**{"tmp.0": FloatType})
        instrs = _fn_instructions_with_table(_float_binary_tac_program(tac.Divide()), st)
        assert any(
            isinstance(i, asm.Binary) and i.operator == asm.DivFloat() and i.assembly_type == asm.Float()
            for i in instrs
        )


def _float_negate_tac_program() -> tac.Program:
    """Build a ``tac.Program`` that negates a float constant.

    Returns:
        A ``tac.Program`` whose sole function negates a float value.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.Unary(operator=tac.Negate(), src=tac.Constant(value=ConstFloat(1.0)), dst=tac.Var("f")),
                    tac.Return(val=tac.Var("f")),
                ),
            ),
        )
    )


class TestFloatNegateCodeGen:
    """Float negation uses ``xorps`` with a ``-0.0f`` sign mask."""

    def test_float_negate_emits_xorps(self) -> None:
        """``Negate`` on a float value emits an ``xorps`` (``XorFloat``) instruction."""
        st = _make_symbol_table(f=FloatType)
        instrs = _fn_instructions_with_table(_float_negate_tac_program(), st)
        assert any(isinstance(i, asm.Binary) and i.operator == asm.XorFloat() for i in instrs)


# ---------------------------------------------------------------------------
# Jump, label, and copy code generation
# ---------------------------------------------------------------------------


class TestJumpAndLabelCodeGen:
    """``tac.Jump``, ``tac.JumpIfZero``, ``tac.JumpIfNotZero``, ``tac.Label``, and ``tac.Copy`` lowering."""

    def _program(self, *body: tac.Instruction) -> tac.Program:
        return tac.Program(top_level=(tac.Function(name="main", body=body),))

    def test_jump_produces_jmp(self) -> None:
        """``tac.Jump(target)`` lowers to ``asm.Jmp(label)``."""
        result = generate(self._program(tac.Jump(target="end"), tac.Return(val=tac.Constant(value=ConstInt(0)))))
        assert any(isinstance(i, asm.Jmp) and i.label == "end" for i in result.definitions[0].instructions)

    def test_label_produces_asm_label(self) -> None:
        """``tac.Label(name)`` lowers to ``asm.Label(name)``."""
        result = generate(self._program(tac.Label(name="loop"), tac.Return(val=tac.Constant(value=ConstInt(0)))))
        assert any(isinstance(i, asm.Label) and i.name == "loop" for i in result.definitions[0].instructions)

    def test_copy_produces_mov(self) -> None:
        """``tac.Copy(src, dst)`` lowers to a single ``asm.Mov``."""
        result = generate(
            self._program(
                tac.Copy(src=tac.Constant(value=ConstInt(7)), dst=tac.Var(value="x")),
                tac.Return(val=tac.Var(value="x")),
            )
        )
        movs = [i for i in result.definitions[0].instructions if isinstance(i, asm.Mov)]
        assert any(isinstance(m.src, asm.Imm) and m.src.value == 7 for m in movs)

    def test_jump_if_zero_produces_cmp_and_jmpcc_e(self) -> None:
        """``tac.JumpIfZero`` lowers to ``Cmp(Imm(0), cond), JmpCC(E, target)``."""
        result = generate(
            self._program(
                tac.JumpIfZero(condition=tac.Var(value="x"), target="end"),
                tac.Return(val=tac.Constant(value=ConstInt(0))),
                tac.Label(name="end"),
                tac.Return(val=tac.Constant(value=ConstInt(0))),
            )
        )
        jmpcc_instrs = [i for i in result.definitions[0].instructions if isinstance(i, asm.JmpCC)]
        assert len(jmpcc_instrs) == 1
        assert jmpcc_instrs[0].cond_code == asm.CondCodes.E
        assert jmpcc_instrs[0].label == "end"

    def test_jump_if_not_zero_produces_jmpcc_ne(self) -> None:
        """``tac.JumpIfNotZero`` lowers to ``JmpCC(NE, target)``."""
        result = generate(
            self._program(
                tac.JumpIfNotZero(condition=tac.Var(value="x"), target="loop"),
                tac.Return(val=tac.Constant(value=ConstInt(0))),
                tac.Label(name="loop"),
                tac.Return(val=tac.Constant(value=ConstInt(0))),
            )
        )
        jmpcc_instrs = [i for i in result.definitions[0].instructions if isinstance(i, asm.JmpCC)]
        assert len(jmpcc_instrs) == 1
        assert jmpcc_instrs[0].cond_code == asm.CondCodes.NE
        assert jmpcc_instrs[0].label == "loop"


# ---------------------------------------------------------------------------
# Cmp pseudo-register replacement
# ---------------------------------------------------------------------------


class TestCmpPseudoReplacement:
    """Pseudo operands inside ``Cmp`` and ``SetCC`` are resolved by pseudo-replacement."""

    def test_cmp_operands_resolved_to_stack(self) -> None:
        """Both operands of every ``Cmp`` instruction are ``Stack`` (not ``Pseudo``) after code gen."""
        result = generate(_relational_tac_program(tac.Equal(), tac.Var("a"), tac.Var("b")))
        for instr in result.definitions[0].instructions:
            if isinstance(instr, asm.Cmp):
                assert not isinstance(instr.op1, asm.Pseudo), "Pseudo survived in Cmp.op1"
                assert not isinstance(instr.op2, asm.Pseudo), "Pseudo survived in Cmp.op2"

    def test_setcc_operand_resolved(self) -> None:
        """The operand of every ``SetCC`` instruction is not a ``Pseudo`` after code gen."""
        result = generate(_relational_tac_program(tac.Equal(), tac.Var("a"), tac.Var("b")))
        for instr in result.definitions[0].instructions:
            if isinstance(instr, asm.SetCC):
                assert not isinstance(instr.operand, asm.Pseudo), "Pseudo survived in SetCC.operand"

    def test_jump_if_zero_cmp_operand_resolved(self) -> None:
        """The ``Cmp(Imm(0), cond)`` emitted for ``JumpIfZero`` has its condition operand resolved."""
        program = tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        tac.JumpIfZero(condition=tac.Var(value="x"), target="end"),
                        tac.Label(name="end"),
                        tac.Return(val=tac.Constant(value=ConstInt(0))),
                    ),
                ),
            )
        )
        result = generate(program)
        for instr in result.definitions[0].instructions:
            if isinstance(instr, asm.Cmp):
                assert not isinstance(instr.op1, asm.Pseudo)
                assert not isinstance(instr.op2, asm.Pseudo)


# ---------------------------------------------------------------------------
# Cmp instruction fixup
# ---------------------------------------------------------------------------


class TestCmpFixup:
    """``_fix_instructions`` rewrites invalid ``Cmp`` operand combinations."""

    def test_cmp_imm_op2_routed_through_r11(self) -> None:
        """When ``Cmp`` op2 is an immediate, it is routed through ``%r11d``."""
        result = generate(_relational_tac_program(tac.Equal()))
        for instr in result.definitions[0].instructions:
            if isinstance(instr, asm.Cmp):
                assert not isinstance(instr.op2, asm.Imm), "Imm survived as Cmp.op2 after fixup"

    def test_cmp_stack_stack_routed_through_r10(self) -> None:
        """When both ``Cmp`` operands are ``Stack``, op1 is routed through ``%r10d``."""
        result = generate(_relational_tac_program(tac.Equal(), tac.Var("a"), tac.Var("b")))
        for instr in result.definitions[0].instructions:
            if isinstance(instr, asm.Cmp):
                assert not (isinstance(instr.op1, asm.Memory) and isinstance(instr.op2, asm.Memory)), (
                    "Cmp(mem, mem) survived fixup"
                )

    def test_cmp_stack_imm_imm_case_fires_first(self) -> None:
        """``Cmp(Stack, Imm)`` is handled by the Imm fixup: op2 goes to ``%r11d``, op1 stays as Stack."""
        # src1=Constant -> Imm becomes op2 of Cmp; src2=Var -> Stack becomes op1 of Cmp
        result = generate(_relational_tac_program(tac.Equal(), tac.Constant(ConstInt(5)), tac.Var("b")))
        cmp_instrs = [i for i in result.definitions[0].instructions if isinstance(i, asm.Cmp)]
        assert len(cmp_instrs) == 1
        cmp = cmp_instrs[0]
        assert cmp.op2 == asm.Register(asm.GPRegisters.R11), "op2 should be routed through R11"
        assert isinstance(cmp.op1, asm.Memory), "op1 (Memory) should be unchanged"


# ---------------------------------------------------------------------------
# Pseudo replacement - static variables become Data operands
# ---------------------------------------------------------------------------


def _program_with_statics(*static_names: str, body: tuple[tac.Instruction, ...]) -> tac.Program:
    """Build a ``tac.Program`` containing static variables and a ``main`` function.

    Args:
        *static_names: Names of static variables to include as ``tac.StaticVariable`` entries.
        body: Instructions for the ``main`` function body.

    Returns:
        A ``tac.Program`` whose ``top_level`` begins with a ``StaticVariable`` for each
        name in ``static_names``, followed by a single ``main`` function.
    """
    statics = tuple(tac.StaticVariable(name=n, globl=True, t=IntType(), init=(tac.IntInit(0),)) for n in static_names)
    return tac.Program(top_level=statics + (tac.Function(name="main", body=body),))


def _fn_instructions(program: tac.Program) -> tuple[asm.Instruction, ...]:
    """Return the instructions of the first ``asm.Function`` in the generated program.

    Args:
        program: A ``tac.Program`` to pass through ``generate``.

    Returns:
        The ``instructions`` tuple of the first ``asm.Function`` definition produced.
    """
    return next(d for d in generate(program).definitions if isinstance(d, asm.Function)).instructions


def _fn_instructions_with_table(program: tac.Program, symbol_table: SymbolTable) -> tuple[asm.Instruction, ...]:
    """Return the instructions of the first ``asm.Function``, passing a symbol table to ``generate``.

    Args:
        program: A ``tac.Program`` to pass through ``generate``.
        symbol_table: The symbol table forwarded to ``generate`` for type-aware code generation.

    Returns:
        The ``instructions`` tuple of the first ``asm.Function`` definition produced.
    """
    return next(d for d in generate(program, symbol_table).definitions if isinstance(d, asm.Function)).instructions


class TestStaticVariablePseudoReplacement:
    """``tac.Var`` references to static-variable names become ``asm.Data``, not ``asm.Memory``."""

    def test_static_var_read_produces_data_operand(self) -> None:
        """Returning a static variable uses ``asm.Data`` as the source operand."""
        program = _program_with_statics("g", body=(tac.Return(val=tac.Var("g")),))
        instrs = _fn_instructions(program)
        assert any(isinstance(i, asm.Mov) and i.src == asm.Data("g") for i in instrs)

    def test_static_var_write_produces_data_dst_operand(self) -> None:
        """Assigning to a static variable uses ``asm.Data`` as the destination operand."""
        program = _program_with_statics(
            "g",
            body=(
                tac.Copy(src=tac.Constant(ConstInt(7)), dst=tac.Var("g")),
                tac.Return(val=tac.Constant(ConstInt(0))),
            ),
        )
        instrs = _fn_instructions(program)
        # The copy becomes Mov(Imm(7), Data("g")), which is valid (imm -> mem).
        assert any(isinstance(i, asm.Mov) and i.dst == asm.Data("g") for i in instrs)

    def test_static_var_does_not_consume_stack_slot(self) -> None:
        """A static variable and a local variable each allocate distinct operand kinds.

        The local gets a ``Stack`` slot; the static gets ``asm.Data``.  Verifies
        that the static name does not accidentally consume a stack slot (which
        would leave the local at the wrong offset, or allocate more frame space
        than needed).
        """
        program = _program_with_statics(
            "g",
            body=(
                tac.Copy(src=tac.Constant(ConstInt(1)), dst=tac.Var("local")),
                tac.Return(val=tac.Var("local")),
            ),
        )
        instrs = _fn_instructions(program)
        # "local" must occupy a stack slot.
        assert any(isinstance(i, asm.Mov) and isinstance(i.dst, asm.Memory) for i in instrs)
        # There must be no stack slot at -8 (which would indicate two locals were allocated,
        # meaning "g" was wrongly treated as a local alongside "local").
        all_stack_offsets = {
            op.offset for i in instrs if isinstance(i, asm.Mov) for op in (i.src, i.dst) if isinstance(op, asm.Memory)
        }
        assert -8 not in all_stack_offsets, "static 'g' was incorrectly assigned a local stack slot"

    def test_no_pseudo_operands_survive_with_static_variable(self) -> None:
        """No ``Pseudo`` operands remain after code generation when a static variable is present."""
        program = _program_with_statics("g", body=(tac.Return(val=tac.Var("g")),))
        instrs = _fn_instructions(program)
        for instr in instrs:
            match instr:
                case asm.Mov(src=src, dst=dst):
                    assert not isinstance(src, asm.Pseudo), "Pseudo survived as Mov.src"
                    assert not isinstance(dst, asm.Pseudo), "Pseudo survived as Mov.dst"
                case asm.Unary(operand=op):
                    assert not isinstance(op, asm.Pseudo), "Pseudo survived as Unary.operand"


# ---------------------------------------------------------------------------
# Instruction fixup - Data (static variable) as a memory operand
# ---------------------------------------------------------------------------


class TestDataOperandFixup:
    """``_fix_instructions`` treats ``asm.Data`` as a memory operand, the same as ``asm.Memory``."""

    def test_mov_data_data_no_mem_mem_after_fixup(self) -> None:
        """``Copy(static_x, static_y)`` produces no ``Mov(mem, mem)`` after fixup."""
        program = _program_with_statics(
            "x",
            "y",
            body=(
                tac.Copy(src=tac.Var("x"), dst=tac.Var("y")),
                tac.Return(val=tac.Constant(ConstInt(0))),
            ),
        )
        for instr in _fn_instructions(program):
            if isinstance(instr, asm.Mov):
                assert not (
                    isinstance(instr.src, (asm.Memory, asm.Data)) and isinstance(instr.dst, (asm.Memory, asm.Data))
                ), "Mov(mem, mem) survived fixup"

    def test_mov_data_stack_no_mem_mem_after_fixup(self) -> None:
        """``Copy(static_g, local)`` produces no ``Mov(Data, Stack)`` after fixup."""
        program = _program_with_statics(
            "g",
            body=(
                tac.Copy(src=tac.Var("g"), dst=tac.Var("local")),
                tac.Return(val=tac.Var("local")),
            ),
        )
        for instr in _fn_instructions(program):
            if isinstance(instr, asm.Mov):
                assert not (
                    isinstance(instr.src, (asm.Memory, asm.Data)) and isinstance(instr.dst, (asm.Memory, asm.Data))
                ), "Mov(mem, mem) survived fixup"

    def test_mov_stack_data_no_mem_mem_after_fixup(self) -> None:
        """``Copy(local, static_g)`` produces no ``Mov(Stack, Data)`` after fixup."""
        program = _program_with_statics(
            "g",
            body=(
                tac.Copy(src=tac.Var("local"), dst=tac.Var("g")),
                tac.Return(val=tac.Constant(ConstInt(0))),
            ),
        )
        for instr in _fn_instructions(program):
            if isinstance(instr, asm.Mov):
                assert not (
                    isinstance(instr.src, (asm.Memory, asm.Data)) and isinstance(instr.dst, (asm.Memory, asm.Data))
                ), "Mov(mem, mem) survived fixup"

    def test_cmp_data_op1_routed_through_r10(self) -> None:
        """When ``Cmp`` op1 is a ``Data`` operand (static variable), it is routed through ``%r10d``."""
        program = _program_with_statics(
            "x",
            "y",
            body=(
                tac.Binary(
                    operator=tac.Equal(),
                    src1=tac.Var("x"),
                    src2=tac.Var("y"),
                    dst=tac.Var("tmp"),
                ),
                tac.Return(val=tac.Var("tmp")),
            ),
        )
        for instr in _fn_instructions(program):
            if isinstance(instr, asm.Cmp):
                assert not isinstance(instr.op1, (asm.Memory, asm.Data)), "memory op1 survived Cmp fixup"


# ---------------------------------------------------------------------------
# Instruction fixup - Push with memory operands
# ---------------------------------------------------------------------------


def _seven_arg_call_program(*extra_top_level: tac.TopLevel, seventh_arg: tac.Val) -> tac.Program:
    """Build a ``tac.Program`` whose ``main`` calls a seven-argument function.

    The first six arguments are constants (consumed by ABI registers); the
    seventh is ``seventh_arg`` and will be pushed onto the stack.

    Args:
        *extra_top_level: Additional ``tac.TopLevel`` entries (e.g. ``StaticVariable``) to
            prepend before the ``main`` function.
        seventh_arg: The TAC value to pass as the seventh (stack-pushed) argument.

    Returns:
        A ``tac.Program`` that calls ``foo`` with seven arguments.
    """
    fn = tac.Function(
        name="main",
        body=(
            tac.FunCall(
                fun_val=tac.Var("foo"),
                args=(
                    tac.Constant(ConstInt(1)),
                    tac.Constant(ConstInt(2)),
                    tac.Constant(ConstInt(3)),
                    tac.Constant(ConstInt(4)),
                    tac.Constant(ConstInt(5)),
                    tac.Constant(ConstInt(6)),
                    seventh_arg,
                ),
                dst=tac.Var("result"),
            ),
            tac.Return(val=tac.Var("result")),
        ),
    )
    return tac.Program(top_level=tuple(extra_top_level) + (fn,))


class TestPushMemOperandFixup:
    """``Push(Stack)`` and ``Push(Data)`` are both expanded through ``%eax``/``%rax``."""

    def test_push_stack_operand_expanded(self) -> None:
        """A stack-passed local variable is not emitted as a direct ``Push(Stack)``."""
        program = _seven_arg_call_program(seventh_arg=tac.Var("x"))
        for instr in _fn_instructions(program):
            if isinstance(instr, asm.Push):
                assert not isinstance(instr.operand, (asm.Memory, asm.Data)), (
                    "Push(mem) survived fixup - should be expanded through %eax/%rax"
                )

    def test_push_data_operand_expanded(self) -> None:
        """A stack-passed static variable is not emitted as a direct ``Push(Data)``."""
        static = tac.StaticVariable(name="g", globl=True, t=IntType(), init=(tac.IntInit(0),))
        program = _seven_arg_call_program(static, seventh_arg=tac.Var("g"))
        for instr in _fn_instructions(program):
            if isinstance(instr, asm.Push):
                assert not isinstance(instr.operand, (asm.Memory, asm.Data)), (
                    "Push(Data) survived fixup - should be expanded through %eax/%rax"
                )


# ---------------------------------------------------------------------------
# Instruction fixup - large Quadword immediates
# ---------------------------------------------------------------------------

_LARGE_LONG: int = 34359738368  # 0x800000000; doesn't fit in a signed 32-bit integer


class TestPushLargeImmFixup:
    """``Push(Quadword, Imm(v))`` where ``v`` doesn't fit in 32 bits is expanded through ``%r10``."""

    def test_push_large_long_imm_not_emitted_directly(self) -> None:
        """A large long constant stack arg produces no ``Push(Imm)`` after fixup."""
        program = _seven_arg_call_program(seventh_arg=tac.Constant(ConstLong(_LARGE_LONG)))
        for instr in _fn_instructions(program):
            if isinstance(instr, asm.Push):
                assert not isinstance(instr.operand, asm.Imm), "Push(Imm(large)) survived fixup"

    def test_push_large_long_imm_routed_through_r10(self) -> None:
        """A large long constant stack arg is loaded into ``%r10`` immediately before the push."""
        program = _seven_arg_call_program(seventh_arg=tac.Constant(ConstLong(_LARGE_LONG)))
        instrs = list(_fn_instructions(program))
        push_idx = next(i for i, instr in enumerate(instrs) if isinstance(instr, asm.Push))
        mov = instrs[push_idx - 1]
        assert isinstance(mov, asm.Mov), "expected Mov before Push for large-imm fixup"
        assert isinstance(mov.src, asm.Imm) and mov.src.value == _LARGE_LONG
        assert isinstance(mov.dst, asm.Register) and mov.dst.reg == asm.GPRegisters.R10


class TestBinaryLargeImmFixup:
    """``Binary(Quadword, op, src1, Imm(v), dst)`` where ``v`` doesn't fit in 32 bits routes src2 through ``%r10``."""

    def _long_binary_instructions(self, op: tac.BinaryOperator) -> tuple[asm.Instruction, ...]:
        """Generate instructions for a Quadword Binary with a large immediate as both src operands.

        Args:
            op: The binary operator to test.

        Returns:
            The assembled instruction sequence after all fixup passes.
        """
        symbol_table: SymbolTable = {"tmp": SymbolTableEntry(t=LongType(), attrs=LocalAttr())}
        program = tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        tac.Binary(
                            operator=op,
                            src1=tac.Constant(ConstLong(_LARGE_LONG)),
                            src2=tac.Constant(ConstLong(_LARGE_LONG)),
                            dst=tac.Var("tmp"),
                        ),
                        tac.Return(val=tac.Var("tmp")),
                    ),
                ),
            )
        )
        return _fn_instructions_with_table(program, symbol_table)

    @pytest.mark.parametrize("op", [tac.Add(), tac.Subtract(), tac.Multiply()])
    def test_binary_large_imm_src2_not_present(self, op: tac.BinaryOperator) -> None:
        """No ``Binary`` instruction retains a large Quadword Imm as ``src2`` after fixup.

        Args:
            op: The binary operator to test (Add, Subtract, Multiply).
        """
        for instr in self._long_binary_instructions(op):
            if isinstance(instr, asm.Binary) and instr.assembly_type == asm.Quadword():
                assert not (
                    isinstance(instr.src2, asm.Imm) and not (-(1 << 31) <= instr.src2.value <= (1 << 31) - 1)
                ), "Binary kept large Imm as src2 after fixup"

    @pytest.mark.parametrize("op", [tac.Add(), tac.Subtract(), tac.Multiply()])
    def test_binary_large_imm_src2_routed_through_r10(self, op: tac.BinaryOperator) -> None:
        """A large Quadword immediate src2 is moved into ``%r10`` before the Binary instruction.

        Args:
            op: The binary operator to test (Add, Subtract, Multiply).
        """
        instrs = list(self._long_binary_instructions(op))
        binary_idx = next(i for i, instr in enumerate(instrs) if isinstance(instr, asm.Binary))
        mov = instrs[binary_idx - 1]
        assert isinstance(mov, asm.Mov), "expected Mov before Binary for large-imm fixup"
        assert isinstance(mov.src, asm.Imm) and mov.src.value == _LARGE_LONG
        assert isinstance(mov.dst, asm.Register) and mov.dst.reg == asm.GPRegisters.R10


class TestCmpBothLargeImmFixup:
    """``Cmp(Quadword, Imm(large), Imm)`` routes both operands through registers rather than
    leaving Imm(large) as op1.
    """

    def test_cmp_large_op1_imm_op2_no_imm_survives(self) -> None:
        """No ``Cmp`` retains a large Imm as ``op1`` when ``op2`` is also an immediate."""
        program = tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        tac.Binary(
                            operator=tac.Equal(),
                            src1=tac.Constant(ConstLong(_LARGE_LONG)),
                            src2=tac.Constant(ConstLong(_LARGE_LONG)),
                            dst=tac.Var("tmp"),
                        ),
                        tac.Return(val=tac.Var("tmp")),
                    ),
                ),
            )
        )
        for instr in _fn_instructions(program):
            if isinstance(instr, asm.Cmp) and instr.assembly_type == asm.Quadword():
                assert not (isinstance(instr.op1, asm.Imm) and not (-(1 << 31) <= instr.op1.value <= (1 << 31) - 1)), (
                    "large Imm survived as Cmp.op1 after fixup"
                )


# ---------------------------------------------------------------------------
# Char type support
# ---------------------------------------------------------------------------


def _make_symbol_table(**vars: type) -> SymbolTable:
    """Build a ``SymbolTable`` mapping each name to the given C type.

    Args:
        **vars: Keyword arguments mapping variable names to their C type classes.

    Returns:
        A ``SymbolTable`` with a ``LocalAttr`` entry for each variable.
    """
    return {name: SymbolTableEntry(t=t(), attrs=LocalAttr()) for name, t in vars.items()}


class TestAsmTypeChar:
    """``_asm_type`` returns ``Byte`` for char-typed TAC values."""

    def test_const_char_returns_byte(self) -> None:
        """``ConstChar`` constant maps to ``Byte`` assembly type."""
        assert _asm_type(tac.Constant(value=ConstChar(65))) == asm.Byte()

    def test_const_uchar_returns_byte(self) -> None:
        """``ConstUChar`` constant maps to ``Byte`` assembly type."""
        assert _asm_type(tac.Constant(value=ConstUChar(200))) == asm.Byte()

    def test_schar_var_returns_byte(self) -> None:
        """``SChar`` variable maps to ``Byte`` assembly type."""
        st = _make_symbol_table(c=SChar)
        assert _asm_type(tac.Var("c"), st) == asm.Byte()

    def test_uchar_var_returns_byte(self) -> None:
        """``UChar`` variable maps to ``Byte`` assembly type."""
        st = _make_symbol_table(c=UChar)
        assert _asm_type(tac.Var("c"), st) == asm.Byte()

    def test_char_var_returns_byte(self) -> None:
        """``Char`` variable maps to ``Byte`` assembly type."""
        st = _make_symbol_table(c=Char)
        assert _asm_type(tac.Var("c"), st) == asm.Byte()

    def test_int_var_still_returns_longword(self) -> None:
        """``IntType`` variable still maps to ``Longword`` (regression guard)."""
        st = _make_symbol_table(x=IntType)
        assert _asm_type(tac.Var("x"), st) == asm.Longword()


class TestRegWidthByte:
    """``_reg_width`` returns ``Width.BYTE`` for ``Byte()``."""

    def test_byte_returns_width_byte(self) -> None:
        """``_reg_width(Byte())`` returns ``Width.BYTE``."""
        assert _reg_width(asm.Byte()) == asm.Width.BYTE

    def test_longword_still_returns_dword(self) -> None:
        """``_reg_width(Longword())`` still returns ``Width.DWORD`` (regression guard)."""
        assert _reg_width(asm.Longword()) == asm.Width.DWORD

    def test_quadword_still_returns_qword(self) -> None:
        """``_reg_width(Quadword())`` still returns ``Width.QWORD`` (regression guard)."""
        assert _reg_width(asm.Quadword()) == asm.Width.QWORD


def _sign_extend_program(src_type: type, dst_type: type, src_name: str = "s", dst_name: str = "d") -> tac.Program:
    """Build a TAC program containing a single ``SignExtend`` instruction.

    Args:
        src_type: C type for the source variable (e.g. ``SChar``).
        dst_type: C type for the destination variable (e.g. ``IntType``).
        src_name: Variable name for the source.
        dst_name: Variable name for the destination.

    Returns:
        A ``tac.Program`` with one ``SignExtend`` and one ``Return``.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.SignExtend(src=tac.Var(src_name), dst=tac.Var(dst_name)),
                    tac.Return(val=tac.Var(dst_name)),
                ),
            ),
        )
    )


def _zero_extend_program(src_type: type, dst_type: type, src_name: str = "s", dst_name: str = "d") -> tac.Program:
    """Build a TAC program containing a single ``ZeroExtend`` instruction.

    Args:
        src_type: C type for the source variable (e.g. ``UChar``).
        dst_type: C type for the destination variable (e.g. ``UIntType``).
        src_name: Variable name for the source.
        dst_name: Variable name for the destination.

    Returns:
        A ``tac.Program`` with one ``ZeroExtend`` and one ``Return``.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.ZeroExtend(src=tac.Var(src_name), dst=tac.Var(dst_name)),
                    tac.Return(val=tac.Var(dst_name)),
                ),
            ),
        )
    )


class TestSignExtendCharCodeGen:
    """``tac.SignExtend`` from char generates ``Movsx`` with correct src/dst types."""

    def test_schar_to_int_generates_movsx_byte_longword(self) -> None:
        """``SignExtend(SChar -> int)`` generates ``Movsx(Byte, Longword)``."""
        st = _make_symbol_table(s=SChar, d=IntType)
        program = _sign_extend_program(SChar, IntType)
        instrs = _fn_instructions_with_table(program, st)
        movsx = next(i for i in instrs if isinstance(i, asm.Movsx))
        assert movsx.src_type == asm.Byte()
        assert movsx.dst_type == asm.Longword()

    def test_schar_to_long_generates_movsx_byte_quadword(self) -> None:
        """``SignExtend(SChar -> long)`` generates ``Movsx(Byte, Quadword)``."""
        st = _make_symbol_table(s=SChar, d=LongType)
        program = _sign_extend_program(SChar, LongType)
        instrs = _fn_instructions_with_table(program, st)
        movsx = next(i for i in instrs if isinstance(i, asm.Movsx))
        assert movsx.src_type == asm.Byte()
        assert movsx.dst_type == asm.Quadword()

    def test_int_to_long_generates_movsx_longword_quadword(self) -> None:
        """``SignExtend(int -> long)`` generates ``Movsx(Longword, Quadword)`` (regression guard)."""
        st = _make_symbol_table(s=IntType, d=LongType)
        program = _sign_extend_program(IntType, LongType)
        instrs = _fn_instructions_with_table(program, st)
        movsx = next(i for i in instrs if isinstance(i, asm.Movsx))
        assert movsx.src_type == asm.Longword()
        assert movsx.dst_type == asm.Quadword()


class TestZeroExtendCharCodeGen:
    """``tac.ZeroExtend`` from uchar generates ``MoveZeroExtend`` with correct src/dst types."""

    def test_uchar_to_uint_generates_movze_byte_longword(self) -> None:
        """``ZeroExtend(UChar -> uint)`` generates ``MoveZeroExtend(Byte, Longword)``."""
        st = _make_symbol_table(s=UChar, d=UIntType)
        program = _zero_extend_program(UChar, UIntType)
        instrs = _fn_instructions_with_table(program, st)
        mze = next(i for i in instrs if isinstance(i, asm.MoveZeroExtend))
        assert mze.src_type == asm.Byte()
        assert mze.dst_type == asm.Longword()

    def test_uchar_to_ulong_generates_movze_byte_quadword(self) -> None:
        """``ZeroExtend(UChar -> ulong)`` generates ``MoveZeroExtend(Byte, Quadword)``."""
        st = _make_symbol_table(s=UChar, d=ULongType)
        program = _zero_extend_program(UChar, ULongType)
        instrs = _fn_instructions_with_table(program, st)
        mze = next(i for i in instrs if isinstance(i, asm.MoveZeroExtend))
        assert mze.src_type == asm.Byte()
        assert mze.dst_type == asm.Quadword()

    def test_uint_to_ulong_generates_mov_longword(self) -> None:
        """``ZeroExtend(uint -> ulong)`` is lowered to ``Mov(Longword)`` after fixup.

        ``movl`` to a 32-bit register implicitly zero-extends to 64 bits, so the
        fixup pass replaces ``MoveZeroExtend(Longword, Quadword)`` with a plain
        ``Mov(Longword)`` instruction.  A ``Negate`` into ``s_copy`` (a non-copy
        instruction) keeps ``s`` live past ``d``'s definition, so ``s`` and ``d``
        interfere and cannot be coalesced away - a plain ``Copy`` would get
        the copy-pair exception and fail to create the needed interference edge.
        """
        st = _make_symbol_table(s=UIntType, d=ULongType, s_copy=UIntType)
        program = tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        tac.ZeroExtend(src=tac.Var("s"), dst=tac.Var("d")),
                        tac.Unary(operator=tac.Negate(), src=tac.Var("s"), dst=tac.Var("s_copy")),
                        tac.Return(val=tac.Var("d")),
                    ),
                ),
            )
        )
        instrs = _fn_instructions_with_table(program, st)
        assert not any(isinstance(i, asm.MoveZeroExtend) for i in instrs)
        mov = next(i for i in instrs if isinstance(i, asm.Mov) and i.assembly_type == asm.Longword())
        assert mov.assembly_type == asm.Longword()


class TestTruncateToCharCodeGen:
    """``tac.Truncate`` to a char destination generates ``Mov(Byte)`` not ``Mov(Longword)``."""

    def test_truncate_to_schar_emits_movb(self) -> None:
        """``Truncate`` with an ``SChar`` destination uses ``Byte`` assembly type.

        A ``Negate(d, d)`` before the ``Truncate`` creates a non-copy definition of
        ``d`` while ``s`` is live (``s`` is used by the subsequent ``Truncate``).
        This adds an interference edge between ``s`` and ``d`` so the coalescer
        cannot merge them - a plain ``Copy`` would get the copy-pair exception and
        fail to produce the required edge.
        """
        st = _make_symbol_table(s=IntType, d=SChar)
        program = tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        tac.Unary(operator=tac.Negate(), src=tac.Var("d"), dst=tac.Var("d")),
                        tac.Truncate(src=tac.Var("s"), dst=tac.Var("d")),
                        tac.Return(val=tac.Var("d")),
                    ),
                ),
            )
        )
        instrs = _fn_instructions_with_table(program, st)
        movs = [i for i in instrs if isinstance(i, asm.Mov)]
        assert any(m.assembly_type == asm.Byte() for m in movs), "expected a Mov(Byte) for Truncate -> SChar"

    def test_truncate_to_int_still_emits_movl(self) -> None:
        """``Truncate`` with an ``IntType`` destination still uses ``Longword`` (regression guard).

        A ``Negate(d, d)`` before the ``Truncate`` creates a non-copy definition of
        ``d`` while ``s`` is live, adding an interference edge so the coalescer
        cannot merge them and eliminate the ``Mov(Longword)``.
        """
        st = _make_symbol_table(s=LongType, d=IntType)
        program = tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        tac.Unary(operator=tac.Negate(), src=tac.Var("d"), dst=tac.Var("d")),
                        tac.Truncate(src=tac.Var("s"), dst=tac.Var("d")),
                        tac.Return(val=tac.Var("d")),
                    ),
                ),
            )
        )
        instrs = _fn_instructions_with_table(program, st)
        movs = [i for i in instrs if isinstance(i, asm.Mov)]
        assert any(m.assembly_type == asm.Longword() for m in movs), "expected a Mov(Longword) for Truncate -> int"


def _funcall_program(arg_name: str) -> tac.Program:
    """Build a TAC program with a single function call passing one named variable.

    Args:
        arg_name: The variable name to pass as the sole argument.

    Returns:
        A ``tac.Program`` calling ``foo(arg_name)`` and returning the result.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.FunCall(
                        fun_val=tac.Var("foo"),
                        args=(tac.Var(arg_name),),
                        dst=tac.Var("ret"),
                    ),
                    tac.Return(val=tac.Var("ret")),
                ),
            ),
        )
    )


class TestCharArgExtension:
    """Char arguments are sign/zero-extended before placement in ABI registers."""

    def test_schar_arg_in_register_uses_movsx(self) -> None:
        """Signed char register argument emits ``Movsx(Byte, Longword)`` (``movsbl``)."""
        st = _make_symbol_table(c=SChar, ret=IntType)
        program = _funcall_program("c")
        instrs = _fn_instructions_with_table(program, st)
        movsx = next((i for i in instrs if isinstance(i, asm.Movsx)), None)
        assert movsx is not None, "expected Movsx for signed char arg"
        assert movsx.src_type == asm.Byte()
        assert movsx.dst_type == asm.Longword()

    def test_uchar_arg_in_register_uses_movzx(self) -> None:
        """Unsigned char register argument emits ``MoveZeroExtend(Byte, Longword)`` (``movzbl``)."""
        st = _make_symbol_table(c=UChar, ret=IntType)
        program = _funcall_program("c")
        instrs = _fn_instructions_with_table(program, st)
        mze = next((i for i in instrs if isinstance(i, asm.MoveZeroExtend)), None)
        assert mze is not None, "expected MoveZeroExtend for unsigned char arg"
        assert mze.src_type == asm.Byte()
        assert mze.dst_type == asm.Longword()

    def test_schar_arg_no_byte_width_register(self) -> None:
        """Signed char register argument never uses a byte-width (%dil) ABI register."""
        st = _make_symbol_table(c=SChar, ret=IntType)
        program = _funcall_program("c")
        instrs = _fn_instructions_with_table(program, st)
        for instr in instrs:
            if isinstance(instr, asm.Mov):
                if isinstance(instr.dst, asm.Register) and instr.dst.reg == asm.GPRegisters.DI:
                    assert instr.dst.width != asm.Width.BYTE, "char arg should not use byte-width DI register"

    def test_uchar_arg_no_byte_width_register(self) -> None:
        """Unsigned char register argument never uses a byte-width (%dil) ABI register."""
        st = _make_symbol_table(c=UChar, ret=IntType)
        program = _funcall_program("c")
        instrs = _fn_instructions_with_table(program, st)
        for instr in instrs:
            if isinstance(instr, asm.Mov):
                if isinstance(instr.dst, asm.Register) and instr.dst.reg == asm.GPRegisters.DI:
                    assert instr.dst.width != asm.Width.BYTE, "uchar arg should not use byte-width DI register"

    def test_int_arg_still_uses_mov(self) -> None:
        """Integer register argument still uses ``Mov`` (regression guard)."""
        st = _make_symbol_table(x=IntType, ret=IntType)
        program = _funcall_program("x")
        instrs = _fn_instructions_with_table(program, st)
        assert not any(isinstance(i, asm.Movsx) for i in instrs), "int arg should not produce Movsx"
        assert not any(isinstance(i, asm.MoveZeroExtend) for i in instrs), "int arg should not produce MoveZeroExtend"

    def test_duplicate_schar_arg_uses_movl_not_second_movsbl(self) -> None:
        """When the same signed-char variable appears in two argument slots, the second
        slot is filled with a ``Mov(Longword)`` from the first register rather than a
        second ``Movsx``.  The test-framework's ``find_reaching_copies`` only recognises
        ``MOV`` (``movl``/``movq``), not ``MOVS`` (``movsbl``), so a duplicate
        ``movsbl`` would cause the framework to report ``None`` for the second argument.
        """
        st = _make_symbol_table(c=SChar, ret=IntType)
        program = tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        tac.FunCall(
                            fun_val=tac.Var("foo"),
                            args=(tac.Var("c"), tac.Var("c")),
                            dst=tac.Var("ret"),
                        ),
                        tac.Return(val=tac.Var("ret")),
                    ),
                ),
            )
        )
        instrs = _fn_instructions_with_table(program, st)
        movsx_count = sum(1 for i in instrs if isinstance(i, asm.Movsx))
        assert movsx_count == 1, f"expected exactly one Movsx for duplicate schar args, got {movsx_count}"
        edi_movl = [
            i
            for i in instrs
            if isinstance(i, asm.Mov)
            and isinstance(i.dst, asm.Register)
            and i.dst.reg == asm.GPRegisters.SI
            and i.assembly_type == asm.Longword()
        ]
        assert edi_movl, "expected a Mov(Longword) into %esi for the duplicate arg"

    def test_duplicate_uchar_arg_uses_movl_not_second_movzbl(self) -> None:
        """When the same unsigned-char variable appears in two argument slots, the second
        slot is filled with a ``Mov(Longword)`` from the first register rather than a
        second ``MoveZeroExtend``.
        """
        st = _make_symbol_table(c=UChar, ret=IntType)
        program = tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        tac.FunCall(
                            fun_val=tac.Var("foo"),
                            args=(tac.Var("c"), tac.Var("c")),
                            dst=tac.Var("ret"),
                        ),
                        tac.Return(val=tac.Var("ret")),
                    ),
                ),
            )
        )
        instrs = _fn_instructions_with_table(program, st)
        mze_count = sum(1 for i in instrs if isinstance(i, asm.MoveZeroExtend))
        assert mze_count == 1, f"expected exactly one MoveZeroExtend for duplicate uchar args, got {mze_count}"
        esi_movl = [
            i
            for i in instrs
            if isinstance(i, asm.Mov)
            and isinstance(i.dst, asm.Register)
            and i.dst.reg == asm.GPRegisters.SI
            and i.assembly_type == asm.Longword()
        ]
        assert esi_movl, "expected a Mov(Longword) into %esi for the duplicate arg"


def _double_to_char_program(dst_name: str) -> tac.Program:
    """Build a TAC program with a single ``DoubleToInt`` instruction targeting a char variable.

    Args:
        dst_name: The destination variable name.

    Returns:
        A ``tac.Program`` that converts a double constant to ``dst_name`` via ``DoubleToInt``.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.DoubleToInt(src=tac.Var("d"), dst=tac.Var(dst_name)),
                    tac.Return(val=tac.Var(dst_name)),
                ),
            ),
        )
    )


def _double_to_uchar_program(dst_name: str) -> tac.Program:
    """Build a TAC program with a single ``DoubleToUInt`` instruction targeting an unsigned char variable.

    Args:
        dst_name: The destination variable name.

    Returns:
        A ``tac.Program`` that converts a double variable to ``dst_name`` via ``DoubleToUInt``.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.DoubleToUInt(src=tac.Var("d"), dst=tac.Var(dst_name)),
                    tac.Return(val=tac.Var(dst_name)),
                ),
            ),
        )
    )


def _char_to_double_program(src_name: str) -> tac.Program:
    """Build a TAC program with a single ``IntToDouble`` instruction from a char variable.

    Args:
        src_name: The source variable name (a char-typed variable).

    Returns:
        A ``tac.Program`` that converts ``src_name`` to double via ``IntToDouble``.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.IntToDouble(src=tac.Var(src_name), dst=tac.Var("d")),
                    tac.Return(val=tac.Var("d")),
                ),
            ),
        )
    )


def _uchar_to_double_program(src_name: str) -> tac.Program:
    """Build a TAC program with a single ``UIntToDouble`` instruction from an unsigned char variable.

    Args:
        src_name: The source variable name (a unsigned char-typed variable).

    Returns:
        A ``tac.Program`` that converts ``src_name`` to double via ``UIntToDouble``.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.UIntToDouble(src=tac.Var(src_name), dst=tac.Var("d")),
                    tac.Return(val=tac.Var("d")),
                ),
            ),
        )
    )


class TestDoubleToCharConversion:
    """``DoubleToInt`` with a char destination converts via int32 scratch then truncates."""

    def test_double_to_schar_emits_cvttsd2si_then_movb(self) -> None:
        """``DoubleToInt(dst=SChar)`` emits ``cvttsd2sil`` into scratch then ``movb`` to dst."""
        st = _make_symbol_table(d=DoubleType, c=SChar)
        instrs = _fn_instructions_with_table(_double_to_char_program("c"), st)
        cvt = next((i for i in instrs if isinstance(i, asm.Cvttsd2si)), None)
        assert cvt is not None, "expected Cvttsd2si"
        assert cvt.assembly_type == asm.Longword(), "cvttsd2si should target int32 scratch"
        movb = next((i for i in instrs if isinstance(i, asm.Mov) and i.assembly_type == asm.Byte()), None)
        assert movb is not None, "expected Mov(Byte) to truncate to char"

    def test_double_to_char_emits_cvttsd2si_then_movb(self) -> None:
        """``DoubleToInt(dst=Char)`` emits the same int32-then-truncate sequence."""
        st = _make_symbol_table(d=DoubleType, c=Char)
        instrs = _fn_instructions_with_table(_double_to_char_program("c"), st)
        assert any(isinstance(i, asm.Cvttsd2si) and i.assembly_type == asm.Longword() for i in instrs)
        assert any(isinstance(i, asm.Mov) and i.assembly_type == asm.Byte() for i in instrs)

    def test_double_to_int_still_uses_cvttsd2si_directly(self) -> None:
        """``DoubleToInt(dst=IntType)`` still emits a single ``cvttsd2sil`` (regression guard)."""
        st = _make_symbol_table(d=DoubleType, x=IntType)
        instrs = _fn_instructions_with_table(_double_to_char_program("x"), st)
        cvts = [i for i in instrs if isinstance(i, asm.Cvttsd2si)]
        assert len(cvts) == 1
        assert cvts[0].assembly_type == asm.Longword()
        assert not any(isinstance(i, asm.Mov) and i.assembly_type == asm.Byte() for i in instrs)


class TestDoubleToUCharConversion:
    """``DoubleToUInt`` with an unsigned char destination converts via int32 scratch then truncates."""

    def test_double_to_uchar_emits_cvttsd2si_then_movb(self) -> None:
        """``DoubleToUInt(dst=UChar)`` emits ``cvttsd2sil`` into scratch then ``movb`` to dst."""
        st = _make_symbol_table(d=DoubleType, c=UChar)
        instrs = _fn_instructions_with_table(_double_to_uchar_program("c"), st)
        cvt = next((i for i in instrs if isinstance(i, asm.Cvttsd2si)), None)
        assert cvt is not None, "expected Cvttsd2si"
        assert cvt.assembly_type == asm.Longword(), "cvttsd2si should target int32 scratch"
        assert any(isinstance(i, asm.Mov) and i.assembly_type == asm.Byte() for i in instrs)

    def test_double_to_uint_still_uses_cvttsd2siq_then_movl(self) -> None:
        """``DoubleToUInt(dst=UIntType)`` still uses the quadword-then-truncate path (regression guard)."""
        st = _make_symbol_table(d=DoubleType, x=UIntType)
        instrs = _fn_instructions_with_table(_double_to_uchar_program("x"), st)
        cvts = [i for i in instrs if isinstance(i, asm.Cvttsd2si)]
        assert len(cvts) == 1
        assert cvts[0].assembly_type == asm.Quadword()


class TestCharToDoubleConversion:
    """``IntToDouble`` with a char source sign-extends to int32 then converts."""

    def test_schar_to_double_emits_movsbl_then_cvtsi2sd(self) -> None:
        """``IntToDouble(src=SChar)`` emits ``movsbl`` then ``cvtsi2sdl``."""
        st = _make_symbol_table(c=SChar, d=DoubleType)
        instrs = _fn_instructions_with_table(_char_to_double_program("c"), st)
        movsx = next((i for i in instrs if isinstance(i, asm.Movsx)), None)
        assert movsx is not None, "expected Movsx (movsbl) to sign-extend char"
        assert movsx.src_type == asm.Byte()
        assert movsx.dst_type == asm.Longword()
        cvt = next((i for i in instrs if isinstance(i, asm.Cvtsi2sd)), None)
        assert cvt is not None, "expected Cvtsi2sd"
        assert cvt.assembly_type == asm.Longword()

    def test_char_to_double_emits_movsbl_then_cvtsi2sd(self) -> None:
        """``IntToDouble(src=Char)`` emits the same sign-extend-then-convert sequence."""
        st = _make_symbol_table(c=Char, d=DoubleType)
        instrs = _fn_instructions_with_table(_char_to_double_program("c"), st)
        assert any(isinstance(i, asm.Movsx) and i.src_type == asm.Byte() for i in instrs)
        assert any(isinstance(i, asm.Cvtsi2sd) and i.assembly_type == asm.Longword() for i in instrs)

    def test_int_to_double_still_uses_cvtsi2sd_directly(self) -> None:
        """``IntToDouble(src=IntType)`` still emits a single ``cvtsi2sdl`` directly (regression guard)."""
        st = _make_symbol_table(x=IntType, d=DoubleType)
        instrs = _fn_instructions_with_table(_char_to_double_program("x"), st)
        assert not any(isinstance(i, asm.Movsx) for i in instrs), "int src should not produce Movsx"
        cvts = [i for i in instrs if isinstance(i, asm.Cvtsi2sd)]
        assert len(cvts) == 1
        assert cvts[0].assembly_type == asm.Longword()


class TestUCharToDoubleConversion:
    """``UIntToDouble`` with an unsigned char source zero-extends to int32 then converts."""

    def test_uchar_to_double_emits_movzbl_then_cvtsi2sd(self) -> None:
        """``UIntToDouble(src=UChar)`` emits ``movzbl`` then ``cvtsi2sdl``."""
        st = _make_symbol_table(c=UChar, d=DoubleType)
        instrs = _fn_instructions_with_table(_uchar_to_double_program("c"), st)
        mze = next((i for i in instrs if isinstance(i, asm.MoveZeroExtend)), None)
        assert mze is not None, "expected MoveZeroExtend (movzbl) to zero-extend uchar"
        assert mze.src_type == asm.Byte()
        assert mze.dst_type == asm.Longword()
        cvt = next((i for i in instrs if isinstance(i, asm.Cvtsi2sd)), None)
        assert cvt is not None, "expected Cvtsi2sd"
        assert cvt.assembly_type == asm.Longword()

    def test_uint_to_double_still_uses_movl_zero_extend_path(self) -> None:
        """``UIntToDouble(src=UIntType)`` still uses the 32-bit zero-extend path (regression guard)."""
        st = _make_symbol_table(x=UIntType, d=DoubleType)
        instrs = _fn_instructions_with_table(_uchar_to_double_program("x"), st)
        assert not any(isinstance(i, asm.MoveZeroExtend) for i in instrs), "uint src should not produce MoveZeroExtend"
        assert any(isinstance(i, asm.Cvtsi2sd) and i.assembly_type == asm.Quadword() for i in instrs)


def _float_to_int_program(dst_name: str) -> tac.Program:
    """Build a TAC program with a single ``FloatToInt`` instruction.

    Args:
        dst_name: The destination variable name.

    Returns:
        A ``tac.Program`` that converts a float variable to ``dst_name`` via ``FloatToInt``.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.FloatToInt(src=tac.Var("f"), dst=tac.Var(dst_name)),
                    tac.Return(val=tac.Var(dst_name)),
                ),
            ),
        )
    )


def _float_to_uint_program(dst_name: str) -> tac.Program:
    """Build a TAC program with a single ``FloatToUInt`` instruction.

    Args:
        dst_name: The destination variable name.

    Returns:
        A ``tac.Program`` that converts a float variable to ``dst_name`` via ``FloatToUInt``.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.FloatToUInt(src=tac.Var("f"), dst=tac.Var(dst_name)),
                    tac.Return(val=tac.Var(dst_name)),
                ),
            ),
        )
    )


def _int_to_float_program(src_name: str) -> tac.Program:
    """Build a TAC program with a single ``IntToFloat`` instruction.

    Args:
        src_name: The source variable name.

    Returns:
        A ``tac.Program`` that converts ``src_name`` to float via ``IntToFloat``.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.IntToFloat(src=tac.Var(src_name), dst=tac.Var("f")),
                    tac.Return(val=tac.Var("f")),
                ),
            ),
        )
    )


def _uint_to_float_program(src_name: str) -> tac.Program:
    """Build a TAC program with a single ``UIntToFloat`` instruction.

    Args:
        src_name: The source variable name.

    Returns:
        A ``tac.Program`` that converts ``src_name`` to float via ``UIntToFloat``.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.UIntToFloat(src=tac.Var(src_name), dst=tac.Var("f")),
                    tac.Return(val=tac.Var("f")),
                ),
            ),
        )
    )


def _float_to_double_program() -> tac.Program:
    """Build a TAC program with a single ``FloatToDouble`` instruction.

    Returns:
        A ``tac.Program`` that widens a float to double via ``FloatToDouble``.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.FloatToDouble(src=tac.Var("f"), dst=tac.Var("d")),
                    tac.Return(val=tac.Var("d")),
                ),
            ),
        )
    )


def _double_to_float_program() -> tac.Program:
    """Build a TAC program with a single ``DoubleToFloat`` instruction.

    Returns:
        A ``tac.Program`` that narrows a double to float via ``DoubleToFloat``.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.DoubleToFloat(src=tac.Var("d"), dst=tac.Var("f")),
                    tac.Return(val=tac.Var("f")),
                ),
            ),
        )
    )


class TestFloatToIntConversion:
    """``FloatToInt`` lowers to ``Cvttss2si``, with a byte-path workaround for char destinations."""

    def test_float_to_int_emits_cvttss2si(self) -> None:
        """``FloatToInt(dst=IntType)`` emits a single ``cvttss2sil``."""
        st = _make_symbol_table(f=FloatType, x=IntType)
        instrs = _fn_instructions_with_table(_float_to_int_program("x"), st)
        cvts = [i for i in instrs if isinstance(i, asm.Cvttss2si)]
        assert len(cvts) == 1
        assert cvts[0].assembly_type == asm.Longword()

    def test_float_to_schar_emits_cvttss2si_then_movb(self) -> None:
        """``FloatToInt(dst=SChar)`` converts via int32 scratch then truncates to byte."""
        st = _make_symbol_table(f=FloatType, c=SChar)
        instrs = _fn_instructions_with_table(_float_to_int_program("c"), st)
        cvt = next((i for i in instrs if isinstance(i, asm.Cvttss2si)), None)
        assert cvt is not None, "expected Cvttss2si"
        assert cvt.assembly_type == asm.Longword(), "should target int32 scratch"
        assert any(isinstance(i, asm.Mov) and i.assembly_type == asm.Byte() for i in instrs)


class TestIntToFloatConversion:
    """``IntToFloat`` lowers to ``Cvtsi2ss``, with a byte-path workaround for char sources."""

    def test_int_to_float_emits_cvtsi2ss(self) -> None:
        """``IntToFloat(src=IntType)`` emits a single ``cvtsi2ssl``."""
        st = _make_symbol_table(x=IntType, f=FloatType)
        instrs = _fn_instructions_with_table(_int_to_float_program("x"), st)
        cvts = [i for i in instrs if isinstance(i, asm.Cvtsi2ss)]
        assert len(cvts) == 1
        assert cvts[0].assembly_type == asm.Longword()

    def test_schar_to_float_emits_movsbl_then_cvtsi2ss(self) -> None:
        """``IntToFloat(src=SChar)`` sign-extends to int32 then converts."""
        st = _make_symbol_table(c=SChar, f=FloatType)
        instrs = _fn_instructions_with_table(_int_to_float_program("c"), st)
        movsx = next((i for i in instrs if isinstance(i, asm.Movsx)), None)
        assert movsx is not None, "expected Movsx (movsbl) to sign-extend char"
        assert movsx.src_type == asm.Byte()
        assert movsx.dst_type == asm.Longword()
        cvt = next((i for i in instrs if isinstance(i, asm.Cvtsi2ss)), None)
        assert cvt is not None, "expected Cvtsi2ss"
        assert cvt.assembly_type == asm.Longword()


class TestUIntToFloatConversion:
    """``UIntToFloat`` lowers similarly to ``UIntToDouble`` but uses ``ss`` instructions."""

    def test_uint_to_float_emits_cvtsi2ss(self) -> None:
        """``UIntToFloat(src=UIntType)`` uses the 32-bit zero-extend path."""
        st = _make_symbol_table(x=UIntType, f=FloatType)
        instrs = _fn_instructions_with_table(_uint_to_float_program("x"), st)
        assert any(isinstance(i, asm.Cvtsi2ss) for i in instrs)

    def test_uchar_to_float_emits_movzbl_then_cvtsi2ss(self) -> None:
        """``UIntToFloat(src=UChar)`` zero-extends to int32 then converts."""
        st = _make_symbol_table(c=UChar, f=FloatType)
        instrs = _fn_instructions_with_table(_uint_to_float_program("c"), st)
        mze = next((i for i in instrs if isinstance(i, asm.MoveZeroExtend)), None)
        assert mze is not None, "expected MoveZeroExtend (movzbl)"
        assert mze.src_type == asm.Byte()
        assert mze.dst_type == asm.Longword()
        assert any(isinstance(i, asm.Cvtsi2ss) for i in instrs)


class TestFloatToUIntConversion:
    """``FloatToUInt`` lowers similarly to ``DoubleToUInt`` but uses ``ss`` instructions."""

    def test_float_to_uint_emits_cvttss2si_then_movl(self) -> None:
        """``FloatToUInt(dst=UIntType)`` converts via quadword scratch then truncates."""
        st = _make_symbol_table(f=FloatType, x=UIntType)
        instrs = _fn_instructions_with_table(_float_to_uint_program("x"), st)
        cvts = [i for i in instrs if isinstance(i, asm.Cvttss2si)]
        assert len(cvts) == 1
        assert cvts[0].assembly_type == asm.Quadword()

    def test_float_to_uchar_emits_cvttss2si_then_movb(self) -> None:
        """``FloatToUInt(dst=UChar)`` converts via int32 scratch then truncates to byte."""
        st = _make_symbol_table(f=FloatType, c=UChar)
        instrs = _fn_instructions_with_table(_float_to_uint_program("c"), st)
        cvt = next((i for i in instrs if isinstance(i, asm.Cvttss2si)), None)
        assert cvt is not None, "expected Cvttss2si"
        assert cvt.assembly_type == asm.Longword(), "should target int32 scratch"
        assert any(isinstance(i, asm.Mov) and i.assembly_type == asm.Byte() for i in instrs)


class TestFloatToDoubleConversion:
    """``FloatToDouble`` lowers to a single ``Cvtss2sd`` instruction."""

    def test_float_to_double_emits_cvtss2sd(self) -> None:
        """``FloatToDouble`` emits a single ``cvtss2sd``."""
        st = _make_symbol_table(f=FloatType, d=DoubleType)
        instrs = _fn_instructions_with_table(_float_to_double_program(), st)
        cvts = [i for i in instrs if isinstance(i, asm.Cvtss2sd)]
        assert len(cvts) == 1


class TestDoubleToFloatConversion:
    """``DoubleToFloat`` lowers to a single ``Cvtsd2ss`` instruction."""

    def test_double_to_float_emits_cvtsd2ss(self) -> None:
        """``DoubleToFloat`` emits a single ``cvtsd2ss``."""
        st = _make_symbol_table(d=DoubleType, f=FloatType)
        instrs = _fn_instructions_with_table(_double_to_float_program(), st)
        cvts = [i for i in instrs if isinstance(i, asm.Cvtsd2ss)]
        assert len(cvts) == 1


# ---------------------------------------------------------------------------
# Instruction fixup - float conversion instructions with memory destinations
# ---------------------------------------------------------------------------


def _float_static(name: str) -> tac.StaticVariable:
    """Return a zero-valued float ``tac.StaticVariable`` with the given name.

    Args:
        name: The identifier used in the TAC program.

    Returns:
        A ``tac.StaticVariable`` of ``FloatType`` initialised to ``0.0``.
    """
    return tac.StaticVariable(name=name, globl=True, t=FloatType(), init=(tac.FloatInit(0.0),))


def _double_static(name: str) -> tac.StaticVariable:
    """Return a zero-valued double ``tac.StaticVariable`` with the given name.

    Args:
        name: The identifier used in the TAC program.

    Returns:
        A ``tac.StaticVariable`` of ``DoubleType`` initialised to ``0.0``.
    """
    return tac.StaticVariable(name=name, globl=True, t=DoubleType(), init=(tac.DoubleInit(0.0),))


def _int_static(name: str) -> tac.StaticVariable:
    """Return a zero-valued int ``tac.StaticVariable`` with the given name.

    Args:
        name: The identifier used in the TAC program.

    Returns:
        A ``tac.StaticVariable`` of ``IntType`` initialised to ``0``.
    """
    return tac.StaticVariable(name=name, globl=True, t=IntType(), init=(tac.IntInit(0),))


class TestCvtss2sdFixup:
    """``_fix_instructions`` routes ``Cvtss2sd`` through ``%xmm14`` when the destination is memory."""

    def test_cvtss2sd_no_mem_dst_after_fixup(self) -> None:
        """``FloatToDouble`` with static (memory) destination never emits ``cvtss2sd`` with a memory dst."""
        program = tac.Program(
            top_level=(
                _float_static("f"),
                _double_static("d"),
                tac.Function(
                    name="main",
                    body=(
                        tac.FloatToDouble(src=tac.Var("f"), dst=tac.Var("d")),
                        tac.Return(val=tac.Constant(ConstInt(0))),
                    ),
                ),
            )
        )
        for instr in _fn_instructions(program):
            if isinstance(instr, asm.Cvtss2sd):
                assert not isinstance(instr.dst, (asm.Memory, asm.Data)), "Cvtss2sd survived with memory dst"


class TestCvtsd2ssFixup:
    """``_fix_instructions`` routes ``Cvtsd2ss`` through ``%xmm14`` when the destination is memory."""

    def test_cvtsd2ss_no_mem_dst_after_fixup(self) -> None:
        """``DoubleToFloat`` with static (memory) destination never emits ``cvtsd2ss`` with a memory dst."""
        program = tac.Program(
            top_level=(
                _double_static("d"),
                _float_static("f"),
                tac.Function(
                    name="main",
                    body=(
                        tac.DoubleToFloat(src=tac.Var("d"), dst=tac.Var("f")),
                        tac.Return(val=tac.Constant(ConstInt(0))),
                    ),
                ),
            )
        )
        for instr in _fn_instructions(program):
            if isinstance(instr, asm.Cvtsd2ss):
                assert not isinstance(instr.dst, (asm.Memory, asm.Data)), "Cvtsd2ss survived with memory dst"


class TestCvttss2siFixup:
    """``_fix_instructions`` routes ``Cvttss2si`` through ``%r11`` when the destination is memory."""

    def test_cvttss2si_no_mem_dst_after_fixup(self) -> None:
        """``FloatToInt`` with static (memory) destination never emits ``cvttss2si`` with a memory dst."""
        program = tac.Program(
            top_level=(
                _float_static("f"),
                _int_static("x"),
                tac.Function(
                    name="main",
                    body=(
                        tac.FloatToInt(src=tac.Var("f"), dst=tac.Var("x")),
                        tac.Return(val=tac.Constant(ConstInt(0))),
                    ),
                ),
            )
        )
        for instr in _fn_instructions(program):
            if isinstance(instr, asm.Cvttss2si):
                assert not isinstance(instr.dst, (asm.Memory, asm.Data)), "Cvttss2si survived with memory dst"


class TestCvtsi2ssFixup:
    """``_fix_instructions`` routes ``Cvtsi2ss`` through ``%xmm14`` when the destination is memory."""

    def test_cvtsi2ss_no_mem_dst_after_fixup(self) -> None:
        """``IntToFloat`` with static (memory) destination never emits ``cvtsi2ss`` with a memory dst."""
        program = tac.Program(
            top_level=(
                _int_static("x"),
                _float_static("f"),
                tac.Function(
                    name="main",
                    body=(
                        tac.IntToFloat(src=tac.Var("x"), dst=tac.Var("f")),
                        tac.Return(val=tac.Constant(ConstInt(0))),
                    ),
                ),
            )
        )
        for instr in _fn_instructions(program):
            if isinstance(instr, asm.Cvtsi2ss):
                assert not isinstance(instr.dst, (asm.Memory, asm.Data)), "Cvtsi2ss survived with memory dst"


# ---------------------------------------------------------------------------
# Union code generation
# ---------------------------------------------------------------------------


class TestUnionCodegen:
    """Union types are handled like structs by the code generator."""

    def test_copy_from_offset_with_union_type_generates_assembly(self) -> None:
        """CopyFromOffset on a union-typed variable generates Mov instructions without error."""
        union_tag = ".UFoo.0"
        union_table = {
            union_tag: StructEntry(
                alignment=4,
                size=4,
                members=(
                    MemberEntry(name="x", member_type=IntType(), offset=0),
                    MemberEntry(name="y", member_type=IntType(), offset=0),
                ),
            )
        }
        sym: SymbolTable = {"u": SymbolTableEntry(t=Union(union_tag), attrs=LocalAttr())}
        program = tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        tac.CopyFromOffset(src="u", offset=0, dst=tac.Var("tmp")),
                        tac.Return(val=tac.Var("tmp")),
                    ),
                ),
            )
        )
        result = generate(program, sym, union_table)
        assert isinstance(result, asm.Program)

    def test_copy_to_offset_with_union_type_generates_assembly(self) -> None:
        """CopyToOffset on a union-typed variable generates Mov instructions without error."""
        union_tag = ".UFoo.0"
        union_table = {
            union_tag: StructEntry(
                alignment=4, size=4, members=(MemberEntry(name="x", member_type=IntType(), offset=0),)
            )
        }
        sym: SymbolTable = {"u": SymbolTableEntry(t=Union(union_tag), attrs=LocalAttr())}
        program = tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        tac.CopyToOffset(src=tac.Constant(ConstInt(42)), dst="u", offset=0),
                        tac.Return(val=tac.Constant(ConstInt(0))),
                    ),
                ),
            )
        )
        result = generate(program, sym, union_table)
        assert isinstance(result, asm.Program)

    def test_copy_between_union_vars_uses_chunked_moves(self) -> None:
        """Copy(src=union_var, dst=union_var) generates chunked moves, not a single scalar Mov."""
        union_tag = ".UFoo.0"
        # Union of size 8: long member dominates.
        union_table = {
            union_tag: StructEntry(
                alignment=8,
                size=8,
                members=(
                    MemberEntry(name="x", member_type=LongType(), offset=0),
                    MemberEntry(name="y", member_type=IntType(), offset=0),
                ),
            )
        }
        sym: SymbolTable = {
            "u1": SymbolTableEntry(t=Union(union_tag), attrs=LocalAttr()),
            "u2": SymbolTableEntry(t=Union(union_tag), attrs=LocalAttr()),
        }
        program = tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        tac.Copy(src=tac.Var("u1"), dst=tac.Var("u2")),
                        tac.Return(val=tac.Constant(ConstInt(0))),
                    ),
                ),
            )
        )
        result = generate(program, sym, union_table)
        instrs = next(d for d in result.definitions if isinstance(d, asm.Function)).instructions
        # Chunked copy generates two Mov pairs (one for each 8-byte chunk).
        # Without fix: Copy falls through to scalar path -> single Mov(Longword) - wrong size.
        movs = [i for i in instrs if isinstance(i, asm.Mov)]
        # At least one move must be Quadword (8-byte) to copy the full 8-byte union.
        assert any(m.assembly_type == asm.Quadword() for m in movs), (
            "Copy of 8-byte union must generate at least one QWORD move"
        )


# ---------------------------------------------------------------------------
# Word assembly type and short/int128 code generation
# ---------------------------------------------------------------------------


class TestAsmTypeShort:
    """``_asm_type`` returns ``Word`` for short-typed TAC values."""

    def test_const_short_returns_word(self) -> None:
        """``ConstShort`` constant maps to ``Word`` assembly type."""
        assert _asm_type(tac.Constant(value=ConstShort(42))) == asm.Word()

    def test_const_ushort_returns_word(self) -> None:
        """``ConstUShort`` constant maps to ``Word`` assembly type."""
        assert _asm_type(tac.Constant(value=ConstUShort(65535))) == asm.Word()

    def test_short_var_returns_word(self) -> None:
        """``ShortType`` variable maps to ``Word`` assembly type."""
        st = _make_symbol_table(s=ShortType)
        assert _asm_type(tac.Var("s"), st) == asm.Word()

    def test_ushort_var_returns_word(self) -> None:
        """``UShortType`` variable maps to ``Word`` assembly type."""
        st = _make_symbol_table(u=UShortType)
        assert _asm_type(tac.Var("u"), st) == asm.Word()

    def test_int_var_still_returns_longword(self) -> None:
        """``IntType`` variable still maps to ``Longword`` (regression guard)."""
        st = _make_symbol_table(x=IntType)
        assert _asm_type(tac.Var("x"), st) == asm.Longword()


class TestRegWidthWord:
    """``_reg_width`` returns ``Width.WORD`` for ``Word()``."""

    def test_word_returns_width_word(self) -> None:
        """``_reg_width(Word())`` returns ``Width.WORD``."""
        assert _reg_width(asm.Word()) == asm.Width.WORD

    def test_byte_still_returns_byte(self) -> None:
        """``_reg_width(Byte())`` still returns ``Width.BYTE`` (regression guard)."""
        assert _reg_width(asm.Byte()) == asm.Width.BYTE

    def test_longword_still_returns_dword(self) -> None:
        """``_reg_width(Longword())`` still returns ``Width.DWORD`` (regression guard)."""
        assert _reg_width(asm.Longword()) == asm.Width.DWORD


class TestTranslateStaticInitShortInt128:
    """``_translate_static_init`` converts short and __int128 TAC inits to asm."""

    def test_short_init_translates_to_asm_short_init(self) -> None:
        """``tac.ShortInit(42)`` maps to ``asm.ShortInit(42)``."""
        assert _translate_static_init(tac.ShortInit(42)) == asm.ShortInit(42)

    def test_ushort_init_translates_to_asm_ushort_init(self) -> None:
        """``tac.UShortInit(65535)`` maps to ``asm.UShortInit(65535)``."""
        assert _translate_static_init(tac.UShortInit(65535)) == asm.UShortInit(65535)

    def test_int128_init_translates_to_asm_int128_init(self) -> None:
        """``tac.Int128Init(big)`` maps to ``asm.Int128Init(big)``."""
        big = (1 << 100) + 7
        assert _translate_static_init(tac.Int128Init(big)) == asm.Int128Init(big)

    def test_uint128_init_translates_to_asm_uint128_init(self) -> None:
        """``tac.UInt128Init(big)`` maps to ``asm.UInt128Init(big)``."""
        big = (1 << 127) + 99
        assert _translate_static_init(tac.UInt128Init(big)) == asm.UInt128Init(big)


class TestAsmTypeLongLong:
    """``_asm_type`` returns ``Quadword`` for ``long long``-typed TAC values."""

    def test_const_longlong_returns_quadword(self) -> None:
        """``ConstLongLong`` constant maps to ``Quadword`` assembly type."""
        assert _asm_type(tac.Constant(value=ConstLongLong(42))) == asm.Quadword()

    def test_const_ulonglong_returns_quadword(self) -> None:
        """``ConstULongLong`` constant maps to ``Quadword`` assembly type."""
        assert _asm_type(tac.Constant(value=ConstULongLong(100))) == asm.Quadword()

    def test_longlong_var_returns_quadword(self) -> None:
        """``LongLongType`` variable maps to ``Quadword`` assembly type."""
        st = _make_symbol_table(ll=LongLongType)
        assert _asm_type(tac.Var("ll"), st) == asm.Quadword()

    def test_ulonglong_var_returns_quadword(self) -> None:
        """``ULongLongType`` variable maps to ``Quadword`` assembly type."""
        st = _make_symbol_table(ull=ULongLongType)
        assert _asm_type(tac.Var("ull"), st) == asm.Quadword()


class TestIsUnsignedLongLong:
    """``_is_unsigned`` correctly classifies ``long long`` and ``unsigned long long``."""

    def test_const_ulonglong_is_unsigned(self) -> None:
        """``ConstULongLong`` is classified as unsigned."""
        assert _is_unsigned(tac.Constant(value=ConstULongLong(0))) is True

    def test_const_longlong_is_not_unsigned(self) -> None:
        """``ConstLongLong`` is classified as signed."""
        assert _is_unsigned(tac.Constant(value=ConstLongLong(0))) is False

    def test_ulonglong_var_is_unsigned(self) -> None:
        """``ULongLongType`` variable is classified as unsigned."""
        st = _make_symbol_table(ull=ULongLongType)
        assert _is_unsigned(tac.Var("ull"), st) is True

    def test_longlong_var_is_not_unsigned(self) -> None:
        """``LongLongType`` variable is classified as signed."""
        st = _make_symbol_table(ll=LongLongType)
        assert _is_unsigned(tac.Var("ll"), st) is False


# ---------------------------------------------------------------------------
# Long double code generation
# ---------------------------------------------------------------------------


def _ld_binary_program(op: tac.BinaryOperator) -> tac.Program:
    """Build a TAC program performing a long-double binary operation.

    Args:
        op: The binary operator to apply.

    Returns:
        A ``tac.Program`` computing ``r = a op b`` where ``a``, ``b``, and ``r``
        are long-double variables.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.Binary(operator=op, src1=tac.Var("a"), src2=tac.Var("b"), dst=tac.Var("r")),
                    tac.Return(val=tac.Var("r")),
                ),
            ),
        )
    )


def _ld_negate_program() -> tac.Program:
    """Build a TAC program that negates a long-double variable.

    Returns:
        A ``tac.Program`` computing ``r = -a`` where both are long-double.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(
                    tac.Unary(operator=tac.Negate(), src=tac.Var("a"), dst=tac.Var("r")),
                    tac.Return(val=tac.Var("r")),
                ),
            ),
        )
    )


def _ld_return_program() -> tac.Program:
    """Build a TAC program that returns a long-double variable.

    Returns:
        A ``tac.Program`` whose sole function returns a long-double variable.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(tac.Return(val=tac.Var("x")),),
            ),
        )
    )


def _ld_symbol_table(**extra: type) -> SymbolTable:
    """Build a symbol table with ``a``, ``b``, ``r``, ``x`` as ``LongDoubleType``.

    Args:
        **extra: Additional variable names mapped to their C type classes.

    Returns:
        A ``SymbolTable`` with long-double entries for standard LD test vars.
    """
    base = {n: SymbolTableEntry(t=LongDoubleType(), attrs=LocalAttr()) for n in ("a", "b", "r", "x")}
    extra_entries = {name: SymbolTableEntry(t=t(), attrs=LocalAttr()) for name, t in extra.items()}
    return {**base, **extra_entries}


class TestLongDoubleAsmType:
    """``_asm_type`` returns ``LongDouble()`` for long-double values."""

    def test_const_long_double_returns_long_double(self) -> None:
        """``ConstLongDouble`` constant maps to ``LongDouble()`` assembly type."""
        assert _asm_type(tac.Constant(value=ConstLongDouble(1.5))) == asm.LongDouble()

    def test_long_double_var_returns_long_double(self) -> None:
        """A ``LongDoubleType`` variable maps to ``LongDouble()``."""
        st = _ld_symbol_table()
        assert _asm_type(tac.Var("x"), st) == asm.LongDouble()


class TestLongDoubleTypeSizeAndAlign:
    """``_type_size`` and ``_static_alignment`` return 16 for ``LongDoubleType``."""

    def test_type_size_is_16(self) -> None:
        """``_type_size(LongDoubleType())`` returns 16."""
        assert _type_size(LongDoubleType()) == 16

    def test_static_alignment_is_16(self) -> None:
        """``_static_alignment(LongDoubleType())`` returns 16."""
        assert _static_alignment(LongDoubleType()) == 16


class TestLongDoubleStaticInit:
    """``_translate_static_init`` maps ``tac.LongDoubleInit`` to ``asm.LongDoubleInit``."""

    def test_long_double_static_init(self) -> None:
        """``tac.LongDoubleInit(3.14)`` translates to ``asm.LongDoubleInit(3.14)``."""
        result = _translate_static_init(tac.LongDoubleInit(3.14))
        assert isinstance(result, asm.LongDoubleInit)
        assert result.value == 3.14


class TestLongDoubleBinaryCodeGen:
    """Long-double binary operations use x87 FPU instructions."""

    def test_long_double_add_emits_fldt_fldt_faddp_fstpt(self) -> None:
        """Long-double addition emits ``Fldt``, ``Fldt``, ``Faddp``, ``Fstpt``."""
        st = _ld_symbol_table()
        instrs = _fn_instructions_with_table(_ld_binary_program(tac.Add()), st)
        assert any(isinstance(i, asm.Faddp) for i in instrs)
        assert any(isinstance(i, asm.Fldt) for i in instrs)
        assert any(isinstance(i, asm.Fstpt) for i in instrs)

    def test_long_double_sub_emits_fsubp(self) -> None:
        """Long-double subtraction emits ``Fsubp`` (which emits ``fsubrp``)."""
        st = _ld_symbol_table()
        instrs = _fn_instructions_with_table(_ld_binary_program(tac.Subtract()), st)
        assert any(isinstance(i, asm.Fsubp) for i in instrs)

    def test_long_double_mul_emits_fmulp(self) -> None:
        """Long-double multiplication emits ``Fmulp``."""
        st = _ld_symbol_table()
        instrs = _fn_instructions_with_table(_ld_binary_program(tac.Multiply()), st)
        assert any(isinstance(i, asm.Fmulp) for i in instrs)

    def test_long_double_div_emits_fdivp(self) -> None:
        """Long-double division emits ``Fdivp`` (which emits ``fdivrp``)."""
        st = _ld_symbol_table()
        instrs = _fn_instructions_with_table(_ld_binary_program(tac.Divide()), st)
        assert any(isinstance(i, asm.Fdivp) for i in instrs)

    def test_long_double_binary_no_mov(self) -> None:
        """Long-double binary operations do not emit a Mov for double/float accumulator copy."""
        st = _ld_symbol_table()
        instrs = _fn_instructions_with_table(_ld_binary_program(tac.Add()), st)
        assert not any(isinstance(i, asm.Binary) for i in instrs), "LD binary must not emit asm.Binary"


class TestLongDoubleNegateCodeGen:
    """Long-double negation uses the x87 ``fchs`` instruction."""

    def test_long_double_negate_emits_fldt_fchs_fstpt(self) -> None:
        """``Negate`` on a long-double emits ``Fldt``, ``Fchs``, ``Fstpt``."""
        st = _ld_symbol_table()
        instrs = _fn_instructions_with_table(_ld_negate_program(), st)
        assert any(isinstance(i, asm.Fldt) for i in instrs)
        assert any(isinstance(i, asm.Fchs) for i in instrs)
        assert any(isinstance(i, asm.Fstpt) for i in instrs)


class TestLongDoubleReturnCodeGen:
    """Returning a long-double value loads it into ST0 then ``ret``."""

    def test_long_double_return_emits_fldt_then_ret(self) -> None:
        """``Return(long_double_var)`` emits ``Fldt`` followed by ``Ret``."""
        st = _ld_symbol_table()
        instrs = _fn_instructions_with_table(_ld_return_program(), st)
        idx_fldt = next((i for i, ins in enumerate(instrs) if isinstance(ins, asm.Fldt)), -1)
        idx_ret = next((i for i, ins in enumerate(instrs) if isinstance(ins, asm.Ret)), -1)
        assert idx_fldt >= 0, "Fldt not found"
        assert idx_ret > idx_fldt, "Ret must follow Fldt"


class TestLongDoubleComparisonCodeGen:
    """Long-double comparisons use ``fucomip`` via ``Fucomip`` instruction."""

    def test_long_double_equal_emits_fucomip_and_fstp0(self) -> None:
        """Equal comparison emits ``Fucomip`` and ``Fstp0`` to clean up x87 stack."""
        st = _ld_symbol_table(**{"res": IntType})
        instrs = _fn_instructions_with_table(_ld_binary_program(tac.Equal()), st)
        assert any(isinstance(i, asm.Fucomip) for i in instrs)
        assert any(isinstance(i, asm.Fstp0) for i in instrs)

    def test_long_double_lt_emits_setb(self) -> None:
        """LessThan comparison sets CF via fucomip; result uses ``SetCC(B)``."""
        st = _ld_symbol_table(**{"res": IntType})
        instrs = _fn_instructions_with_table(_ld_binary_program(tac.LessThan()), st)
        assert any(isinstance(i, asm.SetCC) and i.cond_code == asm.CondCodes.B for i in instrs)

    def test_long_double_gt_emits_seta(self) -> None:
        """GreaterThan comparison uses ``SetCC(A)``."""
        st = _ld_symbol_table(**{"res": IntType})
        instrs = _fn_instructions_with_table(_ld_binary_program(tac.GreaterThan()), st)
        assert any(isinstance(i, asm.SetCC) and i.cond_code == asm.CondCodes.A for i in instrs)


class TestLongDoublePseudoReplacement:
    """Long-double ``Pseudo`` operands get a 16-byte, 16-byte-aligned stack slot."""

    def test_long_double_stack_slot_is_16_bytes(self) -> None:
        """A long-double local variable occupies exactly 16 bytes of frame space."""
        st = _ld_symbol_table()
        program = tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        tac.Copy(src=tac.Var("a"), dst=tac.Var("r")),
                        tac.Return(val=tac.Constant(value=ConstInt(0))),
                    ),
                ),
            )
        )
        instrs = _fn_instructions_with_table(program, st)
        mem_offsets = {
            op.offset
            for i in instrs
            for op in ([getattr(i, "src", None), getattr(i, "dst", None)] + [getattr(i, "operand", None)])
            if op is not None and isinstance(op, asm.Memory)
        }
        # Slots should be 16-byte aligned (multiples of 16 below rbp).
        for off in mem_offsets:
            if off < 0:
                assert off % 16 == 0, f"Stack offset {off} is not 16-byte aligned"


class TestLongDoubleConversions:
    """Long-double conversion instructions map to x87 integer load/store instructions."""

    def _conv_program(self, instr_class: type, src_name: str, dst_name: str) -> tac.Program:
        """Build a single-conversion TAC program.

        Args:
            instr_class: The TAC conversion instruction class.
            src_name: Source variable name.
            dst_name: Destination variable name.

        Returns:
            A ``tac.Program`` wrapping the conversion.
        """
        return tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        instr_class(src=tac.Var(src_name), dst=tac.Var(dst_name)),
                        tac.Return(val=tac.Constant(value=ConstInt(0))),
                    ),
                ),
            )
        )

    def test_int_to_long_double_emits_fildl_fstpt(self) -> None:
        """``IntToLongDouble`` from int32 emits ``Fildl`` and ``Fstpt``."""
        st = {
            "i": SymbolTableEntry(t=IntType(), attrs=LocalAttr()),
            "ld": SymbolTableEntry(
                t=__import__("compiler.syntax.ast.types", fromlist=["LongDoubleType"]).LongDoubleType(),
                attrs=LocalAttr(),
            ),
        }
        instrs = _fn_instructions_with_table(self._conv_program(tac.IntToLongDouble, "i", "ld"), st)
        assert any(isinstance(i, asm.Fildl) for i in instrs)
        assert any(isinstance(i, asm.Fstpt) for i in instrs)

    def test_long_to_long_double_emits_fildq_fstpt(self) -> None:
        """``IntToLongDouble`` from int64 emits ``Fildq`` and ``Fstpt``."""
        st = {
            "i": SymbolTableEntry(t=LongType(), attrs=LocalAttr()),
            "ld": SymbolTableEntry(
                t=__import__("compiler.syntax.ast.types", fromlist=["LongDoubleType"]).LongDoubleType(),
                attrs=LocalAttr(),
            ),
        }
        instrs = _fn_instructions_with_table(self._conv_program(tac.IntToLongDouble, "i", "ld"), st)
        assert any(isinstance(i, asm.Fildq) for i in instrs)
        assert any(isinstance(i, asm.Fstpt) for i in instrs)

    def test_long_double_to_int_emits_fldt_fisttpl(self) -> None:
        """``LongDoubleToInt`` to int32 emits ``Fldt`` and ``Fisttpl``."""
        st = {
            "ld": SymbolTableEntry(
                t=__import__("compiler.syntax.ast.types", fromlist=["LongDoubleType"]).LongDoubleType(),
                attrs=LocalAttr(),
            ),
            "i": SymbolTableEntry(t=IntType(), attrs=LocalAttr()),
        }
        instrs = _fn_instructions_with_table(self._conv_program(tac.LongDoubleToInt, "ld", "i"), st)
        assert any(isinstance(i, asm.Fldt) for i in instrs)
        assert any(isinstance(i, asm.Fisttpl) for i in instrs)

    def test_long_double_to_long_emits_fldt_fisttpq(self) -> None:
        """``LongDoubleToInt`` to int64 emits ``Fldt`` and ``Fisttpq``."""
        st = {
            "ld": SymbolTableEntry(
                t=__import__("compiler.syntax.ast.types", fromlist=["LongDoubleType"]).LongDoubleType(),
                attrs=LocalAttr(),
            ),
            "i": SymbolTableEntry(t=LongType(), attrs=LocalAttr()),
        }
        instrs = _fn_instructions_with_table(self._conv_program(tac.LongDoubleToInt, "ld", "i"), st)
        assert any(isinstance(i, asm.Fldt) for i in instrs)
        assert any(isinstance(i, asm.Fisttpq) for i in instrs)

    def test_double_to_long_double_emits_fldl_fstpt(self) -> None:
        """``DoubleToLongDouble`` emits ``Fldl`` and ``Fstpt``."""
        st = {
            "d": SymbolTableEntry(t=DoubleType(), attrs=LocalAttr()),
            "ld": SymbolTableEntry(
                t=__import__("compiler.syntax.ast.types", fromlist=["LongDoubleType"]).LongDoubleType(),
                attrs=LocalAttr(),
            ),
        }
        instrs = _fn_instructions_with_table(self._conv_program(tac.DoubleToLongDouble, "d", "ld"), st)
        assert any(isinstance(i, asm.Fldl) for i in instrs)
        assert any(isinstance(i, asm.Fstpt) for i in instrs)

    def test_long_double_to_double_emits_fldt_fstpl(self) -> None:
        """``LongDoubleToDouble`` emits ``Fldt`` and ``Fstpl``."""
        st = {
            "ld": SymbolTableEntry(
                t=__import__("compiler.syntax.ast.types", fromlist=["LongDoubleType"]).LongDoubleType(),
                attrs=LocalAttr(),
            ),
            "d": SymbolTableEntry(t=DoubleType(), attrs=LocalAttr()),
        }
        instrs = _fn_instructions_with_table(self._conv_program(tac.LongDoubleToDouble, "ld", "d"), st)
        assert any(isinstance(i, asm.Fldt) for i in instrs)
        assert any(isinstance(i, asm.Fstpl) for i in instrs)

    def test_float_to_long_double_emits_flds_fstpt(self) -> None:
        """``FloatToLongDouble`` emits ``Flds`` and ``Fstpt``."""
        st = {
            "f": SymbolTableEntry(t=FloatType(), attrs=LocalAttr()),
            "ld": SymbolTableEntry(
                t=__import__("compiler.syntax.ast.types", fromlist=["LongDoubleType"]).LongDoubleType(),
                attrs=LocalAttr(),
            ),
        }
        instrs = _fn_instructions_with_table(self._conv_program(tac.FloatToLongDouble, "f", "ld"), st)
        assert any(isinstance(i, asm.Flds) for i in instrs)
        assert any(isinstance(i, asm.Fstpt) for i in instrs)

    def test_long_double_to_float_emits_fldt_fstps(self) -> None:
        """``LongDoubleToFloat`` emits ``Fldt`` and ``Fstps``."""
        st = {
            "ld": SymbolTableEntry(
                t=__import__("compiler.syntax.ast.types", fromlist=["LongDoubleType"]).LongDoubleType(),
                attrs=LocalAttr(),
            ),
            "f": SymbolTableEntry(t=FloatType(), attrs=LocalAttr()),
        }
        instrs = _fn_instructions_with_table(self._conv_program(tac.LongDoubleToFloat, "ld", "f"), st)
        assert any(isinstance(i, asm.Fldt) for i in instrs)
        assert any(isinstance(i, asm.Fstps) for i in instrs)


class TestLongDoubleParam:
    """Long-double function parameters arrive via MEMORY class and are copied with x87."""

    def _ld_param_program(self) -> tac.Program:
        """A function ``foo(long double x)`` that returns ``x``."""
        return tac.Program(
            top_level=(
                tac.Function(
                    name="foo",
                    params=("x",),
                    body=(tac.Return(val=tac.Var("x")),),
                ),
            )
        )

    def _ld_param_st(self) -> SymbolTable:
        """Symbol table: ``x`` is ``LongDoubleType`` local."""
        return {"x": SymbolTableEntry(t=LongDoubleType(), attrs=LocalAttr())}

    def test_ld_param_uses_fldt(self) -> None:
        """A long-double parameter is loaded from the stack with ``Fldt``."""
        instrs = _fn_instructions_with_table(self._ld_param_program(), self._ld_param_st())
        assert any(isinstance(i, asm.Fldt) for i in instrs), "expected Fldt to load ld param from stack"

    def test_ld_param_uses_fstpt(self) -> None:
        """A long-double parameter is stored to its local slot with ``Fstpt``."""
        instrs = _fn_instructions_with_table(self._ld_param_program(), self._ld_param_st())
        assert any(isinstance(i, asm.Fstpt) for i in instrs), "expected Fstpt to store ld param to local"

    def test_ld_param_no_mov(self) -> None:
        """No ``Mov`` is emitted for the long-double parameter copy (x87 handles it)."""

        instrs = _fn_instructions_with_table(self._ld_param_program(), self._ld_param_st())
        param_movs = [i for i in instrs if isinstance(i, asm.Mov) and i.assembly_type == asm.LongDouble()]
        assert not param_movs, "Mov(LongDouble) should not be emitted for ld param"


class TestLongDoubleFuncall:
    """Long-double arguments are pushed as two 8-byte stack slots; returns via ST0."""

    def _ld_arg_program(self) -> tac.Program:
        """Call ``foo(x)`` where ``x`` is a long double, store result in ``r``."""
        return tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        tac.FunCall(
                            fun_val=tac.Var("foo"),
                            args=(tac.Var("x"),),
                            dst=tac.Var("r"),
                        ),
                        tac.Return(val=tac.Var("r")),
                    ),
                ),
            )
        )

    def _ld_arg_st(self, ret_type: type | None = None) -> SymbolTable:
        """Symbol table: ``x`` is LongDoubleType local; ``foo`` has the given return type."""
        ret = (ret_type or IntType)()
        st: SymbolTable = {
            "x": SymbolTableEntry(t=LongDoubleType(), attrs=LocalAttr()),
            "r": SymbolTableEntry(t=ret, attrs=LocalAttr()),
            "foo": SymbolTableEntry(
                t=FunType(params=[LongDoubleType()], ret=ret),
                attrs=FunAttr(defined=False, globl=True),
            ),
        }
        return st

    def test_ld_arg_pushes_two_qwords(self) -> None:
        """A long-double argument emits two ``Push`` instructions (16 bytes on stack)."""
        instrs = _fn_instructions_with_table(self._ld_arg_program(), self._ld_arg_st())
        pushes = [i for i in instrs if isinstance(i, asm.Push)]
        assert len(pushes) == 2, f"expected 2 Push for ld arg, got {len(pushes)}"

    def test_ld_return_emits_fstpt(self) -> None:
        """A call returning long double emits ``Fstpt`` to capture ST0 after the call."""
        instrs = _fn_instructions_with_table(self._ld_arg_program(), self._ld_arg_st(LongDoubleType))
        call_idx = next(i for i, instr in enumerate(instrs) if isinstance(instr, asm.Call))
        post_call = instrs[call_idx + 1 :]
        assert any(isinstance(i, asm.Fstpt) for i in post_call), "expected Fstpt after Call for ld return"


# ---------------------------------------------------------------------------
# Variadic call - %al register
# ---------------------------------------------------------------------------


class TestVariadicCallAlRegister:
    """Calls to variadic functions must set ``%al`` = number of XMM args used."""

    def _variadic_fn_st(self, name: str = "printf") -> SymbolTable:
        """Return a symbol table marking *name* as a variadic function."""
        return {
            name: SymbolTableEntry(
                t=FunType(params=(IntType(),), ret=IntType(), variadic=True),
                attrs=FunAttr(defined=False, globl=True),
            )
        }

    def _variadic_call_program(self, *args: tac.Val, name: str = "printf") -> tac.Program:
        """Build a program that calls variadic *name* with the given args."""
        return tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        tac.FunCall(
                            fun_val=tac.Var(name),
                            args=tuple(args),
                            dst=tac.Var("result"),
                        ),
                        tac.Return(val=tac.Constant(value=ConstInt(0))),
                    ),
                ),
            )
        )

    def test_variadic_call_with_no_fp_args_sets_al_to_zero(self) -> None:
        """Calling a variadic fn with no floating-point args emits ``movb $0, %al``."""
        program = self._variadic_call_program(tac.Constant(ConstInt(42)), name="printf")
        instrs = _fn_instructions_with_table(program, self._variadic_fn_st("printf"))
        call_idx = next(i for i, instr in enumerate(instrs) if isinstance(instr, asm.Call))
        pre_call = instrs[:call_idx]
        al_reg = asm.Register(asm.GPRegisters.AX, asm.Width.BYTE)
        al_movs = [i for i in pre_call if isinstance(i, asm.Mov) and i.dst == al_reg]
        assert len(al_movs) == 1, "expected exactly one movb targeting %al before the call"
        assert al_movs[0].src == asm.Imm(0), f"expected movb $0, %al but got {al_movs[0].src}"

    def test_variadic_call_with_one_fp_arg_sets_al_to_one(self) -> None:
        """Calling a variadic fn with one double arg emits ``movb $1, %al``."""
        program = self._variadic_call_program(
            tac.Constant(ConstInt(1)),
            tac.Constant(ConstDouble(3.14)),
            name="printf",
        )
        instrs = _fn_instructions_with_table(program, self._variadic_fn_st("printf"))
        call_idx = next(i for i, instr in enumerate(instrs) if isinstance(instr, asm.Call))
        pre_call = instrs[:call_idx]
        al_reg = asm.Register(asm.GPRegisters.AX, asm.Width.BYTE)
        al_movs = [i for i in pre_call if isinstance(i, asm.Mov) and i.dst == al_reg]
        assert len(al_movs) == 1, "expected exactly one movb targeting %al before the call"
        assert al_movs[0].src == asm.Imm(1), f"expected movb $1, %al but got {al_movs[0].src}"

    def test_non_variadic_call_does_not_set_al(self) -> None:
        """Calling a regular (non-variadic) fn must NOT emit a ``movb $N, %al``."""
        st: SymbolTable = {
            "foo": SymbolTableEntry(
                t=FunType(params=(IntType(),), ret=IntType(), variadic=False),
                attrs=FunAttr(defined=False, globl=True),
            )
        }
        program = self._variadic_call_program(tac.Constant(ConstInt(1)), name="foo")
        instrs = _fn_instructions_with_table(program, st)
        al_reg = asm.Register(asm.GPRegisters.AX, asm.Width.BYTE)
        al_movs = [i for i in instrs if isinstance(i, asm.Mov) and i.dst == al_reg]
        assert not al_movs, "non-variadic call must not set %al"


# ---------------------------------------------------------------------------
# Function pointer indirect call code generation
# ---------------------------------------------------------------------------


class TestFunctionPointerIndirectCall:
    """A ``FunCall`` whose ``fun_val`` names a ``Pointer(FunType(...))`` variable
    must emit an ``IndirectCall`` instruction, not a direct ``Call``.
    """

    def _fp_call_program(self, fp_name: str = "fp") -> tac.Program:
        """Build a ``tac.Program`` that calls through a function pointer variable.

        Args:
            fp_name: Name of the function pointer variable used as ``fun_val``.

        Returns:
            A minimal program whose sole function calls ``fp_name`` with one integer arg.
        """
        return tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        tac.FunCall(
                            fun_val=tac.Var(fp_name),
                            args=(tac.Constant(ConstInt(1)),),
                            dst=tac.Var("result"),
                        ),
                        tac.Return(val=tac.Var("result")),
                    ),
                ),
            )
        )

    def _fp_symbol_table(self, fp_name: str = "fp") -> SymbolTable:
        """Build a symbol table marking *fp_name* as a local ``Pointer(FunType)`` variable.

        Args:
            fp_name: The function pointer variable name.

        Returns:
            A ``SymbolTable`` with a ``LocalAttr`` entry for *fp_name* and ``result``.
        """
        return {
            fp_name: SymbolTableEntry(
                t=Pointer(FunType(params=(IntType(),), ret=IntType())),
                attrs=LocalAttr(),
            ),
            "result": SymbolTableEntry(t=IntType(), attrs=LocalAttr()),
        }

    def test_function_pointer_call_emits_indirect_call(self) -> None:
        """``FunCall`` through a ``Pointer(FunType)`` local variable emits ``IndirectCall``."""
        instrs = _fn_instructions_with_table(self._fp_call_program(), self._fp_symbol_table())
        indirect = [i for i in instrs if isinstance(i, asm.IndirectCall)]
        assert len(indirect) == 1, f"expected one IndirectCall, got {len(indirect)}"

    def test_function_pointer_call_does_not_emit_direct_call(self) -> None:
        """A function pointer call must not emit a direct ``Call`` instruction."""
        instrs = _fn_instructions_with_table(self._fp_call_program(), self._fp_symbol_table())
        direct = [i for i in instrs if isinstance(i, asm.Call)]
        assert not direct, f"expected no direct Call, got {direct}"

    def test_direct_function_call_still_emits_call(self) -> None:
        """A ``FunCall`` with a ``FunAttr`` callee still emits a direct ``Call`` (regression)."""
        st: SymbolTable = {
            "add": SymbolTableEntry(
                t=FunType(params=(IntType(),), ret=IntType()),
                attrs=FunAttr(defined=False, globl=True),
            ),
            "result": SymbolTableEntry(t=IntType(), attrs=LocalAttr()),
        }
        program = tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        tac.FunCall(
                            fun_val=tac.Var("add"),
                            args=(tac.Constant(ConstInt(1)),),
                            dst=tac.Var("result"),
                        ),
                        tac.Return(val=tac.Var("result")),
                    ),
                ),
            )
        )
        instrs = _fn_instructions_with_table(program, st)
        direct = [i for i in instrs if isinstance(i, asm.Call)]
        indirect = [i for i in instrs if isinstance(i, asm.IndirectCall)]
        assert len(direct) == 1, f"expected one direct Call, got {len(direct)}"
        assert not indirect, f"expected no IndirectCall, got {indirect}"

    def test_function_pointer_call_loads_pointer_into_r10(self) -> None:
        """The function pointer is loaded into ``%r10`` before the indirect call."""
        instrs = _fn_instructions_with_table(self._fp_call_program(), self._fp_symbol_table())
        r10q = asm.Register(asm.GPRegisters.R10, asm.Width.QWORD)
        call_idx = next(i for i, instr in enumerate(instrs) if isinstance(instr, asm.IndirectCall))
        pre_call = instrs[:call_idx]
        r10_loads = [i for i in pre_call if isinstance(i, asm.Mov) and i.dst == r10q]
        assert r10_loads, "expected a Mov into %r10 before the IndirectCall"

    def test_variadic_function_pointer_sets_al(self) -> None:
        """A call through a variadic ``Pointer(FunType(..., variadic=True))`` sets ``%al``."""
        fp_name = "vfp"
        st: SymbolTable = {
            fp_name: SymbolTableEntry(
                t=Pointer(FunType(params=(IntType(),), ret=IntType(), variadic=True)),
                attrs=LocalAttr(),
            ),
            "result": SymbolTableEntry(t=IntType(), attrs=LocalAttr()),
        }
        program = self._fp_call_program(fp_name)
        instrs = _fn_instructions_with_table(program, st)
        al_reg = asm.Register(asm.GPRegisters.AX, asm.Width.BYTE)
        al_movs = [i for i in instrs if isinstance(i, asm.Mov) and i.dst == al_reg]
        assert al_movs, "expected movb $N, %al for variadic function pointer call"


# ---------------------------------------------------------------------------
# Passing a function name as a function-pointer argument
# ---------------------------------------------------------------------------


class TestFunctionNameAsArgument:
    """When a function name (``FunAttr`` entry) is passed as a call argument it must
    be loaded via ``leaq`` (address-of), not ``movq`` (value-at), so the called
    function receives a valid code pointer rather than bytes of machine code.
    """

    def _apply_call_program(self) -> tac.Program:
        """Build a program that passes ``mul`` (a function name) as the first arg to ``apply``."""
        return tac.Program(
            top_level=(
                tac.Function(
                    name="main",
                    body=(
                        tac.FunCall(
                            fun_val=tac.Var("apply"),
                            args=(tac.Var("mul"), tac.Constant(ConstInt(3)), tac.Constant(ConstInt(4))),
                            dst=tac.Var("result"),
                        ),
                        tac.Return(val=tac.Var("result")),
                    ),
                ),
            )
        )

    def _apply_symbol_table(self) -> SymbolTable:
        """Symbol table: ``apply`` and ``mul`` are declared functions; ``result`` is a local."""
        return {
            "apply": SymbolTableEntry(
                t=FunType(
                    params=(Pointer(FunType(params=(IntType(), IntType()), ret=IntType())), IntType(), IntType()),
                    ret=IntType(),
                ),
                attrs=FunAttr(defined=False, globl=True),
            ),
            "mul": SymbolTableEntry(
                t=FunType(params=(IntType(), IntType()), ret=IntType()),
                attrs=FunAttr(defined=False, globl=True),
            ),
            "result": SymbolTableEntry(t=IntType(), attrs=LocalAttr()),
        }

    def test_function_name_arg_uses_lea_not_mov_data(self) -> None:
        """``mul`` passed as an argument must produce a ``Lea`` instruction, not a ``Mov(Data(...))``."""
        instrs = _fn_instructions_with_table(self._apply_call_program(), self._apply_symbol_table())
        mul_data = asm.Data("mul")
        bad_movs = [i for i in instrs if isinstance(i, asm.Mov) and i.src == mul_data]
        assert not bad_movs, f"found Mov(Data('mul'), ...) — should be Lea: {bad_movs}"

    def test_function_name_arg_emits_lea(self) -> None:
        """A ``Lea`` instruction loading the address of ``mul`` must appear before the call."""
        instrs = _fn_instructions_with_table(self._apply_call_program(), self._apply_symbol_table())
        call_idx = next(i for i, instr in enumerate(instrs) if isinstance(instr, asm.Call))
        pre_call = instrs[:call_idx]
        mul_data = asm.Data("mul")
        leas = [i for i in pre_call if isinstance(i, asm.Lea) and i.src == mul_data]
        assert leas, "expected Lea(Data('mul'), ...) before the call"


# ---------------------------------------------------------------------------
# Bit-field code generation
# ---------------------------------------------------------------------------


def _bf_program(instr: tac.Instruction) -> tac.Program:
    """Build a TAC program containing a single bit-field instruction plus a Return.

    Args:
        instr: The ``GetBitField`` or ``SetBitField`` instruction to test.

    Returns:
        A ``tac.Program`` wrapping the instruction in a minimal ``main`` function.
    """
    return tac.Program(
        top_level=(
            tac.Function(
                name="main",
                body=(instr, tac.Return(val=tac.Constant(ConstInt(0)))),
            ),
        )
    )


def _bf_sym(var_name: str) -> SymbolTable:
    """Build a minimal symbol table with a single 4-byte struct variable.

    Args:
        var_name: The name of the aggregate variable.

    Returns:
        A ``SymbolTable`` mapping ``var_name`` to a ``LocalAttr`` entry.
    """
    from compiler.syntax.ast.types import Structure

    return {var_name: SymbolTableEntry(t=Structure(".Sflags.0"), attrs=LocalAttr())}


class TestGetBitFieldCodegen:
    """``GetBitField`` lowers to shift-and-mask sequences in assembly."""

    def test_get_bit_field_generates_valid_assembly(self) -> None:
        """``GetBitField`` produces a valid ``asm.Program`` without error."""
        instr = tac.GetBitField(
            src="s", byte_offset=0, bit_offset=0, bit_width=3, unit_size=4, signed=False, dst=tac.Var("tmp")
        )
        result = generate(_bf_program(instr), _bf_sym("s"))
        assert isinstance(result, asm.Program)

    def test_get_unsigned_bit_field_uses_shr(self) -> None:
        """Unsigned ``GetBitField`` uses logical right shift (``Shr``), not ``Sar``."""
        instr = tac.GetBitField(
            src="s", byte_offset=0, bit_offset=3, bit_width=5, unit_size=4, signed=False, dst=tac.Var("tmp")
        )
        instrs = generate(_bf_program(instr), _bf_sym("s")).definitions[0].instructions
        assert any(isinstance(i, asm.Shr) for i in instrs), "expected Shr for unsigned bit-field read"
        assert not any(isinstance(i, asm.Sar) for i in instrs), "Sar must not appear for unsigned bit-field read"

    def test_get_signed_bit_field_uses_sar_for_sign_extension(self) -> None:
        """Signed ``GetBitField`` uses arithmetic right shift (``Sar``) for sign-extension."""
        instr = tac.GetBitField(
            src="s", byte_offset=0, bit_offset=3, bit_width=5, unit_size=4, signed=True, dst=tac.Var("tmp")
        )
        instrs = generate(_bf_program(instr), _bf_sym("s")).definitions[0].instructions
        assert any(isinstance(i, asm.Sar) for i in instrs), "expected Sar for sign-extension of signed bit-field"

    def test_get_bit_field_at_offset_zero_skips_initial_shift(self) -> None:
        """A bit-field at ``bit_offset=0`` does not emit a right-shift instruction."""
        instr = tac.GetBitField(
            src="s", byte_offset=0, bit_offset=0, bit_width=3, unit_size=4, signed=False, dst=tac.Var("tmp")
        )
        instrs = generate(_bf_program(instr), _bf_sym("s")).definitions[0].instructions
        assert not any(isinstance(i, asm.Shr) for i in instrs), "no Shr needed when bit_offset=0"


class TestSetBitFieldCodegen:
    """``SetBitField`` lowers to a read-modify-write sequence in assembly."""

    def test_set_bit_field_generates_valid_assembly(self) -> None:
        """``SetBitField`` produces a valid ``asm.Program`` without error."""
        instr = tac.SetBitField(
            src=tac.Constant(ConstInt(5)),
            dst="s",
            byte_offset=0,
            bit_offset=3,
            bit_width=5,
            unit_size=4,
        )
        result = generate(_bf_program(instr), _bf_sym("s"))
        assert isinstance(result, asm.Program)

    def test_set_bit_field_uses_shl_to_position_value(self) -> None:
        """``SetBitField`` emits ``Shl`` to shift the source value into the correct bit position."""
        instr = tac.SetBitField(
            src=tac.Constant(ConstInt(5)),
            dst="s",
            byte_offset=0,
            bit_offset=3,
            bit_width=5,
            unit_size=4,
        )
        instrs = generate(_bf_program(instr), _bf_sym("s")).definitions[0].instructions
        assert any(isinstance(i, asm.Shl) for i in instrs), "expected Shl to position src into bit field"

    def test_set_bit_field_at_offset_zero_skips_shl(self) -> None:
        """``SetBitField`` with ``bit_offset=0`` does not emit ``Shl`` for src positioning."""
        instr = tac.SetBitField(
            src=tac.Constant(ConstInt(5)),
            dst="s",
            byte_offset=0,
            bit_offset=0,
            bit_width=3,
            unit_size=4,
        )
        instrs = generate(_bf_program(instr), _bf_sym("s")).definitions[0].instructions
        assert not any(isinstance(i, asm.Shl) for i in instrs), "no Shl needed when bit_offset=0"


class TestAlignAsCodeGen:
    """``_Alignas`` alignment overrides are applied to static variable ``.balign`` directives."""

    def test_alignas_overrides_static_var_alignment(self) -> None:
        """``tac.StaticVariable(align_as=16)`` emits ``asm.StaticVariable(alignment=16)`` for an int."""
        prog = tac.Program(
            top_level=(tac.StaticVariable(name="x", globl=True, t=IntType(), init=(tac.IntInit(0),), align_as=16),)
        )
        asm_prog = generate(prog, {})
        sv = next(d for d in asm_prog.definitions if isinstance(d, asm.StaticVariable) and d.identifier == "x")
        assert sv.alignment == 16

    def test_no_alignas_uses_natural_alignment(self) -> None:
        """``tac.StaticVariable`` without ``align_as`` keeps the natural 4-byte alignment for int."""
        prog = tac.Program(top_level=(tac.StaticVariable(name="x", globl=True, t=IntType(), init=(tac.IntInit(0),)),))
        asm_prog = generate(prog, {})
        sv = next(d for d in asm_prog.definitions if isinstance(d, asm.StaticVariable) and d.identifier == "x")
        assert sv.alignment == 4

    def test_alignas_local_var_aligns_stack_slot(self) -> None:
        """A local int with ``align_as=16`` in the symbol table must be placed at a 16-byte-aligned offset."""
        from compiler.code_generator import _replace_pseudos
        from compiler.types.symbols import LocalAttr, SymbolTableEntry

        sym = {"x": SymbolTableEntry(t=IntType(), attrs=LocalAttr(), align_as=16)}
        pseudo_fn = asm.Function(
            name="main",
            globl=False,
            instructions=(asm.Mov(src=asm.Pseudo("x"), dst=asm.Pseudo("x"), assembly_type=asm.Longword()),),
            gp_callee_saved=(),
        )
        resolved_fn, _ = _replace_pseudos(pseudo_fn, frozenset(), symbol_table=sym)
        mem_ops = [
            op
            for instr in resolved_fn.instructions
            for op in [getattr(instr, "src", None), getattr(instr, "dst", None)]
            if isinstance(op, asm.Memory)
        ]
        assert mem_ops, "expected at least one Memory operand"
        for mem in mem_ops:
            assert mem.offset % 16 == 0, f"stack offset {mem.offset} not 16-byte aligned"
