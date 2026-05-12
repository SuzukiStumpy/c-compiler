"""Tests for the code generation stage.

Verifies that ``generate`` correctly translates each TAC construct into the
equivalent assembly-AST nodes, and that the pseudo-replacement and instruction
fixup passes produce valid, fully resolved assembly.
"""

import pytest

import compiler.syntax.asm as asm
import compiler.syntax.tac as tac
from compiler.code_generator import generate
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
        functions=(
            tac.Function(
                name=name,
                body=(tac.Return(val=tac.Constant(value=value)),),
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
    assert result.functions[0].name == "foo"


def test_single_return_produces_two_instructions() -> None:
    """``return <n>;`` with no temporaries translates to exactly ``Mov`` + ``Ret``."""
    result = generate(_simple_tac_program(value=0))
    assert len(result.functions[0].instructions) == 2


# ---------------------------------------------------------------------------
# Return statement translation
# ---------------------------------------------------------------------------


def test_return_constant_produces_mov_imm_to_register() -> None:
    """``Return(Constant(42))`` produces ``Mov(src=Imm(42), dst=Register())``."""
    result = generate(_simple_tac_program(value=42))
    mov = result.functions[0].instructions[0]
    assert mov == asm.Mov(src=asm.Imm(value=42), dst=asm.Register(asm.Registers.AX))


def test_return_produces_ret_as_second_instruction() -> None:
    """The second instruction for a return statement is always ``Ret()``."""
    result = generate(_simple_tac_program(value=0))
    assert result.functions[0].instructions[1] == asm.Ret()


def test_return_value_is_encoded_in_imm() -> None:
    """The immediate operand value matches the returned integer constant."""
    result = generate(_simple_tac_program(value=99))
    mov = result.functions[0].instructions[0]
    assert isinstance(mov, asm.Mov)
    assert isinstance(mov.src, asm.Imm)
    assert mov.src.value == 99


def test_mov_destination_is_register() -> None:
    """The destination operand of the ``Mov`` is always ``Register()``."""
    result = generate(_simple_tac_program(value=0))
    mov = result.functions[0].instructions[0]
    assert isinstance(mov, asm.Mov)
    assert mov.dst == asm.Register(asm.Registers.AX)


# ---------------------------------------------------------------------------
# Multiple return statements
# ---------------------------------------------------------------------------


def test_multiple_return_statements_produce_instruction_pairs() -> None:
    """Each ``Return`` contributes a ``Mov``+``Ret`` pair; no stack allocation."""
    program = tac.Program(
        functions=(
            tac.Function(
                name="main",
                body=(
                    tac.Return(val=tac.Constant(value=1)),
                    tac.Return(val=tac.Constant(value=2)),
                ),
            ),
        )
    )
    result = generate(program)
    assert len(result.functions[0].instructions) == 4
    assert result.functions[0].instructions[0] == asm.Mov(src=asm.Imm(value=1), dst=asm.Register(asm.Registers.AX))
    assert result.functions[0].instructions[1] == asm.Ret()
    assert result.functions[0].instructions[2] == asm.Mov(src=asm.Imm(value=2), dst=asm.Register(asm.Registers.AX))
    assert result.functions[0].instructions[3] == asm.Ret()


# ---------------------------------------------------------------------------
# Unary operations and stack allocation
# ---------------------------------------------------------------------------


def test_unary_negate_produces_allocate_mov_unary_mov_ret() -> None:
    """``Unary(Negate, Constant(5), Var('t'))`` + ``Return(Var('t'))`` produces the expected sequence."""
    program = tac.Program(
        functions=(
            tac.Function(
                name="main",
                body=(
                    tac.Unary(operator=tac.Negate(), src=tac.Constant(value=5), dst=tac.Var(value="tmp.0")),
                    tac.Return(val=tac.Var(value="tmp.0")),
                ),
            ),
        )
    )
    result = generate(program)
    instrs = result.functions[0].instructions
    # AllocateStack, Mov(Imm(5), Stack(-4)), Unary(Neg, Stack(-4)), Mov(Stack(-4), Register()), Ret()
    assert isinstance(instrs[0], asm.AllocateStack)
    assert instrs[1] == asm.Mov(src=asm.Imm(value=5), dst=asm.Stack(offset=-4))
    assert instrs[2] == asm.Unary(operator=asm.Neg(), operand=asm.Stack(offset=-4))
    assert instrs[3] == asm.Mov(src=asm.Stack(offset=-4), dst=asm.Register(asm.Registers.AX))
    assert instrs[4] == asm.Ret()


def test_unary_complement_uses_not_operator() -> None:
    """``Unary(Complement, ...)`` translates to an ``asm.Unary`` with ``Not()``."""
    program = tac.Program(
        functions=(
            tac.Function(
                name="main",
                body=(
                    tac.Unary(operator=tac.Complement(), src=tac.Constant(value=1), dst=tac.Var(value="tmp.0")),
                    tac.Return(val=tac.Var(value="tmp.0")),
                ),
            ),
        )
    )
    result = generate(program)
    unary_instr = result.functions[0].instructions[2]
    assert isinstance(unary_instr, asm.Unary)
    assert unary_instr.operator == asm.Not()


def test_stack_allocation_is_multiple_of_16() -> None:
    """``AllocateStack`` size is always a multiple of 16 bytes."""
    program = tac.Program(
        functions=(
            tac.Function(
                name="main",
                body=(
                    tac.Unary(operator=tac.Negate(), src=tac.Constant(value=1), dst=tac.Var(value="tmp.0")),
                    tac.Return(val=tac.Var(value="tmp.0")),
                ),
            ),
        )
    )
    result = generate(program)
    alloc = result.functions[0].instructions[0]
    assert isinstance(alloc, asm.AllocateStack)
    assert alloc.size % 16 == 0


def test_no_pseudo_operands_in_output() -> None:
    """No ``Pseudo`` operands remain after code generation."""
    program = tac.Program(
        functions=(
            tac.Function(
                name="main",
                body=(
                    tac.Unary(operator=tac.Negate(), src=tac.Constant(value=3), dst=tac.Var(value="tmp.0")),
                    tac.Return(val=tac.Var(value="tmp.0")),
                ),
            ),
        )
    )
    result = generate(program)
    for instr in result.functions[0].instructions:
        match instr:
            case asm.Mov(src=src, dst=dst):
                assert not isinstance(src, asm.Pseudo)
                assert not isinstance(dst, asm.Pseudo)
            case asm.Unary(operand=operand):
                assert not isinstance(operand, asm.Pseudo)


# ---------------------------------------------------------------------------
# Instruction fixup — Mov(mem, mem)
# ---------------------------------------------------------------------------


def test_chained_temps_produce_no_mem_mem_mov() -> None:
    """Two chained unary ops produce no ``Mov(Stack, Stack)`` instructions."""
    program = tac.Program(
        functions=(
            tac.Function(
                name="main",
                body=(
                    tac.Unary(operator=tac.Complement(), src=tac.Constant(value=2), dst=tac.Var(value="tmp.0")),
                    tac.Unary(operator=tac.Negate(), src=tac.Var(value="tmp.0"), dst=tac.Var(value="tmp.1")),
                    tac.Return(val=tac.Var(value="tmp.1")),
                ),
            ),
        )
    )
    result = generate(program)
    for instr in result.functions[0].instructions:
        if isinstance(instr, asm.Mov):
            assert not (isinstance(instr.src, asm.Stack) and isinstance(instr.dst, asm.Stack)), (
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
        functions=(
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
# Binary operations — Add, Sub, Mul
# ---------------------------------------------------------------------------


def test_binary_add_produces_asm_binary_with_r11_scratch() -> None:
    """``Binary(Add, …)`` uses ``%r11d`` as the scratch accumulator register."""
    result = generate(_binary_tac_program(tac.Add(), tac.Constant(1), tac.Constant(2)))
    binary = result.functions[0].instructions[1]
    assert isinstance(binary, asm.Binary)
    assert binary.dst == asm.Register(asm.Registers.R11)


def test_binary_add_operator_is_asm_add() -> None:
    """``tac.Add()`` maps to ``asm.Add()`` inside the ``Binary`` instruction."""
    result = generate(_binary_tac_program(tac.Add(), tac.Constant(1), tac.Constant(2)))
    binary = result.functions[0].instructions[1]
    assert isinstance(binary, asm.Binary)
    assert binary.operator == asm.Add()


def test_binary_subtract_operator_is_asm_sub() -> None:
    """``tac.Subtract()`` maps to ``asm.Sub()`` inside the ``Binary`` instruction."""
    result = generate(_binary_tac_program(tac.Subtract(), tac.Constant(5), tac.Constant(3)))
    binary = result.functions[0].instructions[1]
    assert isinstance(binary, asm.Binary)
    assert binary.operator == asm.Sub()


def test_binary_multiply_operator_is_asm_mult() -> None:
    """``tac.Multiply()`` maps to ``asm.Mult()`` inside the ``Binary`` instruction."""
    result = generate(_binary_tac_program(tac.Multiply(), tac.Constant(4), tac.Constant(3)))
    binary = result.functions[0].instructions[1]
    assert isinstance(binary, asm.Binary)
    assert binary.operator == asm.Mult()


def test_binary_add_result_moved_from_r11_to_stack() -> None:
    """After the ``Binary`` instruction, the result is moved from ``%r11d`` to the stack slot."""
    result = generate(_binary_tac_program(tac.Add(), tac.Constant(1), tac.Constant(2)))
    # AllocateStack, Binary, Mov(R11, Stack(-4)), Mov(Stack(-4), AX), Ret
    mov = result.functions[0].instructions[2]
    assert mov == asm.Mov(src=asm.Register(asm.Registers.R11), dst=asm.Stack(offset=-4))


def test_binary_add_instruction_count() -> None:
    """``Binary(Add, c, c)`` + ``Return`` produces exactly 5 instructions."""
    result = generate(_binary_tac_program(tac.Add(), tac.Constant(1), tac.Constant(2)))
    assert len(result.functions[0].instructions) == 5


# ---------------------------------------------------------------------------
# Division
# ---------------------------------------------------------------------------


def test_divide_loads_dividend_into_ax() -> None:
    """``Binary(Divide, …)`` moves the dividend into ``%eax`` as the first instruction."""
    result = generate(_binary_tac_program(tac.Divide(), tac.Constant(10), tac.Constant(2)))
    # AllocateStack, Mov(Imm(10), AX), Cdq, Mov(Imm(2), R11), Idiv(R11), Mov(AX, Stack), Mov(Stack, AX), Ret
    assert result.functions[0].instructions[1] == asm.Mov(src=asm.Imm(value=10), dst=asm.Register(asm.Registers.AX))


def test_divide_emits_cdq_after_dividend_load() -> None:
    """``Cdq`` follows the dividend ``Mov`` to sign-extend ``%eax`` into ``%edx:%eax``."""
    result = generate(_binary_tac_program(tac.Divide(), tac.Constant(10), tac.Constant(2)))
    assert isinstance(result.functions[0].instructions[2], asm.Cdq)


def test_divide_result_comes_from_ax() -> None:
    """The quotient ``Mov`` sources from ``%eax``."""
    result = generate(_binary_tac_program(tac.Divide(), tac.Constant(10), tac.Constant(2)))
    # After fixup: AllocateStack, Mov(10, AX), Cdq, Mov(2, R11), Idiv(R11), Mov(AX, Stack), Mov(Stack, AX), Ret
    quotient_mov = result.functions[0].instructions[5]
    assert isinstance(quotient_mov, asm.Mov)
    assert quotient_mov.src == asm.Register(asm.Registers.AX)


# ---------------------------------------------------------------------------
# Remainder
# ---------------------------------------------------------------------------


def test_remainder_result_comes_from_dx() -> None:
    """The remainder ``Mov`` sources from ``%edx``, not ``%eax``."""
    result = generate(_binary_tac_program(tac.Remainder(), tac.Constant(10), tac.Constant(3)))
    # AllocateStack, Mov(10, AX), Cdq, Mov(3, R11), Idiv(R11), Mov(DX, Stack), Mov(Stack, AX), Ret
    remainder_mov = result.functions[0].instructions[5]
    assert isinstance(remainder_mov, asm.Mov)
    assert remainder_mov.src == asm.Register(asm.Registers.DX)


def test_remainder_emits_cdq() -> None:
    """``Cdq`` is emitted for remainder just as for divide."""
    result = generate(_binary_tac_program(tac.Remainder(), tac.Constant(7), tac.Constant(3)))
    assert isinstance(result.functions[0].instructions[2], asm.Cdq)


# ---------------------------------------------------------------------------
# Instruction fixup — Idiv with immediate divisor
# ---------------------------------------------------------------------------


def test_idiv_immediate_divisor_routed_through_r11() -> None:
    """An immediate divisor is moved into ``%r11d`` before ``idivl``."""
    result = generate(_binary_tac_program(tac.Divide(), tac.Constant(10), tac.Constant(2)))
    # AllocateStack, Mov(10, AX), Cdq, Mov(2, R11), Idiv(R11), ...
    mov_r11 = result.functions[0].instructions[3]
    idiv = result.functions[0].instructions[4]
    assert mov_r11 == asm.Mov(src=asm.Imm(value=2), dst=asm.Register(asm.Registers.R11))
    assert idiv == asm.Idiv(operand=asm.Register(asm.Registers.R11))


def test_idiv_memory_divisor_not_routed_through_r11() -> None:
    """A stack-slot divisor is passed directly to ``idivl`` without an extra ``Mov``."""
    program = tac.Program(
        functions=(
            tac.Function(
                name="main",
                body=(
                    tac.Binary(operator=tac.Divide(), src1=tac.Constant(10), src2=tac.Var("b"), dst=tac.Var("tmp.0")),
                    tac.Return(val=tac.Var("tmp.0")),
                ),
            ),
        )
    )
    result = generate(program)
    # Find the Idiv instruction and verify its operand is a Stack slot, not a register
    idiv_instrs = [i for i in result.functions[0].instructions if isinstance(i, asm.Idiv)]
    assert len(idiv_instrs) == 1
    assert isinstance(idiv_instrs[0].operand, asm.Stack)


# ---------------------------------------------------------------------------
# Pseudo replacement for binary instructions
# ---------------------------------------------------------------------------


def test_no_pseudo_operands_after_binary_code_gen() -> None:
    """No ``Pseudo`` operands survive into the final binary instruction output."""
    result = generate(_binary_tac_program(tac.Add(), tac.Constant(1), tac.Constant(2)))
    for instr in result.functions[0].instructions:
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
    """``tac.Binary`` with And/Or/Xor operators lowers to ``asm.Binary`` using ``%r11d``."""

    def _instructions(self, op: tac.BinaryOperator) -> tuple[asm.Instruction, ...]:
        """Generate instructions for ``return 3 <op> 5;`` with two constants."""
        return generate(_binary_tac_program(op, tac.Constant(3), tac.Constant(5))).functions[0].instructions

    @pytest.mark.parametrize(
        ("tac_op", "asm_op_type"),
        [
            (tac.BitwiseAnd(), asm.And),
            (tac.BitwiseOr(), asm.Or),
            (tac.BitwiseXor(), asm.Xor),
        ],
        ids=["and", "or", "xor"],
    )
    def test_operator_and_r11_routing(self, tac_op: tac.BinaryOperator, asm_op_type: type) -> None:
        """Each bitwise TAC op lowers to ``asm.Binary`` with the correct operator and ``%r11d`` as dst."""
        binary = next(i for i in self._instructions(tac_op) if isinstance(i, asm.Binary))
        assert isinstance(binary.operator, asm_op_type)
        assert binary.dst == asm.Register(asm.Registers.R11)

    @pytest.mark.parametrize(
        ("op", "src1", "src2", "expected"),
        [
            (asm.And(), 3, 5, "  movl $3, %r11d\n  andl $5, %r11d"),
            (asm.Or(), 4, 2, "  movl $4, %r11d\n  orl $2, %r11d"),
            (asm.Xor(), 6, 3, "  movl $6, %r11d\n  xorl $3, %r11d"),
        ],
        ids=["and", "or", "xor"],
    )
    def test_emit(self, op: asm.BinaryOperator, src1: int, src2: int, expected: str) -> None:
        """``asm.Binary`` with each bitwise operator emits the correct two-line sequence."""
        instr = asm.Binary(operator=op, src1=asm.Imm(src1), src2=asm.Imm(src2), dst=asm.Register(asm.Registers.R11))
        assert instr.emit() == expected


# ---------------------------------------------------------------------------
# Shift code generation
# ---------------------------------------------------------------------------


def _shift_tac_program(op: tac.BinaryOperator, value: int, count: int) -> tac.Program:
    """Build a ``tac.Program`` that shifts ``value`` by ``count`` using ``op``."""
    return tac.Program(
        functions=(
            tac.Function(
                name="main",
                body=(
                    tac.Binary(
                        operator=op,
                        src1=tac.Constant(value),
                        src2=tac.Constant(count),
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
        return generate(_shift_tac_program(op, value=8, count=2)).functions[0].instructions

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
        assert shift.operand == asm.Register(asm.Registers.R11)

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            (asm.Shl(operand=asm.Register(asm.Registers.R11)), "  shll %cl, %r11d"),
            (asm.Sar(operand=asm.Register(asm.Registers.R11)), "  sarl %cl, %r11d"),
        ],
        ids=["shl", "sar"],
    )
    def test_shift_emit(self, instr: asm.Instruction, expected: str) -> None:
        """Shift instructions emit the correct mnemonic with the implicit ``%cl`` count."""
        assert instr.emit() == expected

    def test_shift_count_moved_to_cx(self) -> None:
        """A ``Mov`` to ``%ecx`` (``CX``) appears before ``Shl`` to load the shift count."""
        instrs = self._instructions(tac.LeftShift())
        cx_movs = [i for i in instrs if isinstance(i, asm.Mov) and i.dst == asm.Register(asm.Registers.CX)]
        assert len(cx_movs) == 1

    def test_shl_emit_stack_operand(self) -> None:
        """``asm.Shl(Stack(-4))`` emits ``shll %cl, -4(%rbp)`` — memory operand is valid."""
        assert asm.Shl(operand=asm.Stack(-4)).emit() == "  shll %cl, -4(%rbp)"

    def test_no_pseudo_operands_after_left_shift_code_gen(self) -> None:
        """No ``Pseudo`` operands survive pseudo-replacement for a shift instruction."""
        result = generate(_shift_tac_program(tac.LeftShift(), 4, 1))
        for instr in result.functions[0].instructions:
            if isinstance(instr, asm.Shl):
                assert not isinstance(instr.operand, asm.Pseudo)


def test_unsupported_instruction_raises_codegen_error() -> None:
    """An unrecognised TAC instruction subclass raises ``CodeGenerationError``."""

    class _FakeInstruction(tac.Instruction):
        pass

    program = tac.Program(
        functions=(
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
        functions=(
            tac.Function(
                name="main",
                body=(
                    tac.Binary(
                        operator=op,
                        src1=src1 if src1 is not None else tac.Constant(value=1),
                        src2=src2 if src2 is not None else tac.Constant(value=2),
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
        setcc = next(i for i in result.functions[0].instructions if isinstance(i, asm.SetCC))
        assert setcc.cond_code == expected_cc

    def test_relational_emits_cmp_instruction(self) -> None:
        """A relational binary op always produces a ``Cmp`` instruction."""
        result = generate(_relational_tac_program(tac.Equal()))
        assert any(isinstance(i, asm.Cmp) for i in result.functions[0].instructions)

    def test_relational_zeroes_destination_before_setcc(self) -> None:
        """``Mov(Imm(0), dst)`` immediately precedes ``SetCC`` in the output."""
        result = generate(_relational_tac_program(tac.Equal()))
        instrs = result.functions[0].instructions
        setcc_idx = next(i for i, instr in enumerate(instrs) if isinstance(instr, asm.SetCC))
        zero_mov = instrs[setcc_idx - 1]
        assert isinstance(zero_mov, asm.Mov)
        assert zero_mov.src == asm.Imm(value=0)

    def test_no_pseudo_operands_in_relational_output(self) -> None:
        """No ``Pseudo`` operands survive into the final output for a relational expression."""
        result = generate(_relational_tac_program(tac.Equal(), tac.Var("a"), tac.Var("b")))
        for instr in result.functions[0].instructions:
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
# Jump, label, and copy code generation
# ---------------------------------------------------------------------------


class TestJumpAndLabelCodeGen:
    """``tac.Jump``, ``tac.JumpIfZero``, ``tac.JumpIfNotZero``, ``tac.Label``, and ``tac.Copy`` lowering."""

    def _program(self, *body: tac.Instruction) -> tac.Program:
        return tac.Program(functions=(tac.Function(name="main", body=body),))

    def test_jump_produces_jmp(self) -> None:
        """``tac.Jump(target)`` lowers to ``asm.Jmp(label)``."""
        result = generate(self._program(tac.Jump(target="end"), tac.Return(val=tac.Constant(value=0))))
        assert any(isinstance(i, asm.Jmp) and i.label == "end" for i in result.functions[0].instructions)

    def test_label_produces_asm_label(self) -> None:
        """``tac.Label(name)`` lowers to ``asm.Label(name)``."""
        result = generate(self._program(tac.Label(name="loop"), tac.Return(val=tac.Constant(value=0))))
        assert any(isinstance(i, asm.Label) and i.name == "loop" for i in result.functions[0].instructions)

    def test_copy_produces_mov(self) -> None:
        """``tac.Copy(src, dst)`` lowers to a single ``asm.Mov``."""
        result = generate(
            self._program(
                tac.Copy(src=tac.Constant(value=7), dst=tac.Var(value="x")),
                tac.Return(val=tac.Var(value="x")),
            )
        )
        movs = [i for i in result.functions[0].instructions if isinstance(i, asm.Mov)]
        assert any(isinstance(m.src, asm.Imm) and m.src.value == 7 for m in movs)

    def test_jump_if_zero_produces_cmp_and_jmpcc_e(self) -> None:
        """``tac.JumpIfZero`` lowers to ``Cmp(Imm(0), cond), JmpCC(E, target)``."""
        result = generate(
            self._program(
                tac.JumpIfZero(condition=tac.Var(value="x"), target="end"),
                tac.Return(val=tac.Constant(value=0)),
            )
        )
        jmpcc_instrs = [i for i in result.functions[0].instructions if isinstance(i, asm.JmpCC)]
        assert len(jmpcc_instrs) == 1
        assert jmpcc_instrs[0].cond_code == asm.CondCodes.E
        assert jmpcc_instrs[0].label == "end"

    def test_jump_if_not_zero_produces_jmpcc_ne(self) -> None:
        """``tac.JumpIfNotZero`` lowers to ``JmpCC(NE, target)``."""
        result = generate(
            self._program(
                tac.JumpIfNotZero(condition=tac.Var(value="x"), target="loop"),
                tac.Return(val=tac.Constant(value=0)),
            )
        )
        jmpcc_instrs = [i for i in result.functions[0].instructions if isinstance(i, asm.JmpCC)]
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
        for instr in result.functions[0].instructions:
            if isinstance(instr, asm.Cmp):
                assert not isinstance(instr.op1, asm.Pseudo), "Pseudo survived in Cmp.op1"
                assert not isinstance(instr.op2, asm.Pseudo), "Pseudo survived in Cmp.op2"

    def test_setcc_operand_resolved(self) -> None:
        """The operand of every ``SetCC`` instruction is not a ``Pseudo`` after code gen."""
        result = generate(_relational_tac_program(tac.Equal(), tac.Var("a"), tac.Var("b")))
        for instr in result.functions[0].instructions:
            if isinstance(instr, asm.SetCC):
                assert not isinstance(instr.operand, asm.Pseudo), "Pseudo survived in SetCC.operand"

    def test_jump_if_zero_cmp_operand_resolved(self) -> None:
        """The ``Cmp(Imm(0), cond)`` emitted for ``JumpIfZero`` has its condition operand resolved."""
        program = tac.Program(
            functions=(
                tac.Function(
                    name="main",
                    body=(
                        tac.JumpIfZero(condition=tac.Var(value="x"), target="end"),
                        tac.Label(name="end"),
                        tac.Return(val=tac.Constant(value=0)),
                    ),
                ),
            )
        )
        result = generate(program)
        for instr in result.functions[0].instructions:
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
        for instr in result.functions[0].instructions:
            if isinstance(instr, asm.Cmp):
                assert not isinstance(instr.op2, asm.Imm), "Imm survived as Cmp.op2 after fixup"

    def test_cmp_stack_stack_routed_through_r10(self) -> None:
        """When both ``Cmp`` operands are ``Stack``, op1 is routed through ``%r10d``."""
        result = generate(_relational_tac_program(tac.Equal(), tac.Var("a"), tac.Var("b")))
        for instr in result.functions[0].instructions:
            if isinstance(instr, asm.Cmp):
                assert not (isinstance(instr.op1, asm.Stack) and isinstance(instr.op2, asm.Stack)), (
                    "Cmp(mem, mem) survived fixup"
                )

    def test_cmp_stack_imm_imm_case_fires_first(self) -> None:
        """``Cmp(Stack, Imm)`` is handled by the Imm fixup: op2 goes to ``%r11d``, op1 stays as Stack."""
        # src1=Constant → Imm becomes op2 of Cmp; src2=Var → Stack becomes op1 of Cmp
        result = generate(_relational_tac_program(tac.Equal(), tac.Constant(5), tac.Var("b")))
        cmp_instrs = [i for i in result.functions[0].instructions if isinstance(i, asm.Cmp)]
        assert len(cmp_instrs) == 1
        cmp = cmp_instrs[0]
        assert cmp.op2 == asm.Register(asm.Registers.R11), "op2 should be routed through R11"
        assert isinstance(cmp.op1, asm.Stack), "op1 (Stack) should be unchanged"
