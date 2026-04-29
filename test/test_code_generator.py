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
        function=tac.Function(
            name=name,
            body=(tac.Return(val=tac.Constant(value=value)),),
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
    assert result.function.name == "foo"


def test_single_return_produces_two_instructions() -> None:
    """``return <n>;`` with no temporaries translates to exactly ``Mov`` + ``Ret``."""
    result = generate(_simple_tac_program(value=0))
    assert len(result.function.instructions) == 2


# ---------------------------------------------------------------------------
# Return statement translation
# ---------------------------------------------------------------------------


def test_return_constant_produces_mov_imm_to_register() -> None:
    """``Return(Constant(42))`` produces ``Mov(src=Imm(42), dst=Register())``."""
    result = generate(_simple_tac_program(value=42))
    mov = result.function.instructions[0]
    assert mov == asm.Mov(src=asm.Imm(value=42), dst=asm.Register(asm.Registers.AX))


def test_return_produces_ret_as_second_instruction() -> None:
    """The second instruction for a return statement is always ``Ret()``."""
    result = generate(_simple_tac_program(value=0))
    assert result.function.instructions[1] == asm.Ret()


def test_return_value_is_encoded_in_imm() -> None:
    """The immediate operand value matches the returned integer constant."""
    result = generate(_simple_tac_program(value=99))
    mov = result.function.instructions[0]
    assert isinstance(mov, asm.Mov)
    assert isinstance(mov.src, asm.Imm)
    assert mov.src.value == 99


def test_mov_destination_is_register() -> None:
    """The destination operand of the ``Mov`` is always ``Register()``."""
    result = generate(_simple_tac_program(value=0))
    mov = result.function.instructions[0]
    assert isinstance(mov, asm.Mov)
    assert mov.dst == asm.Register(asm.Registers.AX)


# ---------------------------------------------------------------------------
# Multiple return statements
# ---------------------------------------------------------------------------


def test_multiple_return_statements_produce_instruction_pairs() -> None:
    """Each ``Return`` contributes a ``Mov``+``Ret`` pair; no stack allocation."""
    program = tac.Program(
        function=tac.Function(
            name="main",
            body=(
                tac.Return(val=tac.Constant(value=1)),
                tac.Return(val=tac.Constant(value=2)),
            ),
        )
    )
    result = generate(program)
    assert len(result.function.instructions) == 4
    assert result.function.instructions[0] == asm.Mov(src=asm.Imm(value=1), dst=asm.Register(asm.Registers.AX))
    assert result.function.instructions[1] == asm.Ret()
    assert result.function.instructions[2] == asm.Mov(src=asm.Imm(value=2), dst=asm.Register(asm.Registers.AX))
    assert result.function.instructions[3] == asm.Ret()


# ---------------------------------------------------------------------------
# Unary operations and stack allocation
# ---------------------------------------------------------------------------


def test_unary_negate_produces_allocate_mov_unary_mov_ret() -> None:
    """``Unary(Negate, Constant(5), Var('t'))`` + ``Return(Var('t'))`` produces the expected sequence."""
    program = tac.Program(
        function=tac.Function(
            name="main",
            body=(
                tac.Unary(operator=tac.Negate(), src=tac.Constant(value=5), dst=tac.Var(value="tmp.0")),
                tac.Return(val=tac.Var(value="tmp.0")),
            ),
        )
    )
    result = generate(program)
    instrs = result.function.instructions
    # AllocateStack, Mov(Imm(5), Stack(-4)), Unary(Neg, Stack(-4)), Mov(Stack(-4), Register()), Ret()
    assert isinstance(instrs[0], asm.AllocateStack)
    assert instrs[1] == asm.Mov(src=asm.Imm(value=5), dst=asm.Stack(offset=-4))
    assert instrs[2] == asm.Unary(operator=asm.Neg(), operand=asm.Stack(offset=-4))
    assert instrs[3] == asm.Mov(src=asm.Stack(offset=-4), dst=asm.Register(asm.Registers.AX))
    assert instrs[4] == asm.Ret()


def test_unary_complement_uses_not_operator() -> None:
    """``Unary(Complement, ...)`` translates to an ``asm.Unary`` with ``Not()``."""
    program = tac.Program(
        function=tac.Function(
            name="main",
            body=(
                tac.Unary(operator=tac.Complement(), src=tac.Constant(value=1), dst=tac.Var(value="tmp.0")),
                tac.Return(val=tac.Var(value="tmp.0")),
            ),
        )
    )
    result = generate(program)
    unary_instr = result.function.instructions[2]
    assert isinstance(unary_instr, asm.Unary)
    assert unary_instr.operator == asm.Not()


def test_stack_allocation_is_multiple_of_16() -> None:
    """``AllocateStack`` size is always a multiple of 16 bytes."""
    program = tac.Program(
        function=tac.Function(
            name="main",
            body=(
                tac.Unary(operator=tac.Negate(), src=tac.Constant(value=1), dst=tac.Var(value="tmp.0")),
                tac.Return(val=tac.Var(value="tmp.0")),
            ),
        )
    )
    result = generate(program)
    alloc = result.function.instructions[0]
    assert isinstance(alloc, asm.AllocateStack)
    assert alloc.size % 16 == 0


def test_no_pseudo_operands_in_output() -> None:
    """No ``Pseudo`` operands remain after code generation."""
    program = tac.Program(
        function=tac.Function(
            name="main",
            body=(
                tac.Unary(operator=tac.Negate(), src=tac.Constant(value=3), dst=tac.Var(value="tmp.0")),
                tac.Return(val=tac.Var(value="tmp.0")),
            ),
        )
    )
    result = generate(program)
    for instr in result.function.instructions:
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
        function=tac.Function(
            name="main",
            body=(
                tac.Unary(operator=tac.Complement(), src=tac.Constant(value=2), dst=tac.Var(value="tmp.0")),
                tac.Unary(operator=tac.Negate(), src=tac.Var(value="tmp.0"), dst=tac.Var(value="tmp.1")),
                tac.Return(val=tac.Var(value="tmp.1")),
            ),
        )
    )
    result = generate(program)
    for instr in result.function.instructions:
        if isinstance(instr, asm.Mov):
            assert not (isinstance(instr.src, asm.Stack) and isinstance(instr.dst, asm.Stack)), (
                f"Invalid Mov(mem, mem) found: {instr}"
            )


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_unsupported_instruction_raises_codegen_error() -> None:
    """An unrecognised TAC instruction subclass raises ``CodeGenerationError``."""

    class _FakeInstruction(tac.Instruction):
        pass

    program = tac.Program(
        function=tac.Function(
            name="main",
            body=(_FakeInstruction(),),
        )
    )
    with pytest.raises(CodeGenerationError):
        generate(program)
