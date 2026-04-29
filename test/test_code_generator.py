"""Tests for the code generation stage.

Verifies that ``generate`` correctly translates each main-AST construct into
the equivalent assembly-AST nodes.
"""

import pytest

import compiler.asm as asm
import compiler.ast as ast
from compiler.code_generator import generate
from compiler.utils.exceptions import CodeGenerationError


def _simple_program(value: int = 0, name: str = "main") -> ast.Program:
    """Build a minimal ``ast.Program`` returning the given constant.

    Args:
        value: The integer value returned by the function.
        name: The function name.

    Returns:
        An ``ast.Program`` containing a single function that returns ``value``.
    """
    return ast.Program(
        functions=(
            ast.Function(
                return_type=ast.IntType(),
                name=name,
                params=(),
                body=(ast.Return(exp=ast.Constant(value=value)),),
            ),
        )
    )


# ---------------------------------------------------------------------------
# Program and function translation
# ---------------------------------------------------------------------------


def test_generates_asm_program() -> None:
    """``generate`` returns an ``asm.Program`` node."""
    result = generate(_simple_program())
    assert isinstance(result, asm.Program)


def test_function_name_is_preserved() -> None:
    """The assembly function carries the same name as the source function."""
    result = generate(_simple_program(name="foo"))
    assert result.function.name == "foo"


def test_single_return_produces_two_instructions() -> None:
    """``return <n>;`` translates to exactly ``Mov`` followed by ``Ret``."""
    result = generate(_simple_program(value=0))
    assert len(result.function.instructions) == 2


# ---------------------------------------------------------------------------
# Return statement translation
# ---------------------------------------------------------------------------


def test_return_constant_produces_mov_imm_to_register() -> None:
    """``return 42;`` produces ``Mov(src=Imm(42), dst=Register())``."""
    result = generate(_simple_program(value=42))
    mov = result.function.instructions[0]
    assert mov == asm.Mov(src=asm.Imm(value=42), dst=asm.Register())


def test_return_produces_ret_as_second_instruction() -> None:
    """The second instruction for a return statement is always ``Ret()``."""
    result = generate(_simple_program(value=0))
    assert result.function.instructions[1] == asm.Ret()


def test_return_value_is_encoded_in_imm() -> None:
    """The immediate operand value matches the returned integer constant."""
    result = generate(_simple_program(value=99))
    mov = result.function.instructions[0]
    assert isinstance(mov, asm.Mov)
    assert isinstance(mov.src, asm.Imm)
    assert mov.src.value == 99


def test_mov_destination_is_register() -> None:
    """The destination operand of the ``Mov`` is always ``Register()``."""
    result = generate(_simple_program(value=0))
    mov = result.function.instructions[0]
    assert isinstance(mov, asm.Mov)
    assert mov.dst == asm.Register()


# ---------------------------------------------------------------------------
# Multiple statements
# ---------------------------------------------------------------------------


def test_multiple_return_statements_produce_instruction_pairs() -> None:
    """Each ``return`` statement contributes a ``Mov``+``Ret`` pair."""
    program = ast.Program(
        functions=(
            ast.Function(
                return_type=ast.IntType(),
                name="main",
                params=(),
                body=(
                    ast.Return(exp=ast.Constant(value=1)),
                    ast.Return(exp=ast.Constant(value=2)),
                ),
            ),
        )
    )
    result = generate(program)
    assert len(result.function.instructions) == 4
    assert result.function.instructions[0] == asm.Mov(src=asm.Imm(value=1), dst=asm.Register())
    assert result.function.instructions[1] == asm.Ret()
    assert result.function.instructions[2] == asm.Mov(src=asm.Imm(value=2), dst=asm.Register())
    assert result.function.instructions[3] == asm.Ret()


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_multiple_functions_raises_codegen_error() -> None:
    """``generate`` raises ``CodeGenerationError`` if the program has more than one function."""

    def _fn(name: str) -> ast.Function:
        return ast.Function(
            return_type=ast.IntType(),
            name=name,
            params=(),
            body=(ast.Return(exp=ast.Constant(value=0)),),
        )

    program = ast.Program(functions=(_fn("foo"), _fn("bar")))
    with pytest.raises(CodeGenerationError, match="exactly one function"):
        generate(program)
