"""Tests for the TAC generation stage.

Verifies that ``generate`` correctly translates ``ast`` nodes into TAC
instruction sequences, with particular attention to binary operations,
instruction ordering, and temporary variable allocation.

Each test constructs ``ast`` nodes directly (without invoking the lexer or
parser) so that TAC generator behaviour is tested in isolation.
"""

import pytest

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from compiler.tac_generator import generate
from compiler.utils.exceptions import CodeGenerationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _program_returning(exp: ast.Exp) -> ast.Program:
    """Wrap an expression in a minimal ``ast.Program`` that returns it.

    Args:
        exp: The expression to return from the generated function.

    Returns:
        An ``ast.Program`` containing a single ``main`` function whose body
        is ``return <exp>;``.
    """
    return ast.Program(
        functions=(
            ast.Function(
                return_type=ast.IntType(),
                name="main",
                params=(),
                body=(ast.Return(exp=exp),),
            ),
        )
    )


def _body(exp: ast.Exp) -> tuple[tac.Instruction, ...]:
    """Translate a single expression and return the resulting TAC instruction body.

    Args:
        exp: The expression to translate.

    Returns:
        The TAC instruction tuple from the generated function body.
    """
    return generate(_program_returning(exp)).function.body


# ---------------------------------------------------------------------------
# Constant and unary (regression — must not be broken by binary additions)
# ---------------------------------------------------------------------------


class TestTacGeneratorExistingConstructs:
    """Constant and unary constructs continue to translate correctly."""

    def test_constant_produces_only_return(self) -> None:
        """``return 42;`` → a single ``Return(Constant(42))`` instruction."""
        instrs = _body(ast.Constant(value=42))
        assert instrs == (tac.Return(val=tac.Constant(value=42)),)

    def test_unary_negate_produces_unary_then_return(self) -> None:
        """``return -1;`` → ``Unary(Negate, Constant(1), tmp.0)`` + ``Return(tmp.0)``."""
        instrs = _body(ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=1)))
        assert instrs == (
            tac.Unary(operator=tac.Negate(), src=tac.Constant(value=1), dst=tac.Var(value="tmp.0")),
            tac.Return(val=tac.Var(value="tmp.0")),
        )


# ---------------------------------------------------------------------------
# Binary operator mapping
# ---------------------------------------------------------------------------


class TestTacGeneratorBinaryOperatorMapping:
    """Each ast binary operator maps to the correct tac binary operator."""

    def _binary_op(self, op: ast.BinaryOperator) -> tac.BinaryOperator:
        """Translate ``return 1 <op> 2;`` and extract the tac operator."""
        instrs = _body(ast.Binary(operator=op, left=ast.Constant(value=1), right=ast.Constant(value=2)))
        assert isinstance(instrs[0], tac.Binary)
        return instrs[0].operator

    def test_add_maps_to_tac_add(self) -> None:
        """``ast.Add()`` → ``tac.Add()``."""
        assert self._binary_op(ast.Add()) == tac.Add()

    def test_subtract_maps_to_tac_subtract(self) -> None:
        """``ast.Subtract()`` → ``tac.Subtract()``."""
        assert self._binary_op(ast.Subtract()) == tac.Subtract()

    def test_multiply_maps_to_tac_multiply(self) -> None:
        """``ast.Multiply()`` → ``tac.Multiply()``."""
        assert self._binary_op(ast.Multiply()) == tac.Multiply()

    def test_divide_maps_to_tac_divide(self) -> None:
        """``ast.Divide()`` → ``tac.Divide()``."""
        assert self._binary_op(ast.Divide()) == tac.Divide()

    def test_remainder_maps_to_tac_remainder(self) -> None:
        """``ast.Remainder()`` → ``tac.Remainder()``."""
        assert self._binary_op(ast.Remainder()) == tac.Remainder()


# ---------------------------------------------------------------------------
# Binary instruction structure
# ---------------------------------------------------------------------------


class TestTacGeneratorBinaryInstructions:
    """Binary expressions produce the correct instruction sequence and operands."""

    def test_two_constants_produce_binary_then_return(self) -> None:
        """``return 1 + 2;`` → ``Binary(Add, Constant(1), Constant(2), tmp.0)`` + ``Return(tmp.0)``."""
        instrs = _body(ast.Binary(operator=ast.Add(), left=ast.Constant(value=1), right=ast.Constant(value=2)))
        assert instrs == (
            tac.Binary(
                operator=tac.Add(),
                src1=tac.Constant(value=1),
                src2=tac.Constant(value=2),
                dst=tac.Var(value="tmp.0"),
            ),
            tac.Return(val=tac.Var(value="tmp.0")),
        )

    def test_binary_result_is_returned(self) -> None:
        """The ``Return`` instruction uses the temporary produced by the ``Binary``."""
        instrs = _body(ast.Binary(operator=ast.Multiply(), left=ast.Constant(value=3), right=ast.Constant(value=4)))
        assert isinstance(instrs[0], tac.Binary)
        assert isinstance(instrs[1], tac.Return)
        assert instrs[1].val == instrs[0].dst

    def test_left_operand_is_src1(self) -> None:
        """The left sub-expression value becomes ``src1`` of the ``Binary`` instruction."""
        instrs = _body(ast.Binary(operator=ast.Subtract(), left=ast.Constant(value=5), right=ast.Constant(value=3)))
        assert isinstance(instrs[0], tac.Binary)
        assert instrs[0].src1 == tac.Constant(value=5)

    def test_right_operand_is_src2(self) -> None:
        """The right sub-expression value becomes ``src2`` of the ``Binary`` instruction."""
        instrs = _body(ast.Binary(operator=ast.Subtract(), left=ast.Constant(value=5), right=ast.Constant(value=3)))
        assert isinstance(instrs[0], tac.Binary)
        assert instrs[0].src2 == tac.Constant(value=3)


# ---------------------------------------------------------------------------
# Instruction ordering and temporary allocation
# ---------------------------------------------------------------------------


class TestTacGeneratorInstructionOrdering:
    """Instructions for compound operands appear in the correct order."""

    def test_left_operand_instructions_precede_right(self) -> None:
        """For ``(-1) + (-2)``, the negation of 1 is emitted before the negation of 2."""
        exp = ast.Binary(
            operator=ast.Add(),
            left=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=1)),
            right=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=2)),
        )
        instrs = _body(exp)
        # Unary(Negate, 1, tmp.0), Unary(Negate, 2, tmp.1), Binary(Add, tmp.0, tmp.1, tmp.2), Return(tmp.2)
        assert len(instrs) == 4
        assert instrs[0] == tac.Unary(operator=tac.Negate(), src=tac.Constant(value=1), dst=tac.Var(value="tmp.0"))
        assert instrs[1] == tac.Unary(operator=tac.Negate(), src=tac.Constant(value=2), dst=tac.Var(value="tmp.1"))
        assert instrs[2] == tac.Binary(
            operator=tac.Add(),
            src1=tac.Var(value="tmp.0"),
            src2=tac.Var(value="tmp.1"),
            dst=tac.Var(value="tmp.2"),
        )
        assert instrs[3] == tac.Return(val=tac.Var(value="tmp.2"))

    def test_nested_binary_right_arm_emitted_before_outer(self) -> None:
        """For ``1 + (2 * 3)``, the multiply instruction precedes the add instruction."""
        exp = ast.Binary(
            operator=ast.Add(),
            left=ast.Constant(value=1),
            right=ast.Binary(operator=ast.Multiply(), left=ast.Constant(value=2), right=ast.Constant(value=3)),
        )
        instrs = _body(exp)
        # Binary(Multiply, 2, 3, tmp.0), Binary(Add, 1, tmp.0, tmp.1), Return(tmp.1)
        assert len(instrs) == 3
        assert instrs[0] == tac.Binary(
            operator=tac.Multiply(),
            src1=tac.Constant(value=2),
            src2=tac.Constant(value=3),
            dst=tac.Var(value="tmp.0"),
        )
        assert instrs[1] == tac.Binary(
            operator=tac.Add(),
            src1=tac.Constant(value=1),
            src2=tac.Var(value="tmp.0"),
            dst=tac.Var(value="tmp.1"),
        )
        assert instrs[2] == tac.Return(val=tac.Var(value="tmp.1"))

    def test_temporaries_are_allocated_in_order(self) -> None:
        """Temporaries are named ``tmp.0``, ``tmp.1``, … in emission order."""
        exp = ast.Binary(
            operator=ast.Add(),
            left=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=1)),
            right=ast.Constant(value=2),
        )
        instrs = _body(exp)
        # Unary(Negate, 1, tmp.0), Binary(Add, tmp.0, 2, tmp.1), Return(tmp.1)
        assert isinstance(instrs[0], tac.Unary)
        assert instrs[0].dst == tac.Var(value="tmp.0")
        assert isinstance(instrs[1], tac.Binary)
        assert instrs[1].dst == tac.Var(value="tmp.1")


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestTacGeneratorErrors:
    """Unsupported constructs raise ``CodeGenerationError``."""

    def test_unsupported_binary_operator_raises(self) -> None:
        """An unrecognised ``BinaryOperator`` subclass raises ``CodeGenerationError``."""

        class _FakeBinaryOp(ast.BinaryOperator):
            pass

        exp = ast.Binary(operator=_FakeBinaryOp(), left=ast.Constant(value=1), right=ast.Constant(value=2))
        with pytest.raises(CodeGenerationError):
            generate(_program_returning(exp))


# ---------------------------------------------------------------------------
# Bitwise and shift operator mapping
# ---------------------------------------------------------------------------


class TestTacGeneratorBitwiseOperatorMapping:
    """Each bitwise / shift AST binary operator maps to the correct TAC operator."""

    def _binary_op(self, op: ast.BinaryOperator) -> tac.BinaryOperator:
        """Translate ``return 1 <op> 2;`` and extract the tac operator."""
        instrs = _body(ast.Binary(operator=op, left=ast.Constant(value=1), right=ast.Constant(value=2)))
        assert isinstance(instrs[0], tac.Binary)
        return instrs[0].operator

    def test_left_shift_maps_to_tac_left_shift(self) -> None:
        """``ast.LeftShift()`` → ``tac.LeftShift()``."""
        assert self._binary_op(ast.LeftShift()) == tac.LeftShift()

    def test_right_shift_maps_to_tac_right_shift(self) -> None:
        """``ast.RightShift()`` → ``tac.RightShift()``."""
        assert self._binary_op(ast.RightShift()) == tac.RightShift()

    def test_bitwise_and_maps_to_tac_bitwise_and(self) -> None:
        """``ast.BitwiseAnd()`` → ``tac.BitwiseAnd()``."""
        assert self._binary_op(ast.BitwiseAnd()) == tac.BitwiseAnd()

    def test_bitwise_xor_maps_to_tac_bitwise_xor(self) -> None:
        """``ast.BitwiseXor()`` → ``tac.BitwiseXor()``."""
        assert self._binary_op(ast.BitwiseXor()) == tac.BitwiseXor()

    def test_bitwise_or_maps_to_tac_bitwise_or(self) -> None:
        """``ast.BitwiseOr()`` → ``tac.BitwiseOr()``."""
        assert self._binary_op(ast.BitwiseOr()) == tac.BitwiseOr()
