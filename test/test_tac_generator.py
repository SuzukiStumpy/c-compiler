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
                body=(ast.S(statement=ast.Return(exp=exp)),),
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


@pytest.mark.parametrize(
    ("ast_op", "expected"),
    [
        (ast.Add(), tac.Add()),
        (ast.Subtract(), tac.Subtract()),
        (ast.Multiply(), tac.Multiply()),
        (ast.Divide(), tac.Divide()),
        (ast.Remainder(), tac.Remainder()),
        (ast.LeftShift(), tac.LeftShift()),
        (ast.RightShift(), tac.RightShift()),
        (ast.BitwiseAnd(), tac.BitwiseAnd()),
        (ast.BitwiseXor(), tac.BitwiseXor()),
        (ast.BitwiseOr(), tac.BitwiseOr()),
        (ast.Equal(), tac.Equal()),
        (ast.NotEqual(), tac.NotEqual()),
        (ast.LessThan(), tac.LessThan()),
        (ast.LessOrEqual(), tac.LessOrEqual()),
        (ast.GreaterThan(), tac.GreaterThan()),
        (ast.GreaterOrEqual(), tac.GreaterOrEqual()),
    ],
)
def test_binary_operator_maps_to_tac_operator(ast_op: ast.BinaryOperator, expected: tac.BinaryOperator) -> None:
    """Each AST binary operator maps to its TAC counterpart as the first instruction's operator."""
    instrs = _body(ast.Binary(operator=ast_op, left=ast.Constant(value=1), right=ast.Constant(value=2)))
    assert isinstance(instrs[0], tac.Binary)
    assert instrs[0].operator == expected


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

    def test_unsupported_unary_operator_raises(self) -> None:
        """An unrecognised ``UnaryOperator`` subclass raises ``CodeGenerationError``."""

        class _FakeUnaryOp(ast.UnaryOperator):
            pass

        exp = ast.Unary(operator=_FakeUnaryOp(), exp=ast.Constant(value=1))
        with pytest.raises(CodeGenerationError):
            generate(_program_returning(exp))

    def test_unsupported_expression_type_raises(self) -> None:
        """An unrecognised ``Exp`` subclass raises ``CodeGenerationError``."""

        class _FakeExp(ast.Exp):
            pass

        with pytest.raises(CodeGenerationError):
            generate(_program_returning(_FakeExp()))

    def test_unsupported_statement_type_raises(self) -> None:
        """An unrecognised ``Statement`` subclass raises ``CodeGenerationError``."""

        class _FakeStmt(ast.Statement):
            pass

        fn = ast.Function(
            return_type=ast.IntType(),
            name="main",
            params=(),
            body=(_FakeStmt(),),
        )
        with pytest.raises(CodeGenerationError):
            generate(ast.Program(functions=(fn,)))

    def test_zero_functions_raises(self) -> None:
        """A program with no functions raises ``CodeGenerationError``."""
        with pytest.raises(CodeGenerationError):
            generate(ast.Program(functions=()))

    def test_multiple_functions_raises(self) -> None:
        """A program with more than one function raises ``CodeGenerationError``."""
        fn = ast.Function(
            return_type=ast.IntType(),
            name="f",
            params=(),
            body=(ast.Return(exp=ast.Constant(value=0)),),
        )
        with pytest.raises(CodeGenerationError):
            generate(ast.Program(functions=(fn, fn)))


# ---------------------------------------------------------------------------
# Unary operator coverage
# ---------------------------------------------------------------------------


class TestTacGeneratorUnaryOperators:
    """All unary operator variants translate to their TAC equivalents."""

    def test_complement_produces_unary_complement(self) -> None:
        """``return ~1;`` → ``Unary(Complement, Constant(1), tmp.0)``."""
        instrs = _body(ast.Unary(operator=ast.Complement(), exp=ast.Constant(value=1)))
        assert isinstance(instrs[0], tac.Unary)
        assert instrs[0].operator == tac.Complement()

    def test_not_produces_unary_not(self) -> None:
        """``return !1;`` → ``Unary(Not, Constant(1), tmp.0)``."""
        instrs = _body(ast.Unary(operator=ast.Not(), exp=ast.Constant(value=1)))
        assert isinstance(instrs[0], tac.Unary)
        assert instrs[0].operator == tac.Not()


# ---------------------------------------------------------------------------
# Logical AND short-circuit
# ---------------------------------------------------------------------------


class TestTacGeneratorAndExpression:
    """``ast.And`` produces a short-circuit JumpIfZero sequence."""

    def test_full_instruction_sequence(self) -> None:
        """``return 1 && 2;`` emits the full short-circuit sequence."""
        instrs = _body(ast.Binary(operator=ast.And(), left=ast.Constant(value=1), right=ast.Constant(value=2)))
        assert instrs == (
            tac.JumpIfZero(condition=tac.Constant(value=1), target="and_false.0"),
            tac.JumpIfZero(condition=tac.Constant(value=2), target="and_false.0"),
            tac.Copy(src=tac.Constant(value=1), dst=tac.Var(value="tmp.0")),
            tac.Jump(target="and_end.1"),
            tac.Label(name="and_false.0"),
            tac.Copy(src=tac.Constant(value=0), dst=tac.Var(value="tmp.0")),
            tac.Label(name="and_end.1"),
            tac.Return(val=tac.Var(value="tmp.0")),
        )

    def test_right_operand_instructions_appear_after_first_jump(self) -> None:
        """Right sub-expression instructions are emitted after the first ``JumpIfZero``."""
        exp = ast.Binary(
            operator=ast.And(),
            left=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=1)),
            right=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=2)),
        )
        instrs = _body(exp)
        assert isinstance(instrs[0], tac.Unary)  # left side: Negate(1)
        assert isinstance(instrs[1], tac.JumpIfZero)  # short-circuit check on left
        assert isinstance(instrs[2], tac.Unary)  # right side: Negate(2) — only after the jump
        assert isinstance(instrs[3], tac.JumpIfZero)  # short-circuit check on right

    def test_result_is_returned(self) -> None:
        """The ``Return`` instruction uses the temporary written by the ``And`` sequence."""
        instrs = _body(ast.Binary(operator=ast.And(), left=ast.Constant(value=1), right=ast.Constant(value=2)))
        ret = instrs[-1]
        assert isinstance(ret, tac.Return)
        assert ret.val == tac.Var(value="tmp.0")

    def test_labels_are_unique_across_nested_and_expressions(self) -> None:
        """Nested ``&&`` expressions each receive distinct label names."""
        exp = ast.Binary(
            operator=ast.And(),
            left=ast.Binary(operator=ast.And(), left=ast.Constant(value=1), right=ast.Constant(value=2)),
            right=ast.Constant(value=3),
        )
        instrs = _body(exp)
        label_names = [i.name for i in instrs if isinstance(i, tac.Label)]
        assert len(label_names) == len(set(label_names))  # all distinct
        assert len(label_names) == 4  # two labels per And expression


# ---------------------------------------------------------------------------
# Logical OR short-circuit
# ---------------------------------------------------------------------------


class TestTacGeneratorOrExpression:
    """``ast.Or`` produces a short-circuit JumpIfNotZero sequence."""

    def test_full_instruction_sequence(self) -> None:
        """``return 1 || 2;`` emits the full short-circuit sequence."""
        instrs = _body(ast.Binary(operator=ast.Or(), left=ast.Constant(value=1), right=ast.Constant(value=2)))
        assert instrs == (
            tac.JumpIfNotZero(condition=tac.Constant(value=1), target="or_true.0"),
            tac.JumpIfNotZero(condition=tac.Constant(value=2), target="or_true.0"),
            tac.Copy(src=tac.Constant(value=0), dst=tac.Var(value="tmp.0")),
            tac.Jump(target="or_end.1"),
            tac.Label(name="or_true.0"),
            tac.Copy(src=tac.Constant(value=1), dst=tac.Var(value="tmp.0")),
            tac.Label(name="or_end.1"),
            tac.Return(val=tac.Var(value="tmp.0")),
        )

    def test_right_operand_instructions_appear_after_first_jump(self) -> None:
        """Right sub-expression instructions are emitted after the first ``JumpIfNotZero``."""
        exp = ast.Binary(
            operator=ast.Or(),
            left=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=1)),
            right=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=2)),
        )
        instrs = _body(exp)
        assert isinstance(instrs[0], tac.Unary)  # left side: Negate(1)
        assert isinstance(instrs[1], tac.JumpIfNotZero)  # short-circuit check on left
        assert isinstance(instrs[2], tac.Unary)  # right side: Negate(2) — only after jump
        assert isinstance(instrs[3], tac.JumpIfNotZero)  # short-circuit check on right

    def test_result_is_returned(self) -> None:
        """The ``Return`` instruction uses the temporary written by the ``Or`` sequence."""
        instrs = _body(ast.Binary(operator=ast.Or(), left=ast.Constant(value=1), right=ast.Constant(value=2)))
        ret = instrs[-1]
        assert isinstance(ret, tac.Return)
        assert ret.val == tac.Var(value="tmp.0")

    def test_labels_are_unique_across_nested_or_expressions(self) -> None:
        """Nested ``||`` expressions each receive distinct label names."""
        exp = ast.Binary(
            operator=ast.Or(),
            left=ast.Binary(operator=ast.Or(), left=ast.Constant(value=1), right=ast.Constant(value=2)),
            right=ast.Constant(value=3),
        )
        instrs = _body(exp)
        label_names = [i.name for i in instrs if isinstance(i, tac.Label)]
        assert len(label_names) == len(set(label_names))
        assert len(label_names) == 4
