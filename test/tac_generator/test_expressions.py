"""Tests for expression-level TAC generation."""

import pytest

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from compiler.syntax.ast.constants import ConstFloat, ConstLongDouble
from compiler.syntax.ast.types import BoolType, LongDoubleType
from compiler.tac_generator import generate
from compiler.utils.exceptions import CodeGenerationError
from test.tac_generator.helpers import _body, _instrs_for_body, _program_returning

# ---------------------------------------------------------------------------
# Constant and unary (regression - must not be broken by binary additions)
# ---------------------------------------------------------------------------


class TestTacGeneratorExistingConstructs:
    """Constant and unary constructs continue to translate correctly."""

    def test_constant_produces_only_return(self) -> None:
        """``return 42;`` -> ``Return(Constant(42))`` with no guard (already returns)."""
        instrs = _body(ast.Constant(value=ast.ConstInt(42)))
        assert instrs == (tac.Return(val=tac.Constant(value=ast.ConstInt(42))),)

    def test_unary_negate_produces_unary_then_return(self) -> None:
        """``return -1;`` -> ``Unary(Negate, Constant(1), tmp.0)`` + ``Return(tmp.0)``."""
        instrs = _body(ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(1))))
        assert instrs == (
            tac.Unary(operator=tac.Negate(), src=tac.Constant(value=ast.ConstInt(1)), dst=tac.Var(value="tmp.0")),
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
    instrs = _body(
        ast.Binary(operator=ast_op, left=ast.Constant(value=ast.ConstInt(1)), right=ast.Constant(value=ast.ConstInt(2)))
    )
    assert isinstance(instrs[0], tac.Binary)
    assert instrs[0].operator == expected


# ---------------------------------------------------------------------------
# Binary instruction structure
# ---------------------------------------------------------------------------


class TestTacGeneratorBinaryInstructions:
    """Binary expressions produce the correct instruction sequence and operands."""

    def test_two_constants_produce_binary_then_return(self) -> None:
        """``return 1 + 2;`` -> ``Binary(Add, Constant(1), Constant(2), tmp.0)`` + ``Return(tmp.0)``."""
        instrs = _body(
            ast.Binary(
                operator=ast.Add(), left=ast.Constant(value=ast.ConstInt(1)), right=ast.Constant(value=ast.ConstInt(2))
            )
        )
        assert instrs == (
            tac.Binary(
                operator=tac.Add(),
                src1=tac.Constant(value=ast.ConstInt(1)),
                src2=tac.Constant(value=ast.ConstInt(2)),
                dst=tac.Var(value="tmp.0"),
            ),
            tac.Return(val=tac.Var(value="tmp.0")),
        )

    def test_binary_result_is_returned(self) -> None:
        """The ``Return`` instruction uses the temporary produced by the ``Binary``."""
        instrs = _body(
            ast.Binary(
                operator=ast.Multiply(),
                left=ast.Constant(value=ast.ConstInt(3)),
                right=ast.Constant(value=ast.ConstInt(4)),
            )
        )
        assert isinstance(instrs[0], tac.Binary)
        assert isinstance(instrs[1], tac.Return)
        assert instrs[1].val == instrs[0].dst

    def test_left_operand_is_src1(self) -> None:
        """The left sub-expression value becomes ``src1`` of the ``Binary`` instruction."""
        instrs = _body(
            ast.Binary(
                operator=ast.Subtract(),
                left=ast.Constant(value=ast.ConstInt(5)),
                right=ast.Constant(value=ast.ConstInt(3)),
            )
        )
        assert isinstance(instrs[0], tac.Binary)
        assert instrs[0].src1 == tac.Constant(value=ast.ConstInt(5))

    def test_right_operand_is_src2(self) -> None:
        """The right sub-expression value becomes ``src2`` of the ``Binary`` instruction."""
        instrs = _body(
            ast.Binary(
                operator=ast.Subtract(),
                left=ast.Constant(value=ast.ConstInt(5)),
                right=ast.Constant(value=ast.ConstInt(3)),
            )
        )
        assert isinstance(instrs[0], tac.Binary)
        assert instrs[0].src2 == tac.Constant(value=ast.ConstInt(3))


# ---------------------------------------------------------------------------
# Instruction ordering and temporary allocation
# ---------------------------------------------------------------------------


class TestTacGeneratorInstructionOrdering:
    """Instructions for compound operands appear in the correct order."""

    def test_left_operand_instructions_precede_right(self) -> None:
        """For ``(-1) + (-2)``, the negation of 1 is emitted before the negation of 2."""
        exp = ast.Binary(
            operator=ast.Add(),
            left=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(1))),
            right=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(2))),
        )
        instrs = _body(exp)
        # Unary(Negate, 1, tmp.0), Unary(Negate, 2, tmp.1), Binary(Add, tmp.0, tmp.1, tmp.2), Return(tmp.2)
        assert len(instrs) == 4
        assert instrs[0] == tac.Unary(
            operator=tac.Negate(), src=tac.Constant(value=ast.ConstInt(1)), dst=tac.Var(value="tmp.0")
        )
        assert instrs[1] == tac.Unary(
            operator=tac.Negate(), src=tac.Constant(value=ast.ConstInt(2)), dst=tac.Var(value="tmp.1")
        )
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
            left=ast.Constant(value=ast.ConstInt(1)),
            right=ast.Binary(
                operator=ast.Multiply(),
                left=ast.Constant(value=ast.ConstInt(2)),
                right=ast.Constant(value=ast.ConstInt(3)),
            ),
        )
        instrs = _body(exp)
        # Binary(Multiply, 2, 3, tmp.0), Binary(Add, 1, tmp.0, tmp.1), Return(tmp.1)
        assert len(instrs) == 3
        assert instrs[0] == tac.Binary(
            operator=tac.Multiply(),
            src1=tac.Constant(value=ast.ConstInt(2)),
            src2=tac.Constant(value=ast.ConstInt(3)),
            dst=tac.Var(value="tmp.0"),
        )
        assert instrs[1] == tac.Binary(
            operator=tac.Add(),
            src1=tac.Constant(value=ast.ConstInt(1)),
            src2=tac.Var(value="tmp.0"),
            dst=tac.Var(value="tmp.1"),
        )
        assert instrs[2] == tac.Return(val=tac.Var(value="tmp.1"))

    def test_temporaries_are_allocated_in_order(self) -> None:
        """Temporaries are named ``tmp.0``, ``tmp.1``, ... in emission order."""
        exp = ast.Binary(
            operator=ast.Add(),
            left=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(1))),
            right=ast.Constant(value=ast.ConstInt(2)),
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

        exp = ast.Binary(
            operator=_FakeBinaryOp(),
            left=ast.Constant(value=ast.ConstInt(1)),
            right=ast.Constant(value=ast.ConstInt(2)),
        )
        with pytest.raises(CodeGenerationError):
            generate(_program_returning(exp), {})

    def test_unsupported_unary_operator_raises(self) -> None:
        """An unrecognised ``UnaryOperator`` subclass raises ``CodeGenerationError``."""

        class _FakeUnaryOp(ast.UnaryOperator):
            pass

        exp = ast.Unary(operator=_FakeUnaryOp(), exp=ast.Constant(value=ast.ConstInt(1)))
        with pytest.raises(CodeGenerationError):
            generate(_program_returning(exp), {})

    def test_unsupported_expression_type_raises(self) -> None:
        """An unrecognised ``Exp`` subclass raises ``CodeGenerationError``."""

        class _FakeExp(ast.Exp):
            pass

        with pytest.raises(CodeGenerationError):
            generate(_program_returning(_FakeExp()), {})

    def test_unsupported_statement_type_raises(self) -> None:
        """An unrecognised ``Statement`` subclass raises ``CodeGenerationError``."""

        class _FakeStmt(ast.Statement):
            pass

        fn = ast.Function(
            name="main",
            params=(),
            body=(ast.S(statement=_FakeStmt()),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        with pytest.raises(CodeGenerationError):
            generate(ast.Program(declarations=(ast.FunDecl(fn),)), {})


# ---------------------------------------------------------------------------
# Unary operator coverage
# ---------------------------------------------------------------------------


class TestTacGeneratorUnaryOperators:
    """All unary operator variants translate to their TAC equivalents."""

    def test_complement_produces_unary_complement(self) -> None:
        """``return ~1;`` -> ``Unary(Complement, Constant(1), tmp.0)``."""
        instrs = _body(ast.Unary(operator=ast.Complement(), exp=ast.Constant(value=ast.ConstInt(1))))
        assert isinstance(instrs[0], tac.Unary)
        assert instrs[0].operator == tac.Complement()

    def test_not_produces_unary_not(self) -> None:
        """``return !1;`` -> ``Unary(Not, Constant(1), tmp.0)``."""
        instrs = _body(ast.Unary(operator=ast.Not(), exp=ast.Constant(value=ast.ConstInt(1))))
        assert isinstance(instrs[0], tac.Unary)
        assert instrs[0].operator == tac.Not()


# ---------------------------------------------------------------------------
# Logical AND short-circuit
# ---------------------------------------------------------------------------


class TestTacGeneratorAndExpression:
    """``ast.And`` produces a short-circuit JumpIfZero sequence."""

    def test_full_instruction_sequence(self) -> None:
        """``return 1 && 2;`` emits the full short-circuit sequence."""
        instrs = _body(
            ast.Binary(
                operator=ast.And(), left=ast.Constant(value=ast.ConstInt(1)), right=ast.Constant(value=ast.ConstInt(2))
            )
        )
        assert instrs == (
            tac.JumpIfZero(condition=tac.Constant(value=ast.ConstInt(1)), target="and_false.0"),
            tac.JumpIfZero(condition=tac.Constant(value=ast.ConstInt(2)), target="and_false.0"),
            tac.Copy(src=tac.Constant(value=ast.ConstInt(1)), dst=tac.Var(value="tmp.0")),
            tac.Jump(target="and_end.1"),
            tac.Label(name="and_false.0"),
            tac.Copy(src=tac.Constant(value=ast.ConstInt(0)), dst=tac.Var(value="tmp.0")),
            tac.Label(name="and_end.1"),
            tac.Return(val=tac.Var(value="tmp.0")),
        )

    def test_right_operand_instructions_appear_after_first_jump(self) -> None:
        """Right sub-expression instructions are emitted after the first ``JumpIfZero``."""
        exp = ast.Binary(
            operator=ast.And(),
            left=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(1))),
            right=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(2))),
        )
        instrs = _body(exp)
        assert isinstance(instrs[0], tac.Unary)  # left side: Negate(1)
        assert isinstance(instrs[1], tac.JumpIfZero)  # short-circuit check on left
        assert isinstance(instrs[2], tac.Unary)  # right side: Negate(2) - only after the jump
        assert isinstance(instrs[3], tac.JumpIfZero)  # short-circuit check on right

    def test_result_is_returned(self) -> None:
        """The ``Return`` instruction uses the temporary written by the ``And`` sequence."""
        instrs = _body(
            ast.Binary(
                operator=ast.And(), left=ast.Constant(value=ast.ConstInt(1)), right=ast.Constant(value=ast.ConstInt(2))
            )
        )
        ret = instrs[-1]
        assert isinstance(ret, tac.Return)
        assert ret.val == tac.Var(value="tmp.0")

    def test_labels_are_unique_across_nested_and_expressions(self) -> None:
        """Nested ``&&`` expressions each receive distinct label names."""
        exp = ast.Binary(
            operator=ast.And(),
            left=ast.Binary(
                operator=ast.And(), left=ast.Constant(value=ast.ConstInt(1)), right=ast.Constant(value=ast.ConstInt(2))
            ),
            right=ast.Constant(value=ast.ConstInt(3)),
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
        instrs = _body(
            ast.Binary(
                operator=ast.Or(), left=ast.Constant(value=ast.ConstInt(1)), right=ast.Constant(value=ast.ConstInt(2))
            )
        )
        assert instrs == (
            tac.JumpIfNotZero(condition=tac.Constant(value=ast.ConstInt(1)), target="or_true.0"),
            tac.JumpIfNotZero(condition=tac.Constant(value=ast.ConstInt(2)), target="or_true.0"),
            tac.Copy(src=tac.Constant(value=ast.ConstInt(0)), dst=tac.Var(value="tmp.0")),
            tac.Jump(target="or_end.1"),
            tac.Label(name="or_true.0"),
            tac.Copy(src=tac.Constant(value=ast.ConstInt(1)), dst=tac.Var(value="tmp.0")),
            tac.Label(name="or_end.1"),
            tac.Return(val=tac.Var(value="tmp.0")),
        )

    def test_right_operand_instructions_appear_after_first_jump(self) -> None:
        """Right sub-expression instructions are emitted after the first ``JumpIfNotZero``."""
        exp = ast.Binary(
            operator=ast.Or(),
            left=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(1))),
            right=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(2))),
        )
        instrs = _body(exp)
        assert isinstance(instrs[0], tac.Unary)  # left side: Negate(1)
        assert isinstance(instrs[1], tac.JumpIfNotZero)  # short-circuit check on left
        assert isinstance(instrs[2], tac.Unary)  # right side: Negate(2) - only after jump
        assert isinstance(instrs[3], tac.JumpIfNotZero)  # short-circuit check on right

    def test_result_is_returned(self) -> None:
        """The ``Return`` instruction uses the temporary written by the ``Or`` sequence."""
        instrs = _body(
            ast.Binary(
                operator=ast.Or(), left=ast.Constant(value=ast.ConstInt(1)), right=ast.Constant(value=ast.ConstInt(2))
            )
        )
        ret = instrs[-1]
        assert isinstance(ret, tac.Return)
        assert ret.val == tac.Var(value="tmp.0")

    def test_labels_are_unique_across_nested_or_expressions(self) -> None:
        """Nested ``||`` expressions each receive distinct label names."""
        exp = ast.Binary(
            operator=ast.Or(),
            left=ast.Binary(
                operator=ast.Or(), left=ast.Constant(value=ast.ConstInt(1)), right=ast.Constant(value=ast.ConstInt(2))
            ),
            right=ast.Constant(value=ast.ConstInt(3)),
        )
        instrs = _body(exp)
        label_names = [i.name for i in instrs if isinstance(i, tac.Label)]
        assert len(label_names) == len(set(label_names))
        assert len(label_names) == 4


# ---------------------------------------------------------------------------
# Variable expressions
# ---------------------------------------------------------------------------


class TestTacGeneratorVarExpression:
    """``ast.Var`` translates to a ``tac.Var`` with no extra instructions."""

    def test_var_produces_no_instructions(self) -> None:
        """A bare variable reference emits no instructions beyond the enclosing statement."""
        instrs = _instrs_for_body((ast.S(statement=ast.Return(exp=ast.Var(name="x"))),))
        assert len(instrs) == 1
        assert isinstance(instrs[0], tac.Return)

    def test_var_name_is_preserved(self) -> None:
        """The variable name is carried through to the ``tac.Var`` value."""
        instrs = _instrs_for_body((ast.S(statement=ast.Return(exp=ast.Var(name=".Vx.0"))),))
        assert isinstance(instrs[0], tac.Return)
        assert instrs[0].val == tac.Var(value=".Vx.0")


# ---------------------------------------------------------------------------
# Assignment expressions
# ---------------------------------------------------------------------------


class TestTacGeneratorAssignmentExpression:
    """``ast.Assignment`` emits a ``Copy`` and returns the lvalue variable."""

    def test_assignment_with_constant_rvalue_emits_copy(self) -> None:
        """``x = 5`` emits a ``Copy(Constant(5), Var("x"))`` instruction followed by the guard."""
        instrs = _instrs_for_body(
            (
                ast.S(
                    statement=ast.Expression(
                        exp=ast.Assignment(lvalue=ast.Var(name="x"), rvalue=ast.Constant(value=ast.ConstInt(5)))
                    )
                ),
            )
        )
        assert instrs == (
            tac.Copy(src=tac.Constant(value=ast.ConstInt(5)), dst=tac.Var(value="x")),
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
        )

    def test_assignment_result_is_the_lvalue_var(self) -> None:
        """The result of an assignment expression is the lvalue variable itself."""
        instrs = _instrs_for_body(
            (
                ast.S(
                    statement=ast.Return(
                        exp=ast.Assignment(lvalue=ast.Var(name="x"), rvalue=ast.Constant(value=ast.ConstInt(7)))
                    )
                ),
            )
        )
        assert isinstance(instrs[0], tac.Copy)
        assert isinstance(instrs[1], tac.Return)
        assert instrs[1].val == tac.Var(value="x")

    def test_assignment_with_expression_rvalue_emits_instructions_before_copy(self) -> None:
        """A non-trivial rvalue is fully evaluated before the ``Copy`` is emitted."""
        instrs = _instrs_for_body(
            (
                ast.S(
                    statement=ast.Expression(
                        exp=ast.Assignment(
                            lvalue=ast.Var(name="x"),
                            rvalue=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(1))),
                        )
                    )
                ),
            )
        )
        assert isinstance(instrs[0], tac.Unary)
        assert isinstance(instrs[1], tac.Copy)
        assert instrs[1].src == instrs[0].dst
        assert instrs[1].dst == tac.Var(value="x")


# ---------------------------------------------------------------------------
# Pre/post increment and decrement desugaring
# ---------------------------------------------------------------------------


class TestTacGeneratorIncDecOperators:
    """Pre/post increment and decrement desugar into Binary + Copy sequences."""

    def test_pre_increment_full_sequence(self) -> None:
        """``return ++x;`` -> ``Binary(Add, x, 1, tmp)``, ``Copy(tmp, x)``, ``Return(x)``."""
        instrs = _body(ast.Unary(operator=ast.PreIncrement(), exp=ast.Var(name="x")))
        assert instrs == (
            tac.Binary(
                operator=tac.Add(),
                src1=tac.Var(value="x"),
                src2=tac.Constant(value=ast.ConstInt(1)),
                dst=tac.Var(value="tmp.0"),
            ),
            tac.Copy(src=tac.Var(value="tmp.0"), dst=tac.Var(value="x")),
            tac.Return(val=tac.Var(value="x")),
        )

    def test_pre_decrement_full_sequence(self) -> None:
        """``return --x;`` -> ``Binary(Sub, x, 1, tmp)``, ``Copy(tmp, x)``, ``Return(x)``."""
        instrs = _body(ast.Unary(operator=ast.PreDecrement(), exp=ast.Var(name="x")))
        assert instrs == (
            tac.Binary(
                operator=tac.Subtract(),
                src1=tac.Var(value="x"),
                src2=tac.Constant(value=ast.ConstInt(1)),
                dst=tac.Var(value="tmp.0"),
            ),
            tac.Copy(src=tac.Var(value="tmp.0"), dst=tac.Var(value="x")),
            tac.Return(val=tac.Var(value="x")),
        )

    def test_post_increment_full_sequence(self) -> None:
        """``return x++;`` -> ``Copy(x, old)``, ``Binary(Add, x, 1, tmp)``, ``Copy(tmp, x)``, ``Return(old)``."""
        instrs = _body(ast.Unary(operator=ast.PostIncrement(), exp=ast.Var(name="x")))
        assert instrs == (
            tac.Copy(src=tac.Var(value="x"), dst=tac.Var(value="tmp.0")),
            tac.Binary(
                operator=tac.Add(),
                src1=tac.Var(value="x"),
                src2=tac.Constant(value=ast.ConstInt(1)),
                dst=tac.Var(value="tmp.1"),
            ),
            tac.Copy(src=tac.Var(value="tmp.1"), dst=tac.Var(value="x")),
            tac.Return(val=tac.Var(value="tmp.0")),
        )

    def test_post_decrement_full_sequence(self) -> None:
        """``return x--;`` -> ``Copy(x, old)``, ``Binary(Sub, x, 1, tmp)``, ``Copy(tmp, x)``, ``Return(old)``."""
        instrs = _body(ast.Unary(operator=ast.PostDecrement(), exp=ast.Var(name="x")))
        assert instrs == (
            tac.Copy(src=tac.Var(value="x"), dst=tac.Var(value="tmp.0")),
            tac.Binary(
                operator=tac.Subtract(),
                src1=tac.Var(value="x"),
                src2=tac.Constant(value=ast.ConstInt(1)),
                dst=tac.Var(value="tmp.1"),
            ),
            tac.Copy(src=tac.Var(value="tmp.1"), dst=tac.Var(value="x")),
            tac.Return(val=tac.Var(value="tmp.0")),
        )

    def test_pre_increment_returns_new_value(self) -> None:
        """``++x`` returns the variable itself (post-increment value)."""
        instrs = _body(ast.Unary(operator=ast.PreIncrement(), exp=ast.Var(name="x")))
        ret = instrs[-1]
        assert isinstance(ret, tac.Return)
        assert ret.val == tac.Var(value="x")

    def test_post_increment_returns_original_value(self) -> None:
        """``x++`` returns the saved copy, not the variable after increment."""
        instrs = _body(ast.Unary(operator=ast.PostIncrement(), exp=ast.Var(name="x")))
        ret = instrs[-1]
        assert isinstance(ret, tac.Return)
        assert ret.val == tac.Var(value="tmp.0")  # the saved copy, not Var("x")
        assert ret.val != tac.Var(value="x")


# ---------------------------------------------------------------------------
# Conditional (ternary) expressions
# ---------------------------------------------------------------------------


class TestTacGeneratorConditionalExpression:
    """``ast.Conditional`` translates to the correct ternary jump-and-copy sequence."""

    def test_full_instruction_sequence(self) -> None:
        """``return 1 ? 2 : 3;`` emits the full ternary sequence."""
        instrs = _body(
            ast.Conditional(
                condition=ast.Constant(value=ast.ConstInt(1)),
                exp_true=ast.Constant(value=ast.ConstInt(2)),
                exp_false=ast.Constant(value=ast.ConstInt(3)),
            )
        )
        assert instrs == (
            tac.JumpIfZero(condition=tac.Constant(value=ast.ConstInt(1)), target="cond_false.0"),
            tac.Copy(src=tac.Constant(value=ast.ConstInt(2)), dst=tac.Var(value="tmp.0")),
            tac.Jump(target="cond_end.1"),
            tac.Label(name="cond_false.0"),
            tac.Copy(src=tac.Constant(value=ast.ConstInt(3)), dst=tac.Var(value="tmp.0")),
            tac.Label(name="cond_end.1"),
            tac.Return(val=tac.Var(value="tmp.0")),
        )

    def test_condition_instructions_precede_jump(self) -> None:
        """A non-trivial condition is fully evaluated before the conditional jump."""
        instrs = _body(
            ast.Conditional(
                condition=ast.Unary(operator=ast.Not(), exp=ast.Constant(value=ast.ConstInt(1))),
                exp_true=ast.Constant(value=ast.ConstInt(2)),
                exp_false=ast.Constant(value=ast.ConstInt(3)),
            )
        )
        assert isinstance(instrs[0], tac.Unary)
        assert instrs[0].operator == tac.Not()
        assert isinstance(instrs[1], tac.JumpIfZero)
        assert instrs[1].condition == instrs[0].dst

    def test_true_branch_instructions_precede_copy(self) -> None:
        """A non-trivial true branch is evaluated before its result is copied into the destination."""
        instrs = _body(
            ast.Conditional(
                condition=ast.Constant(value=ast.ConstInt(1)),
                exp_true=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(2))),
                exp_false=ast.Constant(value=ast.ConstInt(3)),
            )
        )
        # JumpIfZero, then: Unary(Negate, 2, tmp), Copy(tmp, dst)
        assert isinstance(instrs[1], tac.Unary)
        assert isinstance(instrs[2], tac.Copy)
        assert instrs[2].src == instrs[1].dst

    def test_false_branch_instructions_precede_copy(self) -> None:
        """A non-trivial false branch is evaluated before its result is copied into the destination."""
        instrs = _body(
            ast.Conditional(
                condition=ast.Constant(value=ast.ConstInt(1)),
                exp_true=ast.Constant(value=ast.ConstInt(2)),
                exp_false=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(3))),
            )
        )
        # Locate the cond_false label, then check Unary followed by Copy
        false_label_idx = next(i for i, x in enumerate(instrs) if isinstance(x, tac.Label))
        assert isinstance(instrs[false_label_idx + 1], tac.Unary)
        assert isinstance(instrs[false_label_idx + 2], tac.Copy)
        assert instrs[false_label_idx + 2].src == instrs[false_label_idx + 1].dst

    def test_both_branches_write_to_same_destination(self) -> None:
        """Both the true and false branches copy their result into the same temporary."""
        instrs = _body(
            ast.Conditional(
                condition=ast.Constant(value=ast.ConstInt(1)),
                exp_true=ast.Constant(value=ast.ConstInt(2)),
                exp_false=ast.Constant(value=ast.ConstInt(3)),
            )
        )
        copies = [i for i in instrs if isinstance(i, tac.Copy)]
        assert len(copies) == 2
        assert copies[0].dst == copies[1].dst

    def test_result_is_returned(self) -> None:
        """The ``Return`` instruction uses the shared temporary written by both branches."""
        instrs = _body(
            ast.Conditional(
                condition=ast.Constant(value=ast.ConstInt(1)),
                exp_true=ast.Constant(value=ast.ConstInt(2)),
                exp_false=ast.Constant(value=ast.ConstInt(3)),
            )
        )
        copies = [i for i in instrs if isinstance(i, tac.Copy)]
        ret = instrs[-1]
        assert isinstance(ret, tac.Return)
        assert ret.val == copies[0].dst

    def test_labels_unique_across_nested_conditionals(self) -> None:
        """Nested ternary expressions each receive distinct label names."""
        inner = ast.Conditional(
            condition=ast.Constant(value=ast.ConstInt(2)),
            exp_true=ast.Constant(value=ast.ConstInt(3)),
            exp_false=ast.Constant(value=ast.ConstInt(4)),
        )
        outer = ast.Conditional(
            condition=ast.Constant(value=ast.ConstInt(1)),
            exp_true=inner,
            exp_false=ast.Constant(value=ast.ConstInt(5)),
        )
        instrs = _body(outer)
        label_names = [i.name for i in instrs if isinstance(i, tac.Label)]
        assert len(label_names) == len(set(label_names))
        assert len(label_names) == 4  # two labels per ternary


# ---------------------------------------------------------------------------
# String literal expressions (AddrOf(String(...)))
# ---------------------------------------------------------------------------


class TestTacGeneratorStringLiteralExpression:
    """String literals in expression context generate a ``StaticConstant`` and a ``GetAddress`` instruction."""

    def _string_as_expr(self, s: bytes) -> ast.AddrOf:
        """Build a type-annotated ``AddrOf(String(...))`` as produced by the type-checker.

        Args:
            s: Raw byte content of the string (NUL not included).

        Returns:
            An ``AddrOf`` node wrapping a ``String`` with array type, itself typed as
            ``Pointer(Char())``.
        """
        array_type = ast.Array(element=ast.Char(), size=len(s) + 1)
        return ast.AddrOf(
            exp=ast.String(s, exp_type=array_type),
            exp_type=ast.Pointer(ast.Char()),
        )

    def test_string_literal_emits_static_constant(self) -> None:
        """A string literal expression creates one ``StaticConstant`` in the program's top level."""
        sym: dict = {}
        prog = generate(_program_returning(self._string_as_expr(b"hello")), sym)
        static_consts = [t for t in prog.top_level if isinstance(t, tac.StaticConstant)]
        assert len(static_consts) == 1
        assert static_consts[0].init == tac.StringInit(b"hello", True)

    def test_string_literal_static_constant_has_correct_type(self) -> None:
        """The ``StaticConstant`` for a string has type ``Array(Char(), len+1)``."""
        sym: dict = {}
        prog = generate(_program_returning(self._string_as_expr(b"hi")), sym)
        static_consts = [t for t in prog.top_level if isinstance(t, tac.StaticConstant)]
        assert static_consts[0].t == ast.Array(element=ast.Char(), size=3)

    def test_string_literal_emits_get_address(self) -> None:
        """A string literal expression emits a ``GetAddress`` instruction in the function body."""
        sym: dict = {}
        prog = generate(_program_returning(self._string_as_expr(b"hello")), sym)
        static_consts = [t for t in prog.top_level if isinstance(t, tac.StaticConstant)]
        fn_body = prog.functions[0].body
        get_addr = fn_body[0]
        assert isinstance(get_addr, tac.GetAddress)
        assert get_addr.src == tac.Var(static_consts[0].name)

    def test_string_literal_return_value_is_get_address_destination(self) -> None:
        """The ``Return`` instruction uses the temporary written by ``GetAddress``."""
        sym: dict = {}
        prog = generate(_program_returning(self._string_as_expr(b"hello")), sym)
        fn_body = prog.functions[0].body
        get_addr = fn_body[0]
        ret = fn_body[1]
        assert isinstance(get_addr, tac.GetAddress)
        assert isinstance(ret, tac.Return)
        assert ret.val == get_addr.dst

    def test_two_distinct_string_literals_create_two_static_constants(self) -> None:  # noqa: E501
        """Two different string literal expressions each get their own ``StaticConstant``."""
        sym: dict = {}
        prog = generate(
            _program_returning(
                ast.Conditional(
                    condition=ast.Constant(ast.ConstInt(1)),
                    exp_true=self._string_as_expr(b"foo"),
                    exp_false=self._string_as_expr(b"bar"),
                )
            ),
            sym,
        )
        static_consts = [t for t in prog.top_level if isinstance(t, tac.StaticConstant)]
        assert len(static_consts) == 2
        inits = {c.init for c in static_consts}
        assert tac.StringInit(b"foo", True) in inits
        assert tac.StringInit(b"bar", True) in inits


# ---------------------------------------------------------------------------
# Float type casts
# ---------------------------------------------------------------------------


class TestFloatCasts:
    """Float cast expressions produce the correct conversion TAC instructions."""

    def _cast(self, target: ast.Type, src_type: ast.Type, const: ast.Const) -> tuple[tac.Instruction, ...]:
        """Build a cast expression and return the generated TAC instructions."""
        inner = ast.Constant(value=const, exp_type=src_type)
        cast_exp = ast.Cast(target_type=target, exp=inner, exp_type=target)
        return _body(cast_exp)

    def test_int_to_float_produces_int_to_float(self) -> None:
        """``(float)int_val`` emits ``IntToFloat`` instruction."""

        instrs = self._cast(ast.FloatType(), ast.IntType(), ast.ConstInt(3))
        assert any(isinstance(i, tac.IntToFloat) for i in instrs)

    def test_float_to_int_produces_float_to_int(self) -> None:
        """``(int)float_val`` emits ``FloatToInt`` instruction."""
        instrs = self._cast(ast.IntType(), ast.FloatType(), ConstFloat(3.0))
        assert any(isinstance(i, tac.FloatToInt) for i in instrs)

    def test_float_to_double_produces_float_to_double(self) -> None:
        """``(double)float_val`` emits ``FloatToDouble`` instruction."""
        instrs = self._cast(ast.DoubleType(), ast.FloatType(), ConstFloat(3.0))
        assert any(isinstance(i, tac.FloatToDouble) for i in instrs)

    def test_double_to_float_produces_double_to_float(self) -> None:
        """``(float)double_val`` emits ``DoubleToFloat`` instruction."""
        instrs = self._cast(ast.FloatType(), ast.DoubleType(), ast.ConstDouble(3.0))
        assert any(isinstance(i, tac.DoubleToFloat) for i in instrs)

    def test_uint_to_float_produces_uint_to_float(self) -> None:
        """``(float)uint_val`` emits ``UIntToFloat`` instruction."""
        instrs = self._cast(ast.FloatType(), ast.UIntType(), ast.ConstUInt(3))
        assert any(isinstance(i, tac.UIntToFloat) for i in instrs)

    def test_float_to_uint_produces_float_to_uint(self) -> None:
        """``(unsigned)float_val`` emits ``FloatToUInt`` instruction."""
        instrs = self._cast(ast.UIntType(), ast.FloatType(), ConstFloat(3.0))
        assert any(isinstance(i, tac.FloatToUInt) for i in instrs)

    def test_int_to_long_double_produces_int_to_long_double(self) -> None:
        """``(long double)int_val`` emits ``IntToLongDouble``."""
        instrs = self._cast(LongDoubleType(), ast.IntType(), ast.ConstInt(3))
        assert any(isinstance(i, tac.IntToLongDouble) for i in instrs)

    def test_uint_to_long_double_produces_uint_to_long_double(self) -> None:
        """``(long double)uint_val`` emits ``UIntToLongDouble``."""
        instrs = self._cast(LongDoubleType(), ast.UIntType(), ast.ConstUInt(3))
        assert any(isinstance(i, tac.UIntToLongDouble) for i in instrs)

    def test_long_double_to_int_produces_long_double_to_int(self) -> None:
        """``(int)long_double_val`` emits ``LongDoubleToInt``."""
        instrs = self._cast(ast.IntType(), LongDoubleType(), ConstLongDouble(3.0))
        assert any(isinstance(i, tac.LongDoubleToInt) for i in instrs)

    def test_long_double_to_uint_produces_long_double_to_uint(self) -> None:
        """``(unsigned)long_double_val`` emits ``LongDoubleToUInt``."""
        instrs = self._cast(ast.UIntType(), LongDoubleType(), ConstLongDouble(3.0))
        assert any(isinstance(i, tac.LongDoubleToUInt) for i in instrs)

    def test_float_to_long_double_produces_float_to_long_double(self) -> None:
        """``(long double)float_val`` emits ``FloatToLongDouble``."""
        instrs = self._cast(LongDoubleType(), ast.FloatType(), ConstFloat(3.0))
        assert any(isinstance(i, tac.FloatToLongDouble) for i in instrs)

    def test_long_double_to_float_produces_long_double_to_float(self) -> None:
        """``(float)long_double_val`` emits ``LongDoubleToFloat``."""
        instrs = self._cast(ast.FloatType(), LongDoubleType(), ConstLongDouble(3.0))
        assert any(isinstance(i, tac.LongDoubleToFloat) for i in instrs)

    def test_double_to_long_double_produces_double_to_long_double(self) -> None:
        """``(long double)double_val`` emits ``DoubleToLongDouble``."""
        instrs = self._cast(LongDoubleType(), ast.DoubleType(), ast.ConstDouble(3.0))
        assert any(isinstance(i, tac.DoubleToLongDouble) for i in instrs)

    def test_long_double_to_double_produces_long_double_to_double(self) -> None:
        """``(double)long_double_val`` emits ``LongDoubleToDouble``."""
        instrs = self._cast(ast.DoubleType(), LongDoubleType(), ConstLongDouble(3.0))
        assert any(isinstance(i, tac.LongDoubleToDouble) for i in instrs)


class TestBoolCasts:
    """Casts to ``_Bool`` normalise via JumpIfZero rather than plain truncation."""

    def _cast(self, target: ast.Type, src_type: ast.Type, const: ast.Const) -> tuple[tac.Instruction, ...]:
        """Build a cast expression and return the generated TAC instructions."""
        inner = ast.Constant(value=const, exp_type=src_type)
        cast_exp = ast.Cast(target_type=target, exp=inner, exp_type=target)
        return _body(cast_exp)

    def test_int_to_bool_uses_jump_if_zero(self) -> None:
        """``(_Bool)int_val`` emits a ``JumpIfZero`` normalisation sequence."""
        instrs = self._cast(BoolType(), ast.IntType(), ast.ConstInt(5))
        assert any(isinstance(i, tac.JumpIfZero) for i in instrs)

    def test_int_to_bool_no_truncate(self) -> None:
        """``(_Bool)int_val`` must NOT emit a bare ``Truncate`` (which would not normalise)."""
        instrs = self._cast(BoolType(), ast.IntType(), ast.ConstInt(5))
        assert not any(isinstance(i, tac.Truncate) for i in instrs)

    def test_int_to_bool_copies_one_on_nonzero(self) -> None:
        """The true branch of the ``_Bool`` normalisation copies the constant 1."""
        instrs = self._cast(BoolType(), ast.IntType(), ast.ConstInt(5))
        copies = [i for i in instrs if isinstance(i, tac.Copy)]
        values = [i.src for i in copies if isinstance(i, tac.Copy)]
        assert any(v == tac.Constant(ast.ConstBool(1)) for v in values)

    def test_long_to_bool_uses_jump_if_zero(self) -> None:
        """``(_Bool)long_val`` also uses the normalise-via-jump path."""
        instrs = self._cast(BoolType(), ast.LongType(), ast.ConstLong(0))
        assert any(isinstance(i, tac.JumpIfZero) for i in instrs)


class TestAlignOfTacGen:
    """``_Alignof`` and ``_Alignof(T)`` translate to TAC constants."""

    def test_alignof_t_int_returns_constant_4(self) -> None:
        """``_Alignof(int)`` translates to ``Constant(ConstULong(4))``."""
        exp = ast.AlignOfT(t=ast.IntType(), exp_type=ast.ULongType())
        instrs = _body(exp)
        assert instrs == (tac.Return(val=tac.Constant(value=ast.ConstULong(4))),)

    def test_alignof_t_double_returns_constant_8(self) -> None:
        """``_Alignof(double)`` translates to ``Constant(ConstULong(8))``."""
        exp = ast.AlignOfT(t=ast.DoubleType(), exp_type=ast.ULongType())
        instrs = _body(exp)
        assert instrs == (tac.Return(val=tac.Constant(value=ast.ConstULong(8))),)

    def test_alignof_t_long_returns_constant_8(self) -> None:
        """``_Alignof(long)`` translates to ``Constant(ConstULong(8))``."""
        exp = ast.AlignOfT(t=ast.LongType(), exp_type=ast.ULongType())
        instrs = _body(exp)
        assert instrs == (tac.Return(val=tac.Constant(value=ast.ConstULong(8))),)

    def test_alignof_t_char_returns_constant_1(self) -> None:
        """``_Alignof(char)`` translates to ``Constant(ConstULong(1))``."""
        exp = ast.AlignOfT(t=ast.Char(), exp_type=ast.ULongType())
        instrs = _body(exp)
        assert instrs == (tac.Return(val=tac.Constant(value=ast.ConstULong(1))),)

    def test_alignof_expression_uses_exp_type(self) -> None:
        """``_Alignof x`` for an ``int`` var translates to ``Constant(ConstULong(4))``."""
        inner = ast.Var(name="x", exp_type=ast.IntType())
        exp = ast.AlignOf(exp=inner, exp_type=ast.ULongType())
        instrs = _body(exp)
        assert instrs == (tac.Return(val=tac.Constant(value=ast.ConstULong(4))),)
