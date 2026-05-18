"""Tests for expression-level TAC generation."""

import pytest

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from compiler.tac_generator import generate
from compiler.utils.exceptions import CodeGenerationError
from test.tac_generator.helpers import _body, _instrs_for_body, _program_returning

# ---------------------------------------------------------------------------
# Constant and unary (regression — must not be broken by binary additions)
# ---------------------------------------------------------------------------


class TestTacGeneratorExistingConstructs:
    """Constant and unary constructs continue to translate correctly."""

    def test_constant_produces_only_return(self) -> None:
        """``return 42;`` → ``Return(Constant(42))`` then the function guard."""
        instrs = _body(ast.Constant(value=ast.ConstInt(42)))
        assert instrs == (
            tac.Return(val=tac.Constant(value=ast.ConstInt(42))),
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
        )

    def test_unary_negate_produces_unary_then_return(self) -> None:
        """``return -1;`` → ``Unary(Negate, Constant(1), tmp.0)`` + ``Return(tmp.0)`` + guard."""
        instrs = _body(ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(1))))
        assert instrs == (
            tac.Unary(operator=tac.Negate(), src=tac.Constant(value=ast.ConstInt(1)), dst=tac.Var(value="tmp.0")),
            tac.Return(val=tac.Var(value="tmp.0")),
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
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
        """``return 1 + 2;`` → ``Binary(Add, Constant(1), Constant(2), tmp.0)`` + ``Return(tmp.0)`` + guard."""
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
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
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
        # Unary(Negate, 1, tmp.0), Unary(Negate, 2, tmp.1), Binary(Add, tmp.0, tmp.1, tmp.2), Return(tmp.2), guard
        assert len(instrs) == 5
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
        # Binary(Multiply, 2, 3, tmp.0), Binary(Add, 1, tmp.0, tmp.1), Return(tmp.1), guard
        assert len(instrs) == 4
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
        """Temporaries are named ``tmp.0``, ``tmp.1``, … in emission order."""
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
        """``return ~1;`` → ``Unary(Complement, Constant(1), tmp.0)``."""
        instrs = _body(ast.Unary(operator=ast.Complement(), exp=ast.Constant(value=ast.ConstInt(1))))
        assert isinstance(instrs[0], tac.Unary)
        assert instrs[0].operator == tac.Complement()

    def test_not_produces_unary_not(self) -> None:
        """``return !1;`` → ``Unary(Not, Constant(1), tmp.0)``."""
        instrs = _body(ast.Unary(operator=ast.Not(), exp=ast.Constant(value=ast.ConstInt(1))))
        assert isinstance(instrs[0], tac.Unary)
        assert instrs[0].operator == tac.Not()


# ---------------------------------------------------------------------------
# Logical AND short-circuit
# ---------------------------------------------------------------------------


class TestTacGeneratorAndExpression:
    """``ast.And`` produces a short-circuit JumpIfZero sequence."""

    def test_full_instruction_sequence(self) -> None:
        """``return 1 && 2;`` emits the full short-circuit sequence followed by the guard."""
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
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
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
        assert isinstance(instrs[2], tac.Unary)  # right side: Negate(2) — only after the jump
        assert isinstance(instrs[3], tac.JumpIfZero)  # short-circuit check on right

    def test_result_is_returned(self) -> None:
        """The ``Return`` instruction uses the temporary written by the ``And`` sequence."""
        instrs = _body(
            ast.Binary(
                operator=ast.And(), left=ast.Constant(value=ast.ConstInt(1)), right=ast.Constant(value=ast.ConstInt(2))
            )
        )
        ret = instrs[-2]
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
        """``return 1 || 2;`` emits the full short-circuit sequence followed by the guard."""
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
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
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
        assert isinstance(instrs[2], tac.Unary)  # right side: Negate(2) — only after jump
        assert isinstance(instrs[3], tac.JumpIfNotZero)  # short-circuit check on right

    def test_result_is_returned(self) -> None:
        """The ``Return`` instruction uses the temporary written by the ``Or`` sequence."""
        instrs = _body(
            ast.Binary(
                operator=ast.Or(), left=ast.Constant(value=ast.ConstInt(1)), right=ast.Constant(value=ast.ConstInt(2))
            )
        )
        ret = instrs[-2]
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
        assert len(instrs) == 2
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
        """``return ++x;`` → ``Binary(Add, x, 1, tmp)``, ``Copy(tmp, x)``, ``Return(x)``."""
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
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
        )

    def test_pre_decrement_full_sequence(self) -> None:
        """``return --x;`` → ``Binary(Sub, x, 1, tmp)``, ``Copy(tmp, x)``, ``Return(x)``."""
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
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
        )

    def test_post_increment_full_sequence(self) -> None:
        """``return x++;`` → ``Copy(x, old)``, ``Binary(Add, x, 1, tmp)``, ``Copy(tmp, x)``, ``Return(old)``."""
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
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
        )

    def test_post_decrement_full_sequence(self) -> None:
        """``return x--;`` → ``Copy(x, old)``, ``Binary(Sub, x, 1, tmp)``, ``Copy(tmp, x)``, ``Return(old)``."""
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
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
        )

    def test_pre_increment_returns_new_value(self) -> None:
        """``++x`` returns the variable itself (post-increment value)."""
        instrs = _body(ast.Unary(operator=ast.PreIncrement(), exp=ast.Var(name="x")))
        ret = instrs[-2]
        assert isinstance(ret, tac.Return)
        assert ret.val == tac.Var(value="x")

    def test_post_increment_returns_original_value(self) -> None:
        """``x++`` returns the saved copy, not the variable after increment."""
        instrs = _body(ast.Unary(operator=ast.PostIncrement(), exp=ast.Var(name="x")))
        ret = instrs[-2]
        assert isinstance(ret, tac.Return)
        assert ret.val == tac.Var(value="tmp.0")  # the saved copy, not Var("x")
        assert ret.val != tac.Var(value="x")


# ---------------------------------------------------------------------------
# Conditional (ternary) expressions
# ---------------------------------------------------------------------------


class TestTacGeneratorConditionalExpression:
    """``ast.Conditional`` translates to the correct ternary jump-and-copy sequence."""

    def test_full_instruction_sequence(self) -> None:
        """``return 1 ? 2 : 3;`` emits the full ternary sequence followed by the guard."""
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
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
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
        ret = instrs[-2]  # penultimate; last is the function guard
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
