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
        declarations=(
            ast.FunDecl(
                ast.Function(
                    name="main",
                    params=(),
                    body=(ast.S(statement=ast.Return(exp=exp)),),
                )
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
    return generate(_program_returning(exp)).functions[0].body


# ---------------------------------------------------------------------------
# Constant and unary (regression — must not be broken by binary additions)
# ---------------------------------------------------------------------------


class TestTacGeneratorExistingConstructs:
    """Constant and unary constructs continue to translate correctly."""

    def test_constant_produces_only_return(self) -> None:
        """``return 42;`` → ``Return(Constant(42))`` then the function guard."""
        instrs = _body(ast.Constant(value=42))
        assert instrs == (
            tac.Return(val=tac.Constant(value=42)),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_unary_negate_produces_unary_then_return(self) -> None:
        """``return -1;`` → ``Unary(Negate, Constant(1), tmp.0)`` + ``Return(tmp.0)`` + guard."""
        instrs = _body(ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=1)))
        assert instrs == (
            tac.Unary(operator=tac.Negate(), src=tac.Constant(value=1), dst=tac.Var(value="tmp.0")),
            tac.Return(val=tac.Var(value="tmp.0")),
            tac.Return(val=tac.Constant(value=0)),
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
        """``return 1 + 2;`` → ``Binary(Add, Constant(1), Constant(2), tmp.0)`` + ``Return(tmp.0)`` + guard."""
        instrs = _body(ast.Binary(operator=ast.Add(), left=ast.Constant(value=1), right=ast.Constant(value=2)))
        assert instrs == (
            tac.Binary(
                operator=tac.Add(),
                src1=tac.Constant(value=1),
                src2=tac.Constant(value=2),
                dst=tac.Var(value="tmp.0"),
            ),
            tac.Return(val=tac.Var(value="tmp.0")),
            tac.Return(val=tac.Constant(value=0)),
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
        # Unary(Negate, 1, tmp.0), Unary(Negate, 2, tmp.1), Binary(Add, tmp.0, tmp.1, tmp.2), Return(tmp.2), guard
        assert len(instrs) == 5
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
        # Binary(Multiply, 2, 3, tmp.0), Binary(Add, 1, tmp.0, tmp.1), Return(tmp.1), guard
        assert len(instrs) == 4
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
            name="main",
            params=(),
            body=(ast.S(statement=_FakeStmt()),),
        )
        with pytest.raises(CodeGenerationError):
            generate(ast.Program(declarations=(ast.FunDecl(fn),)))


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
        """``return 1 && 2;`` emits the full short-circuit sequence followed by the guard."""
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
            tac.Return(val=tac.Constant(value=0)),
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
        ret = instrs[-2]
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
        """``return 1 || 2;`` emits the full short-circuit sequence followed by the guard."""
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
            tac.Return(val=tac.Constant(value=0)),
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
        ret = instrs[-2]
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


# ---------------------------------------------------------------------------
# Helpers for body-level tests (declarations + statements)
# ---------------------------------------------------------------------------


def _instrs_for_body(body: tuple[ast.BlockItem, ...]) -> tuple[tac.Instruction, ...]:
    """Translate a full function body and return the resulting TAC instructions.

    Args:
        body: The block items to place in the body of ``main``.

    Returns:
        The TAC instruction tuple from the generated function body.
    """
    return (
        generate(
            ast.Program(
                declarations=(
                    ast.FunDecl(
                        ast.Function(
                            name="main",
                            params=(),
                            body=body,
                        )
                    ),
                )
            )
        )
        .functions[0]
        .body
    )


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
                    statement=ast.Expression(exp=ast.Assignment(lvalue=ast.Var(name="x"), rvalue=ast.Constant(value=5)))
                ),
            )
        )
        assert instrs == (
            tac.Copy(src=tac.Constant(value=5), dst=tac.Var(value="x")),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_assignment_result_is_the_lvalue_var(self) -> None:
        """The result of an assignment expression is the lvalue variable itself."""
        instrs = _instrs_for_body(
            (ast.S(statement=ast.Return(exp=ast.Assignment(lvalue=ast.Var(name="x"), rvalue=ast.Constant(value=7)))),)
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
                            rvalue=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=1)),
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
# Null statement
# ---------------------------------------------------------------------------


class TestTacGeneratorNullStatement:
    """``ast.Null`` emits no instructions of its own."""

    def test_null_statement_produces_no_instructions(self) -> None:
        """A ``Null`` statement contributes nothing; only the function guard is present."""
        instrs = _instrs_for_body((ast.S(statement=ast.Null()),))
        assert instrs == (tac.Return(val=tac.Constant(value=0)),)


# ---------------------------------------------------------------------------
# Expression statement
# ---------------------------------------------------------------------------


class TestTacGeneratorExpressionStatement:
    """``ast.Expression`` emits the expression's instructions but discards the result."""

    def test_expression_statement_with_constant_produces_no_instructions(self) -> None:
        """A constant expression statement has no side-effects; only the function guard is present."""
        instrs = _instrs_for_body((ast.S(statement=ast.Expression(exp=ast.Constant(value=42))),))
        assert instrs == (tac.Return(val=tac.Constant(value=0)),)

    def test_expression_statement_emits_expression_instructions(self) -> None:
        """A non-trivial expression statement emits the expression's instructions."""
        instrs = _instrs_for_body(
            (ast.S(statement=ast.Expression(exp=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=1)))),)
        )
        assert len(instrs) == 2
        assert isinstance(instrs[0], tac.Unary)

    def test_expression_statement_result_is_not_returned(self) -> None:
        """An expression statement does not return the expression's computed value."""
        instrs = _instrs_for_body(
            (ast.S(statement=ast.Expression(exp=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=1)))),)
        )
        assert isinstance(instrs[0], tac.Unary)
        expr_result = instrs[0].dst
        assert not any(isinstance(i, tac.Return) and i.val == expr_result for i in instrs)


# ---------------------------------------------------------------------------
# Declaration block items
# ---------------------------------------------------------------------------


class TestTacGeneratorDeclarations:
    """``ast.D`` emits a ``Copy`` only when an initialiser is present."""

    def test_declaration_without_init_produces_no_instructions(self) -> None:
        """A bare ``int x;`` declaration emits no TAC instructions; only the function guard is present."""
        instrs = _instrs_for_body(
            (ast.D(declaration=ast.VarDecl(variable_declaration=ast.VariableDeclaration(name="x", init=None))),)
        )
        assert instrs == (tac.Return(val=tac.Constant(value=0)),)

    def test_declaration_with_constant_init_produces_copy(self) -> None:
        """``int x = 5;`` emits ``Copy(Constant(5), Var("x"))`` followed by the guard."""
        instrs = _instrs_for_body(
            (
                ast.D(
                    declaration=ast.VarDecl(
                        variable_declaration=ast.VariableDeclaration(name="x", init=ast.Constant(value=5))
                    )
                ),
            )
        )
        assert instrs == (
            tac.Copy(src=tac.Constant(value=5), dst=tac.Var(value="x")),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_declaration_copy_destination_is_the_declared_name(self) -> None:
        """The ``Copy`` destination uses the declaration's name verbatim."""
        instrs = _instrs_for_body(
            (
                ast.D(
                    declaration=ast.VarDecl(
                        variable_declaration=ast.VariableDeclaration(name=".Vmy_var.3", init=ast.Constant(value=0))
                    )
                ),
            )
        )
        assert isinstance(instrs[0], tac.Copy)
        assert instrs[0].dst == tac.Var(value=".Vmy_var.3")

    def test_declaration_with_expression_init_emits_instructions_then_copy(self) -> None:
        """``int x = -1;`` emits the negation instruction followed by a ``Copy``."""
        instrs = _instrs_for_body(
            (
                ast.D(
                    declaration=ast.VarDecl(
                        variable_declaration=ast.VariableDeclaration(
                            name="x",
                            init=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=1)),
                        )
                    )
                ),
            )
        )
        assert len(instrs) == 3
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
                operator=tac.Add(), src1=tac.Var(value="x"), src2=tac.Constant(value=1), dst=tac.Var(value="tmp.0")
            ),
            tac.Copy(src=tac.Var(value="tmp.0"), dst=tac.Var(value="x")),
            tac.Return(val=tac.Var(value="x")),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_pre_decrement_full_sequence(self) -> None:
        """``return --x;`` → ``Binary(Sub, x, 1, tmp)``, ``Copy(tmp, x)``, ``Return(x)``."""
        instrs = _body(ast.Unary(operator=ast.PreDecrement(), exp=ast.Var(name="x")))
        assert instrs == (
            tac.Binary(
                operator=tac.Subtract(), src1=tac.Var(value="x"), src2=tac.Constant(value=1), dst=tac.Var(value="tmp.0")
            ),
            tac.Copy(src=tac.Var(value="tmp.0"), dst=tac.Var(value="x")),
            tac.Return(val=tac.Var(value="x")),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_post_increment_full_sequence(self) -> None:
        """``return x++;`` → ``Copy(x, old)``, ``Binary(Add, x, 1, tmp)``, ``Copy(tmp, x)``, ``Return(old)``."""
        instrs = _body(ast.Unary(operator=ast.PostIncrement(), exp=ast.Var(name="x")))
        assert instrs == (
            tac.Copy(src=tac.Var(value="x"), dst=tac.Var(value="tmp.0")),
            tac.Binary(
                operator=tac.Add(), src1=tac.Var(value="x"), src2=tac.Constant(value=1), dst=tac.Var(value="tmp.1")
            ),
            tac.Copy(src=tac.Var(value="tmp.1"), dst=tac.Var(value="x")),
            tac.Return(val=tac.Var(value="tmp.0")),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_post_decrement_full_sequence(self) -> None:
        """``return x--;`` → ``Copy(x, old)``, ``Binary(Sub, x, 1, tmp)``, ``Copy(tmp, x)``, ``Return(old)``."""
        instrs = _body(ast.Unary(operator=ast.PostDecrement(), exp=ast.Var(name="x")))
        assert instrs == (
            tac.Copy(src=tac.Var(value="x"), dst=tac.Var(value="tmp.0")),
            tac.Binary(
                operator=tac.Subtract(), src1=tac.Var(value="x"), src2=tac.Constant(value=1), dst=tac.Var(value="tmp.1")
            ),
            tac.Copy(src=tac.Var(value="tmp.1"), dst=tac.Var(value="x")),
            tac.Return(val=tac.Var(value="tmp.0")),
            tac.Return(val=tac.Constant(value=0)),
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
# If statement
# ---------------------------------------------------------------------------


def _if_instrs(stmt: ast.If) -> tuple[tac.Instruction, ...]:
    """Translate a single ``If`` statement and return the resulting TAC instructions.

    Args:
        stmt: The ``If`` node to translate.

    Returns:
        The TAC instruction tuple from the generated function body.
    """
    return _instrs_for_body((ast.S(statement=stmt),))


class TestTacGeneratorIfStatement:
    """``ast.If`` translates to the correct conditional jump sequence."""

    def test_if_without_else_full_sequence(self) -> None:
        """``if (1) return 2;`` → ``JumpIfZero``, then-body, end-label, guard."""
        instrs = _if_instrs(
            ast.If(
                condition=ast.Constant(value=1),
                then=ast.Return(exp=ast.Constant(value=2)),
                else_=None,
            )
        )
        assert instrs == (
            tac.JumpIfZero(condition=tac.Constant(value=1), target="if_end.0"),
            tac.Return(val=tac.Constant(value=2)),
            tac.Label(name="if_end.0"),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_if_with_else_full_sequence(self) -> None:
        """``if (1) return 2; else return 3;`` → ``JumpIfZero``,
        then-body, ``Jump``, else-label, else-body, end-label, guard."""
        instrs = _if_instrs(
            ast.If(
                condition=ast.Constant(value=1),
                then=ast.Return(exp=ast.Constant(value=2)),
                else_=ast.Return(exp=ast.Constant(value=3)),
            )
        )
        assert instrs == (
            tac.JumpIfZero(condition=tac.Constant(value=1), target="if_else.0"),
            tac.Return(val=tac.Constant(value=2)),
            tac.Jump(target="if_end.1"),
            tac.Label(name="if_else.0"),
            tac.Return(val=tac.Constant(value=3)),
            tac.Label(name="if_end.1"),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_condition_instructions_precede_jump(self) -> None:
        """A non-trivial condition is fully evaluated before the conditional jump."""
        instrs = _if_instrs(
            ast.If(
                condition=ast.Unary(operator=ast.Not(), exp=ast.Constant(value=1)),
                then=ast.Return(exp=ast.Constant(value=0)),
                else_=None,
            )
        )
        assert isinstance(instrs[0], tac.Unary)
        assert instrs[0].operator == tac.Not()
        assert isinstance(instrs[1], tac.JumpIfZero)
        assert instrs[1].condition == instrs[0].dst

    def test_null_then_clause_emits_no_then_instructions(self) -> None:
        """``if (1);`` — a ``Null`` then-branch contributes no instructions between the jump and the end-label."""
        instrs = _if_instrs(
            ast.If(
                condition=ast.Constant(value=1),
                then=ast.Null(),
                else_=None,
            )
        )
        assert instrs == (
            tac.JumpIfZero(condition=tac.Constant(value=1), target="if_end.0"),
            tac.Label(name="if_end.0"),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_null_else_clause_emits_no_else_instructions(self) -> None:
        """``if (1) return 2; else;`` — a ``Null`` else-branch produces adjacent labels."""
        instrs = _if_instrs(
            ast.If(
                condition=ast.Constant(value=1),
                then=ast.Return(exp=ast.Constant(value=2)),
                else_=ast.Null(),
            )
        )
        assert instrs == (
            tac.JumpIfZero(condition=tac.Constant(value=1), target="if_else.0"),
            tac.Return(val=tac.Constant(value=2)),
            tac.Jump(target="if_end.1"),
            tac.Label(name="if_else.0"),
            tac.Label(name="if_end.1"),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_labels_are_unique_across_nested_if_statements(self) -> None:
        """Nested ``if`` statements each receive distinct label names."""
        inner = ast.If(
            condition=ast.Constant(value=2),
            then=ast.Return(exp=ast.Constant(value=1)),
            else_=None,
        )
        outer = ast.If(
            condition=ast.Constant(value=1),
            then=inner,
            else_=None,
        )
        instrs = _if_instrs(outer)
        label_names = [i.name for i in instrs if isinstance(i, tac.Label)]
        assert len(label_names) == len(set(label_names))
        assert len(label_names) == 2  # one end-label per if-without-else


# ---------------------------------------------------------------------------
# Conditional (ternary) expressions
# ---------------------------------------------------------------------------


class TestTacGeneratorConditionalExpression:
    """``ast.Conditional`` translates to the correct ternary jump-and-copy sequence."""

    def test_full_instruction_sequence(self) -> None:
        """``return 1 ? 2 : 3;`` emits the full ternary sequence followed by the guard."""
        instrs = _body(
            ast.Conditional(
                condition=ast.Constant(value=1),
                exp_true=ast.Constant(value=2),
                exp_false=ast.Constant(value=3),
            )
        )
        assert instrs == (
            tac.JumpIfZero(condition=tac.Constant(value=1), target="cond_false.0"),
            tac.Copy(src=tac.Constant(value=2), dst=tac.Var(value="tmp.0")),
            tac.Jump(target="cond_end.1"),
            tac.Label(name="cond_false.0"),
            tac.Copy(src=tac.Constant(value=3), dst=tac.Var(value="tmp.0")),
            tac.Label(name="cond_end.1"),
            tac.Return(val=tac.Var(value="tmp.0")),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_condition_instructions_precede_jump(self) -> None:
        """A non-trivial condition is fully evaluated before the conditional jump."""
        instrs = _body(
            ast.Conditional(
                condition=ast.Unary(operator=ast.Not(), exp=ast.Constant(value=1)),
                exp_true=ast.Constant(value=2),
                exp_false=ast.Constant(value=3),
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
                condition=ast.Constant(value=1),
                exp_true=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=2)),
                exp_false=ast.Constant(value=3),
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
                condition=ast.Constant(value=1),
                exp_true=ast.Constant(value=2),
                exp_false=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=3)),
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
                condition=ast.Constant(value=1),
                exp_true=ast.Constant(value=2),
                exp_false=ast.Constant(value=3),
            )
        )
        copies = [i for i in instrs if isinstance(i, tac.Copy)]
        assert len(copies) == 2
        assert copies[0].dst == copies[1].dst

    def test_result_is_returned(self) -> None:
        """The ``Return`` instruction uses the shared temporary written by both branches."""
        instrs = _body(
            ast.Conditional(
                condition=ast.Constant(value=1),
                exp_true=ast.Constant(value=2),
                exp_false=ast.Constant(value=3),
            )
        )
        copies = [i for i in instrs if isinstance(i, tac.Copy)]
        ret = instrs[-2]  # penultimate; last is the function guard
        assert isinstance(ret, tac.Return)
        assert ret.val == copies[0].dst

    def test_labels_unique_across_nested_conditionals(self) -> None:
        """Nested ternary expressions each receive distinct label names."""
        inner = ast.Conditional(
            condition=ast.Constant(value=2),
            exp_true=ast.Constant(value=3),
            exp_false=ast.Constant(value=4),
        )
        outer = ast.Conditional(
            condition=ast.Constant(value=1),
            exp_true=inner,
            exp_false=ast.Constant(value=5),
        )
        instrs = _body(outer)
        label_names = [i.name for i in instrs if isinstance(i, tac.Label)]
        assert len(label_names) == len(set(label_names))
        assert len(label_names) == 4  # two labels per ternary


# ---------------------------------------------------------------------------
# Helpers for goto / label tests
# ---------------------------------------------------------------------------


def _stmt_instrs(stmt: ast.Statement) -> tuple[tac.Instruction, ...]:
    """Translate a single statement and return the resulting TAC instruction body.

    Args:
        stmt: The statement node to translate.

    Returns:
        The TAC instruction tuple from the generated function body.
    """
    return _instrs_for_body((ast.S(statement=stmt),))


# ---------------------------------------------------------------------------
# Goto statement
# ---------------------------------------------------------------------------


class TestTacGeneratorGotoStatement:
    """``ast.Goto`` translates to an unconditional ``tac.Jump``."""

    def test_goto_produces_jump(self) -> None:
        """``goto foo;`` → ``Jump("foo")`` followed by the function guard."""
        instrs = _stmt_instrs(ast.Goto(target="foo"))
        assert instrs == (
            tac.Jump(target="foo"),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_goto_target_name_is_preserved(self) -> None:
        """The target label name passes through to ``Jump.target`` unchanged."""
        instrs = _stmt_instrs(ast.Goto(target="my_label"))
        jump = instrs[0]
        assert isinstance(jump, tac.Jump)
        assert jump.target == "my_label"

    def test_goto_does_not_produce_a_tac_label(self) -> None:
        """A ``Goto`` statement emits no ``tac.Label`` instruction."""
        instrs = _stmt_instrs(ast.Goto(target="end"))
        assert not any(isinstance(i, tac.Label) for i in instrs)


# ---------------------------------------------------------------------------
# Labeled statement
# ---------------------------------------------------------------------------


class TestTacGeneratorLabeledStatement:
    """``ast.Label`` emits a ``tac.Label`` instruction followed by the inner statement."""

    def test_label_wrapping_null_produces_only_label(self) -> None:
        """``foo:;`` → ``Label("foo")`` then the function guard."""
        instrs = _stmt_instrs(ast.Label(name="foo", statement=ast.Null()))
        assert instrs == (
            tac.Label(name="foo"),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_label_wrapping_return_full_sequence(self) -> None:
        """``foo: return 1;`` → ``Label("foo")``, ``Return(1)``, guard."""
        instrs = _stmt_instrs(ast.Label(name="foo", statement=ast.Return(exp=ast.Constant(value=1))))
        assert instrs == (
            tac.Label(name="foo"),
            tac.Return(val=tac.Constant(value=1)),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_label_name_is_preserved(self) -> None:
        """The source label name appears verbatim in the ``tac.Label`` instruction."""
        instrs = _stmt_instrs(ast.Label(name="my_label", statement=ast.Null()))
        label_instr = instrs[0]
        assert isinstance(label_instr, tac.Label)
        assert label_instr.name == "my_label"

    def test_nested_labels_produce_two_consecutive_label_instructions(self) -> None:
        """``foo: bar: return 0;`` → ``Label("foo")``, ``Label("bar")``, ``Return(0)``, guard."""
        instrs = _stmt_instrs(
            ast.Label(name="foo", statement=ast.Label(name="bar", statement=ast.Return(exp=ast.Constant(value=0))))
        )
        assert instrs == (
            tac.Label(name="foo"),
            tac.Label(name="bar"),
            tac.Return(val=tac.Constant(value=0)),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_goto_and_label_together_in_function(self) -> None:
        """``goto end; end: return 0;`` produces ``Jump``, ``Label``, ``Return``, guard."""
        body = (
            ast.S(statement=ast.Goto(target="end")),
            ast.S(statement=ast.Label(name="end", statement=ast.Return(exp=ast.Constant(value=0)))),
        )
        instrs = _instrs_for_body(body)
        assert instrs == (
            tac.Jump(target="end"),
            tac.Label(name="end"),
            tac.Return(val=tac.Constant(value=0)),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_label_wrapping_compound_block(self) -> None:
        """``foo: { return 0; }`` → ``Label("foo")``, ``Return(0)``, guard."""
        instrs = _stmt_instrs(
            ast.Label(
                name="foo",
                statement=ast.Compound(body=(ast.S(statement=ast.Return(exp=ast.Constant(value=0))),)),
            )
        )
        assert instrs == (
            tac.Label(name="foo"),
            tac.Return(val=tac.Constant(value=0)),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_user_label_name_does_not_clash_with_generated_labels(self) -> None:
        """A user label alongside an ``if`` statement has a distinct name from generated labels."""
        body = (
            ast.S(statement=ast.Label(name="user_lbl", statement=ast.Null())),
            ast.S(
                statement=ast.If(
                    condition=ast.Constant(value=1),
                    then=ast.Return(exp=ast.Constant(value=0)),
                    else_=None,
                )
            ),
        )
        instrs = _instrs_for_body(body)
        label_names = [i.name for i in instrs if isinstance(i, tac.Label)]
        assert "user_lbl" in label_names
        assert len(label_names) == len(set(label_names))


# ---------------------------------------------------------------------------
# Compound statement
# ---------------------------------------------------------------------------


class TestTacGeneratorCompoundStatement:
    """``ast.Compound`` flattens its block items into the surrounding instruction stream."""

    def test_empty_compound_produces_no_instructions(self) -> None:
        """``{}`` contributes no instructions; only the function guard remains."""
        instrs = _stmt_instrs(ast.Compound(body=()))
        assert instrs == (tac.Return(val=tac.Constant(value=0)),)

    def test_compound_with_return_statement(self) -> None:
        """``{ return 42; }`` emits ``Return(42)`` followed by the guard."""
        instrs = _stmt_instrs(ast.Compound(body=(ast.S(statement=ast.Return(exp=ast.Constant(value=42))),)))
        assert instrs == (
            tac.Return(val=tac.Constant(value=42)),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_compound_uninitialised_declaration_produces_no_instructions(self) -> None:
        """``{ int x; }`` has no initialiser and therefore contributes no instructions."""
        instrs = _stmt_instrs(
            ast.Compound(
                body=(
                    ast.D(
                        declaration=ast.VarDecl(variable_declaration=ast.VariableDeclaration(name=".Vx.0", init=None))
                    ),
                )
            )
        )
        assert instrs == (tac.Return(val=tac.Constant(value=0)),)

    def test_compound_initialised_declaration_produces_copy(self) -> None:
        """``{ int x = 5; }`` emits ``Copy(Constant(5), Var(x))`` followed by the guard."""
        instrs = _stmt_instrs(
            ast.Compound(
                body=(
                    ast.D(
                        declaration=ast.VarDecl(
                            variable_declaration=ast.VariableDeclaration(name=".Vx.0", init=ast.Constant(value=5))
                        )
                    ),
                )
            )
        )
        assert instrs == (
            tac.Copy(src=tac.Constant(value=5), dst=tac.Var(value=".Vx.0")),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_compound_items_emitted_in_order(self) -> None:
        """Items inside a compound block appear in declaration-then-use order."""
        instrs = _stmt_instrs(
            ast.Compound(
                body=(
                    ast.D(
                        declaration=ast.VarDecl(
                            variable_declaration=ast.VariableDeclaration(name=".Vx.0", init=ast.Constant(value=1))
                        )
                    ),
                    ast.S(statement=ast.Return(exp=ast.Var(name=".Vx.0"))),
                )
            )
        )
        assert instrs == (
            tac.Copy(src=tac.Constant(value=1), dst=tac.Var(value=".Vx.0")),
            tac.Return(val=tac.Var(value=".Vx.0")),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_compound_flattens_with_surrounding_instructions(self) -> None:
        """Instructions before and after a compound block appear in a single flat sequence with no scope markers."""
        body = (
            ast.D(
                declaration=ast.VarDecl(
                    variable_declaration=ast.VariableDeclaration(name=".Va.0", init=ast.Constant(value=1))
                )
            ),
            ast.S(
                statement=ast.Compound(
                    body=(
                        ast.D(
                            declaration=ast.VarDecl(
                                variable_declaration=ast.VariableDeclaration(name=".Vb.1", init=ast.Constant(value=2))
                            )
                        ),
                    )
                )
            ),
            ast.S(statement=ast.Return(exp=ast.Var(name=".Va.0"))),
        )
        instrs = _instrs_for_body(body)
        assert instrs == (
            tac.Copy(src=tac.Constant(value=1), dst=tac.Var(value=".Va.0")),
            tac.Copy(src=tac.Constant(value=2), dst=tac.Var(value=".Vb.1")),
            tac.Return(val=tac.Var(value=".Va.0")),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_nested_compound_blocks_flatten_completely(self) -> None:
        """``{{ return 1; }}`` produces a single flat ``Return(1)`` with no scope markers."""
        instrs = _stmt_instrs(
            ast.Compound(
                body=(ast.S(statement=ast.Compound(body=(ast.S(statement=ast.Return(exp=ast.Constant(value=1))),))),)
            )
        )
        assert instrs == (
            tac.Return(val=tac.Constant(value=1)),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_compound_as_then_branch_of_if(self) -> None:
        """``if (1) { return 2; }`` translates to the standard conditional jump sequence."""
        instrs = _if_instrs(
            ast.If(
                condition=ast.Constant(value=1),
                then=ast.Compound(body=(ast.S(statement=ast.Return(exp=ast.Constant(value=2))),)),
                else_=None,
            )
        )
        assert instrs == (
            tac.JumpIfZero(condition=tac.Constant(value=1), target="if_end.0"),
            tac.Return(val=tac.Constant(value=2)),
            tac.Label(name="if_end.0"),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_compound_as_else_branch_of_if(self) -> None:
        """``if (1) return 2; else { return 3; }`` translates to the standard if-else sequence."""
        instrs = _if_instrs(
            ast.If(
                condition=ast.Constant(value=1),
                then=ast.Return(exp=ast.Constant(value=2)),
                else_=ast.Compound(body=(ast.S(statement=ast.Return(exp=ast.Constant(value=3))),)),
            )
        )
        assert instrs == (
            tac.JumpIfZero(condition=tac.Constant(value=1), target="if_else.0"),
            tac.Return(val=tac.Constant(value=2)),
            tac.Jump(target="if_end.1"),
            tac.Label(name="if_else.0"),
            tac.Return(val=tac.Constant(value=3)),
            tac.Label(name="if_end.1"),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_label_inside_compound_produces_tac_label(self) -> None:
        """A labeled statement inside a compound block emits a ``tac.Label`` at the correct position."""
        instrs = _stmt_instrs(
            ast.Compound(
                body=(ast.S(statement=ast.Label(name="inner", statement=ast.Return(exp=ast.Constant(value=0)))),)
            )
        )
        assert instrs == (
            tac.Label(name="inner"),
            tac.Return(val=tac.Constant(value=0)),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_unknown_block_item_type_inside_compound_raises(self) -> None:
        """An unrecognised ``BlockItem`` subclass inside a compound block raises ``CodeGenerationError``."""

        class _FakeItem(ast.BlockItem):
            pass

        with pytest.raises(CodeGenerationError):
            _stmt_instrs(ast.Compound(body=(_FakeItem(),)))


# ---------------------------------------------------------------------------
# Break and Continue statements
# ---------------------------------------------------------------------------


class TestTacGeneratorBreakContinue:
    """``Break`` and ``Continue`` translate to unconditional jumps using loop identifiers."""

    def test_break_produces_jump_with_break_prefix(self) -> None:
        """``break;`` (inside a loop with id ``.L.while.0``) → ``Jump("break_.L.while.0")``."""
        instrs = _stmt_instrs(ast.Break(identifier=".L.while.0"))
        assert instrs == (
            tac.Jump(target="break_.L.while.0"),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_continue_produces_jump_with_continue_prefix(self) -> None:
        """``continue;`` (inside a loop with id ``.L.while.0``) → ``Jump("continue_.L.while.0")``."""
        instrs = _stmt_instrs(ast.Continue(identifier=".L.while.0"))
        assert instrs == (
            tac.Jump(target="continue_.L.while.0"),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_break_embeds_identifier_in_target(self) -> None:
        """The full loop identifier string is embedded verbatim in the ``Jump`` target."""
        instrs = _stmt_instrs(ast.Break(identifier=".L.for.3"))
        assert isinstance(instrs[0], tac.Jump)
        assert instrs[0].target == "break_.L.for.3"

    def test_continue_embeds_identifier_in_target(self) -> None:
        """The full loop identifier string is embedded verbatim in the ``Jump`` target."""
        instrs = _stmt_instrs(ast.Continue(identifier=".L.do.7"))
        assert isinstance(instrs[0], tac.Jump)
        assert instrs[0].target == "continue_.L.do.7"


# ---------------------------------------------------------------------------
# While loop
# ---------------------------------------------------------------------------


class TestTacGeneratorWhileLoop:
    """``While`` translates to a condition-at-top loop with continue and break labels."""

    def test_while_full_sequence(self) -> None:
        """``while (1) {}`` → continue-label, JumpIfZero, back-jump, break-label, guard."""
        instrs = _stmt_instrs(ast.While(condition=ast.Constant(value=1), body=ast.Null(), identifier=".L.while.0"))
        assert instrs == (
            tac.Label(name="continue_.L.while.0"),
            tac.JumpIfZero(condition=tac.Constant(value=1), target="break_.L.while.0"),
            tac.Jump(target="continue_.L.while.0"),
            tac.Label(name="break_.L.while.0"),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_while_body_instructions_between_condition_and_back_jump(self) -> None:
        """Body instructions appear between the condition check and the back-jump."""
        instrs = _stmt_instrs(
            ast.While(
                condition=ast.Constant(value=1),
                body=ast.Return(exp=ast.Constant(value=42)),
                identifier=".L.while.0",
            )
        )
        assert instrs == (
            tac.Label(name="continue_.L.while.0"),
            tac.JumpIfZero(condition=tac.Constant(value=1), target="break_.L.while.0"),
            tac.Return(val=tac.Constant(value=42)),
            tac.Jump(target="continue_.L.while.0"),
            tac.Label(name="break_.L.while.0"),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_while_non_constant_condition_evaluated_before_jump(self) -> None:
        """A non-trivial condition is fully evaluated before the conditional jump."""
        instrs = _stmt_instrs(
            ast.While(
                condition=ast.Unary(operator=ast.Not(), exp=ast.Var(name="x")),
                body=ast.Null(),
                identifier=".L.while.0",
            )
        )
        assert isinstance(instrs[0], tac.Label)  # continue label
        assert isinstance(instrs[1], tac.Unary)  # condition evaluation
        assert isinstance(instrs[2], tac.JumpIfZero)
        assert instrs[2].condition == instrs[1].dst

    def test_while_back_jump_targets_continue_label(self) -> None:
        """The back-jump at the end of the body targets the continue label, not the break label."""
        instrs = _stmt_instrs(ast.While(condition=ast.Constant(value=1), body=ast.Null(), identifier=".L.while.0"))
        back_jump = next(i for i in instrs if isinstance(i, tac.Jump))
        assert back_jump.target == "continue_.L.while.0"

    def test_while_condition_jump_targets_break_label(self) -> None:
        """The exit condition check jumps to the break label on a false condition."""
        instrs = _stmt_instrs(ast.While(condition=ast.Constant(value=1), body=ast.Null(), identifier=".L.while.0"))
        cond_jump = next(i for i in instrs if isinstance(i, tac.JumpIfZero))
        assert cond_jump.target == "break_.L.while.0"

    def test_break_inside_while_jumps_to_break_label(self) -> None:
        """A ``Break`` inside the loop body produces a ``Jump`` targeting the loop's break label."""
        instrs = _stmt_instrs(
            ast.While(
                condition=ast.Constant(value=1),
                body=ast.Break(identifier=".L.while.0"),
                identifier=".L.while.0",
            )
        )
        jumps = [i for i in instrs if isinstance(i, tac.Jump)]
        assert any(j.target == "break_.L.while.0" for j in jumps)

    def test_continue_inside_while_jumps_to_continue_label(self) -> None:
        """A ``Continue`` inside the loop body produces a ``Jump`` targeting the loop's continue label."""
        instrs = _stmt_instrs(
            ast.While(
                condition=ast.Constant(value=1),
                body=ast.Continue(identifier=".L.while.0"),
                identifier=".L.while.0",
            )
        )
        jumps = [i for i in instrs if isinstance(i, tac.Jump)]
        assert any(j.target == "continue_.L.while.0" for j in jumps)

    def test_nested_while_loops_have_distinct_labels(self) -> None:
        """Nested while loops with distinct identifiers produce non-overlapping label sets."""
        instrs = _stmt_instrs(
            ast.While(
                condition=ast.Constant(value=1),
                body=ast.While(condition=ast.Constant(value=1), body=ast.Null(), identifier=".L.while.1"),
                identifier=".L.while.0",
            )
        )
        label_names = [i.name for i in instrs if isinstance(i, tac.Label)]
        assert len(label_names) == len(set(label_names))
        assert len(label_names) == 4  # continue + break for each of the two loops


# ---------------------------------------------------------------------------
# DoWhile loop
# ---------------------------------------------------------------------------


class TestTacGeneratorDoWhileLoop:
    """``DoWhile`` translates to a condition-at-bottom loop where the body always executes first."""

    def test_do_while_full_sequence(self) -> None:
        """``do {} while (1);`` → start-label, continue-label, JumpIfNotZero, break-label, guard."""
        instrs = _stmt_instrs(ast.DoWhile(body=ast.Null(), condition=ast.Constant(value=1), identifier=".L.do.0"))
        assert instrs == (
            tac.Label(name="start_.L.do.0"),
            tac.Label(name="continue_.L.do.0"),
            tac.JumpIfNotZero(condition=tac.Constant(value=1), target="start_.L.do.0"),
            tac.Label(name="break_.L.do.0"),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_do_while_body_precedes_condition(self) -> None:
        """Body instructions appear before the condition check."""
        instrs = _stmt_instrs(
            ast.DoWhile(
                body=ast.Expression(exp=ast.Assignment(lvalue=ast.Var(name="x"), rvalue=ast.Constant(value=1))),
                condition=ast.Constant(value=1),
                identifier=".L.do.0",
            )
        )
        copy_idx = next(i for i, x in enumerate(instrs) if isinstance(x, tac.Copy))
        cond_jump_idx = next(i for i, x in enumerate(instrs) if isinstance(x, tac.JumpIfNotZero))
        assert copy_idx < cond_jump_idx

    def test_do_while_uses_jump_if_not_zero(self) -> None:
        """The back-edge fires on a non-zero condition (loop continues); ``JumpIfZero`` is not emitted."""
        instrs = _stmt_instrs(ast.DoWhile(body=ast.Null(), condition=ast.Constant(value=1), identifier=".L.do.0"))
        assert any(isinstance(i, tac.JumpIfNotZero) for i in instrs)
        assert not any(isinstance(i, tac.JumpIfZero) for i in instrs)

    def test_do_while_back_jump_targets_start_label(self) -> None:
        """The conditional back-jump at the bottom of the loop targets the start label."""
        instrs = _stmt_instrs(ast.DoWhile(body=ast.Null(), condition=ast.Constant(value=1), identifier=".L.do.0"))
        jump = next(i for i in instrs if isinstance(i, tac.JumpIfNotZero))
        assert jump.target == "start_.L.do.0"

    def test_break_inside_do_while_jumps_to_break_label(self) -> None:
        """A ``Break`` inside a do-while body produces a ``Jump`` to the break label."""
        instrs = _stmt_instrs(
            ast.DoWhile(
                body=ast.Break(identifier=".L.do.0"),
                condition=ast.Constant(value=1),
                identifier=".L.do.0",
            )
        )
        jumps = [i for i in instrs if isinstance(i, tac.Jump)]
        assert any(j.target == "break_.L.do.0" for j in jumps)


# ---------------------------------------------------------------------------
# For loop
# ---------------------------------------------------------------------------


class TestTacGeneratorForLoop:
    """``For`` translates to a standard init/test/step sequence with optional-expression elision."""

    def test_for_unconditional_full_sequence(self) -> None:
        """``for (;;) {}`` → start-label, continue-label, back-jump, break-label; no ``JumpIfZero``."""
        instrs = _stmt_instrs(
            ast.For(init=ast.InitExp(exp=None), condition=None, post=None, body=ast.Null(), identifier=".L.for.0")
        )
        assert instrs == (
            tac.Label(name="start_.L.for.0"),
            tac.Label(name="continue_.L.for.0"),
            tac.Jump(target="start_.L.for.0"),
            tac.Label(name="break_.L.for.0"),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_for_unconditional_emits_no_jump_if_zero(self) -> None:
        """Absent condition emits no ``JumpIfZero`` guard at all."""
        instrs = _stmt_instrs(
            ast.For(init=ast.InitExp(exp=None), condition=None, post=None, body=ast.Null(), identifier=".L.for.0")
        )
        assert not any(isinstance(i, tac.JumpIfZero) for i in instrs)

    def test_for_with_condition_full_sequence(self) -> None:
        """``for (;1;) {}`` → start-label, JumpIfZero, body, continue-label, back-jump, break-label."""
        instrs = _stmt_instrs(
            ast.For(
                init=ast.InitExp(exp=None),
                condition=ast.Constant(value=1),
                post=None,
                body=ast.Null(),
                identifier=".L.for.0",
            )
        )
        assert instrs == (
            tac.Label(name="start_.L.for.0"),
            tac.JumpIfZero(condition=tac.Constant(value=1), target="break_.L.for.0"),
            tac.Label(name="continue_.L.for.0"),
            tac.Jump(target="start_.L.for.0"),
            tac.Label(name="break_.L.for.0"),
            tac.Return(val=tac.Constant(value=0)),
        )

    def test_for_init_expression_emitted_before_start_label(self) -> None:
        """An init expression is emitted before the loop's start label."""
        instrs = _stmt_instrs(
            ast.For(
                init=ast.InitExp(exp=ast.Assignment(lvalue=ast.Var(name="i"), rvalue=ast.Constant(value=0))),
                condition=None,
                post=None,
                body=ast.Null(),
                identifier=".L.for.0",
            )
        )
        copy_idx = next(i for i, x in enumerate(instrs) if isinstance(x, tac.Copy))
        start_idx = next(i for i, x in enumerate(instrs) if isinstance(x, tac.Label) and "start_" in x.name)
        assert copy_idx < start_idx

    def test_for_absent_init_expression_emits_nothing_before_start_label(self) -> None:
        """An absent init clause (``InitExp(None)``) produces no instructions before the start label."""
        instrs = _stmt_instrs(
            ast.For(init=ast.InitExp(exp=None), condition=None, post=None, body=ast.Null(), identifier=".L.for.0")
        )
        assert instrs[0] == tac.Label(name="start_.L.for.0")

    def test_for_init_declaration_with_init_emits_copy_before_start_label(self) -> None:
        """``for (int i = 0;;)`` emits ``Copy(0, i)`` as the very first instruction."""
        instrs = _stmt_instrs(
            ast.For(
                init=ast.InitDecl(
                    declaration=ast.VarDecl(
                        variable_declaration=ast.VariableDeclaration(name=".Vi.0", init=ast.Constant(value=0))
                    )
                ),
                condition=None,
                post=None,
                body=ast.Null(),
                identifier=".L.for.0",
            )
        )
        assert instrs[0] == tac.Copy(src=tac.Constant(value=0), dst=tac.Var(value=".Vi.0"))
        assert instrs[1] == tac.Label(name="start_.L.for.0")

    def test_for_init_declaration_without_init_produces_no_copy(self) -> None:
        """``for (int i;;)`` — a declaration without an initialiser emits no ``Copy``."""
        instrs = _stmt_instrs(
            ast.For(
                init=ast.InitDecl(
                    declaration=ast.VarDecl(variable_declaration=ast.VariableDeclaration(name=".Vi.0", init=None))
                ),
                condition=None,
                post=None,
                body=ast.Null(),
                identifier=".L.for.0",
            )
        )
        assert instrs[0] == tac.Label(name="start_.L.for.0")

    def test_for_post_expression_emitted_after_continue_label(self) -> None:
        """The post expression is emitted after the continue label, not before it."""
        instrs = _stmt_instrs(
            ast.For(
                init=ast.InitExp(exp=None),
                condition=None,
                post=ast.Assignment(lvalue=ast.Var(name="i"), rvalue=ast.Constant(value=1)),
                body=ast.Null(),
                identifier=".L.for.0",
            )
        )
        continue_idx = next(i for i, x in enumerate(instrs) if isinstance(x, tac.Label) and "continue_" in x.name)
        copy_idx = next(i for i, x in enumerate(instrs) if isinstance(x, tac.Copy))
        assert continue_idx < copy_idx

    def test_for_all_three_labels_are_present_and_correctly_named(self) -> None:
        """``start_``, ``continue_``, and ``break_`` labels all carry the loop identifier."""
        instrs = _stmt_instrs(
            ast.For(
                init=ast.InitExp(exp=None),
                condition=ast.Constant(value=1),
                post=None,
                body=ast.Null(),
                identifier=".L.for.0",
            )
        )
        label_names = {i.name for i in instrs if isinstance(i, tac.Label)}
        assert "start_.L.for.0" in label_names
        assert "continue_.L.for.0" in label_names
        assert "break_.L.for.0" in label_names

    def test_for_break_jumps_to_break_label(self) -> None:
        """A ``Break`` inside a for-loop body produces a ``Jump`` to the break label."""
        instrs = _stmt_instrs(
            ast.For(
                init=ast.InitExp(exp=None),
                condition=None,
                post=None,
                body=ast.Break(identifier=".L.for.0"),
                identifier=".L.for.0",
            )
        )
        jumps = [i for i in instrs if isinstance(i, tac.Jump)]
        assert any(j.target == "break_.L.for.0" for j in jumps)

    def test_for_continue_jumps_to_continue_label(self) -> None:
        """A ``Continue`` inside a for-loop body produces a ``Jump`` to the continue label."""
        instrs = _stmt_instrs(
            ast.For(
                init=ast.InitExp(exp=None),
                condition=None,
                post=None,
                body=ast.Continue(identifier=".L.for.0"),
                identifier=".L.for.0",
            )
        )
        jumps = [i for i in instrs if isinstance(i, tac.Jump)]
        assert any(j.target == "continue_.L.for.0" for j in jumps)


# ---------------------------------------------------------------------------
# Switch statement
# ---------------------------------------------------------------------------

_SWITCH_ID = ".L.switch.0"


def _switch_stmt(
    control: ast.Exp,
    cases: tuple[ast.Statement, ...],
    identifier: str = _SWITCH_ID,
) -> ast.Switch:
    """Build a ``Switch`` with a ``Compound`` body from a sequence of case/default statements.

    Args:
        control: The controlling expression.
        cases: Statement nodes forming the switch body (typically ``Case``/``Default``).
        identifier: The switch identifier to attach.

    Returns:
        A fully-constructed ``ast.Switch`` node ready for TAC translation.
    """
    return ast.Switch(
        control=control,
        body=ast.Compound(body=tuple(ast.S(statement=s) for s in cases)),
        identifier=identifier,
    )


class TestTacGeneratorSwitch:
    """``Switch`` emits a comparison-based dispatch table followed by the body and a break label."""

    def test_switch_no_cases_jumps_straight_to_break(self) -> None:
        """An empty switch body emits no comparisons and jumps directly to the break label."""
        instrs = _stmt_instrs(_switch_stmt(ast.Constant(value=1), ()))
        label_names = [i.name for i in instrs if isinstance(i, tac.Label)]
        assert f"break_{_SWITCH_ID}" in label_names
        # Only jump should be the unconditional one to break
        jumps = [i for i in instrs if isinstance(i, tac.Jump)]
        assert jumps == [tac.Jump(target=f"break_{_SWITCH_ID}")]

    def test_switch_one_case_emits_comparison_and_conditional_jump(self) -> None:
        """A single ``case 1:`` emits ``Binary(Equal)`` + ``JumpIfNotZero`` before the body."""
        instrs = _stmt_instrs(
            _switch_stmt(
                ast.Var(name="x"),
                (ast.Case(value=ast.Constant(value=1), statement=ast.Null(), identifier=_SWITCH_ID),),
            )
        )
        binaries = [i for i in instrs if isinstance(i, tac.Binary)]
        assert len(binaries) == 1
        assert isinstance(binaries[0].operator, tac.Equal)
        assert binaries[0].src1 == tac.Var(value="x")
        assert binaries[0].src2 == tac.Constant(value=1)
        cond_jumps = [i for i in instrs if isinstance(i, tac.JumpIfNotZero)]
        assert len(cond_jumps) == 1
        assert cond_jumps[0].target == f"case_{_SWITCH_ID}_1"

    def test_switch_dispatch_table_precedes_body(self) -> None:
        """All comparison+jump pairs appear before the first case label in the instruction stream."""
        instrs = _stmt_instrs(
            _switch_stmt(
                ast.Var(name="x"),
                (
                    ast.Case(value=ast.Constant(value=1), statement=ast.Null(), identifier=_SWITCH_ID),
                    ast.Case(value=ast.Constant(value=2), statement=ast.Null(), identifier=_SWITCH_ID),
                ),
            )
        )
        first_case_label_idx = next(
            i for i, instr in enumerate(instrs) if isinstance(instr, tac.Label) and "case_" in instr.name
        )
        cond_jump_indices = [i for i, instr in enumerate(instrs) if isinstance(instr, tac.JumpIfNotZero)]
        assert all(idx < first_case_label_idx for idx in cond_jump_indices)

    def test_switch_without_default_fallback_jumps_to_break(self) -> None:
        """When no ``default`` is present, the dispatch table ends with ``Jump(break_<id>)``."""
        instrs = _stmt_instrs(
            _switch_stmt(
                ast.Var(name="x"),
                (ast.Case(value=ast.Constant(value=1), statement=ast.Null(), identifier=_SWITCH_ID),),
            )
        )
        unconditional_jumps = [i for i in instrs if isinstance(i, tac.Jump)]
        assert any(j.target == f"break_{_SWITCH_ID}" for j in unconditional_jumps)

    def test_switch_with_default_fallback_jumps_to_default(self) -> None:
        """When a ``default`` is present, the dispatch table ends with ``Jump(default_<id>)``."""
        instrs = _stmt_instrs(
            _switch_stmt(
                ast.Var(name="x"),
                (
                    ast.Case(value=ast.Constant(value=1), statement=ast.Null(), identifier=_SWITCH_ID),
                    ast.Default(statement=ast.Null(), identifier=_SWITCH_ID),
                ),
            )
        )
        unconditional_jumps = [i for i in instrs if isinstance(i, tac.Jump)]
        assert any(j.target == f"default_{_SWITCH_ID}" for j in unconditional_jumps)

    def test_switch_break_label_is_last_label_in_body(self) -> None:
        """The ``break_<id>`` label is emitted at the end of the switch, before the function guard."""
        instrs = _stmt_instrs(
            _switch_stmt(
                ast.Var(name="x"),
                (ast.Case(value=ast.Constant(value=1), statement=ast.Null(), identifier=_SWITCH_ID),),
            )
        )
        labels = [i for i in instrs if isinstance(i, tac.Label)]
        assert labels[-1].name == f"break_{_SWITCH_ID}"

    def test_switch_case_label_name_encodes_value(self) -> None:
        """The case label for ``case 42:`` is ``case_<id>_42``."""
        instrs = _stmt_instrs(
            _switch_stmt(
                ast.Var(name="x"),
                (ast.Case(value=ast.Constant(value=42), statement=ast.Null(), identifier=_SWITCH_ID),),
            )
        )
        label_names = [i.name for i in instrs if isinstance(i, tac.Label)]
        assert f"case_{_SWITCH_ID}_42" in label_names

    def test_switch_default_label_name(self) -> None:
        """The default label is ``default_<id>``."""
        instrs = _stmt_instrs(
            _switch_stmt(
                ast.Var(name="x"),
                (ast.Default(statement=ast.Null(), identifier=_SWITCH_ID),),
            )
        )
        label_names = [i.name for i in instrs if isinstance(i, tac.Label)]
        assert f"default_{_SWITCH_ID}" in label_names

    def test_switch_multiple_cases_emit_comparisons_in_order(self) -> None:
        """Two cases emit two comparison blocks; the first case's jump appears before the second's."""
        instrs = _stmt_instrs(
            _switch_stmt(
                ast.Var(name="x"),
                (
                    ast.Case(value=ast.Constant(value=10), statement=ast.Null(), identifier=_SWITCH_ID),
                    ast.Case(value=ast.Constant(value=20), statement=ast.Null(), identifier=_SWITCH_ID),
                ),
            )
        )
        cond_jumps = [i for i in instrs if isinstance(i, tac.JumpIfNotZero)]
        assert cond_jumps[0].target == f"case_{_SWITCH_ID}_10"
        assert cond_jumps[1].target == f"case_{_SWITCH_ID}_20"

    def test_switch_break_inside_body_jumps_to_break_label(self) -> None:
        """A ``Break`` inside a case body produces a ``Jump`` to ``break_<id>``."""
        instrs = _stmt_instrs(
            _switch_stmt(
                ast.Var(name="x"),
                (
                    ast.Case(
                        value=ast.Constant(value=1),
                        statement=ast.Break(identifier=_SWITCH_ID),
                        identifier=_SWITCH_ID,
                    ),
                ),
            )
        )
        jumps = [i for i in instrs if isinstance(i, tac.Jump)]
        assert any(j.target == f"break_{_SWITCH_ID}" for j in jumps)

    def test_switch_full_sequence_with_two_cases_and_default(self) -> None:
        """Full instruction sequence for ``switch(x) { case 1: return 1; case 2: return 2; default: return 0; }``."""
        instrs = _stmt_instrs(
            _switch_stmt(
                ast.Var(name="x"),
                (
                    ast.Case(
                        value=ast.Constant(value=1),
                        statement=ast.Return(exp=ast.Constant(value=1)),
                        identifier=_SWITCH_ID,
                    ),
                    ast.Case(
                        value=ast.Constant(value=2),
                        statement=ast.Return(exp=ast.Constant(value=2)),
                        identifier=_SWITCH_ID,
                    ),
                    ast.Default(
                        statement=ast.Return(exp=ast.Constant(value=0)),
                        identifier=_SWITCH_ID,
                    ),
                ),
            )
        )
        # Dispatch: cmp x==1, jump; cmp x==2, jump; jump default
        assert isinstance(instrs[0], tac.Binary) and instrs[0].src2 == tac.Constant(value=1)
        assert isinstance(instrs[1], tac.JumpIfNotZero) and instrs[1].target == f"case_{_SWITCH_ID}_1"
        assert isinstance(instrs[2], tac.Binary) and instrs[2].src2 == tac.Constant(value=2)
        assert isinstance(instrs[3], tac.JumpIfNotZero) and instrs[3].target == f"case_{_SWITCH_ID}_2"
        assert isinstance(instrs[4], tac.Jump) and instrs[4].target == f"default_{_SWITCH_ID}"
        # Body: case 1 label, return 1, case 2 label, return 2, default label, return 0
        assert isinstance(instrs[5], tac.Label) and instrs[5].name == f"case_{_SWITCH_ID}_1"
        assert isinstance(instrs[6], tac.Return) and instrs[6].val == tac.Constant(value=1)
        assert isinstance(instrs[7], tac.Label) and instrs[7].name == f"case_{_SWITCH_ID}_2"
        assert isinstance(instrs[8], tac.Return) and instrs[8].val == tac.Constant(value=2)
        assert isinstance(instrs[9], tac.Label) and instrs[9].name == f"default_{_SWITCH_ID}"
        assert isinstance(instrs[10], tac.Return) and instrs[10].val == tac.Constant(value=0)
        assert isinstance(instrs[11], tac.Label) and instrs[11].name == f"break_{_SWITCH_ID}"


class TestTacGeneratorCaseDefault:
    """``Case`` and ``Default`` in isolation produce a label followed by the governed statement."""

    def test_case_emits_label_then_statement(self) -> None:
        """``case 5: return 5;`` → ``Label("case_<id>_5")`` + ``Return(5)``."""
        instrs = _stmt_instrs(
            ast.Case(
                value=ast.Constant(value=5),
                statement=ast.Return(exp=ast.Constant(value=5)),
                identifier=_SWITCH_ID,
            )
        )
        assert isinstance(instrs[0], tac.Label)
        assert instrs[0].name == f"case_{_SWITCH_ID}_5"
        assert isinstance(instrs[1], tac.Return)
        assert instrs[1].val == tac.Constant(value=5)

    def test_default_emits_label_then_statement(self) -> None:
        """``default: return 0;`` → ``Label("default_<id>")`` + ``Return(0)``."""
        instrs = _stmt_instrs(
            ast.Default(
                statement=ast.Return(exp=ast.Constant(value=0)),
                identifier=_SWITCH_ID,
            )
        )
        assert isinstance(instrs[0], tac.Label)
        assert instrs[0].name == f"default_{_SWITCH_ID}"
        assert isinstance(instrs[1], tac.Return)
        assert instrs[1].val == tac.Constant(value=0)


# ---------------------------------------------------------------------------
# Multiple functions and forward declarations
# ---------------------------------------------------------------------------


def _make_function(name: str, body: ast.Block, params: tuple[ast.Parameter, ...] = ()) -> ast.Function:
    """Build a minimal ``ast.Function`` definition.

    Args:
        name: The function name.
        body: The block of statements forming the function body.
        params: Optional parameter list; defaults to empty.

    Returns:
        An ``ast.Function`` node with the given name, params, and body.
    """
    return ast.Function(name=name, params=params, body=body)


class TestTacGeneratorMultipleFunctions:
    """``generate`` correctly handles programs with more than one function definition."""

    def test_two_functions_produce_two_tac_functions(self) -> None:
        """A program with two definitions yields two entries in ``tac.Program.functions``."""
        prog = ast.Program(
            declarations=(
                ast.FunDecl(_make_function("foo", (ast.S(statement=ast.Return(exp=ast.Constant(value=1))),))),
                ast.FunDecl(_make_function("bar", (ast.S(statement=ast.Return(exp=ast.Constant(value=2))),))),
            )
        )
        result = generate(prog)
        assert len(result.functions) == 2

    def test_function_names_are_preserved_in_order(self) -> None:
        """The TAC functions appear in the same order as the source definitions."""
        prog = ast.Program(
            declarations=(
                ast.FunDecl(_make_function("alpha", (ast.S(statement=ast.Return(exp=ast.Constant(value=0))),))),
                ast.FunDecl(_make_function("beta", (ast.S(statement=ast.Return(exp=ast.Constant(value=0))),))),
            )
        )
        result = generate(prog)
        assert result.functions[0].name == "alpha"
        assert result.functions[1].name == "beta"

    def test_forward_declaration_is_skipped(self) -> None:
        """A function with ``body=None`` (forward declaration) is excluded from the TAC output."""
        prog = ast.Program(
            declarations=(
                ast.FunDecl(ast.Function(name="declared_only", params=(), body=None)),
                ast.FunDecl(_make_function("defined", (ast.S(statement=ast.Return(exp=ast.Constant(value=0))),))),
            )
        )
        result = generate(prog)
        assert len(result.functions) == 1
        assert result.functions[0].name == "defined"

    def test_all_forward_declarations_produces_empty_function_list(self) -> None:
        """A program consisting solely of forward declarations yields an empty ``functions`` tuple."""
        prog = ast.Program(
            declarations=(
                ast.FunDecl(ast.Function(name="f", params=(), body=None)),
                ast.FunDecl(ast.Function(name="g", params=(), body=None)),
            )
        )
        result = generate(prog)
        assert result.functions == ()

    def test_temporaries_are_unique_across_functions(self) -> None:
        """Temporaries allocated in different functions have distinct names."""
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    _make_function(
                        "f",
                        (ast.S(statement=ast.Return(exp=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=1)))),),
                    )
                ),
                ast.FunDecl(
                    _make_function(
                        "g",
                        (ast.S(statement=ast.Return(exp=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=2)))),),
                    )
                ),
            )
        )
        result = generate(prog)
        f_temps = {i.dst for i in result.functions[0].body if isinstance(i, tac.Unary)}
        g_temps = {i.dst for i in result.functions[1].body if isinstance(i, tac.Unary)}
        assert len(f_temps) == 1
        assert len(g_temps) == 1
        assert f_temps.isdisjoint(g_temps)


# ---------------------------------------------------------------------------
# Function parameters
# ---------------------------------------------------------------------------


class TestTacGeneratorFunctionParameters:
    """Parameter names are extracted from ``ast.Parameter`` nodes and stored in ``tac.Function.params``."""

    def test_single_parameter_name_preserved(self) -> None:
        """A function with one parameter stores its name in ``tac.Function.params``."""
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    _make_function(
                        "f",
                        (ast.S(statement=ast.Return(exp=ast.Constant(value=0))),),
                        params=(ast.Parameter(param_type=ast.IntType(), name="x"),),
                    )
                ),
            )
        )
        result = generate(prog)
        assert result.functions[0].params == ("x",)

    def test_multiple_parameters_names_preserved_in_order(self) -> None:
        """Multiple parameters are stored in declaration order."""
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    _make_function(
                        "add",
                        (ast.S(statement=ast.Return(exp=ast.Constant(value=0))),),
                        params=(
                            ast.Parameter(param_type=ast.IntType(), name="a"),
                            ast.Parameter(param_type=ast.IntType(), name="b"),
                        ),
                    )
                ),
            )
        )
        result = generate(prog)
        assert result.functions[0].params == ("a", "b")

    def test_no_parameters_yields_empty_params(self) -> None:
        """A function with no parameters has an empty ``params`` tuple."""
        prog = ast.Program(
            declarations=(
                ast.FunDecl(_make_function("main", (ast.S(statement=ast.Return(exp=ast.Constant(value=0))),))),
            )
        )
        result = generate(prog)
        assert result.functions[0].params == ()


# ---------------------------------------------------------------------------
# FunDecl as block item
# ---------------------------------------------------------------------------


class TestTacGeneratorFunDeclBlockItem:
    """A ``FunDecl`` inside a function body is silently ignored (no instructions emitted)."""

    def test_fun_decl_block_item_produces_no_instructions(self) -> None:
        """A forward declaration inside a block contributes no TAC instructions."""
        forward = ast.FunDecl(function_declaration=ast.Function(name="helper", params=(), body=None))
        instrs = _instrs_for_body((ast.D(declaration=forward),))
        assert instrs == (tac.Return(val=tac.Constant(value=0)),)


# ---------------------------------------------------------------------------
# Function call expression
# ---------------------------------------------------------------------------


class TestTacGeneratorFunctionCall:
    """``ast.FunctionCall`` lowers to a ``tac.FunCall`` instruction with evaluated arguments."""

    def test_no_arg_call_produces_funcall(self) -> None:
        """``foo()`` emits a single ``FunCall`` before the ``Return``."""
        instrs = _body(ast.FunctionCall(name="foo", args=[]))
        funcalls = [i for i in instrs if isinstance(i, tac.FunCall)]
        assert len(funcalls) == 1
        assert funcalls[0].name == "foo"

    def test_no_arg_call_has_empty_args(self) -> None:
        """A zero-argument call stores an empty args tuple in ``FunCall``."""
        instrs = _body(ast.FunctionCall(name="foo", args=[]))
        funcall = next(i for i in instrs if isinstance(i, tac.FunCall))
        assert funcall.args == ()

    def test_call_result_stored_in_temp(self) -> None:
        """The return value of a call is stored in a temporary that is then returned."""
        instrs = _body(ast.FunctionCall(name="foo", args=[]))
        funcall = next(i for i in instrs if isinstance(i, tac.FunCall))
        ret = instrs[-2]  # penultimate; last is the function guard
        assert isinstance(ret, tac.Return)
        assert ret.val == funcall.dst

    def test_constant_arg_becomes_tac_constant(self) -> None:
        """A constant argument is passed directly as a ``tac.Constant``."""
        instrs = _body(ast.FunctionCall(name="foo", args=[ast.Constant(value=42)]))
        funcall = next(i for i in instrs if isinstance(i, tac.FunCall))
        assert funcall.args == (tac.Constant(value=42),)

    def test_multiple_constant_args_preserved_in_order(self) -> None:
        """Multiple constant arguments appear in the ``FunCall.args`` tuple in declaration order."""
        instrs = _body(ast.FunctionCall(name="add", args=[ast.Constant(value=1), ast.Constant(value=2)]))
        funcall = next(i for i in instrs if isinstance(i, tac.FunCall))
        assert funcall.args == (tac.Constant(value=1), tac.Constant(value=2))

    def test_expression_arg_emitted_before_funcall(self) -> None:
        """A non-trivial argument expression is fully evaluated before the ``FunCall`` is emitted."""
        instrs = _body(
            ast.FunctionCall(
                name="f",
                args=[ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=1))],
            )
        )
        unary_idx = next(i for i, x in enumerate(instrs) if isinstance(x, tac.Unary))
        funcall_idx = next(i for i, x in enumerate(instrs) if isinstance(x, tac.FunCall))
        assert unary_idx < funcall_idx

    def test_expression_arg_result_passed_to_funcall(self) -> None:
        """The temporary produced by evaluating an argument expression is passed as the arg value."""
        instrs = _body(
            ast.FunctionCall(
                name="f",
                args=[ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=1))],
            )
        )
        unary = next(i for i in instrs if isinstance(i, tac.Unary))
        funcall = next(i for i in instrs if isinstance(i, tac.FunCall))
        assert funcall.args == (unary.dst,)

    def test_multiple_expression_args_evaluated_left_to_right(self) -> None:
        """Arguments with side-effects are evaluated strictly left to right."""
        instrs = _body(
            ast.FunctionCall(
                name="f",
                args=[
                    ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=1)),
                    ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=2)),
                ],
            )
        )
        unaries = [i for i in instrs if isinstance(i, tac.Unary)]
        funcall = next(i for i in instrs if isinstance(i, tac.FunCall))
        assert len(unaries) == 2
        assert unaries[0].src == tac.Constant(value=1)
        assert unaries[1].src == tac.Constant(value=2)
        assert funcall.args == (unaries[0].dst, unaries[1].dst)
