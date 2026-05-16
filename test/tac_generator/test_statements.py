"""Tests for statement-level TAC generation."""

import pytest

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from compiler.utils.exceptions import CodeGenerationError
from test.tac_generator.helpers import _instrs_for_body

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
# Helpers for if tests
# ---------------------------------------------------------------------------


def _if_instrs(stmt: ast.If) -> tuple[tac.Instruction, ...]:
    """Translate a single ``If`` statement and return the resulting TAC instructions.

    Args:
        stmt: The ``If`` node to translate.

    Returns:
        The TAC instruction tuple from the generated function body.
    """
    return _instrs_for_body((ast.S(statement=stmt),))


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
# If statement
# ---------------------------------------------------------------------------


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
