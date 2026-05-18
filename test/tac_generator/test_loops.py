"""Tests for loop-level TAC generation."""

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from test.tac_generator.helpers import _instrs_for_body

# ---------------------------------------------------------------------------
# Helpers for statement tests
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
# Break and Continue statements
# ---------------------------------------------------------------------------


class TestTacGeneratorBreakContinue:
    """``Break`` and ``Continue`` translate to unconditional jumps using loop identifiers."""

    def test_break_produces_jump_with_break_prefix(self) -> None:
        """``break;`` (inside a loop with id ``.L.while.0``) → ``Jump("break_.L.while.0")``."""
        instrs = _stmt_instrs(ast.Break(identifier=".L.while.0"))
        assert instrs == (
            tac.Jump(target="break_.L.while.0"),
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
        )

    def test_continue_produces_jump_with_continue_prefix(self) -> None:
        """``continue;`` (inside a loop with id ``.L.while.0``) → ``Jump("continue_.L.while.0")``."""
        instrs = _stmt_instrs(ast.Continue(identifier=".L.while.0"))
        assert instrs == (
            tac.Jump(target="continue_.L.while.0"),
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
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
        instrs = _stmt_instrs(
            ast.While(condition=ast.Constant(value=ast.ConstInt(1)), body=ast.Null(), identifier=".L.while.0")
        )
        assert instrs == (
            tac.Label(name="continue_.L.while.0"),
            tac.JumpIfZero(condition=tac.Constant(value=ast.ConstInt(1)), target="break_.L.while.0"),
            tac.Jump(target="continue_.L.while.0"),
            tac.Label(name="break_.L.while.0"),
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
        )

    def test_while_body_instructions_between_condition_and_back_jump(self) -> None:
        """Body instructions appear between the condition check and the back-jump."""
        instrs = _stmt_instrs(
            ast.While(
                condition=ast.Constant(value=ast.ConstInt(1)),
                body=ast.Return(exp=ast.Constant(value=ast.ConstInt(42))),
                identifier=".L.while.0",
            )
        )
        assert instrs == (
            tac.Label(name="continue_.L.while.0"),
            tac.JumpIfZero(condition=tac.Constant(value=ast.ConstInt(1)), target="break_.L.while.0"),
            tac.Return(val=tac.Constant(value=ast.ConstInt(42))),
            tac.Jump(target="continue_.L.while.0"),
            tac.Label(name="break_.L.while.0"),
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
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
        instrs = _stmt_instrs(
            ast.While(condition=ast.Constant(value=ast.ConstInt(1)), body=ast.Null(), identifier=".L.while.0")
        )
        back_jump = next(i for i in instrs if isinstance(i, tac.Jump))
        assert back_jump.target == "continue_.L.while.0"

    def test_while_condition_jump_targets_break_label(self) -> None:
        """The exit condition check jumps to the break label on a false condition."""
        instrs = _stmt_instrs(
            ast.While(condition=ast.Constant(value=ast.ConstInt(1)), body=ast.Null(), identifier=".L.while.0")
        )
        cond_jump = next(i for i in instrs if isinstance(i, tac.JumpIfZero))
        assert cond_jump.target == "break_.L.while.0"

    def test_break_inside_while_jumps_to_break_label(self) -> None:
        """A ``Break`` inside the loop body produces a ``Jump`` targeting the loop's break label."""
        instrs = _stmt_instrs(
            ast.While(
                condition=ast.Constant(value=ast.ConstInt(1)),
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
                condition=ast.Constant(value=ast.ConstInt(1)),
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
                condition=ast.Constant(value=ast.ConstInt(1)),
                body=ast.While(condition=ast.Constant(value=ast.ConstInt(1)), body=ast.Null(), identifier=".L.while.1"),
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
        instrs = _stmt_instrs(
            ast.DoWhile(body=ast.Null(), condition=ast.Constant(value=ast.ConstInt(1)), identifier=".L.do.0")
        )
        assert instrs == (
            tac.Label(name="start_.L.do.0"),
            tac.Label(name="continue_.L.do.0"),
            tac.JumpIfNotZero(condition=tac.Constant(value=ast.ConstInt(1)), target="start_.L.do.0"),
            tac.Label(name="break_.L.do.0"),
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
        )

    def test_do_while_body_precedes_condition(self) -> None:
        """Body instructions appear before the condition check."""
        instrs = _stmt_instrs(
            ast.DoWhile(
                body=ast.Expression(
                    exp=ast.Assignment(lvalue=ast.Var(name="x"), rvalue=ast.Constant(value=ast.ConstInt(1)))
                ),
                condition=ast.Constant(value=ast.ConstInt(1)),
                identifier=".L.do.0",
            )
        )
        copy_idx = next(i for i, x in enumerate(instrs) if isinstance(x, tac.Copy))
        cond_jump_idx = next(i for i, x in enumerate(instrs) if isinstance(x, tac.JumpIfNotZero))
        assert copy_idx < cond_jump_idx

    def test_do_while_uses_jump_if_not_zero(self) -> None:
        """The back-edge fires on a non-zero condition (loop continues); ``JumpIfZero`` is not emitted."""
        instrs = _stmt_instrs(
            ast.DoWhile(body=ast.Null(), condition=ast.Constant(value=ast.ConstInt(1)), identifier=".L.do.0")
        )
        assert any(isinstance(i, tac.JumpIfNotZero) for i in instrs)
        assert not any(isinstance(i, tac.JumpIfZero) for i in instrs)

    def test_do_while_back_jump_targets_start_label(self) -> None:
        """The conditional back-jump at the bottom of the loop targets the start label."""
        instrs = _stmt_instrs(
            ast.DoWhile(body=ast.Null(), condition=ast.Constant(value=ast.ConstInt(1)), identifier=".L.do.0")
        )
        jump = next(i for i in instrs if isinstance(i, tac.JumpIfNotZero))
        assert jump.target == "start_.L.do.0"

    def test_break_inside_do_while_jumps_to_break_label(self) -> None:
        """A ``Break`` inside a do-while body produces a ``Jump`` to the break label."""
        instrs = _stmt_instrs(
            ast.DoWhile(
                body=ast.Break(identifier=".L.do.0"),
                condition=ast.Constant(value=ast.ConstInt(1)),
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
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
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
                condition=ast.Constant(value=ast.ConstInt(1)),
                post=None,
                body=ast.Null(),
                identifier=".L.for.0",
            )
        )
        assert instrs == (
            tac.Label(name="start_.L.for.0"),
            tac.JumpIfZero(condition=tac.Constant(value=ast.ConstInt(1)), target="break_.L.for.0"),
            tac.Label(name="continue_.L.for.0"),
            tac.Jump(target="start_.L.for.0"),
            tac.Label(name="break_.L.for.0"),
            tac.Return(val=tac.Constant(value=ast.ConstInt(0))),
        )

    def test_for_init_expression_emitted_before_start_label(self) -> None:
        """An init expression is emitted before the loop's start label."""
        instrs = _stmt_instrs(
            ast.For(
                init=ast.InitExp(
                    exp=ast.Assignment(lvalue=ast.Var(name="i"), rvalue=ast.Constant(value=ast.ConstInt(0)))
                ),
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
                        variable_declaration=ast.VariableDeclaration(
                            name=".Vi.0", init=ast.Constant(value=ast.ConstInt(0)), var_type=ast.IntType()
                        )
                    )
                ),
                condition=None,
                post=None,
                body=ast.Null(),
                identifier=".L.for.0",
            )
        )
        assert instrs[0] == tac.Copy(src=tac.Constant(value=ast.ConstInt(0)), dst=tac.Var(value=".Vi.0"))
        assert instrs[1] == tac.Label(name="start_.L.for.0")

    def test_for_init_declaration_without_init_produces_no_copy(self) -> None:
        """``for (int i;;)`` — a declaration without an initialiser emits no ``Copy``."""
        instrs = _stmt_instrs(
            ast.For(
                init=ast.InitDecl(
                    declaration=ast.VarDecl(
                        variable_declaration=ast.VariableDeclaration(name=".Vi.0", init=None, var_type=ast.IntType())
                    )
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
                post=ast.Assignment(lvalue=ast.Var(name="i"), rvalue=ast.Constant(value=ast.ConstInt(1))),
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
                condition=ast.Constant(value=ast.ConstInt(1)),
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
