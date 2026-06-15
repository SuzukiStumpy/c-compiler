"""Tests for switch statement TAC generation."""

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from test.tac_generator.helpers import _instrs_for_body

# ---------------------------------------------------------------------------
# Switch statement
# ---------------------------------------------------------------------------

_SWITCH_ID = ".L.switch.0"


def _stmt_instrs(stmt: ast.Statement) -> tuple[tac.Instruction, ...]:
    """Translate a single statement and return the resulting TAC instruction body.

    Args:
        stmt: The statement node to translate.

    Returns:
        The TAC instruction tuple from the generated function body.
    """
    return _instrs_for_body((ast.S(statement=stmt),))


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
        instrs = _stmt_instrs(_switch_stmt(ast.Constant(value=ast.ConstInt(1)), ()))
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
                (ast.Case(value=ast.Constant(value=ast.ConstInt(1)), statement=ast.Null(), identifier=_SWITCH_ID),),
            )
        )
        binaries = [i for i in instrs if isinstance(i, tac.Binary)]
        assert len(binaries) == 1
        assert isinstance(binaries[0].operator, tac.Equal)
        assert binaries[0].src1 == tac.Var(value="x")
        assert binaries[0].src2 == tac.Constant(value=ast.ConstInt(1))
        cond_jumps = [i for i in instrs if isinstance(i, tac.JumpIfNotZero)]
        assert len(cond_jumps) == 1
        assert cond_jumps[0].target == f"case_{_SWITCH_ID}_1"

    def test_switch_dispatch_table_precedes_body(self) -> None:
        """All comparison+jump pairs appear before the first case label in the instruction stream."""
        instrs = _stmt_instrs(
            _switch_stmt(
                ast.Var(name="x"),
                (
                    ast.Case(value=ast.Constant(value=ast.ConstInt(1)), statement=ast.Null(), identifier=_SWITCH_ID),
                    ast.Case(value=ast.Constant(value=ast.ConstInt(2)), statement=ast.Null(), identifier=_SWITCH_ID),
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
                (ast.Case(value=ast.Constant(value=ast.ConstInt(1)), statement=ast.Null(), identifier=_SWITCH_ID),),
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
                    ast.Case(value=ast.Constant(value=ast.ConstInt(1)), statement=ast.Null(), identifier=_SWITCH_ID),
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
                (ast.Case(value=ast.Constant(value=ast.ConstInt(1)), statement=ast.Null(), identifier=_SWITCH_ID),),
            )
        )
        labels = [i for i in instrs if isinstance(i, tac.Label)]
        assert labels[-1].name == f"break_{_SWITCH_ID}"

    def test_switch_case_label_name_encodes_value(self) -> None:
        """The case label for ``case 42:`` is ``case_<id>_42``."""
        instrs = _stmt_instrs(
            _switch_stmt(
                ast.Var(name="x"),
                (ast.Case(value=ast.Constant(value=ast.ConstInt(42)), statement=ast.Null(), identifier=_SWITCH_ID),),
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
                    ast.Case(value=ast.Constant(value=ast.ConstInt(10)), statement=ast.Null(), identifier=_SWITCH_ID),
                    ast.Case(value=ast.Constant(value=ast.ConstInt(20)), statement=ast.Null(), identifier=_SWITCH_ID),
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
                        value=ast.Constant(value=ast.ConstInt(1)),
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
                        value=ast.Constant(value=ast.ConstInt(1)),
                        statement=ast.Return(exp=ast.Constant(value=ast.ConstInt(1))),
                        identifier=_SWITCH_ID,
                    ),
                    ast.Case(
                        value=ast.Constant(value=ast.ConstInt(2)),
                        statement=ast.Return(exp=ast.Constant(value=ast.ConstInt(2))),
                        identifier=_SWITCH_ID,
                    ),
                    ast.Default(
                        statement=ast.Return(exp=ast.Constant(value=ast.ConstInt(0))),
                        identifier=_SWITCH_ID,
                    ),
                ),
            )
        )
        # Dispatch: cmp x==1, jump; cmp x==2, jump; jump default
        assert isinstance(instrs[0], tac.Binary) and instrs[0].src2 == tac.Constant(value=ast.ConstInt(1))
        assert isinstance(instrs[1], tac.JumpIfNotZero) and instrs[1].target == f"case_{_SWITCH_ID}_1"
        assert isinstance(instrs[2], tac.Binary) and instrs[2].src2 == tac.Constant(value=ast.ConstInt(2))
        assert isinstance(instrs[3], tac.JumpIfNotZero) and instrs[3].target == f"case_{_SWITCH_ID}_2"
        assert isinstance(instrs[4], tac.Jump) and instrs[4].target == f"default_{_SWITCH_ID}"
        # Body: case 1 label, return 1, case 2 label, return 2, default label, return 0
        assert isinstance(instrs[5], tac.Label) and instrs[5].name == f"case_{_SWITCH_ID}_1"
        assert isinstance(instrs[6], tac.Return) and instrs[6].val == tac.Constant(value=ast.ConstInt(1))
        assert isinstance(instrs[7], tac.Label) and instrs[7].name == f"case_{_SWITCH_ID}_2"
        assert isinstance(instrs[8], tac.Return) and instrs[8].val == tac.Constant(value=ast.ConstInt(2))
        assert isinstance(instrs[9], tac.Label) and instrs[9].name == f"default_{_SWITCH_ID}"
        assert isinstance(instrs[10], tac.Return) and instrs[10].val == tac.Constant(value=ast.ConstInt(0))
        assert isinstance(instrs[11], tac.Label) and instrs[11].name == f"break_{_SWITCH_ID}"


class TestTacGeneratorCaseDefault:
    """``Case`` and ``Default`` in isolation produce a label followed by the governed statement."""

    def test_case_emits_label_then_statement(self) -> None:
        """``case 5: return 5;`` -> ``Label("case_<id>_5")`` + ``Return(5)``."""
        instrs = _stmt_instrs(
            ast.Case(
                value=ast.Constant(value=ast.ConstInt(5)),
                statement=ast.Return(exp=ast.Constant(value=ast.ConstInt(5))),
                identifier=_SWITCH_ID,
            )
        )
        assert isinstance(instrs[0], tac.Label)
        assert instrs[0].name == f"case_{_SWITCH_ID}_5"
        assert isinstance(instrs[1], tac.Return)
        assert instrs[1].val == tac.Constant(value=ast.ConstInt(5))

    def test_default_emits_label_then_statement(self) -> None:
        """``default: return 0;`` -> ``Label("default_<id>")`` + ``Return(0)``."""
        instrs = _stmt_instrs(
            ast.Default(
                statement=ast.Return(exp=ast.Constant(value=ast.ConstInt(0))),
                identifier=_SWITCH_ID,
            )
        )
        assert isinstance(instrs[0], tac.Label)
        assert instrs[0].name == f"default_{_SWITCH_ID}"
        assert isinstance(instrs[1], tac.Return)
        assert instrs[1].val == tac.Constant(value=ast.ConstInt(0))
