"""Tests for control-flow resolution in the semantic analysis stage.

Covers goto/label validation, if-statement resolution, loop identifier
assignment (while/do-while/for), for-init storage class restrictions, and
switch identifier assignment.
"""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import _make_prog, _program, _resolved_body, _vd

# ---------------------------------------------------------------------------
# Goto statement resolution
# ---------------------------------------------------------------------------


class TestResolveGoto:
    """``Goto`` nodes are validated against the function's label set."""

    def test_goto_valid_label_passes(self) -> None:
        """``goto end;`` where ``end:`` exists in the same function resolves successfully."""
        body = (
            ast.S(statement=ast.Goto(target="end")),
            ast.S(statement=ast.Label(name="end", statement=ast.Return(exp=ast.Constant(value=0)))),
        )
        resolved = _resolved_body(body)
        assert resolved[0] == ast.S(statement=ast.Goto(target="end"))

    def test_goto_forward_reference_is_valid(self) -> None:
        """A ``goto`` that precedes its target label in the body is permitted."""
        body = (
            ast.S(statement=ast.Goto(target="later")),
            ast.S(statement=ast.Return(exp=ast.Constant(value=1))),
            ast.S(statement=ast.Label(name="later", statement=ast.Return(exp=ast.Constant(value=0)))),
        )
        resolved = _resolved_body(body)
        assert resolved[0] == ast.S(statement=ast.Goto(target="later"))

    def test_goto_label_inside_if_branch_is_valid(self) -> None:
        """A label nested inside an ``if`` branch is still reachable by ``goto``."""
        body = (
            ast.S(statement=ast.Goto(target="nested")),
            ast.S(
                statement=ast.If(
                    condition=ast.Constant(value=1),
                    then=ast.Label(name="nested", statement=ast.Return(exp=ast.Constant(value=0))),
                    else_=None,
                )
            ),
        )
        resolved = _resolved_body(body)
        assert resolved[0] == ast.S(statement=ast.Goto(target="nested"))

    def test_goto_undefined_label_raises(self) -> None:
        """A ``goto`` targeting a label not declared anywhere in the function raises ``SemanticError``."""
        body = (ast.S(statement=ast.Goto(target="nonexistent")),)
        with pytest.raises(SemanticError, match="undefined label"):
            resolve(_program(body))

    def test_goto_target_name_not_confused_with_variable(self) -> None:
        """A variable name identical to a label name does not satisfy the goto check."""
        body = (
            _vd("target"),
            ast.S(statement=ast.Goto(target="target")),
        )
        with pytest.raises(SemanticError, match="undefined label"):
            resolve(_program(body))


# ---------------------------------------------------------------------------
# Labeled statement resolution
# ---------------------------------------------------------------------------


class TestResolveLabeledStatement:
    """``Label`` nodes are validated, deduplicated, and their inner statements resolved."""

    def test_label_passes_through_unchanged(self) -> None:
        """A label wrapping a ``Null`` statement is returned with the name intact."""
        body = (ast.S(statement=ast.Label(name="foo", statement=ast.Null())),)
        resolved = _resolved_body(body)
        assert resolved[0] == ast.S(statement=ast.Label(name="foo", statement=ast.Null()))

    def test_label_resolves_inner_statement_variable(self) -> None:
        """Variable references inside a labeled statement are resolved."""
        body = (
            _vd("x"),
            ast.S(statement=ast.Label(name="lbl", statement=ast.Return(exp=ast.Var(name="x")))),
        )
        resolved = _resolved_body(body)
        assert resolved[1] == ast.S(statement=ast.Label(name="lbl", statement=ast.Return(exp=ast.Var(name=".Vx.0"))))

    def test_duplicate_label_at_top_level_raises(self) -> None:
        """Two labels with the same name at the top level of a function raise ``SemanticError``."""
        body = (
            ast.S(statement=ast.Label(name="foo", statement=ast.Null())),
            ast.S(statement=ast.Label(name="foo", statement=ast.Null())),
        )
        with pytest.raises(SemanticError, match="Duplicate label"):
            resolve(_program(body))

    def test_duplicate_label_in_if_branch_raises(self) -> None:
        """A label that duplicates one already defined inside an ``if`` branch raises ``SemanticError``."""
        body = (
            ast.S(statement=ast.Label(name="foo", statement=ast.Null())),
            ast.S(
                statement=ast.If(
                    condition=ast.Constant(value=1),
                    then=ast.Label(name="foo", statement=ast.Null()),
                    else_=None,
                )
            ),
        )
        with pytest.raises(SemanticError, match="Duplicate label"):
            resolve(_program(body))

    def test_duplicate_label_across_both_if_branches_raises(self) -> None:
        """The same label name in both the then- and else-branches raises ``SemanticError``."""
        body = (
            ast.S(
                statement=ast.If(
                    condition=ast.Constant(value=1),
                    then=ast.Label(name="foo", statement=ast.Null()),
                    else_=ast.Label(name="foo", statement=ast.Null()),
                )
            ),
        )
        with pytest.raises(SemanticError, match="Duplicate label"):
            resolve(_program(body))

    def test_distinct_labels_in_same_function_are_accepted(self) -> None:
        """Multiple labels with different names in the same function are all valid."""
        body = (
            ast.S(statement=ast.Label(name="a", statement=ast.Null())),
            ast.S(statement=ast.Label(name="b", statement=ast.Null())),
            ast.S(statement=ast.Label(name="c", statement=ast.Return(exp=ast.Constant(value=0)))),
        )
        resolved = _resolved_body(body)
        assert len(resolved) == 3

    def test_same_label_name_in_different_functions_is_accepted(self) -> None:
        """The same label name may appear in two separate functions without conflict."""

        def _fn(name: str) -> ast.Function:
            return ast.Function(
                name=name,
                params=(),
                body=(ast.S(statement=ast.Label(name="lbl", statement=ast.Return(exp=ast.Constant(value=0)))),),
            )

        program = _make_prog(_fn("foo"), _fn("bar"))
        resolved = resolve(program)
        assert len(resolved.declarations) == 2


# ---------------------------------------------------------------------------
# If statement resolution
# ---------------------------------------------------------------------------


class TestResolveIfStatement:
    """``If`` statements propagate variable resolution into condition and both branches."""

    def test_if_without_else_condition_is_resolved(self) -> None:
        """The condition of an ``if``-without-``else`` has variable references renamed."""
        body = (
            _vd("x"),
            ast.S(
                statement=ast.If(
                    condition=ast.Var(name="x"),
                    then=ast.Return(exp=ast.Constant(value=0)),
                    else_=None,
                )
            ),
        )
        resolved = _resolved_body(body)
        item = resolved[1]
        assert isinstance(item, ast.S)
        assert isinstance(item.statement, ast.If)
        assert item.statement.condition == ast.Var(name=".Vx.0")

    def test_if_without_else_then_branch_is_resolved(self) -> None:
        """The then-branch of an ``if``-without-``else`` has variable references renamed."""
        body = (
            _vd("x"),
            ast.S(
                statement=ast.If(
                    condition=ast.Constant(value=1),
                    then=ast.Return(exp=ast.Var(name="x")),
                    else_=None,
                )
            ),
        )
        resolved = _resolved_body(body)
        item = resolved[1]
        assert isinstance(item, ast.S)
        assert isinstance(item.statement, ast.If)
        assert item.statement.then == ast.Return(exp=ast.Var(name=".Vx.0"))

    def test_if_with_else_all_three_parts_are_resolved(self) -> None:
        """The condition, then-branch, and else-branch all have variable references renamed."""
        body = (
            _vd("x"),
            _vd("y"),
            ast.S(
                statement=ast.If(
                    condition=ast.Var(name="x"),
                    then=ast.Return(exp=ast.Var(name="x")),
                    else_=ast.Return(exp=ast.Var(name="y")),
                )
            ),
        )
        resolved = _resolved_body(body)
        item = resolved[2]
        assert isinstance(item, ast.S)
        assert item.statement == ast.If(
            condition=ast.Var(name=".Vx.0"),
            then=ast.Return(exp=ast.Var(name=".Vx.0")),
            else_=ast.Return(exp=ast.Var(name=".Vy.1")),
        )

    def test_undeclared_variable_in_condition_raises(self) -> None:
        """An undeclared variable in the ``if`` condition raises ``SemanticError``."""
        body = (
            ast.S(
                statement=ast.If(
                    condition=ast.Var(name="undeclared"),
                    then=ast.Return(exp=ast.Constant(value=0)),
                    else_=None,
                )
            ),
        )
        with pytest.raises(SemanticError, match="Undeclared"):
            resolve(_program(body))

    def test_undeclared_variable_in_then_branch_raises(self) -> None:
        """An undeclared variable in the then-branch raises ``SemanticError``."""
        body = (
            ast.S(
                statement=ast.If(
                    condition=ast.Constant(value=1),
                    then=ast.Return(exp=ast.Var(name="undeclared")),
                    else_=None,
                )
            ),
        )
        with pytest.raises(SemanticError, match="Undeclared"):
            resolve(_program(body))

    def test_undeclared_variable_in_else_branch_raises(self) -> None:
        """An undeclared variable in the else-branch raises ``SemanticError``."""
        body = (
            ast.S(
                statement=ast.If(
                    condition=ast.Constant(value=1),
                    then=ast.Return(exp=ast.Constant(value=0)),
                    else_=ast.Return(exp=ast.Var(name="undeclared")),
                )
            ),
        )
        with pytest.raises(SemanticError, match="Undeclared"):
            resolve(_program(body))

    def test_nested_if_variable_references_resolved(self) -> None:
        """Variable references inside a nested ``if`` are resolved at all levels."""
        body = (
            _vd("x"),
            _vd("y"),
            ast.S(
                statement=ast.If(
                    condition=ast.Var(name="x"),
                    then=ast.If(
                        condition=ast.Var(name="y"),
                        then=ast.Return(exp=ast.Var(name="x")),
                        else_=None,
                    ),
                    else_=None,
                )
            ),
        )
        resolved = _resolved_body(body)
        outer_if = resolved[2].statement
        assert isinstance(outer_if, ast.If)
        assert outer_if.condition == ast.Var(name=".Vx.0")
        inner_if = outer_if.then
        assert isinstance(inner_if, ast.If)
        assert inner_if.condition == ast.Var(name=".Vy.1")
        assert inner_if.then == ast.Return(exp=ast.Var(name=".Vx.0"))


# ---------------------------------------------------------------------------
# Loop identifier assignment
# ---------------------------------------------------------------------------


class TestLoopIdentifiers:
    """Loop constructs are assigned unique identifiers; Break/Continue are tagged with their enclosing loop's id."""

    # ------------------------------------------------------------------
    # Identifier format
    # ------------------------------------------------------------------

    def test_while_loop_identifier_uses_while_prefix(self) -> None:
        """A resolved ``While`` node carries an identifier beginning with ``.L.while.``."""
        body = (ast.S(statement=ast.While(condition=ast.Constant(value=1), body=ast.Null())),)
        resolved = _resolved_body(body)
        loop = resolved[0].statement
        assert isinstance(loop, ast.While)
        assert loop.identifier.startswith(".L.while.")

    def test_do_while_loop_identifier_uses_do_prefix(self) -> None:
        """A resolved ``DoWhile`` node carries an identifier beginning with ``.L.do.``."""
        body = (ast.S(statement=ast.DoWhile(body=ast.Null(), condition=ast.Constant(value=1))),)
        resolved = _resolved_body(body)
        loop = resolved[0].statement
        assert isinstance(loop, ast.DoWhile)
        assert loop.identifier.startswith(".L.do.")

    def test_for_loop_identifier_uses_for_prefix(self) -> None:
        """A resolved ``For`` node carries an identifier beginning with ``.L.for.``."""
        body = (ast.S(statement=ast.For(init=ast.InitExp(exp=None), condition=None, post=None, body=ast.Null())),)
        resolved = _resolved_body(body)
        loop = resolved[0].statement
        assert isinstance(loop, ast.For)
        assert loop.identifier.startswith(".L.for.")

    # ------------------------------------------------------------------
    # Loop counter independence
    # ------------------------------------------------------------------

    def test_loop_counter_independent_of_variable_counter(self) -> None:
        """The first loop's identifier ends in ``.0`` regardless of how many variables were declared first."""
        body = (
            _vd("a"),
            _vd("b"),
            _vd("c"),
            ast.S(statement=ast.While(condition=ast.Constant(value=1), body=ast.Null())),
        )
        resolved = _resolved_body(body)
        loop = resolved[3].statement
        assert isinstance(loop, ast.While)
        assert loop.identifier == ".L.while.0"

    # ------------------------------------------------------------------
    # Break and Continue tagging — each loop type
    # ------------------------------------------------------------------

    def test_break_inside_while_tagged_with_loop_identifier(self) -> None:
        """A ``Break`` directly inside a ``While`` body receives the loop's identifier."""
        body = (ast.S(statement=ast.While(condition=ast.Constant(value=1), body=ast.Break())),)
        resolved = _resolved_body(body)
        loop = resolved[0].statement
        assert isinstance(loop, ast.While)
        assert loop.body == ast.Break(loop.identifier)

    def test_continue_inside_while_tagged_with_loop_identifier(self) -> None:
        """A ``Continue`` directly inside a ``While`` body receives the loop's identifier."""
        body = (ast.S(statement=ast.While(condition=ast.Constant(value=1), body=ast.Continue())),)
        resolved = _resolved_body(body)
        loop = resolved[0].statement
        assert isinstance(loop, ast.While)
        assert loop.body == ast.Continue(loop.identifier)

    def test_break_inside_do_while_tagged_with_loop_identifier(self) -> None:
        """A ``Break`` inside a ``DoWhile`` body receives the loop's identifier."""
        body = (ast.S(statement=ast.DoWhile(body=ast.Break(), condition=ast.Constant(value=1))),)
        resolved = _resolved_body(body)
        loop = resolved[0].statement
        assert isinstance(loop, ast.DoWhile)
        assert loop.body == ast.Break(loop.identifier)

    def test_continue_inside_do_while_tagged_with_loop_identifier(self) -> None:
        """A ``Continue`` inside a ``DoWhile`` body receives the loop's identifier."""
        body = (ast.S(statement=ast.DoWhile(body=ast.Continue(), condition=ast.Constant(value=1))),)
        resolved = _resolved_body(body)
        loop = resolved[0].statement
        assert isinstance(loop, ast.DoWhile)
        assert loop.body == ast.Continue(loop.identifier)

    def test_break_inside_for_tagged_with_loop_identifier(self) -> None:
        """A ``Break`` inside a ``For`` body receives the loop's identifier."""
        body = (ast.S(statement=ast.For(init=ast.InitExp(exp=None), condition=None, post=None, body=ast.Break())),)
        resolved = _resolved_body(body)
        loop = resolved[0].statement
        assert isinstance(loop, ast.For)
        assert loop.body == ast.Break(loop.identifier)

    def test_continue_inside_for_tagged_with_loop_identifier(self) -> None:
        """A ``Continue`` inside a ``For`` body receives the loop's identifier."""
        body = (ast.S(statement=ast.For(init=ast.InitExp(exp=None), condition=None, post=None, body=ast.Continue())),)
        resolved = _resolved_body(body)
        loop = resolved[0].statement
        assert isinstance(loop, ast.For)
        assert loop.body == ast.Continue(loop.identifier)

    def test_break_and_continue_in_same_loop_share_identifier(self) -> None:
        """Both ``Break`` and ``Continue`` inside the same loop carry identical identifiers."""
        body = (
            ast.S(
                statement=ast.While(
                    condition=ast.Constant(value=1),
                    body=ast.Compound(
                        body=(
                            ast.S(statement=ast.Break()),
                            ast.S(statement=ast.Continue()),
                        )
                    ),
                )
            ),
        )
        resolved = _resolved_body(body)
        loop = resolved[0].statement
        assert isinstance(loop, ast.While)
        compound = loop.body
        assert isinstance(compound, ast.Compound)
        break_id = compound.body[0].statement.identifier
        continue_id = compound.body[1].statement.identifier
        assert break_id == loop.identifier
        assert continue_id == loop.identifier

    # ------------------------------------------------------------------
    # Sequential and nested loops
    # ------------------------------------------------------------------

    def test_sequential_loops_receive_distinct_identifiers(self) -> None:
        """Two consecutive loops in the same function body each get a distinct identifier."""
        body = (
            ast.S(statement=ast.While(condition=ast.Constant(value=1), body=ast.Null())),
            ast.S(statement=ast.While(condition=ast.Constant(value=1), body=ast.Null())),
        )
        resolved = _resolved_body(body)
        first = resolved[0].statement
        second = resolved[1].statement
        assert isinstance(first, ast.While)
        assert isinstance(second, ast.While)
        assert first.identifier != second.identifier

    def test_nested_loops_receive_distinct_identifiers(self) -> None:
        """An inner loop and its enclosing outer loop are assigned different identifiers."""
        body = (
            ast.S(
                statement=ast.While(
                    condition=ast.Constant(value=1),
                    body=ast.While(condition=ast.Constant(value=1), body=ast.Null()),
                )
            ),
        )
        resolved = _resolved_body(body)
        outer = resolved[0].statement
        assert isinstance(outer, ast.While)
        inner = outer.body
        assert isinstance(inner, ast.While)
        assert outer.identifier != inner.identifier

    def test_break_in_inner_loop_tagged_with_inner_identifier(self) -> None:
        """A ``Break`` inside an inner loop is tagged with the inner loop's identifier, not the outer's."""
        body = (
            ast.S(
                statement=ast.While(
                    condition=ast.Constant(value=1),
                    body=ast.While(condition=ast.Constant(value=1), body=ast.Break()),
                )
            ),
        )
        resolved = _resolved_body(body)
        outer = resolved[0].statement
        assert isinstance(outer, ast.While)
        inner = outer.body
        assert isinstance(inner, ast.While)
        assert inner.body == ast.Break(inner.identifier)
        assert inner.body != ast.Break(outer.identifier)

    def test_break_after_inner_loop_tagged_with_outer_identifier(self) -> None:
        """After an inner loop exits, a ``Break`` in the outer loop body uses the outer identifier."""
        body = (
            ast.S(
                statement=ast.While(
                    condition=ast.Constant(value=1),
                    body=ast.Compound(
                        body=(
                            ast.S(statement=ast.While(condition=ast.Constant(value=1), body=ast.Null())),
                            ast.S(statement=ast.Break()),
                        )
                    ),
                )
            ),
        )
        resolved = _resolved_body(body)
        outer = resolved[0].statement
        assert isinstance(outer, ast.While)
        compound = outer.body
        assert isinstance(compound, ast.Compound)
        break_stmt = compound.body[1].statement
        assert isinstance(break_stmt, ast.Break)
        assert break_stmt.identifier == outer.identifier

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_break_outside_any_loop_raises(self) -> None:
        """A ``Break`` at function scope (not inside any loop) raises ``SemanticError``."""
        body = (ast.S(statement=ast.Break()),)
        with pytest.raises(SemanticError, match="not within a loop"):
            resolve(_program(body))

    def test_continue_outside_any_loop_raises(self) -> None:
        """A ``Continue`` at function scope (not inside any loop) raises ``SemanticError``."""
        body = (ast.S(statement=ast.Continue()),)
        with pytest.raises(SemanticError, match="not within a loop"):
            resolve(_program(body))

    def test_break_inside_if_but_outside_loop_raises(self) -> None:
        """A ``Break`` nested inside an ``If`` that is not itself inside a loop raises ``SemanticError``."""
        body = (
            ast.S(
                statement=ast.If(
                    condition=ast.Constant(value=1),
                    then=ast.Break(),
                    else_=None,
                )
            ),
        )
        with pytest.raises(SemanticError, match="not within a loop"):
            resolve(_program(body))

    def test_continue_inside_if_but_outside_loop_raises(self) -> None:
        """A ``Continue`` nested inside an ``If`` that is not itself inside a loop raises ``SemanticError``."""
        body = (
            ast.S(
                statement=ast.If(
                    condition=ast.Constant(value=1),
                    then=ast.Continue(),
                    else_=None,
                )
            ),
        )
        with pytest.raises(SemanticError, match="not within a loop"):
            resolve(_program(body))


# ---------------------------------------------------------------------------
# For-loop initialiser validation
# ---------------------------------------------------------------------------


class TestForInitStorageClass:
    """Storage class specifiers are forbidden in ``for`` loop initialisers."""

    def test_for_init_static_raises(self) -> None:
        """A ``static`` declaration in a ``for`` init clause raises ``SemanticError``."""
        body = (
            ast.S(
                ast.For(
                    init=ast.InitDecl(
                        ast.VarDecl(ast.VariableDeclaration("i", ast.Constant(0), ast.StorageClass.Static))
                    ),
                    condition=None,
                    post=None,
                    body=ast.Null(),
                )
            ),
        )
        with pytest.raises(SemanticError, match="storage class"):
            resolve(_program(body))

    def test_for_init_extern_raises(self) -> None:
        """An ``extern`` declaration in a ``for`` init clause raises ``SemanticError``."""
        body = (
            ast.S(
                ast.For(
                    init=ast.InitDecl(ast.VarDecl(ast.VariableDeclaration("i", None, ast.StorageClass.Extern))),
                    condition=None,
                    post=None,
                    body=ast.Null(),
                )
            ),
        )
        with pytest.raises(SemanticError, match="storage class"):
            resolve(_program(body))

    def test_for_init_no_storage_class_is_valid(self) -> None:
        """A plain declaration in a ``for`` init clause is accepted."""
        body = (
            ast.S(
                ast.For(
                    init=ast.InitDecl(ast.VarDecl(ast.VariableDeclaration("i", ast.Constant(0)))),
                    condition=None,
                    post=None,
                    body=ast.Null(),
                )
            ),
        )
        resolve(_program(body))  # must not raise


# ---------------------------------------------------------------------------
# Switch identifier assignment
# ---------------------------------------------------------------------------


class TestSwitchIdentifiers:
    """Switch constructs are assigned unique identifiers; Case/Default/Break are tagged correctly."""

    # ------------------------------------------------------------------
    # Identifier format and counter
    # ------------------------------------------------------------------

    def test_switch_identifier_uses_switch_prefix(self) -> None:
        """A resolved ``Switch`` node carries an identifier beginning with ``.L.switch.``."""
        body = (ast.S(statement=ast.Switch(control=ast.Constant(value=1), body=ast.Null())),)
        resolved = _resolved_body(body)
        switch = resolved[0].statement
        assert isinstance(switch, ast.Switch)
        assert switch.identifier.startswith(".L.switch.")

    def test_switch_counter_independent_of_variable_counter(self) -> None:
        """The first switch's identifier ends in ``.0`` regardless of how many variables were declared first."""
        body = (
            _vd("a"),
            _vd("b"),
            ast.S(statement=ast.Switch(control=ast.Constant(value=1), body=ast.Null())),
        )
        resolved = _resolved_body(body)
        switch = resolved[2].statement
        assert isinstance(switch, ast.Switch)
        assert switch.identifier == ".L.switch.0"

    def test_sequential_switches_receive_distinct_identifiers(self) -> None:
        """Two consecutive ``Switch`` statements each receive a distinct identifier."""
        body = (
            ast.S(statement=ast.Switch(control=ast.Constant(value=1), body=ast.Null())),
            ast.S(statement=ast.Switch(control=ast.Constant(value=2), body=ast.Null())),
        )
        resolved = _resolved_body(body)
        first = resolved[0].statement
        second = resolved[1].statement
        assert isinstance(first, ast.Switch)
        assert isinstance(second, ast.Switch)
        assert first.identifier != second.identifier

    def test_nested_switches_receive_distinct_identifiers(self) -> None:
        """An inner ``Switch`` and its enclosing outer ``Switch`` receive different identifiers."""
        body = (
            ast.S(
                statement=ast.Switch(
                    control=ast.Constant(value=1),
                    body=ast.Switch(control=ast.Constant(value=2), body=ast.Null()),
                )
            ),
        )
        resolved = _resolved_body(body)
        outer = resolved[0].statement
        assert isinstance(outer, ast.Switch)
        inner = outer.body
        assert isinstance(inner, ast.Switch)
        assert outer.identifier != inner.identifier

    # ------------------------------------------------------------------
    # Case and Default identifier assignment
    # ------------------------------------------------------------------

    def test_case_identifier_matches_enclosing_switch(self) -> None:
        """A resolved ``Case`` carries the identifier of its enclosing ``Switch``."""
        body = (
            ast.S(
                statement=ast.Switch(
                    control=ast.Constant(value=1),
                    body=ast.Case(value=ast.Constant(value=1), statement=ast.Null()),
                )
            ),
        )
        resolved = _resolved_body(body)
        switch = resolved[0].statement
        assert isinstance(switch, ast.Switch)
        case_ = switch.body
        assert isinstance(case_, ast.Case)
        assert case_.identifier == switch.identifier

    def test_default_identifier_matches_enclosing_switch(self) -> None:
        """A resolved ``Default`` carries the identifier of its enclosing ``Switch``."""
        body = (
            ast.S(
                statement=ast.Switch(
                    control=ast.Constant(value=1),
                    body=ast.Default(statement=ast.Null()),
                )
            ),
        )
        resolved = _resolved_body(body)
        switch = resolved[0].statement
        assert isinstance(switch, ast.Switch)
        default_ = switch.body
        assert isinstance(default_, ast.Default)
        assert default_.identifier == switch.identifier

    def test_case_in_inner_switch_gets_inner_switch_identifier(self) -> None:
        """A ``Case`` nested inside an inner ``Switch`` is tagged with the inner switch's id, not the outer's."""
        body = (
            ast.S(
                statement=ast.Switch(
                    control=ast.Constant(value=1),
                    body=ast.Switch(
                        control=ast.Constant(value=2),
                        body=ast.Case(value=ast.Constant(value=2), statement=ast.Null()),
                    ),
                )
            ),
        )
        resolved = _resolved_body(body)
        outer = resolved[0].statement
        assert isinstance(outer, ast.Switch)
        inner = outer.body
        assert isinstance(inner, ast.Switch)
        case_ = inner.body
        assert isinstance(case_, ast.Case)
        assert case_.identifier == inner.identifier
        assert case_.identifier != outer.identifier

    def test_case_and_default_in_compound_body_both_tagged(self) -> None:
        """Both ``Case`` and ``Default`` inside a compound switch body carry the switch's identifier."""
        body = (
            ast.S(
                statement=ast.Switch(
                    control=ast.Constant(value=1),
                    body=ast.Compound(
                        body=(
                            ast.S(statement=ast.Case(value=ast.Constant(value=1), statement=ast.Null())),
                            ast.S(statement=ast.Default(statement=ast.Null())),
                        )
                    ),
                )
            ),
        )
        resolved = _resolved_body(body)
        switch = resolved[0].statement
        assert isinstance(switch, ast.Switch)
        compound = switch.body
        assert isinstance(compound, ast.Compound)
        case_ = compound.body[0].statement
        default_ = compound.body[1].statement
        assert isinstance(case_, ast.Case)
        assert isinstance(default_, ast.Default)
        assert case_.identifier == switch.identifier
        assert default_.identifier == switch.identifier

    # ------------------------------------------------------------------
    # Break tagging
    # ------------------------------------------------------------------

    def test_break_inside_switch_tagged_with_switch_identifier(self) -> None:
        """A ``Break`` directly inside a ``Switch`` body is tagged with the switch's identifier."""
        body = (ast.S(statement=ast.Switch(control=ast.Constant(value=1), body=ast.Break())),)
        resolved = _resolved_body(body)
        switch = resolved[0].statement
        assert isinstance(switch, ast.Switch)
        assert switch.body == ast.Break(switch.identifier)

    def test_break_inside_switch_inside_loop_targets_switch_not_loop(self) -> None:
        """When a switch is nested inside a loop, ``break`` exits the switch, not the loop."""
        body = (
            ast.S(
                statement=ast.While(
                    condition=ast.Constant(value=1),
                    body=ast.Switch(control=ast.Constant(value=1), body=ast.Break()),
                )
            ),
        )
        resolved = _resolved_body(body)
        loop = resolved[0].statement
        assert isinstance(loop, ast.While)
        switch = loop.body
        assert isinstance(switch, ast.Switch)
        assert switch.body == ast.Break(switch.identifier)
        assert switch.body != ast.Break(loop.identifier)

    def test_break_after_switch_inside_loop_targets_loop(self) -> None:
        """After the inner switch closes, a ``Break`` in the outer loop body targets the loop."""
        body = (
            ast.S(
                statement=ast.While(
                    condition=ast.Constant(value=1),
                    body=ast.Compound(
                        body=(
                            ast.S(statement=ast.Switch(control=ast.Constant(value=1), body=ast.Null())),
                            ast.S(statement=ast.Break()),
                        )
                    ),
                )
            ),
        )
        resolved = _resolved_body(body)
        loop = resolved[0].statement
        assert isinstance(loop, ast.While)
        compound = loop.body
        assert isinstance(compound, ast.Compound)
        break_stmt = compound.body[1].statement
        assert isinstance(break_stmt, ast.Break)
        assert break_stmt.identifier == loop.identifier

    # ------------------------------------------------------------------
    # Continue inside switch
    # ------------------------------------------------------------------

    def test_continue_inside_switch_but_outside_loop_raises(self) -> None:
        """A ``Continue`` inside a bare switch (no enclosing loop) raises ``SemanticError``."""
        body = (ast.S(statement=ast.Switch(control=ast.Constant(value=1), body=ast.Continue())),)
        with pytest.raises(SemanticError, match="not within a loop"):
            resolve(_program(body))

    def test_continue_inside_switch_inside_loop_targets_loop(self) -> None:
        """A ``Continue`` inside a switch that is itself inside a loop targets the enclosing loop."""
        body = (
            ast.S(
                statement=ast.While(
                    condition=ast.Constant(value=1),
                    body=ast.Switch(control=ast.Constant(value=1), body=ast.Continue()),
                )
            ),
        )
        resolved = _resolved_body(body)
        loop = resolved[0].statement
        assert isinstance(loop, ast.While)
        switch = loop.body
        assert isinstance(switch, ast.Switch)
        assert switch.body == ast.Continue(loop.identifier)
        assert switch.body != ast.Continue(switch.identifier)

    # ------------------------------------------------------------------
    # Variable resolution inside switch constructs
    # ------------------------------------------------------------------

    def test_switch_control_expression_variable_is_resolved(self) -> None:
        """A variable reference in the switch control expression is renamed to its unique identifier."""
        body = (
            _vd("x"),
            ast.S(statement=ast.Switch(control=ast.Var(name="x"), body=ast.Null())),
        )
        resolved = _resolved_body(body)
        switch = resolved[1].statement
        assert isinstance(switch, ast.Switch)
        assert switch.control == ast.Var(name=".Vx.0")

    def test_case_governed_statement_variable_is_resolved(self) -> None:
        """A variable reference in the statement governed by a ``Case`` is renamed."""
        body = (
            _vd("x"),
            ast.S(
                statement=ast.Switch(
                    control=ast.Constant(value=1),
                    body=ast.Case(
                        value=ast.Constant(value=1),
                        statement=ast.Return(exp=ast.Var(name="x")),
                    ),
                )
            ),
        )
        resolved = _resolved_body(body)
        switch = resolved[1].statement
        assert isinstance(switch, ast.Switch)
        case_ = switch.body
        assert isinstance(case_, ast.Case)
        assert case_.statement == ast.Return(exp=ast.Var(name=".Vx.0"))

    def test_default_governed_statement_variable_is_resolved(self) -> None:
        """A variable reference in the statement governed by a ``Default`` is renamed."""
        body = (
            _vd("x"),
            ast.S(
                statement=ast.Switch(
                    control=ast.Constant(value=1),
                    body=ast.Default(statement=ast.Return(exp=ast.Var(name="x"))),
                )
            ),
        )
        resolved = _resolved_body(body)
        switch = resolved[1].statement
        assert isinstance(switch, ast.Switch)
        default_ = switch.body
        assert isinstance(default_, ast.Default)
        assert default_.statement == ast.Return(exp=ast.Var(name=".Vx.0"))

    # ------------------------------------------------------------------
    # Goto label collection inside switch bodies
    # ------------------------------------------------------------------

    def test_label_inside_switch_body_reachable_by_goto(self) -> None:
        """A ``goto`` label nested inside a switch body is collected in the first pass and is a valid target."""
        body = (
            ast.S(statement=ast.Goto(target="lbl")),
            ast.S(
                statement=ast.Switch(
                    control=ast.Constant(value=1),
                    body=ast.Case(
                        value=ast.Constant(value=1),
                        statement=ast.Label(name="lbl", statement=ast.Null()),
                    ),
                )
            ),
        )
        resolved = _resolved_body(body)
        assert resolved[0] == ast.S(statement=ast.Goto(target="lbl"))

    def test_duplicate_label_inside_switch_raises(self) -> None:
        """A label appearing twice within a switch body raises ``SemanticError``."""
        body = (
            ast.S(statement=ast.Label(name="lbl", statement=ast.Null())),
            ast.S(
                statement=ast.Switch(
                    control=ast.Constant(value=1),
                    body=ast.Case(
                        value=ast.Constant(value=1),
                        statement=ast.Label(name="lbl", statement=ast.Null()),
                    ),
                )
            ),
        )
        with pytest.raises(SemanticError, match="Duplicate label"):
            resolve(_program(body))

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_break_outside_loop_or_switch_raises(self) -> None:
        """A ``Break`` at function scope (not inside any loop or switch) raises ``SemanticError``."""
        body = (ast.S(statement=ast.Break()),)
        with pytest.raises(SemanticError, match="not within a loop or switch"):
            resolve(_program(body))

    def test_case_outside_switch_raises(self) -> None:
        """A ``Case`` appearing outside any enclosing ``Switch`` raises ``SemanticError``."""
        body = (ast.S(statement=ast.Case(value=ast.Constant(value=1), statement=ast.Null())),)
        with pytest.raises(SemanticError, match="not within a switch"):
            resolve(_program(body))

    def test_default_outside_switch_raises(self) -> None:
        """A ``Default`` appearing outside any enclosing ``Switch`` raises ``SemanticError``."""
        body = (ast.S(statement=ast.Default(statement=ast.Null())),)
        with pytest.raises(SemanticError, match="not within a switch"):
            resolve(_program(body))

    def test_duplicate_case_value_in_same_switch_raises(self) -> None:
        """Two ``Case`` labels with the same integer value in one switch raise ``SemanticError``."""
        body = (
            ast.S(
                statement=ast.Switch(
                    control=ast.Constant(value=1),
                    body=ast.Compound(
                        body=(
                            ast.S(statement=ast.Case(value=ast.Constant(value=5), statement=ast.Null())),
                            ast.S(statement=ast.Case(value=ast.Constant(value=4), statement=ast.Null())),
                            ast.S(statement=ast.Case(value=ast.Constant(value=5), statement=ast.Null())),
                        )
                    ),
                )
            ),
        )
        with pytest.raises(SemanticError, match="Duplicate case value"):
            resolve(_program(body))

    def test_same_case_value_in_different_switches_is_accepted(self) -> None:
        """The same integer value may appear as a ``Case`` in two separate ``Switch`` statements."""
        body = (
            ast.S(
                statement=ast.Switch(
                    control=ast.Constant(value=1),
                    body=ast.Case(value=ast.Constant(value=5), statement=ast.Null()),
                )
            ),
            ast.S(
                statement=ast.Switch(
                    control=ast.Constant(value=2),
                    body=ast.Case(value=ast.Constant(value=5), statement=ast.Null()),
                )
            ),
        )
        resolved = _resolved_body(body)
        assert len(resolved) == 2

    def test_same_case_value_in_nested_switches_is_accepted(self) -> None:
        """The same integer value may appear as a ``Case`` in an outer and inner switch independently."""
        body = (
            ast.S(
                statement=ast.Switch(
                    control=ast.Constant(value=1),
                    body=ast.Case(
                        value=ast.Constant(value=5),
                        statement=ast.Switch(
                            control=ast.Constant(value=2),
                            body=ast.Case(value=ast.Constant(value=5), statement=ast.Null()),
                        ),
                    ),
                )
            ),
        )
        resolved = _resolved_body(body)
        assert len(resolved) == 1

    def test_duplicate_default_in_same_switch_raises(self) -> None:
        """Two ``Default`` labels within a single switch raise ``SemanticError``."""
        body = (
            ast.S(
                statement=ast.Switch(
                    control=ast.Constant(value=1),
                    body=ast.Compound(
                        body=(
                            ast.S(statement=ast.Default(statement=ast.Null())),
                            ast.S(statement=ast.Default(statement=ast.Null())),
                        )
                    ),
                )
            ),
        )
        with pytest.raises(SemanticError, match="Multiple 'default'"):
            resolve(_program(body))

    def test_default_in_each_of_two_sequential_switches_is_accepted(self) -> None:
        """A ``Default`` in each of two separate sequential switches is permitted."""
        body = (
            ast.S(
                statement=ast.Switch(
                    control=ast.Constant(value=1),
                    body=ast.Default(statement=ast.Null()),
                )
            ),
            ast.S(
                statement=ast.Switch(
                    control=ast.Constant(value=2),
                    body=ast.Default(statement=ast.Null()),
                )
            ),
        )
        resolved = _resolved_body(body)
        assert len(resolved) == 2

    def test_default_in_each_of_two_nested_switches_is_accepted(self) -> None:
        """A ``Default`` in both an outer and an inner switch is permitted."""
        body = (
            ast.S(
                statement=ast.Switch(
                    control=ast.Constant(value=1),
                    body=ast.Compound(
                        body=(
                            ast.S(
                                statement=ast.Default(
                                    statement=ast.Switch(
                                        control=ast.Constant(value=2),
                                        body=ast.Default(statement=ast.Null()),
                                    )
                                )
                            ),
                        )
                    ),
                )
            ),
        )
        resolved = _resolved_body(body)
        assert len(resolved) == 1
