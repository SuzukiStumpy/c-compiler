"""Tests for the semantic analysis stage.

Verifies that ``SemanticAnalyser`` and the ``resolve`` convenience function
correctly rename variable declarations to unique internal identifiers, reject
semantic violations (duplicate declarations, undeclared variables/functions,
invalid lvalues, function redefinitions), validate function declarations, and
leave non-variable constructs unchanged.

Each test constructs ``ast`` nodes directly so that the semantic analyser is
exercised in isolation, without invoking the lexer or parser.
"""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import FunctionBinding, SemanticAnalyser, VariableBinding, resolve
from compiler.syntax.ast.types import Linkage
from compiler.utils.exceptions import SemanticError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vd(name: str, init: ast.Exp | None = None) -> ast.D:
    """Return a ``D`` block item wrapping a variable declaration.

    Args:
        name: The source-level variable name.
        init: Optional initialiser expression.

    Returns:
        An ``ast.D`` wrapping ``ast.VarDecl(ast.VariableDeclaration(name, init))``.
    """
    return ast.D(ast.VarDecl(ast.VariableDeclaration(name, init)))


def _program(body: tuple[ast.BlockItem, ...]) -> ast.Program:
    """Wrap a sequence of block items in a minimal single-function ``ast.Program``.

    Args:
        body: The block items to place in the body of ``main``.

    Returns:
        An ``ast.Program`` containing a single ``main`` function with the
        given body.
    """
    return ast.Program(
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


def _make_prog(*fns: ast.Function) -> ast.Program:
    """Wrap one or more ``Function`` nodes in an ``ast.Program``.

    Args:
        *fns: The ``Function`` nodes to include in the program, in source order.

    Returns:
        An ``ast.Program`` whose ``declarations`` tuple contains each function
        wrapped in an ``ast.FunDecl``.
    """
    return ast.Program(declarations=tuple(ast.FunDecl(fn) for fn in fns))


def _get_fn(program: ast.Program, index: int = 0) -> ast.Function:
    """Return the *n*-th function declaration from a resolved program.

    Args:
        program: The resolved ``ast.Program``.
        index: Zero-based index into the list of ``FunDecl`` declarations.

    Returns:
        The ``ast.Function`` at the given position among the program's
        ``FunDecl`` entries.
    """
    fns = [d.function_declaration for d in program.declarations if isinstance(d, ast.FunDecl)]
    return fns[index]


def _resolved_body(body: tuple[ast.BlockItem, ...]) -> tuple[ast.BlockItem, ...]:
    """Resolve a function body and return the resolved block items.

    Args:
        body: The block items to resolve.

    Returns:
        The resolved block items from the single function in the resolved
        program.
    """
    return _get_fn(resolve(_program(body))).body


# ---------------------------------------------------------------------------
# Variable declaration resolution
# ---------------------------------------------------------------------------


class TestResolveDeclaration:
    """``resolve_variable_declaration`` renames variables and enforces scope rules."""

    def test_declaration_assigns_unique_suffix(self) -> None:
        """The first declaration in a fresh analyser receives the unique name ``.Vx.0``."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(ast.VariableDeclaration(name="x", init=None))
        assert result.name == ".Vx.0"

    def test_declaration_preserves_none_init(self) -> None:
        """A declaration without an initialiser keeps ``init=None`` after resolution."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(ast.VariableDeclaration(name="x", init=None))
        assert result.init is None

    def test_declaration_with_constant_init_passes_through(self) -> None:
        """A constant initialiser is returned unchanged."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(ast.VariableDeclaration(name="x", init=ast.Constant(value=5)))
        assert result.init == ast.Constant(value=5)

    def test_declaration_with_var_init_renames_init_var(self) -> None:
        """An initialiser that references an already-declared variable is renamed."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(ast.VariableDeclaration(name="y", init=None))  # y → .Vy.0
        result = analyser.resolve_variable_declaration(ast.VariableDeclaration(name="x", init=ast.Var(name="y")))
        assert result.init == ast.Var(name=".Vy.0")

    def test_two_declarations_receive_distinct_suffixes(self) -> None:
        """Successive declarations in the same analyser get incrementing suffixes."""
        analyser = SemanticAnalyser()
        r1 = analyser.resolve_variable_declaration(ast.VariableDeclaration(name="x", init=None))
        r2 = analyser.resolve_variable_declaration(ast.VariableDeclaration(name="y", init=None))
        assert r1.name == ".Vx.0"
        assert r2.name == ".Vy.1"

    def test_duplicate_declaration_raises(self) -> None:
        """Declaring the same name twice in the same scope raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(ast.VariableDeclaration(name="x", init=None))
        with pytest.raises(SemanticError, match="Duplicate"):
            analyser.resolve_variable_declaration(ast.VariableDeclaration(name="x", init=None))

    def test_self_referential_init_is_accepted(self) -> None:
        """``int x = x;`` is accepted: C scoping places ``x`` in scope before the initialiser.

        Reading ``x`` before it is initialised is undefined behaviour at runtime,
        but is not a constraint violation and must not be rejected by the analyser.
        """
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(ast.VariableDeclaration(name="x", init=ast.Var(name="x")))
        assert result.name == ".Vx.0"
        assert result.init == ast.Var(name=".Vx.0")

    def test_short_circuit_init_referencing_own_variable_is_accepted(self) -> None:
        """``int a = 0 && a;`` is accepted: ``a`` is in scope and the right side is never evaluated."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration(
                name="a", init=ast.Binary(operator=ast.And(), left=ast.Constant(value=0), right=ast.Var(name="a"))
            )
        )
        assert result.name == ".Va.0"


# ---------------------------------------------------------------------------
# Expression resolution
# ---------------------------------------------------------------------------


class TestResolveExp:
    """``resolve_exp`` substitutes ``Var`` nodes and validates lvalues."""

    def _analyser_with(self, *names: str) -> SemanticAnalyser:
        """Return an analyser that has already declared each name in order.

        Args:
            *names: Source-level variable names to pre-declare.

        Returns:
            A ``SemanticAnalyser`` whose symbol table maps each name to its
            unique internal identifier.
        """
        analyser = SemanticAnalyser()
        for name in names:
            analyser.resolve_variable_declaration(ast.VariableDeclaration(name=name, init=None))
        return analyser

    def test_constant_is_returned_unchanged(self) -> None:
        """``Constant`` nodes carry no variable references and pass through as-is."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_exp(ast.Constant(value=42))
        assert result == ast.Constant(value=42)

    def test_known_var_is_renamed(self) -> None:
        """A ``Var`` whose name is in the symbol table is replaced by its unique name."""
        analyser = self._analyser_with("x")
        result = analyser.resolve_exp(ast.Var(name="x"))
        assert result == ast.Var(name=".Vx.0")

    def test_undeclared_var_raises(self) -> None:
        """A ``Var`` not present in the symbol table raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        with pytest.raises(SemanticError, match="Undeclared"):
            analyser.resolve_exp(ast.Var(name="undeclared"))

    def test_unary_recurses_into_operand(self) -> None:
        """``Unary`` expressions have their sub-expression resolved."""
        analyser = self._analyser_with("x")
        result = analyser.resolve_exp(ast.Unary(operator=ast.Negate(), exp=ast.Var(name="x")))
        assert result == ast.Unary(operator=ast.Negate(), exp=ast.Var(name=".Vx.0"))

    def test_binary_recurses_into_both_operands(self) -> None:
        """Both sides of a ``Binary`` expression are resolved independently."""
        analyser = self._analyser_with("x", "y")
        result = analyser.resolve_exp(ast.Binary(operator=ast.Add(), left=ast.Var(name="x"), right=ast.Var(name="y")))
        assert result == ast.Binary(operator=ast.Add(), left=ast.Var(name=".Vx.0"), right=ast.Var(name=".Vy.1"))

    def test_assignment_with_var_lvalue_resolves_both_sides(self) -> None:
        """An assignment whose lvalue is a ``Var`` resolves both lvalue and rvalue."""
        analyser = self._analyser_with("x")
        result = analyser.resolve_exp(ast.Assignment(lvalue=ast.Var(name="x"), rvalue=ast.Constant(value=10)))
        assert result == ast.Assignment(lvalue=ast.Var(name=".Vx.0"), rvalue=ast.Constant(value=10))

    def test_assignment_with_non_var_lvalue_raises(self) -> None:
        """An assignment whose lvalue is not a ``Var`` raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        with pytest.raises(SemanticError, match="Invalid lvalue"):
            analyser.resolve_exp(ast.Assignment(lvalue=ast.Constant(value=1), rvalue=ast.Constant(value=2)))

    def test_conditional_recurses_into_all_three_branches(self) -> None:
        """All three sub-expressions of a ``Conditional`` are resolved independently."""
        analyser = self._analyser_with("x", "y")
        result = analyser.resolve_exp(
            ast.Conditional(
                condition=ast.Var(name="x"),
                exp_true=ast.Var(name="x"),
                exp_false=ast.Var(name="y"),
            )
        )
        assert result == ast.Conditional(
            condition=ast.Var(name=".Vx.0"),
            exp_true=ast.Var(name=".Vx.0"),
            exp_false=ast.Var(name=".Vy.1"),
        )

    def test_conditional_undeclared_variable_raises(self) -> None:
        """An undeclared variable in any branch of a ``Conditional`` raises ``SemanticError``."""
        analyser = self._analyser_with("x")
        with pytest.raises(SemanticError, match="Undeclared"):
            analyser.resolve_exp(
                ast.Conditional(
                    condition=ast.Var(name="x"),
                    exp_true=ast.Constant(value=1),
                    exp_false=ast.Var(name="undeclared"),
                )
            )

    def test_unknown_expression_type_raises(self) -> None:
        """An unrecognised ``Exp`` subclass raises ``SemanticError``."""

        class _FakeExp(ast.Exp):
            pass

        analyser = SemanticAnalyser()
        with pytest.raises(SemanticError, match="Unhandled expression type"):
            analyser.resolve_exp(_FakeExp())


# ---------------------------------------------------------------------------
# Statement resolution
# ---------------------------------------------------------------------------


class TestResolveStatement:
    """Statements propagate variable resolution correctly into their expressions."""

    def test_return_statement_var_is_renamed(self) -> None:
        """``return x;`` has the variable reference renamed after declaration."""
        body = (
            _vd("x"),
            ast.S(statement=ast.Return(exp=ast.Var(name="x"))),
        )
        resolved = _resolved_body(body)
        assert resolved[1] == ast.S(statement=ast.Return(exp=ast.Var(name=".Vx.0")))

    def test_expression_statement_var_is_renamed(self) -> None:
        """An expression statement containing a variable reference is resolved."""
        body = (
            _vd("x"),
            ast.S(statement=ast.Expression(exp=ast.Var(name="x"))),
        )
        resolved = _resolved_body(body)
        assert resolved[1] == ast.S(statement=ast.Expression(exp=ast.Var(name=".Vx.0")))

    def test_null_statement_passes_through(self) -> None:
        """A ``Null`` statement contains no expressions and is returned unchanged."""
        body = (ast.S(statement=ast.Null()),)
        resolved = _resolved_body(body)
        assert resolved[0] == ast.S(statement=ast.Null())

    def test_unknown_statement_type_raises(self) -> None:
        """An unrecognised ``Statement`` subclass raises ``SemanticError``."""

        class _FakeStmt(ast.Statement):
            pass

        with pytest.raises(SemanticError, match="Unhandled statement type"):
            resolve(_program((ast.S(statement=_FakeStmt()),)))


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
# Full program resolution
# ---------------------------------------------------------------------------


class TestResolveProgram:
    """``resolve`` correctly renames variables and preserves program structure."""

    def test_declaration_name_is_uniquified(self) -> None:
        """A declared variable's name is replaced by its unique internal identifier."""
        body = (_vd("x"),)
        resolved = _resolved_body(body)
        assert resolved[0] == ast.D(ast.VarDecl(ast.VariableDeclaration(".Vx.0", None)))

    def test_declaration_and_use_renamed_consistently(self) -> None:
        """The same unique name is used in both the declaration and its subsequent reference."""
        body = (
            _vd("x", ast.Constant(value=5)),
            ast.S(statement=ast.Return(exp=ast.Var(name="x"))),
        )
        resolved = _resolved_body(body)
        decl = resolved[0]
        ret = resolved[1]
        assert isinstance(decl, ast.D)
        assert isinstance(ret, ast.S)
        assert decl.declaration.variable_declaration.name == ".Vx.0"
        assert ret.statement == ast.Return(exp=ast.Var(name=".Vx.0"))

    def test_assignment_expression_in_statement_resolves(self) -> None:
        """``x = 10;`` after ``int x;`` resolves the lvalue to the unique name."""
        body = (
            _vd("x"),
            ast.S(
                statement=ast.Expression(
                    exp=ast.Assignment(
                        lvalue=ast.Var(name="x"),
                        rvalue=ast.Constant(value=10),
                    )
                )
            ),
        )
        resolved = _resolved_body(body)
        assert resolved[1] == ast.S(
            statement=ast.Expression(
                exp=ast.Assignment(
                    lvalue=ast.Var(name=".Vx.0"),
                    rvalue=ast.Constant(value=10),
                )
            )
        )

    def test_return_constant_requires_no_resolution(self) -> None:
        """``return 42;`` contains no variables and passes through unchanged."""
        body = (ast.S(statement=ast.Return(exp=ast.Constant(value=42))),)
        resolved = _resolved_body(body)
        assert resolved[0] == ast.S(statement=ast.Return(exp=ast.Constant(value=42)))

    def test_duplicate_declaration_raises(self) -> None:
        """Two declarations of the same name in a function body raise ``SemanticError``."""
        body = (_vd("x"), _vd("x"))
        with pytest.raises(SemanticError, match="Duplicate"):
            resolve(_program(body))

    def test_undeclared_variable_in_return_raises(self) -> None:
        """Returning an undeclared variable raises ``SemanticError``."""
        body = (ast.S(statement=ast.Return(exp=ast.Var(name="x"))),)
        with pytest.raises(SemanticError, match="Undeclared"):
            resolve(_program(body))

    def test_function_name_and_params_preserved(self) -> None:
        """Resolution does not alter the function's name or params."""
        body = (ast.S(statement=ast.Return(exp=ast.Constant(value=0))),)
        resolved = resolve(_program(body))
        fn = _get_fn(resolved)
        assert fn.name == "main"
        assert fn.params == ()

    def test_variable_declarations_not_visible_across_functions(self) -> None:
        """A variable declared in one function is not visible in a subsequent function.

        ``_resolve_function`` resets the variable map to an empty scope before
        processing each function; declarations from ``fn1`` must not leak into ``fn2``.
        """
        fn1 = ast.Function(
            name="fn1",
            params=(),
            body=(
                _vd("x", ast.Constant(value=0)),
                ast.S(statement=ast.Return(exp=ast.Var(name="x"))),
            ),
        )
        fn2 = ast.Function(
            name="fn2",
            params=(),
            body=(ast.S(statement=ast.Return(exp=ast.Var(name="x"))),),
        )
        with pytest.raises(SemanticError, match="Undeclared"):
            resolve(_make_prog(fn1, fn2))


# ---------------------------------------------------------------------------
# Counter isolation between analyser instances
# ---------------------------------------------------------------------------


class TestCounterIsolation:
    """Each ``SemanticAnalyser`` instance starts its counter from zero."""

    def test_two_instances_both_produce_zero_suffix(self) -> None:
        """Independent instances both produce ``.Vx.0`` for their first declaration of ``x``."""
        r1 = SemanticAnalyser().resolve_variable_declaration(ast.VariableDeclaration(name="x", init=None))
        r2 = SemanticAnalyser().resolve_variable_declaration(ast.VariableDeclaration(name="x", init=None))
        assert r1.name == ".Vx.0"
        assert r2.name == ".Vx.0"

    def test_second_call_on_same_instance_increments(self) -> None:
        """A single instance increments its counter across consecutive declarations."""
        analyser = SemanticAnalyser()
        r1 = analyser.resolve_variable_declaration(ast.VariableDeclaration(name="a", init=None))
        r2 = analyser.resolve_variable_declaration(ast.VariableDeclaration(name="b", init=None))
        assert r1.name == ".Va.0"
        assert r2.name == ".Vb.1"


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
# Compound statement (block scoping)
# ---------------------------------------------------------------------------


class TestResolveCompound:
    """``Compound`` statements introduce a new scope for variable declarations."""

    def test_empty_compound_block_resolves(self) -> None:
        """An empty compound block resolves to an empty ``Compound`` node."""
        body = (ast.S(statement=ast.Compound(body=())),)
        resolved = _resolved_body(body)
        assert resolved[0] == ast.S(statement=ast.Compound(body=()))

    def test_inner_declaration_gets_unique_name(self) -> None:
        """A variable declared inside a compound block receives a unique internal name."""
        body = (
            ast.S(
                statement=ast.Compound(
                    body=(
                        _vd("x"),
                        ast.S(statement=ast.Return(exp=ast.Var(name="x"))),
                    )
                )
            ),
        )
        resolved = _resolved_body(body)
        compound = resolved[0].statement
        assert isinstance(compound, ast.Compound)
        decl = compound.body[0]
        ret = compound.body[1]
        assert isinstance(decl, ast.D)
        assert decl.declaration.variable_declaration.name == ".Vx.0"
        assert ret == ast.S(statement=ast.Return(exp=ast.Var(name=".Vx.0")))

    def test_outer_variable_visible_inside_compound_block(self) -> None:
        """A variable declared in the outer scope is visible inside a nested block."""
        body = (
            _vd("x", ast.Constant(value=1)),
            ast.S(statement=ast.Compound(body=(ast.S(statement=ast.Return(exp=ast.Var(name="x"))),))),
        )
        resolved = _resolved_body(body)
        compound = resolved[1].statement
        assert isinstance(compound, ast.Compound)
        assert compound.body[0] == ast.S(statement=ast.Return(exp=ast.Var(name=".Vx.0")))

    def test_shadowing_inner_declaration_permitted(self) -> None:
        """Declaring a variable with the same name as an outer variable is allowed (shadowing)."""
        body = (
            _vd("x", ast.Constant(value=1)),
            ast.S(
                statement=ast.Compound(
                    body=(
                        _vd("x", ast.Constant(value=2)),
                        ast.S(statement=ast.Return(exp=ast.Var(name="x"))),
                    )
                )
            ),
        )
        resolved = _resolved_body(body)
        outer_decl = resolved[0]
        assert isinstance(outer_decl, ast.D)
        outer_name = outer_decl.declaration.variable_declaration.name  # e.g. ".Vx.0"
        compound = resolved[1].statement
        assert isinstance(compound, ast.Compound)
        inner_decl = compound.body[0]
        assert isinstance(inner_decl, ast.D)
        inner_name = inner_decl.declaration.variable_declaration.name  # e.g. ".Vx.1"
        assert inner_name != outer_name
        # The return inside the block uses the inner (shadow) binding.
        assert compound.body[1] == ast.S(statement=ast.Return(exp=ast.Var(name=inner_name)))

    def test_outer_binding_restored_after_compound_block(self) -> None:
        """After a compound block that shadows a name, the outer binding is visible again."""
        body = (
            _vd("x", ast.Constant(value=1)),
            ast.S(statement=ast.Compound(body=(_vd("x", ast.Constant(value=2)),))),
            ast.S(statement=ast.Return(exp=ast.Var(name="x"))),
        )
        resolved = _resolved_body(body)
        outer_name = resolved[0].declaration.variable_declaration.name
        # The return after the compound block must use the outer binding.
        assert resolved[2] == ast.S(statement=ast.Return(exp=ast.Var(name=outer_name)))

    def test_duplicate_in_same_inner_block_raises(self) -> None:
        """Two declarations of the same name within one compound block raise ``SemanticError``."""
        body = (ast.S(statement=ast.Compound(body=(_vd("x"), _vd("x")))),)
        with pytest.raises(SemanticError, match="Duplicate"):
            resolve(_program(body))

    def test_inner_variable_not_visible_after_block(self) -> None:
        """A variable declared inside a compound block is out of scope after it closes."""
        body = (
            ast.S(statement=ast.Compound(body=(_vd("x"),))),
            ast.S(statement=ast.Return(exp=ast.Var(name="x"))),
        )
        with pytest.raises(SemanticError, match="Undeclared"):
            resolve(_program(body))

    def test_label_inside_compound_block_is_reachable_by_goto(self) -> None:
        """A label nested inside a compound block is still collected and validatable by ``goto``."""
        body = (
            ast.S(statement=ast.Goto(target="inner_label")),
            ast.S(
                statement=ast.Compound(
                    body=(
                        ast.S(
                            statement=ast.Label(
                                name="inner_label",
                                statement=ast.Return(exp=ast.Constant(value=0)),
                            )
                        ),
                    )
                )
            ),
        )
        resolved = _resolved_body(body)
        assert resolved[0] == ast.S(statement=ast.Goto(target="inner_label"))

    def test_duplicate_label_inside_compound_block_raises(self) -> None:
        """A label declared twice (once inside a compound block, once outside) raises ``SemanticError``."""
        body = (
            ast.S(statement=ast.Label(name="foo", statement=ast.Null())),
            ast.S(statement=ast.Compound(body=(ast.S(statement=ast.Label(name="foo", statement=ast.Null())),))),
        )
        with pytest.raises(SemanticError, match="Duplicate label"):
            resolve(_program(body))

    def test_sibling_blocks_may_reuse_same_variable_name(self) -> None:
        """Two sequential compound blocks may both declare a variable with the same name.

        After the first block exits the name is no longer in scope; the second
        block starts a fresh scope of its own and is free to declare the same
        source-level name without conflict.
        """
        body = (
            ast.S(statement=ast.Compound(body=(_vd("x"),))),
            ast.S(statement=ast.Compound(body=(_vd("x"),))),
            ast.S(statement=ast.Return(exp=ast.Constant(value=0))),
        )
        resolved = _resolved_body(body)
        first_name = resolved[0].statement.body[0].declaration.variable_declaration.name
        second_name = resolved[1].statement.body[0].declaration.variable_declaration.name
        # Both resolved successfully and received distinct unique names.
        assert first_name != second_name

    def test_counter_not_reset_between_compound_blocks(self) -> None:
        """The unique-name counter increments monotonically across compound block boundaries.

        Entering or exiting a compound block must not reset the counter; the
        three declarations ``a``, ``b``, and ``c`` must receive consecutive
        suffixes regardless of which block each lives in.
        """
        body = (
            _vd("a"),
            ast.S(statement=ast.Compound(body=(_vd("b"),))),
            _vd("c"),
            ast.S(statement=ast.Return(exp=ast.Constant(value=0))),
        )
        resolved = _resolved_body(body)
        a_name = resolved[0].declaration.variable_declaration.name
        b_name = resolved[1].statement.body[0].declaration.variable_declaration.name
        c_name = resolved[2].declaration.variable_declaration.name
        assert a_name == ".Va.0"
        assert b_name == ".Vb.1"
        assert c_name == ".Vc.2"

    def test_assignment_lvalue_to_outer_variable_inside_block(self) -> None:
        """An assignment whose lvalue is a variable from an outer scope resolves correctly.

        ``test_outer_variable_visible_inside_compound_block`` covers reading;
        this test covers the lvalue (write) path through ``resolve_exp``.
        """
        body = (
            _vd("x", ast.Constant(value=0)),
            ast.S(
                statement=ast.Compound(
                    body=(
                        ast.S(
                            statement=ast.Expression(
                                exp=ast.Assignment(
                                    lvalue=ast.Var(name="x"),
                                    rvalue=ast.Constant(value=5),
                                )
                            )
                        ),
                    )
                )
            ),
        )
        resolved = _resolved_body(body)
        outer_name = resolved[0].declaration.variable_declaration.name
        inner_stmt = resolved[1].statement.body[0]
        assert inner_stmt == ast.S(
            statement=ast.Expression(
                exp=ast.Assignment(
                    lvalue=ast.Var(name=outer_name),
                    rvalue=ast.Constant(value=5),
                )
            )
        )

    def test_shadow_initialiser_resolves_to_new_binding(self) -> None:
        """A shadow declaration's initialiser sees the new (inner) binding, not the outer one.

        C11 §6.2.1 places a block-scope identifier in scope immediately after
        its declarator, before any ``=`` initialiser.  ``int x = x;`` in an
        inner scope where an outer ``x`` exists therefore self-initialises from
        the freshly created inner binding rather than from the outer one.
        """
        body = (
            _vd("x", ast.Constant(value=1)),
            ast.S(statement=ast.Compound(body=(_vd("x", ast.Var(name="x")),))),
        )
        resolved = _resolved_body(body)
        outer_name = resolved[0].declaration.variable_declaration.name
        inner_decl = resolved[1].statement.body[0].declaration.variable_declaration
        # The inner x must be distinct from the outer x.
        assert inner_decl.name != outer_name
        # Its initialiser references itself (the new inner binding), not the outer x.
        assert inner_decl.init == ast.Var(name=inner_decl.name)

    def test_undeclared_variable_inside_compound_block_raises(self) -> None:
        """A reference to a variable that was never declared, inside a compound block, raises ``SemanticError``."""
        body = (ast.S(statement=ast.Compound(body=(ast.S(statement=ast.Return(exp=ast.Var(name="never_declared"))),))),)
        with pytest.raises(SemanticError, match="Undeclared"):
            resolve(_program(body))

    def test_compound_as_then_branch_of_if_resolves(self) -> None:
        """A compound block used as the then-branch of an ``if`` is fully resolved."""
        body = (
            _vd("x", ast.Constant(value=1)),
            ast.S(
                statement=ast.If(
                    condition=ast.Var(name="x"),
                    then=ast.Compound(
                        body=(
                            _vd("y", ast.Var(name="x")),
                            ast.S(statement=ast.Return(exp=ast.Var(name="y"))),
                        )
                    ),
                    else_=None,
                )
            ),
        )
        resolved = _resolved_body(body)
        outer_name = resolved[0].declaration.variable_declaration.name
        if_stmt = resolved[1].statement
        assert isinstance(if_stmt, ast.If)
        assert if_stmt.condition == ast.Var(name=outer_name)
        compound = if_stmt.then
        assert isinstance(compound, ast.Compound)
        inner_decl = compound.body[0].declaration.variable_declaration
        assert inner_decl.init == ast.Var(name=outer_name)
        assert compound.body[1] == ast.S(statement=ast.Return(exp=ast.Var(name=inner_decl.name)))

    def test_compound_as_else_branch_of_if_resolves(self) -> None:
        """A compound block used as the else-branch of an ``if`` is fully resolved."""
        body = (
            _vd("x", ast.Constant(value=1)),
            ast.S(
                statement=ast.If(
                    condition=ast.Constant(value=0),
                    then=ast.Null(),
                    else_=ast.Compound(body=(ast.S(statement=ast.Return(exp=ast.Var(name="x"))),)),
                )
            ),
        )
        resolved = _resolved_body(body)
        outer_name = resolved[0].declaration.variable_declaration.name
        if_stmt = resolved[1].statement
        assert isinstance(if_stmt, ast.If)
        compound = if_stmt.else_
        assert isinstance(compound, ast.Compound)
        assert compound.body[0] == ast.S(statement=ast.Return(exp=ast.Var(name=outer_name)))

    def test_unknown_block_item_type_inside_compound_raises(self) -> None:
        """An unrecognised ``BlockItem`` subclass inside a compound block raises ``SemanticError``.

        The ``_`` branch of ``_resolve_block_item`` must fire for any subclass
        that is neither ``ast.S`` nor ``ast.D``.
        """

        class _FakeItem(ast.BlockItem):
            pass

        body = (ast.S(statement=ast.Compound(body=(_FakeItem(),))),)
        with pytest.raises(SemanticError, match="Unhandled block item type"):
            resolve(_program(body))

    def test_goto_inside_compound_targets_outer_label(self) -> None:
        """A ``goto`` inside a compound block may target a label declared outside the block.

        Labels are collected in a function-wide first pass, so a label at the
        top level of a function is visible to a ``goto`` nested inside any
        compound block within that function.
        """
        body = (
            ast.S(statement=ast.Compound(body=(ast.S(statement=ast.Goto(target="end")),))),
            ast.S(statement=ast.Label(name="end", statement=ast.Return(exp=ast.Constant(value=0)))),
        )
        resolved = _resolved_body(body)
        inner_goto = resolved[0].statement.body[0]
        assert inner_goto == ast.S(statement=ast.Goto(target="end"))

    def test_label_wrapping_compound_block_resolves(self) -> None:
        """A ``Label`` whose inner statement is a ``Compound`` block is fully resolved.

        Tests the ``Label → _resolve_statement(Compound) → _resolve_block`` path
        and verifies that variable references inside the compound are resolved
        using the enclosing scope.
        """
        body = (
            _vd("x", ast.Constant(value=1)),
            ast.S(
                statement=ast.Label(
                    name="lbl",
                    statement=ast.Compound(body=(ast.S(statement=ast.Return(exp=ast.Var(name="x"))),)),
                )
            ),
        )
        resolved = _resolved_body(body)
        outer_name = resolved[0].declaration.variable_declaration.name
        label_stmt = resolved[1].statement
        assert isinstance(label_stmt, ast.Label)
        assert label_stmt.name == "lbl"
        compound = label_stmt.statement
        assert isinstance(compound, ast.Compound)
        assert compound.body[0] == ast.S(statement=ast.Return(exp=ast.Var(name=outer_name)))

    def test_deeply_nested_compound_blocks(self) -> None:
        """Variables at each nesting level receive distinct unique names and the correct binding is used."""
        body = (
            _vd("x", ast.Constant(value=1)),
            ast.S(
                statement=ast.Compound(
                    body=(
                        _vd("x", ast.Constant(value=2)),
                        ast.S(
                            statement=ast.Compound(
                                body=(
                                    _vd("x", ast.Constant(value=3)),
                                    ast.S(statement=ast.Return(exp=ast.Var(name="x"))),
                                )
                            )
                        ),
                    )
                )
            ),
        )
        resolved = _resolved_body(body)
        outer_name = resolved[0].declaration.variable_declaration.name
        mid_compound = resolved[1].statement
        assert isinstance(mid_compound, ast.Compound)
        mid_name = mid_compound.body[0].declaration.variable_declaration.name
        inner_compound = mid_compound.body[1].statement
        assert isinstance(inner_compound, ast.Compound)
        inner_name = inner_compound.body[0].declaration.variable_declaration.name
        # All three must be distinct.
        assert len({outer_name, mid_name, inner_name}) == 3
        # The return inside the innermost block uses the innermost binding.
        assert inner_compound.body[1] == ast.S(statement=ast.Return(exp=ast.Var(name=inner_name)))


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


# ---------------------------------------------------------------------------
# Function declaration and definition validation
# ---------------------------------------------------------------------------


class TestFunctionDeclarations:
    """Function declarations and definitions are tracked in the program-wide function map."""

    def _fn(self, name: str, body: tuple[ast.BlockItem, ...] | None = None) -> ast.Function:
        """Build a ``Function`` node with the given name and optional body.

        Args:
            name: The function name.
            body: The function body block items, or ``None`` for a declaration.

        Returns:
            An ``ast.Function`` with ``params=()`` and the given body.
        """
        return ast.Function(name=name, params=(), body=body)

    def test_top_level_declaration_only_is_valid(self) -> None:
        """A single top-level function declaration (no body) resolves without error."""
        program = _make_prog(self._fn("foo", None), self._fn("main", (ast.S(ast.Return(ast.Constant(0))),)))
        resolved = resolve(program)
        assert len(resolved.declarations) == 2

    def test_top_level_definition_is_valid(self) -> None:
        """A top-level function definition with a body resolves without error."""
        program = _make_prog(self._fn("main", (ast.S(ast.Return(ast.Constant(0))),)))
        resolved = resolve(program)
        assert _get_fn(resolved).name == "main"

    def test_declaration_followed_by_definition_is_valid(self) -> None:
        """A forward declaration of a function followed by its definition is accepted."""
        program = _make_prog(
            self._fn("foo", None),
            self._fn("foo", (ast.S(ast.Return(ast.Constant(0))),)),
        )
        resolved = resolve(program)
        assert len(resolved.declarations) == 2

    def test_multiple_declarations_of_same_function_are_valid(self) -> None:
        """Multiple forward declarations of the same function name are permitted."""
        program = _make_prog(
            self._fn("foo", None),
            self._fn("foo", None),
            self._fn("main", (ast.S(ast.Return(ast.Constant(0))),)),
        )
        resolved = resolve(program)
        assert len(resolved.declarations) == 3

    def test_duplicate_definition_raises(self) -> None:
        """Two definitions of the same function at the top level raise ``SemanticError``."""
        program = _make_prog(
            self._fn("foo", (ast.S(ast.Return(ast.Constant(0))),)),
            self._fn("foo", (ast.S(ast.Return(ast.Constant(1))),)),
        )
        with pytest.raises(SemanticError, match="Redefinition"):
            resolve(program)

    def test_block_level_function_declaration_is_valid(self) -> None:
        """A function declaration appearing as a block item inside another function is accepted."""
        foo_decl = ast.D(ast.FunDecl(ast.Function(name="foo", params=(), body=None)))
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            ast.S(ast.Return(ast.Constant(0))),
        )
        resolved = resolve(_program(body))
        assert _get_fn(resolved).body[0] == foo_decl

    def test_repeated_block_level_declarations_of_same_function_are_valid(self) -> None:
        """Multiple declarations of the same function within a single block are all accepted.

        This corresponds to the user's example::

            int main(void) {
                int foo(void);
                int foo(void);
                return foo();
            }
        """
        foo_decl = ast.D(ast.FunDecl(ast.Function(name="foo", params=(), body=None)))
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            foo_decl,
            ast.S(ast.Return(ast.FunctionCall("foo", []))),
        )
        resolved = resolve(_program(body))
        assert len(_get_fn(resolved).body) == 3

    def test_call_to_declared_function_is_valid(self) -> None:
        """A ``FunctionCall`` to a function that has been declared resolves without error."""
        foo_decl = ast.D(ast.FunDecl(ast.Function(name="foo", params=(), body=None)))
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            ast.S(ast.Return(ast.FunctionCall("foo", []))),
        )
        resolved = resolve(_program(body))
        return_stmt = _get_fn(resolved).body[1].statement
        assert isinstance(return_stmt, ast.Return)
        assert return_stmt.exp == ast.FunctionCall("foo", [])

    def test_call_to_top_level_defined_function_declared_earlier_is_valid(self) -> None:
        """A ``FunctionCall`` to a function defined earlier in source order resolves without error."""
        foo = ast.Function(
            name="foo",
            params=(),
            body=(ast.S(ast.Return(ast.Constant(0))),),
        )
        bar = ast.Function(
            name="bar",
            params=(),
            body=(ast.S(ast.Return(ast.FunctionCall("foo", []))),),
        )
        program = _make_prog(foo, bar)
        resolved = resolve(program)
        return_stmt = _get_fn(resolved, 1).body[0].statement
        assert isinstance(return_stmt, ast.Return)
        assert return_stmt.exp == ast.FunctionCall("foo", [])

    def test_call_to_top_level_function_defined_later_raises(self) -> None:
        """A ``FunctionCall`` to a function whose declaration has not yet appeared raises ``SemanticError``.

        Functions must be declared or defined before use; there is no implicit
        forward declaration in C.
        """
        bar = ast.Function(
            name="bar",
            params=(),
            body=(ast.S(ast.Return(ast.FunctionCall("foo", []))),),
        )
        foo = ast.Function(
            name="foo",
            params=(),
            body=(ast.S(ast.Return(ast.Constant(0))),),
        )
        program = _make_prog(bar, foo)
        with pytest.raises(SemanticError, match="undeclared function"):
            resolve(program)

    def test_call_to_undeclared_function_raises(self) -> None:
        """A ``FunctionCall`` to a name not in the function map raises ``SemanticError``."""
        body: tuple[ast.BlockItem, ...] = (ast.S(ast.Return(ast.FunctionCall("ghost", []))),)
        with pytest.raises(SemanticError, match="undeclared function"):
            resolve(_program(body))

    def test_function_call_args_are_resolved(self) -> None:
        """Arguments of a ``FunctionCall`` have their variable references renamed."""
        foo_decl = ast.D(
            ast.FunDecl(
                ast.Function(
                    name="foo",
                    params=(ast.Parameter(param_type=ast.IntType(), name="a"),),
                    body=None,
                )
            )
        )
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            _vd("x", ast.Constant(1)),
            ast.S(ast.Return(ast.FunctionCall("foo", [ast.Var("x")]))),
        )
        resolved = resolve(_program(body))
        return_stmt = _get_fn(resolved).body[2].statement
        assert isinstance(return_stmt, ast.Return)
        call = return_stmt.exp
        assert isinstance(call, ast.FunctionCall)
        assert call.args == [ast.Var(".Vx.0")]


# ---------------------------------------------------------------------------
# Function parameters
# ---------------------------------------------------------------------------


class TestFunctionParameters:
    """Function parameters are renamed with unique identifiers like local variables."""

    def test_parameter_receives_unique_name(self) -> None:
        """A function parameter is given a unique internal name in the resolved AST."""
        fn = ast.Function(
            name="foo",
            params=(ast.Parameter(param_type=ast.IntType(), name="x"),),
            body=(ast.S(ast.Return(ast.Var("x"))),),
        )
        resolved = resolve(_make_prog(fn))
        param = _get_fn(resolved).params[0]
        assert param.name.startswith(".Vx.")

    def test_parameter_reference_in_body_uses_unique_name(self) -> None:
        """The body of a function refers to renamed parameters via unique internal names."""
        fn = ast.Function(
            name="foo",
            params=(ast.Parameter(param_type=ast.IntType(), name="x"),),
            body=(ast.S(ast.Return(ast.Var("x"))),),
        )
        resolved = resolve(_make_prog(fn))
        param_name = _get_fn(resolved).params[0].name
        return_stmt = _get_fn(resolved).body[0].statement
        assert isinstance(return_stmt, ast.Return)
        assert return_stmt.exp == ast.Var(param_name)

    def test_duplicate_parameter_names_raise(self) -> None:
        """Two parameters with the same name in a single function raise ``SemanticError``."""
        fn = ast.Function(
            name="foo",
            params=(
                ast.Parameter(param_type=ast.IntType(), name="x"),
                ast.Parameter(param_type=ast.IntType(), name="x"),
            ),
            body=(ast.S(ast.Return(ast.Var("x"))),),
        )
        with pytest.raises(SemanticError, match="Duplicate"):
            resolve(_make_prog(fn))

    def test_declaration_body_is_none_skips_resolution(self) -> None:
        """A ``Function`` node with ``body=None`` is returned unchanged — no resolution needed."""
        fn = ast.Function(name="foo", params=(), body=None)
        resolved = resolve(_make_prog(fn))
        assert _get_fn(resolved) is fn

    def test_duplicate_params_in_top_level_declaration_raises(self) -> None:
        """Duplicate parameter names in a top-level forward declaration raise ``SemanticError``.

        Corresponds to ``int foo(int a, int a);`` at file scope.
        """
        foo_decl = ast.Function(
            name="foo",
            params=(
                ast.Parameter(param_type=ast.IntType(), name="a"),
                ast.Parameter(param_type=ast.IntType(), name="a"),
            ),
            body=None,
        )
        main_def = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(ast.Constant(0))),),
        )
        with pytest.raises(SemanticError, match="Duplicate parameter name 'a'"):
            resolve(_make_prog(foo_decl, main_def))

    def test_duplicate_params_in_block_level_declaration_raises(self) -> None:
        """Duplicate parameter names in a block-scope function declaration raise ``SemanticError``.

        Corresponds to ``int foo(int a, int a);`` appearing inside a function body.
        """
        foo_decl = ast.D(
            ast.FunDecl(
                ast.Function(
                    name="foo",
                    params=(
                        ast.Parameter(param_type=ast.IntType(), name="a"),
                        ast.Parameter(param_type=ast.IntType(), name="a"),
                    ),
                    body=None,
                )
            )
        )
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            ast.S(ast.Return(ast.Constant(0))),
        )
        with pytest.raises(SemanticError, match="Duplicate parameter name 'a'"):
            resolve(_program(body))

    def test_distinct_params_in_top_level_declaration_are_valid(self) -> None:
        """A forward declaration whose parameter names are all distinct resolves without error."""
        foo_decl = ast.Function(
            name="foo",
            params=(
                ast.Parameter(param_type=ast.IntType(), name="a"),
                ast.Parameter(param_type=ast.IntType(), name="b"),
            ),
            body=None,
        )
        main_def = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(ast.Constant(0))),),
        )
        resolved = resolve(_make_prog(foo_decl, main_def))
        assert len(resolved.declarations) == 2

    def test_same_param_name_in_declaration_and_definition_is_valid(self) -> None:
        """The same parameter name may be reused between a declaration and its matching definition."""
        foo_decl = ast.Function(
            name="foo",
            params=(ast.Parameter(param_type=ast.IntType(), name="x"),),
            body=None,
        )
        foo_def = ast.Function(
            name="foo",
            params=(ast.Parameter(param_type=ast.IntType(), name="x"),),
            body=(ast.S(ast.Return(ast.Var("x"))),),
        )
        resolved = resolve(_make_prog(foo_decl, foo_def))
        assert len(resolved.declarations) == 2

    def test_nested_function_definition_raises(self) -> None:
        """A function definition appearing as a block item inside another function raises ``SemanticError``.

        C forbids nested function definitions; only forward declarations (no body)
        may appear at block scope.
        """
        nested_def = ast.D(
            ast.FunDecl(
                ast.Function(
                    name="inner",
                    params=(),
                    body=(ast.S(ast.Return(ast.Constant(0))),),
                )
            )
        )
        body: tuple[ast.BlockItem, ...] = (
            nested_def,
            ast.S(ast.Return(ast.Constant(0))),
        )
        with pytest.raises(SemanticError, match="Nested function definition"):
            resolve(_program(body))

    def test_nested_function_definition_inside_compound_raises(self) -> None:
        """A nested definition buried inside a compound block also raises ``SemanticError``."""
        nested_def = ast.D(
            ast.FunDecl(
                ast.Function(
                    name="inner",
                    params=(),
                    body=(ast.S(ast.Return(ast.Constant(0))),),
                )
            )
        )
        body: tuple[ast.BlockItem, ...] = (
            ast.S(
                statement=ast.Compound(
                    body=(
                        nested_def,
                        ast.S(ast.Return(ast.Constant(0))),
                    )
                )
            ),
        )
        with pytest.raises(SemanticError, match="Nested function definition"):
            resolve(_program(body))

    def test_block_scope_function_declaration_no_body_is_valid(self) -> None:
        """A forward declaration (no body) at block scope is still permitted."""
        foo_decl = ast.D(ast.FunDecl(ast.Function(name="foo", params=(), body=None)))
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            ast.S(ast.Return(ast.Constant(0))),
        )
        resolved = resolve(_program(body))
        assert _get_fn(resolved).body[0] == foo_decl

    def test_block_scope_static_function_declaration_raises(self) -> None:
        """A block-scope function declaration with ``static`` storage class raises ``SemanticError``."""
        foo_decl = ast.D(
            ast.FunDecl(ast.Function(name="foo", params=(), body=None, storage_class=ast.StorageClass.Static))
        )
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            ast.S(ast.Return(ast.FunctionCall("foo", []))),
        )
        with pytest.raises(SemanticError, match="static"):
            resolve(_program(body))


# ---------------------------------------------------------------------------
# Cross-kind redeclaration: function ↔ variable in the same scope
# ---------------------------------------------------------------------------


class TestCrossKindRedeclaration:
    """Declaring the same name as both a function and a variable in one block scope is an error."""

    def test_function_then_variable_same_scope_raises(self) -> None:
        """A variable declaration following a function declaration of the same name raises ``SemanticError``."""
        foo_decl = ast.D(ast.FunDecl(ast.Function(name="foo", params=(), body=None)))
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            _vd("foo"),
            ast.S(ast.Return(ast.Constant(0))),
        )
        with pytest.raises(SemanticError, match="'foo' redeclared as a variable"):
            resolve(_program(body))

    def test_variable_then_function_same_scope_raises(self) -> None:
        """A function declaration following a variable declaration of the same name raises ``SemanticError``."""
        foo_decl = ast.D(ast.FunDecl(ast.Function(name="foo", params=(), body=None)))
        body: tuple[ast.BlockItem, ...] = (
            _vd("foo"),
            foo_decl,
            ast.S(ast.Return(ast.Constant(0))),
        )
        with pytest.raises(SemanticError, match="'foo' redeclared as a function"):
            resolve(_program(body))

    def test_top_level_function_shadowed_by_local_variable_is_valid(self) -> None:
        """A local variable may shadow a top-level function name without error.

        The top-level function is in ``_function_map`` but is not in
        ``_current_block_funs`` for the inner function body (that set is reset
        at the start of each function definition), so no conflict is raised.
        """
        foo_top = ast.Function(name="foo", params=(), body=None)
        main_body: tuple[ast.BlockItem, ...] = (
            _vd("foo", ast.Constant(0)),
            ast.S(ast.Return(ast.Constant(0))),
        )
        main_fn = ast.Function(name="main", params=(), body=main_body)
        resolved = resolve(_make_prog(foo_top, main_fn))
        assert len(resolved.declarations) == 2

    def test_function_in_outer_scope_variable_in_inner_scope_is_valid(self) -> None:
        """A variable in an inner block may shadow a function declared in an outer block.

        The outer block's ``_current_block_funs`` is saved and cleared when the
        inner block is entered, so the inner variable declaration sees no conflict.
        """
        foo_decl = ast.D(ast.FunDecl(ast.Function(name="foo", params=(), body=None)))
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            ast.S(
                statement=ast.Compound(
                    body=(
                        _vd("foo", ast.Constant(0)),
                        ast.S(ast.Return(ast.Constant(0))),
                    )
                )
            ),
        )
        resolved = resolve(_program(body))
        assert len(_get_fn(resolved).body) == 2

    def test_variable_in_outer_scope_function_in_inner_scope_is_valid(self) -> None:
        """A function declaration in an inner block is independent of a variable in the outer block."""
        foo_decl = ast.D(ast.FunDecl(ast.Function(name="foo", params=(), body=None)))
        body: tuple[ast.BlockItem, ...] = (
            _vd("foo", ast.Constant(0)),
            ast.S(
                statement=ast.Compound(
                    body=(
                        foo_decl,
                        ast.S(ast.Return(ast.Constant(0))),
                    )
                )
            ),
        )
        resolved = resolve(_program(body))
        assert len(_get_fn(resolved).body) == 2

    def test_extern_variable_conflicts_with_prior_file_scope_function_raises(self) -> None:
        """A block-scope ``extern`` variable whose name was previously declared as a function raises ``SemanticError``.

        Both refer to the same external entity; declaring it as both a function and a
        variable is a conflict.
        """
        foo_fn = ast.Function(name="foo", params=(), body=(ast.S(ast.Return(ast.Constant(0))),))
        main_body: tuple[ast.BlockItem, ...] = (
            ast.D(ast.VarDecl(ast.VariableDeclaration("foo", None, ast.StorageClass.Extern))),
            ast.S(ast.Return(ast.Constant(0))),
        )
        main_fn = ast.Function(name="main", params=(), body=main_body)
        with pytest.raises(SemanticError, match="redeclared as a variable"):
            resolve(_make_prog(foo_fn, main_fn))

    def test_function_declaration_conflicts_with_prior_extern_variable_raises(self) -> None:
        """A block-scope function declaration whose name was declared as ``extern`` in an outer scope raises.

        An ``extern`` variable binding has external linkage; a function of the same
        name in an inner scope would refer to the same external entity but with a
        conflicting type, which is an error.
        """
        body: tuple[ast.BlockItem, ...] = (
            ast.D(ast.VarDecl(ast.VariableDeclaration("foo", None, ast.StorageClass.Extern))),
            ast.S(
                statement=ast.Compound(
                    body=(
                        ast.D(ast.FunDecl(ast.Function(name="foo", params=(), body=None))),
                        ast.S(ast.Return(ast.Constant(0))),
                    )
                )
            ),
        )
        with pytest.raises(SemanticError, match="redeclared as a function"):
            resolve(_program(body))


# ---------------------------------------------------------------------------
# Type checking: function/variable name misuse
# ---------------------------------------------------------------------------


class TestTypeChecking:
    """The analyser rejects function names used as variables and variables called as functions."""

    def test_function_name_used_as_variable_raises(self) -> None:
        """Referencing a function name in a ``Var`` expression raises ``SemanticError``.

        ``foo`` is declared as a function; using it bare (without a call) in a
        ``return`` expression is a type error.
        """
        foo_decl = ast.D(ast.FunDecl(ast.Function(name="foo", params=(), body=None)))
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            ast.S(ast.Return(ast.Var("foo"))),
        )
        with pytest.raises(SemanticError, match="Function name 'foo' used as a variable"):
            resolve(_program(body))

    def test_variable_called_as_function_raises(self) -> None:
        """Calling a local variable as though it were a function raises ``SemanticError``.

        ``x`` is declared as an ``int``; ``x()`` is a type error.
        """
        body: tuple[ast.BlockItem, ...] = (
            _vd("x", ast.Constant(1)),
            ast.S(ast.Return(ast.FunctionCall("x", []))),
        )
        with pytest.raises(SemanticError, match="Variable 'x' called as a function"):
            resolve(_program(body))

    def test_function_name_in_binary_expression_raises(self) -> None:
        """A function name appearing in an arithmetic expression raises ``SemanticError``."""
        foo_decl = ast.D(ast.FunDecl(ast.Function(name="foo", params=(), body=None)))
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            ast.S(ast.Return(ast.Binary(ast.Add(), ast.Var("foo"), ast.Constant(1)))),
        )
        with pytest.raises(SemanticError, match="Function name 'foo' used as a variable"):
            resolve(_program(body))

    def test_parameter_called_as_function_raises(self) -> None:
        """A function parameter called as a function raises ``SemanticError``.

        ``p`` is a parameter of type ``int``; ``p()`` is a type error.
        """
        fn = ast.Function(
            name="foo",
            params=(ast.Parameter(param_type=ast.IntType(), name="p"),),
            body=(ast.S(ast.Return(ast.FunctionCall("p", []))),),
        )
        with pytest.raises(SemanticError, match="Variable 'p' called as a function"):
            resolve(_make_prog(fn))

    def test_variable_shadowing_function_name_is_usable_as_variable(self) -> None:
        """A local variable may shadow a function name; the variable is then accessible as a ``Var``.

        Inside the block, ``foo`` refers to the local ``int`` variable.
        After the block exits the function-name binding is visible again.
        """
        foo_decl = ast.D(ast.FunDecl(ast.Function(name="foo", params=(), body=None)))
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            ast.S(
                statement=ast.Compound(
                    body=(
                        _vd("foo", ast.Constant(42)),
                        ast.S(ast.Return(ast.Var("foo"))),
                    )
                )
            ),
        )
        resolved = resolve(_program(body))
        compound = _get_fn(resolved).body[1].statement
        assert isinstance(compound, ast.Compound)
        inner_name = compound.body[0].declaration.variable_declaration.name
        assert compound.body[1] == ast.S(ast.Return(ast.Var(inner_name)))

    def test_variable_shadowing_function_in_inner_scope_calling_as_function_raises(self) -> None:
        """A variable that shadows a function name in an inner scope cannot be called as a function.

        The inner-scope ``int foo = 0;`` is a valid shadow (different scope from the
        outer ``int foo(void);``), but ``foo()`` inside that block sees ``foo`` as a
        variable and raises a type error.
        """
        foo_decl = ast.D(ast.FunDecl(ast.Function(name="foo", params=(), body=None)))
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            ast.S(
                statement=ast.Compound(
                    body=(
                        _vd("foo", ast.Constant(0)),
                        ast.S(ast.Return(ast.FunctionCall("foo", []))),
                    )
                )
            ),
        )
        with pytest.raises(SemanticError, match="Variable 'foo' called as a function"):
            resolve(_program(body))


# ---------------------------------------------------------------------------
# Linkage tracking
# ---------------------------------------------------------------------------


class TestLinkage:
    """Linkage is recorded correctly in ``VariableBinding`` and ``FunctionBinding``."""

    def test_variable_binding_has_no_linkage(self) -> None:
        """A local variable declaration carries ``Linkage.NONE``."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(ast.VariableDeclaration(name="x", init=None))
        binding = analyser._variable_map["x"]
        assert isinstance(binding, VariableBinding)
        assert binding.linkage is Linkage.NONE

    def test_function_binding_has_external_linkage(self) -> None:
        """A function declaration carries ``Linkage.EXTERNAL``."""
        analyser = SemanticAnalyser()
        analyser._register_fun_declaration(ast.Function(name="foo", params=(), body=None))
        binding = analyser._function_map["foo"]
        assert isinstance(binding, FunctionBinding)
        assert binding.linkage is Linkage.EXTERNAL

    def test_function_definition_has_external_linkage(self) -> None:
        """A function definition (with a body) also carries ``Linkage.EXTERNAL``."""
        analyser = SemanticAnalyser()
        analyser._register_fun_declaration(
            ast.Function(name="bar", params=(), body=(ast.S(ast.Return(ast.Constant(0))),))
        )
        binding = analyser._function_map["bar"]
        assert binding.linkage is Linkage.EXTERNAL

    def test_variable_linkage_preserved_across_scope_entry(self) -> None:
        """``Linkage.NONE`` is preserved on a binding when a new nested scope is entered."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(ast.VariableDeclaration(name="x", init=None))
        analyser._enter_scope()
        binding = analyser._variable_map["x"]
        assert binding.linkage is Linkage.NONE

    def test_function_binding_records_param_types(self) -> None:
        """``FunctionBinding.param_types`` captures the parameter types of the declaration."""
        analyser = SemanticAnalyser()
        analyser._register_fun_declaration(
            ast.Function(
                name="foo",
                params=(ast.Parameter(param_type=ast.IntType(), name="x"),),
                body=None,
            )
        )
        binding = analyser._function_map["foo"]
        assert binding.param_types == (ast.IntType(),)

    def test_function_binding_no_params_records_empty_tuple(self) -> None:
        """A function with no parameters stores an empty ``param_types`` tuple."""
        analyser = SemanticAnalyser()
        analyser._register_fun_declaration(ast.Function(name="foo", params=(), body=None))
        assert analyser._function_map["foo"].param_types == ()


# ---------------------------------------------------------------------------
# Conflicting function declarations
# ---------------------------------------------------------------------------


class TestConflictingDeclarations:
    """Declarations of the same function with mismatched parameter types raise ``SemanticError``."""

    def _int_param(self, name: str) -> ast.Parameter:
        """Return an ``int`` parameter with the given name."""
        return ast.Parameter(param_type=ast.IntType(), name=name)

    def test_forward_decl_and_definition_different_param_count_raises(self) -> None:
        """A forward declaration with one parameter and a definition with two raises ``SemanticError``."""
        foo_decl = ast.Function(name="foo", params=(self._int_param("a"),), body=None)
        foo_def = ast.Function(
            name="foo",
            params=(self._int_param("a"), self._int_param("b")),
            body=(ast.S(ast.Return(ast.Constant(0))),),
        )
        main = ast.Function(name="main", params=(), body=(ast.S(ast.Return(ast.Constant(5))),))
        with pytest.raises(SemanticError, match="Conflicting declarations"):
            resolve(_make_prog(foo_decl, main, foo_def))

    def test_two_declarations_different_param_count_raises(self) -> None:
        """Two forward declarations of the same function with different arities raise ``SemanticError``."""
        foo_zero = ast.Function(name="foo", params=(), body=None)
        foo_one = ast.Function(name="foo", params=(self._int_param("a"),), body=None)
        main = ast.Function(name="main", params=(), body=(ast.S(ast.Return(ast.Constant(0))),))
        with pytest.raises(SemanticError, match="Conflicting declarations"):
            resolve(_make_prog(foo_zero, foo_one, main))

    def test_block_level_declaration_conflicts_with_top_level_raises(self) -> None:
        """A block-scope declaration that disagrees with the top-level declaration raises ``SemanticError``."""
        foo_top = ast.Function(name="foo", params=(self._int_param("a"),), body=None)
        foo_inner = ast.D(ast.FunDecl(ast.Function(name="foo", params=(), body=None)))
        body: tuple[ast.BlockItem, ...] = (
            foo_inner,
            ast.S(ast.Return(ast.Constant(0))),
        )
        main = ast.Function(name="main", params=(), body=body)
        with pytest.raises(SemanticError, match="Conflicting declarations"):
            resolve(_make_prog(foo_top, main))

    def test_matching_declarations_are_valid(self) -> None:
        """Repeated declarations of the same function with identical parameter types are accepted."""
        foo_decl_a = ast.Function(name="foo", params=(self._int_param("a"),), body=None)
        foo_decl_b = ast.Function(name="foo", params=(self._int_param("x"),), body=None)
        foo_def = ast.Function(
            name="foo",
            params=(self._int_param("p"),),
            body=(ast.S(ast.Return(ast.Var("p"))),),
        )
        main = ast.Function(name="main", params=(), body=(ast.S(ast.Return(ast.Constant(0))),))
        resolved = resolve(_make_prog(foo_decl_a, foo_decl_b, foo_def, main))
        assert len(resolved.declarations) == 4

    def test_zero_param_matching_declarations_are_valid(self) -> None:
        """Two forward declarations of the same zero-parameter function are accepted."""
        foo_a = ast.Function(name="foo", params=(), body=None)
        foo_b = ast.Function(name="foo", params=(), body=None)
        main = ast.Function(name="main", params=(), body=(ast.S(ast.Return(ast.Constant(0))),))
        resolved = resolve(_make_prog(foo_a, foo_b, main))
        assert len(resolved.declarations) == 3


# ---------------------------------------------------------------------------
# Argument count checking
# ---------------------------------------------------------------------------


class TestArgumentCount:
    """Function calls are validated against the declared parameter count."""

    def _fn_decl(self, name: str, n_params: int) -> ast.D:
        """Return a block-scope forward declaration of ``name`` with ``n_params`` int params."""
        params = tuple(ast.Parameter(param_type=ast.IntType(), name=f"p{i}") for i in range(n_params))
        return ast.D(ast.FunDecl(ast.Function(name=name, params=params, body=None)))

    def test_too_few_arguments_raises(self) -> None:
        """Calling a one-parameter function with zero arguments raises ``SemanticError``."""
        body: tuple[ast.BlockItem, ...] = (
            self._fn_decl("foo", 1),
            ast.S(ast.Return(ast.FunctionCall("foo", []))),
        )
        with pytest.raises(SemanticError, match="called with 0 argument"):
            resolve(_program(body))

    def test_too_many_arguments_raises(self) -> None:
        """Calling a zero-parameter function with one argument raises ``SemanticError``."""
        body: tuple[ast.BlockItem, ...] = (
            self._fn_decl("foo", 0),
            ast.S(ast.Return(ast.FunctionCall("foo", [ast.Constant(1)]))),
        )
        with pytest.raises(SemanticError, match="called with 1 argument"):
            resolve(_program(body))

    def test_correct_argument_count_is_valid(self) -> None:
        """A call whose argument count matches the declaration resolves without error."""
        body: tuple[ast.BlockItem, ...] = (
            self._fn_decl("foo", 2),
            ast.S(ast.Return(ast.FunctionCall("foo", [ast.Constant(1), ast.Constant(2)]))),
        )
        resolved = resolve(_program(body))
        ret = _get_fn(resolved).body[1].statement
        assert isinstance(ret, ast.Return)
        assert isinstance(ret.exp, ast.FunctionCall)

    def test_zero_argument_call_to_zero_param_function_is_valid(self) -> None:
        """A call with no arguments to a zero-parameter function resolves without error."""
        body: tuple[ast.BlockItem, ...] = (
            self._fn_decl("foo", 0),
            ast.S(ast.Return(ast.FunctionCall("foo", []))),
        )
        resolved = resolve(_program(body))
        assert _get_fn(resolved).body[1] == ast.S(ast.Return(ast.FunctionCall("foo", [])))


# ---------------------------------------------------------------------------
# Function declaration shadowing outer variable
# ---------------------------------------------------------------------------


class TestFunctionShadowsVariable:
    """A block-scope function declaration may shadow an outer variable of the same name."""

    def test_inner_function_decl_shadows_outer_variable_call_resolves(self) -> None:
        """Calling a function whose name matches an outer variable resolves without error.

        The inner ``int foo(void);`` declaration places ``"foo"`` in
        ``_current_block_funs``, which suppresses the "variable called as
        function" check even though ``"foo"`` is also present in
        ``_variable_map`` from the outer scope.
        """
        foo_top = ast.Function(name="foo", params=(), body=None)
        body: tuple[ast.BlockItem, ...] = (
            _vd("foo", ast.Constant(3)),
            ast.S(
                statement=ast.Compound(
                    body=(
                        ast.D(ast.FunDecl(ast.Function(name="foo", params=(), body=None))),
                        ast.S(ast.Expression(ast.FunctionCall("foo", []))),
                        ast.S(ast.Return(ast.Constant(0))),
                    )
                )
            ),
        )
        main = ast.Function(name="main", params=(), body=body)
        resolved = resolve(_make_prog(foo_top, main))
        assert len(resolved.declarations) == 2

    def test_outer_variable_visible_after_inner_scope_exits(self) -> None:
        """After the inner block exits, the outer variable binding is restored.

        ``return foo;`` after the inner block should resolve to the outer
        variable, not raise "function name used as variable".
        """
        foo_top = ast.Function(name="foo", params=(), body=None)
        body: tuple[ast.BlockItem, ...] = (
            _vd("foo", ast.Constant(3)),
            ast.S(
                statement=ast.Compound(
                    body=(
                        ast.D(ast.FunDecl(ast.Function(name="foo", params=(), body=None))),
                        ast.S(ast.Return(ast.Constant(0))),
                    )
                )
            ),
            ast.S(ast.Return(ast.Var("foo"))),
        )
        main = ast.Function(name="main", params=(), body=body)
        resolved = resolve(_make_prog(foo_top, main))
        outer_name = _get_fn(resolved, 1).body[0].declaration.variable_declaration.name
        ret_stmt = _get_fn(resolved, 1).body[2].statement
        assert isinstance(ret_stmt, ast.Return)
        assert ret_stmt.exp == ast.Var(outer_name)

    def test_var_reference_inside_inner_scope_with_function_decl_raises(self) -> None:
        """Using the shadowed name as a plain ``Var`` inside the inner scope raises ``SemanticError``.

        ``_current_block_funs`` contains ``"foo"`` (the inner function
        declaration), so the ``Var("foo")`` path raises immediately.
        """
        foo_top = ast.Function(name="foo", params=(), body=None)
        body: tuple[ast.BlockItem, ...] = (
            _vd("foo", ast.Constant(3)),
            ast.S(
                statement=ast.Compound(
                    body=(
                        ast.D(ast.FunDecl(ast.Function(name="foo", params=(), body=None))),
                        ast.S(ast.Return(ast.Var("foo"))),
                    )
                )
            ),
        )
        main = ast.Function(name="main", params=(), body=body)
        with pytest.raises(SemanticError, match="Function name 'foo' used as a variable"):
            resolve(_make_prog(foo_top, main))


# ---------------------------------------------------------------------------
# File-scope variable declarations
# ---------------------------------------------------------------------------


def _file_var(name: str, init: ast.Exp | None = None, sc: ast.StorageClass | None = None) -> ast.VarDecl:
    """Return a top-level ``VarDecl`` with the given name, optional initialiser, and optional storage class.

    Args:
        name: The variable name.
        init: Optional initialiser expression.
        sc: Optional storage class specifier.

    Returns:
        An ``ast.VarDecl`` wrapping a ``VariableDeclaration``.
    """
    return ast.VarDecl(ast.VariableDeclaration(name, init, sc))


def _prog_with_decls(*decls: ast.Declaration) -> ast.Program:
    """Build an ``ast.Program`` from an arbitrary mix of top-level declarations.

    Args:
        *decls: Any combination of ``ast.FunDecl`` and ``ast.VarDecl`` nodes.

    Returns:
        An ``ast.Program`` whose ``declarations`` tuple contains all given nodes.
    """
    return ast.Program(declarations=decls)


class TestFileScopeVariableDeclarations:
    """File-scope variable declarations are validated for linkage and duplicate definitions."""

    def test_file_scope_variable_is_not_renamed(self) -> None:
        """A file-scope variable keeps its original name after resolution."""
        main_fn = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(ast.Var("g"))),),
        )
        program = _prog_with_decls(_file_var("g", ast.Constant(0)), ast.FunDecl(main_fn))
        resolved = resolve(program)
        var_decl = resolved.declarations[0].variable_declaration
        assert var_decl.name == "g"

    def test_file_scope_variable_visible_in_function_body(self) -> None:
        """A file-scope variable declared before a function is visible inside that function."""
        main_fn = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(ast.Var("g"))),),
        )
        program = _prog_with_decls(_file_var("g", ast.Constant(0)), ast.FunDecl(main_fn))
        resolved = resolve(program)
        ret = _get_fn(resolved).body[0].statement
        assert isinstance(ret, ast.Return)
        assert ret.exp == ast.Var("g")

    def test_file_scope_variable_not_visible_before_declaration(self) -> None:
        """A file-scope variable declared after a function is NOT visible inside that function."""
        main_fn = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(ast.Var("g"))),),
        )
        program = _prog_with_decls(ast.FunDecl(main_fn), _file_var("g", ast.Constant(0)))
        with pytest.raises(SemanticError, match="Undeclared"):
            resolve(program)

    def test_static_file_scope_variable_has_internal_linkage(self) -> None:
        """A ``static`` file-scope variable is recorded with ``Linkage.INTERNAL``."""
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("x", None, ast.StorageClass.Static))
        assert analyser._variable_map["x"].linkage is Linkage.INTERNAL

    def test_extern_file_scope_variable_has_external_linkage(self) -> None:
        """An ``extern`` file-scope variable is recorded with ``Linkage.EXTERNAL``."""
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("x", None, ast.StorageClass.Extern))
        assert analyser._variable_map["x"].linkage is Linkage.EXTERNAL

    def test_no_storage_class_file_scope_has_external_linkage(self) -> None:
        """A file-scope variable with no storage class specifier has ``Linkage.EXTERNAL``."""
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("x", None))
        assert analyser._variable_map["x"].linkage is Linkage.EXTERNAL

    def test_file_scope_variable_after_function_of_same_name_raises(self) -> None:
        """A file-scope variable declaration after a function of the same name raises ``SemanticError``."""
        foo_fn = ast.Function(name="foo", params=(), body=(ast.S(ast.Return(ast.Constant(0))),))
        main_fn = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(ast.FunctionCall("foo", []))),),
        )
        with pytest.raises(SemanticError, match="redeclared as a variable"):
            resolve(_prog_with_decls(ast.FunDecl(foo_fn), ast.FunDecl(main_fn), _file_var("foo", ast.Constant(3))))

    def test_function_after_file_scope_variable_of_same_name_raises(self) -> None:
        """A function declaration after a file-scope variable of the same name raises ``SemanticError``."""
        foo_fn = ast.Function(name="foo", params=(), body=(ast.S(ast.Return(ast.Constant(0))),))
        with pytest.raises(SemanticError, match="redeclared as a function"):
            resolve(_prog_with_decls(_file_var("foo", ast.Constant(3)), ast.FunDecl(foo_fn)))

    def test_file_scope_variable_non_constant_init_raises(self) -> None:
        """A file-scope variable with a non-constant initialiser raises ``SemanticError``.

        All file-scope variables have static storage duration, so their
        initialisers must be compile-time constants.  An expression referencing
        another variable (e.g. ``int b = 1 + a``) is therefore invalid.
        """
        main_fn = ast.Function(name="main", params=(), body=(ast.S(ast.Return(ast.Constant(0))),))
        with pytest.raises(SemanticError, match="constant expression"):
            resolve(
                _prog_with_decls(
                    _file_var("a", ast.Constant(10)),
                    _file_var("b", ast.Binary(ast.Add, ast.Constant(1), ast.Var("a"))),
                    ast.FunDecl(main_fn),
                )
            )

    def test_file_scope_variable_constant_init_is_valid(self) -> None:
        """A file-scope variable with a ``Constant`` initialiser is accepted."""
        analyser = SemanticAnalyser()
        result = analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("x", ast.Constant(7)))
        assert result.init == ast.Constant(7)

    def test_file_scope_extern_then_static_raises(self) -> None:
        """A ``static`` file-scope declaration conflicts with a prior ``extern`` declaration of the same name.

        ``int x; static int x;`` changes the linkage from external to internal,
        which is a constraint violation.
        """
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("x", None))
        with pytest.raises(SemanticError, match="linkage"):
            analyser._resolve_file_scope_variable_declaration(
                ast.VariableDeclaration("x", None, ast.StorageClass.Static)
            )

    def test_file_scope_static_then_extern_raises(self) -> None:
        """A non-``static`` file-scope declaration conflicts with a prior ``static`` declaration of the same name."""
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("x", None, ast.StorageClass.Static))
        with pytest.raises(SemanticError, match="linkage"):
            analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("x", None))

    def test_block_extern_then_file_static_raises(self) -> None:
        """A ``static`` file-scope declaration conflicts with a block-scope ``extern`` in a preceding function.

        The block-scope ``extern int x`` establishes external linkage for ``x``
        in ``_file_scope_linkage``.  The subsequent ``static int x`` at file
        scope attempts to change that linkage to internal, which C forbids.
        """
        main_fn = ast.Function(
            name="main",
            params=(),
            body=(
                ast.D(ast.VarDecl(ast.VariableDeclaration("x", None, ast.StorageClass.Extern))),
                ast.S(ast.Return(ast.Var("x"))),
            ),
        )
        with pytest.raises(SemanticError, match="linkage"):
            resolve(
                _prog_with_decls(
                    ast.FunDecl(main_fn),
                    _file_var("x", ast.Constant(10), sc=ast.StorageClass.Static),
                )
            )

    def test_file_static_then_block_extern_is_valid(self) -> None:
        """A block-scope ``extern`` after a file-scope ``static`` declaration is valid.

        C11 §6.2.2p4: an ``extern`` declaration in a scope where a prior
        declaration is visible inherits the established linkage.  The block-scope
        ``extern int x`` therefore refers to the same internal-linkage object as
        ``static int x`` and is not a conflict.
        """
        main_fn = ast.Function(
            name="main",
            params=(),
            body=(
                ast.D(ast.VarDecl(ast.VariableDeclaration("x", None, ast.StorageClass.Extern))),
                ast.S(ast.Return(ast.Var("x"))),
            ),
        )
        resolved = resolve(
            _prog_with_decls(
                _file_var("x", ast.Constant(10), sc=ast.StorageClass.Static),
                ast.FunDecl(main_fn),
            )
        )
        assert resolved is not None

    def test_file_scope_static_then_extern_specifier_is_valid(self) -> None:
        """A file-scope ``extern`` declaration after a ``static`` declaration of the same name is valid.

        C11 §6.2.2p4: the ``extern`` declaration inherits the prior internal
        linkage; it is not a redeclaration with a conflicting linkage.
        """
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(
            ast.VariableDeclaration("x", ast.Constant(1), ast.StorageClass.Static)
        )
        result = analyser._resolve_file_scope_variable_declaration(
            ast.VariableDeclaration("x", None, ast.StorageClass.Extern)
        )
        assert result.name == "x"
        assert analyser._variable_map["x"].linkage is Linkage.INTERNAL

    def test_two_file_scope_definitions_raises(self) -> None:
        """Two file-scope definitions of the same variable (both with initialisers) raise ``SemanticError``.

        ``int foo = 3; int foo = 4;`` produces two definitions; C permits at
        most one definition per name in a translation unit.
        """
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("foo", ast.Constant(3)))
        with pytest.raises(SemanticError, match="Duplicate definition"):
            analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("foo", ast.Constant(4)))

    def test_tentative_then_definition_is_valid(self) -> None:
        """A tentative definition (no initialiser) followed by a definition is accepted."""
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("foo", None))
        result = analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("foo", ast.Constant(4)))
        assert result.init == ast.Constant(4)

    def test_definition_then_tentative_is_valid(self) -> None:
        """A definition followed by a tentative definition (no initialiser) is accepted."""
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("foo", ast.Constant(3)))
        result = analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("foo", None))
        assert result.init is None

    def test_two_tentative_definitions_are_valid(self) -> None:
        """Two tentative definitions (no initialisers) of the same name are permitted."""
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("foo", None))
        result = analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("foo", None))
        assert result.name == "foo"


# ---------------------------------------------------------------------------
# Block-scope storage class specifiers
# ---------------------------------------------------------------------------


class TestBlockScopeStorageClass:
    """Block-scope ``static`` and ``extern`` declarations are validated and resolved correctly."""

    def test_block_scope_static_is_renamed(self) -> None:
        """A ``static`` local variable receives a unique internal name like any other local."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(ast.VariableDeclaration("x", None, ast.StorageClass.Static))
        assert result.name.startswith(".Vx.")

    def test_block_scope_static_preserves_storage_class(self) -> None:
        """The ``Static`` storage class is preserved on the resolved ``VariableDeclaration``."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(ast.VariableDeclaration("x", None, ast.StorageClass.Static))
        assert result.storage_class is ast.StorageClass.Static

    def test_block_scope_static_has_internal_linkage(self) -> None:
        """A ``static`` local variable is recorded in ``_variable_map`` with ``Linkage.INTERNAL``."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(ast.VariableDeclaration("x", None, ast.StorageClass.Static))
        assert analyser._variable_map["x"].linkage is Linkage.INTERNAL

    def test_block_scope_static_duplicate_raises(self) -> None:
        """Declaring the same ``static`` local twice in the same block raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(ast.VariableDeclaration("x", None, ast.StorageClass.Static))
        with pytest.raises(SemanticError, match="Duplicate"):
            analyser.resolve_variable_declaration(ast.VariableDeclaration("x", None, ast.StorageClass.Static))

    def test_block_scope_extern_is_not_renamed(self) -> None:
        """A block-scope ``extern`` declaration keeps its original source name."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(ast.VariableDeclaration("g", None, ast.StorageClass.Extern))
        assert result.name == "g"

    def test_block_scope_extern_has_external_linkage(self) -> None:
        """A block-scope ``extern`` declaration is recorded with ``Linkage.EXTERNAL``."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(ast.VariableDeclaration("g", None, ast.StorageClass.Extern))
        assert analyser._variable_map["g"].linkage is Linkage.EXTERNAL

    def test_block_scope_extern_with_init_raises(self) -> None:
        """A block-scope ``extern`` declaration with an initialiser raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        with pytest.raises(SemanticError, match="may not have an initialiser"):
            analyser.resolve_variable_declaration(
                ast.VariableDeclaration("g", ast.Constant(5), ast.StorageClass.Extern)
            )

    def test_block_scope_extern_repeated_is_valid(self) -> None:
        """Two ``extern`` declarations of the same name in the same block are permitted."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(ast.VariableDeclaration("g", None, ast.StorageClass.Extern))
        result = analyser.resolve_variable_declaration(ast.VariableDeclaration("g", None, ast.StorageClass.Extern))
        assert result.name == "g"

    def test_block_scope_extern_after_non_extern_raises(self) -> None:
        """An ``extern`` declaration after a non-``extern`` declaration of the same name raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(ast.VariableDeclaration("x", None))
        with pytest.raises(SemanticError, match="previously declared without linkage"):
            analyser.resolve_variable_declaration(ast.VariableDeclaration("x", None, ast.StorageClass.Extern))

    def test_block_scope_non_extern_after_extern_raises(self) -> None:
        """A non-``extern`` declaration after an ``extern`` declaration of the same name raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(ast.VariableDeclaration("x", None, ast.StorageClass.Extern))
        with pytest.raises(SemanticError, match="previously declared with linkage"):
            analyser.resolve_variable_declaration(ast.VariableDeclaration("x", None))

    def test_block_scope_extern_resolves_in_function_body(self) -> None:
        """A block-scope ``extern`` declaration makes the variable usable inside the function."""
        body: tuple[ast.BlockItem, ...] = (
            ast.D(ast.VarDecl(ast.VariableDeclaration("g", None, ast.StorageClass.Extern))),
            ast.S(ast.Return(ast.Var("g"))),
        )
        resolved = _resolved_body(body)
        ret = resolved[1].statement
        assert isinstance(ret, ast.Return)
        assert ret.exp == ast.Var("g")

    def test_block_scope_static_non_constant_init_raises(self) -> None:
        """A ``static`` local variable with a non-constant initialiser raises ``SemanticError``.

        Variables with static storage duration are initialised at program start, so
        their initialisers must be compile-time constants.  A reference to another
        variable (e.g. ``static int b = a * 2``) is therefore invalid.
        """
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(ast.VariableDeclaration("a", ast.Constant(1)))
        with pytest.raises(SemanticError, match="constant expression"):
            analyser.resolve_variable_declaration(
                ast.VariableDeclaration(
                    "b",
                    ast.Binary(ast.Multiply, ast.Var("a"), ast.Constant(2)),
                    ast.StorageClass.Static,
                )
            )

    def test_block_scope_static_constant_init_is_valid(self) -> None:
        """A ``static`` local variable with a ``Constant`` initialiser is accepted."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration("x", ast.Constant(42), ast.StorageClass.Static)
        )
        assert result.init == ast.Constant(42)

    def test_block_scope_extern_after_static_local_raises(self) -> None:
        """A block-scope ``extern`` after a ``static`` local of the same name raises ``SemanticError``.

        A block-scope ``static`` variable has no linkage (C11 §6.2.2p3 only
        confers internal linkage on *file-scope* ``static``).  A subsequent
        ``extern`` declaration in the same block therefore conflicts, exactly
        as it would with a plain local variable.
        """
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(ast.VariableDeclaration("x", ast.Constant(0), ast.StorageClass.Static))
        with pytest.raises(SemanticError, match="without linkage"):
            analyser.resolve_variable_declaration(ast.VariableDeclaration("x", None, ast.StorageClass.Extern))

    def test_block_scope_static_shadows_file_scope_variable(self) -> None:
        """A ``static`` local variable shadows a file-scope variable of the same name.

        Inside the function the ``static`` local's unique internal name is used;
        the file-scope variable is not accessible under the original name.
        """
        g_decl = _file_var("g", ast.Constant(0))
        main_fn = ast.Function(
            name="main",
            params=(),
            body=(
                ast.D(ast.VarDecl(ast.VariableDeclaration("g", ast.Constant(1), ast.StorageClass.Static))),
                ast.S(ast.Return(ast.Var("g"))),
            ),
        )
        program = _prog_with_decls(g_decl, ast.FunDecl(main_fn))
        resolved = resolve(program)
        fn = _get_fn(resolved)
        static_name = fn.body[0].declaration.variable_declaration.name
        assert static_name.startswith(".Vg.")
        ret = fn.body[1].statement
        assert isinstance(ret, ast.Return)
        assert ret.exp == ast.Var(static_name)


# ---------------------------------------------------------------------------
# Function linkage conflicts
# ---------------------------------------------------------------------------


class TestFunctionLinkage:
    """Function declarations must agree on linkage across the entire translation unit."""

    def _fn(
        self,
        name: str,
        body: tuple[ast.BlockItem, ...] | None = None,
        static: bool = False,
        extern: bool = False,
    ) -> ast.Function:
        """Build a ``Function`` node with an optional storage class.

        Args:
            name: The function name.
            body: The function body block items, or ``None`` for a forward declaration.
            static: If ``True``, sets ``storage_class=StorageClass.Static``.
            extern: If ``True``, sets ``storage_class=StorageClass.Extern``.
                Mutually exclusive with ``static``.

        Returns:
            An ``ast.Function`` with ``params=()`` and the given body and linkage.
        """
        if static:
            sc: ast.StorageClass | None = ast.StorageClass.Static
        elif extern:
            sc = ast.StorageClass.Extern
        else:
            sc = None
        return ast.Function(name=name, params=(), body=body, storage_class=sc)

    def test_extern_then_static_function_raises(self) -> None:
        """A ``static`` definition after a non-``static`` declaration raises ``SemanticError``.

        ``int foo(void);`` gives ``foo`` external linkage.  The subsequent
        ``static int foo(void) { ... }`` attempts to change it to internal
        linkage, which C forbids.
        """
        program = _make_prog(
            self._fn("foo", None),
            self._fn("foo", (ast.S(ast.Return(ast.Constant(0))),), static=True),
            self._fn("main", (ast.S(ast.Return(ast.FunctionCall("foo", []))),)),
        )
        with pytest.raises(SemanticError, match="linkage"):
            resolve(program)

    def test_static_then_non_static_declaration_is_valid(self) -> None:
        """A non-``static`` declaration after a ``static`` declaration of the same function is valid.

        C11 §6.2.2p5: a function declaration with no storage-class specifier is
        treated exactly as ``extern``.  §6.2.2p4: ``extern`` inherits the linkage
        of the prior visible declaration.  ``int foo(void);`` after
        ``static int foo(void) {...}`` therefore inherits internal linkage and is
        not a conflict.
        """
        program = _make_prog(
            self._fn("foo", (ast.S(ast.Return(ast.Constant(0))),), static=True),
            self._fn("foo", None),
            self._fn("main", (ast.S(ast.Return(ast.FunctionCall("foo", []))),)),
        )
        resolved = resolve(program)
        assert len(resolved.declarations) == 3

    def test_static_then_extern_specifier_declaration_is_valid(self) -> None:
        """A file-scope ``extern`` declaration after a ``static`` declaration is valid.

        The ``extern`` inherits the prior internal linkage; it is not a conflict.
        """
        program = _make_prog(
            self._fn("foo", None, static=True),
            self._fn("foo", (ast.S(ast.Return(ast.Constant(0))),), extern=True),
            self._fn("main", (ast.S(ast.Return(ast.FunctionCall("foo", []))),)),
        )
        resolved = resolve(program)
        assert len(resolved.declarations) == 3

    def test_block_scope_function_declaration_inherits_internal_linkage(self) -> None:
        """A block-scope function declaration inherits the internal linkage of a prior file-scope declaration.

        C11 §6.2.2p4/p5: the block-scope ``int my_fun(void);`` (treated as
        ``extern``) inherits the internal linkage established by the earlier
        ``static int my_fun(void);`` at file scope.
        """
        my_fun_decl = ast.D(ast.FunDecl(ast.Function(name="my_fun", params=(), body=None)))
        caller_fn = ast.Function(
            name="caller",
            params=(),
            body=(
                my_fun_decl,
                ast.S(ast.Return(ast.FunctionCall("my_fun", []))),
            ),
        )
        program = _make_prog(
            self._fn("my_fun", None, static=True),
            caller_fn,
            self._fn("my_fun", (ast.S(ast.Return(ast.Constant(0))),), static=True),
            self._fn("main", (ast.S(ast.Return(ast.FunctionCall("my_fun", []))),)),
        )
        resolved = resolve(program)
        assert resolved is not None

    def test_static_function_definition_is_valid(self) -> None:
        """A ``static`` function defined with no prior declaration is accepted."""
        program = _make_prog(
            self._fn("foo", (ast.S(ast.Return(ast.Constant(0))),), static=True),
            self._fn("main", (ast.S(ast.Return(ast.FunctionCall("foo", []))),)),
        )
        resolved = resolve(program)
        assert len(resolved.declarations) == 2

    def test_static_declaration_then_static_definition_is_valid(self) -> None:
        """A ``static`` forward declaration followed by a ``static`` definition is accepted."""
        program = _make_prog(
            self._fn("foo", None, static=True),
            self._fn("foo", (ast.S(ast.Return(ast.Constant(0))),), static=True),
            self._fn("main", (ast.S(ast.Return(ast.FunctionCall("foo", []))),)),
        )
        resolved = resolve(program)
        assert len(resolved.declarations) == 3

    def test_function_binding_records_internal_linkage_for_static(self) -> None:
        """``_function_map`` stores ``Linkage.INTERNAL`` for a ``static`` function."""
        analyser = SemanticAnalyser()
        analyser._register_fun_declaration(self._fn("foo", None, static=True))
        assert analyser._function_map["foo"].linkage is Linkage.INTERNAL

    def test_function_binding_records_external_linkage_for_non_static(self) -> None:
        """``_function_map`` stores ``Linkage.EXTERNAL`` for a non-``static`` function."""
        analyser = SemanticAnalyser()
        analyser._register_fun_declaration(self._fn("foo", None))
        assert analyser._function_map["foo"].linkage is Linkage.EXTERNAL
