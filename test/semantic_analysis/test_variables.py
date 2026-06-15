"""Tests for variable declaration, expression, statement, program, and compound resolution.

Verifies that ``SemanticAnalyser`` correctly renames variable declarations to
unique internal identifiers, rejects semantic violations (duplicate declarations,
undeclared variables, invalid lvalues), and leaves non-variable constructs
unchanged.
"""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import SemanticAnalyser, resolve
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import _get_fn, _make_prog, _program, _resolved_body, _vd

# ---------------------------------------------------------------------------
# Variable declaration resolution
# ---------------------------------------------------------------------------


class TestResolveDeclaration:
    """``resolve_variable_declaration`` renames variables and enforces scope rules."""

    def test_declaration_assigns_unique_suffix(self) -> None:
        """The first declaration in a fresh analyser receives the unique name ``.Vx.0``."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration(name="x", init=None, var_type=ast.IntType())
        )
        assert result.name == ".Vx.0"

    def test_declaration_preserves_none_init(self) -> None:
        """A declaration without an initialiser keeps ``init=None`` after resolution."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration(name="x", init=None, var_type=ast.IntType())
        )
        assert result.init is None

    def test_declaration_with_constant_init_passes_through(self) -> None:
        """A constant initialiser is returned wrapped in ``SingleInit``."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration(name="x", init=ast.Constant(value=5), var_type=ast.IntType())
        )
        assert result.init == ast.SingleInit(ast.Constant(value=5))

    def test_declaration_with_var_init_renames_init_var(self) -> None:
        """An initialiser that references an already-declared variable is renamed."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(  # y -> .Vy.0
            ast.VariableDeclaration(name="y", init=None, var_type=ast.IntType())
        )
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration(name="x", init=ast.Var(name="y"), var_type=ast.IntType())
        )
        assert result.init == ast.SingleInit(ast.Var(name=".Vy.0"))

    def test_two_declarations_receive_distinct_suffixes(self) -> None:
        """Successive declarations in the same analyser get incrementing suffixes."""
        analyser = SemanticAnalyser()
        r1 = analyser.resolve_variable_declaration(ast.VariableDeclaration(name="x", init=None, var_type=ast.IntType()))
        r2 = analyser.resolve_variable_declaration(ast.VariableDeclaration(name="y", init=None, var_type=ast.IntType()))
        assert r1.name == ".Vx.0"
        assert r2.name == ".Vy.1"

    def test_duplicate_declaration_raises(self) -> None:
        """Declaring the same name twice in the same scope raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(ast.VariableDeclaration(name="x", init=None, var_type=ast.IntType()))
        with pytest.raises(SemanticError, match="Duplicate"):
            analyser.resolve_variable_declaration(ast.VariableDeclaration(name="x", init=None, var_type=ast.IntType()))

    def test_self_referential_init_is_accepted(self) -> None:
        """``int x = x;`` is accepted: C scoping places ``x`` in scope before the initialiser.

        Reading ``x`` before it is initialised is undefined behaviour at runtime,
        but is not a constraint violation and must not be rejected by the analyser.
        """
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration(name="x", init=ast.Var(name="x"), var_type=ast.IntType())
        )
        assert result.name == ".Vx.0"
        assert result.init == ast.SingleInit(ast.Var(name=".Vx.0"))

    def test_short_circuit_init_referencing_own_variable_is_accepted(self) -> None:
        """``int a = 0 && a;`` is accepted: ``a`` is in scope and the right side is never evaluated."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration(
                name="a",
                init=ast.Binary(operator=ast.And(), left=ast.Constant(value=0), right=ast.Var(name="a")),
                var_type=ast.IntType(),
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
            analyser.resolve_variable_declaration(ast.VariableDeclaration(name=name, init=None, var_type=ast.IntType()))
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
# Full program resolution
# ---------------------------------------------------------------------------


class TestResolveProgram:
    """``resolve`` correctly renames variables and preserves program structure."""

    def test_declaration_name_is_uniquified(self) -> None:
        """A declared variable's name is replaced by its unique internal identifier."""
        body = (_vd("x"),)
        resolved = _resolved_body(body)
        assert resolved[0] == ast.D(ast.VarDecl(ast.VariableDeclaration(".Vx.0", None, var_type=ast.IntType())))

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
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        fn2 = ast.Function(
            name="fn2",
            params=(),
            body=(ast.S(statement=ast.Return(exp=ast.Var(name="x"))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
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
        r1 = SemanticAnalyser().resolve_variable_declaration(
            ast.VariableDeclaration(name="x", init=None, var_type=ast.IntType())
        )
        r2 = SemanticAnalyser().resolve_variable_declaration(
            ast.VariableDeclaration(name="x", init=None, var_type=ast.IntType())
        )
        assert r1.name == ".Vx.0"
        assert r2.name == ".Vx.0"

    def test_second_call_on_same_instance_increments(self) -> None:
        """A single instance increments its counter across consecutive declarations."""
        analyser = SemanticAnalyser()
        r1 = analyser.resolve_variable_declaration(ast.VariableDeclaration(name="a", init=None, var_type=ast.IntType()))
        r2 = analyser.resolve_variable_declaration(ast.VariableDeclaration(name="b", init=None, var_type=ast.IntType()))
        assert r1.name == ".Va.0"
        assert r2.name == ".Vb.1"


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

        C11 sec. 6.2.1 places a block-scope identifier in scope immediately after
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
        assert inner_decl.init == ast.SingleInit(ast.Var(name=inner_decl.name))

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
        assert inner_decl.init == ast.SingleInit(ast.Var(name=outer_name))
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

        Tests the ``Label -> _resolve_statement(Compound) -> _resolve_block`` path
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
# _type_check_exp - per-form type annotation
# ---------------------------------------------------------------------------


class TestTypeCheckExp:
    """``_type_check_exp`` annotates expression nodes with their C types."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _analyser_with_vars(self, **vars_and_types: ast.Type) -> tuple[SemanticAnalyser, dict[str, str]]:
        """Declare variables in a fresh analyser and return it alongside a name map.

        Args:
            **vars_and_types: Source-level variable names mapped to their declared C types.

        Returns:
            A ``(analyser, name_map)`` tuple where ``name_map`` maps each source name
            to the unique internal name assigned during declaration.
        """
        analyser = SemanticAnalyser()
        name_map: dict[str, str] = {}
        for src_name, var_type in vars_and_types.items():
            resolved = analyser.resolve_variable_declaration(ast.VariableDeclaration(src_name, None, var_type))
            name_map[src_name] = resolved.name
        return analyser, name_map

    # ------------------------------------------------------------------
    # Short-circuit: already-typed expressions
    # ------------------------------------------------------------------

    def test_already_typed_expression_returned_unchanged(self) -> None:
        """An expression whose ``exp_type`` is already set is returned as the same object."""
        analyser, name_map = self._analyser_with_vars(x=ast.IntType())
        deliberately_wrong = ast.Var(name_map["x"], exp_type=ast.LongType())
        result = analyser._type_check_exp(deliberately_wrong)
        assert result is deliberately_wrong
        assert result.exp_type == ast.LongType()

    # ------------------------------------------------------------------
    # Var
    # ------------------------------------------------------------------

    def test_var_int_variable_receives_int_type(self) -> None:
        """A ``Var`` referencing a declared ``int`` variable is annotated with ``IntType``."""
        analyser, name_map = self._analyser_with_vars(x=ast.IntType())
        result = analyser._type_check_exp(ast.Var(name_map["x"]))
        assert result.exp_type == ast.IntType()

    def test_var_long_variable_receives_long_type(self) -> None:
        """A ``Var`` referencing a declared ``long`` variable is annotated with ``LongType``."""
        analyser, name_map = self._analyser_with_vars(x=ast.LongType())
        result = analyser._type_check_exp(ast.Var(name_map["x"]))
        assert result.exp_type == ast.LongType()

    def test_var_with_function_name_has_pointer_to_fun_type(self) -> None:
        """A ``Var`` whose resolved name maps to a ``FunAttr`` is typed as a pointer to that function type.

        C function-name decay: a function name used as a value yields a pointer to the function's type.
        """
        analyser = SemanticAnalyser()
        analyser._register_fun_declaration(
            ast.Function("foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
        )
        result = analyser._type_check_exp(ast.Var("foo"))
        assert result.exp_type == ast.Pointer(ast.FunType(params=(), ret=ast.IntType()))

    def test_var_undeclared_raises(self) -> None:
        """A ``Var`` whose name is absent from ``_type_table`` raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        with pytest.raises(SemanticError, match="Undeclared"):
            analyser._type_check_exp(ast.Var("ghost"))

    def test_var_type_propagated_through_resolve_exp(self) -> None:
        """``resolve_exp`` produces a ``Var`` whose ``exp_type`` matches the declared type."""
        analyser, _ = self._analyser_with_vars(x=ast.LongType())
        result = analyser.resolve_exp(ast.Var("x"))
        assert result.exp_type == ast.LongType()

    # ------------------------------------------------------------------
    # Constant
    # ------------------------------------------------------------------

    def test_const_int_receives_int_type(self) -> None:
        """A ``Constant`` wrapping a ``ConstInt`` is annotated with ``IntType``."""
        analyser = SemanticAnalyser()
        result = analyser._type_check_exp(ast.Constant(ast.ConstInt(42)))
        assert result.exp_type == ast.IntType()

    def test_const_long_receives_long_type(self) -> None:
        """A ``Constant`` wrapping a ``ConstLong`` is annotated with ``LongType``."""
        analyser = SemanticAnalyser()
        result = analyser._type_check_exp(ast.Constant(ast.ConstLong(42)))
        assert result.exp_type == ast.LongType()

    # ------------------------------------------------------------------
    # Unary
    # ------------------------------------------------------------------

    def test_unary_not_always_produces_int(self) -> None:
        """``Not`` applied to a ``long`` operand still yields ``IntType``."""
        analyser = SemanticAnalyser()
        long_const = ast.Constant(ast.ConstLong(1), exp_type=ast.LongType())
        result = analyser._type_check_exp(ast.Unary(ast.Not(), long_const))
        assert result.exp_type == ast.IntType()

    def test_unary_negate_propagates_operand_type(self) -> None:
        """``Negate`` applied to a ``long`` operand yields ``LongType``."""
        analyser = SemanticAnalyser()
        long_const = ast.Constant(ast.ConstLong(1), exp_type=ast.LongType())
        result = analyser._type_check_exp(ast.Unary(ast.Negate(), long_const))
        assert result.exp_type == ast.LongType()

    def test_unary_complement_propagates_operand_type(self) -> None:
        """``Complement`` applied to an ``int`` operand yields ``IntType``."""
        analyser = SemanticAnalyser()
        int_const = ast.Constant(ast.ConstInt(1), exp_type=ast.IntType())
        result = analyser._type_check_exp(ast.Unary(ast.Complement(), int_const))
        assert result.exp_type == ast.IntType()

    # ------------------------------------------------------------------
    # Binary - arithmetic
    # ------------------------------------------------------------------

    def test_binary_int_plus_int_produces_int(self) -> None:
        """``int + int`` is annotated with ``IntType``."""
        analyser = SemanticAnalyser()
        lhs = ast.Constant(ast.ConstInt(1), exp_type=ast.IntType())
        rhs = ast.Constant(ast.ConstInt(2), exp_type=ast.IntType())
        result = analyser._type_check_exp(ast.Binary(ast.Add(), lhs, rhs))
        assert result.exp_type == ast.IntType()

    def test_binary_long_plus_int_produces_long(self) -> None:
        """``long + int`` is annotated with ``LongType``."""
        analyser = SemanticAnalyser()
        lhs = ast.Constant(ast.ConstLong(1), exp_type=ast.LongType())
        rhs = ast.Constant(ast.ConstInt(2), exp_type=ast.IntType())
        result = analyser._type_check_exp(ast.Binary(ast.Add(), lhs, rhs))
        assert result.exp_type == ast.LongType()

    def test_binary_int_plus_long_produces_long(self) -> None:
        """``int + long`` is annotated with ``LongType``."""
        analyser = SemanticAnalyser()
        lhs = ast.Constant(ast.ConstInt(1), exp_type=ast.IntType())
        rhs = ast.Constant(ast.ConstLong(2), exp_type=ast.LongType())
        result = analyser._type_check_exp(ast.Binary(ast.Add(), lhs, rhs))
        assert result.exp_type == ast.LongType()

    # ------------------------------------------------------------------
    # Binary - logical and comparison (always int)
    # ------------------------------------------------------------------

    def test_binary_comparison_long_operands_still_produces_int(self) -> None:
        """Comparison operators always yield ``IntType``, even with ``long`` operands."""
        analyser = SemanticAnalyser()
        lhs = ast.Constant(ast.ConstLong(1), exp_type=ast.LongType())
        rhs = ast.Constant(ast.ConstLong(2), exp_type=ast.LongType())
        for op in (
            ast.Equal(),
            ast.NotEqual(),
            ast.LessThan(),
            ast.LessOrEqual(),
            ast.GreaterThan(),
            ast.GreaterOrEqual(),
        ):
            result = analyser._type_check_exp(ast.Binary(op, lhs, rhs))
            assert result.exp_type == ast.IntType(), f"Expected IntType for {type(op).__name__}"

    def test_binary_logical_long_operands_still_produces_int(self) -> None:
        """Logical ``And`` and ``Or`` always yield ``IntType``, even with ``long`` operands."""
        analyser = SemanticAnalyser()
        lhs = ast.Constant(ast.ConstLong(1), exp_type=ast.LongType())
        rhs = ast.Constant(ast.ConstLong(2), exp_type=ast.LongType())
        for op in (ast.And(), ast.Or()):
            result = analyser._type_check_exp(ast.Binary(op, lhs, rhs))
            assert result.exp_type == ast.IntType(), f"Expected IntType for {type(op).__name__}"

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def test_assignment_type_matches_lvalue(self) -> None:
        """An ``Assignment`` is annotated with the type of its lvalue."""
        analyser, name_map = self._analyser_with_vars(x=ast.LongType())
        lvalue = ast.Var(name_map["x"], exp_type=ast.LongType())
        rvalue = ast.Constant(ast.ConstInt(5), exp_type=ast.IntType())
        result = analyser._type_check_exp(ast.Assignment(lvalue, rvalue))
        assert result.exp_type == ast.LongType()

    # ------------------------------------------------------------------
    # Conditional
    # ------------------------------------------------------------------

    def test_conditional_int_branches_produce_int(self) -> None:
        """A conditional with two ``int`` branches is annotated with ``IntType``."""
        analyser = SemanticAnalyser()
        cond = ast.Constant(ast.ConstInt(1), exp_type=ast.IntType())
        true_br = ast.Constant(ast.ConstInt(2), exp_type=ast.IntType())
        false_br = ast.Constant(ast.ConstInt(3), exp_type=ast.IntType())
        result = analyser._type_check_exp(ast.Conditional(cond, true_br, false_br))
        assert result.exp_type == ast.IntType()

    def test_conditional_long_true_branch_produces_long(self) -> None:
        """A conditional with a ``long`` true branch is annotated with ``LongType``."""
        analyser = SemanticAnalyser()
        cond = ast.Constant(ast.ConstInt(1), exp_type=ast.IntType())
        true_br = ast.Constant(ast.ConstLong(2), exp_type=ast.LongType())
        false_br = ast.Constant(ast.ConstInt(3), exp_type=ast.IntType())
        result = analyser._type_check_exp(ast.Conditional(cond, true_br, false_br))
        assert result.exp_type == ast.LongType()

    def test_conditional_long_false_branch_produces_long(self) -> None:
        """A conditional with a ``long`` false branch is annotated with ``LongType``."""
        analyser = SemanticAnalyser()
        cond = ast.Constant(ast.ConstInt(1), exp_type=ast.IntType())
        true_br = ast.Constant(ast.ConstInt(2), exp_type=ast.IntType())
        false_br = ast.Constant(ast.ConstLong(3), exp_type=ast.LongType())
        result = analyser._type_check_exp(ast.Conditional(cond, true_br, false_br))
        assert result.exp_type == ast.LongType()

    # ------------------------------------------------------------------
    # Cast
    # ------------------------------------------------------------------

    def test_cast_annotated_with_target_type(self) -> None:
        """A ``Cast`` node is annotated with its explicit target type."""
        analyser = SemanticAnalyser()
        int_const = ast.Constant(ast.ConstInt(5), exp_type=ast.IntType())
        result = analyser._type_check_exp(ast.Cast(ast.LongType(), int_const))
        assert result.exp_type == ast.LongType()

    # ------------------------------------------------------------------
    # FunctionCall
    # ------------------------------------------------------------------

    def test_function_call_annotated_with_return_type(self) -> None:
        """A ``FunctionCall`` is annotated with the callee's declared return type."""
        analyser = SemanticAnalyser()
        analyser._register_fun_declaration(
            ast.Function("foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.LongType()))
        )
        result = analyser._type_check_exp(ast.FunctionCall(ast.Var("foo"), []))
        assert result.exp_type == ast.LongType()

    def test_function_call_int_return(self) -> None:
        """A ``FunctionCall`` to an ``int``-returning function is annotated with ``IntType``."""
        analyser = SemanticAnalyser()
        analyser._register_fun_declaration(
            ast.Function("bar", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
        )
        result = analyser._type_check_exp(ast.FunctionCall(ast.Var("bar"), []))
        assert result.exp_type == ast.IntType()

    # ------------------------------------------------------------------
    # Recursive type-checking of untyped children
    # ------------------------------------------------------------------

    def test_untyped_child_is_typed_recursively(self) -> None:
        """When a child expression has no ``exp_type``, ``_type_check_exp`` recurses to set it.

        Mixed int/long arithmetic wraps the narrower operand in a Cast so both sides share
        the common LongType before the Binary node is returned.
        """
        analyser = SemanticAnalyser()
        # ConstInt with no exp_type - _type_check_exp must annotate it
        untyped_int = ast.Constant(ast.ConstInt(1))
        untyped_long = ast.Constant(ast.ConstLong(2))
        result = analyser._type_check_exp(ast.Binary(ast.Add(), untyped_int, untyped_long))
        assert result.exp_type == ast.LongType()
        # The int operand must be wrapped in a Cast(LongType) for promotion.
        assert isinstance(result.left, ast.Cast)
        assert result.left.target_type == ast.LongType()
        assert result.left.exp.exp_type == ast.IntType()
        # The long operand is already the common type - no Cast needed.
        assert result.right.exp_type == ast.LongType()
