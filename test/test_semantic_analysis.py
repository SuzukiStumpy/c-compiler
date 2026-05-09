"""Tests for the semantic analysis stage.

Verifies that ``SemanticAnalyser`` and the ``resolve`` convenience function
correctly rename variable declarations to unique internal identifiers, reject
semantic violations (duplicate declarations, undeclared variables, invalid
lvalues), and leave non-variable constructs unchanged.

Each test constructs ``ast`` nodes directly so that the semantic analyser is
exercised in isolation, without invoking the lexer or parser.
"""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import SemanticAnalyser, resolve
from compiler.utils.exceptions import SemanticError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _program(body: tuple[ast.BlockItem, ...]) -> ast.Program:
    """Wrap a sequence of block items in a minimal single-function ``ast.Program``.

    Args:
        body: The block items to place in the body of ``main``.

    Returns:
        An ``ast.Program`` containing a single ``main`` function with the
        given body.
    """
    return ast.Program(
        functions=(
            ast.Function(
                return_type=ast.IntType(),
                name="main",
                params=(),
                body=body,
            ),
        )
    )


def _resolved_body(body: tuple[ast.BlockItem, ...]) -> tuple[ast.BlockItem, ...]:
    """Resolve a function body and return the resolved block items.

    Args:
        body: The block items to resolve.

    Returns:
        The resolved block items from the single function in the resolved
        program.
    """
    return resolve(_program(body)).functions[0].body


# ---------------------------------------------------------------------------
# Declaration resolution
# ---------------------------------------------------------------------------


class TestResolveDeclaration:
    """``resolve_declaration`` renames variables and enforces scope rules."""

    def test_declaration_assigns_unique_suffix(self) -> None:
        """The first declaration in a fresh analyser receives the suffix ``.0``."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_declaration(ast.Declaration(name="x", init=None))
        assert result.name == "x.0"

    def test_declaration_preserves_none_init(self) -> None:
        """A declaration without an initialiser keeps ``init=None`` after resolution."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_declaration(ast.Declaration(name="x", init=None))
        assert result.init is None

    def test_declaration_with_constant_init_passes_through(self) -> None:
        """A constant initialiser is returned unchanged."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_declaration(ast.Declaration(name="x", init=ast.Constant(value=5)))
        assert result.init == ast.Constant(value=5)

    def test_declaration_with_var_init_renames_init_var(self) -> None:
        """An initialiser that references an already-declared variable is renamed."""
        analyser = SemanticAnalyser()
        analyser.resolve_declaration(ast.Declaration(name="y", init=None))  # y → y.0
        result = analyser.resolve_declaration(ast.Declaration(name="x", init=ast.Var(name="y")))
        assert result.init == ast.Var(name="y.0")

    def test_two_declarations_receive_distinct_suffixes(self) -> None:
        """Successive declarations in the same analyser get incrementing suffixes."""
        analyser = SemanticAnalyser()
        r1 = analyser.resolve_declaration(ast.Declaration(name="x", init=None))
        r2 = analyser.resolve_declaration(ast.Declaration(name="y", init=None))
        assert r1.name == "x.0"
        assert r2.name == "y.1"

    def test_duplicate_declaration_raises(self) -> None:
        """Declaring the same name twice in the same scope raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        analyser.resolve_declaration(ast.Declaration(name="x", init=None))
        with pytest.raises(SemanticError, match="Duplicate"):
            analyser.resolve_declaration(ast.Declaration(name="x", init=None))

    def test_self_referential_init_raises(self) -> None:
        """``int x = x;`` raises ``SemanticError`` because ``x`` is not yet in scope."""
        analyser = SemanticAnalyser()
        with pytest.raises(SemanticError, match="Undeclared"):
            analyser.resolve_declaration(ast.Declaration(name="x", init=ast.Var(name="x")))


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
            analyser.resolve_declaration(ast.Declaration(name=name, init=None))
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
        assert result == ast.Var(name="x.0")

    def test_undeclared_var_raises(self) -> None:
        """A ``Var`` not present in the symbol table raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        with pytest.raises(SemanticError, match="Undeclared"):
            analyser.resolve_exp(ast.Var(name="undeclared"))

    def test_unary_recurses_into_operand(self) -> None:
        """``Unary`` expressions have their sub-expression resolved."""
        analyser = self._analyser_with("x")
        result = analyser.resolve_exp(ast.Unary(operator=ast.Negate(), exp=ast.Var(name="x")))
        assert result == ast.Unary(operator=ast.Negate(), exp=ast.Var(name="x.0"))

    def test_binary_recurses_into_both_operands(self) -> None:
        """Both sides of a ``Binary`` expression are resolved independently."""
        analyser = self._analyser_with("x", "y")
        result = analyser.resolve_exp(ast.Binary(operator=ast.Add(), left=ast.Var(name="x"), right=ast.Var(name="y")))
        assert result == ast.Binary(operator=ast.Add(), left=ast.Var(name="x.0"), right=ast.Var(name="y.1"))

    def test_assignment_with_var_lvalue_resolves_both_sides(self) -> None:
        """An assignment whose lvalue is a ``Var`` resolves both lvalue and rvalue."""
        analyser = self._analyser_with("x")
        result = analyser.resolve_exp(ast.Assignment(lvalue=ast.Var(name="x"), rvalue=ast.Constant(value=10)))
        assert result == ast.Assignment(lvalue=ast.Var(name="x.0"), rvalue=ast.Constant(value=10))

    def test_assignment_with_non_var_lvalue_raises(self) -> None:
        """An assignment whose lvalue is not a ``Var`` raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        with pytest.raises(SemanticError, match="Invalid lvalue"):
            analyser.resolve_exp(ast.Assignment(lvalue=ast.Constant(value=1), rvalue=ast.Constant(value=2)))

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
            ast.D(declaration=ast.Declaration(name="x", init=None)),
            ast.S(statement=ast.Return(exp=ast.Var(name="x"))),
        )
        resolved = _resolved_body(body)
        assert resolved[1] == ast.S(statement=ast.Return(exp=ast.Var(name="x.0")))

    def test_expression_statement_var_is_renamed(self) -> None:
        """An expression statement containing a variable reference is resolved."""
        body = (
            ast.D(declaration=ast.Declaration(name="x", init=None)),
            ast.S(statement=ast.Expression(exp=ast.Var(name="x"))),
        )
        resolved = _resolved_body(body)
        assert resolved[1] == ast.S(statement=ast.Expression(exp=ast.Var(name="x.0")))

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
        body = (ast.D(declaration=ast.Declaration(name="x", init=None)),)
        resolved = _resolved_body(body)
        assert resolved[0] == ast.D(declaration=ast.Declaration(name="x.0", init=None))

    def test_declaration_and_use_renamed_consistently(self) -> None:
        """The same unique name is used in both the declaration and its subsequent reference."""
        body = (
            ast.D(declaration=ast.Declaration(name="x", init=ast.Constant(value=5))),
            ast.S(statement=ast.Return(exp=ast.Var(name="x"))),
        )
        resolved = _resolved_body(body)
        decl = resolved[0]
        ret = resolved[1]
        assert isinstance(decl, ast.D)
        assert isinstance(ret, ast.S)
        assert decl.declaration.name == "x.0"
        assert ret.statement == ast.Return(exp=ast.Var(name="x.0"))

    def test_assignment_expression_in_statement_resolves(self) -> None:
        """``x = 10;`` after ``int x;`` resolves the lvalue to the unique name."""
        body = (
            ast.D(declaration=ast.Declaration(name="x", init=None)),
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
                    lvalue=ast.Var(name="x.0"),
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
        body = (
            ast.D(declaration=ast.Declaration(name="x", init=None)),
            ast.D(declaration=ast.Declaration(name="x", init=None)),
        )
        with pytest.raises(SemanticError, match="Duplicate"):
            resolve(_program(body))

    def test_undeclared_variable_in_return_raises(self) -> None:
        """Returning an undeclared variable raises ``SemanticError``."""
        body = (ast.S(statement=ast.Return(exp=ast.Var(name="x"))),)
        with pytest.raises(SemanticError, match="Undeclared"):
            resolve(_program(body))

    def test_function_name_and_return_type_preserved(self) -> None:
        """Resolution does not alter the function's name, return type, or params."""
        body = (ast.S(statement=ast.Return(exp=ast.Constant(value=0))),)
        resolved = resolve(_program(body))
        fn = resolved.functions[0]
        assert fn.name == "main"
        assert fn.return_type == ast.IntType()
        assert fn.params == ()


# ---------------------------------------------------------------------------
# Counter isolation between analyser instances
# ---------------------------------------------------------------------------


class TestCounterIsolation:
    """Each ``SemanticAnalyser`` instance starts its counter from zero."""

    def test_two_instances_both_produce_zero_suffix(self) -> None:
        """Independent instances assign ``.0`` to their first declaration."""
        r1 = SemanticAnalyser().resolve_declaration(ast.Declaration(name="x", init=None))
        r2 = SemanticAnalyser().resolve_declaration(ast.Declaration(name="x", init=None))
        assert r1.name == "x.0"
        assert r2.name == "x.0"

    def test_second_call_on_same_instance_increments(self) -> None:
        """A single instance increments its counter across consecutive declarations."""
        analyser = SemanticAnalyser()
        r1 = analyser.resolve_declaration(ast.Declaration(name="a", init=None))
        r2 = analyser.resolve_declaration(ast.Declaration(name="b", init=None))
        assert r1.name == "a.0"
        assert r2.name == "b.1"
