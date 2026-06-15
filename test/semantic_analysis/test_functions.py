"""Tests for function declaration, parameter, type-checking, linkage, and conflict resolution.

Verifies that ``SemanticAnalyser`` correctly validates function declarations and
definitions, renames parameters, rejects type/linkage conflicts, enforces correct
argument counts, handles function-name shadowing, and records linkage information.
"""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import FunctionBinding, SemanticAnalyser, VariableBinding, resolve
from compiler.syntax.ast.types import Linkage
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import _get_fn, _make_prog, _program, _vd

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
        return ast.Function(name=name, params=(), body=body, fun_type=ast.FunType(params=(), ret=ast.IntType()))

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
        foo_decl = ast.D(
            ast.FunDecl(
                ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
            )
        )
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
        foo_decl = ast.D(
            ast.FunDecl(
                ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
            )
        )
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            foo_decl,
            ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), []))),
        )
        resolved = resolve(_program(body))
        assert len(_get_fn(resolved).body) == 3

    def test_call_to_declared_function_is_valid(self) -> None:
        """A ``FunctionCall`` to a function that has been declared resolves without error."""
        foo_decl = ast.D(
            ast.FunDecl(
                ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
            )
        )
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), []))),
        )
        resolved = resolve(_program(body))
        return_stmt = _get_fn(resolved).body[1].statement
        assert isinstance(return_stmt, ast.Return)
        assert return_stmt.exp == ast.FunctionCall(ast.Var("foo"), [])

    def test_call_to_top_level_defined_function_declared_earlier_is_valid(self) -> None:
        """A ``FunctionCall`` to a function defined earlier in source order resolves without error."""
        foo = ast.Function(
            name="foo",
            params=(),
            body=(ast.S(ast.Return(ast.Constant(0))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        bar = ast.Function(
            name="bar",
            params=(),
            body=(ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), []))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        program = _make_prog(foo, bar)
        resolved = resolve(program)
        return_stmt = _get_fn(resolved, 1).body[0].statement
        assert isinstance(return_stmt, ast.Return)
        assert return_stmt.exp == ast.FunctionCall(ast.Var("foo"), [])

    def test_call_to_top_level_function_defined_later_raises(self) -> None:
        """A ``FunctionCall`` to a function whose declaration has not yet appeared raises ``SemanticError``.

        Functions must be declared or defined before use; there is no implicit
        forward declaration in C.
        """
        bar = ast.Function(
            name="bar",
            params=(),
            body=(ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), []))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        foo = ast.Function(
            name="foo",
            params=(),
            body=(ast.S(ast.Return(ast.Constant(0))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        program = _make_prog(bar, foo)
        with pytest.raises(SemanticError, match="undeclared function"):
            resolve(program)

    def test_call_to_undeclared_function_raises(self) -> None:
        """A ``FunctionCall`` to a name not in the function map raises ``SemanticError``."""
        body: tuple[ast.BlockItem, ...] = (ast.S(ast.Return(ast.FunctionCall(ast.Var("ghost"), []))),)
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
                    fun_type=ast.FunType(params=(), ret=ast.IntType()),
                )
            )
        )
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            _vd("x", ast.Constant(1)),
            ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), [ast.Var("x")]))),
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
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
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
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
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
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        with pytest.raises(SemanticError, match="Duplicate"):
            resolve(_make_prog(fn))

    def test_declaration_body_is_none_skips_resolution(self) -> None:
        """A ``Function`` node with ``body=None`` and no struct types produces an equal resolved function."""
        fn = ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
        resolved = resolve(_make_prog(fn))
        assert _get_fn(resolved) == fn

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
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        main_def = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(ast.Constant(0))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
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
                    fun_type=ast.FunType(params=(), ret=ast.IntType()),
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
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        main_def = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(ast.Constant(0))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        resolved = resolve(_make_prog(foo_decl, main_def))
        assert len(resolved.declarations) == 2

    def test_same_param_name_in_declaration_and_definition_is_valid(self) -> None:
        """The same parameter name may be reused between a declaration and its matching definition."""
        foo_decl = ast.Function(
            name="foo",
            params=(ast.Parameter(param_type=ast.IntType(), name="x"),),
            body=None,
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        foo_def = ast.Function(
            name="foo",
            params=(ast.Parameter(param_type=ast.IntType(), name="x"),),
            body=(ast.S(ast.Return(ast.Var("x"))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
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
                    fun_type=ast.FunType(params=(), ret=ast.IntType()),
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
                    fun_type=ast.FunType(params=(), ret=ast.IntType()),
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
        foo_decl = ast.D(
            ast.FunDecl(
                ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
            )
        )
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            ast.S(ast.Return(ast.Constant(0))),
        )
        resolved = resolve(_program(body))
        assert _get_fn(resolved).body[0] == foo_decl

    def test_block_scope_static_function_declaration_raises(self) -> None:
        """A block-scope function declaration with ``static`` storage class raises ``SemanticError``."""
        foo_decl = ast.D(
            ast.FunDecl(
                ast.Function(
                    name="foo",
                    params=(),
                    body=None,
                    fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    storage_class=ast.StorageClass.Static,
                )
            )
        )
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), []))),
        )
        with pytest.raises(SemanticError, match="static"):
            resolve(_program(body))


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
        foo_decl = ast.Function(
            name="foo", params=(self._int_param("a"),), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType())
        )
        foo_def = ast.Function(
            name="foo",
            params=(self._int_param("a"), self._int_param("b")),
            body=(ast.S(ast.Return(ast.Constant(0))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        main = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(ast.Constant(5))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        with pytest.raises(SemanticError, match="Conflicting declarations"):
            resolve(_make_prog(foo_decl, main, foo_def))

    def test_two_declarations_different_param_count_raises(self) -> None:
        """Two forward declarations of the same function with different arities raise ``SemanticError``."""
        foo_zero = ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
        foo_one = ast.Function(
            name="foo", params=(self._int_param("a"),), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType())
        )
        main = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(ast.Constant(0))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        with pytest.raises(SemanticError, match="Conflicting declarations"):
            resolve(_make_prog(foo_zero, foo_one, main))

    def test_block_level_declaration_conflicts_with_top_level_raises(self) -> None:
        """A block-scope declaration that disagrees with the top-level declaration raises ``SemanticError``."""
        foo_top = ast.Function(
            name="foo", params=(self._int_param("a"),), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType())
        )
        foo_inner = ast.D(
            ast.FunDecl(
                ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
            )
        )
        body: tuple[ast.BlockItem, ...] = (
            foo_inner,
            ast.S(ast.Return(ast.Constant(0))),
        )
        main = ast.Function(name="main", params=(), body=body, fun_type=ast.FunType(params=(), ret=ast.IntType()))
        with pytest.raises(SemanticError, match="Conflicting declarations"):
            resolve(_make_prog(foo_top, main))

    def test_matching_declarations_are_valid(self) -> None:
        """Repeated declarations of the same function with identical parameter types are accepted."""
        foo_decl_a = ast.Function(
            name="foo", params=(self._int_param("a"),), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType())
        )
        foo_decl_b = ast.Function(
            name="foo", params=(self._int_param("x"),), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType())
        )
        foo_def = ast.Function(
            name="foo",
            params=(self._int_param("p"),),
            body=(ast.S(ast.Return(ast.Var("p"))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        main = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(ast.Constant(0))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        resolved = resolve(_make_prog(foo_decl_a, foo_decl_b, foo_def, main))
        assert len(resolved.declarations) == 4

    def test_zero_param_matching_declarations_are_valid(self) -> None:
        """Two forward declarations of the same zero-parameter function are accepted."""
        foo_a = ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
        foo_b = ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
        main = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(ast.Constant(0))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
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
        return ast.D(
            ast.FunDecl(
                ast.Function(name=name, params=params, body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
            )
        )

    def test_too_few_arguments_raises(self) -> None:
        """Calling a one-parameter function with zero arguments raises ``SemanticError``."""
        body: tuple[ast.BlockItem, ...] = (
            self._fn_decl("foo", 1),
            ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), []))),
        )
        with pytest.raises(SemanticError, match="called with 0 argument"):
            resolve(_program(body))

    def test_too_many_arguments_raises(self) -> None:
        """Calling a zero-parameter function with one argument raises ``SemanticError``."""
        body: tuple[ast.BlockItem, ...] = (
            self._fn_decl("foo", 0),
            ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), [ast.Constant(1)]))),
        )
        with pytest.raises(SemanticError, match="called with 1 argument"):
            resolve(_program(body))

    def test_correct_argument_count_is_valid(self) -> None:
        """A call whose argument count matches the declaration resolves without error."""
        body: tuple[ast.BlockItem, ...] = (
            self._fn_decl("foo", 2),
            ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), [ast.Constant(1), ast.Constant(2)]))),
        )
        resolved = resolve(_program(body))
        ret = _get_fn(resolved).body[1].statement
        assert isinstance(ret, ast.Return)
        assert isinstance(ret.exp, ast.FunctionCall)

    def test_zero_argument_call_to_zero_param_function_is_valid(self) -> None:
        """A call with no arguments to a zero-parameter function resolves without error."""
        body: tuple[ast.BlockItem, ...] = (
            self._fn_decl("foo", 0),
            ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), []))),
        )
        resolved = resolve(_program(body))
        assert _get_fn(resolved).body[1] == ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), [])))


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
        foo_top = ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
        body: tuple[ast.BlockItem, ...] = (
            _vd("foo", ast.Constant(3)),
            ast.S(
                statement=ast.Compound(
                    body=(
                        ast.D(
                            ast.FunDecl(
                                ast.Function(
                                    name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType())
                                )
                            )
                        ),
                        ast.S(ast.Expression(ast.FunctionCall(ast.Var("foo"), []))),
                        ast.S(ast.Return(ast.Constant(0))),
                    )
                )
            ),
        )
        main = ast.Function(name="main", params=(), body=body, fun_type=ast.FunType(params=(), ret=ast.IntType()))
        resolved = resolve(_make_prog(foo_top, main))
        assert len(resolved.declarations) == 2

    def test_outer_variable_visible_after_inner_scope_exits(self) -> None:
        """After the inner block exits, the outer variable binding is restored.

        ``return foo;`` after the inner block should resolve to the outer
        variable, not raise "function name used as variable".
        """
        foo_top = ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
        body: tuple[ast.BlockItem, ...] = (
            _vd("foo", ast.Constant(3)),
            ast.S(
                statement=ast.Compound(
                    body=(
                        ast.D(
                            ast.FunDecl(
                                ast.Function(
                                    name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType())
                                )
                            )
                        ),
                        ast.S(ast.Return(ast.Constant(0))),
                    )
                )
            ),
            ast.S(ast.Return(ast.Var("foo"))),
        )
        main = ast.Function(name="main", params=(), body=body, fun_type=ast.FunType(params=(), ret=ast.IntType()))
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
        foo_top = ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
        body: tuple[ast.BlockItem, ...] = (
            _vd("foo", ast.Constant(3)),
            ast.S(
                statement=ast.Compound(
                    body=(
                        ast.D(
                            ast.FunDecl(
                                ast.Function(
                                    name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType())
                                )
                            )
                        ),
                        ast.S(ast.Return(ast.Var("foo"))),
                    )
                )
            ),
        )
        main = ast.Function(name="main", params=(), body=body, fun_type=ast.FunType(params=(), ret=ast.IntType()))
        with pytest.raises(SemanticError, match="Function name 'foo' used as a variable"):
            resolve(_make_prog(foo_top, main))


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
        foo_decl = ast.D(
            ast.FunDecl(
                ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
            )
        )
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
            ast.S(ast.Return(ast.FunctionCall(ast.Var("x"), []))),
        )
        with pytest.raises(SemanticError, match="Variable 'x' called as a function"):
            resolve(_program(body))

    def test_function_name_in_binary_expression_raises(self) -> None:
        """A function name appearing in an arithmetic expression raises ``SemanticError``."""
        foo_decl = ast.D(
            ast.FunDecl(
                ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
            )
        )
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
            body=(ast.S(ast.Return(ast.FunctionCall(ast.Var("p"), []))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        with pytest.raises(SemanticError, match="Variable 'p' called as a function"):
            resolve(_make_prog(fn))

    def test_variable_shadowing_function_name_is_usable_as_variable(self) -> None:
        """A local variable may shadow a function name; the variable is then accessible as a ``Var``.

        Inside the block, ``foo`` refers to the local ``int`` variable.
        After the block exits the function-name binding is visible again.
        """
        foo_decl = ast.D(
            ast.FunDecl(
                ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
            )
        )
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
        foo_decl = ast.D(
            ast.FunDecl(
                ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
            )
        )
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            ast.S(
                statement=ast.Compound(
                    body=(
                        _vd("foo", ast.Constant(0)),
                        ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), []))),
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
        analyser.resolve_variable_declaration(ast.VariableDeclaration(name="x", init=None, var_type=ast.IntType()))
        binding = analyser._variable_map["x"]
        assert isinstance(binding, VariableBinding)
        assert binding.linkage is Linkage.NONE

    def test_function_binding_has_external_linkage(self) -> None:
        """A function declaration carries ``Linkage.EXTERNAL``."""
        analyser = SemanticAnalyser()
        analyser._register_fun_declaration(
            ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
        )
        binding = analyser._function_map["foo"]
        assert isinstance(binding, FunctionBinding)
        assert binding.linkage is Linkage.EXTERNAL

    def test_function_definition_has_external_linkage(self) -> None:
        """A function definition (with a body) also carries ``Linkage.EXTERNAL``."""
        analyser = SemanticAnalyser()
        analyser._register_fun_declaration(
            ast.Function(
                name="bar",
                params=(),
                body=(ast.S(ast.Return(ast.Constant(0))),),
                fun_type=ast.FunType(params=(), ret=ast.IntType()),
            )
        )
        binding = analyser._function_map["bar"]
        assert binding.linkage is Linkage.EXTERNAL

    def test_variable_linkage_preserved_across_scope_entry(self) -> None:
        """``Linkage.NONE`` is preserved on a binding when a new nested scope is entered."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(ast.VariableDeclaration(name="x", init=None, var_type=ast.IntType()))
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
                fun_type=ast.FunType(params=(), ret=ast.IntType()),
            )
        )
        binding = analyser._function_map["foo"]
        assert binding.param_types == (ast.IntType(),)

    def test_function_binding_no_params_records_empty_tuple(self) -> None:
        """A function with no parameters stores an empty ``param_types`` tuple."""
        analyser = SemanticAnalyser()
        analyser._register_fun_declaration(
            ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
        )
        assert analyser._function_map["foo"].param_types == ()


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
        return ast.Function(
            name=name, params=(), body=body, fun_type=ast.FunType(params=(), ret=ast.IntType()), storage_class=sc
        )

    def test_extern_then_static_function_raises(self) -> None:
        """A ``static`` definition after a non-``static`` declaration raises ``SemanticError``.

        ``int foo(void);`` gives ``foo`` external linkage.  The subsequent
        ``static int foo(void) { ... }`` attempts to change it to internal
        linkage, which C forbids.
        """
        program = _make_prog(
            self._fn("foo", None),
            self._fn("foo", (ast.S(ast.Return(ast.Constant(0))),), static=True),
            self._fn("main", (ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), []))),)),
        )
        with pytest.raises(SemanticError, match="linkage"):
            resolve(program)

    def test_static_then_non_static_declaration_is_valid(self) -> None:
        """A non-``static`` declaration after a ``static`` declaration of the same function is valid.

        C11 sec. 6.2.2p5: a function declaration with no storage-class specifier is
        treated exactly as ``extern``.  sec. 6.2.2p4: ``extern`` inherits the linkage
        of the prior visible declaration.  ``int foo(void);`` after
        ``static int foo(void) {...}`` therefore inherits internal linkage and is
        not a conflict.
        """
        program = _make_prog(
            self._fn("foo", (ast.S(ast.Return(ast.Constant(0))),), static=True),
            self._fn("foo", None),
            self._fn("main", (ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), []))),)),
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
            self._fn("main", (ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), []))),)),
        )
        resolved = resolve(program)
        assert len(resolved.declarations) == 3

    def test_block_scope_function_declaration_inherits_internal_linkage(self) -> None:
        """A block-scope function declaration inherits the internal linkage of a prior file-scope declaration.

        C11 sec. 6.2.2p4/p5: the block-scope ``int my_fun(void);`` (treated as
        ``extern``) inherits the internal linkage established by the earlier
        ``static int my_fun(void);`` at file scope.
        """
        my_fun_decl = ast.D(
            ast.FunDecl(
                ast.Function(name="my_fun", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
            )
        )
        caller_fn = ast.Function(
            name="caller",
            params=(),
            body=(
                my_fun_decl,
                ast.S(ast.Return(ast.FunctionCall(ast.Var("my_fun"), []))),
            ),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        program = _make_prog(
            self._fn("my_fun", None, static=True),
            caller_fn,
            self._fn("my_fun", (ast.S(ast.Return(ast.Constant(0))),), static=True),
            self._fn("main", (ast.S(ast.Return(ast.FunctionCall(ast.Var("my_fun"), []))),)),
        )
        resolved = resolve(program)
        assert resolved is not None

    def test_static_function_definition_is_valid(self) -> None:
        """A ``static`` function defined with no prior declaration is accepted."""
        program = _make_prog(
            self._fn("foo", (ast.S(ast.Return(ast.Constant(0))),), static=True),
            self._fn("main", (ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), []))),)),
        )
        resolved = resolve(program)
        assert len(resolved.declarations) == 2

    def test_static_declaration_then_static_definition_is_valid(self) -> None:
        """A ``static`` forward declaration followed by a ``static`` definition is accepted."""
        program = _make_prog(
            self._fn("foo", None, static=True),
            self._fn("foo", (ast.S(ast.Return(ast.Constant(0))),), static=True),
            self._fn("main", (ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), []))),)),
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


class TestVariadicFunctions:
    """Semantic analysis accepts variadic function declarations and relaxes arg-count checks."""

    def _variadic_fn_decl(self, name: str = "foo") -> ast.Function:
        """Return a variadic function declaration ``int name(int x, ...)``."""
        return ast.Function(
            name=name,
            params=(ast.Parameter(param_type=ast.IntType(), name="x"),),
            body=None,
            fun_type=ast.FunType(params=(ast.IntType(),), ret=ast.IntType(), variadic=True),
        )

    def _call(self, fn_name: str, *args: ast.Exp) -> ast.FunctionCall:
        """Return an un-typed ``FunctionCall`` AST node."""
        return ast.FunctionCall(callee=ast.Var(fn_name), args=list(args), exp_type=None)

    def _int_const(self, v: int = 0) -> ast.Constant:
        return ast.Constant(value=ast.ConstInt(v))

    def test_variadic_decl_resolves_without_error(self) -> None:
        """A variadic forward declaration resolves successfully."""
        main = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(self._int_const())),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        program = ast.Program(declarations=(ast.FunDecl(self._variadic_fn_decl()), ast.FunDecl(main)))
        resolve(program)  # must not raise

    def test_variadic_call_with_extra_args_resolves_without_error(self) -> None:
        """Calling a variadic fn with more args than declared params is accepted."""
        decl = self._variadic_fn_decl("printf")
        call_stmt = ast.S(ast.Expression(self._call("printf", self._int_const(1), self._int_const(2))))
        main = ast.Function(
            name="main",
            params=(),
            body=(call_stmt, ast.S(ast.Return(self._int_const()))),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        program = ast.Program(declarations=(ast.FunDecl(decl), ast.FunDecl(main)))
        resolve(program)  # must not raise

    def test_variadic_call_with_exact_args_resolves_without_error(self) -> None:
        """Calling a variadic fn with exactly the declared arg count is also accepted."""
        decl = self._variadic_fn_decl("printf")
        call_stmt = ast.S(ast.Expression(self._call("printf", self._int_const(1))))
        main = ast.Function(
            name="main",
            params=(),
            body=(call_stmt, ast.S(ast.Return(self._int_const()))),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        program = ast.Program(declarations=(ast.FunDecl(decl), ast.FunDecl(main)))
        resolve(program)  # must not raise

    def test_variadic_call_with_too_few_args_raises(self) -> None:
        """Calling a variadic fn with fewer args than declared params raises ``SemanticError``."""
        decl = self._variadic_fn_decl("printf")
        call_stmt = ast.S(ast.Expression(self._call("printf")))
        main = ast.Function(
            name="main",
            params=(),
            body=(call_stmt, ast.S(ast.Return(self._int_const()))),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        program = ast.Program(declarations=(ast.FunDecl(decl), ast.FunDecl(main)))
        with pytest.raises(SemanticError):
            resolve(program)

    def test_non_variadic_call_still_enforces_exact_arg_count(self) -> None:
        """Extra args to a non-variadic fn still raise ``SemanticError``."""
        decl = ast.Function(
            name="foo",
            params=(ast.Parameter(param_type=ast.IntType(), name="x"),),
            body=None,
            fun_type=ast.FunType(params=(ast.IntType(),), ret=ast.IntType(), variadic=False),
        )
        call_stmt = ast.S(ast.Expression(self._call("foo", self._int_const(1), self._int_const(2))))
        main = ast.Function(
            name="main",
            params=(),
            body=(call_stmt, ast.S(ast.Return(self._int_const()))),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        program = ast.Program(declarations=(ast.FunDecl(decl), ast.FunDecl(main)))
        with pytest.raises(SemanticError):
            resolve(program)

    def test_variadic_flag_preserved_in_symbol_table(self) -> None:
        """After resolution, the symbol table entry for a variadic fn has ``FunType.variadic == True``."""
        from compiler.semantic_analysis import SemanticAnalyser

        analyser = SemanticAnalyser()
        analyser._register_fun_declaration(self._variadic_fn_decl("printf"))
        entry = analyser._symbol_table["printf"]
        assert isinstance(entry.t, ast.FunType)
        assert entry.t.variadic is True
