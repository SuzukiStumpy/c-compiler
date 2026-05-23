"""Tests for file-scope variable declarations, block-scope storage classes, and cross-kind redeclaration.

Verifies that the semantic analyser correctly validates linkage, enforces
storage class rules for block-scope declarations, and rejects redeclarations
that mix variable and function kinds at the same scope.
"""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import SemanticAnalyser, resolve
from compiler.syntax.ast.constants import ConstDouble, ConstInt
from compiler.syntax.ast.types import Linkage
from compiler.types.symbols import PointerInit, StaticAttr, ULongInit
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import (
    _file_var,
    _get_fn,
    _make_prog,
    _prog_with_decls,
    _program,
    _resolved_body,
    _vd,
)

# ---------------------------------------------------------------------------
# File-scope variable declarations
# ---------------------------------------------------------------------------


class TestFileScopeVariableDeclarations:
    """File-scope variable declarations are validated for linkage and duplicate definitions."""

    def test_file_scope_variable_is_not_renamed(self) -> None:
        """A file-scope variable keeps its original name after resolution."""
        main_fn = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(ast.Var("g"))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        program = _prog_with_decls(_file_var("g", ast.Constant(ConstInt(0))), ast.FunDecl(main_fn))
        resolved = resolve(program)
        var_decl = resolved.declarations[0].variable_declaration
        assert var_decl.name == "g"

    def test_file_scope_variable_visible_in_function_body(self) -> None:
        """A file-scope variable declared before a function is visible inside that function."""
        main_fn = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(ast.Var("g"))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        program = _prog_with_decls(_file_var("g", ast.Constant(ConstInt(0))), ast.FunDecl(main_fn))
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
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        program = _prog_with_decls(ast.FunDecl(main_fn), _file_var("g", ast.Constant(ConstInt(0))))
        with pytest.raises(SemanticError, match="Undeclared"):
            resolve(program)

    def test_static_file_scope_variable_has_internal_linkage(self) -> None:
        """A ``static`` file-scope variable is recorded with ``Linkage.INTERNAL``."""
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(
            ast.VariableDeclaration("x", None, ast.IntType(), ast.StorageClass.Static)
        )
        assert analyser._variable_map["x"].linkage is Linkage.INTERNAL

    def test_extern_file_scope_variable_has_external_linkage(self) -> None:
        """An ``extern`` file-scope variable is recorded with ``Linkage.EXTERNAL``."""
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(
            ast.VariableDeclaration("x", None, ast.IntType(), ast.StorageClass.Extern)
        )
        assert analyser._variable_map["x"].linkage is Linkage.EXTERNAL

    def test_no_storage_class_file_scope_has_external_linkage(self) -> None:
        """A file-scope variable with no storage class specifier has ``Linkage.EXTERNAL``."""
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("x", None, ast.IntType()))
        assert analyser._variable_map["x"].linkage is Linkage.EXTERNAL

    def test_file_scope_variable_after_function_of_same_name_raises(self) -> None:
        """A file-scope variable declaration after a function of the same name raises ``SemanticError``."""
        foo_fn = ast.Function(
            name="foo",
            params=(),
            body=(ast.S(ast.Return(ast.Constant(ConstInt(0)))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        main_fn = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(ast.FunctionCall(ast.Var("foo"), []))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        with pytest.raises(SemanticError, match="redeclared as a variable"):
            resolve(
                _prog_with_decls(ast.FunDecl(foo_fn), ast.FunDecl(main_fn), _file_var("foo", ast.Constant(ConstInt(3))))
            )

    def test_function_after_file_scope_variable_of_same_name_raises(self) -> None:
        """A function declaration after a file-scope variable of the same name raises ``SemanticError``."""
        foo_fn = ast.Function(
            name="foo",
            params=(),
            body=(ast.S(ast.Return(ast.Constant(ConstInt(0)))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        with pytest.raises(SemanticError, match="redeclared as a function"):
            resolve(_prog_with_decls(_file_var("foo", ast.Constant(ConstInt(3))), ast.FunDecl(foo_fn)))

    def test_file_scope_variable_non_constant_init_raises(self) -> None:
        """A file-scope variable with a non-constant initialiser raises ``SemanticError``.

        All file-scope variables have static storage duration, so their
        initialisers must be compile-time constants.  An expression referencing
        another variable (e.g. ``int b = 1 + a``) is therefore invalid.
        """
        main_fn = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(ast.Constant(ConstInt(0)))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        with pytest.raises(SemanticError, match="constant expression"):
            resolve(
                _prog_with_decls(
                    _file_var("a", ast.Constant(ConstInt(10))),
                    _file_var("b", ast.Binary(ast.Add, ast.Constant(ConstInt(1)), ast.Var("a"))),
                    ast.FunDecl(main_fn),
                )
            )

    def test_file_scope_variable_constant_init_is_valid(self) -> None:
        """A file-scope variable with a ``Constant`` initialiser is accepted."""
        analyser = SemanticAnalyser()
        result = analyser._resolve_file_scope_variable_declaration(
            ast.VariableDeclaration("x", ast.Constant(ConstInt(7)), ast.IntType())
        )
        assert result.init == ast.Constant(ConstInt(7))

    def test_file_scope_extern_then_static_raises(self) -> None:
        """A ``static`` file-scope declaration conflicts with a prior ``extern`` declaration of the same name.

        ``int x; static int x;`` changes the linkage from external to internal,
        which is a constraint violation.
        """
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("x", None, ast.IntType()))
        with pytest.raises(SemanticError, match="linkage"):
            analyser._resolve_file_scope_variable_declaration(
                ast.VariableDeclaration("x", None, ast.IntType(), ast.StorageClass.Static)
            )

    def test_file_scope_static_then_extern_raises(self) -> None:
        """A non-``static`` file-scope declaration conflicts with a prior ``static`` declaration of the same name."""
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(
            ast.VariableDeclaration("x", None, ast.IntType(), ast.StorageClass.Static)
        )
        with pytest.raises(SemanticError, match="linkage"):
            analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("x", None, ast.IntType()))

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
                ast.D(ast.VarDecl(ast.VariableDeclaration("x", None, ast.IntType(), ast.StorageClass.Extern))),
                ast.S(ast.Return(ast.Var("x"))),
            ),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        with pytest.raises(SemanticError, match="linkage"):
            resolve(
                _prog_with_decls(
                    ast.FunDecl(main_fn),
                    _file_var("x", ast.Constant(ConstInt(10)), sc=ast.StorageClass.Static),
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
                ast.D(ast.VarDecl(ast.VariableDeclaration("x", None, ast.IntType(), ast.StorageClass.Extern))),
                ast.S(ast.Return(ast.Var("x"))),
            ),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        resolved = resolve(
            _prog_with_decls(
                _file_var("x", ast.Constant(ConstInt(10)), sc=ast.StorageClass.Static),
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
            ast.VariableDeclaration("x", ast.Constant(ConstInt(1)), ast.IntType(), ast.StorageClass.Static)
        )
        result = analyser._resolve_file_scope_variable_declaration(
            ast.VariableDeclaration("x", None, ast.IntType(), ast.StorageClass.Extern)
        )
        assert result.name == "x"
        assert analyser._variable_map["x"].linkage is Linkage.INTERNAL

    def test_two_file_scope_definitions_raises(self) -> None:
        """Two file-scope definitions of the same variable (both with initialisers) raise ``SemanticError``.

        ``int foo = 3; int foo = 4;`` produces two definitions; C permits at
        most one definition per name in a translation unit.
        """
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(
            ast.VariableDeclaration("foo", ast.Constant(ConstInt(3)), ast.IntType())
        )
        with pytest.raises(SemanticError, match="Duplicate definition"):
            analyser._resolve_file_scope_variable_declaration(
                ast.VariableDeclaration("foo", ast.Constant(ConstInt(4)), ast.IntType())
            )

    def test_tentative_then_definition_is_valid(self) -> None:
        """A tentative definition (no initialiser) followed by a definition is accepted."""
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("foo", None, ast.IntType()))
        result = analyser._resolve_file_scope_variable_declaration(
            ast.VariableDeclaration("foo", ast.Constant(ConstInt(4)), ast.IntType())
        )
        assert result.init == ast.Constant(ConstInt(4))

    def test_definition_then_tentative_is_valid(self) -> None:
        """A definition followed by a tentative definition (no initialiser) is accepted."""
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(
            ast.VariableDeclaration("foo", ast.Constant(ConstInt(3)), ast.IntType())
        )
        result = analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("foo", None, ast.IntType()))
        assert result.init is None

    def test_two_tentative_definitions_are_valid(self) -> None:
        """Two tentative definitions (no initialisers) of the same name are permitted."""
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("foo", None, ast.IntType()))
        result = analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("foo", None, ast.IntType()))
        assert result.name == "foo"


# ---------------------------------------------------------------------------
# Block-scope storage class specifiers
# ---------------------------------------------------------------------------


class TestBlockScopeStorageClass:
    """Block-scope ``static`` and ``extern`` declarations are validated and resolved correctly."""

    def test_block_scope_static_is_renamed(self) -> None:
        """A ``static`` local variable receives a unique internal name like any other local."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration("x", None, ast.IntType(), ast.StorageClass.Static)
        )
        assert result.name.startswith(".Vx.")

    def test_block_scope_static_preserves_storage_class(self) -> None:
        """The ``Static`` storage class is preserved on the resolved ``VariableDeclaration``."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration("x", None, ast.IntType(), ast.StorageClass.Static)
        )
        assert result.storage_class is ast.StorageClass.Static

    def test_block_scope_static_has_internal_linkage(self) -> None:
        """A ``static`` local variable is recorded in ``_variable_map`` with ``Linkage.INTERNAL``."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(
            ast.VariableDeclaration("x", None, ast.IntType(), ast.StorageClass.Static)
        )
        assert analyser._variable_map["x"].linkage is Linkage.INTERNAL

    def test_block_scope_static_duplicate_raises(self) -> None:
        """Declaring the same ``static`` local twice in the same block raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(
            ast.VariableDeclaration("x", None, ast.IntType(), ast.StorageClass.Static)
        )
        with pytest.raises(SemanticError, match="Duplicate"):
            analyser.resolve_variable_declaration(
                ast.VariableDeclaration("x", None, ast.IntType(), ast.StorageClass.Static)
            )

    def test_block_scope_extern_is_not_renamed(self) -> None:
        """A block-scope ``extern`` declaration keeps its original source name."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration("g", None, ast.IntType(), ast.StorageClass.Extern)
        )
        assert result.name == "g"

    def test_block_scope_extern_has_external_linkage(self) -> None:
        """A block-scope ``extern`` declaration is recorded with ``Linkage.EXTERNAL``."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(
            ast.VariableDeclaration("g", None, ast.IntType(), ast.StorageClass.Extern)
        )
        assert analyser._variable_map["g"].linkage is Linkage.EXTERNAL

    def test_block_scope_extern_with_init_raises(self) -> None:
        """A block-scope ``extern`` declaration with an initialiser raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        with pytest.raises(SemanticError, match="may not have an initialiser"):
            analyser.resolve_variable_declaration(
                ast.VariableDeclaration("g", ast.Constant(ConstInt(5)), ast.IntType(), ast.StorageClass.Extern)
            )

    def test_block_scope_extern_repeated_is_valid(self) -> None:
        """Two ``extern`` declarations of the same name in the same block are permitted."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(
            ast.VariableDeclaration("g", None, ast.IntType(), ast.StorageClass.Extern)
        )
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration("g", None, ast.IntType(), ast.StorageClass.Extern)
        )
        assert result.name == "g"

    def test_block_scope_extern_after_non_extern_raises(self) -> None:
        """An ``extern`` declaration after a non-``extern`` declaration of the same name raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(ast.VariableDeclaration("x", None, ast.IntType()))
        with pytest.raises(SemanticError, match="previously declared without linkage"):
            analyser.resolve_variable_declaration(
                ast.VariableDeclaration("x", None, ast.IntType(), ast.StorageClass.Extern)
            )

    def test_block_scope_non_extern_after_extern_raises(self) -> None:
        """A non-``extern`` declaration after an ``extern`` declaration of the same name raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(
            ast.VariableDeclaration("x", None, ast.IntType(), ast.StorageClass.Extern)
        )
        with pytest.raises(SemanticError, match="previously declared with linkage"):
            analyser.resolve_variable_declaration(ast.VariableDeclaration("x", None, ast.IntType()))

    def test_block_scope_extern_resolves_in_function_body(self) -> None:
        """A block-scope ``extern`` declaration makes the variable usable inside the function."""
        body: tuple[ast.BlockItem, ...] = (
            ast.D(ast.VarDecl(ast.VariableDeclaration("g", None, ast.IntType(), ast.StorageClass.Extern))),
            ast.S(ast.Return(ast.Var("g"))),
        )
        resolved = _resolved_body(body)
        ret = resolved[1].statement
        assert isinstance(ret, ast.Return)
        assert ret.exp == ast.Var("g")

    def test_block_scope_extern_shadows_outer_local_in_nested_block(self) -> None:
        """``extern int a`` in a nested block resolves ``a`` to the global name, shadowing the outer local.

        This is the key variable-visibility rule tested here: when an inner block
        declares ``extern int a``, all subsequent references to ``a`` *within that
        block* must resolve to the external symbol ``"a"``, not to the renamed local
        ``"a"`` that exists in the enclosing function scope.
        """
        body: tuple[ast.BlockItem, ...] = (
            # int a = 3;  — outer local, will be renamed to something like ".Va.0"
            ast.D(ast.VarDecl(ast.VariableDeclaration("a", ast.Constant(ConstInt(3)), ast.IntType()))),
            # { extern int a;  return a; }
            ast.S(
                ast.Compound(
                    body=(
                        ast.D(ast.VarDecl(ast.VariableDeclaration("a", None, ast.IntType(), ast.StorageClass.Extern))),
                        ast.S(ast.Return(ast.Var("a"))),
                    )
                )
            ),
        )
        resolved = _resolved_body(body)

        # The outer local must have been renamed — its name should not be the bare "a".
        outer_local_name = resolved[0].declaration.variable_declaration.name
        assert outer_local_name != "a", "outer local 'a' was not renamed"

        # The return inside the nested block must use the global "a", not the renamed local.
        compound = resolved[1].statement
        assert isinstance(compound, ast.Compound)
        ret = compound.body[1].statement
        assert isinstance(ret, ast.Return)
        assert ret.exp == ast.Var("a"), (
            f"expected return of global 'a', got {ret.exp!r} — outer local was {outer_local_name!r}"
        )

    def test_block_scope_static_non_constant_init_raises(self) -> None:
        """A ``static`` local variable with a non-constant initialiser raises ``SemanticError``.

        Variables with static storage duration are initialised at program start, so
        their initialisers must be compile-time constants.  A reference to another
        variable (e.g. ``static int b = a * 2``) is therefore invalid.
        """
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(ast.VariableDeclaration("a", ast.Constant(ConstInt(1)), ast.IntType()))
        with pytest.raises(SemanticError, match="constant expression"):
            analyser.resolve_variable_declaration(
                ast.VariableDeclaration(
                    "b",
                    ast.Binary(ast.Multiply, ast.Var("a"), ast.Constant(ConstInt(2))),
                    ast.IntType(),
                    ast.StorageClass.Static,
                )
            )

    def test_block_scope_static_constant_init_is_valid(self) -> None:
        """A ``static`` local variable with a ``Constant`` initialiser is accepted."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration("x", ast.Constant(ConstInt(42)), ast.IntType(), ast.StorageClass.Static)
        )
        assert result.init == ast.Constant(ConstInt(42))

    def test_block_scope_extern_after_static_local_raises(self) -> None:
        """A block-scope ``extern`` after a ``static`` local of the same name raises ``SemanticError``.

        A block-scope ``static`` variable has no linkage (C11 §6.2.2p3 only
        confers internal linkage on *file-scope* ``static``).  A subsequent
        ``extern`` declaration in the same block therefore conflicts, exactly
        as it would with a plain local variable.
        """
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(
            ast.VariableDeclaration("x", ast.Constant(ConstInt(0)), ast.IntType(), ast.StorageClass.Static)
        )
        with pytest.raises(SemanticError, match="without linkage"):
            analyser.resolve_variable_declaration(
                ast.VariableDeclaration("x", None, ast.IntType(), ast.StorageClass.Extern)
            )

    def test_block_scope_static_shadows_file_scope_variable(self) -> None:
        """A ``static`` local variable shadows a file-scope variable of the same name.

        Inside the function the ``static`` local's unique internal name is used;
        the file-scope variable is not accessible under the original name.
        """
        g_decl = _file_var("g", ast.Constant(ConstInt(0)))
        main_fn = ast.Function(
            name="main",
            params=(),
            body=(
                ast.D(
                    ast.VarDecl(
                        ast.VariableDeclaration("g", ast.Constant(ConstInt(1)), ast.IntType(), ast.StorageClass.Static)
                    )
                ),
                ast.S(ast.Return(ast.Var("g"))),
            ),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
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
# Cross-kind redeclaration: function ↔ variable in the same scope
# ---------------------------------------------------------------------------


class TestCrossKindRedeclaration:
    """Declaring the same name as both a function and a variable in one block scope is an error."""

    def test_function_then_variable_same_scope_raises(self) -> None:
        """A variable declaration following a function declaration of the same name raises ``SemanticError``."""
        foo_decl = ast.D(
            ast.FunDecl(
                ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
            )
        )
        body: tuple[ast.BlockItem, ...] = (
            foo_decl,
            _vd("foo"),
            ast.S(ast.Return(ast.Constant(ConstInt(0)))),
        )
        with pytest.raises(SemanticError, match="'foo' redeclared as a variable"):
            resolve(_program(body))

    def test_variable_then_function_same_scope_raises(self) -> None:
        """A function declaration following a variable declaration of the same name raises ``SemanticError``."""
        foo_decl = ast.D(
            ast.FunDecl(
                ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
            )
        )
        body: tuple[ast.BlockItem, ...] = (
            _vd("foo"),
            foo_decl,
            ast.S(ast.Return(ast.Constant(ConstInt(0)))),
        )
        with pytest.raises(SemanticError, match="'foo' redeclared as a function"):
            resolve(_program(body))

    def test_top_level_function_shadowed_by_local_variable_is_valid(self) -> None:
        """A local variable may shadow a top-level function name without error.

        The top-level function is in ``_function_map`` but is not in
        ``_current_block_funs`` for the inner function body (that set is reset
        at the start of each function definition), so no conflict is raised.
        """
        foo_top = ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
        main_body: tuple[ast.BlockItem, ...] = (
            _vd("foo", ast.Constant(ConstInt(0))),
            ast.S(ast.Return(ast.Constant(ConstInt(0)))),
        )
        main_fn = ast.Function(
            name="main", params=(), body=main_body, fun_type=ast.FunType(params=(), ret=ast.IntType())
        )
        resolved = resolve(_make_prog(foo_top, main_fn))
        assert len(resolved.declarations) == 2

    def test_function_in_outer_scope_variable_in_inner_scope_is_valid(self) -> None:
        """A variable in an inner block may shadow a function declared in an outer block.

        The outer block's ``_current_block_funs`` is saved and cleared when the
        inner block is entered, so the inner variable declaration sees no conflict.
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
                        _vd("foo", ast.Constant(ConstInt(0))),
                        ast.S(ast.Return(ast.Constant(ConstInt(0)))),
                    )
                )
            ),
        )
        resolved = resolve(_program(body))
        assert len(_get_fn(resolved).body) == 2

    def test_variable_in_outer_scope_function_in_inner_scope_is_valid(self) -> None:
        """A function declaration in an inner block is independent of a variable in the outer block."""
        foo_decl = ast.D(
            ast.FunDecl(
                ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
            )
        )
        body: tuple[ast.BlockItem, ...] = (
            _vd("foo", ast.Constant(ConstInt(0))),
            ast.S(
                statement=ast.Compound(
                    body=(
                        foo_decl,
                        ast.S(ast.Return(ast.Constant(ConstInt(0)))),
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
        foo_fn = ast.Function(
            name="foo",
            params=(),
            body=(ast.S(ast.Return(ast.Constant(ConstInt(0)))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        main_body: tuple[ast.BlockItem, ...] = (
            ast.D(ast.VarDecl(ast.VariableDeclaration("foo", None, ast.IntType(), ast.StorageClass.Extern))),
            ast.S(ast.Return(ast.Constant(ConstInt(0)))),
        )
        main_fn = ast.Function(
            name="main", params=(), body=main_body, fun_type=ast.FunType(params=(), ret=ast.IntType())
        )
        with pytest.raises(SemanticError, match="redeclared as a variable"):
            resolve(_make_prog(foo_fn, main_fn))

    def test_function_declaration_conflicts_with_prior_extern_variable_raises(self) -> None:
        """A block-scope function declaration whose name was declared as ``extern`` in an outer scope raises.

        An ``extern`` variable binding has external linkage; a function of the same
        name in an inner scope would refer to the same external entity but with a
        conflicting type, which is an error.
        """
        body: tuple[ast.BlockItem, ...] = (
            ast.D(ast.VarDecl(ast.VariableDeclaration("foo", None, ast.IntType(), ast.StorageClass.Extern))),
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
                        ast.S(ast.Return(ast.Constant(ConstInt(0)))),
                    )
                )
            ),
        )
        with pytest.raises(SemanticError, match="redeclared as a function"):
            resolve(_program(body))


# ---------------------------------------------------------------------------
# Static pointer initialisers
# ---------------------------------------------------------------------------


class TestStaticPointerInitialisers:
    """Static variables of pointer type accept null pointer constants and addresses of static objects."""

    # --- file-scope ---

    def test_file_scope_pointer_null_init_stores_ulong_init(self) -> None:
        """A file-scope ``int *`` initialised to ``0`` stores ``ULongInit(0)`` in the symbol table."""
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(
            ast.VariableDeclaration("p", ast.Constant(ConstInt(0)), ast.Pointer(ast.IntType()))
        )
        entry = analyser._symbol_table["p"]
        assert isinstance(entry.attrs, StaticAttr)
        assert entry.attrs.init == ULongInit(0)

    def test_file_scope_pointer_addr_of_static_stores_pointer_init(self) -> None:
        """A file-scope ``int *`` initialised with ``&a`` stores ``PointerInit("a")`` in the symbol table."""
        analyser = SemanticAnalyser()
        analyser._resolve_file_scope_variable_declaration(ast.VariableDeclaration("a", None, ast.IntType()))
        analyser._resolve_file_scope_variable_declaration(
            ast.VariableDeclaration("p", ast.AddrOf(ast.Var("a")), ast.Pointer(ast.IntType()))
        )
        entry = analyser._symbol_table["p"]
        assert isinstance(entry.attrs, StaticAttr)
        assert entry.attrs.init == PointerInit("a")

    def test_file_scope_pointer_non_constant_init_raises(self) -> None:
        """A file-scope pointer initialised with a non-constant, non-address expression raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        with pytest.raises(SemanticError, match="constant"):
            analyser._resolve_file_scope_variable_declaration(
                ast.VariableDeclaration(
                    "p",
                    ast.Binary(ast.Add(), ast.Constant(ConstInt(1)), ast.Constant(ConstInt(2))),
                    ast.Pointer(ast.IntType()),
                )
            )

    def test_file_scope_pointer_double_init_raises(self) -> None:
        """A file-scope pointer initialised with a ``double`` constant raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        with pytest.raises(SemanticError, match="pointer"):
            analyser._resolve_file_scope_variable_declaration(
                ast.VariableDeclaration("p", ast.Constant(ConstDouble(0.0)), ast.Pointer(ast.IntType()))
            )

    def test_file_scope_pointer_addr_of_local_raises(self) -> None:
        """A file-scope pointer initialised with the address of a non-static variable raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        # 'q' is not in the variable map at file scope, so &q cannot be resolved.
        with pytest.raises(SemanticError):
            analyser._resolve_file_scope_variable_declaration(
                ast.VariableDeclaration("p", ast.AddrOf(ast.Var("q")), ast.Pointer(ast.IntType()))
            )

    # --- block-scope static ---

    def test_block_scope_static_pointer_null_init_stores_ulong_init(self) -> None:
        """A block-scope ``static int *`` initialised to ``0`` stores ``ULongInit(0)`` in the symbol table."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration("p", ast.Constant(ConstInt(0)), ast.Pointer(ast.IntType()), ast.StorageClass.Static)
        )
        entry = analyser._symbol_table[result.name]
        assert isinstance(entry.attrs, StaticAttr)
        assert entry.attrs.init == ULongInit(0)

    def test_block_scope_static_pointer_addr_of_static_stores_pointer_init(self) -> None:
        """A block-scope ``static int *`` initialised with ``&a`` (another static) stores ``PointerInit``."""
        analyser = SemanticAnalyser()
        a_result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration("a", None, ast.IntType(), ast.StorageClass.Static)
        )
        p_result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration("p", ast.AddrOf(ast.Var("a")), ast.Pointer(ast.IntType()), ast.StorageClass.Static)
        )
        entry = analyser._symbol_table[p_result.name]
        assert isinstance(entry.attrs, StaticAttr)
        assert entry.attrs.init == PointerInit(a_result.name)

    def test_block_scope_static_pointer_addr_of_local_raises(self) -> None:
        """A block-scope ``static int *`` initialised with the address of a local variable raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        analyser.resolve_variable_declaration(ast.VariableDeclaration("a", None, ast.IntType()))
        with pytest.raises(SemanticError, match="static storage duration"):
            analyser.resolve_variable_declaration(
                ast.VariableDeclaration(
                    "p", ast.AddrOf(ast.Var("a")), ast.Pointer(ast.IntType()), ast.StorageClass.Static
                )
            )

    def test_block_scope_static_pointer_non_constant_init_raises(self) -> None:
        """A block-scope ``static int *`` initialised with a non-constant expression raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        with pytest.raises(SemanticError, match="constant"):
            analyser.resolve_variable_declaration(
                ast.VariableDeclaration(
                    "p",
                    ast.Binary(ast.Add(), ast.Constant(ConstInt(1)), ast.Constant(ConstInt(2))),
                    ast.Pointer(ast.IntType()),
                    ast.StorageClass.Static,
                )
            )
