"""Tests for symbol-table population during semantic analysis.

Verifies that ``resolve_with_symbols`` returns a ``SymbolTable`` whose entries
correctly reflect static-duration variables (file-scope and block-scope
``static``/``extern``) and function declarations.
"""

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve_with_symbols
from compiler.types.symbols import FunAttr, Initial, NoInitializer, StaticAttr, Tentative
from test.semantic_analysis.helpers import _file_var, _get_fn, _prog_with_decls, _program


class TestSymbolTablePopulation:
    """``resolve_with_symbols`` returns a symbol table that mirrors the identifier attributes
    established during semantic analysis."""

    # ------------------------------------------------------------------
    # File-scope variables
    # ------------------------------------------------------------------

    def test_file_scope_variable_with_init_produces_initial_entry(self) -> None:
        """A file-scope ``int x = 5;`` produces ``StaticAttr(Initial(5), globl=True)``."""
        _, symbols = resolve_with_symbols(_prog_with_decls(_file_var("x", ast.Constant(5))))
        entry = symbols["x"]
        assert isinstance(entry.attrs, StaticAttr)
        assert isinstance(entry.attrs.init, Initial)
        assert entry.attrs.init.value == 5
        assert entry.attrs.globl is True

    def test_file_scope_static_variable_with_init_produces_internal_entry(self) -> None:
        """A ``static int x = 3;`` at file scope produces ``StaticAttr(Initial(3), globl=False)``."""
        _, symbols = resolve_with_symbols(_prog_with_decls(_file_var("x", ast.Constant(3), sc=ast.StorageClass.Static)))
        entry = symbols["x"]
        assert isinstance(entry.attrs, StaticAttr)
        assert isinstance(entry.attrs.init, Initial)
        assert entry.attrs.init.value == 3
        assert entry.attrs.globl is False

    def test_file_scope_tentative_variable_produces_tentative_entry(self) -> None:
        """A tentative definition ``int x;`` at file scope produces ``StaticAttr(Tentative, globl=True)``."""
        _, symbols = resolve_with_symbols(_prog_with_decls(_file_var("x")))
        entry = symbols["x"]
        assert isinstance(entry.attrs, StaticAttr)
        assert isinstance(entry.attrs.init, Tentative)
        assert entry.attrs.globl is True

    def test_file_scope_extern_without_init_produces_no_initializer_entry(self) -> None:
        """A non-defining ``extern int x;`` at file scope produces ``StaticAttr(NoInitializer, globl=True)``."""
        _, symbols = resolve_with_symbols(_prog_with_decls(_file_var("x", sc=ast.StorageClass.Extern)))
        entry = symbols["x"]
        assert isinstance(entry.attrs, StaticAttr)
        assert isinstance(entry.attrs.init, NoInitializer)
        assert entry.attrs.globl is True

    def test_tentative_then_definition_yields_initial_value(self) -> None:
        """A tentative definition followed by a concrete definition resolves to ``Initial``."""
        _, symbols = resolve_with_symbols(_prog_with_decls(_file_var("x"), _file_var("x", ast.Constant(7))))
        entry = symbols["x"]
        assert isinstance(entry.attrs, StaticAttr)
        assert isinstance(entry.attrs.init, Initial)
        assert entry.attrs.init.value == 7

    def test_definition_then_tentative_preserves_initial_value(self) -> None:
        """An initial definition followed by a tentative definition keeps ``Initial``."""
        _, symbols = resolve_with_symbols(_prog_with_decls(_file_var("x", ast.Constant(9)), _file_var("x")))
        entry = symbols["x"]
        assert isinstance(entry.attrs, StaticAttr)
        assert isinstance(entry.attrs.init, Initial)
        assert entry.attrs.init.value == 9

    def test_extern_then_tentative_yields_tentative(self) -> None:
        """A non-defining ``extern`` followed by a tentative definition upgrades to ``Tentative``."""
        _, symbols = resolve_with_symbols(_prog_with_decls(_file_var("x", sc=ast.StorageClass.Extern), _file_var("x")))
        entry = symbols["x"]
        assert isinstance(entry.attrs, StaticAttr)
        assert isinstance(entry.attrs.init, Tentative)

    # ------------------------------------------------------------------
    # Block-scope static / extern
    # ------------------------------------------------------------------

    def test_block_scope_static_with_init_produces_initial_entry(self) -> None:
        """A block-scope ``static int x = 5;`` produces ``StaticAttr(Initial(5), globl=False)``
        keyed by the unique renamed identifier."""
        body = (
            ast.D(ast.VarDecl(ast.VariableDeclaration("x", ast.Constant(5), ast.StorageClass.Static))),
            ast.S(ast.Return(ast.Constant(0))),
        )
        _, symbols = resolve_with_symbols(_program(body))
        static_entries = {
            name: e
            for name, e in symbols.items()
            if isinstance(e.attrs, StaticAttr) and not isinstance(e.attrs.init, NoInitializer)
        }
        assert len(static_entries) == 1
        name, entry = next(iter(static_entries.items()))
        assert name.startswith(".Vx.")
        assert isinstance(entry.attrs.init, Initial)
        assert entry.attrs.init.value == 5
        assert entry.attrs.globl is False

    def test_block_scope_static_without_init_produces_tentative_entry(self) -> None:
        """A block-scope ``static int x;`` produces ``StaticAttr(Tentative, globl=False)``."""
        body = (
            ast.D(ast.VarDecl(ast.VariableDeclaration("x", None, ast.StorageClass.Static))),
            ast.S(ast.Return(ast.Constant(0))),
        )
        _, symbols = resolve_with_symbols(_program(body))
        static_entries = {name: e for name, e in symbols.items() if isinstance(e.attrs, StaticAttr)}
        assert len(static_entries) == 1
        _, entry = next(iter(static_entries.items()))
        assert isinstance(entry.attrs.init, Tentative)
        assert entry.attrs.globl is False

    def test_block_scope_extern_produces_no_initializer_entry(self) -> None:
        """A block-scope ``extern int x;`` produces ``StaticAttr(NoInitializer, globl=True)``
        keyed by the original (unrenamed) name."""
        body = (
            ast.D(ast.VarDecl(ast.VariableDeclaration("g", None, ast.StorageClass.Extern))),
            ast.S(ast.Return(ast.Constant(0))),
        )
        _, symbols = resolve_with_symbols(_program(body))
        entry = symbols.get("g")
        assert entry is not None
        assert isinstance(entry.attrs, StaticAttr)
        assert isinstance(entry.attrs.init, NoInitializer)
        assert entry.attrs.globl is True

    def test_file_scope_definition_takes_precedence_over_block_extern(self) -> None:
        """When a file-scope definition ``int g = 4;`` exists and a block-scope ``extern int g;``
        also appears, the symbol table retains ``Initial(4)`` rather than ``NoInitializer``."""
        body = (
            ast.D(ast.VarDecl(ast.VariableDeclaration("g", None, ast.StorageClass.Extern))),
            ast.S(ast.Return(ast.Constant(0))),
        )
        program = _prog_with_decls(
            _file_var("g", ast.Constant(4)),
            ast.FunDecl(ast.Function(name="main", params=(), body=body)),
        )
        _, symbols = resolve_with_symbols(program)
        entry = symbols["g"]
        assert isinstance(entry.attrs, StaticAttr)
        assert isinstance(entry.attrs.init, Initial)
        assert entry.attrs.init.value == 4

    # ------------------------------------------------------------------
    # Functions
    # ------------------------------------------------------------------

    def test_function_forward_declaration_produces_not_defined_fun_attr(self) -> None:
        """A forward declaration ``void foo(void);`` produces ``FunAttr(defined=False, globl=True)``."""
        program = _prog_with_decls(
            ast.FunDecl(ast.Function(name="foo", params=(), body=None)),
            ast.FunDecl(ast.Function(name="main", params=(), body=(ast.S(ast.Return(ast.Constant(0))),))),
        )
        _, symbols = resolve_with_symbols(program)
        entry = symbols["foo"]
        assert isinstance(entry.attrs, FunAttr)
        assert entry.attrs.defined is False
        assert entry.attrs.globl is True

    def test_function_definition_produces_defined_fun_attr(self) -> None:
        """A function definition produces ``FunAttr(defined=True, globl=True)``."""
        fn = ast.Function(name="main", params=(), body=(ast.S(ast.Return(ast.Constant(0))),))
        _, symbols = resolve_with_symbols(_prog_with_decls(ast.FunDecl(fn)))
        entry = symbols["main"]
        assert isinstance(entry.attrs, FunAttr)
        assert entry.attrs.defined is True
        assert entry.attrs.globl is True

    def test_static_function_produces_internal_fun_attr(self) -> None:
        """A ``static`` function definition produces ``FunAttr(defined=True, globl=False)``."""
        program = _prog_with_decls(
            ast.FunDecl(
                ast.Function(
                    name="helper",
                    params=(),
                    body=(ast.S(ast.Return(ast.Constant(0))),),
                    storage_class=ast.StorageClass.Static,
                )
            ),
            ast.FunDecl(ast.Function(name="main", params=(), body=(ast.S(ast.Return(ast.Constant(0))),))),
        )
        _, symbols = resolve_with_symbols(program)
        entry = symbols["helper"]
        assert isinstance(entry.attrs, FunAttr)
        assert entry.attrs.defined is True
        assert entry.attrs.globl is False

    def test_forward_decl_then_definition_produces_defined_fun_attr(self) -> None:
        """A forward declaration followed by a definition yields ``FunAttr(defined=True)``."""
        program = _prog_with_decls(
            ast.FunDecl(ast.Function(name="foo", params=(), body=None)),
            ast.FunDecl(ast.Function(name="foo", params=(), body=(ast.S(ast.Return(ast.Constant(0))),))),
            ast.FunDecl(ast.Function(name="main", params=(), body=(ast.S(ast.Return(ast.Constant(0))),))),
        )
        _, symbols = resolve_with_symbols(program)
        entry = symbols["foo"]
        assert isinstance(entry.attrs, FunAttr)
        assert entry.attrs.defined is True

    # ------------------------------------------------------------------
    # ast.Function.globl propagation
    # ------------------------------------------------------------------

    def test_resolved_external_function_has_globl_true(self) -> None:
        """A function without ``static`` is resolved with ``globl=True`` on its ``ast.Function`` node."""
        program = _prog_with_decls(
            ast.FunDecl(ast.Function(name="main", params=(), body=(ast.S(ast.Return(ast.Constant(0))),)))
        )
        resolved, _ = resolve_with_symbols(program)
        fn = _get_fn(resolved)
        assert fn.globl is True

    def test_resolved_static_function_has_globl_false(self) -> None:
        """A ``static`` function is resolved with ``globl=False`` on its ``ast.Function`` node."""
        program = _prog_with_decls(
            ast.FunDecl(
                ast.Function(
                    name="helper",
                    params=(),
                    body=(ast.S(ast.Return(ast.Constant(0))),),
                    storage_class=ast.StorageClass.Static,
                )
            ),
            ast.FunDecl(ast.Function(name="main", params=(), body=(ast.S(ast.Return(ast.Constant(0))),))),
        )
        resolved, _ = resolve_with_symbols(program)
        helper_fn = _get_fn(resolved, index=0)
        assert helper_fn.globl is False
