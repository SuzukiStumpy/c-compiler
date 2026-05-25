"""Tests for symbol-table population during semantic analysis.

Verifies that ``resolve_with_symbols`` returns a ``SymbolTable`` whose entries
correctly reflect static-duration variables (file-scope and block-scope
``static``/``extern``) and function declarations.
"""

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve_with_symbols
from compiler.syntax.ast.constants import ConstInt
from compiler.types.symbols import (
    ConstantAttr,
    FunAttr,
    Initial,
    IntInit,
    NoInitializer,
    StaticAttr,
    StringInit,
    Tentative,
)
from test.semantic_analysis.helpers import _file_var, _get_fn, _prog_with_decls, _program


class TestSymbolTablePopulation:
    """``resolve_with_symbols`` returns a symbol table that mirrors the identifier attributes
    established during semantic analysis."""

    # ------------------------------------------------------------------
    # File-scope variables
    # ------------------------------------------------------------------

    def test_file_scope_variable_with_init_produces_initial_entry(self) -> None:
        """A file-scope ``int x = 5;`` produces ``StaticAttr(Initial((IntInit(5),)), globl=True)``."""
        _, symbols = resolve_with_symbols(_prog_with_decls(_file_var("x", ast.Constant(ConstInt(5)))))
        entry = symbols["x"]
        assert isinstance(entry.attrs, StaticAttr)
        assert isinstance(entry.attrs.init, Initial)
        assert entry.attrs.init.initializers == (IntInit(5),)
        assert entry.attrs.globl is True

    def test_file_scope_static_variable_with_init_produces_internal_entry(self) -> None:
        """A ``static int x = 3;`` at file scope produces ``StaticAttr(Initial((IntInit(3),)), globl=False)``."""
        _, symbols = resolve_with_symbols(
            _prog_with_decls(_file_var("x", ast.Constant(ConstInt(3)), sc=ast.StorageClass.Static))
        )
        entry = symbols["x"]
        assert isinstance(entry.attrs, StaticAttr)
        assert isinstance(entry.attrs.init, Initial)
        assert entry.attrs.init.initializers == (IntInit(3),)
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
        _, symbols = resolve_with_symbols(_prog_with_decls(_file_var("x"), _file_var("x", ast.Constant(ConstInt(7)))))
        entry = symbols["x"]
        assert isinstance(entry.attrs, StaticAttr)
        assert isinstance(entry.attrs.init, Initial)
        assert entry.attrs.init.initializers == (IntInit(7),)

    def test_definition_then_tentative_preserves_initial_value(self) -> None:
        """An initial definition followed by a tentative definition keeps ``Initial``."""
        _, symbols = resolve_with_symbols(_prog_with_decls(_file_var("x", ast.Constant(ConstInt(9))), _file_var("x")))
        entry = symbols["x"]
        assert isinstance(entry.attrs, StaticAttr)
        assert isinstance(entry.attrs.init, Initial)
        assert entry.attrs.init.initializers == (IntInit(9),)

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
        """A block-scope ``static int x = 5;`` produces ``StaticAttr(Initial((IntInit(5),)), globl=False)``
        keyed by the unique renamed identifier."""
        body = (
            ast.D(
                ast.VarDecl(
                    ast.VariableDeclaration("x", ast.Constant(ConstInt(5)), ast.IntType(), ast.StorageClass.Static)
                )
            ),
            ast.S(ast.Return(ast.Constant(ConstInt(0)))),
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
        assert entry.attrs.init.initializers == (IntInit(5),)
        assert entry.attrs.globl is False

    def test_block_scope_static_without_init_produces_tentative_entry(self) -> None:
        """A block-scope ``static int x;`` produces ``StaticAttr(Tentative, globl=False)``."""
        body = (
            ast.D(ast.VarDecl(ast.VariableDeclaration("x", None, ast.IntType(), ast.StorageClass.Static))),
            ast.S(ast.Return(ast.Constant(ConstInt(0)))),
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
            ast.D(ast.VarDecl(ast.VariableDeclaration("g", None, ast.IntType(), ast.StorageClass.Extern))),
            ast.S(ast.Return(ast.Constant(ConstInt(0)))),
        )
        _, symbols = resolve_with_symbols(_program(body))
        entry = symbols.get("g")
        assert entry is not None
        assert isinstance(entry.attrs, StaticAttr)
        assert isinstance(entry.attrs.init, NoInitializer)
        assert entry.attrs.globl is True

    def test_file_scope_definition_takes_precedence_over_block_extern(self) -> None:
        """When a file-scope definition ``int g = 4;`` exists and a block-scope ``extern int g;``
        also appears, the symbol table retains ``Initial((IntInit(4),))`` rather than ``NoInitializer``."""
        body = (
            ast.D(ast.VarDecl(ast.VariableDeclaration("g", None, ast.IntType(), ast.StorageClass.Extern))),
            ast.S(ast.Return(ast.Constant(ConstInt(0)))),
        )
        program = _prog_with_decls(
            _file_var("g", ast.Constant(ConstInt(4))),
            ast.FunDecl(
                ast.Function(name="main", params=(), body=body, fun_type=ast.FunType(params=(), ret=ast.IntType()))
            ),
        )
        _, symbols = resolve_with_symbols(program)
        entry = symbols["g"]
        assert isinstance(entry.attrs, StaticAttr)
        assert isinstance(entry.attrs.init, Initial)
        assert entry.attrs.init.initializers == (IntInit(4),)

    # ------------------------------------------------------------------
    # Functions
    # ------------------------------------------------------------------

    def test_function_forward_declaration_produces_not_defined_fun_attr(self) -> None:
        """A forward declaration ``void foo(void);`` produces ``FunAttr(defined=False, globl=True)``."""
        program = _prog_with_decls(
            ast.FunDecl(
                ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
            ),
            ast.FunDecl(
                ast.Function(
                    name="main",
                    params=(),
                    body=(ast.S(ast.Return(ast.Constant(ConstInt(0)))),),
                    fun_type=ast.FunType(params=(), ret=ast.IntType()),
                )
            ),
        )
        _, symbols = resolve_with_symbols(program)
        entry = symbols["foo"]
        assert isinstance(entry.attrs, FunAttr)
        assert entry.attrs.defined is False
        assert entry.attrs.globl is True

    def test_function_definition_produces_defined_fun_attr(self) -> None:
        """A function definition produces ``FunAttr(defined=True, globl=True)``."""
        fn = ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(ast.Constant(ConstInt(0)))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
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
                    body=(ast.S(ast.Return(ast.Constant(ConstInt(0)))),),
                    fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    storage_class=ast.StorageClass.Static,
                )
            ),
            ast.FunDecl(
                ast.Function(
                    name="main",
                    params=(),
                    body=(ast.S(ast.Return(ast.Constant(ConstInt(0)))),),
                    fun_type=ast.FunType(params=(), ret=ast.IntType()),
                )
            ),
        )
        _, symbols = resolve_with_symbols(program)
        entry = symbols["helper"]
        assert isinstance(entry.attrs, FunAttr)
        assert entry.attrs.defined is True
        assert entry.attrs.globl is False

    def test_forward_decl_then_definition_produces_defined_fun_attr(self) -> None:
        """A forward declaration followed by a definition yields ``FunAttr(defined=True)``."""
        program = _prog_with_decls(
            ast.FunDecl(
                ast.Function(name="foo", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
            ),
            ast.FunDecl(
                ast.Function(
                    name="foo",
                    params=(),
                    body=(ast.S(ast.Return(ast.Constant(ConstInt(0)))),),
                    fun_type=ast.FunType(params=(), ret=ast.IntType()),
                )
            ),
            ast.FunDecl(
                ast.Function(
                    name="main",
                    params=(),
                    body=(ast.S(ast.Return(ast.Constant(ConstInt(0)))),),
                    fun_type=ast.FunType(params=(), ret=ast.IntType()),
                )
            ),
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
            ast.FunDecl(
                ast.Function(
                    name="main",
                    params=(),
                    body=(ast.S(ast.Return(ast.Constant(ConstInt(0)))),),
                    fun_type=ast.FunType(params=(), ret=ast.IntType()),
                )
            )
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
                    body=(ast.S(ast.Return(ast.Constant(ConstInt(0)))),),
                    fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    storage_class=ast.StorageClass.Static,
                )
            ),
            ast.FunDecl(
                ast.Function(
                    name="main",
                    params=(),
                    body=(ast.S(ast.Return(ast.Constant(ConstInt(0)))),),
                    fun_type=ast.FunType(params=(), ret=ast.IntType()),
                )
            ),
        )
        resolved, _ = resolve_with_symbols(program)
        helper_fn = _get_fn(resolved, index=0)
        assert helper_fn.globl is False


# ---------------------------------------------------------------------------
# String-literal constant-attr entries
# ---------------------------------------------------------------------------


class TestStringLiteralConstantAttr:
    """When a ``char *`` variable is initialised with a string literal, the analyser creates a
    hidden ``ConstantAttr(StringInit(...))`` symbol-table entry for the string object itself."""

    def _char_ptr_with_string(self, string_bytes: bytes) -> tuple[ast.Program, dict]:
        """Build and resolve a program that declares ``char *s = <literal>;`` at file scope.

        Args:
            string_bytes: The raw byte content of the string literal.

        Returns:
            A ``(resolved_program, symbol_table)`` pair.
        """
        decl = _file_var("s", ast.SingleInit(ast.String(string_bytes)), var_type=ast.Pointer(ast.Char()))
        program = _prog_with_decls(decl)
        resolved, symbols = resolve_with_symbols(program)
        return resolved, symbols

    def test_file_scope_char_ptr_string_creates_constant_attr_entry(self) -> None:
        """``char *s = "hello";`` at file scope produces a hidden ``ConstantAttr`` symbol entry."""
        _, symbols = self._char_ptr_with_string(b"hello")
        hidden = [name for name, entry in symbols.items() if isinstance(entry.attrs, ConstantAttr)]
        assert len(hidden) == 1

    def test_file_scope_char_ptr_string_hidden_name_starts_with_lstr(self) -> None:
        """The hidden string object name starts with ``.Lstr.`` (assembler-local label)."""
        _, symbols = self._char_ptr_with_string(b"hello")
        hidden = [name for name, entry in symbols.items() if isinstance(entry.attrs, ConstantAttr)]
        assert hidden[0].startswith(".Lstr.")

    def test_file_scope_char_ptr_string_constant_attr_holds_string_init(self) -> None:
        """The hidden entry's ``ConstantAttr`` wraps a ``StringInit`` with correct bytes and NUL flag."""
        _, symbols = self._char_ptr_with_string(b"hello")
        hidden_entry = next(entry for entry in symbols.values() if isinstance(entry.attrs, ConstantAttr))
        assert hidden_entry.attrs.init == StringInit(b"hello", True)

    def test_file_scope_char_ptr_string_pointer_var_has_static_attr(self) -> None:
        """The ``char *s`` variable itself gets a ``StaticAttr``, not a ``ConstantAttr``."""
        _, symbols = self._char_ptr_with_string(b"hello")
        assert isinstance(symbols["s"].attrs, StaticAttr)

    def test_file_scope_char_ptr_empty_string_hidden_entry_exists(self) -> None:
        """``char *s = "";`` still creates a hidden ``ConstantAttr`` entry for the empty string."""
        _, symbols = self._char_ptr_with_string(b"")
        hidden = [name for name, entry in symbols.items() if isinstance(entry.attrs, ConstantAttr)]
        assert len(hidden) == 1

    def test_two_string_literals_produce_distinct_hidden_names(self) -> None:
        """Two ``char *`` file-scope variables initialised with string literals get distinct hidden names."""
        decl_a = _file_var("a", ast.SingleInit(ast.String(b"foo")), var_type=ast.Pointer(ast.Char()))
        decl_b = _file_var("b", ast.SingleInit(ast.String(b"bar")), var_type=ast.Pointer(ast.Char()))
        program = _prog_with_decls(decl_a, decl_b)
        _, symbols = resolve_with_symbols(program)
        hidden = [name for name, entry in symbols.items() if isinstance(entry.attrs, ConstantAttr)]
        assert len(hidden) == 2
        assert hidden[0] != hidden[1]

    def test_hidden_string_object_type_is_array_of_char(self) -> None:
        """The hidden symbol's type is ``Array(Char, len+1)``."""
        _, symbols = self._char_ptr_with_string(b"hi")
        hidden_entry = next(entry for entry in symbols.values() if isinstance(entry.attrs, ConstantAttr))
        assert hidden_entry.t == ast.Array(ast.Char(), 3)
