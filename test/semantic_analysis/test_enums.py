"""Tests for semantic analysis of enum declarations and enumerator constant references."""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve, resolve_with_symbols
from compiler.syntax.ast.declarations import EnumDeclaration, EnumeratorDeclaration
from compiler.types.symbols import EnumConstAttr, EnumEntry
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import _prog_with_decls

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enum_decl(tag: str, *members: tuple[str, int]) -> ast.EnumDecl:
    """Return an ``EnumDecl`` AST node for testing.

    Args:
        tag: The source-level enum tag name.
        *members: Zero or more ``(name, value)`` pairs forming the enumerator list.

    Returns:
        An ``ast.EnumDecl`` node.
    """
    enumerators = tuple(EnumeratorDeclaration(name=n, value=v) for n, v in members)
    return ast.EnumDecl(EnumDeclaration(tag=tag, members=enumerators))


def _var_decl_enum(name: str, tag: str) -> ast.VarDecl:
    """Return a ``VarDecl`` with an ``EnumType`` for testing.

    Args:
        name: The variable name.
        tag: The source-level enum tag used as the variable type.

    Returns:
        An ``ast.VarDecl`` whose ``var_type`` is ``EnumType(tag)``.
    """
    return ast.VarDecl(ast.VariableDeclaration(name, None, ast.EnumType(tag)))


# ---------------------------------------------------------------------------
# Enum tag resolution
# ---------------------------------------------------------------------------


class TestEnumTagResolution:
    """Enum declarations produce uniquified internal tags during resolution."""

    def test_enum_tag_is_uniquified(self) -> None:
        """A resolved enum tag starts with ``.E`` and is not the source tag."""
        prog = _prog_with_decls(_enum_decl("Color", ("RED", 0), ("GREEN", 1)))
        resolved = resolve(prog)
        decl = resolved.declarations[0]
        assert isinstance(decl, ast.EnumDecl)
        assert decl.enum_declaration.tag != "Color"
        assert decl.enum_declaration.tag is not None
        assert decl.enum_declaration.tag.startswith(".E")

    def test_enum_appears_in_enum_type_table(self) -> None:
        """The resolved enum tag is present in the enum type table."""
        prog = _prog_with_decls(_enum_decl("E", ("A", 0)))
        _, _, _, enum_table = resolve_with_symbols(prog)
        assert len(enum_table) == 1

    def test_enum_entry_contains_member_names(self) -> None:
        """The ``EnumEntry`` in the type table lists enumerator names."""
        prog = _prog_with_decls(_enum_decl("Color", ("RED", 0), ("GREEN", 1), ("BLUE", 2)))
        _, _, _, enum_table = resolve_with_symbols(prog)
        entry = next(iter(enum_table.values()))
        assert isinstance(entry, EnumEntry)
        assert entry.members == ("RED", "GREEN", "BLUE")


# ---------------------------------------------------------------------------
# Enumerator constants in symbol table
# ---------------------------------------------------------------------------


class TestEnumConstantSymbols:
    """Each enumerator is registered in the symbol table with ``EnumConstAttr``."""

    def test_enumerator_has_enum_const_attr(self) -> None:
        """``RED`` is registered with ``EnumConstAttr(0)`` in the symbol table."""
        prog = _prog_with_decls(_enum_decl("Color", ("RED", 0), ("GREEN", 1)))
        _, sym_table, _, _ = resolve_with_symbols(prog)
        assert "RED" in sym_table
        assert isinstance(sym_table["RED"].attrs, EnumConstAttr)
        assert sym_table["RED"].attrs.value == 0

    def test_all_enumerators_registered(self) -> None:
        """All three members of ``Color`` appear in the symbol table."""
        prog = _prog_with_decls(_enum_decl("Color", ("RED", 0), ("GREEN", 1), ("BLUE", 2)))
        _, sym_table, _, _ = resolve_with_symbols(prog)
        for name, expected_val in [("RED", 0), ("GREEN", 1), ("BLUE", 2)]:
            assert name in sym_table
            assert sym_table[name].attrs == EnumConstAttr(expected_val)


# ---------------------------------------------------------------------------
# Enumerator constant references folded to integer constants
# ---------------------------------------------------------------------------


class TestEnumConstantFolding:
    """References to enumerator names are replaced by ``Constant(ConstInt(value))``."""

    def test_var_reference_to_enumerator_folds_to_constant(self) -> None:
        """Using ``RED`` in an expression yields ``Constant(ConstInt(0))``."""
        prog = ast.Program(
            declarations=(
                _enum_decl("Color", ("RED", 0), ("GREEN", 1)),
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=(ast.S(ast.Return(exp=ast.Var(name="RED"))),),
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )
        resolved = resolve(prog)
        fn = next(d.function_declaration for d in resolved.declarations if isinstance(d, ast.FunDecl))
        assert fn.body is not None
        ret_stmt = fn.body[0]
        assert isinstance(ret_stmt, ast.S)
        assert isinstance(ret_stmt.statement, ast.Return)
        folded = ret_stmt.statement.exp
        assert isinstance(folded, ast.Constant)
        assert isinstance(folded.value, ast.ConstInt)
        assert folded.value.val == 0

    def test_enumerator_with_nonzero_value_folds_correctly(self) -> None:
        """``GREEN`` (value 1) is folded to ``Constant(ConstInt(1))``."""
        prog = ast.Program(
            declarations=(
                _enum_decl("Color", ("RED", 0), ("GREEN", 1)),
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=(ast.S(ast.Return(exp=ast.Var(name="GREEN"))),),
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )
        resolved = resolve(prog)
        fn = next(d.function_declaration for d in resolved.declarations if isinstance(d, ast.FunDecl))
        assert fn.body is not None
        folded = fn.body[0].statement.exp  # type: ignore[union-attr]
        assert isinstance(folded, ast.Constant)
        assert folded.value == ast.ConstInt(1)


# ---------------------------------------------------------------------------
# Enum-typed variables
# ---------------------------------------------------------------------------


class TestEnumTypedVariables:
    """Variables declared with an ``enum`` type are resolved correctly."""

    def test_enum_typed_variable_resolves(self) -> None:
        """``enum Color x;`` after a ``Color`` definition resolves without error."""
        prog = _prog_with_decls(
            _enum_decl("Color", ("RED", 0)),
            _var_decl_enum("x", "Color"),
        )
        resolve(prog)  # must not raise

    def test_enum_type_resolves_tag(self) -> None:
        """The ``EnumType`` on the resolved ``VarDecl`` carries a unique internal tag."""
        prog = _prog_with_decls(
            _enum_decl("Color", ("RED", 0)),
            _var_decl_enum("x", "Color"),
        )
        resolved = resolve(prog)
        var = next(d for d in resolved.declarations if isinstance(d, ast.VarDecl))
        assert isinstance(var.variable_declaration.var_type, ast.EnumType)
        tag = var.variable_declaration.var_type.tag
        assert tag is not None
        assert tag.startswith(".E")

    def test_undeclared_enum_tag_raises(self) -> None:
        """Using an undeclared enum tag raises ``SemanticError``."""
        prog = _prog_with_decls(_var_decl_enum("x", "Undeclared"))
        with pytest.raises(SemanticError):
            resolve(prog)


# ---------------------------------------------------------------------------
# Duplicate / conflict checks
# ---------------------------------------------------------------------------


class TestEnumTagConflicts:
    """Enum tags may not conflict with struct or union tags in the same scope."""

    def test_enum_tag_conflicts_with_struct_tag_raises(self) -> None:
        """Declaring ``struct Foo`` and ``enum Foo`` in the same scope raises."""
        prog = _prog_with_decls(
            ast.StructDecl(
                ast.StructDeclaration(
                    tag="Foo", members=(ast.MemberDeclaration(member_name="x", member_type=ast.IntType()),)
                )
            ),
            _enum_decl("Foo", ("A", 0)),
        )
        with pytest.raises(SemanticError):
            resolve(prog)

    def test_duplicate_enumerator_name_raises(self) -> None:
        """Two enumerators with the same name in one enum definition raises."""
        prog = _prog_with_decls(_enum_decl("E", ("A", 0), ("A", 1)))
        with pytest.raises(SemanticError):
            resolve(prog)
