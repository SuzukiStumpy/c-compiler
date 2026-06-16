"""Tests for semantic analysis of union declarations and member-access expressions."""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve, resolve_with_symbols
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import _prog_with_decls


def _union_decl(tag: str, *members: ast.MemberDeclaration) -> ast.UnionDecl:
    """Return a ``UnionDecl`` for testing.

    Args:
        tag: The source-level union tag name.
        *members: Zero or more ``MemberDeclaration`` nodes forming the body.
            An empty list produces a forward declaration.

    Returns:
        An ``ast.UnionDecl`` node.
    """
    return ast.UnionDecl(ast.UnionDeclaration(tag=tag, members=members))


def _member(name: str, member_type: ast.Type) -> ast.MemberDeclaration:
    """Return a ``MemberDeclaration`` with the given name and type.

    Args:
        name: The member name.
        member_type: The declared C type of the member.

    Returns:
        An ``ast.MemberDeclaration`` node.
    """
    return ast.MemberDeclaration(member_name=name, member_type=member_type)


class TestUnionTagResolution:
    """Union declarations produce uniquified internal tags during variable resolution."""

    def test_union_tag_is_uniquified(self) -> None:
        """A union declaration replaces the source tag with a unique internal name."""
        prog = _prog_with_decls(_union_decl("Foo", _member("x", ast.IntType())))
        resolved = resolve(prog)
        resolved_decl = resolved.declarations[0]
        assert isinstance(resolved_decl, ast.UnionDecl)
        tag = resolved_decl.union_declaration.tag
        assert tag != "Foo"
        assert tag.startswith(".U")

    def test_forward_declaration_is_accepted(self) -> None:
        """A bare ``union Tag;`` forward declaration resolves without error."""
        prog = _prog_with_decls(_union_decl("Foo"))
        resolve(prog)  # must not raise

    def test_union_tag_appears_in_struct_type_table(self) -> None:
        """The resolved union tag is present in the type table returned by ``resolve_with_symbols``."""
        prog = _prog_with_decls(_union_decl("Foo", _member("x", ast.IntType())))
        _, _, struct_table, _ = resolve_with_symbols(prog)
        assert len(struct_table) == 1
        tag = next(iter(struct_table))
        assert tag.startswith(".U")


class TestUnionLayout:
    """Union layout: all members at offset 0, size = max member size, align = max member align."""

    def test_single_int_member(self) -> None:
        """A single ``int`` member produces a 4-byte union with 4-byte alignment."""
        prog = _prog_with_decls(_union_decl("U", _member("x", ast.IntType())))
        _, _, struct_table, _ = resolve_with_symbols(prog)
        entry = next(iter(struct_table.values()))
        assert entry.size == 4
        assert entry.alignment == 4
        assert len(entry.members) == 1
        assert entry.members[0].name == "x"
        assert entry.members[0].offset == 0

    def test_two_int_members_both_at_offset_zero(self) -> None:
        """``union { int x; int y; }`` - both members at offset 0, size 4."""
        prog = _prog_with_decls(_union_decl("U", _member("x", ast.IntType()), _member("y", ast.IntType())))
        _, _, struct_table, _ = resolve_with_symbols(prog)
        entry = next(iter(struct_table.values()))
        assert entry.size == 4
        assert entry.alignment == 4
        for m in entry.members:
            assert m.offset == 0

    def test_int_and_long_size_is_long(self) -> None:
        """``union { int x; long y; }`` - size=8, align=8, both at offset 0."""
        prog = _prog_with_decls(_union_decl("U", _member("x", ast.IntType()), _member("y", ast.LongType())))
        _, _, struct_table, _ = resolve_with_symbols(prog)
        entry = next(iter(struct_table.values()))
        assert entry.size == 8
        assert entry.alignment == 8
        for m in entry.members:
            assert m.offset == 0

    def test_trailing_padding_rounds_to_alignment(self) -> None:
        """``union { char c; int x; }`` - size=4 (rounded to int align), align=4."""
        prog = _prog_with_decls(_union_decl("U", _member("c", ast.Char()), _member("x", ast.IntType())))
        _, _, struct_table, _ = resolve_with_symbols(prog)
        entry = next(iter(struct_table.values()))
        assert entry.size == 4
        assert entry.alignment == 4


class TestUnionVariable:
    """Variable declared with a union type gets its tag uniquified."""

    def test_union_variable_type_tag_is_uniquified(self) -> None:
        """A variable declared as ``union Foo u;`` has its tag uniquified."""
        prog = _prog_with_decls(
            _union_decl("Foo", _member("x", ast.IntType())),
            ast.VarDecl(ast.VariableDeclaration("u", None, ast.Union("Foo"))),
        )
        resolved = resolve(prog)
        var_decl = next(d for d in resolved.declarations if isinstance(d, ast.VarDecl))
        assert isinstance(var_decl.variable_declaration.var_type, ast.Union)
        assert var_decl.variable_declaration.var_type.tag.startswith(".U")


class TestUnionErrors:
    """Semantic errors for malformed union declarations."""

    def test_redefinition_of_union_raises(self) -> None:
        """Two complete definitions of the same union tag in the same scope raise ``SemanticError``."""
        prog = _prog_with_decls(
            _union_decl("U", _member("x", ast.IntType())),
            _union_decl("U", _member("y", ast.LongType())),
        )
        with pytest.raises(SemanticError):
            resolve(prog)

    def test_duplicate_member_names_raise(self) -> None:
        """Two members with the same name in a union raise ``SemanticError``."""
        prog = _prog_with_decls(_union_decl("U", _member("x", ast.IntType()), _member("x", ast.LongType())))
        with pytest.raises(SemanticError):
            resolve(prog)
