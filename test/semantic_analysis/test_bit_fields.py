"""Tests for semantic analysis of bit-field declarations within structs and unions.

Covers layout computation, packing, type/width validation, and struct-type-table
entries produced by ``resolve_with_symbols``.
"""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve_with_symbols
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import _prog_with_decls


def _struct_decl(tag: str, *members: ast.MemberDeclaration) -> ast.StructDecl:
    """Return a ``StructDecl`` for testing.

    Args:
        tag: The source-level struct tag name.
        *members: Zero or more ``MemberDeclaration`` nodes.

    Returns:
        An ``ast.StructDecl`` node.
    """
    return ast.StructDecl(ast.StructDeclaration(tag=tag, members=members))


def _union_decl(tag: str, *members: ast.MemberDeclaration) -> ast.UnionDecl:
    """Return a ``UnionDecl`` for testing.

    Args:
        tag: The source-level union tag name.
        *members: Zero or more ``MemberDeclaration`` nodes.

    Returns:
        An ``ast.UnionDecl`` node.
    """
    return ast.UnionDecl(ast.UnionDeclaration(tag=tag, members=members))


def _bf(name: str, member_type: ast.Type, width: int) -> ast.MemberDeclaration:
    """Return a named bit-field ``MemberDeclaration``.

    Args:
        name: The member name.
        member_type: The declared C type (e.g. ``ast.IntType()``).
        width: The bit-field width in bits.

    Returns:
        An ``ast.MemberDeclaration`` with ``bit_width`` set.
    """
    return ast.MemberDeclaration(member_name=name, member_type=member_type, bit_width=width)


def _unnamed_bf(member_type: ast.Type, width: int) -> ast.MemberDeclaration:
    """Return an unnamed (padding) bit-field ``MemberDeclaration``.

    Args:
        member_type: The declared C type.
        width: The bit-field width; 0 forces alignment to the next storage unit.

    Returns:
        An ``ast.MemberDeclaration`` with ``member_name=None`` and ``bit_width`` set.
    """
    return ast.MemberDeclaration(member_name=None, member_type=member_type, bit_width=width)


def _member(name: str, member_type: ast.Type) -> ast.MemberDeclaration:
    """Return a regular (non-bit-field) ``MemberDeclaration``.

    Args:
        name: The member name.
        member_type: The declared C type.

    Returns:
        An ``ast.MemberDeclaration`` with no ``bit_width``.
    """
    return ast.MemberDeclaration(member_name=name, member_type=member_type)


def _resolve(decl: ast.Declaration) -> dict:
    """Resolve a single top-level declaration and return the struct-type-table.

    Args:
        decl: A ``StructDecl`` or ``UnionDecl`` to resolve.

    Returns:
        The populated ``StructTypeTable`` mapping resolved tags to ``StructEntry``.
    """
    _, _, struct_table, _ = resolve_with_symbols(_prog_with_decls(decl))
    return struct_table


class TestBitFieldLayout:
    """Bit-field packing and layout in struct declarations."""

    def test_single_named_bit_field_has_bit_offset_zero(self) -> None:
        """A single ``int x : 3;`` field is placed at byte 0, bit 0."""
        table = _resolve(_struct_decl("S", _bf("x", ast.IntType(), 3)))
        entry = next(iter(table.values()))
        assert len(entry.members) == 1
        m = entry.members[0]
        assert m.name == "x"
        assert m.offset == 0
        assert m.bit_offset == 0

    def test_single_named_bit_field_struct_size_is_storage_unit(self) -> None:
        """``struct { int x : 3; }`` occupies a full 4-byte storage unit."""
        table = _resolve(_struct_decl("S", _bf("x", ast.IntType(), 3)))
        entry = next(iter(table.values()))
        assert entry.size == 4
        assert entry.alignment == 4

    def test_two_bit_fields_pack_into_same_unit(self) -> None:
        """``int x : 3; int y : 5;`` packs into the same 4-byte unit."""
        table = _resolve(_struct_decl("S", _bf("x", ast.IntType(), 3), _bf("y", ast.IntType(), 5)))
        entry = next(iter(table.values()))
        assert entry.size == 4
        assert len(entry.members) == 2
        x = next(m for m in entry.members if m.name == "x")
        y = next(m for m in entry.members if m.name == "y")
        assert x.offset == 0 and x.bit_offset == 0
        assert y.offset == 0 and y.bit_offset == 3

    def test_bit_field_overflow_starts_new_unit(self) -> None:
        """Fields that don't fit in the current unit start a new storage unit."""
        # 30 bits used; next 5-bit field doesn't fit in 32-bit unit -> new unit at byte 4
        table = _resolve(
            _struct_decl(
                "S",
                _bf("x", ast.IntType(), 30),
                _bf("y", ast.IntType(), 5),
            )
        )
        entry = next(iter(table.values()))
        assert entry.size == 8
        x = next(m for m in entry.members if m.name == "x")
        y = next(m for m in entry.members if m.name == "y")
        assert x.offset == 0 and x.bit_offset == 0
        assert y.offset == 4 and y.bit_offset == 0

    def test_unnamed_bit_field_not_in_members(self) -> None:
        """Unnamed bit-field padding does not appear in the ``StructEntry.members`` list."""
        table = _resolve(_struct_decl("S", _unnamed_bf(ast.IntType(), 5), _bf("x", ast.IntType(), 3)))
        entry = next(iter(table.values()))
        assert all(m.name != "" for m in entry.members)
        assert len(entry.members) == 1
        m = entry.members[0]
        assert m.name == "x"
        assert m.bit_offset == 5

    def test_zero_width_unnamed_forces_next_unit(self) -> None:
        """``int : 0;`` forces the next field to the start of a new storage unit."""
        table = _resolve(
            _struct_decl(
                "S",
                _bf("x", ast.IntType(), 3),
                _unnamed_bf(ast.IntType(), 0),
                _bf("y", ast.IntType(), 2),
            )
        )
        entry = next(iter(table.values()))
        x = next(m for m in entry.members if m.name == "x")
        y = next(m for m in entry.members if m.name == "y")
        assert x.offset == 0 and x.bit_offset == 0
        assert y.offset == 4 and y.bit_offset == 0

    def test_regular_member_after_bit_fields_is_aligned(self) -> None:
        """A regular member following bit fields is placed after the completed unit, aligned."""
        # struct { int x : 3; long y; }
        # x packs into int unit at offset 0 (4 bytes)
        # y must be at offset 8 (long alignment = 8)
        table = _resolve(_struct_decl("S", _bf("x", ast.IntType(), 3), _member("y", ast.LongType())))
        entry = next(iter(table.values()))
        y = next(m for m in entry.members if m.name == "y")
        assert y.offset == 8
        assert y.bit_offset is None

    def test_bool_bit_field(self) -> None:
        """``_Bool b : 1;`` uses a 1-byte storage unit."""
        table = _resolve(_struct_decl("S", _bf("b", ast.BoolType(), 1)))
        entry = next(iter(table.values()))
        assert entry.size == 1
        assert entry.alignment == 1
        m = entry.members[0]
        assert m.bit_offset == 0


class TestBitFieldValidation:
    """Type and width constraint validation for bit-field declarations."""

    def test_width_exceeds_type_bits_raises(self) -> None:
        """A bit-field width larger than the underlying type's bit-width is rejected."""
        with pytest.raises(SemanticError):
            _resolve(_struct_decl("S", _bf("x", ast.IntType(), 33)))

    def test_invalid_bit_field_type_raises(self) -> None:
        """A bit-field declared with a non-integer type (e.g. ``double``) is rejected."""
        with pytest.raises(SemanticError):
            _resolve(_struct_decl("S", _bf("x", ast.DoubleType(), 3)))

    def test_pointer_bit_field_type_raises(self) -> None:
        """A bit-field declared with a pointer type is rejected."""
        with pytest.raises(SemanticError):
            _resolve(_struct_decl("S", _bf("x", ast.Pointer(ast.IntType()), 3)))


class TestBitFieldInUnion:
    """Bit-field members in union declarations."""

    def test_bit_field_in_union_offset_zero(self) -> None:
        """A bit-field in a union is placed at byte offset 0, bit 0."""
        table = _resolve(_union_decl("U", _bf("x", ast.IntType(), 3)))
        entry = next(iter(table.values()))
        assert len(entry.members) == 1
        m = entry.members[0]
        assert m.offset == 0
        assert m.bit_offset == 0

    def test_union_size_is_full_storage_unit(self) -> None:
        """A union with only bit-fields has size equal to the largest storage unit."""
        table = _resolve(_union_decl("U", _bf("x", ast.IntType(), 3), _bf("y", ast.IntType(), 5)))
        entry = next(iter(table.values()))
        assert entry.size == 4
