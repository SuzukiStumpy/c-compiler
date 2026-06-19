"""Tests for semantic analysis of struct declarations and member-access expressions."""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve, resolve_with_symbols
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import _prog_with_decls


def _struct_decl(tag: str, *members: ast.MemberDeclaration) -> ast.StructDecl:
    """Return a ``StructDecl`` for testing.

    Args:
        tag: The source-level struct tag name.
        *members: Zero or more ``MemberDeclaration`` nodes forming the body.
            An empty list produces a forward declaration.

    Returns:
        An ``ast.StructDecl`` node.
    """
    return ast.StructDecl(ast.StructDeclaration(tag=tag, members=members))


def _member(name: str, member_type: ast.Type) -> ast.MemberDeclaration:
    """Return a ``MemberDeclaration`` with the given name and type.

    Args:
        name: The member name.
        member_type: The declared C type of the member.

    Returns:
        An ``ast.MemberDeclaration`` node.
    """
    return ast.MemberDeclaration(member_name=name, member_type=member_type)


class TestStructTagResolution:
    """Struct declarations produce uniquified internal tags during variable resolution."""

    def test_struct_tag_is_uniquified(self) -> None:
        """A struct declaration replaces the source tag with a unique internal name."""
        prog = _prog_with_decls(_struct_decl("Foo", _member("x", ast.IntType())))
        resolved = resolve(prog)
        resolved_decl = resolved.declarations[0]
        assert isinstance(resolved_decl, ast.StructDecl)
        tag = resolved_decl.struct_declaration.tag
        assert tag != "Foo"
        assert tag.startswith(".S")

    def test_forward_declaration_is_accepted(self) -> None:
        """A bare ``struct Tag;`` forward declaration resolves without error."""
        prog = _prog_with_decls(_struct_decl("Foo"))
        resolve(prog)  # must not raise

    def test_struct_tag_appears_in_struct_type_table(self) -> None:
        """The resolved struct tag is present in the ``StructTypeTable``."""
        prog = _prog_with_decls(_struct_decl("Foo", _member("x", ast.IntType())))
        _, _, struct_table, _ = resolve_with_symbols(prog)
        assert len(struct_table) == 1
        tag = next(iter(struct_table))
        assert tag.startswith(".S")

    def test_struct_layout_single_int_member(self) -> None:
        """A single ``int`` member produces a 4-byte struct with 4-byte alignment."""
        prog = _prog_with_decls(_struct_decl("S", _member("x", ast.IntType())))
        _, _, struct_table, _ = resolve_with_symbols(prog)
        entry = next(iter(struct_table.values()))
        assert entry.size == 4
        assert entry.alignment == 4
        assert len(entry.members) == 1
        assert entry.members[0].name == "x"
        assert entry.members[0].offset == 0

    def test_struct_layout_int_then_long_has_padding(self) -> None:
        """``struct { int x; long y; }`` pads ``x`` to align ``y`` at offset 8."""
        prog = _prog_with_decls(_struct_decl("S", _member("x", ast.IntType()), _member("y", ast.LongType())))
        _, _, struct_table, _ = resolve_with_symbols(prog)
        entry = next(iter(struct_table.values()))
        x_entry = next(m for m in entry.members if m.name == "x")
        y_entry = next(m for m in entry.members if m.name == "y")
        assert x_entry.offset == 0
        assert y_entry.offset == 8
        assert entry.size == 16
        assert entry.alignment == 8

    def test_struct_variable_type_is_resolved(self) -> None:
        """A variable declared as ``struct Foo s;`` has its type tag uniquified."""
        prog = _prog_with_decls(
            _struct_decl("Foo", _member("x", ast.IntType())),
            ast.VarDecl(ast.VariableDeclaration("s", None, ast.Structure("Foo"))),
        )
        resolved = resolve(prog)
        var_decl = next(d for d in resolved.declarations if isinstance(d, ast.VarDecl))
        assert isinstance(var_decl.variable_declaration.var_type, ast.Structure)
        assert var_decl.variable_declaration.var_type.tag.startswith(".S")

    def test_redefinition_of_struct_raises(self) -> None:
        """Two complete definitions of the same struct tag in the same scope raise ``SemanticError``."""
        prog = _prog_with_decls(
            _struct_decl("Foo", _member("x", ast.IntType())),
            _struct_decl("Foo", _member("y", ast.IntType())),
        )
        with pytest.raises(SemanticError):
            resolve(prog)

    def test_undeclared_struct_tag_raises(self) -> None:
        """Using an undeclared struct tag in a variable type raises ``SemanticError``."""
        prog = _prog_with_decls(
            ast.VarDecl(ast.VariableDeclaration("s", None, ast.Structure("Undeclared"))),
        )
        with pytest.raises(SemanticError):
            resolve(prog)


class TestStructMemberValidation:
    """Semantic analysis rejects invalid struct member declarations."""

    def test_duplicate_member_name_raises(self) -> None:
        """Two members with the same name in one struct raise ``SemanticError``."""
        prog = _prog_with_decls(_struct_decl("Foo", _member("x", ast.IntType()), _member("x", ast.LongType())))
        with pytest.raises(SemanticError):
            resolve(prog)

    def test_void_member_raises(self) -> None:
        """A member of type ``void`` raises ``SemanticError``."""
        prog = _prog_with_decls(_struct_decl("Foo", _member("v", ast.VoidType())))
        with pytest.raises(SemanticError):
            resolve(prog)


class TestDotTypeChecking:
    """``Dot`` expressions have the member's type and obey lvalue rules."""

    def _dot_program(self) -> ast.Program:
        """Build a minimal program containing a struct variable and a dot-access return.

        Returns:
            A program with ``struct S { int x; }; int main(void) { struct S s; return s.x; }``.
        """
        struct_type = ast.Structure("S")
        s_decl = ast.D(ast.VarDecl(ast.VariableDeclaration("s", None, struct_type)))
        dot_exp = ast.Dot(
            structure=ast.Var("s", exp_type=None),
            member="x",
        )
        return ast.Program(
            declarations=(
                _struct_decl("S", _member("x", ast.IntType())),
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=(s_decl, ast.S(ast.Return(exp=dot_exp))),
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )

    def test_dot_result_type_is_member_type(self) -> None:
        """After type-checking, the ``Dot`` expression carries the member's type."""
        resolved = resolve(self._dot_program())
        main_fn = next(d.function_declaration for d in resolved.declarations if isinstance(d, ast.FunDecl))
        ret_stmt = next(
            item.statement
            for item in main_fn.body
            if isinstance(item, ast.S) and isinstance(item.statement, ast.Return)
        )
        assert isinstance(ret_stmt, ast.Return)
        assert isinstance(ret_stmt.exp, ast.Dot)
        assert ret_stmt.exp.exp_type == ast.IntType()

    def test_dot_on_non_struct_raises(self) -> None:
        """Applying ``.`` to a non-struct expression raises ``SemanticError``."""
        dot_exp = ast.Dot(structure=ast.Var("x", exp_type=None), member="y")
        body: tuple[ast.BlockItem, ...] = (
            ast.D(ast.VarDecl(ast.VariableDeclaration("x", None, ast.IntType()))),
            ast.S(ast.Return(exp=dot_exp)),
        )
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=body,
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )
        with pytest.raises(SemanticError):
            resolve(prog)

    def test_dot_unknown_member_raises(self) -> None:
        """Accessing a non-existent member via ``.`` raises ``SemanticError``."""
        struct_type = ast.Structure("S")
        s_decl = ast.D(ast.VarDecl(ast.VariableDeclaration("s", None, struct_type)))
        dot_exp = ast.Dot(structure=ast.Var("s", exp_type=None), member="nonexistent")
        prog = ast.Program(
            declarations=(
                _struct_decl("S", _member("x", ast.IntType())),
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=(s_decl, ast.S(ast.Return(exp=dot_exp))),
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )
        with pytest.raises(SemanticError):
            resolve(prog)


class TestArrowTypeChecking:
    """``Arrow`` expressions have the member's type and require a pointer-to-struct operand."""

    def _arrow_program(self) -> ast.Program:
        """Build a minimal program containing a struct pointer and an arrow-access return.

        Returns:
            A program with ``struct S { int x; }; int main(void) { struct S *p; return p->x; }``.
        """
        ptr_type = ast.Pointer(ast.Structure("S"))
        p_decl = ast.D(ast.VarDecl(ast.VariableDeclaration("p", None, ptr_type)))
        arrow_exp = ast.Arrow(pointer=ast.Var("p", exp_type=None), member="x")
        return ast.Program(
            declarations=(
                _struct_decl("S", _member("x", ast.IntType())),
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=(p_decl, ast.S(ast.Return(exp=arrow_exp))),
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )

    def test_arrow_result_type_is_member_type(self) -> None:
        """After type-checking, the ``Arrow`` expression carries the member's type."""
        resolved = resolve(self._arrow_program())
        main_fn = next(d.function_declaration for d in resolved.declarations if isinstance(d, ast.FunDecl))
        ret_stmt = next(
            item.statement
            for item in main_fn.body
            if isinstance(item, ast.S) and isinstance(item.statement, ast.Return)
        )
        assert isinstance(ret_stmt, ast.Return)
        assert isinstance(ret_stmt.exp, ast.Arrow)
        assert ret_stmt.exp.exp_type == ast.IntType()

    def test_arrow_on_non_pointer_raises(self) -> None:
        """Applying ``->`` to a non-pointer expression raises ``SemanticError``."""
        arrow_exp = ast.Arrow(pointer=ast.Var("x", exp_type=None), member="y")
        body: tuple[ast.BlockItem, ...] = (
            ast.D(ast.VarDecl(ast.VariableDeclaration("x", None, ast.IntType()))),
            ast.S(ast.Return(exp=arrow_exp)),
        )
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=body,
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )
        with pytest.raises(SemanticError):
            resolve(prog)

    def test_arrow_on_pointer_to_non_struct_raises(self) -> None:
        """Applying ``->`` to a ``int *`` raises ``SemanticError``."""
        arrow_exp = ast.Arrow(pointer=ast.Var("p", exp_type=None), member="x")
        body: tuple[ast.BlockItem, ...] = (
            ast.D(ast.VarDecl(ast.VariableDeclaration("p", None, ast.Pointer(ast.IntType())))),
            ast.S(ast.Return(exp=arrow_exp)),
        )
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=body,
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )
        with pytest.raises(SemanticError):
            resolve(prog)

    def test_arrow_unknown_member_raises(self) -> None:
        """Accessing a non-existent member via ``->`` raises ``SemanticError``."""
        ptr_type = ast.Pointer(ast.Structure("S"))
        p_decl = ast.D(ast.VarDecl(ast.VariableDeclaration("p", None, ptr_type)))
        arrow_exp = ast.Arrow(pointer=ast.Var("p", exp_type=None), member="nonexistent")
        prog = ast.Program(
            declarations=(
                _struct_decl("S", _member("x", ast.IntType())),
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=(p_decl, ast.S(ast.Return(exp=arrow_exp))),
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )
        with pytest.raises(SemanticError):
            resolve(prog)


def _anon_struct_member(*members: "ast.MemberDeclaration") -> "ast.MemberDeclaration":
    """Return a ``MemberDeclaration`` representing an anonymous struct member.

    Args:
        *members: Members of the anonymous inner struct.

    Returns:
        An ``ast.MemberDeclaration`` with ``member_name=None`` and a ``StructDecl`` in ``nested_decl``.
    """
    tag = "(anon_struct_test)"
    inner_decl = ast.StructDecl(ast.StructDeclaration(tag=tag, members=members))
    return ast.MemberDeclaration(member_name=None, member_type=ast.Structure(tag=tag), nested_decl=inner_decl)


def _anon_union_member(*members: "ast.MemberDeclaration") -> "ast.MemberDeclaration":
    """Return a ``MemberDeclaration`` representing an anonymous union member.

    Args:
        *members: Members of the anonymous inner union.

    Returns:
        An ``ast.MemberDeclaration`` with ``member_name=None`` and a ``UnionDecl`` in ``nested_decl``.
    """
    tag = "(anon_union_test)"
    inner_decl = ast.UnionDecl(ast.UnionDeclaration(tag=tag, members=members))
    return ast.MemberDeclaration(member_name=None, member_type=ast.Union(tag=tag), nested_decl=inner_decl)


class TestAnonymousStructMemberLayout:
    """Anonymous struct members promote their fields into the enclosing struct."""

    def test_anonymous_struct_member_single_field_promoted(self) -> None:
        """An anonymous struct with ``int y`` promotes ``y`` at correct offset in outer struct."""
        outer = _struct_decl("Outer", _anon_struct_member(_member("y", ast.IntType())))
        prog = _prog_with_decls(outer)
        _, _, struct_table, _ = resolve_with_symbols(prog)
        outer_entry = next(e for e in struct_table.values() if any(m.name == "y" for m in e.members))
        y_entry = next(m for m in outer_entry.members if m.name == "y")
        assert y_entry.offset == 0
        assert y_entry.member_type == ast.IntType()

    def test_anonymous_struct_after_named_member(self) -> None:
        """An anonymous struct after ``int x`` places promoted fields after ``x``."""
        outer = _struct_decl(
            "Outer",
            _member("x", ast.IntType()),
            _anon_struct_member(_member("y", ast.IntType())),
        )
        prog = _prog_with_decls(outer)
        _, _, struct_table, _ = resolve_with_symbols(prog)
        # Outer struct has both "x" and "y"; inner only has "y".
        outer_entry = next(
            e
            for e in struct_table.values()
            if any(m.name == "x" for m in e.members) and any(m.name == "y" for m in e.members)
        )
        x_entry = next(m for m in outer_entry.members if m.name == "x")
        y_entry = next(m for m in outer_entry.members if m.name == "y")
        assert x_entry.offset == 0
        assert y_entry.offset == 4

    def test_anonymous_struct_two_fields_both_promoted(self) -> None:
        """Both fields of an anonymous struct are promoted to the outer struct."""
        outer = _struct_decl(
            "Outer",
            _anon_struct_member(_member("a", ast.IntType()), _member("b", ast.LongType())),
        )
        prog = _prog_with_decls(outer)
        _, _, struct_table, _ = resolve_with_symbols(prog)
        outer_entry = next(e for e in struct_table.values() if any(m.name == "a" for m in e.members))
        names = {m.name for m in outer_entry.members}
        assert "a" in names
        assert "b" in names

    def test_anonymous_union_member_fields_promoted_at_same_offset(self) -> None:
        """An anonymous union's fields are all promoted at the base offset of the anonymous member."""
        outer = _struct_decl(
            "Outer",
            _member("x", ast.IntType()),
            _anon_union_member(_member("a", ast.IntType()), _member("b", ast.LongType())),
        )
        prog = _prog_with_decls(outer)
        _, _, struct_table, _ = resolve_with_symbols(prog)
        # Outer struct has "x", "a", "b"; inner union only has "a" and "b".
        outer_entry = next(e for e in struct_table.values() if any(m.name == "x" for m in e.members))
        a_entry = next(m for m in outer_entry.members if m.name == "a")
        b_entry = next(m for m in outer_entry.members if m.name == "b")
        assert a_entry.offset == b_entry.offset

    def test_anonymous_struct_inner_entry_registered_in_table(self) -> None:
        """The anonymous inner struct is also registered in the struct type table."""
        outer = _struct_decl("Outer", _anon_struct_member(_member("y", ast.IntType())))
        prog = _prog_with_decls(outer)
        _, _, struct_table, _ = resolve_with_symbols(prog)
        assert len(struct_table) == 2

    def test_outer_struct_size_accounts_for_anonymous_member(self) -> None:
        """Outer struct size includes the anonymous struct's footprint."""
        outer = _struct_decl(
            "Outer",
            _member("x", ast.IntType()),
            _anon_struct_member(_member("y", ast.IntType())),
        )
        prog = _prog_with_decls(outer)
        _, _, struct_table, _ = resolve_with_symbols(prog)
        outer_entry = next(
            e
            for e in struct_table.values()
            if any(m.name == "x" for m in e.members) and any(m.name == "y" for m in e.members)
        )
        assert outer_entry.size == 8
