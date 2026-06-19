"""Semantic analysis tests for flexible array members (C11 §6.7.2.1p18)."""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import _resolved_body


def _struct_with_fam(
    tag: str, *extra_members: ast.MemberDeclaration, fam_element: ast.Type = ast.Char()
) -> ast.StructDecl:
    """Build a ``StructDecl`` whose last member is a flexible array member.

    Args:
        tag: The struct tag.
        *extra_members: Normal members that precede the FAM.
        fam_element: Element type of the flexible array member.

    Returns:
        A ``StructDecl`` with a FAM named ``data`` as the last member.
    """
    fam = ast.MemberDeclaration(member_name="data", member_type=ast.Array(fam_element, size=None))
    return ast.StructDecl(ast.StructDeclaration(tag=tag, members=(*extra_members, fam)))


def _int_member(name: str) -> ast.MemberDeclaration:
    """Shorthand: ``int <name>`` member declaration."""
    return ast.MemberDeclaration(member_name=name, member_type=ast.IntType())


def _make_program(*decls: ast.Declaration) -> ast.Program:
    """Wrap declarations in a minimal program with an ``int main()``."""
    main = ast.FunDecl(
        ast.Function(
            name="main",
            params=(),
            body=(ast.S(ast.Return(exp=ast.Constant(ast.ConstInt(0)))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
    )
    return ast.Program(declarations=(*decls, main))


class TestFlexibleArrayMemberValidation:
    """SA validates FAM placement rules per C11 §6.7.2.1p18."""

    def test_fam_as_last_member_is_valid(self) -> None:
        """``struct S { int n; char data[]; };`` resolves without error."""
        struct_decl = _struct_with_fam("S", _int_member("n"))
        prog = _make_program(struct_decl)
        resolve(prog)  # must not raise

    def test_fam_not_last_member_raises(self) -> None:
        """FAM followed by another member raises ``SemanticError``."""
        fam = ast.MemberDeclaration(member_name="data", member_type=ast.Array(ast.Char(), size=None))
        extra = ast.MemberDeclaration(member_name="n", member_type=ast.IntType())
        struct_decl = ast.StructDecl(ast.StructDeclaration(tag="S", members=(fam, extra)))
        prog = _make_program(struct_decl)
        with pytest.raises(SemanticError, match="[Ff]lexible array member"):
            resolve(prog)

    def test_fam_only_member_raises(self) -> None:
        """A struct with only a FAM and no preceding members raises ``SemanticError``."""
        struct_decl = _struct_with_fam("S")  # no extra members
        prog = _make_program(struct_decl)
        with pytest.raises(SemanticError, match="[Ff]lexible array member"):
            resolve(prog)


class TestFlexibleArrayMemberLayout:
    """sizeof a struct with FAM matches sizeof without FAM (FAM has zero contribution)."""

    def test_sizeof_struct_excludes_fam(self) -> None:
        """sizeof(struct { int n; char data[]; }) == sizeof(struct { int n; })."""
        from compiler.semantic_analysis import resolve_with_symbols

        struct_with_fam = _struct_with_fam("WithFAM", _int_member("n"))
        struct_without = ast.StructDecl(
            ast.StructDeclaration(
                tag="Without",
                members=(_int_member("n"),),
            )
        )
        prog = _make_program(struct_with_fam, struct_without)
        _, _, struct_type_table, _ = resolve_with_symbols(prog)

        with_tag = next(t for t in struct_type_table if "WithFAM" in t)
        without_tag = next(t for t in struct_type_table if "Without" in t)

        size_with = struct_type_table[with_tag].size
        size_without = struct_type_table[without_tag].size
        assert size_with == size_without


class TestArrayCompoundLiteralSizeInference:
    """``(T[]){...}`` compound literals: SA infers array size from initializer count."""

    def _make_lit(self, *values: int) -> ast.CompoundLiteral:
        """Build ``(int[]){v0, v1, ...}`` compound literal."""
        return ast.CompoundLiteral(
            target_type=ast.Array(ast.IntType(), size=None),
            init=ast.CompoundInit(initializers=[ast.SingleInit(ast.Constant(ast.ConstInt(v))) for v in values]),
        )

    def _resolve_compound_literal(self, lit: ast.CompoundLiteral, n: int) -> ast.CompoundLiteral:
        """Resolve ``lit`` via a ``int *p = <lit>;`` statement; returns resolved literal."""
        # Arrays decay to pointers, so int *p = (int[]){...} is valid.
        body: tuple[ast.BlockItem, ...] = (
            ast.D(ast.VarDecl(ast.VariableDeclaration("p", ast.SingleInit(lit), ast.Pointer(ast.IntType())))),
            ast.S(ast.Return(exp=ast.Constant(ast.ConstInt(0)))),
        )
        resolved_body = _resolved_body(body)
        decl_item = resolved_body[0]
        assert isinstance(decl_item, ast.D)
        init = decl_item.declaration.variable_declaration.init  # type: ignore[union-attr]
        assert isinstance(init, ast.SingleInit)
        # After array decay, SA wraps the literal in AddrOf (array→pointer conversion).
        inner = init.exp
        while isinstance(inner, (ast.Cast, ast.AddrOf)):
            inner = inner.exp if isinstance(inner, ast.Cast) else inner.exp
        assert isinstance(inner, ast.CompoundLiteral)
        return inner

    def test_inferred_size_int_array(self) -> None:
        """``(int[]){1, 2, 3}`` resolves with ``Array(IntType, 3)`` as exp_type."""
        resolved_lit = self._resolve_compound_literal(self._make_lit(1, 2, 3), 3)
        assert isinstance(resolved_lit.exp_type, ast.Array)
        assert resolved_lit.exp_type.size == 3

    def test_inferred_size_two_elements(self) -> None:
        """``(int[]){10, 20}`` resolves with size 2."""
        resolved_lit = self._resolve_compound_literal(self._make_lit(10, 20), 2)
        assert isinstance(resolved_lit.exp_type, ast.Array)
        assert resolved_lit.exp_type.size == 2
