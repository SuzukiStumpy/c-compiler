"""Tests for TAC generation of union member-access expressions.

Union members all sit at offset 0, so CopyFromOffset/CopyToOffset are emitted
with offset=0 regardless of which member is accessed.
"""

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from compiler.tac_generator import generate
from compiler.types.symbols import LocalAttr, MemberEntry, StructEntry, StructTypeTable, SymbolTable, SymbolTableEntry


def _make_union_table(tag: str, *members: tuple[str, ast.Type]) -> StructTypeTable:
    """Build a ``StructTypeTable`` for a single union where all members have offset 0.

    Args:
        tag: The (already-uniquified) union tag.
        *members: Tuples of ``(member_name, member_type)``; offset is always 0.

    Returns:
        A ``StructTypeTable`` with the single union entry.  Size is the size
        of the largest member (simplified to 8 for long, 4 for int).
    """
    member_entries = tuple(MemberEntry(name=n, member_type=t, offset=0) for n, t in members)
    size = 8 if any(isinstance(t, ast.LongType) for _, t in members) else 4
    return {tag: StructEntry(alignment=size, size=size, members=member_entries)}


def _dot_body(
    var_name: str,
    union_tag: str,
    member_name: str,
    member_type: ast.Type,
    union_table: StructTypeTable,
) -> tuple[tac.Instruction, ...]:
    """Translate ``return <var>.<member>;`` for a union variable and return instruction body.

    Args:
        var_name: Name of the local union variable.
        union_tag: The uniquified union tag.
        member_name: The member to access.
        member_type: The C type of the member.
        union_table: The type table passed to ``generate``.

    Returns:
        The tuple of TAC instructions from the generated function body.
    """
    dot_exp = ast.Dot(
        structure=ast.Var(var_name, exp_type=ast.Union(union_tag)),
        member=member_name,
        exp_type=member_type,
    )
    prog = ast.Program(
        declarations=(
            ast.FunDecl(
                ast.Function(
                    name="main",
                    params=(),
                    body=(ast.S(ast.Return(exp=dot_exp)),),
                    fun_type=ast.FunType(params=(), ret=member_type),
                )
            ),
        )
    )
    sym: SymbolTable = {var_name: SymbolTableEntry(t=ast.Union(union_tag), attrs=LocalAttr())}
    return generate(prog, sym, union_table).functions[0].body


class TestUnionDotAccess:
    """Dot access on a union variable emits CopyFromOffset with offset 0."""

    def test_int_member_offset_is_zero(self) -> None:
        """``return u.x;`` on a ``union { int x; long y; }`` emits CopyFromOffset(offset=0)."""
        tag = ".UFoo.0"
        table = _make_union_table(tag, ("x", ast.IntType()), ("y", ast.LongType()))
        body = _dot_body("u", tag, "x", ast.IntType(), table)
        copy_instrs = [i for i in body if isinstance(i, tac.CopyFromOffset)]
        assert len(copy_instrs) == 1
        assert copy_instrs[0].src == "u"
        assert copy_instrs[0].offset == 0

    def test_long_member_offset_is_zero(self) -> None:
        """``return u.y;`` on a ``union { int x; long y; }`` also emits offset=0."""
        tag = ".UFoo.0"
        table = _make_union_table(tag, ("x", ast.IntType()), ("y", ast.LongType()))
        body = _dot_body("u", tag, "y", ast.LongType(), table)
        copy_instrs = [i for i in body if isinstance(i, tac.CopyFromOffset)]
        assert len(copy_instrs) == 1
        assert copy_instrs[0].offset == 0


class TestUnionArrowAccess:
    """Arrow access on a union pointer emits CopyFromOffset(offset=0) without AddPtr."""

    def test_arrow_no_addptr_emitted(self) -> None:
        """``return p->x;`` on a union pointer should NOT emit AddPtr (offset=0)."""
        tag = ".UFoo.0"
        table = _make_union_table(tag, ("x", ast.IntType()), ("y", ast.LongType()))
        ptr_name = "p"
        arrow_exp = ast.Arrow(
            pointer=ast.Var(ptr_name, exp_type=ast.Pointer(ast.Union(tag))),
            member="x",
            exp_type=ast.IntType(),
        )
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=(ast.S(ast.Return(exp=arrow_exp)),),
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )
        sym: SymbolTable = {ptr_name: SymbolTableEntry(t=ast.Pointer(ast.Union(tag)), attrs=LocalAttr())}
        body = generate(prog, sym, table).functions[0].body
        assert not any(isinstance(i, tac.AddPtr) for i in body)
