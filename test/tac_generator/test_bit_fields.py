"""Tests for TAC generation of bit-field member-access expressions.

Each test constructs ``ast`` nodes directly (without invoking the lexer,
parser, or semantic analyser) and verifies the exact TAC instruction
sequence emitted for reads and writes of bit-field struct members.
"""

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from compiler.tac_generator import generate
from compiler.types.symbols import LocalAttr, MemberEntry, StructEntry, StructTypeTable, SymbolTable, SymbolTableEntry


def _make_bf_struct_table(
    tag: str,
    *members: tuple[str, ast.Type, int, int | None, int | None],
) -> StructTypeTable:
    """Build a ``StructTypeTable`` for a struct that may include bit-field members.

    Args:
        tag: The (already-uniquified) struct tag.
        *members: Tuples of ``(name, type, byte_offset, bit_offset, bit_width)``.
            Pass ``None`` for ``bit_offset`` and ``bit_width`` for regular members.

    Returns:
        A ``StructTypeTable`` containing the single struct entry.
    """
    entries = tuple(
        MemberEntry(name=n, member_type=t, offset=o, bit_offset=bo, bit_width=bw) for n, t, o, bo, bw in members
    )
    return {tag: StructEntry(alignment=4, size=4, members=entries)}


def _dot_read_body(
    var_name: str,
    struct_tag: str,
    member_name: str,
    member_type: ast.Type,
    struct_table: StructTypeTable,
) -> tuple[tac.Instruction, ...]:
    """Translate ``return <var>.<member>;`` and return the instruction body.

    Args:
        var_name: Name of the local struct variable.
        struct_tag: The uniquified struct tag.
        member_name: The member to access.
        member_type: The C type of the member.
        struct_table: The struct type table passed to ``generate``.

    Returns:
        The tuple of TAC instructions from the generated function body.
    """
    dot_exp = ast.Dot(
        structure=ast.Var(var_name, exp_type=ast.Structure(struct_tag)),
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
    sym: SymbolTable = {var_name: SymbolTableEntry(t=ast.Structure(struct_tag), attrs=LocalAttr())}
    return generate(prog, sym, struct_table).functions[0].body


def _dot_write_body(
    var_name: str,
    struct_tag: str,
    member_name: str,
    member_type: ast.Type,
    rhs: ast.Exp,
    struct_table: StructTypeTable,
) -> tuple[tac.Instruction, ...]:
    """Translate ``<var>.<member> = <rhs>;`` and return the instruction body.

    Args:
        var_name: Name of the local struct variable.
        struct_tag: The uniquified struct tag.
        member_name: The member to write.
        member_type: The C type of the member.
        rhs: The expression assigned to the member.
        struct_table: The struct type table passed to ``generate``.

    Returns:
        The tuple of TAC instructions from the generated function body.
    """
    lhs = ast.Dot(
        structure=ast.Var(var_name, exp_type=ast.Structure(struct_tag)),
        member=member_name,
        exp_type=member_type,
    )
    assign_exp = ast.Assignment(lvalue=lhs, rvalue=rhs, exp_type=member_type)
    prog = ast.Program(
        declarations=(
            ast.FunDecl(
                ast.Function(
                    name="main",
                    params=(),
                    body=(ast.S(ast.Expression(exp=assign_exp)),),
                    fun_type=ast.FunType(params=(), ret=ast.VoidType()),
                )
            ),
        )
    )
    sym: SymbolTable = {var_name: SymbolTableEntry(t=ast.Structure(struct_tag), attrs=LocalAttr())}
    return generate(prog, sym, struct_table).functions[0].body


class TestBitFieldDotRead:
    """``Dot`` access to a bit-field member emits ``GetBitField``."""

    def test_dot_read_unsigned_bit_field_emits_get_bit_field(self) -> None:
        """``s.x`` where ``x`` is an unsigned 3-bit field emits ``GetBitField``."""
        tag = ".Sflags.0"
        table = _make_bf_struct_table(tag, ("x", ast.UIntType(), 0, 0, 3))
        body = _dot_read_body("s", tag, "x", ast.UIntType(), table)
        bf_instrs = [i for i in body if isinstance(i, tac.GetBitField)]
        assert len(bf_instrs) == 1
        bf = bf_instrs[0]
        assert bf.src == "s"
        assert bf.byte_offset == 0
        assert bf.bit_offset == 0
        assert bf.bit_width == 3
        assert bf.signed is False

    def test_dot_read_signed_bit_field_has_signed_flag(self) -> None:
        """``s.x`` where ``x`` is a signed bit-field sets ``signed=True``."""
        tag = ".Sflags.0"
        table = _make_bf_struct_table(tag, ("x", ast.IntType(), 0, 0, 5))
        body = _dot_read_body("s", tag, "x", ast.IntType(), table)
        bf_instrs = [i for i in body if isinstance(i, tac.GetBitField)]
        assert len(bf_instrs) == 1
        assert bf_instrs[0].signed is True

    def test_dot_read_bit_field_at_bit_offset_3(self) -> None:
        """A bit-field packed after a 3-bit field has ``bit_offset=3``."""
        tag = ".Sflags.0"
        table = _make_bf_struct_table(
            tag,
            ("x", ast.UIntType(), 0, 0, 3),
            ("y", ast.UIntType(), 0, 3, 5),
        )
        body = _dot_read_body("s", tag, "y", ast.UIntType(), table)
        bf_instrs = [i for i in body if isinstance(i, tac.GetBitField)]
        assert len(bf_instrs) == 1
        bf = bf_instrs[0]
        assert bf.byte_offset == 0
        assert bf.bit_offset == 3
        assert bf.bit_width == 5

    def test_dot_read_bit_field_does_not_emit_copy_from_offset(self) -> None:
        """Bit-field reads do not use ``CopyFromOffset``."""
        tag = ".Sflags.0"
        table = _make_bf_struct_table(tag, ("x", ast.UIntType(), 0, 0, 3))
        body = _dot_read_body("s", tag, "x", ast.UIntType(), table)
        assert not any(isinstance(i, tac.CopyFromOffset) for i in body)

    def test_dot_read_bit_field_dst_used_in_return(self) -> None:
        """The ``dst`` of ``GetBitField`` is the value passed to ``Return``."""
        tag = ".Sflags.0"
        table = _make_bf_struct_table(tag, ("x", ast.UIntType(), 0, 0, 3))
        body = _dot_read_body("s", tag, "x", ast.UIntType(), table)
        bf = next(i for i in body if isinstance(i, tac.GetBitField))
        ret = next(i for i in body if isinstance(i, tac.Return))
        assert ret.val == bf.dst


class TestBitFieldDotWrite:
    """Assignment to a bit-field member via ``Dot`` emits ``SetBitField``."""

    def test_dot_write_bit_field_emits_set_bit_field(self) -> None:
        """``s.x = 5;`` where ``x`` is a 3-bit field emits ``SetBitField``."""
        tag = ".Sflags.0"
        table = _make_bf_struct_table(tag, ("x", ast.UIntType(), 0, 0, 3))
        rhs = ast.Constant(ast.ConstInt(5), exp_type=ast.UIntType())
        body = _dot_write_body("s", tag, "x", ast.UIntType(), rhs, table)
        bf_instrs = [i for i in body if isinstance(i, tac.SetBitField)]
        assert len(bf_instrs) == 1
        bf = bf_instrs[0]
        assert bf.dst == "s"
        assert bf.byte_offset == 0
        assert bf.bit_offset == 0
        assert bf.bit_width == 3

    def test_dot_write_bit_field_does_not_emit_copy_to_offset(self) -> None:
        """Bit-field writes do not use ``CopyToOffset``."""
        tag = ".Sflags.0"
        table = _make_bf_struct_table(tag, ("x", ast.UIntType(), 0, 0, 3))
        rhs = ast.Constant(ast.ConstInt(5), exp_type=ast.UIntType())
        body = _dot_write_body("s", tag, "x", ast.UIntType(), rhs, table)
        assert not any(isinstance(i, tac.CopyToOffset) for i in body)
