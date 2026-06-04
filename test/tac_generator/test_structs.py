"""Tests for TAC generation of struct member-access expressions.

Each test constructs ``ast`` nodes directly (without invoking the lexer,
parser, or semantic analyser) and verifies the exact TAC instruction
sequence emitted for ``Dot``, ``Arrow``, and member-assignment constructs.
"""

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from compiler.tac_generator import generate
from compiler.types.symbols import LocalAttr, MemberEntry, StructEntry, StructTypeTable, SymbolTable, SymbolTableEntry


def _make_struct_table(tag: str, *members: tuple[str, ast.Type, int]) -> StructTypeTable:
    """Build a ``StructTypeTable`` for a single struct with the given members.

    Args:
        tag: The (already-uniquified) struct tag.
        *members: Tuples of ``(member_name, member_type, byte_offset)``.

    Returns:
        A ``StructTypeTable`` containing the single struct entry.
    """
    member_entries = tuple(MemberEntry(name=n, member_type=t, offset=o) for n, t, o in members)
    total_size = member_entries[-1].offset + 4 if member_entries else 0
    alignment = max((4 for _ in member_entries), default=1)
    return {tag: StructEntry(alignment=alignment, size=total_size, members=member_entries)}


def _dot_body(
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
        member_type: The C type of the member (used as the ``Dot`` expression type).
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


def _arrow_body(
    ptr_name: str,
    struct_tag: str,
    member_name: str,
    member_type: ast.Type,
    struct_table: StructTypeTable,
) -> tuple[tac.Instruction, ...]:
    """Translate ``return <ptr>-><member>;`` and return the instruction body.

    Args:
        ptr_name: Name of the local pointer variable.
        struct_tag: The uniquified struct tag.
        member_name: The member to access.
        member_type: The C type of the member.
        struct_table: The struct type table passed to ``generate``.

    Returns:
        The tuple of TAC instructions from the generated function body.
    """
    ptr_type = ast.Pointer(ast.Structure(struct_tag))
    arrow_exp = ast.Arrow(
        pointer=ast.Var(ptr_name, exp_type=ptr_type),
        member=member_name,
        exp_type=member_type,
    )
    prog = ast.Program(
        declarations=(
            ast.FunDecl(
                ast.Function(
                    name="main",
                    params=(),
                    body=(ast.S(ast.Return(exp=arrow_exp)),),
                    fun_type=ast.FunType(params=(), ret=member_type),
                )
            ),
        )
    )
    sym: SymbolTable = {ptr_name: SymbolTableEntry(t=ptr_type, attrs=LocalAttr())}
    return generate(prog, sym, struct_table).functions[0].body


class TestDotRead:
    """``Dot`` expressions on local struct variables emit ``CopyFromOffset``."""

    def test_dot_at_offset_zero_emits_copy_from_offset(self) -> None:
        """``s.x`` where ``x`` is at offset 0 emits ``CopyFromOffset(src="s", offset=0, ...``."""
        tag = ".Spoint.0"
        table = _make_struct_table(tag, ("x", ast.IntType(), 0))
        body = _dot_body("s", tag, "x", ast.IntType(), table)
        copy_instrs = [i for i in body if isinstance(i, tac.CopyFromOffset)]
        assert len(copy_instrs) == 1
        assert copy_instrs[0].src == "s"
        assert copy_instrs[0].offset == 0

    def test_dot_at_offset_four_emits_correct_offset(self) -> None:
        """``s.y`` where ``y`` is at offset 4 emits ``CopyFromOffset(src="s", offset=4, ...``."""
        tag = ".Spoint.0"
        table = _make_struct_table(tag, ("x", ast.IntType(), 0), ("y", ast.IntType(), 4))
        body = _dot_body("s", tag, "y", ast.IntType(), table)
        copy_instrs = [i for i in body if isinstance(i, tac.CopyFromOffset)]
        assert len(copy_instrs) == 1
        assert copy_instrs[0].src == "s"
        assert copy_instrs[0].offset == 4

    def test_dot_destination_is_used_in_return(self) -> None:
        """The destination of ``CopyFromOffset`` is the value passed to ``Return``."""
        tag = ".Spoint.0"
        table = _make_struct_table(tag, ("x", ast.IntType(), 0))
        body = _dot_body("s", tag, "x", ast.IntType(), table)
        copy_instr = next(i for i in body if isinstance(i, tac.CopyFromOffset))
        ret_instr = next(i for i in body if isinstance(i, tac.Return))
        assert ret_instr.val == copy_instr.dst

    def test_no_add_ptr_for_plain_dot_access(self) -> None:
        """A ``Dot`` on a local variable (not a pointer) emits no ``AddPtr``."""
        tag = ".Spoint.0"
        table = _make_struct_table(tag, ("x", ast.IntType(), 0))
        body = _dot_body("s", tag, "x", ast.IntType(), table)
        assert not any(isinstance(i, tac.AddPtr) for i in body)


class TestNestedDotRead:
    """Consecutive ``Dot`` accesses accumulate offsets into a single ``CopyFromOffset``."""

    def test_nested_dot_accumulates_offsets(self) -> None:
        """``s.a.b`` where ``a`` is at offset 0 and ``b`` is at offset 4 emits offset 4."""
        outer_tag = ".Souter.0"
        inner_tag = ".Sinner.0"
        inner_table_entry = StructEntry(
            alignment=4,
            size=8,
            members=(MemberEntry(name="b", member_type=ast.IntType(), offset=4),),
        )
        outer_table_entry = StructEntry(
            alignment=4,
            size=8,
            members=(MemberEntry(name="a", member_type=ast.Structure(inner_tag), offset=0),),
        )
        struct_table: StructTypeTable = {
            inner_tag: inner_table_entry,
            outer_tag: outer_table_entry,
        }
        inner_dot = ast.Dot(
            structure=ast.Var("s", exp_type=ast.Structure(outer_tag)),
            member="a",
            exp_type=ast.Structure(inner_tag),
        )
        outer_dot = ast.Dot(structure=inner_dot, member="b", exp_type=ast.IntType())
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=(ast.S(ast.Return(exp=outer_dot)),),
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )
        sym: SymbolTable = {"s": SymbolTableEntry(t=ast.Structure(outer_tag), attrs=LocalAttr())}
        body = generate(prog, sym, struct_table).functions[0].body
        copy_instrs = [i for i in body if isinstance(i, tac.CopyFromOffset)]
        assert len(copy_instrs) == 1
        assert copy_instrs[0].src == "s"
        assert copy_instrs[0].offset == 4

    def test_nested_dot_emits_no_intermediate_copy(self) -> None:
        """``s.a.b`` emits exactly one ``CopyFromOffset``, not one per dot."""
        outer_tag = ".Souter.0"
        inner_tag = ".Sinner.0"
        struct_table: StructTypeTable = {
            inner_tag: StructEntry(4, 8, (MemberEntry("b", ast.IntType(), 4),)),
            outer_tag: StructEntry(4, 8, (MemberEntry("a", ast.Structure(inner_tag), 0),)),
        }
        inner_dot = ast.Dot(
            structure=ast.Var("s", exp_type=ast.Structure(outer_tag)),
            member="a",
            exp_type=ast.Structure(inner_tag),
        )
        outer_dot = ast.Dot(structure=inner_dot, member="b", exp_type=ast.IntType())
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=(ast.S(ast.Return(exp=outer_dot)),),
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )
        sym: SymbolTable = {"s": SymbolTableEntry(t=ast.Structure(outer_tag), attrs=LocalAttr())}
        body = generate(prog, sym, struct_table).functions[0].body
        assert sum(1 for i in body if isinstance(i, tac.CopyFromOffset)) == 1


class TestArrowRead:
    """``Arrow`` expressions on struct pointers emit ``Load``, with ``AddPtr`` for non-zero offsets."""

    def test_arrow_at_offset_zero_emits_load_without_add_ptr(self) -> None:
        """``p->x`` where ``x`` is at offset 0 emits ``Load`` directly from ``p``."""
        tag = ".Spoint.0"
        table = _make_struct_table(tag, ("x", ast.IntType(), 0))
        body = _arrow_body("p", tag, "x", ast.IntType(), table)
        load_instrs = [i for i in body if isinstance(i, tac.Load)]
        assert len(load_instrs) == 1
        assert load_instrs[0].src_ptr == tac.Var(value="p")
        assert not any(isinstance(i, tac.AddPtr) for i in body)

    def test_arrow_at_nonzero_offset_emits_add_ptr_then_load(self) -> None:
        """``p->y`` where ``y`` is at offset 4 emits ``AddPtr`` followed by ``Load``."""
        tag = ".Spoint.0"
        table = _make_struct_table(tag, ("x", ast.IntType(), 0), ("y", ast.IntType(), 4))
        body = _arrow_body("p", tag, "y", ast.IntType(), table)
        add_ptr_instrs = [i for i in body if isinstance(i, tac.AddPtr)]
        load_instrs = [i for i in body if isinstance(i, tac.Load)]
        assert len(add_ptr_instrs) == 1
        assert len(load_instrs) == 1
        assert add_ptr_instrs[0].ptr == tac.Var(value="p")
        assert add_ptr_instrs[0].index == tac.Constant(ast.ConstLong(4))
        assert add_ptr_instrs[0].scale == 1

    def test_arrow_nonzero_load_uses_add_ptr_destination(self) -> None:
        """The ``Load`` after an ``AddPtr`` reads from the pointer computed by ``AddPtr``."""
        tag = ".Spoint.0"
        table = _make_struct_table(tag, ("x", ast.IntType(), 0), ("y", ast.IntType(), 4))
        body = _arrow_body("p", tag, "y", ast.IntType(), table)
        add_ptr = next(i for i in body if isinstance(i, tac.AddPtr))
        load = next(i for i in body if isinstance(i, tac.Load))
        assert load.src_ptr == add_ptr.dst

    def test_arrow_result_used_in_return(self) -> None:
        """The destination of the ``Load`` instruction is the value returned."""
        tag = ".Spoint.0"
        table = _make_struct_table(tag, ("x", ast.IntType(), 0))
        body = _arrow_body("p", tag, "x", ast.IntType(), table)
        load = next(i for i in body if isinstance(i, tac.Load))
        ret = next(i for i in body if isinstance(i, tac.Return))
        assert ret.val == load.dst


class TestDotAssignment:
    """Assigning to a ``Dot`` lvalue emits ``CopyToOffset``."""

    def test_dot_assignment_emits_copy_to_offset(self) -> None:
        """``s.x = 5;`` emits ``CopyToOffset(src=Constant(ConstInt(5)), dst="s", offset=0)``."""
        tag = ".Spoint.0"
        table = _make_struct_table(tag, ("x", ast.IntType(), 0))
        assign_exp = ast.Assignment(
            lvalue=ast.Dot(
                structure=ast.Var("s", exp_type=ast.Structure(tag)),
                member="x",
                exp_type=ast.IntType(),
            ),
            rvalue=ast.Constant(ast.ConstInt(5), exp_type=ast.IntType()),
            exp_type=ast.IntType(),
        )
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=(
                            ast.S(ast.Expression(exp=assign_exp)),
                            ast.S(ast.Return(exp=ast.Constant(ast.ConstInt(0), exp_type=ast.IntType()))),
                        ),
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )
        sym: SymbolTable = {"s": SymbolTableEntry(t=ast.Structure(tag), attrs=LocalAttr())}
        body = generate(prog, sym, table).functions[0].body
        copy_instrs = [i for i in body if isinstance(i, tac.CopyToOffset)]
        assert len(copy_instrs) == 1
        assert copy_instrs[0].dst == "s"
        assert copy_instrs[0].offset == 0
        assert copy_instrs[0].src == tac.Constant(ast.ConstInt(5))

    def test_dot_assignment_at_nonzero_offset(self) -> None:
        """``s.y = 7;`` where ``y`` is at offset 4 emits ``CopyToOffset`` with offset 4."""
        tag = ".Spoint.0"
        table = _make_struct_table(tag, ("x", ast.IntType(), 0), ("y", ast.IntType(), 4))
        assign_exp = ast.Assignment(
            lvalue=ast.Dot(
                structure=ast.Var("s", exp_type=ast.Structure(tag)),
                member="y",
                exp_type=ast.IntType(),
            ),
            rvalue=ast.Constant(ast.ConstInt(7), exp_type=ast.IntType()),
            exp_type=ast.IntType(),
        )
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=(
                            ast.S(ast.Expression(exp=assign_exp)),
                            ast.S(ast.Return(exp=ast.Constant(ast.ConstInt(0), exp_type=ast.IntType()))),
                        ),
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )
        sym: SymbolTable = {"s": SymbolTableEntry(t=ast.Structure(tag), attrs=LocalAttr())}
        body = generate(prog, sym, table).functions[0].body
        copy_instrs = [i for i in body if isinstance(i, tac.CopyToOffset)]
        assert len(copy_instrs) == 1
        assert copy_instrs[0].dst == "s"
        assert copy_instrs[0].offset == 4
