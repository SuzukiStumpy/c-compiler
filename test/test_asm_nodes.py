"""Tests for the assembly AST node definitions.

Verifies construction, equality, immutability, and type hierarchy for all
node types in the ``compiler.syntax.asm`` package.
"""

import dataclasses

import pytest

from compiler.syntax.asm import (
    AllocateStack,
    Byte,
    Call,
    Cmp,
    CondCodes,
    DeallocateStack,
    Function,
    GPRegisters,
    Imm,
    Instruction,
    Int128Init,
    Jmp,
    JmpCC,
    Label,
    Longword,
    Memory,
    Mov,
    MoveZeroExtend,
    Movsx,
    Operand,
    Program,
    Push,
    Quadword,
    Register,
    Ret,
    SetCC,
    ShortInit,
    StaticConstant,
    StaticVariable,
    StringInit,
    UInt128Init,
    UShortInit,
    Width,
    Word,
)

# ---------------------------------------------------------------------------
# Operand nodes
# ---------------------------------------------------------------------------


class TestImm:
    """``Imm`` is a concrete frozen dataclass representing an immediate value."""

    def test_construction(self) -> None:
        """``Imm`` can be constructed with an integer value."""
        assert Imm(value=42).value == 42

    def test_is_operand(self) -> None:
        """``Imm`` is a subclass of ``Operand``."""
        assert isinstance(Imm(value=0), Operand)

    def test_equality(self) -> None:
        """Two ``Imm`` nodes with the same value are equal."""
        assert Imm(value=7) == Imm(value=7)

    def test_inequality(self) -> None:
        """``Imm`` nodes with different values are not equal."""
        assert Imm(value=1) != Imm(value=2)

    def test_immutable(self) -> None:
        """``Imm`` is frozen - attribute assignment raises ``FrozenInstanceError``."""
        imm = Imm(value=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            imm.value = 99  # type: ignore[misc]


class TestRegister:
    """``Register`` wraps a ``GPRegisters`` enum value and an optional ``Width``."""

    def test_construction(self) -> None:
        """``Register`` can be constructed with just a register name."""
        Register(GPRegisters.AX)

    def test_is_operand(self) -> None:
        """``Register`` is a subclass of ``Operand``."""
        assert isinstance(Register(GPRegisters.AX), Operand)

    def test_equality(self) -> None:
        """Two ``Register`` instances with the same reg and default width are equal."""
        assert Register(GPRegisters.AX) == Register(GPRegisters.AX)

    def test_not_equal_to_imm(self) -> None:
        """``Register`` is not equal to an ``Imm`` node."""
        assert Register(GPRegisters.AX) != Imm(value=0)

    def test_default_width_is_dword(self) -> None:
        """The default width for a ``Register`` is ``Width.DWORD``."""
        assert Register(GPRegisters.AX).width == Width.DWORD

    def test_explicit_byte_width(self) -> None:
        """``Register`` can be constructed with ``Width.BYTE``."""
        assert Register(GPRegisters.AX, Width.BYTE).width == Width.BYTE

    def test_different_widths_not_equal(self) -> None:
        """``Register`` nodes for the same logical register at different widths are not equal."""
        assert Register(GPRegisters.AX, Width.DWORD) != Register(GPRegisters.AX, Width.BYTE)


# ---------------------------------------------------------------------------
# Instruction nodes
# ---------------------------------------------------------------------------


class TestMov:
    """``Mov`` is a concrete frozen dataclass for a move instruction."""

    def test_construction(self) -> None:
        """``Mov`` stores its source and destination operands."""
        mov = Mov(src=Imm(value=3), dst=Register(GPRegisters.AX))
        assert mov.src == Imm(value=3)
        assert mov.dst == Register(GPRegisters.AX)

    def test_is_instruction(self) -> None:
        """``Mov`` is a subclass of ``Instruction``."""
        assert isinstance(Mov(src=Imm(value=0), dst=Register(GPRegisters.AX)), Instruction)

    def test_equality(self) -> None:
        """Two ``Mov`` nodes with identical operands are equal."""
        assert Mov(src=Imm(value=1), dst=Register(GPRegisters.AX)) == Mov(
            src=Imm(value=1), dst=Register(GPRegisters.AX)
        )

    def test_inequality_different_src(self) -> None:
        """``Mov`` nodes with different source operands are not equal."""
        assert Mov(src=Imm(value=1), dst=Register(GPRegisters.AX)) != Mov(
            src=Imm(value=2), dst=Register(GPRegisters.AX)
        )

    def test_immutable(self) -> None:
        """``Mov`` is frozen - attribute assignment raises ``FrozenInstanceError``."""
        mov = Mov(src=Imm(value=0), dst=Register(GPRegisters.AX))
        with pytest.raises(dataclasses.FrozenInstanceError):
            mov.src = Imm(value=99)  # type: ignore[misc]


class TestRet:
    """``Ret`` is a no-field frozen dataclass for a return instruction."""

    def test_construction(self) -> None:
        """``Ret`` can be constructed with no arguments."""
        Ret()

    def test_is_instruction(self) -> None:
        """``Ret`` is a subclass of ``Instruction``."""
        assert isinstance(Ret(), Instruction)

    def test_equality(self) -> None:
        """Two ``Ret`` instances are equal."""
        assert Ret() == Ret()

    def test_not_equal_to_mov(self) -> None:
        """``Ret`` is not equal to a ``Mov`` node."""
        assert Ret() != Mov(src=Imm(value=0), dst=Register(GPRegisters.AX))


# ---------------------------------------------------------------------------
# Declaration nodes
# ---------------------------------------------------------------------------


class TestFunction:
    """``Function`` is a frozen dataclass holding a name and instruction sequence."""

    def test_construction(self) -> None:
        """``Function`` stores its name and instruction tuple."""
        fn = Function(
            name="main", globl=True, instructions=(Mov(src=Imm(value=0), dst=Register(GPRegisters.AX)), Ret())
        )
        assert fn.name == "main"
        assert len(fn.instructions) == 2

    def test_empty_instructions(self) -> None:
        """``Function`` accepts an empty instruction tuple."""
        fn = Function(name="main", globl=False, instructions=())
        assert fn.instructions == ()

    def test_equality(self) -> None:
        """Two ``Function`` nodes with the same name and instructions are equal."""
        fn1 = Function(name="foo", globl=False, instructions=(Ret(),))
        fn2 = Function(name="foo", globl=False, instructions=(Ret(),))
        assert fn1 == fn2

    def test_immutable(self) -> None:
        """``Function`` is frozen - attribute assignment raises ``FrozenInstanceError``."""
        fn = Function(name="main", globl=False, instructions=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            fn.name = "other"  # type: ignore[misc]

    def test_no_comment_block_when_source_file_absent(self) -> None:
        """When ``source_file`` is empty no comment block appears in the emitted text."""
        fn = Function(name="main", globl=True, instructions=(Ret(),))
        assert "###" not in fn.emit()
        assert "# Function:" not in fn.emit()

    def test_comment_block_emitted_when_source_file_set(self) -> None:
        """When ``source_file`` is non-empty the comment block appears before the directives."""
        fn = Function(name="main", globl=True, source_file="foo.c", source_line=5, instructions=(Ret(),))
        assert "###" in fn.emit()
        assert "# Function: main" in fn.emit()
        assert "# Source: foo.c:5" in fn.emit()

    def test_comment_block_precedes_globl_directive(self) -> None:
        """The ``###`` comment block appears before the ``.globl`` directive."""
        fn = Function(name="main", globl=True, source_file="foo.c", source_line=1, instructions=(Ret(),))
        lines = fn.emit().splitlines()
        hash_idx = next(i for i, ln in enumerate(lines) if ln == "###")
        globl_idx = next(i for i, ln in enumerate(lines) if ".globl" in ln)
        assert hash_idx < globl_idx

    def test_blank_line_before_comment_block(self) -> None:
        """A blank line immediately precedes the opening ``###``."""
        fn = Function(name="main", globl=True, source_file="foo.c", source_line=1, instructions=(Ret(),))
        lines = fn.emit().splitlines()
        hash_idx = next(i for i, ln in enumerate(lines) if ln == "###")
        assert lines[hash_idx - 1] == ""

    def test_existing_tests_unaffected_without_source_file(self) -> None:
        """Existing callers that omit ``source_file`` get the same output as before."""
        fn = Function(
            name="main", globl=True, instructions=(Mov(src=Imm(value=0), dst=Register(GPRegisters.AX)), Ret())
        )
        expected = (
            "      .globl main\n"
            "      .text\n"
            "main:\n"
            "      pushq %rbp\n"
            "      movq  %rsp, %rbp\n"
            "      movl  $0, %eax\n"
            "      movq  %rbp, %rsp\n"
            "      popq  %rbp\n"
            "      ret"
        )
        assert fn.emit() == expected


class TestProgram:
    """``Program`` is the root assembly AST node holding function definitions."""

    def test_construction(self) -> None:
        """``Program`` stores its top-level definitions."""
        fn = Function(name="main", globl=True, instructions=(Ret(),))
        prog = Program(definitions=(fn,))
        assert prog.definitions == (fn,)

    def test_equality(self) -> None:
        """Two ``Program`` nodes wrapping equal definitions are equal."""
        fn = Function(name="main", globl=True, instructions=(Ret(),))
        assert Program(definitions=(fn,)) == Program(definitions=(fn,))

    def test_immutable(self) -> None:
        """``Program`` is frozen - attribute assignment raises ``FrozenInstanceError``."""
        prog = Program(definitions=(Function(name="main", globl=False, instructions=()),))
        with pytest.raises(dataclasses.FrozenInstanceError):
            prog.definitions = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# New instruction nodes
# ---------------------------------------------------------------------------


class TestCmp:
    """``Cmp`` is a frozen dataclass for the ``cmpl`` comparison instruction."""

    def test_construction(self) -> None:
        """``Cmp`` stores its two operands."""
        cmp = Cmp(op1=Imm(value=1), op2=Register(GPRegisters.AX))
        assert cmp.op1 == Imm(value=1)
        assert cmp.op2 == Register(GPRegisters.AX)

    def test_is_instruction(self) -> None:
        """``Cmp`` is a subclass of ``Instruction``."""
        assert isinstance(Cmp(op1=Imm(value=0), op2=Register(GPRegisters.AX)), Instruction)

    def test_equality(self) -> None:
        """Two ``Cmp`` nodes with identical operands are equal."""
        assert Cmp(op1=Imm(value=1), op2=Register(GPRegisters.AX)) == Cmp(
            op1=Imm(value=1), op2=Register(GPRegisters.AX)
        )

    def test_inequality_different_op1(self) -> None:
        """``Cmp`` nodes with different ``op1`` values are not equal."""
        assert Cmp(op1=Imm(value=1), op2=Register(GPRegisters.AX)) != Cmp(
            op1=Imm(value=2), op2=Register(GPRegisters.AX)
        )

    def test_immutable(self) -> None:
        """``Cmp`` is frozen - attribute assignment raises ``FrozenInstanceError``."""
        cmp = Cmp(op1=Imm(value=0), op2=Register(GPRegisters.AX))
        with pytest.raises(dataclasses.FrozenInstanceError):
            cmp.op1 = Imm(value=99)  # type: ignore[misc]


class TestJmp:
    """``Jmp`` is a frozen dataclass for an unconditional jump instruction."""

    def test_construction(self) -> None:
        """``Jmp`` stores the target label name as a string."""
        assert Jmp(label="my_label").label == "my_label"

    def test_is_instruction(self) -> None:
        """``Jmp`` is a subclass of ``Instruction``."""
        assert isinstance(Jmp(label="foo"), Instruction)

    def test_equality(self) -> None:
        """Two ``Jmp`` nodes with the same label are equal."""
        assert Jmp(label="loop") == Jmp(label="loop")

    def test_inequality(self) -> None:
        """``Jmp`` nodes with different labels are not equal."""
        assert Jmp(label="a") != Jmp(label="b")


class TestJmpCC:
    """``JmpCC`` is a frozen dataclass for a conditional jump instruction."""

    def test_construction(self) -> None:
        """``JmpCC`` stores its condition code and label."""
        j = JmpCC(cond_code=CondCodes.E, label="done")
        assert j.cond_code == CondCodes.E
        assert j.label == "done"

    def test_is_instruction(self) -> None:
        """``JmpCC`` is a subclass of ``Instruction``."""
        assert isinstance(JmpCC(cond_code=CondCodes.NE, label="foo"), Instruction)

    def test_equality(self) -> None:
        """Two ``JmpCC`` nodes with the same code and label are equal."""
        assert JmpCC(cond_code=CondCodes.L, label="x") == JmpCC(cond_code=CondCodes.L, label="x")

    def test_inequality_different_cond_code(self) -> None:
        """``JmpCC`` nodes with different condition codes are not equal."""
        assert JmpCC(cond_code=CondCodes.E, label="x") != JmpCC(cond_code=CondCodes.NE, label="x")

    def test_inequality_different_label(self) -> None:
        """``JmpCC`` nodes with different labels are not equal."""
        assert JmpCC(cond_code=CondCodes.E, label="a") != JmpCC(cond_code=CondCodes.E, label="b")


class TestSetCC:
    """``SetCC`` is a frozen dataclass for the conditional byte-set instruction."""

    def test_construction(self) -> None:
        """``SetCC`` stores its condition code and operand."""
        s = SetCC(cond_code=CondCodes.GE, operand=Register(GPRegisters.AX, Width.BYTE))
        assert s.cond_code == CondCodes.GE
        assert s.operand == Register(GPRegisters.AX, Width.BYTE)

    def test_is_instruction(self) -> None:
        """``SetCC`` is a subclass of ``Instruction``."""
        assert isinstance(
            SetCC(cond_code=CondCodes.E, operand=Memory(reg=Register(GPRegisters.BP, Width.QWORD), offset=-4)),
            Instruction,
        )

    def test_equality(self) -> None:
        """Two ``SetCC`` nodes with the same code and operand are equal."""
        op = Register(GPRegisters.AX, Width.BYTE)
        assert SetCC(cond_code=CondCodes.E, operand=op) == SetCC(cond_code=CondCodes.E, operand=op)

    def test_inequality_different_cond_code(self) -> None:
        """``SetCC`` nodes with different condition codes are not equal."""
        op = Memory(reg=Register(GPRegisters.BP, Width.QWORD), offset=-4)
        assert SetCC(cond_code=CondCodes.E, operand=op) != SetCC(cond_code=CondCodes.NE, operand=op)


class TestLabel:
    """``Label`` is a frozen dataclass representing a local label declaration."""

    def test_construction(self) -> None:
        """``Label`` stores the label name."""
        assert Label(name="loop_start").name == "loop_start"

    def test_is_instruction(self) -> None:
        """``Label`` is a subclass of ``Instruction``."""
        assert isinstance(Label(name="foo"), Instruction)

    def test_equality(self) -> None:
        """Two ``Label`` nodes with the same name are equal."""
        assert Label(name="end") == Label(name="end")

    def test_immutable(self) -> None:
        """``Label`` is frozen - attribute assignment raises ``FrozenInstanceError``."""
        lbl = Label(name="start")
        with pytest.raises(dataclasses.FrozenInstanceError):
            lbl.name = "other"  # type: ignore[misc]


class TestAllocateStack:
    """``AllocateStack`` is a frozen dataclass for a stack-reservation instruction."""

    def test_construction(self) -> None:
        """``AllocateStack`` stores the byte count."""
        assert AllocateStack(size=16).size == 16

    def test_is_instruction(self) -> None:
        """``AllocateStack`` is a subclass of ``Instruction``."""
        assert isinstance(AllocateStack(size=0), Instruction)

    def test_equality(self) -> None:
        """Two ``AllocateStack`` nodes with the same size are equal."""
        assert AllocateStack(size=32) == AllocateStack(size=32)

    def test_inequality(self) -> None:
        """``AllocateStack`` nodes with different sizes are not equal."""
        assert AllocateStack(size=16) != AllocateStack(size=32)

    def test_immutable(self) -> None:
        """``AllocateStack`` is frozen - attribute assignment raises ``FrozenInstanceError``."""
        a = AllocateStack(size=8)
        with pytest.raises(dataclasses.FrozenInstanceError):
            a.size = 16  # type: ignore[misc]


class TestDeallocateStack:
    """``DeallocateStack`` is a frozen dataclass for a stack-reclamation instruction."""

    def test_construction(self) -> None:
        """``DeallocateStack`` stores the byte count."""
        assert DeallocateStack(size=16).size == 16

    def test_is_instruction(self) -> None:
        """``DeallocateStack`` is a subclass of ``Instruction``."""
        assert isinstance(DeallocateStack(size=0), Instruction)

    def test_equality(self) -> None:
        """Two ``DeallocateStack`` nodes with the same size are equal."""
        assert DeallocateStack(size=32) == DeallocateStack(size=32)

    def test_inequality(self) -> None:
        """``DeallocateStack`` nodes with different sizes are not equal."""
        assert DeallocateStack(size=16) != DeallocateStack(size=32)

    def test_immutable(self) -> None:
        """``DeallocateStack`` is frozen - attribute assignment raises ``FrozenInstanceError``."""
        d = DeallocateStack(size=8)
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.size = 16  # type: ignore[misc]


class TestPush:
    """``Push`` is a frozen dataclass for a stack-push instruction."""

    def test_construction(self) -> None:
        """``Push`` stores the operand to push."""
        p = Push(operand=Register(GPRegisters.AX, Width.QWORD))
        assert p.operand == Register(GPRegisters.AX, Width.QWORD)

    def test_is_instruction(self) -> None:
        """``Push`` is a subclass of ``Instruction``."""
        assert isinstance(Push(operand=Imm(value=0)), Instruction)

    def test_equality(self) -> None:
        """Two ``Push`` nodes with the same operand are equal."""
        assert Push(operand=Imm(value=1)) == Push(operand=Imm(value=1))

    def test_inequality(self) -> None:
        """``Push`` nodes with different operands are not equal."""
        assert Push(operand=Imm(value=1)) != Push(operand=Imm(value=2))

    def test_immutable(self) -> None:
        """``Push`` is frozen - attribute assignment raises ``FrozenInstanceError``."""
        p = Push(operand=Imm(value=0))
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.operand = Imm(value=99)  # type: ignore[misc]


class TestCall:
    """``Call`` is a frozen dataclass for a function-call instruction."""

    def test_construction(self) -> None:
        """``Call`` stores the function name."""
        assert Call(name="foo").name == "foo"

    def test_is_instruction(self) -> None:
        """``Call`` is a subclass of ``Instruction``."""
        assert isinstance(Call(name="foo"), Instruction)

    def test_equality(self) -> None:
        """Two ``Call`` nodes with the same name are equal."""
        assert Call(name="bar") == Call(name="bar")

    def test_inequality(self) -> None:
        """``Call`` nodes with different names are not equal."""
        assert Call(name="foo") != Call(name="bar")

    def test_immutable(self) -> None:
        """``Call`` is frozen - attribute assignment raises ``FrozenInstanceError``."""
        c = Call(name="foo")
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.name = "bar"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Byte-width instruction emit
# ---------------------------------------------------------------------------

_AX_BYTE = Register(GPRegisters.AX, Width.BYTE)
_AX_DWORD = Register(GPRegisters.AX, Width.DWORD)
_AX_QWORD = Register(GPRegisters.AX, Width.QWORD)
_R10_BYTE = Register(GPRegisters.R10, Width.BYTE)
_R10_DWORD = Register(GPRegisters.R10, Width.DWORD)
_R10_QWORD = Register(GPRegisters.R10, Width.QWORD)


class TestMovByteEmit:
    """``Mov(Byte)`` emits ``movb``."""

    def test_mov_byte_reg_reg(self) -> None:
        """``Mov(Byte, reg, reg)`` emits ``movb``."""
        assert Mov(src=_R10_BYTE, dst=_AX_BYTE, assembly_type=Byte()).emit() == "      movb  %r10b, %al"

    def test_mov_longword_emits_movl(self) -> None:
        """``Mov(Longword)`` still emits ``movl`` (regression guard)."""
        assert Mov(src=_R10_DWORD, dst=_AX_DWORD, assembly_type=Longword()).emit() == "      movl  %r10d, %eax"

    def test_mov_quadword_emits_movq(self) -> None:
        """``Mov(Quadword)`` still emits ``movq`` (regression guard)."""
        assert Mov(src=_R10_QWORD, dst=_AX_QWORD, assembly_type=Quadword()).emit() == "      movq  %r10, %rax"


class TestCmpByteEmit:
    """``Cmp(Byte)`` emits ``cmpb``."""

    def test_cmp_byte_emits_cmpb(self) -> None:
        """``Cmp(Byte, reg, reg)`` emits ``cmpb``."""
        assert Cmp(op1=_R10_BYTE, op2=_AX_BYTE, assembly_type=Byte()).emit() == "      cmpb  %r10b, %al"

    def test_cmp_longword_regression(self) -> None:
        """``Cmp(Longword)`` still emits ``cmpl``."""
        assert Cmp(op1=_R10_DWORD, op2=_AX_DWORD, assembly_type=Longword()).emit() == "      cmpl  %r10d, %eax"


class TestMovsxEmit:
    """``Movsx.emit()`` selects the correct sign-extending mnemonic."""

    def test_byte_to_longword_emits_movsbl(self) -> None:
        """``Movsx(Byte, Longword)`` emits ``movsbl``."""
        instr = Movsx(src_type=Byte(), dst_type=Longword(), src=_R10_BYTE, dst=_AX_DWORD)
        assert instr.emit() == "      movsbl %r10b, %eax"

    def test_byte_to_quadword_emits_movsbq(self) -> None:
        """``Movsx(Byte, Quadword)`` emits ``movsbq``."""
        instr = Movsx(src_type=Byte(), dst_type=Quadword(), src=_R10_BYTE, dst=_AX_QWORD)
        assert instr.emit() == "      movsbq %r10b, %rax"

    def test_longword_to_quadword_emits_movslq(self) -> None:
        """``Movsx(Longword, Quadword)`` emits ``movslq`` (regression guard)."""
        instr = Movsx(src_type=Longword(), dst_type=Quadword(), src=_R10_DWORD, dst=_AX_QWORD)
        assert instr.emit() == "      movslq %r10d, %rax"


class TestMoveZeroExtendEmit:
    """``MoveZeroExtend.emit()`` selects the correct zero-extending mnemonic."""

    def test_byte_to_longword_emits_movzbl(self) -> None:
        """``MoveZeroExtend(Byte, Longword)`` emits ``movzbl``."""
        instr = MoveZeroExtend(src_type=Byte(), dst_type=Longword(), src=_R10_BYTE, dst=_AX_DWORD)
        assert instr.emit() == "      movzbl %r10b, %eax"

    def test_byte_to_quadword_emits_movzbq(self) -> None:
        """``MoveZeroExtend(Byte, Quadword)`` emits ``movzbq``."""
        instr = MoveZeroExtend(src_type=Byte(), dst_type=Quadword(), src=_R10_BYTE, dst=_AX_QWORD)
        assert instr.emit() == "      movzbq %r10b, %rax"

    def test_longword_to_quadword_emits_movl(self) -> None:
        """``MoveZeroExtend(Longword, Quadword)`` emits ``movl`` (regression guard)."""
        instr = MoveZeroExtend(src_type=Longword(), dst_type=Quadword(), src=_R10_DWORD, dst=_AX_DWORD)
        assert instr.emit() == "      movl  %r10d, %eax"


# ---------------------------------------------------------------------------
# StaticVariable / StaticConstant - StringInit comment
# ---------------------------------------------------------------------------


class TestStaticVariableStringComment:
    """``StaticVariable.emit()`` prepends a human-readable comment before ``StringInit`` bytes."""

    def test_simple_string_comment_present(self) -> None:
        """A comment line appears immediately before the ``.byte`` directive."""
        var = StaticVariable(identifier="s", globl=False, alignment=1, init=(StringInit(b"Hi", null_terminated=True),))
        lines = var.emit().splitlines()
        comment_idx = next(i for i, ln in enumerate(lines) if "# String:" in ln)
        assert lines[comment_idx + 1].startswith("      .byte")

    def test_comment_shows_printable_chars(self) -> None:
        """Printable ASCII bytes appear verbatim in the comment."""
        var = StaticVariable(
            identifier="s", globl=False, alignment=1, init=(StringInit(b"Hello", null_terminated=False),)
        )
        assert "      # String: Hello" in var.emit()

    def test_comment_appends_null_terminator(self) -> None:
        """When ``null_terminated=True`` the comment ends with ``\\0``."""
        var = StaticVariable(identifier="s", globl=False, alignment=1, init=(StringInit(b"Hi", null_terminated=True),))
        assert "      # String: Hi\\0" in var.emit()

    def test_comment_no_null_when_not_terminated(self) -> None:
        """When ``null_terminated=False`` the comment does not end with ``\\0``."""
        var = StaticVariable(identifier="s", globl=False, alignment=1, init=(StringInit(b"Hi", null_terminated=False),))
        lines = [ln for ln in var.emit().splitlines() if "# String:" in ln]
        assert lines[0] == "      # String: Hi"

    def test_comment_escapes_newline(self) -> None:
        """Newline bytes are rendered as ``\\n`` in the comment."""
        var = StaticVariable(
            identifier="s", globl=False, alignment=1, init=(StringInit(b"a\nb", null_terminated=False),)
        )
        assert r"\n" in var.emit()

    def test_comment_escapes_tab(self) -> None:
        """Tab bytes are rendered as ``\\t`` in the comment."""
        var = StaticVariable(
            identifier="s", globl=False, alignment=1, init=(StringInit(b"a\tb", null_terminated=False),)
        )
        assert r"\t" in var.emit()

    def test_comment_escapes_non_printable(self) -> None:
        """Non-printable bytes outside named escapes are rendered as ``\\xNN``."""
        var = StaticVariable(
            identifier="s", globl=False, alignment=1, init=(StringInit(bytes([0x01]), null_terminated=False),)
        )
        assert r"\x01" in var.emit()


class TestStaticConstantStringComment:
    """``StaticConstant.emit()`` prepends a human-readable comment before ``StringInit`` bytes."""

    def test_comment_present(self) -> None:
        """A comment line appears immediately before the ``.byte`` directive."""
        const = StaticConstant(identifier="s", alignment=1, init=StringInit(b"Hi", null_terminated=True))
        lines = const.emit().splitlines()
        comment_idx = next(i for i, ln in enumerate(lines) if "# String:" in ln)
        assert lines[comment_idx + 1].startswith("      .byte")

    def test_comment_content(self) -> None:
        """Comment shows printable content plus ``\\0`` for null-terminated strings."""
        const = StaticConstant(identifier="s", alignment=1, init=StringInit(b"Hi", null_terminated=True))
        assert "      # String: Hi\\0" in const.emit()


# ---------------------------------------------------------------------------
# Word AssemblyType
# ---------------------------------------------------------------------------


class TestWordAssemblyType:
    """``Word`` is a concrete frozen dataclass for 16-bit assembly type."""

    def test_word_is_assembly_type(self) -> None:
        """``Word()`` is an instance of ``AssemblyType``."""
        from compiler.syntax.asm.values import AssemblyType

        assert isinstance(Word(), AssemblyType)

    def test_word_equality(self) -> None:
        """Two ``Word()`` instances compare equal."""
        assert Word() == Word()

    def test_word_not_equal_to_byte(self) -> None:
        """``Word()`` is not equal to ``Byte()``."""
        assert Word() != Byte()

    def test_word_not_equal_to_longword(self) -> None:
        """``Word()`` is not equal to ``Longword()``."""
        assert Word() != Longword()


# ---------------------------------------------------------------------------
# Short and __int128 StaticVariable emission
# ---------------------------------------------------------------------------


class TestShortStaticVariableEmit:
    """``StaticVariable.emit()`` handles ``ShortInit`` and ``UShortInit``."""

    def test_short_init_nonzero_emits_short_directive(self) -> None:
        """``ShortInit(42)`` emits a ``.short 42`` directive."""
        var = StaticVariable(identifier="s", globl=False, alignment=2, init=(ShortInit(42),))
        assert "      .short 42" in var.emit()

    def test_short_init_zero_goes_to_bss(self) -> None:
        """``ShortInit(0)`` triggers ``.bss`` section selection."""
        var = StaticVariable(identifier="s", globl=False, alignment=2, init=(ShortInit(0),))
        assert ".bss" in var.emit()

    def test_ushort_init_nonzero_emits_short_directive(self) -> None:
        """``UShortInit(65535)`` emits a ``.short 65535`` directive."""
        var = StaticVariable(identifier="u", globl=False, alignment=2, init=(UShortInit(65535),))
        assert "      .short 65535" in var.emit()

    def test_ushort_init_zero_goes_to_bss(self) -> None:
        """``UShortInit(0)`` triggers ``.bss`` section selection."""
        var = StaticVariable(identifier="u", globl=False, alignment=2, init=(UShortInit(0),))
        assert ".bss" in var.emit()

    def test_short_init_negative_emits_signed_value(self) -> None:
        """``ShortInit(-1)`` emits a ``.short -1`` directive."""
        var = StaticVariable(identifier="s", globl=False, alignment=2, init=(ShortInit(-1),))
        assert "      .short -1" in var.emit()


class TestInt128StaticVariableEmit:
    """``StaticVariable.emit()`` handles ``Int128Init`` and ``UInt128Init``."""

    def test_int128_init_nonzero_emits_two_quad_directives(self) -> None:
        """``Int128Init(1)`` emits two ``.quad`` directives (lo=1, hi=0)."""
        var = StaticVariable(identifier="x", globl=False, alignment=16, init=(Int128Init(1),))
        lines = [ln.strip() for ln in var.emit().splitlines()]
        quads = [ln for ln in lines if ln.startswith(".quad")]
        assert len(quads) == 2
        assert quads[0] == ".quad 1"
        assert quads[1] == ".quad 0"

    def test_int128_init_zero_goes_to_bss(self) -> None:
        """``Int128Init(0)`` triggers ``.bss`` section selection."""
        var = StaticVariable(identifier="x", globl=False, alignment=16, init=(Int128Init(0),))
        assert ".bss" in var.emit()

    def test_uint128_init_large_value_splits_correctly(self) -> None:
        """``UInt128Init`` with a value spanning both halves emits correct lo/hi quads."""
        val = (0xDEADBEEFCAFEBABE << 64) | 0x0102030405060708
        var = StaticVariable(identifier="x", globl=False, alignment=16, init=(UInt128Init(val),))
        lines = [ln.strip() for ln in var.emit().splitlines()]
        quads = [ln for ln in lines if ln.startswith(".quad")]
        assert len(quads) == 2
        assert quads[0] == f".quad {0x0102030405060708}"
        assert quads[1] == f".quad {0xDEADBEEFCAFEBABE}"

    def test_int128_negative_splits_to_signed_halves(self) -> None:
        """``Int128Init(-1)`` emits lo=-1 as unsigned 64-bit max and hi=-1 as unsigned 64-bit max."""
        var = StaticVariable(identifier="x", globl=False, alignment=16, init=(Int128Init(-1),))
        lines = [ln.strip() for ln in var.emit().splitlines()]
        quads = [ln for ln in lines if ln.startswith(".quad")]
        assert len(quads) == 2
        mask64 = (1 << 64) - 1
        assert quads[0] == f".quad {mask64}"
        assert quads[1] == f".quad {mask64}"
