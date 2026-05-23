"""Tests for the assembly AST node definitions.

Verifies construction, equality, immutability, and type hierarchy for all
node types in the ``compiler.syntax.asm`` package.
"""

import dataclasses

import pytest

from compiler.syntax.asm import (
    AllocateStack,
    Call,
    Cmp,
    CondCodes,
    DeallocateStack,
    Function,
    Imm,
    Instruction,
    Jmp,
    JmpCC,
    Label,
    Memory,
    Mov,
    Operand,
    Program,
    Push,
    Register,
    Registers,
    Ret,
    SetCC,
    Width,
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
        """``Imm`` is frozen — attribute assignment raises ``FrozenInstanceError``."""
        imm = Imm(value=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            imm.value = 99  # type: ignore[misc]


class TestRegister:
    """``Register`` wraps a ``Registers`` enum value and an optional ``Width``."""

    def test_construction(self) -> None:
        """``Register`` can be constructed with just a register name."""
        Register(Registers.AX)

    def test_is_operand(self) -> None:
        """``Register`` is a subclass of ``Operand``."""
        assert isinstance(Register(Registers.AX), Operand)

    def test_equality(self) -> None:
        """Two ``Register`` instances with the same reg and default width are equal."""
        assert Register(Registers.AX) == Register(Registers.AX)

    def test_not_equal_to_imm(self) -> None:
        """``Register`` is not equal to an ``Imm`` node."""
        assert Register(Registers.AX) != Imm(value=0)

    def test_default_width_is_dword(self) -> None:
        """The default width for a ``Register`` is ``Width.DWORD``."""
        assert Register(Registers.AX).width == Width.DWORD

    def test_explicit_byte_width(self) -> None:
        """``Register`` can be constructed with ``Width.BYTE``."""
        assert Register(Registers.AX, Width.BYTE).width == Width.BYTE

    def test_different_widths_not_equal(self) -> None:
        """``Register`` nodes for the same logical register at different widths are not equal."""
        assert Register(Registers.AX, Width.DWORD) != Register(Registers.AX, Width.BYTE)


# ---------------------------------------------------------------------------
# Instruction nodes
# ---------------------------------------------------------------------------


class TestMov:
    """``Mov`` is a concrete frozen dataclass for a move instruction."""

    def test_construction(self) -> None:
        """``Mov`` stores its source and destination operands."""
        mov = Mov(src=Imm(value=3), dst=Register(Registers.AX))
        assert mov.src == Imm(value=3)
        assert mov.dst == Register(Registers.AX)

    def test_is_instruction(self) -> None:
        """``Mov`` is a subclass of ``Instruction``."""
        assert isinstance(Mov(src=Imm(value=0), dst=Register(Registers.AX)), Instruction)

    def test_equality(self) -> None:
        """Two ``Mov`` nodes with identical operands are equal."""
        assert Mov(src=Imm(value=1), dst=Register(Registers.AX)) == Mov(src=Imm(value=1), dst=Register(Registers.AX))

    def test_inequality_different_src(self) -> None:
        """``Mov`` nodes with different source operands are not equal."""
        assert Mov(src=Imm(value=1), dst=Register(Registers.AX)) != Mov(src=Imm(value=2), dst=Register(Registers.AX))

    def test_immutable(self) -> None:
        """``Mov`` is frozen — attribute assignment raises ``FrozenInstanceError``."""
        mov = Mov(src=Imm(value=0), dst=Register(Registers.AX))
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
        assert Ret() != Mov(src=Imm(value=0), dst=Register(Registers.AX))


# ---------------------------------------------------------------------------
# Declaration nodes
# ---------------------------------------------------------------------------


class TestFunction:
    """``Function`` is a frozen dataclass holding a name and instruction sequence."""

    def test_construction(self) -> None:
        """``Function`` stores its name and instruction tuple."""
        fn = Function(name="main", globl=True, instructions=(Mov(src=Imm(value=0), dst=Register(Registers.AX)), Ret()))
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
        """``Function`` is frozen — attribute assignment raises ``FrozenInstanceError``."""
        fn = Function(name="main", globl=False, instructions=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            fn.name = "other"  # type: ignore[misc]


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
        """``Program`` is frozen — attribute assignment raises ``FrozenInstanceError``."""
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
        cmp = Cmp(op1=Imm(value=1), op2=Register(Registers.AX))
        assert cmp.op1 == Imm(value=1)
        assert cmp.op2 == Register(Registers.AX)

    def test_is_instruction(self) -> None:
        """``Cmp`` is a subclass of ``Instruction``."""
        assert isinstance(Cmp(op1=Imm(value=0), op2=Register(Registers.AX)), Instruction)

    def test_equality(self) -> None:
        """Two ``Cmp`` nodes with identical operands are equal."""
        assert Cmp(op1=Imm(value=1), op2=Register(Registers.AX)) == Cmp(op1=Imm(value=1), op2=Register(Registers.AX))

    def test_inequality_different_op1(self) -> None:
        """``Cmp`` nodes with different ``op1`` values are not equal."""
        assert Cmp(op1=Imm(value=1), op2=Register(Registers.AX)) != Cmp(op1=Imm(value=2), op2=Register(Registers.AX))

    def test_immutable(self) -> None:
        """``Cmp`` is frozen — attribute assignment raises ``FrozenInstanceError``."""
        cmp = Cmp(op1=Imm(value=0), op2=Register(Registers.AX))
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
        s = SetCC(cond_code=CondCodes.GE, operand=Register(Registers.AX, Width.BYTE))
        assert s.cond_code == CondCodes.GE
        assert s.operand == Register(Registers.AX, Width.BYTE)

    def test_is_instruction(self) -> None:
        """``SetCC`` is a subclass of ``Instruction``."""
        assert isinstance(
            SetCC(cond_code=CondCodes.E, operand=Memory(reg=Register(Registers.BP, Width.QWORD), offset=-4)),
            Instruction,
        )

    def test_equality(self) -> None:
        """Two ``SetCC`` nodes with the same code and operand are equal."""
        op = Register(Registers.AX, Width.BYTE)
        assert SetCC(cond_code=CondCodes.E, operand=op) == SetCC(cond_code=CondCodes.E, operand=op)

    def test_inequality_different_cond_code(self) -> None:
        """``SetCC`` nodes with different condition codes are not equal."""
        op = Memory(reg=Register(Registers.BP, Width.QWORD), offset=-4)
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
        """``Label`` is frozen — attribute assignment raises ``FrozenInstanceError``."""
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
        """``AllocateStack`` is frozen — attribute assignment raises ``FrozenInstanceError``."""
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
        """``DeallocateStack`` is frozen — attribute assignment raises ``FrozenInstanceError``."""
        d = DeallocateStack(size=8)
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.size = 16  # type: ignore[misc]


class TestPush:
    """``Push`` is a frozen dataclass for a stack-push instruction."""

    def test_construction(self) -> None:
        """``Push`` stores the operand to push."""
        p = Push(operand=Register(Registers.AX, Width.QWORD))
        assert p.operand == Register(Registers.AX, Width.QWORD)

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
        """``Push`` is frozen — attribute assignment raises ``FrozenInstanceError``."""
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
        """``Call`` is frozen — attribute assignment raises ``FrozenInstanceError``."""
        c = Call(name="foo")
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.name = "bar"  # type: ignore[misc]
