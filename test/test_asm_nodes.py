"""Tests for the assembly AST node definitions.

Verifies construction, equality, immutability, and type hierarchy for all
node types in the ``compiler.asm`` package.
"""

import dataclasses

import pytest

from compiler.asm import Function, Imm, Instruction, Mov, Operand, Program, Register, Ret

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
    """``Register`` is a no-field frozen dataclass representing ``%eax``."""

    def test_construction(self) -> None:
        """``Register`` can be constructed with no arguments."""
        Register()

    def test_is_operand(self) -> None:
        """``Register`` is a subclass of ``Operand``."""
        assert isinstance(Register(), Operand)

    def test_equality(self) -> None:
        """Two ``Register`` instances are equal."""
        assert Register() == Register()

    def test_not_equal_to_imm(self) -> None:
        """``Register`` is not equal to an ``Imm`` node."""
        assert Register() != Imm(value=0)


# ---------------------------------------------------------------------------
# Instruction nodes
# ---------------------------------------------------------------------------


class TestMov:
    """``Mov`` is a concrete frozen dataclass for a move instruction."""

    def test_construction(self) -> None:
        """``Mov`` stores its source and destination operands."""
        mov = Mov(src=Imm(value=3), dst=Register())
        assert mov.src == Imm(value=3)
        assert mov.dst == Register()

    def test_is_instruction(self) -> None:
        """``Mov`` is a subclass of ``Instruction``."""
        assert isinstance(Mov(src=Imm(value=0), dst=Register()), Instruction)

    def test_equality(self) -> None:
        """Two ``Mov`` nodes with identical operands are equal."""
        assert Mov(src=Imm(value=1), dst=Register()) == Mov(src=Imm(value=1), dst=Register())

    def test_inequality_different_src(self) -> None:
        """``Mov`` nodes with different source operands are not equal."""
        assert Mov(src=Imm(value=1), dst=Register()) != Mov(src=Imm(value=2), dst=Register())

    def test_immutable(self) -> None:
        """``Mov`` is frozen — attribute assignment raises ``FrozenInstanceError``."""
        mov = Mov(src=Imm(value=0), dst=Register())
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
        assert Ret() != Mov(src=Imm(value=0), dst=Register())


# ---------------------------------------------------------------------------
# Declaration nodes
# ---------------------------------------------------------------------------


class TestFunction:
    """``Function`` is a frozen dataclass holding a name and instruction sequence."""

    def test_construction(self) -> None:
        """``Function`` stores its name and instruction tuple."""
        fn = Function(name="main", instructions=(Mov(src=Imm(value=0), dst=Register()), Ret()))
        assert fn.name == "main"
        assert len(fn.instructions) == 2

    def test_empty_instructions(self) -> None:
        """``Function`` accepts an empty instruction tuple."""
        fn = Function(name="main", instructions=())
        assert fn.instructions == ()

    def test_equality(self) -> None:
        """Two ``Function`` nodes with the same name and instructions are equal."""
        fn1 = Function(name="foo", instructions=(Ret(),))
        fn2 = Function(name="foo", instructions=(Ret(),))
        assert fn1 == fn2

    def test_immutable(self) -> None:
        """``Function`` is frozen — attribute assignment raises ``FrozenInstanceError``."""
        fn = Function(name="main", instructions=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            fn.name = "other"  # type: ignore[misc]


class TestProgram:
    """``Program`` is the root assembly AST node holding a single function."""

    def test_construction(self) -> None:
        """``Program`` stores its single function definition."""
        fn = Function(name="main", instructions=(Ret(),))
        prog = Program(function=fn)
        assert prog.function is fn

    def test_equality(self) -> None:
        """Two ``Program`` nodes wrapping equal functions are equal."""
        fn = Function(name="main", instructions=(Ret(),))
        assert Program(function=fn) == Program(function=fn)

    def test_immutable(self) -> None:
        """``Program`` is frozen — attribute assignment raises ``FrozenInstanceError``."""
        prog = Program(function=Function(name="main", instructions=()))
        with pytest.raises(dataclasses.FrozenInstanceError):
            prog.function = Function(name="other", instructions=())  # type: ignore[misc]
