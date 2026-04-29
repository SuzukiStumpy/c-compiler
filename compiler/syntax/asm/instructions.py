"""Instruction nodes for the assembly AST.

Defines the ``Instruction`` abstract base class and its concrete variants.
Each instruction form supported by the code generator is represented as a
subclass of ``Instruction``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from compiler.syntax.asm.operands import Operand


class Instruction(ABC):
    """Abstract base for all assembly instruction nodes.

    Each instruction form is represented as a concrete subclass. Consumers
    should use structural pattern matching (``match``/``case``) to dispatch on
    the concrete variant rather than inspecting fields directly.
    """

    @abstractmethod
    def emit(self) -> str:
        """Return the assembly text for this instruction, indented by two spaces.

        Returns:
            A string beginning with two spaces followed by the mnemonic and
            any operands, ready to be written directly to an assembly file.
        """


@dataclass(frozen=True, slots=True)
class Mov(Instruction):
    """A ``movl`` instruction that copies a value from one operand to another.

    Attributes:
        src: The source operand whose value is copied.
        dst: The destination operand that receives the value.
    """

    src: Operand
    dst: Operand

    def emit(self) -> str:
        """Return a ``movl`` instruction in AT&T syntax.

        Returns:
            A string of the form ``  movl <src>, <dst>``.
        """
        return f"  movl {self.src.emit()}, {self.dst.emit()}"


@dataclass(frozen=True, slots=True)
class Ret(Instruction):
    """A ``ret`` instruction that returns from the current function.

    Carries no fields — the instruction has no operands.
    """

    def emit(self) -> str:
        """Return the ``ret`` mnemonic.

        Returns:
            The string ``  ret``.
        """
        return "  ret"
