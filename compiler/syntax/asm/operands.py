"""Operand nodes for the assembly AST.

Defines the ``Operand`` abstract base class and its concrete variants. Each
operand form supported by the code generator is represented as a subclass of
``Operand``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class Operand(ABC):
    """Abstract base for all assembly operand nodes.

    Each operand form is represented as a concrete subclass. Consumers should
    use structural pattern matching (``match``/``case``) to dispatch on the
    concrete variant rather than inspecting fields directly.
    """

    @abstractmethod
    def emit(self) -> str:
        """Return the assembly text representation of this operand.

        Returns:
            A string suitable for use as an operand in an assembly instruction.
        """


@dataclass(frozen=True, slots=True)
class Imm(Operand):
    """An immediate (literal integer) operand.

    Attributes:
        value: The integer value of the immediate.
    """

    value: int

    def emit(self) -> str:
        """Return the immediate in AT&T syntax (e.g. ``$42``).

        Returns:
            The immediate value prefixed with ``$``.
        """
        return f"${self.value}"


@dataclass(frozen=True, slots=True)
class Register(Operand):
    """The ``%eax`` general-purpose register.

    Carries no fields — in the current grammar only a single register is
    used, so its identity is its type alone.
    """

    def emit(self) -> str:
        """Return the register name in AT&T syntax.

        Returns:
            The string ``%eax``.
        """
        return "%eax"
