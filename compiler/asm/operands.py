"""Operand nodes for the assembly AST.

Defines the ``Operand`` abstract base class and its concrete variants. Each
operand form supported by the code generator is represented as a subclass of
``Operand``.
"""

from abc import ABC
from dataclasses import dataclass


class Operand(ABC):
    """Abstract base for all assembly operand nodes.

    Each operand form is represented as a concrete subclass. Consumers should
    use structural pattern matching (``match``/``case``) to dispatch on the
    concrete variant rather than inspecting fields directly.
    """


@dataclass(frozen=True, slots=True)
class Imm(Operand):
    """An immediate (literal integer) operand.

    Attributes:
        value: The integer value of the immediate.
    """

    value: int


@dataclass(frozen=True, slots=True)
class Register(Operand):
    """The ``%eax`` general-purpose register.

    Carries no fields — in the current grammar only a single register is
    used, so its identity is its type alone.
    """
