"""Unary operator nodes for the assembly AST.

Defines the ``UnaryOperator`` abstract base class and its concrete variants.
Each operator form corresponds to a single x86 mnemonic (``negl``, ``notl``)
and is represented as a concrete frozen dataclass subclassing ``UnaryOperator``.

Consumers should use structural pattern matching (``match``/``case``) or call
``emit()`` rather than inspecting the concrete type directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class UnaryOperator(ABC):
    """Abstract base for all assembly unary operator nodes.

    Each operator is represented as a concrete subclass carrying no data fields
    — its identity is its type alone.  Consumers should use structural pattern
    matching (``match``/``case``) to dispatch on the concrete variant rather than
    inspecting the type directly.
    """

    @abstractmethod
    def emit(self) -> str:
        """Return the assembly mnemonic for this operator.

        Returns:
            The instruction mnemonic string (e.g. ``"negl"`` or ``"notl"``).
        """


@dataclass(frozen=True, slots=True)
class Neg(UnaryOperator):
    """The ``negl`` instruction: arithmetic (two's-complement) negation.

    Negates its single operand in place.
    """

    def emit(self) -> str:
        """Return the ``negl`` mnemonic.

        Returns:
            The string ``"negl"``.
        """
        return "negl"


@dataclass(frozen=True, slots=True)
class Not(UnaryOperator):
    """The ``notl`` instruction: bitwise complement.

    Inverts every bit of its single operand in place.
    """

    def emit(self) -> str:
        """Return the ``notl`` mnemonic.

        Returns:
            The string ``"notl"``.
        """
        return "notl"
