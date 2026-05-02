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


class BinaryOperator(ABC):
    """Abstract base for all assembly binary operator nodes.

    Each operator is represented as a concrete subclass carrying no data fields
    — its identity is its type alone.  Consumers should use structural pattern
    matching (``match``/``case``) to dispatch on the concrete variant rather than
    inspecting the type directly.
    """

    @abstractmethod
    def emit(self) -> str:
        """Return the assembly mnemonic for this operator.

        Returns:
            The instruction mnemonic string (e.g. ``"addl"``, ``"subl"``, or
            ``"imull"``).
        """


@dataclass(frozen=True, slots=True)
class Add(BinaryOperator):
    """The ``addl`` instruction: addition.

    Adds the first operand to the second operand, storing the result in place
    in the second operand.
    """

    def emit(self) -> str:
        """Return the ``addl`` mnemonic.

        Returns:
            The string ``"addl"``.
        """
        return "addl"


@dataclass(frozen=True, slots=True)
class Sub(BinaryOperator):
    """The ``subl`` instruction: subtraction.

    Subtracts the first operand from the second operand, storing the result in
    place in the second operand.
    """

    def emit(self) -> str:
        """Return the ``subl`` mnemonic.

        Returns:
            The string ``"subl"``.
        """
        return "subl"


@dataclass(frozen=True, slots=True)
class Mult(BinaryOperator):
    """The ``imull`` instruction: multiplication.

    Multiplies the first operand by the second operand, storing the result in
    place in the second operand.
    """

    def emit(self) -> str:
        """Return the ``imull`` mnemonic.

        Returns:
            The string ``"imull"``.
        """
        return "imull"


@dataclass(frozen=True, slots=True)
class And(BinaryOperator):
    """The ``andl`` instruction: bitwise AND.

    Computes the bitwise AND of two operands, storing the result in the second.
    """

    def emit(self) -> str:
        """Return the ``andl`` mnemonic.

        Returns:
            The string ``"andl"``.
        """
        return "andl"


@dataclass(frozen=True, slots=True)
class Or(BinaryOperator):
    """The ``orl`` instruction: bitwise OR.

    Computes the bitwise OR of two operands, storing the result in the second.
    """

    def emit(self) -> str:
        """Return the ``orl`` mnemonic.

        Returns:
            The string ``"orl"``.
        """
        return "orl"


@dataclass(frozen=True, slots=True)
class Xor(BinaryOperator):
    """The ``xorl`` instruction: bitwise exclusive OR.

    Computes the bitwise XOR of two operands, storing the result in the second.
    """

    def emit(self) -> str:
        """Return the ``xorl`` mnemonic.

        Returns:
            The string ``"xorl"``.
        """
        return "xorl"
