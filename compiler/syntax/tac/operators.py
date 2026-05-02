"""Operator nodes for the tac AST.

Defines the ``UnaryOperator`` and ``BinaryOperator`` abstract base classes and their
concrete variants.  Each instruction form supported by the code generator is
represented as a subclass of their respective abstract base.
"""

from abc import ABC
from dataclasses import dataclass


class UnaryOperator(ABC):
    """Abstract base for all tac Unary Operation nodes.

    Each instruction form is represented as a concrete subclass. Consumers
    should use structural pattern matching (``match``/``case``) to dispatch on
    the concrete variant rather than inspecting fields directly.

    Unary operators are completely self-defined so they contain no attributes
    """


@dataclass(frozen=True, slots=True)
class Complement(UnaryOperator):
    """Represents the binary complement of an expression."""


@dataclass(frozen=True, slots=True)
class Negate(UnaryOperator):
    """Represents the negation of an expression."""


class BinaryOperator(ABC):
    """Abstract base for all tac Binary Operation nodes.

    Each instruction form is represented as a concrete subclass. Consumers
    should use structural pattern matching (``match``/``case``) to dispatch on
    the concrete variant rather than inspecting fields directly.

    Binary operators are completely self-defined so they contain no attributes
    """


@dataclass(frozen=True, slots=True)
class Add(BinaryOperator):
    """Represents the addition of two expressions."""


@dataclass(frozen=True, slots=True)
class Subtract(BinaryOperator):
    """Represents the subtraction of two expressions."""


@dataclass(frozen=True, slots=True)
class Multiply(BinaryOperator):
    """Represents the multiplication of two expressions."""


@dataclass(frozen=True, slots=True)
class Divide(BinaryOperator):
    """Represents the division of two expressions."""


@dataclass(frozen=True, slots=True)
class Remainder(BinaryOperator):
    """Represents the remainder of the division of two expressions."""


@dataclass(frozen=True, slots=True)
class LeftShift(BinaryOperator):
    """Represents a bitwise left shift (``<<``) of two expressions."""


@dataclass(frozen=True, slots=True)
class RightShift(BinaryOperator):
    """Represents an arithmetic bitwise right shift (``>>``) of two expressions."""


@dataclass(frozen=True, slots=True)
class BitwiseAnd(BinaryOperator):
    """Represents the bitwise AND (``&``) of two expressions."""


@dataclass(frozen=True, slots=True)
class BitwiseXor(BinaryOperator):
    """Represents the bitwise exclusive OR (``^``) of two expressions."""


@dataclass(frozen=True, slots=True)
class BitwiseOr(BinaryOperator):
    """Represents the bitwise OR (``|``) of two expressions."""
