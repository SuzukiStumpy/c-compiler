"""Operator nodes for the AST.

Defines the ``UnaryOperator`` and ``BinaryOperator`` abstract base classes and
their concrete variants.  Each operator form supported by the compiler is
represented as a subclass of the appropriate base.  New operator types are added
here as language support is extended.
"""

from abc import ABC
from dataclasses import dataclass


class UnaryOperator(ABC):
    """Abstract base for all unary operator nodes.

    Each operator supported by the compiler is represented by a concrete subclass.
    Consumers should use structural pattern matching (``match``/``case``) to
    dispatch on the concrete variant rather than inspecting fields directly.
    """


@dataclass(frozen=True, slots=True)
class Complement(UnaryOperator):
    """A bitwise complement of an expression.

    Attributes:
        None
    """


@dataclass(frozen=True, slots=True)
class Negate(UnaryOperator):
    """Enacts unary negation for an expression.

    Attributes:
        None
    """


class BinaryOperator(ABC):
    """Abstract base for all binary operator nodes.

    Each operator supported by the compiler is represented by a concrete subclass.
    Consumers should use structural pattern matching (``match``/``case``) to
    dispatch on the concrete variant rather than inspecting fields directly.
    """


@dataclass(frozen=True, slots=True)
class Add(BinaryOperator):
    """Addition of two expressions (``+``).

    Attributes:
        None
    """


@dataclass(frozen=True, slots=True)
class Subtract(BinaryOperator):
    """Subtraction of one expression from another (``-``).

    Attributes:
        None
    """


@dataclass(frozen=True, slots=True)
class Multiply(BinaryOperator):
    """Multiplication of two expressions (``*``).

    Attributes:
        None
    """


@dataclass(frozen=True, slots=True)
class Divide(BinaryOperator):
    """Integer division of two expressions (``/``).

    Attributes:
        None
    """


@dataclass(frozen=True, slots=True)
class Remainder(BinaryOperator):
    """Remainder after integer division of two expressions (``%``).

    Attributes:
        None
    """
