"""Operator nodes for the AST.

Defines the ``UnaryOperator`` abstract base class and its concrete variants.
Each operator form supported by the compiler is represented as a subclass of
``UnaryOperator``.  New operator types will be added here as language support is
extended.
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
    """Enacts unary negation for an expression

    Attributes:
        None
    """
