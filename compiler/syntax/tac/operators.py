"""Operator nodes for the tac AST.

Defines the ``UnaryOperator`` abstract base class and its concrete variants.
Each instruction form supported by the code generator is represented as a
subclass of ``UnaryOperator``.
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
