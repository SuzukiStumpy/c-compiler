"""Value nodes for the tac AST.

Defines the ``Val`` abstract base class and its concrete variants.
Each instruction form supported by the code generator is represented as a
subclass of ``Val``.
"""

from abc import ABC
from dataclasses import dataclass


class Val(ABC):
    """Abstract base for all tac Value nodes.

    Each instruction form is represented as a concrete subclass. Consumers
    should use structural pattern matching (``match``/``case``) to dispatch on
    the concrete variant rather than inspecting fields directly.
    """


@dataclass(frozen=True, slots=True)
class Constant(Val):
    """Class representing an integer constant within the tac.

    Attributes:
        value: The value of the constant
    """

    value: int


@dataclass(frozen=True, slots=True)
class Var(Val):
    """Class representing a variable/identifier name within the tac

    Attributes:
        value: The string value (name) of the identifier
    """

    value: str
