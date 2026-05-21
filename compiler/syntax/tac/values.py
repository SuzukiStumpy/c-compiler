"""Value nodes for the tac AST.

Defines the ``Val`` abstract base class and its concrete variants, plus the
``StaticInit`` abstract base class and its concrete variants used to represent
the initial value of a static variable in the data segment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from compiler.syntax.ast.constants import Const


class Val(ABC):
    """Abstract base for all tac Value nodes.

    Each instruction form is represented as a concrete subclass. Consumers
    should use structural pattern matching (``match``/``case``) to dispatch on
    the concrete variant rather than inspecting fields directly.
    """


@dataclass(frozen=True, slots=True)
class Constant(Val):
    """Class representing a typed constant within the tac.

    Attributes:
        value: The constant, carrying both the integer value and its C type
            (``ConstInt`` or ``ConstLong``).
    """

    value: Const


@dataclass(frozen=True, slots=True)
class Var(Val):
    """Class representing a variable/identifier name within the tac

    Attributes:
        value: The string value (name) of the identifier
    """

    value: str


class StaticInit(ABC):
    """Abstract base for static variable initialiser nodes.

    Concrete variants carry the initial value for a static variable as it
    appears in the data segment. They are distinct from ``Constant`` and
    ``Var``: those are instruction operands; these are data-segment
    initialisers only.

    All concrete subclasses must provide a ``value`` attribute of the
    appropriate numeric type.
    """

    @property
    @abstractmethod
    def value(self) -> int | float:
        """The numeric value of the initialiser."""


@dataclass(frozen=True, slots=True)
class IntInit(StaticInit):
    """Initialiser for a static variable of C ``int`` type.

    Attributes:
        value: The initial integer value, as a plain Python int.
    """

    value: int


@dataclass(frozen=True, slots=True)
class LongInit(StaticInit):
    """Initialiser for a static variable of C ``long`` type.

    Attributes:
        value: The initial integer value, as a plain Python int.
    """

    value: int


@dataclass(frozen=True, slots=True)
class DoubleInit(StaticInit):
    """Initialiser for a static variable of C ``double`` type.

    Attributes:
        value: The initial double value, as a plain Python float.
    """

    value: float
