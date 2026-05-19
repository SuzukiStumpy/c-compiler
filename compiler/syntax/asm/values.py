from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto


class AssemblyType(Enum):
    """Enumerated type identifying the type of assembly instruction and operand size to emit."""

    Longword = auto()
    Quadword = auto()


class StaticInit(ABC):
    """Abstract base for static variable initialiser nodes.

    Concrete variants carry the initial value for a static variable as it
    appears in the data segment. They are distinct from ``Constant`` and
    ``Var``: those are instruction operands; these are data-segment
    initialisers only.

    All concrete subclasses must provide a ``value`` attribute of type ``int``.
    Dataclass fields named ``value`` satisfy this contract automatically.
    """

    @property
    @abstractmethod
    def value(self) -> int:
        """The integer value of the initialiser."""


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
