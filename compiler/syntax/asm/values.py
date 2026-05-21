import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto


class AssemblyType(Enum):
    """Enumerated type identifying the type of assembly instruction and operand size to emit."""

    Longword = auto()
    Quadword = auto()
    Double = auto()


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
    """Initialiser for a static variable or constant of C ``double`` type.

    When used with ``StaticConstant``, the value is emitted as a raw 64-bit
    little-endian word (``.quad``) in ``.rodata`` so that the bit pattern is
    preserved exactly.  When used with ``StaticVariable``, zero goes to ``.bss``
    and non-zero values go to ``.data``.

    Attributes:
        value: The initial double value, as a plain Python float.
    """

    value: float

    def as_bits(self) -> int:
        """Return the IEEE 754 bit pattern of ``value`` as an unsigned 64-bit integer.

        Returns:
            An unsigned integer whose binary representation is identical to the
            double's IEEE 754 encoding, suitable for emission as ``.quad <bits>``.
        """
        (bits,) = struct.unpack("Q", struct.pack("d", self.value))
        return bits
