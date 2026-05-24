import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass


class AssemblyType(ABC):
    """Abstract base for assembly type descriptors.

    Concrete subclasses identify the operand size and register family for an
    instruction or data-segment element.  Consumers should use ``isinstance``
    checks or structural pattern matching to dispatch on the concrete variant.
    """


@dataclass(frozen=True, slots=True)
class Longword(AssemblyType):
    """32-bit integer assembly type (C ``int`` / ``unsigned int``)."""


@dataclass(frozen=True, slots=True)
class Quadword(AssemblyType):
    """64-bit integer assembly type (C ``long`` / ``unsigned long`` / any pointer)."""


@dataclass(frozen=True, slots=True)
class Double(AssemblyType):
    """64-bit IEEE 754 floating-point assembly type (C ``double``)."""


@dataclass(frozen=True, slots=True)
class ByteArray(AssemblyType):
    """Variable-length byte region in the data segment.

    Used for aggregate types such as arrays whose total size does not map to a
    single scalar register width.  The ``size`` bytes are emitted as a
    ``.zero`` or ``.byte`` sequence; ``alignment`` sets the ``.balign``
    directive.

    Attributes:
        size: Total size of the region in bytes.
        alignment: Required alignment of the region in bytes (must be a power of two).
    """

    size: int
    alignment: int


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


@dataclass(frozen=True, slots=True)
class ZeroInit(StaticInit):
    """Zero-fill initialiser spanning ``bytes`` consecutive bytes in the data segment.

    Emitted as ``.zero <bytes>`` regardless of whether the variable lives in
    ``.bss`` or ``.data``.  The assembler zeroes the region in both cases.

    Attributes:
        bytes: The number of consecutive zero bytes to emit.
    """

    bytes: int

    @property
    def value(self) -> int:
        """Return zero — every ``ZeroInit`` initialiser is zero-valued."""
        return 0
