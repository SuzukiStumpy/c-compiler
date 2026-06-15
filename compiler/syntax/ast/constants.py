"""Represent various constant types within the AST"""

import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Const(ABC):
    """Abstract base class for all constant AST nodes."""

    @property
    @abstractmethod
    def val(self) -> int | float:
        """The numeric value of this constant."""


@dataclass(frozen=True, slots=True)
class ConstInt(Const):
    """Represents an integer constant within the C code.

    Attributes:
        val: The integer value the constant represents.
    """

    val: int


@dataclass(frozen=True, slots=True)
class ConstLong(Const):
    """Represents a long integer constant within the C code.

    Attributes:
        val: The integer value the constant represents.
    """

    val: int


@dataclass(frozen=True, slots=True)
class ConstUInt(Const):
    """Represents an unsigned integer constant within the C code.

    Attributes:
        val: The integer value the constant represents.
    """

    val: int


@dataclass(frozen=True, slots=True)
class ConstULong(Const):
    """Represents an unsigned long integer constant within the C code.

    Attributes:
        val: The integer value the constant represents.
    """

    val: int


@dataclass(frozen=True, slots=True, eq=False)
class ConstDouble(Const):
    """Represents an double constant within the C code.

    Attributes:
        val: The double value the constant represents.  Important: float in Python _is_
            an IEEE 764 64-bit number (equivalent of a C double).  Storing directly in
            this format will remove the issues of rounding errors when we generate the
            Assembly code later
    """

    val: float

    def __eq__(self, other: object) -> bool:
        """Compare bit-exactly so that ``+0.0`` and ``-0.0`` are treated as distinct.

        Args:
            other: Object to compare against.

        Returns:
            ``True`` when *other* is a ``ConstDouble`` with the same IEEE 754 bit
            pattern as ``self``.
        """
        if not isinstance(other, ConstDouble):
            return NotImplemented
        return struct.pack("d", self.val) == struct.pack("d", other.val)

    def __hash__(self) -> int:
        """Hash by IEEE 754 bit pattern so that ``+0.0`` and ``-0.0`` hash differently.

        Returns:
            An integer hash consistent with the bit-exact ``__eq__``.
        """
        return hash(struct.pack("d", self.val))


@dataclass(frozen=True, slots=True)
class ConstChar(Const):
    """Represents a signed character constant within the C code

    Attributes:
        val: The integer value representing the character
    """

    val: int


@dataclass(frozen=True, slots=True)
class ConstUChar(Const):
    """Represents an unsigned character constant within the C code

    Attributes:
        val: The (unsigned) integer value representing the character
    """

    val: int


@dataclass(frozen=True, slots=True)
class ConstShort(Const):
    """Represents a signed 16-bit short integer constant within the C code.

    Appears in zero-initialisers and after truncation from wider types.

    Attributes:
        val: The signed integer value in the range [-32768, 32767].
    """

    val: int


@dataclass(frozen=True, slots=True)
class ConstUShort(Const):
    """Represents an unsigned 16-bit short integer constant within the C code.

    Appears in zero-initialisers and after truncation from wider types.

    Attributes:
        val: The unsigned integer value in the range [0, 65535].
    """

    val: int


@dataclass(frozen=True, slots=True)
class ConstInt128(Const):
    """Represents a signed 128-bit integer constant within the C code.

    Python ``int`` is arbitrary-precision so the full 128-bit range is stored
    without loss.

    Attributes:
        val: The signed integer value in the range [-2**127, 2**127 - 1].
    """

    val: int


@dataclass(frozen=True, slots=True)
class ConstUInt128(Const):
    """Represents an unsigned 128-bit integer constant within the C code.

    Python ``int`` is arbitrary-precision so the full 128-bit range is stored
    without loss.

    Attributes:
        val: The unsigned integer value in the range [0, 2**128 - 1].
    """

    val: int
