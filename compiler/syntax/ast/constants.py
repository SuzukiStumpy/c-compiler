"""Represent various constant types within the AST"""

import struct
from abc import ABC
from dataclasses import dataclass


class Const(ABC):
    """Abstract base class for all constant AST nodes."""


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
