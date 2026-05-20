"""Represent various constant types within the AST"""

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


@dataclass(frozen=True, slots=True)
class ConstDouble(Const):
    """Represents an double constant within the C code.

    Attributes:
        val: The double value the constant represents.  Important: float in Python _is_
            an IEEE 764 64-bit number (equivalent of a C double).  Storing directly in
            this format will remove the issues of rounding errors when we generate the
            Assembly code later
    """

    val: float
