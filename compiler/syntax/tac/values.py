"""Value nodes for the tac AST.

Defines the ``Val`` abstract base class and its concrete variants, plus the
``StaticInit`` abstract base class and its concrete variants used to represent
the initial value of a static variable in the data segment.
"""

from abc import ABC
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
    """


@dataclass(frozen=True, slots=True)
class CharInit(StaticInit):
    """Initialiser for a static variable of C ``char``, ``signed char``, or ``unsigned char`` type.

    Emits a single byte (``.byte v``) in the data segment.

    Attributes:
        value: The initial byte value, as a plain Python int.
    """

    value: int


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


@dataclass(frozen=True, slots=True)
class FloatInit(StaticInit):
    """Initialiser for a static variable of C ``float`` type.

    Attributes:
        value: The initial float value, as a plain Python float.
    """

    value: float


@dataclass(frozen=True, slots=True)
class LongDoubleInit(StaticInit):
    """Initialiser for a static variable of C ``long double`` type (80-bit x87 extended precision).

    On x86-64 Linux ``long double`` occupies 16 bytes (10 bytes of value plus 6 bytes
    padding).  The value is stored as a Python float (double precision).

    Attributes:
        value: The initial long double value, as a plain Python float.
    """

    value: float


@dataclass(frozen=True, slots=True)
class ShortInit(StaticInit):
    """Initialiser for a static variable of C ``short`` / ``signed short`` type.

    Emits a 2-byte (``.short v``) entry in the data segment.

    Attributes:
        value: The initial signed 16-bit value.
    """

    value: int


@dataclass(frozen=True, slots=True)
class UShortInit(StaticInit):
    """Initialiser for a static variable of C ``unsigned short`` type.

    Emits a 2-byte (``.short v``) entry in the data segment.

    Attributes:
        value: The initial unsigned 16-bit value.
    """

    value: int


@dataclass(frozen=True, slots=True)
class Int128Init(StaticInit):
    """Initialiser for a static variable of GCC ``__int128`` type.

    Emits two consecutive 8-byte (``.quad``) entries: low 64 bits then high 64 bits.

    Attributes:
        value: The initial signed 128-bit value stored as an arbitrary-precision Python int.
    """

    value: int


@dataclass(frozen=True, slots=True)
class UInt128Init(StaticInit):
    """Initialiser for a static variable of GCC ``unsigned __int128`` type.

    Emits two consecutive 8-byte (``.quad``) entries: low 64 bits then high 64 bits.

    Attributes:
        value: The initial unsigned 128-bit value stored as an arbitrary-precision Python int.
    """

    value: int


@dataclass(frozen=True, slots=True)
class ZeroInit(StaticInit):
    """Zero-fill initialiser spanning ``bytes`` consecutive bytes in the data segment.

    Used for tentative definitions and for explicitly zero-initialised scalars and arrays.
    Distinct from ``DoubleInit(0.0)`` which preserves the double type for positive zero;
    ``ZeroInit`` must not be used where negative zero (``-0.0``) is required.

    Attributes:
        bytes: The number of consecutive zero bytes to emit.
    """

    bytes: int

    @property
    def value(self) -> int:
        """Return zero - the numeric value of any ``ZeroInit`` initialiser."""
        return 0


@dataclass(frozen=True, slots=True)
class StringInit(StaticInit):
    """String content initialiser for a character array or string-literal static object.

    Attributes:
        string: The raw byte content of the string (NUL terminator not included).
        null_terminated: When ``True``, a NUL byte is appended after ``string`` during
            emission; when ``False``, the bytes are written as-is.
    """

    string: bytes
    null_terminated: bool


@dataclass(frozen=True, slots=True)
class PointerInit(StaticInit):
    """Pointer initialiser set to the address of a named static object.

    Used when a static pointer variable is initialised with the address of another
    static object (e.g. a string-literal constant or a file-scope variable).

    Attributes:
        name: The resolved (unique) name of the static object whose address is used.
    """

    name: str
