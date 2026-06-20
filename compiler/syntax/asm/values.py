import struct
from abc import ABC
from dataclasses import dataclass


class AssemblyType(ABC):
    """Abstract base for assembly type descriptors.

    Concrete subclasses identify the operand size and register family for an
    instruction or data-segment element.  Consumers should use ``isinstance``
    checks or structural pattern matching to dispatch on the concrete variant.
    """


@dataclass(frozen=True, slots=True)
class Byte(AssemblyType):
    """8-bit integer assembly type (C ``char`` / ``signed char`` / ``unsigned char``)."""


@dataclass(frozen=True, slots=True)
class Word(AssemblyType):
    """16-bit integer assembly type (C ``short`` / ``unsigned short``)."""


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
class Float(AssemblyType):
    """32-bit IEEE 754 floating-point assembly type (C ``float``)."""


@dataclass(frozen=True, slots=True)
class LongDouble(AssemblyType):
    """80-bit x87 extended-precision assembly type (C ``long double``).

    Occupies 16 bytes on the stack (10 bytes of value plus 6 bytes of padding)
    and is accessed via x87 FPU instructions (``fldt``/``fstpt``).
    """


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
    """


@dataclass(frozen=True, slots=True)
class CharInit(StaticInit):
    """Initialiser for a static variable of C ``char``, ``signed char``, or ``unsigned char`` type.

    Emitted as ``.byte v`` - a single byte in the data segment.

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
class FloatInit(StaticInit):
    """Initialiser for a static variable or constant of C ``float`` type.

    The value is emitted as a raw 32-bit little-endian word (``.long <bits>``) in
    ``.rodata`` so that the bit pattern is preserved exactly (including ``-0.0f``).

    Attributes:
        value: The initial float value, as a plain Python float.
    """

    value: float

    def as_bits(self) -> int:
        """Return the IEEE 754 single-precision bit pattern as an unsigned 32-bit integer.

        Returns:
            An unsigned integer whose binary representation is the float's IEEE 754 encoding,
            suitable for emission as ``.long <bits>``.
        """
        (bits,) = struct.unpack("I", struct.pack("f", self.value))
        return bits


@dataclass(frozen=True, slots=True)
class LongDoubleInit(StaticInit):
    """Initialiser for a static variable of C ``long double`` type (80-bit x87 extended precision).

    On x86-64 Linux ``long double`` occupies 16 bytes: the first 10 bytes hold the 80-bit
    value (64-bit mantissa + 15-bit exponent + sign bit) and the remaining 6 bytes are
    zero-padded.  The value is stored as a Python float (double precision) and converted
    to the 80-bit format at emit time using ``struct.pack("e...")``.

    When emitted, non-zero values produce two ``.quad`` directives (mantissa and
    exponent/sign word) plus a ``.zero 6`` padding directive in ``.data``.  Zero
    values go to ``.bss``.

    Attributes:
        value: The initial long double value, stored as a Python float.
    """

    value: float

    def as_bits(self) -> tuple[int, int]:
        """Return the 80-bit x87 representation as ``(mantissa_bits, exp_sign_bits)``.

        The 80-bit extended precision format uses an explicit integer bit (unlike
        IEEE 754 double/float).  The method converts the Python float to this format
        by unpacking from the 80-bit struct format (if available) or approximating
        from double precision.

        Returns:
            A pair ``(mantissa, exp_sign)`` where ``mantissa`` is the 64-bit
            significand (integer bit explicit) and ``exp_sign`` is the combined
            15-bit biased exponent and sign bit as a 16-bit unsigned integer
            (packed in the low 16 bits of the second element).
        """
        import math

        v = self.value
        if v == 0.0:
            return (0, 0)
        sign = 1 if math.copysign(1.0, v) < 0 else 0
        v_abs = abs(v)
        exp_unbiased = math.frexp(v_abs)[1] - 1
        exp_biased = exp_unbiased + 16383
        mantissa_f = math.ldexp(v_abs, 63 - exp_unbiased)
        mantissa = int(mantissa_f) & 0xFFFF_FFFF_FFFF_FFFF
        exp_sign = (sign << 15) | (exp_biased & 0x7FFF)
        return (mantissa, exp_sign)


@dataclass(frozen=True, slots=True)
class ShortInit(StaticInit):
    """Initialiser for a static variable of C ``short`` / ``signed short`` type.

    Emitted as ``.short v`` - a 2-byte value in the data segment.

    Attributes:
        value: The initial signed 16-bit value.
    """

    value: int


@dataclass(frozen=True, slots=True)
class UShortInit(StaticInit):
    """Initialiser for a static variable of C ``unsigned short`` type.

    Emitted as ``.short v`` - a 2-byte value in the data segment.

    Attributes:
        value: The initial unsigned 16-bit value.
    """

    value: int


@dataclass(frozen=True, slots=True)
class Int128Init(StaticInit):
    """Initialiser for a static variable of GCC ``__int128`` type.

    Emitted as two consecutive ``.quad`` directives: low 64 bits then high 64 bits,
    matching the little-endian memory layout required by the SysV AMD64 ABI.

    Attributes:
        value: The initial signed 128-bit value stored as an arbitrary-precision Python int.
    """

    value: int


@dataclass(frozen=True, slots=True)
class UInt128Init(StaticInit):
    """Initialiser for a static variable of GCC ``unsigned __int128`` type.

    Emitted as two consecutive ``.quad`` directives: low 64 bits then high 64 bits.

    Attributes:
        value: The initial unsigned 128-bit value stored as an arbitrary-precision Python int.
    """

    value: int


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
        """Return zero - every ``ZeroInit`` initialiser is zero-valued."""
        return 0


@dataclass(frozen=True, slots=True)
class StringInit(StaticInit):
    """String content initialiser for a character array or string-literal static object.

    Attributes:
        string: The raw encoded byte content of the string (NUL terminator not included).
        null_terminated: When ``True``, ``element_size`` NUL bytes are appended during
            emission; when ``False``, the bytes are written as-is.
        element_size: Bytes per element (1 for ``char``, 2 for ``char16_t``, 4 for
            ``wchar_t`` / ``char32_t``).  Controls the width of the NUL terminator.
    """

    string: bytes
    null_terminated: bool
    element_size: int = 1


@dataclass(frozen=True, slots=True)
class PointerInit(StaticInit):
    """Pointer initialiser set to the address of a named static object.

    Used when a static pointer variable is initialised with the address of another
    static object (e.g. a string-literal constant or a file-scope variable).

    Attributes:
        name: The resolved (unique) name of the static object whose address is used.
    """

    name: str
