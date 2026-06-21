"""C type nodes for the AST.

Defines the ``Type`` abstract base class and its concrete variants. Each
distinct C type supported by the compiler is represented as a subclass of
``Type``. New types (pointers, arrays, structs, etc.) are added here as
language support is extended.

Also defines ``Linkage``, the linkage classification for declared C identifiers.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from compiler.syntax.ast.expressions import Exp


class Type(ABC):
    """Abstract base for all C type nodes.

    Each distinct C type supported by the compiler is represented as a
    concrete subclass. Consumers should use ``isinstance`` checks or
    structural pattern matching (``match``/``case``) rather than inspecting
    string representations.
    """


@dataclass(frozen=True, slots=True)
class IntType(Type):
    """Represents the C ``int`` type.

    Used wherever ``int`` appears as a return type or parameter type in the
    source. The node carries no fields - its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class LongType(Type):
    """Represents the C ``long`` type.

    Used wherever ``long`` appears as a return type or parameter type in the
    source.  The node carries no fields - its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class UIntType(Type):
    """Represents the C ``unsigned int`` type.

    Used wherever ``unsigned int`` appears as a return type or parameter type in the
    source. The node carries no fields - its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class ULongType(Type):
    """Represents the C ``unsigned long`` type.

    Used wherever ``unsigned long`` appears as a return type or parameter type in the
    source.  The node carries no fields - its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class DoubleType(Type):
    """Represents the C ``double`` type.

    Used wherever ``double`` appears as a return type or parameter type in the
    source.  The node carries no fields - its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class FloatType(Type):
    """Represents the C ``float`` type (32-bit IEEE 754 single precision).

    Used wherever ``float`` appears as a return type, parameter type, or variable type
    in the source.  The node carries no fields - its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class LongDoubleType(Type):
    """Represents the C ``long double`` type (80-bit x87 extended precision on x86-64).

    On x86-64 Linux the type occupies 16 bytes (10 bytes of mantissa/exponent plus
    6 bytes of alignment padding).  Values live in x87 FPU registers (ST0-ST7) and
    are passed/returned via the x87 stack (MEMORY class in the SysV AMD64 ABI).
    The node carries no fields - its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class BoolType(Type):
    """Represents the C ``_Bool`` type (C99 § 6.2.5 p2).

    A 1-byte unsigned integer whose only valid values are 0 and 1.  Any
    nonzero value assigned or converted to ``_Bool`` is normalised to 1.
    The node carries no fields - its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class VoidType(Type):
    """Represents the C ``void`` type.

    Used wherever ``void`` appears as a return type or parameter type in the
    source.  The node carries no fields - its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class Pointer(Type):
    """Represents a pointer within the C code

    Attributes:
        referenced: The type of the object the pointer references
    """

    referenced: Type


@dataclass(frozen=True, slots=True)
class Char(Type):
    """Represents a character within the C code

    Used whenever ``char`` appears as a return type or parameter type in the
    source.  The node carries no fields - its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class UChar(Type):
    """Represents an unsigned character within the C code

    Used whenever ``unsigned char`` appears as a return type or parameter type in the
    source.  The node carries no fields - its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class SChar(Type):
    """Represents a signed character within the C code

    Used whenever ``signed char`` appears as a return type or parameter type in the
    source.  The node carries no fields - its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class ShortType(Type):
    """Represents the C ``short`` type (16-bit signed integer).

    Used wherever ``short``, ``short int``, ``signed short``, or ``signed short int``
    appears as a return type, parameter type, or variable type. The node carries
    no fields - its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class UShortType(Type):
    """Represents the C ``unsigned short`` type (16-bit unsigned integer).

    Used wherever ``unsigned short`` or ``unsigned short int`` appears as a
    return type, parameter type, or variable type. The node carries no fields -
    its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class LongLongType(Type):
    """Represents the C ``long long`` type (64-bit signed integer).

    On x86-64 Linux this is identical in size and representation to ``long``
    (both 8 bytes), but ``long long`` and ``long`` are distinct types in C.
    The node carries no fields - its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class ULongLongType(Type):
    """Represents the C ``unsigned long long`` type (64-bit unsigned integer).

    On x86-64 Linux this is identical in size and representation to ``unsigned long``
    (both 8 bytes), but the two types are distinct in C.
    The node carries no fields - its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class Int128Type(Type):
    """Represents the GCC ``__int128`` extension type (128-bit signed integer).

    This is a compiler extension, not part of standard C11.  On x86-64 (SysV ABI)
    the type is 16 bytes wide with 16-byte alignment.  It is passed and returned
    in two consecutive integer registers (lo word first).  The node carries no
    fields - its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class UInt128Type(Type):
    """Represents the GCC ``unsigned __int128`` extension type (128-bit unsigned integer).

    See ``Int128Type`` for ABI notes.  The node carries no fields - its identity
    is its type alone.
    """


@dataclass(frozen=True, slots=True)
class FunType(Type):
    """Represents the type signature of a function definition.

    Attributes:
        params: An ordered list of types of each parameter to the function.
        ret: The return type of the function.
        variadic: ``True`` when the function accepts a variable number of
            arguments after the declared parameters (C ``...`` ellipsis).
    """

    params: tuple[Type, ...]
    ret: Type
    variadic: bool = False


@dataclass(frozen=True, slots=True)
class Array(Type):
    """Represents an array within the C code.

    Attributes:
        element: The type of element within the array.
        size: The number of elements as a concrete ``int`` (after semantic analysis),
            an unevaluated constant ``Exp`` (as produced by the parser), or ``None``
            for an incomplete array (flexible array member or ``T[]`` before size inference).
    """

    element: Type
    size: int | Exp | None


@dataclass(frozen=True, slots=True)
class Structure(Type):
    """Represents a ``struct`` type referred to by its tag name.

    This node appears wherever a ``struct Tag`` type is used as a variable
    type, parameter type, or return type.  The actual member layout is
    carried by the corresponding ``StructDeclaration`` node looked up via
    the tag.

    Attributes:
        tag: The struct tag name as it appears after the ``struct`` keyword
            in the source.
    """

    tag: str


@dataclass(frozen=True, slots=True)
class Union(Type):
    """Represents a ``union`` type referred to by its tag name.

    This node appears wherever a ``union Tag`` type is used as a variable
    type, parameter type, or return type.  The actual member layout is
    carried by the corresponding ``UnionDeclaration`` node looked up via
    the tag.

    Attributes:
        tag: The union tag name as it appears after the ``union`` keyword
            in the source.
    """

    tag: str


@dataclass(frozen=True, slots=True)
class EnumType(Type):
    """Represents a ``enum`` type referred to by its optional tag name.

    This node appears wherever an ``enum Tag`` type is used as a variable
    type, parameter type, or return type.  The underlying integer type is
    always ``int`` (C11 §6.7.2.2p4); the tag allows the semantic analyser
    to look up the enumeration's member names and values.

    Attributes:
        tag: The enum tag name as it appears after the ``enum`` keyword in
            the source, or ``None`` for an anonymous enum.
    """

    tag: str | None


@dataclass(frozen=True, slots=True)
class QualifiedType(Type):
    """A C type wrapped with one or more type qualifiers (C11 §6.7.3).

    Qualifiers modify the outermost type without changing its underlying
    identity.  The ``base`` field holds the unqualified type.  Each flag
    records whether the corresponding qualifier was present.

    Attributes:
        base: The underlying unqualified C type.
        const: ``True`` when the ``const`` qualifier is present.
        volatile: ``True`` when the ``volatile`` qualifier is present.
        restrict: ``True`` when the ``restrict`` qualifier is present.
    """

    base: Type
    const: bool = False
    volatile: bool = False
    restrict: bool = False


def unqualify(t: Type) -> Type:
    """Strip the outermost ``QualifiedType`` wrapper, if present.

    Args:
        t: A C type, potentially qualified.

    Returns:
        ``t.base`` when ``t`` is a ``QualifiedType``; ``t`` unchanged otherwise.
    """
    return t.base if isinstance(t, QualifiedType) else t


def is_const(t: Type) -> bool:
    """Return ``True`` when ``t`` carries the ``const`` qualifier at its outermost level.

    Args:
        t: A C type to inspect.

    Returns:
        ``True`` if and only if ``t`` is a ``QualifiedType`` with ``const=True``.
    """
    return isinstance(t, QualifiedType) and t.const


def is_volatile(t: Type) -> bool:
    """Return ``True`` when ``t`` carries the ``volatile`` qualifier at its outermost level.

    Args:
        t: A C type to inspect.

    Returns:
        ``True`` if and only if ``t`` is a ``QualifiedType`` with ``volatile=True``.
    """
    return isinstance(t, QualifiedType) and t.volatile


def is_restrict(t: Type) -> bool:
    """Return ``True`` when ``t`` carries the ``restrict`` qualifier at its outermost level.

    Args:
        t: A C type to inspect.

    Returns:
        ``True`` if and only if ``t`` is a ``QualifiedType`` with ``restrict=True``.
    """
    return isinstance(t, QualifiedType) and t.restrict


class Linkage(Enum):
    """Linkage classification for a declared C identifier (C11 sec. 6.2.2).

    ``NONE`` applies to identifiers with no linkage - local (block-scope)
    variables.  ``INTERNAL`` applies to identifiers with internal linkage -
    file-scope or static-local variables qualified with ``static``.
    ``EXTERNAL`` applies to identifiers with external linkage - functions and
    file-scope variables not qualified with ``static``.
    """

    NONE = auto()
    INTERNAL = auto()
    EXTERNAL = auto()
