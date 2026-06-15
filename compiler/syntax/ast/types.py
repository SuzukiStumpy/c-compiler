"""C type nodes for the AST.

Defines the ``Type`` abstract base class and its concrete variants. Each
distinct C type supported by the compiler is represented as a subclass of
``Type``. New types (pointers, arrays, structs, etc.) are added here as
language support is extended.

Also defines ``Linkage``, the linkage classification for declared C identifiers.
"""

from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto


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
class FunType(Type):
    """Represents the type signature of a function definition.

    Attributes:
        params: An ordered list of types of each parameter to the function.
        ret: The return type of the function.
    """

    params: tuple[Type, ...]
    ret: Type


@dataclass(frozen=True, slots=True)
class Array(Type):
    """Represents an array within the C code

    Attributes:
        element: The type of element within the array
        size: The size of each element within the array
    """

    element: Type
    size: int


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
