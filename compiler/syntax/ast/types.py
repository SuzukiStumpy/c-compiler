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
    source. The node carries no fields — its identity is its type alone.
    """


@dataclass(frozen=True, slots=True)
class LongType(Type):
    """Represents the C ``long`` type.

    Used wherever ``long`` appears as a return type or parameter type in the
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


class Linkage(Enum):
    """Linkage classification for a declared C identifier (C11 §6.2.2).

    ``NONE`` applies to identifiers with no linkage — local (block-scope)
    variables.  ``INTERNAL`` applies to identifiers with internal linkage —
    file-scope or static-local variables qualified with ``static``.
    ``EXTERNAL`` applies to identifiers with external linkage — functions and
    file-scope variables not qualified with ``static``.
    """

    NONE = auto()
    INTERNAL = auto()
    EXTERNAL = auto()
