"""Symbol table types shared between the semantic analysis and TAC generation stages.

The symbol table maps every name that has been resolved by the semantic analyser
to a ``SymbolTableEntry`` describing how that name should be treated by downstream
passes.

Grammar::

    symbol_table_entry = SymbolTableEntry(type, attrs)
    attrs              = StaticAttr(initial_value, bool globl)
                       | FunAttr(bool defined, bool globl)
                       | LocalAttr
    initial_value      = Initial(int | float)
                       | ULongInit(int)
                       | PointerInit(str)
                       | Tentative
                       | NoInitializer

``StaticAttr`` is attached to every variable with static storage duration:
file-scope variables (external or internal linkage) and block-scope ``static``
or ``extern`` variables.

``FunAttr`` is attached to every function declaration or definition.

``Initial(i)``      — the variable was given a constant initialiser ``i``.
``Tentative``       — the variable is a tentative definition (no initialiser);
                      it will be zero-initialised at link time.
``NoInitializer``   — the name appears only in a non-defining declaration (e.g.
                      ``extern int x;`` with no preceding definition); it will
                      not generate any data-section output.
"""

from abc import ABC
from dataclasses import dataclass

from compiler.syntax.ast.types import Type


class InitialValue(ABC):
    """Abstract base for the possible initial-value states of a static variable."""


@dataclass(frozen=True, slots=True)
class Initial(InitialValue):
    """The variable was defined with a constant initialiser.

    Attributes:
        value: The numeric value supplied as the initialiser.  ``int`` for all
            integer types; ``float`` for ``double``.
    """

    value: int | float


@dataclass(frozen=True, slots=True)
class ULongInit(InitialValue):
    """Initialiser for a static pointer variable set to the null pointer (64-bit zero).

    Attributes:
        value: Always zero; carried for structural uniformity with other numeric inits.
    """

    value: int = 0


@dataclass(frozen=True, slots=True)
class PointerInit(InitialValue):
    """Initialiser for a static pointer variable set to the address of a named static object.

    Attributes:
        name: The resolved (unique) name of the static variable whose address is used as
            the initialiser.  For file-scope variables this is the original source name;
            for block-scope ``static`` variables it is the unique internal identifier
            assigned by the semantic analyser.
    """

    name: str


@dataclass(frozen=True, slots=True)
class Tentative(InitialValue):
    """The variable is a tentative definition; zero-initialised if no definition is seen."""


@dataclass(frozen=True, slots=True)
class NoInitializer(InitialValue):
    """The name appears only in a non-defining declaration; generates no data-section output."""


class SymbolAttrs(ABC):
    """Abstract base for symbol-table attribute variants."""


@dataclass(frozen=True, slots=True)
class StaticAttr(SymbolAttrs):
    """Attributes for a variable with static storage duration.

    Attributes:
        init:  The initial-value classification for the variable.
        globl: ``True`` when the symbol has external linkage (should appear in
               the ``.globl`` directive); ``False`` for internal linkage.
    """

    init: InitialValue
    globl: bool


@dataclass(frozen=True, slots=True)
class LocalAttr(SymbolAttrs):
    """Attributes for a local (automatic-storage-duration) variable or compiler temporary.

    Carries no fields — its presence in the symbol table records the variable's
    type (via ``SymbolTableEntry.t``) so that downstream passes can determine
    register width without inspecting the original declaration.
    """


@dataclass(frozen=True, slots=True)
class FunAttr(SymbolAttrs):
    """Attributes for a function declaration or definition.

    Attributes:
        defined: ``True`` once a function body has been registered for this name.
        globl:   ``True`` when the function has external linkage.
    """

    defined: bool
    globl: bool


@dataclass(frozen=True, slots=True)
class SymbolTableEntry:
    """A single entry in the program-wide symbol table.

    Attributes:
        t: The C type of the symbol, carried through from the AST.
        attrs: The attribute variant that describes how this symbol should be
               treated by downstream compilation passes.
    """

    t: Type
    attrs: SymbolAttrs


SymbolTable = dict[str, SymbolTableEntry]
"""Mapping from resolved identifier name to its symbol-table entry."""
