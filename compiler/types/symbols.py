"""Symbol table types shared between the semantic analysis and TAC generation stages.

The symbol table maps every name that has been resolved by the semantic analyser
to a ``SymbolTableEntry`` describing how that name should be treated by downstream
passes.

Grammar::

    symbol_table_entry = SymbolTableEntry(type, attrs)
    attrs              = StaticAttr(initial_value, bool globl)
                       | FunAttr(bool defined, bool globl)
                       | LocalAttr
                       | ConstantAttr(static_init init)
    initial_value      = Initial(tuple[StaticInit, ...])
                       | ULongInit(int)
                       | PointerInit(str)
                       | Tentative
                       | NoInitializer
    static_init        = IntInit(int)
                       | LongInit(int)
                       | DoubleInit(float)
                       | CharInit(int)
                       | UCharInit(int)
                       | ZeroInit(int bytes)
                       | StringInit(bytes string, bool null_terminated)

``StaticAttr`` is attached to every variable with static storage duration:
file-scope variables (external or internal linkage) and block-scope ``static``
or ``extern`` variables.

``FunAttr`` is attached to every function declaration or definition.

``Initial(inits)``  - the variable was given a constant initialiser; ``inits`` is
                      a flat ordered tuple of ``StaticInit`` elements.  For scalar
                      objects this is a single-element tuple; for arrays it is a
                      flat sequence with consecutive zero regions coalesced into
                      ``ZeroInit`` nodes.
``Tentative``       - the variable is a tentative definition (no initialiser);
                      it will be zero-initialised at link time.
``NoInitializer``   - the name appears only in a non-defining declaration (e.g.
                      ``extern int x;`` with no preceding definition); it will
                      not generate any data-section output.
"""

from abc import ABC
from dataclasses import dataclass

from compiler.syntax.ast.types import Type


class StaticInit(ABC):
    """Abstract base for scalar initial-value elements within an ``Initial`` tuple.

    Concrete variants carry a single scalar value or a zero-fill region.  They
    appear as elements in ``Initial.initializers`` and are distinct from
    ``tac.StaticInit``, which lives at the TAC level.
    """


@dataclass(frozen=True, slots=True)
class IntInit(StaticInit):
    """4-byte integer initialiser element (C ``int`` / ``unsigned int``).

    Attributes:
        value: The integer value to store.  Never zero - use ``ZeroInit(4)``
            for a zero-valued 4-byte element.
    """

    value: int


@dataclass(frozen=True, slots=True)
class LongInit(StaticInit):
    """8-byte integer initialiser element (C ``long`` / ``unsigned long``).

    Attributes:
        value: The integer value to store.  Never zero - use ``ZeroInit(8)``
            for a zero-valued 8-byte element.
    """

    value: int


@dataclass(frozen=True, slots=True)
class DoubleInit(StaticInit):
    """8-byte floating-point initialiser element (C ``double``).

    Attributes:
        value: The double value to store.  Never zero - use ``ZeroInit(8)``
            for a zero-valued 8-byte element.
    """

    value: float


@dataclass(frozen=True, slots=True)
class CharInit(StaticInit):
    """1-byte signed character initialiser element (C ``char`` / ``signed char``).

    Attributes:
        value: The integer value to store.  Never zero - use ``ZeroInit(1)``
            for a zero-valued 1-byte signed character element.
    """

    value: int


@dataclass(frozen=True, slots=True)
class UCharInit(StaticInit):
    """1-byte unsigned character initialiser element (C ``unsigned char``).

    Attributes:
        value: The integer value to store.  Never zero - use ``ZeroInit(1)``
            for a zero-valued 1-byte unsigned character element.
    """

    value: int


@dataclass(frozen=True, slots=True)
class ShortInit(StaticInit):
    """2-byte signed short integer initialiser element (C ``short`` / ``signed short``).

    Attributes:
        value: The signed integer value to store.  Never zero - use ``ZeroInit(2)``
            for a zero-valued 2-byte element.
    """

    value: int


@dataclass(frozen=True, slots=True)
class UShortInit(StaticInit):
    """2-byte unsigned short integer initialiser element (C ``unsigned short``).

    Attributes:
        value: The unsigned integer value to store.  Never zero - use ``ZeroInit(2)``
            for a zero-valued 2-byte element.
    """

    value: int


@dataclass(frozen=True, slots=True)
class Int128Init(StaticInit):
    """16-byte signed 128-bit integer initialiser element (GCC ``__int128``).

    Python ``int`` is arbitrary-precision so the full 128-bit signed range is
    stored without loss.  Never zero - use ``ZeroInit(16)`` for a zero-valued element.

    Attributes:
        value: The signed integer value.
    """

    value: int


@dataclass(frozen=True, slots=True)
class UInt128Init(StaticInit):
    """16-byte unsigned 128-bit integer initialiser element (GCC ``unsigned __int128``).

    Python ``int`` is arbitrary-precision so the full 128-bit unsigned range is
    stored without loss.  Never zero - use ``ZeroInit(16)`` for a zero-valued element.

    Attributes:
        value: The unsigned integer value.
    """

    value: int


@dataclass(frozen=True, slots=True)
class ZeroInit(StaticInit):
    """Zero-fill initialiser element spanning ``bytes`` bytes.

    Used for explicitly zero-initialised scalars and for zero-padding regions
    within aggregate initialisers.  Adjacent ``ZeroInit`` nodes are coalesced
    by the semantic analyser so a single ``ZeroInit`` spans the whole region.

    Attributes:
        bytes: The number of consecutive zero bytes to emit.
    """

    bytes: int


@dataclass(frozen=True, slots=True)
class StringInit(StaticInit):
    """String content initialiser for a character array or string-literal static object.

    Used both when a string literal initialises a ``char``/``signed char``/``unsigned char``
    array and when the compiler emits a hidden static object for a string-literal constant.

    Attributes:
        string: The raw byte content of the string (NUL terminator not included).
        null_terminated: When ``True``, a NUL byte is appended after ``string`` during
            emission; when ``False``, the bytes are written as-is (used when the array
            is smaller than the full string or the terminator is handled separately).
    """

    string: bytes
    null_terminated: bool


class InitialValue(ABC):
    """Abstract base for the possible initial-value states of a static variable."""


@dataclass(frozen=True, slots=True)
class Initial(InitialValue):
    """The variable was defined with a constant initialiser.

    Holds an ordered flat tuple of ``StaticInit`` elements.  For scalar objects
    this is a single-element tuple; for arrays it is a flat sequence of element
    values with consecutive zero regions coalesced into ``ZeroInit`` nodes.

    Attributes:
        initializers: Ordered tuple of scalar initialiser elements.
    """

    initializers: tuple[StaticInit, ...]


@dataclass(frozen=True, slots=True)
class ULongInit(InitialValue):
    """Initialiser for a static pointer variable set to the null pointer (64-bit zero).

    Attributes:
        value: Always zero; carried for structural uniformity with other numeric inits.
    """

    value: int = 0


@dataclass(frozen=True, slots=True)
class PointerInit(InitialValue, StaticInit):
    """Initialiser for a static pointer variable set to the address of a named static object.

    Serves a dual role: as a top-level ``InitialValue`` for pointer-typed static variables
    (e.g. ``int *p = &x;``) and as a scalar element within an aggregate ``StaticInit``
    sequence (e.g. a struct member pointing to a string literal).

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
class ConstantAttr(SymbolAttrs):
    """Attributes for a compiler-generated read-only string-literal constant.

    Attached to the hidden static objects the compiler creates for each distinct
    string literal.  The object lives in ``.rodata`` and is never exported.

    Attributes:
        init: The single ``StaticInit`` element that describes the content of
            the constant (typically a ``StringInit``).
    """

    init: StaticInit


@dataclass(frozen=True, slots=True)
class LocalAttr(SymbolAttrs):
    """Attributes for a local (automatic-storage-duration) variable or compiler temporary.

    Carries no fields - its presence in the symbol table records the variable's
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


@dataclass(frozen=True, slots=True)
class MemberEntry:
    """Layout information for a single member within a struct type.

    Attributes:
        name: The member name as it appears in the source.
        member_type: The resolved C type of the member.
        offset: Byte offset of the member from the start of the enclosing struct.
    """

    name: str
    member_type: Type
    offset: int


@dataclass(frozen=True, slots=True)
class StructEntry:
    """Layout information for a complete, defined struct type.

    Populated by the semantic analyser when it encounters a struct definition
    (a ``StructDeclaration`` with a non-empty member list).  Consumed by
    downstream passes (TAC generation, code generation) to compute member
    offsets and struct sizes without re-walking the AST.

    Attributes:
        alignment: Required alignment of the struct in bytes.  Equal to the
            maximum alignment of any member.
        size: Total padded size of the struct in bytes (rounded up to a multiple
            of ``alignment`` to satisfy C11 sec. 6.7.2.1p15).
        members: Ordered tuple of ``MemberEntry`` values, one per declared field.
    """

    alignment: int
    size: int
    members: tuple[MemberEntry, ...]


StructTypeTable = dict[str, StructEntry]
"""Mapping from resolved struct tag to its layout entry."""
