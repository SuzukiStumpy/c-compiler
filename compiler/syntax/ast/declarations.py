"""Program structure nodes for the AST.

Defines the top-level constructs of a C translation unit: the ``VariableDeclaration``
node, the ``Declaration`` sum type (``FunDecl``, ``VarDecl``, and ``StructDecl``),
the ``BlockItem`` sum type (``S`` and ``D``) that composes statements and declarations
into a block, the ``Block`` type alias, the ``Compound`` statement node, the
``ForInit`` sum type (``InitDecl`` and ``InitExp``) for ``for``-loop initialisers,
the ``For`` statement node, the ``MemberDeclaration`` and ``StructDeclaration`` nodes
for struct type definitions, and the ``Parameter``, ``Function``, and ``Program`` nodes.
"""

from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto

from compiler.syntax.ast.expressions import Exp, Initializer, SingleInit
from compiler.syntax.ast.statements import Statement
from compiler.syntax.ast.types import Type


class StorageClass(Enum):
    """Defines the storage class on function and variable declarations"""

    Static = auto()
    Extern = auto()


@dataclass(frozen=True, slots=True)
class VariableDeclaration:
    """A variable declaration, optionally with an initialiser and storage class.

    Attributes:
        name: The name of the variable being declared.
        init: The initialiser - a ``SingleInit`` wrapping a scalar expression,
            a ``CompoundInit`` for brace-enclosed aggregate initialisers, or
            ``None`` if the variable is declared without an initial value.
        var_type: The type of this variable.
        storage_class: The storage class specifier, or ``None`` if omitted.
    """

    name: str
    init: Initializer | None
    var_type: Type
    storage_class: StorageClass | None = None

    def __post_init__(self) -> None:
        """Wrap a bare ``Exp`` passed as ``init`` in a ``SingleInit`` node.

        Allows callers that pre-date the ``Initializer`` wrapper (tests, semantic
        analyser internals) to continue passing an ``Exp`` directly without
        breaking the type contract.
        """
        if isinstance(self.init, Exp):
            object.__setattr__(self, "init", SingleInit(exp=self.init))


class Declaration(ABC):
    """Abstract base for declaration nodes.

    A declaration is either a variable declaration (``VarDecl``) or a function
    declaration (``FunDecl``). Consumers should use structural pattern matching
    (``match``/``case``) to dispatch on the concrete variant.
    """


class BlockItem(ABC):
    """Abstract base for items that may appear inside a block.

    A block is a sequence of ``BlockItem`` nodes, each of which is either a
    statement (``S``) or a declaration (``D``). Consumers should use
    structural pattern matching (``match``/``case``) to dispatch on the
    concrete variant.
    """


@dataclass(frozen=True, slots=True)
class S(BlockItem):
    """A statement wrapped as a block item.

    Attributes:
        statement: The statement node.
    """

    statement: Statement


@dataclass(frozen=True, slots=True)
class D(BlockItem):
    """A declaration wrapped as a block item.

    Attributes:
        declaration: The declaration node - either a ``FunDecl`` or a ``VarDecl``.
    """

    declaration: Declaration


# A block is an ordered sequence of block items, corresponding to the grammar
# production: block = Block(block_item*).
Block = tuple[BlockItem, ...]


@dataclass(frozen=True, slots=True)
class Compound(Statement):
    """A compound statement: a ``{ block_item* }`` block used as a statement.

    Attributes:
        body: The ordered sequence of block items enclosed by the braces.
    """

    body: Block


@dataclass(frozen=True, slots=True)
class Parameter:
    """A single function parameter declaration.

    Attributes:
        param_type: The declared C type of this parameter.
        name: The parameter name as it appears in the source.
    """

    param_type: Type
    name: str


@dataclass(frozen=True, slots=True)
class Function:
    """A function declaration or definition.

    Represents a C function including its name, parameter list, and optional
    body. When ``body`` is ``None`` the node represents a forward declaration
    (prototype); when ``body`` is a ``Block`` the node is a full definition.

    Attributes:
        name: The function name as it appears in the source.
        params: The ordered sequence of parameters. Empty for functions declared
            with ``void`` or ``()`` in the parameter list.
        body: The block comprising the function body, or ``None`` for a forward
            declaration.
        fun_type: The type signature for this function (carries types for parameters
            along with the return type).
        globl: ``True`` when the function has external linkage and should be
            visible to the linker.  Set by the semantic analysis stage; defaults
            to ``False`` before analysis runs.
        storage_class: The optional storage-class specifier (``static``).
        source_file: Path of the source file in which the function appears, as
            recorded by the lexer's linemarker tracking.  Empty string when not
            available (e.g. in hand-constructed ASTs used by tests).
        source_line: 1-indexed line number of the first token of the function
            declaration in ``source_file``.  Zero when not available.
        is_inline: ``True`` when the ``inline`` function specifier appears on the
            declaration.  Stored for completeness; the compiler does not perform
            inlining and treats this flag as informational only.
    """

    name: str
    params: tuple[Parameter, ...]
    body: Block | None
    fun_type: Type
    globl: bool = False
    storage_class: StorageClass | None = None
    source_file: str = ""
    source_line: int = 0
    is_inline: bool = False


@dataclass(frozen=True, slots=True)
class FunDecl(Declaration):
    """A function declaration wrapped as a declaration node.

    Used when a function declaration appears as a block item (``D``).

    Attributes:
        function_declaration: The underlying ``Function`` node.  The storage
            class, if any, is carried on ``function_declaration.storage_class``.
    """

    function_declaration: Function


@dataclass(frozen=True, slots=True)
class VarDecl(Declaration):
    """A variable declaration wrapped as a declaration node.

    Used when a variable declaration appears as a block item (``D``).

    Attributes:
        variable_declaration: The underlying ``VariableDeclaration`` node.
    """

    variable_declaration: VariableDeclaration


@dataclass(frozen=True, slots=True)
class MemberDeclaration:
    """A single field declaration within a struct or union body.

    Attributes:
        member_name: The name of the struct member as it appears in the source,
            or ``None`` for an unnamed bit-field (padding) declaration.
        member_type: The declared C type of the member.
        bit_width: For bit-field declarations, the number of bits to occupy.
            ``None`` for regular (non-bit-field) members.  A width of zero is
            only valid when ``member_name`` is also ``None``; it forces the
            next bit-field to begin at the next storage-unit boundary.
    """

    member_name: str | None
    member_type: Type
    bit_width: int | None = None


@dataclass(frozen=True, slots=True)
class StructDeclaration:
    """A ``struct`` type definition, binding a tag to an ordered list of members.

    Attributes:
        tag: The struct tag name as it appears after the ``struct`` keyword.
        members: The ordered sequence of field declarations.  Empty only for an
            incomplete (forward) struct declaration.
    """

    tag: str
    members: tuple[MemberDeclaration, ...]


@dataclass(frozen=True, slots=True)
class StructDecl(Declaration):
    """A struct definition wrapped as a top-level declaration node.

    Attributes:
        struct_declaration: The underlying ``StructDeclaration`` node carrying
            the tag name and member list.
    """

    struct_declaration: StructDeclaration


@dataclass(frozen=True, slots=True)
class UnionDeclaration:
    """A ``union`` type definition, binding a tag to an ordered list of members.

    Attributes:
        tag: The union tag name as it appears after the ``union`` keyword.
        members: The ordered sequence of field declarations.  Empty only for an
            incomplete (forward) union declaration.
    """

    tag: str
    members: tuple[MemberDeclaration, ...]


@dataclass(frozen=True, slots=True)
class UnionDecl(Declaration):
    """A union definition wrapped as a top-level declaration node.

    Attributes:
        union_declaration: The underlying ``UnionDeclaration`` node carrying
            the tag name and member list.
    """

    union_declaration: UnionDeclaration


@dataclass(frozen=True, slots=True)
class EnumeratorDeclaration:
    """A single enumerator within an enum body, with its computed integer value.

    The parser computes enumerator values at parse time using a running counter
    that starts at 0 and increments by 1 for each enumerator without an explicit
    value.  An explicit ``= constant`` resets the running counter.

    Attributes:
        name: The enumerator identifier as it appears in the source.
        value: The integer value assigned to this enumerator, computed by the
            parser at parse time.
    """

    name: str
    value: int


@dataclass(frozen=True, slots=True)
class EnumDeclaration:
    """An ``enum`` type definition, binding an optional tag to an ordered enumerator list.

    Attributes:
        tag: The enum tag name as it appears after the ``enum`` keyword, or
            ``None`` for an anonymous enum.
        members: The ordered sequence of enumerator declarations.  Each member
            carries its computed integer value as resolved by the parser.
    """

    tag: str | None
    members: tuple[EnumeratorDeclaration, ...]


@dataclass(frozen=True, slots=True)
class EnumDecl(Declaration):
    """An enum definition wrapped as a top-level declaration node.

    Attributes:
        enum_declaration: The underlying ``EnumDeclaration`` node carrying the
            optional tag name and enumerator list.
    """

    enum_declaration: EnumDeclaration


@dataclass(frozen=True, slots=True)
class TypedefDecl(Declaration):
    """A typedef declaration, binding an alias name to an aliased type.

    Resolved entirely by the parser; downstream stages do not see typedef names.
    Semantic analysis and later stages should skip ``TypedefDecl`` nodes.

    Attributes:
        name: The alias identifier introduced by the typedef.
        aliased_type: The concrete ``Type`` that ``name`` maps to.
    """

    name: str
    aliased_type: Type


@dataclass(frozen=True, slots=True)
class Program:
    """The root AST node representing a complete translation unit.

    A translation unit consists of one or more top-level function declarations
    or definitions.

    Attributes:
        declarations: The ordered sequence of top-level function and variable declarations.
            At least one function is required by the grammar.
    """

    declarations: tuple[Declaration, ...]


class ForInit(ABC):
    """Abstract base for the initialiser clause of a ``for`` loop.

    Consumers should use structural pattern matching (``match``/``case``) to
    dispatch on the concrete variant: ``InitDecl`` for a declaration, or
    ``InitExp`` for an expression (which may be ``None`` when the clause is
    absent).
    """


@dataclass(frozen=True, slots=True)
class InitDecl(ForInit):
    """A declaration used as the initialiser clause of a ``for`` loop.

    Attributes:
        declaration: The variable declaration that initialises the loop.
            Always a ``VarDecl``; the parser rejects function declarations
            in this position.
    """

    declaration: Declaration


@dataclass(frozen=True, slots=True)
class InitExp(ForInit):
    """An expression used as the initialiser clause of a ``for`` loop.

    ``exp`` is ``None`` when the ``for`` loop has no initialiser (``for (;;)``).

    Attributes:
        exp: The initialiser expression, or ``None`` if absent.
    """

    exp: Exp | None


@dataclass(frozen=True, slots=True)
class For(Statement):
    """A ``for`` loop.

    The initialiser runs exactly once before the first iteration. The loop body
    executes while ``condition`` is truthy; after each iteration ``post`` is
    evaluated if present.

    Attributes:
        init: The initialiser clause - either a declaration (``InitDecl``) or
            an optional expression (``InitExp``).
        condition: The loop condition tested before each iteration, or ``None``
            for an unbounded loop.
        post: The expression evaluated after each iteration, or ``None`` if absent.
        body: The statement executed on each iteration.
        identifier: The unique identifier for this loop, assigned during the
            loop-labelling pass of semantic analysis.
    """

    init: ForInit
    condition: Exp | None
    post: Exp | None
    body: Statement
    identifier: str = ""
