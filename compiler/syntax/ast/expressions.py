"""Expression nodes for the AST.

Defines the ``Exp`` abstract base class and its concrete variants. Each
expression form supported by the compiler is represented as a subclass of
``Exp``. New expression types (function calls, etc.) are added here as language
support is extended.
"""

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto

from compiler.syntax.ast.constants import Const
from compiler.syntax.ast.operators import BinaryOperator, UnaryOperator
from compiler.syntax.ast.types import Type


class StringEncoding(Enum):
    """Encoding of a string or character literal as written in C source.

    Determines the element type used for the resulting array and the number
    of bytes per character stored in :attr:`String.string`.

    Attributes:
        CHAR: Plain ``"..."`` or ``u8"..."`` — ``char`` elements, 1 byte each.
        WCHAR: ``L"..."`` — ``wchar_t`` elements; 4 bytes each (LE int32) on Linux x86-64.
        UTF16: ``u"..."`` — ``char16_t`` elements; 2 bytes each (LE uint16).
        UTF32: ``U"..."`` — ``char32_t`` elements; 4 bytes each (LE uint32).
    """

    CHAR = auto()
    WCHAR = auto()
    UTF16 = auto()
    UTF32 = auto()


@dataclass(frozen=True, slots=True)
class Exp(ABC):
    """Abstract base for all expression nodes.

    Each expression form supported by the compiler is represented as a
    concrete subclass. Consumers should use structural pattern matching
    (``match``/``case``) to dispatch on the concrete variant rather than
    inspecting fields directly.

    Attributes:
        exp_type: The resolved C type of this expression, populated by the
            semantic analysis stage. ``None`` before type annotation is complete.
    """

    exp_type: Type | None = field(default=None, kw_only=True, compare=False, hash=False, repr=False)


@dataclass(frozen=True, slots=True)
class Constant(Exp):
    """A constant expression.

    Attributes:
        value: The value of the constant as parsed from the source.
    """

    value: Const


@dataclass(frozen=True, slots=True)
class Var(Exp):
    """A named variable in the program.

    Attributes:
        name: The name of the variable.
    """

    name: str


@dataclass(frozen=True, slots=True)
class Cast(Exp):
    """Represents a type cast expression.

    Attributes:
        target_type: The type we want to cast the expression to.
        exp: The expression we are casting.
    """

    target_type: Type
    exp: Exp


@dataclass(frozen=True, slots=True)
class Unary(Exp):
    """A unary operator expression.

    Attributes:
        operator: The unary operator which will act upon the subexpression
        exp:      The subexpression on which the operator acts
    """

    operator: UnaryOperator
    exp: Exp


@dataclass(frozen=True, slots=True)
class Binary(Exp):
    """A binary operator expression.

    Attributes:
        operator: The binary operator which combines the two subexpressions
        left:     The left-hand operand
        right:    The right-hand operand
    """

    operator: BinaryOperator
    left: Exp
    right: Exp


@dataclass(frozen=True, slots=True)
class Assignment(Exp):
    """An assignment expression.

    Attributes:
        lvalue: The left side of the assignment (ie: the thing being assigned to)
        rvalue: The expression that we're assiging to the lvalue.
    """

    lvalue: Exp
    rvalue: Exp


@dataclass(frozen=True, slots=True)
class CompoundAssignment(Exp):
    """A compound assignment expression (``+=``, ``-=``, ``*=``, etc.).

    Stores the lvalue and rhs separately so that the TAC generator can evaluate
    the lvalue address exactly once (C11 sec. 6.5.16.2), unlike the naive desugaring
    ``Assignment(lvalue, Binary(op, lvalue, rhs))`` which would evaluate lvalue twice.

    Attributes:
        lvalue: The left-hand side; its address is computed once.
        op: The underlying binary operator (``Add``, ``Subtract``, ...).
        rvalue: The right-hand side expression.
    """

    lvalue: Exp
    op: BinaryOperator
    rvalue: Exp


@dataclass(frozen=True, slots=True)
class Conditional(Exp):
    """A ternary (``?:``) expression.

    Attributes:
        condition: The condition to be tested for truthiness
        exp_true: The expression to be evaluated if condition is truthy
        exp_false: The expression to be evaluated if condition is falsy
    """

    condition: Exp
    exp_true: Exp
    exp_false: Exp


@dataclass(frozen=True, slots=True)
class FunctionCall(Exp):
    """A function call expression.

    The callee is an arbitrary expression so that both direct calls (where the
    callee is a ``Var`` naming a function) and indirect calls through a function
    pointer (where the callee is a ``Var`` or ``Dereference`` whose type is a
    pointer-to-function) are represented uniformly.

    Attributes:
        callee: Expression that yields the callable.  A ``Var`` for a named
            function call; a ``Dereference`` (or other expression) for an
            indirect call through a pointer.
        args: Ordered list of argument expressions.
    """

    callee: Exp
    args: list[Exp]


@dataclass(frozen=True, slots=True)
class Dereference(Exp):
    """Represents the dereferencing of a pointer type

    Attributes:
        exp: The expression which yields the address of the object the pointer
            points at.
    """

    exp: Exp


@dataclass(frozen=True, slots=True)
class AddrOf(Exp):
    """Represents the address which a pointer object is pointing at.

    Attributes:
        exp: The expression which yields the relevant address
    """

    exp: Exp


@dataclass(frozen=True, slots=True)
class Subscript(Exp):
    """Represents a subscripted expression in the C code of the form:
    ```exp1[exp2]```

    Attributes:
        exp1: The first (outer) subexpression
        exp2: The second (inner) subexpression
    """

    exp1: Exp
    exp2: Exp


@dataclass(frozen=True, slots=True)
class String(Exp):
    """Represents a string literal in C source code, e.g. ``"hello world!"``.

    For plain and ``u8``-prefixed strings the bytes are stored one byte per
    character (ASCII / UTF-8).  For wide and Unicode-prefixed strings the bytes
    are stored in little-endian target encoding: 4 bytes per element for
    ``L"..."`` (wchar_t = int32 on Linux x86-64), 2 bytes per element for
    ``u"..."`` (char16_t = uint16), 4 bytes per element for ``U"..."``
    (char32_t = uint32).  Escape sequences have already been resolved and
    adjacent same-encoding tokens have been concatenated.

    Attributes:
        string: The encoded byte content of the literal (NUL terminator not
            included).  Element count = ``len(string) // encoding.element_size``.
        encoding: The character encoding that determines the element type and
            bytes-per-element of the literal.
    """

    string: bytes
    encoding: StringEncoding = StringEncoding.CHAR


@dataclass(frozen=True, slots=True)
class SizeOf(Exp):
    """Expression returning the size of the type of the given expression.  Note that the
    expression is _not_ evaluated

    Attributes:
        exp: The expression we wish to know the size of
    """

    exp: Exp


@dataclass(frozen=True, slots=True)
class SizeOfT(Exp):
    """Expression returning the size of the specified type.

    Attributes:
        t: The type whose size we wish to return
    """

    t: Type


@dataclass(frozen=True, slots=True)
class AlignOf(Exp):
    """Expression returning the alignment of the type of the given expression.

    The expression is not evaluated; only its type is inspected.

    Attributes:
        exp: The expression whose type alignment is queried.
    """

    exp: Exp


@dataclass(frozen=True, slots=True)
class AlignOfT(Exp):
    """Expression returning the alignment requirement of the specified type.

    Attributes:
        t: The type whose alignment requirement is returned.
    """

    t: Type


class Initializer(ABC):
    """Abstract base for initializer objects.  Wraps the SingleInit and CompoundInit
    AST Nodes
    """


@dataclass(frozen=True, slots=True)
class SingleInit(Initializer):
    """Represents a variable initializer for a scalar variable type

    Attributes:
        exp: The expression with which we are initialising the variable
    """

    exp: Exp


@dataclass(frozen=True, slots=True)
class CompoundInit(Initializer):
    """Represents a variable initializer for compound variable types

    Attributes:
        initializers: An ordered list of initializations (could be empty)
        init_type: The C type this compound initializer targets, or ``None`` for
            array initializers.  Set to ``Structure(tag)`` by ``_typecheck_init``
            when the target type is a struct so that ``_flatten_helper`` can emit
            inter-member and trailing padding when building the static-init list.
        union_member_index: For union ``CompoundInit`` nodes produced by a designated
            initialiser (e.g. ``{.f = 3.14}``), the zero-based index of the union
            member being initialised.  ``None`` means the first member (index 0),
            which preserves backward compatibility with non-designated union inits.
    """

    initializers: list[Initializer]
    init_type: Type | None = None
    union_member_index: int | None = None


@dataclass(frozen=True, slots=True)
class MemberDesignatorInit(Initializer):
    """A designated initialiser targeting a struct or union member by name: ``.member = init``.

    Produced by the parser for ``{ .member = value }`` syntax and expanded by
    semantic analysis into a positional ``CompoundInit`` before type-checking.

    Attributes:
        member: The name of the struct or union member being initialised.
        init: The sub-initialiser for that member's value.
    """

    member: str
    init: Initializer


@dataclass(frozen=True, slots=True)
class IndexDesignatorInit(Initializer):
    """A designated initialiser targeting an array element by index: ``[expr] = init``.

    Produced by the parser for ``{ [constant] = value }`` syntax and expanded by
    semantic analysis into a positional ``CompoundInit`` before type-checking.

    Attributes:
        index: The constant expression giving the target array index.
        init: The sub-initialiser for that element's value.
    """

    index: Exp
    init: Initializer


@dataclass(frozen=True, slots=True)
class Dot(Exp):
    """Member access on a structure value: ``structure.member``.

    Attributes:
        structure: The expression yielding the structure object.
        member: The name of the member being accessed.
    """

    structure: Exp
    member: str


@dataclass(frozen=True, slots=True)
class Arrow(Exp):
    """Member access through a pointer to a structure: ``pointer->member``.

    Attributes:
        pointer: The expression yielding the pointer to the structure.
        member: The name of the member being accessed.
    """

    pointer: Exp
    member: str


@dataclass(frozen=True, slots=True)
class CompoundLiteral(Exp):
    """A compound literal expression: ``(Type){initializer-list}``.

    Compound literals (C11 §6.5.2.5) produce an lvalue of the given type
    initialised with the supplied initialiser list.  At block scope the object
    has automatic storage duration; at file scope it has static storage duration.

    The parser creates this node whenever a cast-like ``(type)`` prefix is
    immediately followed by ``{``; semantic analysis validates the target type
    and type-checks the initialiser, then the TAC generator allocates a fresh
    temporary variable and emits the corresponding initialisation code.

    Attributes:
        target_type: The declared C type of the compound-literal object.
        init: The ``SingleInit`` or ``CompoundInit`` initialiser.
    """

    target_type: Type
    init: Initializer


@dataclass(frozen=True, slots=True)
class VaStart(Exp):
    """``__builtin_va_start(ap, last)`` — initialise a ``va_list``.

    Lowers to a ``tac.VaStart`` instruction that populates the
    ``__va_list_tag`` struct from the calling-convention register counts.

    Attributes:
        ap: Lvalue expression of type ``__va_list_tag`` (the ``va_list`` object).
        last: The last named parameter expression; not used after semantic
            analysis but retained for error-reporting.
    """

    ap: Exp
    last: Exp


@dataclass(frozen=True, slots=True)
class VaArg(Exp):
    """``__builtin_va_arg(ap, T)`` — fetch the next variadic argument of type ``T``.

    Attributes:
        ap: Lvalue expression of type ``__va_list_tag`` (the ``va_list`` object).
        arg_type: The C type of the argument to retrieve.
    """

    ap: Exp
    arg_type: Type


@dataclass(frozen=True, slots=True)
class VaEnd(Exp):
    """``__builtin_va_end(ap)`` — mark end of variadic argument traversal.

    On x86-64 Linux this is a no-op; the node is retained for completeness
    and to allow future implementations to insert cleanup code.

    Attributes:
        ap: Lvalue expression of type ``__va_list_tag`` (the ``va_list`` object).
    """

    ap: Exp


@dataclass(frozen=True, slots=True)
class VaCopy(Exp):
    """``__builtin_va_copy(dest, src)`` — copy one ``va_list`` to another.

    Copies the state of ``src`` into ``dest`` so that ``dest`` can traverse
    the same variadic arguments independently.

    Attributes:
        dest: Lvalue expression of type ``__va_list_tag`` — the destination.
        src: Expression of type ``__va_list_tag`` — the source.
    """

    dest: Exp
    src: Exp


@dataclass(frozen=True, slots=True)
class BuiltinUnreachable(Exp):
    """``__builtin_unreachable()`` — marks a code path as unreachable.

    Has void type.  The code generator emits a ``ud2`` instruction, which
    raises ``SIGILL`` if control ever reaches it.
    """


@dataclass(frozen=True, slots=True)
class OffsetOf(Exp):
    """``__builtin_offsetof(type, member)`` — byte offset of a struct member.

    Resolved to a ``ULongType`` constant during semantic analysis.

    Attributes:
        t: The struct or union type whose layout is queried.
        member: The name of the member whose offset is requested.
    """

    t: Type
    member: str


@dataclass(frozen=True, slots=True)
class Generic(Exp):
    """``_Generic(controlling-expression, association-list)`` — type-generic selection (C11 §6.5.1.1).

    Selects one expression from the association list based on the type of the
    controlling expression.  The controlling expression is not evaluated; only
    its type is inspected.  Folded entirely to the selected branch by semantic
    analysis so that no ``Generic`` node ever appears in the resolved AST.

    Attributes:
        controlling_exp: The expression whose type drives the selection.
        associations: Ordered sequence of ``(type, expression)`` pairs.  A
            ``None`` type denotes the ``default`` association.
    """

    controlling_exp: Exp
    associations: tuple[tuple[Type | None, Exp], ...]
