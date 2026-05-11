"""Program structure nodes for the AST.

Defines the top-level constructs of a C translation unit: ``Parameter``,
``Function``, ``Program``, ``Declaration``, the ``BlockItem`` sum type
(``S`` and ``D``) that composes statements and declarations into a block,
the ``Block`` type alias, and the ``Compound`` statement node.
"""

from abc import ABC
from dataclasses import dataclass

from compiler.syntax.ast.expressions import Exp
from compiler.syntax.ast.statements import Statement
from compiler.syntax.ast.types import Type


@dataclass(frozen=True, slots=True)
class Declaration:
    """A variable declaration, optionally with an initialiser.

    Attributes:
        name: The name of the variable being declared.
        init: The initialisation expression, or ``None`` if the variable is
            declared without an initial value.
    """

    name: str
    init: Exp | None


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
        declaration: The declaration node.
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
    """A function definition.

    Represents a complete C function including its return type, name, parameter
    list, and body. The body is an ordered sequence of block items — statements
    and declarations interleaved — executed when the function is called.

    Attributes:
        return_type: The declared return type of the function.
        name: The function name as it appears in the source.
        params: The ordered sequence of parameters. Empty for functions declared
            with ``void`` in the parameter list.
        body: The ordered sequence of block items comprising the function body.
            May be empty.
    """

    return_type: Type
    name: str
    params: tuple[Parameter, ...]
    body: tuple[BlockItem, ...]


@dataclass(frozen=True, slots=True)
class Program:
    """The root AST node representing a complete translation unit.

    A translation unit consists of one or more top-level function definitions.
    In the current minimal grammar only a single ``main`` function is valid, but
    the node holds a sequence of functions in anticipation of future language
    support (multiple functions, forward declarations, etc.).

    Attributes:
        functions: The ordered sequence of top-level function definitions.
            At least one function is required by the grammar.
    """

    functions: tuple[Function, ...]
