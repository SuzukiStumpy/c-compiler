"""Program structure nodes for the AST.

Defines the top-level constructs of a C translation unit: the ``VariableDeclaration``
node, the ``Declaration`` sum type (``FunDecl`` and ``VarDecl``), the ``BlockItem``
sum type (``S`` and ``D``) that composes statements and declarations into a block,
the ``Block`` type alias, the ``Compound`` statement node, the ``ForInit`` sum type
(``InitDecl`` and ``InitExp``) for ``for``-loop initialisers, the ``For`` statement
node, and the ``Parameter``, ``Function``, and ``Program`` nodes.
"""

from abc import ABC
from dataclasses import dataclass

from compiler.syntax.ast.expressions import Exp
from compiler.syntax.ast.statements import Statement
from compiler.syntax.ast.types import Type


@dataclass(frozen=True, slots=True)
class VariableDeclaration:
    """A variable declaration, optionally with an initialiser.

    Attributes:
        name: The name of the variable being declared.
        init: The initialisation expression, or ``None`` if the variable is
            declared without an initial value.
    """

    name: str
    init: Exp | None


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
        declaration: The declaration node — either a ``FunDecl`` or a ``VarDecl``.
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
    """

    name: str
    params: tuple[Parameter, ...]
    body: Block | None


@dataclass(frozen=True, slots=True)
class FunDecl(Declaration):
    """A function declaration wrapped as a declaration node.

    Used when a function declaration appears as a block item (``D``).

    Attributes:
        function_declaration: The underlying ``Function`` node.
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
class Program:
    """The root AST node representing a complete translation unit.

    A translation unit consists of one or more top-level function declarations
    or definitions.

    Attributes:
        functions: The ordered sequence of top-level function declarations.
            At least one function is required by the grammar.
    """

    functions: tuple[Function, ...]


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
        init: The initialiser clause — either a declaration (``InitDecl``) or
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
