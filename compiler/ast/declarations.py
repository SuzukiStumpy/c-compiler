"""Program structure nodes for the AST.

Defines the top-level constructs of a C translation unit: ``Parameter``,
``Function``, and ``Program``. These are the product types in the grammar —
they have no abstract base because each has exactly one form.
"""

from dataclasses import dataclass

from compiler.ast.statements import Statement
from compiler.ast.types import Type


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
    list, and body. The body is an ordered sequence of statements executed when
    the function is called.

    Attributes:
        return_type: The declared return type of the function.
        name: The function name as it appears in the source.
        params: The ordered sequence of parameters. Empty for functions declared
            with ``void`` in the parameter list.
        body: The ordered sequence of statements comprising the function body.
            At least one statement is required by the grammar.
    """

    return_type: Type
    name: str
    params: tuple[Parameter, ...]
    body: tuple[Statement, ...]


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
