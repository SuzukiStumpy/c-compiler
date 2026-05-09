"""Statement nodes for the AST.

Defines the ``Statement`` abstract base class and its concrete variants. Each
statement form supported by the compiler is represented as a subclass of
``Statement``. New statement types (if, while, for, assignments, etc.) are
added here as language support is extended.
"""

from abc import ABC
from dataclasses import dataclass

from compiler.syntax.ast.expressions import Exp


class Statement(ABC):
    """Abstract base for all statement nodes.

    Each statement form supported by the compiler is represented as a
    concrete subclass. Consumers should use structural pattern matching
    (``match``/``case``) to dispatch on the concrete variant rather than
    inspecting fields directly.
    """


@dataclass(frozen=True, slots=True)
class Return(Statement):
    """A ``return`` statement.

    Attributes:
        exp: The expression whose value is returned to the caller.
    """

    exp: Exp


@dataclass(frozen=True, slots=True)
class Expression(Statement):
    """An expression statement (ie: an expression which doesn't return a value)

    Attributes:
        exp: The expression contained within the statement
    """

    exp: Exp


@dataclass(frozen=True, slots=True)
class Null(Statement):
    """An empty statement for when a placeholder is needed in the source."""


@dataclass(frozen=True, slots=True)
class If(Statement):
    """An if statement along with optional else clause

    Attributes:
        condition: The condition to check
        then: the statement to execute if the condition is truthy
        else_: the statement to execute if the condition is falsy
    """

    condition: Exp
    then: Statement
    else_: Statement | None


@dataclass(frozen=True, slots=True)
class Goto(Statement):
    """A goto statement - sends program flow directly to the statement
    identifed by the specified label within the current function scope.

    Attributes:
        target: The label that we are targeting.
    """

    target: str


@dataclass(frozen=True, slots=True)
class Label(Statement):
    """A labeled statement, associating an identifier with the statement that
    immediately follows it so that it may be targeted by a ``goto``.

    Attributes:
        name: The label identifier.
        statement: The statement governed by this label.
    """

    name: str
    statement: Statement
