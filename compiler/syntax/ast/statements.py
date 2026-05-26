"""Statement nodes for the AST.

Defines the ``Statement`` abstract base class and its concrete variants. Each
statement form supported by the compiler is represented as a subclass of
``Statement``.
"""

from abc import ABC
from dataclasses import dataclass

from compiler.syntax.ast.expressions import Constant, Exp


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
        exp: The (optional) expression whose value is returned to the caller.
    """

    exp: Exp | None


@dataclass(frozen=True, slots=True)
class Expression(Statement):
    """An expression statement (an expression evaluated for its side effects).

    Attributes:
        exp: The expression contained within the statement.
    """

    exp: Exp


@dataclass(frozen=True, slots=True)
class Null(Statement):
    """An empty statement; used as a placeholder where a statement is syntactically required."""


@dataclass(frozen=True, slots=True)
class If(Statement):
    """An ``if`` statement with an optional ``else`` clause.

    Attributes:
        condition: The condition to evaluate.
        then: The statement executed when ``condition`` is truthy.
        else_: The statement executed when ``condition`` is falsy, or ``None``
            if there is no ``else`` branch.
    """

    condition: Exp
    then: Statement
    else_: Statement | None


@dataclass(frozen=True, slots=True)
class Goto(Statement):
    """A ``goto`` statement; transfers control unconditionally to a named label
    within the current function scope.

    Attributes:
        target: The identifier of the target label.
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


@dataclass(frozen=True, slots=True)
class Break(Statement):
    """A ``break`` statement; transfers control to just after the enclosing loop or switch.

    Attributes:
        identifier: The identifier of the enclosing loop or switch, assigned during
            the loop-labelling pass of semantic analysis.
    """

    identifier: str = ""


@dataclass(frozen=True, slots=True)
class Continue(Statement):
    """A ``continue`` statement; transfers control to the update expression of the enclosing loop.

    Attributes:
        identifier: The identifier of the enclosing loop, assigned during the
            loop-labelling pass of semantic analysis.
    """

    identifier: str = ""


@dataclass(frozen=True, slots=True)
class While(Statement):
    """A ``while`` loop.

    The condition is tested before each iteration; the body executes only while
    the condition is truthy.

    Attributes:
        condition: The expression tested before each iteration.
        body: The statement executed on each iteration.
        identifier: The unique identifier for this loop, assigned during the
            loop-labelling pass of semantic analysis.
    """

    condition: Exp
    body: Statement
    identifier: str = ""


@dataclass(frozen=True, slots=True)
class DoWhile(Statement):
    """A ``do``/``while`` loop.

    The body is always executed at least once; the condition is tested after
    each iteration and the loop continues while it is truthy.

    Attributes:
        body: The statement executed on each iteration.
        condition: The expression tested after each iteration.
        identifier: The unique identifier for this loop, assigned during the
            loop-labelling pass of semantic analysis.
    """

    body: Statement
    condition: Exp
    identifier: str = ""


@dataclass(frozen=True, slots=True)
class Switch(Statement):
    """A ``switch`` statement.

    The body is a general statement (typically a ``Compound``) that may contain
    ``Case`` and ``Default`` nodes at any depth.  Fall-through between cases is
    the default behaviour; a ``Break`` exits the switch.

    Attributes:
        control: The controlling expression evaluated once and matched against
            each ``Case`` value.
        body: The statement forming the body of the switch, usually a compound
            block containing ``Case`` and ``Default`` nodes.
        identifier: The unique identifier for this switch, assigned during
            semantic analysis and used to resolve ``Break`` targets.
    """

    control: Exp
    body: Statement
    identifier: str = ""


@dataclass(frozen=True, slots=True)
class Case(Statement):
    """A ``case`` label within an enclosing ``switch``.

    Acts as a jump target: if the switch control expression equals ``value``,
    execution jumps here.  If no ``Break`` follows, execution falls through into
    the next case.

    Attributes:
        value: The integer constant matched against the enclosing ``switch``
            control expression.  Must be a ``Constant`` node; non-constant
            case values are rejected by the parser.
        statement: The statement immediately governed by this label.
        identifier: The unique identifier for this case, derived from the
            enclosing switch's identifier during semantic analysis.
    """

    value: Constant
    statement: Statement
    identifier: str = ""


@dataclass(frozen=True, slots=True)
class Default(Statement):
    """The ``default`` label within an enclosing ``switch``.

    Acts as the fallback jump target when no ``Case`` value matches the control
    expression.  At most one ``Default`` may appear per ``switch``, but it may
    appear at any position within the body.

    Attributes:
        statement: The statement immediately governed by this label.
        identifier: The unique identifier for this default, derived from the
            enclosing switch's identifier during semantic analysis.
    """

    statement: Statement
    identifier: str = ""
