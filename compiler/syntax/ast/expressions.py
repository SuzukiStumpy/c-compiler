"""Expression nodes for the AST.

Defines the ``Exp`` abstract base class and its concrete variants. Each
expression form supported by the compiler is represented as a subclass of
``Exp``. New expression types (function calls, etc.) are added here as language
support is extended.
"""

from abc import ABC
from dataclasses import dataclass

from compiler.syntax.ast.operators import BinaryOperator, UnaryOperator


class Exp(ABC):
    """Abstract base for all expression nodes.

    Each expression form supported by the compiler is represented as a
    concrete subclass. Consumers should use structural pattern matching
    (``match``/``case``) to dispatch on the concrete variant rather than
    inspecting fields directly.
    """


@dataclass(frozen=True, slots=True)
class Constant(Exp):
    """An integer constant expression.

    Attributes:
        value: The integer value of the constant as parsed from the source.
    """

    value: int


@dataclass(frozen=True, slots=True)
class Var(Exp):
    """A named variable in the program.

    Attributes:
        name: The name of the variable.
    """

    name: str


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
