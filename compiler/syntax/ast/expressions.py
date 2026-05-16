"""Expression nodes for the AST.

Defines the ``Exp`` abstract base class and its concrete variants. Each
expression form supported by the compiler is represented as a subclass of
``Exp``. New expression types (function calls, etc.) are added here as language
support is extended.
"""

from abc import ABC
from dataclasses import dataclass

from compiler.syntax.ast.constants import Const
from compiler.syntax.ast.operators import BinaryOperator, UnaryOperator
from compiler.syntax.ast.types import Type


class Exp(ABC):
    """Abstract base for all expression nodes.

    Each expression form supported by the compiler is represented as a
    concrete subclass. Consumers should use structural pattern matching
    (``match``/``case``) to dispatch on the concrete variant rather than
    inspecting fields directly.
    """


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
    """A function call within the source code.

    Attributes:
        name: The name of the function.
        args: a list of function arguments
    """

    name: str
    args: list[Exp]
