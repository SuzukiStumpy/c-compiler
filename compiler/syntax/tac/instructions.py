"""Instruction nodes for the tac AST.

Defines the ``Instruction`` abstract base class and its concrete variants.
Each instruction form supported by the code generator is represented as a
subclass of ``Instruction``.
"""

from abc import ABC
from dataclasses import dataclass

from compiler.syntax.tac.operators import BinaryOperator, UnaryOperator
from compiler.syntax.tac.values import Val


class Instruction(ABC):
    """Abstract base for all tac instruction nodes.

    Each instruction form is represented as a concrete subclass. Consumers
    should use structural pattern matching (``match``/``case``) to dispatch on
    the concrete variant rather than inspecting fields directly.
    """


@dataclass(frozen=True, slots=True)
class Return(Instruction):
    """Represents a return value from a function.

    Attributes:
        val:  The value (of val type) being returned.
    """

    val: Val


@dataclass(frozen=True, slots=True)
class Unary(Instruction):
    """Represents a unary expression within the tac

    Attributes:
        operator: The type of the operation
        src:  The source value for the expression
        dst:  The destination for the result of the expression
    """

    operator: UnaryOperator
    src: Val
    dst: Val


@dataclass(frozen=True, slots=True)
class Binary(Instruction):
    """Represents a binary expression within the tac

    Attributes:
        operator: The type of the operation
        src1: The first source value for the expression
        src2: The second source value for the expression
        dst:  The destination for the result of the expression
    """

    operator: BinaryOperator
    src1: Val
    src2: Val
    dst: Val


@dataclass(frozen=True, slots=True)
class Copy(Instruction):
    """Represents a copy operation with the tac

    Attributes:
        src: The source location to copy from
        dst: The destination location to copy to
    """

    src: Val
    dst: Val


@dataclass(frozen=True, slots=True)
class Jump(Instruction):
    """Represents an unconditional jump within the tac

    Attributes:
        target: The labelled location to jump to
    """

    target: str


@dataclass(frozen=True, slots=True)
class JumpIfZero(Instruction):
    """Represents a conditional jump if the condition evaluates
    to zero.

    Attributes:
        condition: The condition to evaluate
        target: The labelled location to jump to
    """

    condition: Val
    target: str


@dataclass(frozen=True, slots=True)
class JumpIfNotZero(Instruction):
    """Represents a conditional jump if the condition evaluates
    to something other than zero

    Attributes:
        condition: The condition to evaluate
        target: The labelled location to jump to
    """

    condition: Val
    target: str


@dataclass(frozen=True, slots=True)
class Label(Instruction):
    """Represents a named location within the code.  Generally used
    as a jump target.

    Attributes:
        name: The name of the label
    """

    name: str
