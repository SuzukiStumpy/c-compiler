"""Instruction nodes for the assembly AST.

Defines the ``Instruction`` abstract base class and its concrete variants.
Each instruction form supported by the code generator is represented as a
subclass of ``Instruction``.
"""

from abc import ABC
from dataclasses import dataclass

from compiler.asm.operands import Operand


class Instruction(ABC):
    """Abstract base for all assembly instruction nodes.

    Each instruction form is represented as a concrete subclass. Consumers
    should use structural pattern matching (``match``/``case``) to dispatch on
    the concrete variant rather than inspecting fields directly.
    """


@dataclass(frozen=True, slots=True)
class Mov(Instruction):
    """A ``movl`` instruction that copies a value from one operand to another.

    Attributes:
        src: The source operand whose value is copied.
        dst: The destination operand that receives the value.
    """

    src: Operand
    dst: Operand


@dataclass(frozen=True, slots=True)
class Ret(Instruction):
    """A ``ret`` instruction that returns from the current function.

    Carries no fields — the instruction has no operands.
    """
