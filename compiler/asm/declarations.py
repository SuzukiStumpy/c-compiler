"""Program structure nodes for the assembly AST.

Defines the top-level constructs of an assembly translation unit: ``Function``
and ``Program``. These are the product types in the grammar — each has exactly
one form.
"""

from dataclasses import dataclass

from compiler.asm.instructions import Instruction


@dataclass(frozen=True, slots=True)
class Function:
    """An assembly function definition.

    Represents a labelled sequence of instructions that forms a single
    callable function in the generated assembly.

    Attributes:
        name: The function name used as the assembly label.
        instructions: The ordered sequence of instructions in the function
            body. May be empty before code generation fills it in, but will
            contain at least one instruction in fully generated output.
    """

    name: str
    instructions: tuple[Instruction, ...]


@dataclass(frozen=True, slots=True)
class Program:
    """The root assembly AST node representing a complete translation unit.

    In the current grammar a translation unit contains exactly one function
    definition.

    Attributes:
        function: The single top-level function definition.
    """

    function: Function
