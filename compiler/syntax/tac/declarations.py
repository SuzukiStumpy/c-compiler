"""Program structure nodes for the tac AST.

Defines the top-level constructs of a tac translation unit: ``Function``
and ``Program``.  These are the product types in the grammar - each has exactly
one form.
"""

from dataclasses import dataclass

from compiler.syntax.tac.instructions import Instruction


@dataclass(frozen=True, slots=True)
class Function:
    """A tac function definition.

    Represents a labelled sequence of instructions that forms a single callable
    function in the generated assembly.

    Attributes:
        name: The function name used as the assembly label
        body: The ordered sequence of instructions in the function body.  May
              be empty before code generation fills it in, but will contain at
              least one instruction in fully generated output.
    """

    name: str
    body: tuple[Instruction, ...]


@dataclass(frozen=True, slots=True)
class Program:
    """A tac program definition.

    Represents the root of the tac AST and should contain the function definitions.

    In the current implementation Program consists of _exactly_ one function
    definition

    Attributes:
        function: The function that is the entry point for the program
    """

    function: Function
