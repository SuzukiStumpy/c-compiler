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
        name:   The function name used as the assembly label
        params: The ordered sequence of parameters that can be passed to the function
        body:   The ordered sequence of instructions in the function body.  May
                be empty before code generation fills it in, but will contain at
                least one instruction in fully generated output.
    """

    name: str
    params: tuple[str, ...] = ()
    body: tuple[Instruction, ...] = ()


@dataclass(frozen=True, slots=True)
class Program:
    """A tac program definition.

    Represents the root of the tac AST and contains all function definitions
    in the translation unit.

    Attributes:
        functions: The ordered sequence of function definitions in the program.
            At least one function is required.
    """

    functions: tuple[Function, ...]
