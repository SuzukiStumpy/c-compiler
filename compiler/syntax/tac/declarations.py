"""Program structure nodes for the tac AST.

Defines the top-level constructs of a tac translation unit: ``Function``
and ``Program``.  These are the product types in the grammar - each has exactly
one form.
"""

from abc import ABC
from dataclasses import dataclass

from compiler.syntax.ast.types import Type
from compiler.syntax.tac.instructions import Instruction
from compiler.syntax.tac.values import StaticInit


class TopLevel(ABC):
    """Abstract class representing the top level of a TACKY program.  Encapsulates global
    function definitions and static variables."""


@dataclass(frozen=True, slots=True)
class Function(TopLevel):
    """A tac function definition.

    Represents a labelled sequence of instructions that forms a single callable
    function in the generated assembly.

    Attributes:
        name:   The function name used as the assembly label
        globl: Whether to mark a function definition with the ``.globl`` directive in the assembly
        params: The ordered sequence of parameters that can be passed to the function
        body:   The ordered sequence of instructions in the function body.  May
                be empty before code generation fills it in, but will contain at
                least one instruction in fully generated output.
        source_file: Path of the source file where this function was declared.
            Propagated from ``ast.Function``; empty when not available.
        source_line: 1-indexed line number of the function declaration.
            Propagated from ``ast.Function``; zero when not available.
    """

    name: str
    globl: bool = False
    params: tuple[str, ...] = ()
    body: tuple[Instruction, ...] = ()
    source_file: str = ""
    source_line: int = 0


@dataclass(frozen=True, slots=True)
class StaticVariable(TopLevel):
    """A globally defined static variable declaration.

    Attributes:
        name: The symbol representing the variable.
        globl: Whether the variable is declared with the ``.globl`` directive in the assembly.
        t: The C type of the variable, carried through from the main AST.
        init: The ordered sequence of zero or more initialisers for the variable. Scalars have
            exactly one entry; arrays have one entry per element; zero-initialised variables
            have an empty tuple.
        align_as: Explicit alignment override in bytes from ``_Alignas``, or ``None`` to use
            the natural type alignment.
    """

    name: str
    globl: bool
    t: Type
    init: tuple[StaticInit, ...]
    align_as: int | None = None


@dataclass(frozen=True, slots=True)
class StaticConstant(TopLevel):
    """A globally defined static constant declaration.

    Attributes:
        name: The symbol representing the constant.
        t: The type of the constant
        init: The value of the constant
    """

    name: str
    t: Type
    init: StaticInit


@dataclass(frozen=True, slots=True)
class Program:
    """A tac program definition.

    Represents the root of the tac AST and contains all ``TopLevel`` definitions
    in the translation unit.

    Attributes:
        top_level: The ordered sequence of ``TopLevel`` definitions in the program.
            At least one function is required.
    """

    top_level: tuple[TopLevel, ...]

    @property
    def functions(self) -> tuple[Function, ...]:
        """Return only the ``Function`` entries from ``top_level``.

        Returns:
            A tuple containing every ``Function`` in ``top_level``, in order.
        """
        return tuple(t for t in self.top_level if isinstance(t, Function))
