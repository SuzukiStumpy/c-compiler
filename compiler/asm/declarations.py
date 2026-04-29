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

    def emit(self) -> str:
        """Return the assembly text for this function definition.

        Produces a ``.globl`` directive, a label, and the emitted text of
        each instruction, all joined by newlines.

        Returns:
            A multi-line string of the form::

                  .globl <name>
                <name>:
                  <instruction 1>
                  <instruction 2>
                  ...
        """
        lines = [f"  .globl {self.name}", f"{self.name}:"]
        lines.extend(instr.emit() for instr in self.instructions)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Program:
    """The root assembly AST node representing a complete translation unit.

    In the current grammar a translation unit contains exactly one function
    definition.

    Attributes:
        function: The single top-level function definition.
    """

    function: Function

    def emit(self) -> str:
        """Return the assembly text for the full translation unit.

        Delegates to the single contained function definition.  Platform-
        specific additions (name mangling, section directives) are applied
        by the emitter before calling this method.

        Returns:
            The assembly text of the contained function.
        """
        return self.function.emit()
