"""Program structure nodes for the assembly AST.

Defines the top-level constructs of an assembly translation unit: ``Function``
and ``Program``. These are the product types in the grammar — each has exactly
one form.
"""

from dataclasses import dataclass

from compiler.syntax.asm.instructions import Instruction


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

        When the instruction list contains an ``AllocateStack`` instruction,
        a standard frame-pointer prologue (``pushq %rbp; movq %rsp, %rbp``) is
        inserted immediately after the label, and a matching epilogue
        (``movq %rbp, %rsp; popq %rbp``) is inserted before every ``Ret``.

        Returns:
            A multi-line string of the form::

                  .globl <name>
                <name>:
                  [pushq %rbp]
                  [movq %rsp, %rbp]
                  <instruction 1>
                  ...
                  [movq %rbp, %rsp]
                  [popq %rbp]
                  ret
        """
        lines = [
            f"  .globl {self.name}",
            f"{self.name}:",
            "  pushq %rbp",
            "  movq %rsp, %rbp",
        ]
        for instr in self.instructions:
            lines.append(instr.emit())
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Program:
    """The root assembly AST node representing a complete translation unit.

    Attributes:
        functions: The ordered sequence of top-level function definitions.
            At least one function is required.
    """

    functions: tuple[Function, ...]

    def emit(self) -> str:
        """Return the assembly text for the full translation unit.

        Delegates to each contained function definition and joins their output.
        Platform-specific additions (name mangling, section directives) are
        applied by the emitter before calling this method.

        Returns:
            The assembly text of all contained functions, separated by newlines.
        """
        return "\n".join(fn.emit() for fn in self.functions)
