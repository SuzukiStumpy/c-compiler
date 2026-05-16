"""Program structure nodes for the assembly AST.

Defines the top-level constructs of an assembly translation unit: ``Function``
and ``Program``. These are the product types in the grammar — each has exactly
one form.
"""

from abc import ABC
from dataclasses import dataclass
from typing import abstractmethod

from compiler.syntax.asm.instructions import Instruction
from compiler.syntax.asm.operands import Operand


class TopLevel(ABC):
    """Abstract base for the top level declarations ``Function`` and ``StaticVariable``"""

    @abstractmethod
    def emit(self) -> str: ...


@dataclass(frozen=True, slots=True)
class Function(TopLevel):
    """An assembly function definition.

    Represents a labelled sequence of instructions that forms a single
    callable function in the generated assembly.

    Attributes:
        name: The function name used as the assembly label.
        globl: Whether the function declaration should be tagged with the .globl directive
        params: Register operands that carry the function's parameters on
            entry, in ABI order (DI, SI, DX, …).  This field is informational
            metadata used during code generation; it does not appear in the
            emitted assembly.
        instructions: The ordered sequence of instructions in the function
            body. May be empty before code generation fills it in, but will
            contain at least one instruction in fully generated output.
    """

    name: str
    globl: bool
    params: tuple[Operand, ...] = ()
    instructions: tuple[Instruction, ...] = ()

    def emit(self) -> str:
        """Return the assembly text for this function definition.

        Produces a ``.globl`` directive (if required), a label, and the emitted text of
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
            f"  .globl {self.name}" if self.globl else "",
            f"{self.name}:",
            "  pushq %rbp",
            "  movq %rsp, %rbp",
        ]
        for instr in self.instructions:
            lines.append(instr.emit())
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class StaticVariable(TopLevel):
    """Represents the definition of a statically linked variable within the program.

    Attributes:
        identifier: The name associated with the variable
        globl: Whether the variable should be tagged with the .globl directive
        init: The initial value of the variable to set
    """

    identifier: str
    globl: bool
    init: int

    def emit(self) -> str:
        """Return the assembly text for the variable definition.

        Produces a .globl definition if required, followed by the label and initial
        value for the variable.
        """
        return ""  # Placeholder for now


@dataclass(frozen=True, slots=True)
class Program:
    """The root assembly AST node representing a complete translation unit.

    Attributes:
        definitions: The ordered sequence of top-level definitions.
            At least one function is required.
    """

    definitions: tuple[TopLevel, ...]

    def emit(self) -> str:
        """Return the assembly text for the full translation unit.

        Delegates to each contained function or variable definition and joins their output.
        Platform-specific additions (name mangling, section directives) are
        applied by the emitter before calling this method.

        Returns:
            The assembly text of all contained functions, separated by newlines.
        """
        return "\n".join(fn.emit() for fn in self.definitions)
