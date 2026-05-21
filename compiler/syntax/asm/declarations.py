"""Program structure nodes for the assembly AST.

Defines the top-level constructs of an assembly translation unit: ``Function``
and ``Program``. These are the product types in the grammar — each has exactly
one form.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from compiler.syntax.asm.instructions import Instruction
from compiler.syntax.asm.operands import Operand
from compiler.syntax.asm.values import DoubleInit, IntInit, LongInit, StaticInit


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
            "   .text",
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
        alignment: The number of bytes to align the variable to in memory
        init: The initial value of the variable to set
    """

    identifier: str
    globl: bool
    alignment: int
    init: StaticInit

    def emit(self) -> str:
        """Return the assembly text for the variable definition.

        Produces a .globl definition if required, followed by the label and initial
        value for the variable.
        """
        lines = []

        if self.globl:
            lines.append(f"   .globl {self.identifier}")

        is_zero_int = self.init.value == 0 and isinstance(self.init, (IntInit, LongInit))
        lines.append("   .bss" if is_zero_int else "   .data")
        lines.append(f"   .balign {self.alignment}")
        lines.append(f"{self.identifier}:")

        match self.init:
            case IntInit(value=0):
                lines.append("   .zero 4")
            case IntInit(value=v):
                lines.append(f"   .long {v}")
            case LongInit(value=0):
                lines.append("   .zero 8")
            case LongInit(value=v):
                lines.append(f"   .quad {v}")
            case DoubleInit(value=0.0):
                lines.append("   .zero 8")
            case DoubleInit() as d:
                lines.append(f"   .quad {d.as_bits()}")

        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class StaticConstant(TopLevel):
    """Represents the definition of a statically linked constant within the program.

    Attributes:
        identifier: The name associated with the value
        alignment: The number of bytes to align the value to in memory
        init: The value of the constant to set
    """

    identifier: str
    alignment: int
    init: StaticInit

    def emit(self) -> str:
        """Return the assembly text for the constant definition.

        Constants are placed in ``.rodata`` (read-only data).  Double values are
        emitted as their raw IEEE 754 bit pattern via ``.quad`` to guarantee
        exact bit-level fidelity.

        Returns:
            A multi-line string containing the section directive, alignment
            directive, label, and data directive.
        """
        lines = [
            "   .section .rodata",
            f"   .balign {self.alignment}",
            f"{self.identifier}:",
        ]
        match self.init:
            case DoubleInit() as d:
                lines.append(f"   .quad {d.as_bits()}")
            case IntInit(value=v):
                lines.append(f"   .long {v}")
            case LongInit(value=v):
                lines.append(f"   .quad {v}")
        return "\n".join(lines)


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
