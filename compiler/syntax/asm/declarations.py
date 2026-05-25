"""Program structure nodes for the assembly AST.

Defines the top-level constructs of an assembly translation unit: ``Function``
and ``Program``. These are the product types in the grammar — each has exactly
one form.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from compiler.syntax.asm.instructions import Instruction, _asm_line
from compiler.syntax.asm.operands import Operand
from compiler.syntax.asm.values import CharInit, DoubleInit, IntInit, LongInit, StaticInit, StringInit, ZeroInit


def _string_comment(s: bytes, null_terminated: bool) -> str:
    """Build an assembly comment showing the human-readable form of string bytes.

    Non-printable bytes are rendered as escape sequences; printable ASCII is kept as-is.
    A trailing ``\\0`` is appended when ``null_terminated`` is ``True``.

    Args:
        s: The raw string bytes (NUL terminator not included).
        null_terminated: When ``True``, appends ``\\0`` to the comment text.

    Returns:
        A string of the form ``"      ; String: <content>"``, indented to match
        the directive column used by ``_asm_line``.
    """
    _ESCAPE: dict[int, str] = {0: r"\0", 9: r"\t", 10: r"\n", 13: r"\r", 92: r"\\"}
    parts: list[str] = []
    for b in s:
        if b in _ESCAPE:
            parts.append(_ESCAPE[b])
        elif 32 <= b < 127:
            parts.append(chr(b))
        else:
            parts.append(f"\\x{b:02x}")
    if null_terminated:
        parts.append(r"\0")
    return "      # String: " + "".join(parts)


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
            _asm_line(".globl", self.name) if self.globl else "",
            _asm_line(".text"),
            f"{self.name}:",
            _asm_line("pushq", "%rbp"),
            _asm_line("movq", "%rsp", "%rbp"),
        ]
        for instr in self.instructions:
            lines.append(instr.emit())
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class StaticVariable(TopLevel):
    """Represents the definition of a statically linked variable within the program.

    Attributes:
        identifier: The name associated with the variable.
        globl: Whether the variable should be tagged with the ``.globl`` directive.
        alignment: The number of bytes to align the variable to in memory.
        init: Ordered sequence of zero or more initialiser elements.  Scalars have
            exactly one entry; arrays have one entry per element (or a single
            ``ZeroInit`` spanning the whole region).  An empty tuple is treated as
            fully zero-initialised.
    """

    identifier: str
    globl: bool
    alignment: int
    init: tuple[StaticInit, ...]

    def emit(self) -> str:
        """Return the assembly text for the variable definition.

        Chooses ``.bss`` when every initialiser element is zero-valued (``ZeroInit``
        or a scalar integer init with value 0); otherwise uses ``.data``.  Each
        element is emitted as the appropriate assembler directive.

        Returns:
            A multi-line assembly string containing the section, alignment, label,
            and data directives.
        """
        lines = []

        if self.globl:
            lines.append(_asm_line(".globl", self.identifier))

        in_bss = all(
            isinstance(i, ZeroInit) or (isinstance(i, (CharInit, IntInit, LongInit)) and i.value == 0)
            for i in self.init
        )
        lines.append(_asm_line(".bss" if in_bss else ".data"))
        lines.append(_asm_line(".balign", str(self.alignment)))
        lines.append(f"{self.identifier}:")

        for init in self.init:
            match init:
                case CharInit(value=v):
                    lines.append(_asm_line(".byte", str(v)))
                case ZeroInit(bytes=n):
                    lines.append(_asm_line(".zero", str(n)))
                case IntInit(value=0):
                    lines.append(_asm_line(".zero", "4"))
                case IntInit(value=v):
                    lines.append(_asm_line(".long", str(v)))
                case LongInit(value=0):
                    lines.append(_asm_line(".zero", "8"))
                case LongInit(value=v):
                    lines.append(_asm_line(".quad", str(v)))
                case DoubleInit(value=0.0):
                    lines.append(_asm_line(".zero", "8"))
                case DoubleInit() as d:
                    lines.append(_asm_line(".quad", str(d.as_bits())))
                case StringInit(string=s, null_terminated=nt):
                    data = list(s) + ([0] if nt else [])
                    lines.append(_string_comment(s, nt))
                    lines.append(_asm_line(".byte", ", ".join(str(b) for b in data)))

        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class StaticConstant(TopLevel):
    """Represents the definition of a statically linked constant within the program.

    Attributes:
        identifier: The name associated with the value
        alignment: The number of bytes to align the value to in memory
        init: The value of the constant to set
        section: The assembler section directive body (platform-specific).
            Defaults to ``.rodata`` (Linux/ELF); the emitter replaces this with
            ``__TEXT,__const`` on macOS before calling ``emit()``.
    """

    identifier: str
    alignment: int
    init: StaticInit
    section: str = ".rodata"

    def emit(self) -> str:
        """Return the assembly text for the constant definition.

        Constants are placed in the section named by ``self.section``.  Double values are
        emitted as their raw IEEE 754 bit pattern via ``.quad`` to guarantee
        exact bit-level fidelity.

        Returns:
            A multi-line string containing the section directive, alignment
            directive, label, and data directive.
        """
        lines = [
            _asm_line(".section", self.section),
            _asm_line(".balign", str(self.alignment)),
            f"{self.identifier}:",
        ]
        match self.init:
            case DoubleInit() as d:
                lines.append(_asm_line(".quad", str(d.as_bits())))
            case IntInit(value=v):
                lines.append(_asm_line(".long", str(v)))
            case LongInit(value=v):
                lines.append(_asm_line(".quad", str(v)))
            case StringInit(string=s, null_terminated=nt):
                data = list(s) + ([0] if nt else [])
                lines.append(_string_comment(s, nt))
                lines.append(_asm_line(".byte", ", ".join(str(b) for b in data)))
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
