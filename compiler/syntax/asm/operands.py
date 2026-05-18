"""Operand nodes for the assembly AST.

Defines the ``Operand`` abstract base class and its concrete variants. Each
operand form supported by the code generator is represented as a subclass of
``Operand``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum


class Width(IntEnum):
    """Byte width of a hardware register access.

    Used to select the correct register name when a logical register can be
    addressed at multiple granularities (e.g. ``%eax`` vs ``%al``).
    """

    BYTE = 1
    DWORD = 4
    QWORD = 8


class Registers(Enum):
    """Logical hardware registers and their width-specific assembly names.

    Each member's value is a ``{Width: str}`` mapping from access width to the
    AT&T register name for that width.  Add a new entry to the dict to support
    additional widths without touching any other code.
    """

    AX = {Width.QWORD: "rax", Width.DWORD: "eax", Width.BYTE: "al"}
    CX = {Width.QWORD: "rcx", Width.DWORD: "ecx", Width.BYTE: "cl"}
    DX = {Width.QWORD: "rdx", Width.DWORD: "edx", Width.BYTE: "dl"}
    DI = {Width.QWORD: "rdi", Width.DWORD: "edi", Width.BYTE: "dil"}
    SI = {Width.QWORD: "rsi", Width.DWORD: "esi", Width.BYTE: "sil"}
    R8 = {Width.QWORD: "r8", Width.DWORD: "r8d", Width.BYTE: "r8b"}
    R9 = {Width.QWORD: "r9", Width.DWORD: "r9d", Width.BYTE: "r9b"}
    R10 = {Width.QWORD: "r10", Width.DWORD: "r10d", Width.BYTE: "r10b"}
    R11 = {Width.QWORD: "r11", Width.DWORD: "r11d", Width.BYTE: "r11b"}
    SP = {Width.QWORD: "", Width.DWORD: "", Width.BYTE: ""}


class Operand(ABC):
    """Abstract base for all assembly operand nodes.

    Each operand form is represented as a concrete subclass. Consumers should
    use structural pattern matching (``match``/``case``) to dispatch on the
    concrete variant rather than inspecting fields directly.
    """

    @abstractmethod
    def emit(self) -> str:
        """Return the assembly text representation of this operand.

        Returns:
            A string suitable for use as an operand in an assembly instruction.
        """


@dataclass(frozen=True, slots=True)
class Imm(Operand):
    """An immediate (literal integer) operand.

    Attributes:
        value: The integer value of the immediate.
    """

    value: int

    def emit(self) -> str:
        """Return the immediate in AT&T syntax (e.g. ``$42``).

        Returns:
            The immediate value prefixed with ``$``.
        """
        return f"${self.value}"


@dataclass(frozen=True, slots=True)
class Register(Operand):
    """A hardware register operand at a specific byte width.

    Wraps a ``Registers`` enum value and a ``Width`` so that both the logical
    register and its access size are explicit in the AST.

    Attributes:
        reg: The logical register (e.g. ``Registers.AX``).
        width: The access width; defaults to ``Width.DWORD`` (32-bit).
    """

    reg: Registers
    width: Width = Width.DWORD

    def emit(self) -> str:
        """Return the register name in AT&T syntax.

        Returns:
            A string of the form ``%<name>``, e.g. ``%eax``, ``%al``, ``%r10b``.
        """
        return f"%{self.reg.value[self.width]}"


@dataclass(frozen=True, slots=True)
class Pseudo(Operand):
    """A pseudo-register representing a named temporary variable.

    Pseudo-registers are produced by the initial TAC-to-assembly translation
    pass and must be resolved to ``Stack`` operands by a subsequent fixup pass
    before emission.

    Attributes:
        name: The variable name from the TAC representation (e.g. ``"tmp.0"``).
    """

    name: str

    def emit(self) -> str:
        """Raise; pseudo-registers must be resolved before emission.

        Raises:
            AssertionError: Always — ``Pseudo`` operands must be replaced with
                ``Stack`` operands by the pseudo-replacement pass before any
                code is emitted.

        Returns:
            Never returns normally.
        """
        raise AssertionError(f"Pseudo operand '{self.name}' was not resolved before emission")


@dataclass(frozen=True, slots=True)
class Stack(Operand):
    """A stack memory operand at a fixed byte offset from ``%rbp``.

    Used to represent local variables and temporaries after pseudo-register
    replacement.  Offsets are negative for variables in the current frame
    (e.g. ``-4(%rbp)`` for the first 4-byte local).

    Attributes:
        offset: The signed byte offset from ``%rbp``.
    """

    offset: int

    def emit(self) -> str:
        """Return the stack reference in AT&T syntax.

        Returns:
            A string of the form ``<offset>(%rbp)``, e.g. ``-4(%rbp)``.
        """
        return f"{self.offset}(%rbp)"


@dataclass(frozen=True, slots=True)
class Data(Operand):
    """A RIP-relative reference to a statically-allocated variable.

    Used to address global and static variables via position-independent,
    RIP-relative addressing on x86-64.

    Attributes:
        name: The symbol name of the static variable.
    """

    name: str

    def emit(self) -> str:
        """Return the RIP-relative reference in AT&T syntax.

        Returns:
            A string of the form ``<name>(%rip)``, e.g. ``x(%rip)``.
        """
        return f"{self.name}(%rip)"
