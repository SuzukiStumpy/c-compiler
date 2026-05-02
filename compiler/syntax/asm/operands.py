"""Operand nodes for the assembly AST.

Defines the ``Operand`` abstract base class and its concrete variants. Each
operand form supported by the code generator is represented as a subclass of
``Operand``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum


class Registers(StrEnum):
    """Names of the hardware registers used during code generation."""

    AX = "eax"
    DX = "edx"
    R10 = "r10d"
    R11 = "r11d"


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
    """A hardware register operand.

    Wraps a ``Registers`` enum value so that the specific register (e.g.
    ``%ax``, ``%r10``) is explicit in the AST rather than hard-coded.

    Attributes:
        reg: The register to use.
    """

    reg: Registers

    def emit(self) -> str:
        """Return the register name in AT&T syntax.

        Returns:
            A string of the form ``%<name>``, e.g. ``%ax`` or ``%r10``.
        """
        return f"%{self.reg.value}"


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
