"""Unary operator nodes for the assembly AST.

Defines the ``UnaryOperator`` abstract base class and its concrete variants.
Each operator form corresponds to a pair of x86 mnemonics (``negl``/``negq``,
``notl``/``notq``) and is represented as a concrete frozen dataclass subclassing
``UnaryOperator``.

Consumers should use structural pattern matching (``match``/``case``) or call
``emit()`` rather than inspecting the concrete type directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from compiler.syntax.asm.values import AssemblyType, Longword


class UnaryOperator(ABC):
    """Abstract base for all assembly unary operator nodes.

    Each operator is represented as a concrete subclass carrying no data fields
    — its identity is its type alone.  Consumers should use structural pattern
    matching (``match``/``case``) to dispatch on the concrete variant rather than
    inspecting the type directly.
    """

    @abstractmethod
    def emit(self, assembly_type: AssemblyType) -> str:
        """Return the assembly mnemonic for this operator at the given operand size.

        Args:
            assembly_type: Selects between the 32-bit (``l``) and 64-bit (``q``) form.

        Returns:
            The instruction mnemonic string (e.g. ``"negl"`` or ``"negq"``).
        """


@dataclass(frozen=True, slots=True)
class Neg(UnaryOperator):
    """The ``negl``/``negq`` instruction: arithmetic (two's-complement) negation.

    Negates its single operand in place.
    """

    def emit(self, assembly_type: AssemblyType) -> str:
        """Return the ``negl`` or ``negq`` mnemonic.

        Args:
            assembly_type: Selects the 32-bit or 64-bit form.

        Returns:
            ``"negl"`` for Longword, ``"negq"`` for Quadword.
        """
        return "negl" if assembly_type == Longword() else "negq"


@dataclass(frozen=True, slots=True)
class Not(UnaryOperator):
    """The ``notl``/``notq`` instruction: bitwise complement.

    Inverts every bit of its single operand in place.
    """

    def emit(self, assembly_type: AssemblyType) -> str:
        """Return the ``notl`` or ``notq`` mnemonic.

        Args:
            assembly_type: Selects the 32-bit or 64-bit form.

        Returns:
            ``"notl"`` for Longword, ``"notq"`` for Quadword.
        """
        return "notl" if assembly_type == Longword() else "notq"


class BinaryOperator(ABC):
    """Abstract base for all assembly binary operator nodes.

    Each operator is represented as a concrete subclass carrying no data fields
    — its identity is its type alone.  Consumers should use structural pattern
    matching (``match``/``case``) to dispatch on the concrete variant rather than
    inspecting the type directly.
    """

    @abstractmethod
    def emit(self, assembly_type: AssemblyType) -> str:
        """Return the assembly mnemonic for this operator at the given operand size.

        Args:
            assembly_type: Selects between the 32-bit (``l``) and 64-bit (``q``) form.

        Returns:
            The instruction mnemonic string (e.g. ``"addl"`` or ``"addq"``).
        """


@dataclass(frozen=True, slots=True)
class Add(BinaryOperator):
    """The ``addl``/``addq`` instruction: addition."""

    def emit(self, assembly_type: AssemblyType) -> str:
        """Return the ``addl`` or ``addq`` mnemonic.

        Args:
            assembly_type: Selects the 32-bit or 64-bit form.

        Returns:
            ``"addl"`` for Longword, ``"addq"`` for Quadword.
        """
        return "addl" if assembly_type == Longword() else "addq"


@dataclass(frozen=True, slots=True)
class Sub(BinaryOperator):
    """The ``subl``/``subq`` instruction: subtraction."""

    def emit(self, assembly_type: AssemblyType) -> str:
        """Return the ``subl`` or ``subq`` mnemonic.

        Args:
            assembly_type: Selects the 32-bit or 64-bit form.

        Returns:
            ``"subl"`` for Longword, ``"subq"`` for Quadword.
        """
        return "subl" if assembly_type == Longword() else "subq"


@dataclass(frozen=True, slots=True)
class Mult(BinaryOperator):
    """The ``imull``/``imulq`` instruction: signed multiplication."""

    def emit(self, assembly_type: AssemblyType) -> str:
        """Return the ``imull`` or ``imulq`` mnemonic.

        Args:
            assembly_type: Selects the 32-bit or 64-bit form.

        Returns:
            ``"imull"`` for Longword, ``"imulq"`` for Quadword.
        """
        return "imull" if assembly_type == Longword() else "imulq"


@dataclass(frozen=True, slots=True)
class And(BinaryOperator):
    """The ``andl``/``andq`` instruction: bitwise AND."""

    def emit(self, assembly_type: AssemblyType) -> str:
        """Return the ``andl`` or ``andq`` mnemonic.

        Args:
            assembly_type: Selects the 32-bit or 64-bit form.

        Returns:
            ``"andl"`` for Longword, ``"andq"`` for Quadword.
        """
        return "andl" if assembly_type == Longword() else "andq"


@dataclass(frozen=True, slots=True)
class Or(BinaryOperator):
    """The ``orl``/``orq`` instruction: bitwise OR."""

    def emit(self, assembly_type: AssemblyType) -> str:
        """Return the ``orl`` or ``orq`` mnemonic.

        Args:
            assembly_type: Selects the 32-bit or 64-bit form.

        Returns:
            ``"orl"`` for Longword, ``"orq"`` for Quadword.
        """
        return "orl" if assembly_type == Longword() else "orq"


@dataclass(frozen=True, slots=True)
class Xor(BinaryOperator):
    """The ``xorl``/``xorq`` instruction: bitwise exclusive OR."""

    def emit(self, assembly_type: AssemblyType) -> str:
        """Return the ``xorl`` or ``xorq`` mnemonic.

        Args:
            assembly_type: Selects the 32-bit or 64-bit form.

        Returns:
            ``"xorl"`` for Longword, ``"xorq"`` for Quadword.
        """
        return "xorl" if assembly_type == Longword() else "xorq"


@dataclass(frozen=True, slots=True)
class AddDouble(BinaryOperator):
    """The ``addsd`` instruction: scalar double-precision floating-point addition."""

    def emit(self, assembly_type: AssemblyType) -> str:
        """Return the ``addsd`` mnemonic.

        Args:
            assembly_type: Ignored for SSE operators; present to satisfy the abstract interface.

        Returns:
            ``"addsd"`` unconditionally.
        """
        return "addsd"


@dataclass(frozen=True, slots=True)
class SubDouble(BinaryOperator):
    """The ``subsd`` instruction: scalar double-precision floating-point subtraction."""

    def emit(self, assembly_type: AssemblyType) -> str:
        """Return the ``subsd`` mnemonic.

        Args:
            assembly_type: Ignored for SSE operators; present to satisfy the abstract interface.

        Returns:
            ``"subsd"`` unconditionally.
        """
        return "subsd"


@dataclass(frozen=True, slots=True)
class MultDouble(BinaryOperator):
    """The ``mulsd`` instruction: scalar double-precision floating-point multiplication."""

    def emit(self, assembly_type: AssemblyType) -> str:
        """Return the ``mulsd`` mnemonic.

        Args:
            assembly_type: Ignored for SSE operators; present to satisfy the abstract interface.

        Returns:
            ``"mulsd"`` unconditionally.
        """
        return "mulsd"


@dataclass(frozen=True, slots=True)
class DivDouble(BinaryOperator):
    """The ``divsd`` instruction: scalar double-precision floating-point division."""

    def emit(self, assembly_type: AssemblyType) -> str:
        """Return the ``divsd`` mnemonic.

        Args:
            assembly_type: Ignored for SSE operators; present to satisfy the abstract interface.

        Returns:
            ``"divsd"`` unconditionally.
        """
        return "divsd"


@dataclass(frozen=True, slots=True)
class XorDouble(BinaryOperator):
    """The ``xorpd`` instruction: bitwise XOR of packed double-precision floating-point values.

    Used to flip the sign bit of a double by XOR-ing with a constant whose sign bit is set.
    The memory operand must be 16-byte aligned for strict SSE2 compliance, though modern
    x86-64 processors tolerate 8-byte alignment.
    """

    def emit(self, assembly_type: AssemblyType) -> str:
        """Return the ``xorpd`` mnemonic.

        Args:
            assembly_type: Ignored for SSE operators; present to satisfy the abstract interface.

        Returns:
            ``"xorpd"`` unconditionally.
        """
        return "xorpd"
