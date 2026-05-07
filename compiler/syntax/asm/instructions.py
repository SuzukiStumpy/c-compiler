"""Instruction nodes for the assembly AST.

Defines the ``Instruction`` abstract base class and its concrete variants.
Each instruction form supported by the code generator is represented as a
subclass of ``Instruction``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from compiler.syntax.asm.operands import Operand, Register
from compiler.syntax.asm.operators import Add, And, BinaryOperator, Mult, Or, Sub, UnaryOperator, Xor
from compiler.utils.exceptions import CodeGenerationError


class CondCodes(StrEnum):
    """Condition codes which will be applied to the conditional set and jump commands"""

    E = "e"
    NE = "ne"
    L = "l"
    LE = "le"
    G = "g"
    GE = "ge"


class Instruction(ABC):
    """Abstract base for all assembly instruction nodes.

    Each instruction form is represented as a concrete subclass. Consumers
    should use structural pattern matching (``match``/``case``) to dispatch on
    the concrete variant rather than inspecting fields directly.
    """

    @abstractmethod
    def emit(self) -> str:
        """Return the assembly text for this instruction, indented by two spaces.

        Returns:
            A string beginning with two spaces followed by the mnemonic and
            any operands, ready to be written directly to an assembly file.
        """


@dataclass(frozen=True, slots=True)
class Mov(Instruction):
    """A ``movl`` instruction that copies a value from one operand to another.

    Attributes:
        src: The source operand whose value is copied.
        dst: The destination operand that receives the value.
    """

    src: Operand
    dst: Operand

    def emit(self) -> str:
        """Return a ``movl`` instruction in AT&T syntax.

        Returns:
            A string of the form ``  movl <src>, <dst>``.
        """
        return f"  movl {self.src.emit()}, {self.dst.emit()}"


@dataclass(frozen=True, slots=True)
class Ret(Instruction):
    """A ``ret`` instruction that returns from the current function.

    Carries no fields — the instruction has no operands.
    """

    def emit(self) -> str:
        """Return the ``ret`` mnemonic.

        Returns:
            The string ``
                movq  %rbp, %rsp
                popq  %rbp, %rsp
                ret
            ``.
        """
        return "  movq  %rbp, %rsp\n  popq  %rbp\n  ret"


@dataclass(frozen=True, slots=True)
class Unary(Instruction):
    """A unary instruction that reads and modifies a single operand in place.

    Covers ``negl`` (arithmetic negation) and ``notl`` (bitwise complement).

    Attributes:
        operator: The specific unary operation to perform.
        operand: The operand to read and modify in place.
    """

    operator: UnaryOperator
    operand: Operand

    def emit(self) -> str:
        """Return a unary instruction in AT&T syntax.

        Returns:
            A string of the form ``  <mnemonic> <operand>``,
            e.g. ``  negl %eax``.
        """
        return f"  {self.operator.emit()} {self.operand.emit()}"


@dataclass(frozen=True, slots=True)
class AllocateStack(Instruction):
    """A ``subq`` instruction that reserves space on the stack for local variables.

    This instruction is prepended to function bodies by the pseudo-register
    replacement pass whenever temporaries must be spilled to the stack.  The
    ``size`` is always a positive multiple of 16 to maintain ABI alignment.

    Attributes:
        size: Number of bytes to subtract from ``%rsp``.
    """

    size: int

    def emit(self) -> str:
        """Return a ``subq`` instruction in AT&T syntax.

        Returns:
            A string of the form ``  subq $<size>, %rsp``,
            e.g. ``  subq $16, %rsp``.
        """
        return f"  subq ${self.size}, %rsp"


@dataclass(frozen=True, slots=True)
class Binary(Instruction):
    """A binary instruction that reads two operands and writes the result to the second.

    Covers ``addl``, ``subl``, and ``imull``.  Division and remainder are
    handled by ``Cdq`` and ``Idiv`` instead, since they require a fixed
    register protocol that does not fit the two-operand model.

    Attributes:
        operator: The specific binary operation to perform.
        src1: The left (first) source operand.
        src2: The right (second) source operand.
        dst: The destination register (also the accumulator for the operation).
    """

    operator: BinaryOperator
    src1: Operand
    src2: Operand
    dst: Register

    def emit(self) -> str:
        """Return a binary instruction sequence in AT&T syntax.

        Moves ``src1`` into ``dst``, then applies the operator with ``src2``.

        Returns:
            A two-line string: ``movl src1, dst`` followed by ``op src2, dst``.

        Raises:
            CodeGenerationError: If the operator is not one of the supported
                variants (``Add``, ``Sub``, ``Mult``).
        """
        match self.operator:
            case Add() | Sub() | Mult() | And() | Or() | Xor():
                return "\n".join(
                    [
                        f"  movl {self.src1.emit()}, {self.dst.emit()}",
                        f"  {self.operator.emit()} {self.src2.emit()}, {self.dst.emit()}",
                    ]
                )
            case _:
                raise CodeGenerationError(f"unsupported binary operator in emit: {type(self.operator).__name__}")


@dataclass(frozen=True, slots=True)
class Cdq(Instruction):
    """A ``cdq`` instruction that sign-extends ``%eax`` into ``%edx:%eax``.

    Must be emitted immediately before ``Idiv`` to prepare the full 64-bit
    dividend.  Carries no fields.
    """

    def emit(self) -> str:
        """Return the ``cdq`` mnemonic.

        Returns:
            The string ``"  cdq"``.
        """
        return "  cdq"


@dataclass(frozen=True, slots=True)
class Idiv(Instruction):
    """An ``idivl`` instruction that performs signed 32-bit integer division.

    Divides the ``%edx:%eax`` dividend by ``operand``.  The quotient is stored
    in ``%eax`` and the remainder in ``%edx``.  ``operand`` must not be an
    immediate value — the instruction fixup pass is responsible for routing
    any ``Imm`` divisor through ``%r11d`` before emission.

    Attributes:
        operand: The divisor; a register or memory location (never an immediate).
    """

    operand: Operand

    def emit(self) -> str:
        """Return an ``idivl`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  idivl <operand>"``.
        """
        return f"  idivl {self.operand.emit()}"


@dataclass(frozen=True, slots=True)
class Shl(Instruction):
    """A ``shll`` instruction that shifts ``operand`` left by the count in ``%cl``.

    The shift count must have been placed in ``%ecx`` by the caller before this
    instruction is emitted.  ``operand`` is updated in place.

    Attributes:
        operand: The value to be shifted; a register or memory location.
    """

    operand: Operand

    def emit(self) -> str:
        """Return a ``shll`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  shll %cl, <operand>"``.
        """
        return f"  shll %cl, {self.operand.emit()}"


@dataclass(frozen=True, slots=True)
class Sar(Instruction):
    """A ``sarl`` instruction that arithmetic-shifts ``operand`` right by the count in ``%cl``.

    Arithmetic right shift sign-extends the result, which is the correct
    behaviour for signed ``int`` values in C.  The shift count must have been
    placed in ``%ecx`` by the caller before this instruction is emitted.
    ``operand`` is updated in place.

    Attributes:
        operand: The value to be shifted; a register or memory location.
    """

    operand: Operand

    def emit(self) -> str:
        """Return a ``sarl`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  sarl %cl, <operand>"``.
        """
        return f"  sarl %cl, {self.operand.emit()}"


@dataclass(frozen=True, slots=True)
class Cmp(Instruction):
    """A ``cmpl`` instruction that compares two operands and sets the flags register.

    The operands are subtracted (``op2 - op1``) for the purposes of flag
    setting; neither is modified.  The result is consumed by a subsequent
    ``JmpCC`` or ``SetCC`` instruction.

    Attributes:
        op1: The right-hand (subtracted) operand.
        op2: The left-hand (base) operand.
    """

    op1: Operand
    op2: Operand

    def emit(self) -> str:
        """Return a ``cmpl`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  cmpl <op1>, <op2>"``.
        """
        return f"  cmpl {self.op1.emit()}, {self.op2.emit()}"


@dataclass(frozen=True, slots=True)
class Jmp(Instruction):
    """An unconditional ``jmp`` instruction that transfers control to a label.

    Attributes:
        label: The name of the target label (without the ``.L`` prefix).
    """

    label: str

    def emit(self) -> str:
        """Return a ``jmp`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  jmp .L<label>"``.
        """
        return f"  jmp .L{self.label}"


@dataclass(frozen=True, slots=True)
class JmpCC(Instruction):
    """A conditional jump instruction that transfers control based on the flags register.

    The condition is evaluated against the flags set by the most recent ``Cmp``
    instruction.  Control is transferred to ``label`` when the condition holds.

    Attributes:
        cond_code: The condition that must be satisfied for the jump to occur.
        label: The name of the target label (without the ``.L`` prefix).
    """

    cond_code: CondCodes
    label: str

    def emit(self) -> str:
        """Return a conditional jump instruction in AT&T syntax.

        Returns:
            A string of the form ``"  j<cond_code> .L<label>"``.
        """
        return f"  j{self.cond_code.value} .L{self.label}"


@dataclass(frozen=True, slots=True)
class SetCC(Instruction):
    """A ``set<cc>`` instruction that writes 0 or 1 into a byte operand based on the flags register.

    The condition is evaluated against the flags set by the most recent ``Cmp``
    instruction.  The destination operand receives 1 if the condition holds and
    0 otherwise.  The operand must be an 8-bit register or memory location.

    Attributes:
        cond_code: The condition that determines the value written.
        operand: The byte-sized destination operand.
    """

    cond_code: CondCodes
    operand: Operand

    def emit(self) -> str:
        """Return a conditional set instruction in AT&T syntax.

        Returns:
            A string of the form ``"  set<cond_code> <operand>"``.
        """
        return f"  set{self.cond_code.value} {self.operand.emit()}"


@dataclass(frozen=True, slots=True)
class Label(Instruction):
    """A label declaration that marks a position in the instruction stream.

    Emitted as a ``.L``-prefixed local label followed by a colon, making it
    visible to the assembler but not exported to the linker symbol table.

    Attributes:
        name: The label identifier, without the ``.L`` prefix or trailing colon.
    """

    name: str

    def emit(self) -> str:
        """Return a local label declaration in assembler syntax.

        Returns:
            A string of the form ``".L<name>:"``, flush with the left margin
            (no leading spaces) so the assembler treats it as a label rather
            than an instruction.
        """
        return f".L{self.name}:"
