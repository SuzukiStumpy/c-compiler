"""Instruction nodes for the assembly AST.

Defines the ``Instruction`` abstract base class and its concrete variants.
Each instruction form supported by the code generator is represented as a
subclass of ``Instruction``.

Where an instruction carries an ``assembly_type`` field, it selects between
the 32-bit (``Longword``) and 64-bit (``Quadword``) form of the mnemonic.
The field is placed last and defaults to ``Longword()`` so that
callers that pre-date long-integer support continue to work unchanged.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from compiler.syntax.asm.operands import Operand, Register, Width
from compiler.syntax.asm.operators import (
    Add,
    AddDouble,
    AddFloat,
    And,
    BinaryOperator,
    DivDouble,
    DivFloat,
    Mult,
    MultDouble,
    MultFloat,
    Or,
    Sub,
    SubDouble,
    SubFloat,
    UnaryOperator,
    Xor,
    XorDouble,
    XorFloat,
)
from compiler.syntax.asm.values import AssemblyType, Byte, Double, Float, Longword, Word
from compiler.utils.exceptions import CodeGenerationError

_MNEMONIC_COL = 6
_OPERAND_COL = 12


def _asm_line(mnemonic: str, *operands: str) -> str:
    """Format one assembly line: mnemonic at column 6, operands at column 12.

    Args:
        mnemonic: The instruction mnemonic or assembler directive (e.g. ``movl``, ``.globl``).
        *operands: Zero or more operand strings joined by ``", "``.

    Returns:
        A string with six leading spaces, the mnemonic left-justified in a
        six-character field, and the operands starting at column 12.  When the
        mnemonic exceeds six characters exactly one space separates it from the
        operands.
    """
    prefix = " " * _MNEMONIC_COL
    if not operands:
        return f"{prefix}{mnemonic}"
    ops = ", ".join(operands)
    pad = " " * max(1, _OPERAND_COL - _MNEMONIC_COL - len(mnemonic))
    return f"{prefix}{mnemonic}{pad}{ops}"


class CondCodes(StrEnum):
    """Condition codes which will be applied to the conditional set and jump commands"""

    E = "e"
    NE = "ne"
    L = "l"
    LE = "le"
    G = "g"
    GE = "ge"
    A = "a"
    AE = "ae"
    B = "b"
    BE = "be"
    P = "p"
    NP = "np"


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
    """A ``movl``/``movq`` instruction that copies a value from one operand to another.

    Attributes:
        src: The source operand whose value is copied.
        dst: The destination operand that receives the value.
        assembly_type: Selects between 32-bit (``movl``) and 64-bit (``movq``); defaults
            to ``Longword``.
    """

    src: Operand
    dst: Operand
    assembly_type: AssemblyType = Longword()

    def emit(self) -> str:
        """Return a ``movb``, ``movl``, ``movq``, or ``movsd`` instruction in AT&T syntax.

        Returns:
            A string of the form ``  movb <src>, <dst>``, ``  movl <src>, <dst>``,
            ``  movq <src>, <dst>``, or ``  movsd <src>, <dst>``.
        """
        if self.assembly_type == Double():
            return _asm_line("movsd", self.src.emit(), self.dst.emit())
        if self.assembly_type == Float():
            return _asm_line("movss", self.src.emit(), self.dst.emit())
        if self.assembly_type == Byte():
            suffix = "b"
        elif self.assembly_type == Longword():
            suffix = "l"
        else:
            suffix = "q"
        return _asm_line(f"mov{suffix}", self.src.emit(), self.dst.emit())


@dataclass(frozen=True, slots=True)
class Movsx(Instruction):
    """A ``movslq`` instruction that sign-extends a 32-bit value to 64 bits.

    Always operates longword -> quadword; no ``assembly_type`` field is needed.

    Attributes:
        src_type: The type of the source operand.
        dst_type: The type of the destination operand.
        src: The 32-bit (Longword) source operand.
        dst: The 64-bit (Quadword) destination operand (must be a register after fixup).
    """

    src_type: AssemblyType
    dst_type: AssemblyType
    src: Operand
    dst: Operand

    def emit(self) -> str:
        """Return a sign-extending move instruction in AT&T syntax.

        Selects the mnemonic based on ``src_type`` and ``dst_type``:

        * ``Byte -> Longword`` -> ``movsbl``
        * ``Byte -> Quadword`` -> ``movsbq``
        * ``Longword -> Quadword`` -> ``movslq``

        Returns:
            A string of the form ``  <mnemonic> <src>, <dst>``.
        """
        if self.src_type == Byte():
            mnemonic = "movsbl" if self.dst_type == Longword() else "movsbq"
        else:
            mnemonic = "movslq"
        return _asm_line(mnemonic, self.src.emit(), self.dst.emit())


@dataclass(frozen=True, slots=True)
class MoveZeroExtend(Instruction):
    """Move a longword (4-byte) value and convert to a quadword (8-byte) value without
    sign-extension.

    Always operates longword -> quadword; no ``assembly_type`` field is needed.

    Attributes:
        src_type: The type of the source operand
        dst_type: The type of the destination operand
        src: The 32-bit (Longword) source operand.
        dst: The 64-bit (Quadword) destination operand (must be a register after fixup).
    """

    src_type: AssemblyType
    dst_type: AssemblyType
    src: Operand
    dst: Operand

    def emit(self) -> str:
        """Return a zero-extending move instruction in AT&T syntax.

        Selects the mnemonic based on ``src_type`` and ``dst_type``:

        * ``Byte -> Longword`` -> ``movzbl``
        * ``Byte -> Quadword`` -> ``movzbq``
        * ``Word -> Longword`` -> ``movzwl`` (dst forced to 32-bit register form)
        * ``Word -> Quadword`` -> ``movzwl`` (32-bit write auto-zeroes upper 32 bits)
        * ``Longword -> Quadword`` -> ``movl`` (writing to a 32-bit register automatically
          zeroes the upper 32 bits of the 64-bit register on x86-64)

        Returns:
            A string of the form ``  <mnemonic> <src>, <dst>``.
        """
        if self.src_type == Byte():
            mnemonic = "movzbl" if self.dst_type == Longword() else "movzbq"
            return _asm_line(mnemonic, self.src.emit(), self.dst.emit())
        if self.src_type == Word():
            dst = Register(self.dst.reg, Width.DWORD) if isinstance(self.dst, Register) else self.dst
            return _asm_line("movzwl", self.src.emit(), dst.emit())
        mnemonic = "movl"
        return _asm_line(mnemonic, self.src.emit(), self.dst.emit())


@dataclass(frozen=True, slots=True)
class Ret(Instruction):
    """A ``ret`` instruction that returns from the current function.

    Carries no fields - the instruction has no operands.
    """

    def emit(self) -> str:
        """Return the frame-teardown sequence followed by ``ret``.

        Returns:
            The string ``  movq  %rbp, %rsp\\n  popq  %rbp\\n  ret``.
        """
        return "\n".join([_asm_line("movq", "%rbp", "%rsp"), _asm_line("popq", "%rbp"), _asm_line("ret")])


@dataclass(frozen=True, slots=True)
class Unary(Instruction):
    """A unary instruction that reads and modifies a single operand in place.

    Covers ``negl``/``negq`` (arithmetic negation) and ``notl``/``notq`` (bitwise complement).

    Attributes:
        operator: The specific unary operation to perform.
        operand: The operand to read and modify in place.
        assembly_type: Selects the 32-bit or 64-bit form; defaults to ``Longword``.
    """

    operator: UnaryOperator
    operand: Operand
    assembly_type: AssemblyType = Longword()

    def emit(self) -> str:
        """Return a unary instruction in AT&T syntax.

        Returns:
            A string of the form ``  <mnemonic> <operand>``,
            e.g. ``  negl %eax`` or ``  negq %rax``.
        """
        return _asm_line(self.operator.emit(self.assembly_type), self.operand.emit())


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
        return _asm_line("subq", f"${self.size}", "%rsp")


@dataclass(frozen=True, slots=True)
class DeallocateStack(Instruction):
    """An ``addq`` instruction that reclaims space from the stack.

    Attributes:
        size: Number of bytes to add to ``%rsp``.
    """

    size: int

    def emit(self) -> str:
        """Return an ``addq`` instruction in AT&T syntax.

        Returns:
            A string of the form ``  addq $<size>, %rsp``,
            e.g. ``  addq $16, %rsp``.
        """
        return _asm_line("addq", f"${self.size}", "%rsp")


@dataclass(frozen=True, slots=True)
class Binary(Instruction):
    """A binary instruction that reads two operands and writes the result to the second.

    Covers ``addl``/``addq``, ``subl``/``subq``, ``imull``/``imulq``, and the bitwise
    variants.  Division and remainder are handled by ``Cdq``/``Cqo`` and ``Idiv``.

    Attributes:
        operator: The specific binary operation to perform.
        src1: The left (first) source operand.
        src2: The right (second) source operand.
        dst: The destination operand (also the accumulator for the operation).  When
            ``src1 == dst`` the copy prefix is suppressed so the instruction emits as
            a single in-place operation (e.g. ``addl %ecx, %eax`` rather than
            ``movl %eax, %eax; addl %ecx, %eax``).
        assembly_type: Selects the 32-bit or 64-bit form; defaults to ``Longword``.
    """

    operator: BinaryOperator
    src1: Operand
    src2: Operand
    dst: Operand
    assembly_type: AssemblyType = Longword()

    def emit(self) -> str:
        """Return a binary instruction sequence in AT&T syntax.

        When ``src1 != dst`` first copies ``src1`` into ``dst``, then applies the
        operator with ``src2``.  When ``src1 == dst`` the copy is suppressed and
        only the operator instruction is emitted - the caller is responsible for
        ensuring ``dst`` already holds ``src1``'s value (typically via a preceding
        ``Mov`` that the register allocator may coalesce away).

        Integer operators use ``movl``/``movq`` and the operator's integer mnemonic.
        SSE operators use ``movsd`` and the operator's scalar-double mnemonic.

        Returns:
            A one- or two-line string: optionally ``mov? src1, dst`` followed by
            ``op src2, dst``.

        Raises:
            CodeGenerationError: If the operator is not one of the supported variants.
        """
        match self.operator:
            case Add() | Sub() | Mult() | And() | Or() | Xor():
                if self.assembly_type == Byte():
                    suffix = "b"
                elif self.assembly_type == Longword():
                    suffix = "l"
                else:
                    suffix = "q"
                lines = []
                if self.src1 != self.dst:
                    lines.append(_asm_line(f"mov{suffix}", self.src1.emit(), self.dst.emit()))
                lines.append(_asm_line(self.operator.emit(self.assembly_type), self.src2.emit(), self.dst.emit()))
                return "\n".join(lines)
            case AddDouble() | SubDouble() | MultDouble() | DivDouble() | XorDouble():
                lines = []
                if self.src1 != self.dst:
                    lines.append(_asm_line("movsd", self.src1.emit(), self.dst.emit()))
                lines.append(_asm_line(self.operator.emit(self.assembly_type), self.src2.emit(), self.dst.emit()))
                return "\n".join(lines)
            case AddFloat() | SubFloat() | MultFloat() | DivFloat() | XorFloat():
                lines = []
                if self.src1 != self.dst:
                    lines.append(_asm_line("movss", self.src1.emit(), self.dst.emit()))
                lines.append(_asm_line(self.operator.emit(self.assembly_type), self.src2.emit(), self.dst.emit()))
                return "\n".join(lines)
            case _:
                raise CodeGenerationError(f"unsupported binary operator in emit: {type(self.operator).__name__}")


@dataclass(frozen=True, slots=True)
class Cdq(Instruction):
    """A ``cdq`` (Longword) or ``cqo`` (Quadword) instruction.

    Sign-extends the accumulator into the double-width pair: ``cdq`` extends
    ``%eax`` into ``%edx:%eax``; ``cqo`` extends ``%rax`` into ``%rdx:%rax``.
    Must be emitted immediately before ``Idiv``.

    Attributes:
        assembly_type: Selects ``cdq`` (Longword) or ``cqo`` (Quadword);
            defaults to ``Longword``.
    """

    assembly_type: AssemblyType = Longword()

    def emit(self) -> str:
        """Return ``cdq`` or ``cqo``.

        Returns:
            ``"  cdq"`` for Longword or ``"  cqo"`` for Quadword.
        """
        return _asm_line("cdq" if self.assembly_type == Longword() else "cqo")


@dataclass(frozen=True, slots=True)
class Idiv(Instruction):
    """An ``idivl``/``idivq`` instruction that performs signed integer division.

    Divides the double-width accumulator pair by ``operand``.  The quotient is
    stored in ``%eax``/``%rax`` and the remainder in ``%edx``/``%rdx``.  ``operand``
    must not be an immediate value - the instruction fixup pass routes any ``Imm``
    divisor through a scratch register.

    Attributes:
        operand: The divisor; a register or memory location (never an immediate).
        assembly_type: Selects ``idivl`` (Longword) or ``idivq`` (Quadword);
            defaults to ``Longword``.
    """

    operand: Operand
    assembly_type: AssemblyType = Longword()

    def emit(self) -> str:
        """Return an ``idivb``, ``idivl``, or ``idivq`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  idivb <operand>"``, ``"  idivl <operand>"``,
            or ``"  idivq <operand>"``.
        """
        if self.assembly_type == Byte():
            suffix = "b"
        elif self.assembly_type == Longword():
            suffix = "l"
        else:
            suffix = "q"
        return _asm_line(f"idiv{suffix}", self.operand.emit())


@dataclass(frozen=True, slots=True)
class Div(Instruction):
    """A ``divl``/``divq`` instruction that performs unsigned integer division.

    Divides the zero-extended accumulator pair by ``operand``.  The quotient is
    stored in ``%eax``/``%rax`` and the remainder in ``%edx``/``%rdx``.  The caller
    must zero ``%edx``/``%rdx`` before this instruction (rather than using ``cdq``/``cqo``).
    ``operand`` must not be an immediate value - the instruction fixup pass routes any ``Imm``
    divisor through a scratch register.

    Attributes:
        operand: The divisor; a register or memory location (never an immediate).
        assembly_type: Selects ``divl`` (Longword) or ``divq`` (Quadword);
            defaults to ``Longword``.
    """

    operand: Operand
    assembly_type: AssemblyType = Longword()

    def emit(self) -> str:
        """Return a ``divb``, ``divl``, or ``divq`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  divb <operand>"``, ``"  divl <operand>"``,
            or ``"  divq <operand>"``.
        """
        if self.assembly_type == Byte():
            suffix = "b"
        elif self.assembly_type == Longword():
            suffix = "l"
        else:
            suffix = "q"
        return _asm_line(f"div{suffix}", self.operand.emit())


@dataclass(frozen=True, slots=True)
class Adc(Instruction):
    """An ``adcb``/``adcl``/``adcq`` instruction: ``dst = dst + src + CF``.

    Used as the high-half step in 128-bit addition, where the carry flag from the
    low-half ``addq`` propagates into the high-half sum.

    Attributes:
        src: The source operand (register, memory, or immediate).
        dst: The destination operand (register or memory; cannot be an immediate).
        assembly_type: Selects the 8-bit, 32-bit, or 64-bit form; defaults to ``Quadword``.
    """

    src: Operand
    dst: Operand
    assembly_type: AssemblyType = Longword()

    def emit(self) -> str:
        """Return an ``adcb``, ``adcl``, or ``adcq`` instruction in AT&T syntax.

        Args:
            None

        Returns:
            A string of the form ``"  adc<suffix> <src>, <dst>"``.
        """
        if self.assembly_type == Byte():
            suffix = "b"
        elif self.assembly_type == Longword():
            suffix = "l"
        else:
            suffix = "q"
        return _asm_line(f"adc{suffix}", self.src.emit(), self.dst.emit())


@dataclass(frozen=True, slots=True)
class Sbb(Instruction):
    """An ``sbbb``/``sbbl``/``sbbq`` instruction: ``dst = dst - src - CF``.

    Used as the high-half step in 128-bit subtraction, where the borrow flag from
    the low-half ``subq`` propagates into the high-half difference.

    Attributes:
        src: The source operand (register, memory, or immediate).
        dst: The destination operand (register or memory; cannot be an immediate).
        assembly_type: Selects the 8-bit, 32-bit, or 64-bit form; defaults to ``Quadword``.
    """

    src: Operand
    dst: Operand
    assembly_type: AssemblyType = Longword()

    def emit(self) -> str:
        """Return an ``sbbb``, ``sbbl``, or ``sbbq`` instruction in AT&T syntax.

        Args:
            None

        Returns:
            A string of the form ``"  sbb<suffix> <src>, <dst>"``.
        """
        if self.assembly_type == Byte():
            suffix = "b"
        elif self.assembly_type == Longword():
            suffix = "l"
        else:
            suffix = "q"
        return _asm_line(f"sbb{suffix}", self.src.emit(), self.dst.emit())


@dataclass(frozen=True, slots=True)
class Shl(Instruction):
    """A ``shll``/``shlq`` instruction that shifts ``operand`` left by the count in ``%cl``.

    The shift count must have been placed in ``%ecx`` by the caller.

    Attributes:
        operand: The value to be shifted; a register or memory location.
        assembly_type: Selects ``shll`` (Longword) or ``shlq`` (Quadword);
            defaults to ``Longword``.
    """

    operand: Operand
    assembly_type: AssemblyType = Longword()

    def emit(self) -> str:
        """Return a ``shlb``, ``shll``, or ``shlq`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  shlb %cl, <operand>"``, ``"  shll %cl, <operand>"``,
            or ``"  shlq %cl, <operand>"``.
        """
        if self.assembly_type == Byte():
            suffix = "b"
        elif self.assembly_type == Longword():
            suffix = "l"
        else:
            suffix = "q"
        return _asm_line(f"shl{suffix}", "%cl", self.operand.emit())


@dataclass(frozen=True, slots=True)
class Sar(Instruction):
    """A ``sarl``/``sarq`` instruction that arithmetic-shifts ``operand`` right by the count in ``%cl``.

    Arithmetic right shift sign-extends, which is correct for signed C integers.

    Attributes:
        operand: The value to be shifted; a register or memory location.
        assembly_type: Selects ``sarl`` (Longword) or ``sarq`` (Quadword);
            defaults to ``Longword``.
    """

    operand: Operand
    assembly_type: AssemblyType = Longword()

    def emit(self) -> str:
        """Return a ``sarb``, ``sarl``, or ``sarq`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  sarb %cl, <operand>"``, ``"  sarl %cl, <operand>"``,
            or ``"  sarq %cl, <operand>"``.
        """
        if self.assembly_type == Byte():
            suffix = "b"
        elif self.assembly_type == Longword():
            suffix = "l"
        else:
            suffix = "q"
        return _asm_line(f"sar{suffix}", "%cl", self.operand.emit())


@dataclass(frozen=True, slots=True)
class Shr(Instruction):
    """A ``shrl``/``shrq`` instruction that logically shifts ``operand`` right by the count in ``%cl``.

    Logical right shift fills the vacated high bits with zeroes, which is correct for unsigned C
    integers.  For signed right shift (arithmetic, sign-filling) use ``Sar`` instead.

    Attributes:
        operand: The value to be shifted; a register or memory location.
        assembly_type: Selects ``shrl`` (Longword) or ``shrq`` (Quadword);
            defaults to ``Longword``.
    """

    operand: Operand
    assembly_type: AssemblyType = Longword()

    def emit(self) -> str:
        """Return a ``shrb``, ``shrl``, or ``shrq`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  shrb %cl, <operand>"``, ``"  shrl %cl, <operand>"``,
            or ``"  shrq %cl, <operand>"``.
        """
        if self.assembly_type == Byte():
            suffix = "b"
        elif self.assembly_type == Longword():
            suffix = "l"
        else:
            suffix = "q"
        return _asm_line(f"shr{suffix}", "%cl", self.operand.emit())


@dataclass(frozen=True, slots=True)
class Cmp(Instruction):
    """A ``cmpl``/``cmpq``/``ucomisd``/``ucomiss`` instruction that compares two operands and sets the flags register.

    The operands are subtracted (``op2 - op1``) for flag-setting; neither is modified.
    When ``assembly_type`` is ``Double`` the instruction emitted is ``ucomisd``; when
    ``Float`` it is ``ucomiss``.  Both set CF/ZF/PF (compatible with the A/AE/B/BE unsigned
    condition codes).

    Attributes:
        op1: The right-hand (subtracted) operand.
        op2: The left-hand (base) operand.
        assembly_type: Selects ``cmpl`` (Longword), ``cmpq`` (Quadword), ``ucomisd`` (Double),
            or ``ucomiss`` (Float); defaults to ``Longword``.
    """

    op1: Operand
    op2: Operand
    assembly_type: AssemblyType = Longword()

    def emit(self) -> str:
        """Return a ``cmpb``, ``cmpl``, ``cmpq``, ``ucomisd``, or ``ucomiss`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  cmpb <op1>, <op2>"``, ``"  cmpl <op1>, <op2>"``,
            ``"  cmpq <op1>, <op2>"``, ``"  ucomisd <op1>, <op2>"``, or ``"  ucomiss <op1>, <op2>"``.
        """
        if self.assembly_type == Double():
            return _asm_line("ucomisd", self.op1.emit(), self.op2.emit())
        if self.assembly_type == Float():
            return _asm_line("ucomiss", self.op1.emit(), self.op2.emit())
        if self.assembly_type == Byte():
            suffix = "b"
        elif self.assembly_type == Longword():
            suffix = "l"
        else:
            suffix = "q"
        return _asm_line(f"cmp{suffix}", self.op1.emit(), self.op2.emit())


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
        return _asm_line("jmp", f".L{self.label}")


@dataclass(frozen=True, slots=True)
class JmpCC(Instruction):
    """A conditional jump instruction that transfers control based on the flags register.

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
        return _asm_line(f"j{self.cond_code.value}", f".L{self.label}")


@dataclass(frozen=True, slots=True)
class SetCC(Instruction):
    """A ``set<cc>`` instruction that writes 0 or 1 into a byte operand based on the flags register.

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
        return _asm_line(f"set{self.cond_code.value}", self.operand.emit())


@dataclass(frozen=True, slots=True)
class Label(Instruction):
    """A label declaration that marks a position in the instruction stream.

    Attributes:
        name: The label identifier, without the ``.L`` prefix or trailing colon.
    """

    name: str

    def emit(self) -> str:
        """Return a local label declaration in assembler syntax.

        Returns:
            A string of the form ``".L<name>:"``, flush with the left margin.
        """
        return f".L{self.name}:"


@dataclass(frozen=True, slots=True)
class Push(Instruction):
    """A ``pushq`` instruction that saves a value onto the stack.

    Attributes:
        operand: The value to push onto the stack.
        assembly_type: The width of the value being pushed; used by the fixup pass
            to select the right load instruction when ``operand`` is a memory location.
            Defaults to ``Longword``.
    """

    operand: Operand
    assembly_type: AssemblyType = Longword()

    def emit(self) -> str:
        """Return a ``pushq`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  pushq <operand>"``.
        """
        return _asm_line("pushq", self.operand.emit())


@dataclass(frozen=True, slots=True)
class Pop(Instruction):
    """A ``popq`` instruction that pops the topmost value from the stack into a register

    Attributes:
        target: The register into which the value will be popped
    """

    target: Register

    def emit(self) -> str:
        """Return a ``popq`` instruction which pops the topmost value from the stack

        Returns:
            A string of the form ``"  popq <register>"``.
        """
        return _asm_line("popq", self.target.emit())


@dataclass(frozen=True, slots=True)
class Call(Instruction):
    """A ``call`` instruction that transfers control to a named function.

    Attributes:
        name: The symbol name of the function to call.
    """

    name: str

    def emit(self) -> str:
        """Return a ``call`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  call <name>"``.
        """
        return _asm_line("call", self.name)


@dataclass(frozen=True, slots=True)
class IndirectCall(Instruction):
    """A ``call`` instruction that transfers control through a register or memory operand.

    Used to lower calls through function pointer variables.  The target operand holds
    the run-time address of the callee, so the emitted form is ``call *<target>``.

    Attributes:
        target: The operand containing the function pointer value (typically a register).
    """

    target: Operand

    def emit(self) -> str:
        """Return an indirect ``call`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  call *<target>"``, e.g. ``"  call *%r10"``.
        """
        return _asm_line("call", f"*{self.target.emit()}")


@dataclass(frozen=True, slots=True)
class Cvttsd2si(Instruction):
    """A ``cvttsd2sil``/``cvttsd2siq`` instruction that truncates a scalar double to a signed integer.

    The source must be an XMM register or memory location; the destination is a general-purpose
    register.  ``assembly_type`` selects the *destination* integer width.

    Attributes:
        src: The scalar double source (XMM register or memory).
        dst: The signed-integer destination (general-purpose register).
        assembly_type: Selects 32-bit (``Longword``) or 64-bit (``Quadword``) integer output;
            defaults to ``Longword``.
    """

    src: Operand
    dst: Operand
    assembly_type: AssemblyType = Longword()

    def emit(self) -> str:
        """Return a ``cvttsd2sil`` or ``cvttsd2siq`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  cvttsd2sil <src>, <dst>"`` or ``"  cvttsd2siq <src>, <dst>"``.
        """
        suffix = "l" if self.assembly_type == Longword() else "q"
        return _asm_line(f"cvttsd2si{suffix}", self.src.emit(), self.dst.emit())


@dataclass(frozen=True, slots=True)
class Cvtsi2sd(Instruction):
    """A ``cvtsi2sdl``/``cvtsi2sdq`` instruction that converts a signed integer to a scalar double.

    The source is a general-purpose register or memory location; the destination must be an XMM
    register.  ``assembly_type`` selects the *source* integer width.

    Attributes:
        src: The signed-integer source (general-purpose register or memory).
        dst: The scalar double destination (XMM register).
        assembly_type: Selects 32-bit (``Longword``) or 64-bit (``Quadword``) integer input;
            defaults to ``Longword``.
    """

    src: Operand
    dst: Operand
    assembly_type: AssemblyType = Longword()

    def emit(self) -> str:
        """Return a ``cvtsi2sdl`` or ``cvtsi2sdq`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  cvtsi2sdl <src>, <dst>"`` or ``"  cvtsi2sdq <src>, <dst>"``.
        """
        suffix = "l" if self.assembly_type == Longword() else "q"
        return _asm_line(f"cvtsi2sd{suffix}", self.src.emit(), self.dst.emit())


@dataclass(frozen=True, slots=True)
class Cvttss2si(Instruction):
    """A ``cvttss2sil``/``cvttss2siq`` instruction: truncate scalar single-precision float to signed integer.

    The source must be an XMM register or memory location; the destination is a general-purpose
    register.  ``assembly_type`` selects the *destination* integer width.

    Attributes:
        src: The scalar float source (XMM register or memory).
        dst: The signed-integer destination (general-purpose register).
        assembly_type: Selects 32-bit (``Longword``) or 64-bit (``Quadword``) integer output;
            defaults to ``Longword``.
    """

    src: Operand
    dst: Operand
    assembly_type: AssemblyType = Longword()

    def emit(self) -> str:
        """Return a ``cvttss2sil`` or ``cvttss2siq`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  cvttss2sil <src>, <dst>"`` or ``"  cvttss2siq <src>, <dst>"``.
        """
        suffix = "l" if self.assembly_type == Longword() else "q"
        return _asm_line(f"cvttss2si{suffix}", self.src.emit(), self.dst.emit())


@dataclass(frozen=True, slots=True)
class Cvtsi2ss(Instruction):
    """A ``cvtsi2ssl``/``cvtsi2ssq`` instruction: convert signed integer to scalar single-precision float.

    The source is a general-purpose register or memory location; the destination must be an XMM
    register.  ``assembly_type`` selects the *source* integer width.

    Attributes:
        src: The signed-integer source (general-purpose register or memory).
        dst: The scalar float destination (XMM register).
        assembly_type: Selects 32-bit (``Longword``) or 64-bit (``Quadword``) integer input;
            defaults to ``Longword``.
    """

    src: Operand
    dst: Operand
    assembly_type: AssemblyType = Longword()

    def emit(self) -> str:
        """Return a ``cvtsi2ssl`` or ``cvtsi2ssq`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  cvtsi2ssl <src>, <dst>"`` or ``"  cvtsi2ssq <src>, <dst>"``.
        """
        suffix = "l" if self.assembly_type == Longword() else "q"
        return _asm_line(f"cvtsi2ss{suffix}", self.src.emit(), self.dst.emit())


@dataclass(frozen=True, slots=True)
class Cvtss2sd(Instruction):
    """A ``cvtss2sd`` instruction: widen scalar single-precision float to double-precision.

    Attributes:
        src: The scalar float source (XMM register or memory).
        dst: The scalar double destination (XMM register).
    """

    src: Operand
    dst: Operand

    def emit(self) -> str:
        """Return a ``cvtss2sd`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  cvtss2sd <src>, <dst>"``.
        """
        return _asm_line("cvtss2sd", self.src.emit(), self.dst.emit())


@dataclass(frozen=True, slots=True)
class Cvtsd2ss(Instruction):
    """A ``cvtsd2ss`` instruction: narrow scalar double-precision float to single-precision.

    Attributes:
        src: The scalar double source (XMM register or memory).
        dst: The scalar float destination (XMM register).
    """

    src: Operand
    dst: Operand

    def emit(self) -> str:
        """Return a ``cvtsd2ss`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  cvtsd2ss <src>, <dst>"``.
        """
        return _asm_line("cvtsd2ss", self.src.emit(), self.dst.emit())


@dataclass(frozen=True, slots=True)
class Lea(Instruction):
    """A ``leaq`` instruction that loads the effective address of a memory operand into a register.

    Used to lower ``tac.GetAddress``: given a ``Memory`` operand that describes where a variable
    lives in the current frame, ``leaq`` computes its address and writes the pointer value into
    ``dst`` without performing a memory access.

    Attributes:
        src: The memory operand whose address is to be computed (e.g. ``-4(%rbp)``).
        dst: The destination register that receives the computed address.
    """

    src: Operand
    dst: Operand

    def emit(self) -> str:
        """Return a ``leaq`` instruction in AT&T syntax.

        Returns:
            A string of the form ``"  leaq <src>, <dst>"``,
            e.g. ``"  leaq -4(%rbp), %rax"``.
        """
        return _asm_line("leaq", self.src.emit(), self.dst.emit())


# ---------------------------------------------------------------------------
# x87 FPU instructions for ``long double`` (80-bit extended precision)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fldt(Instruction):
    """Load 80-bit extended-precision value from memory onto the x87 FPU stack (ST0).

    Attributes:
        src: Memory operand containing the 10-byte (80-bit) value to push.
    """

    src: Operand

    def emit(self) -> str:
        """Return a ``fldt`` instruction in AT&T syntax."""
        return _asm_line("fldt", self.src.emit())


@dataclass(frozen=True, slots=True)
class Fstpt(Instruction):
    """Store ST0 as 80-bit extended-precision to memory and pop the x87 stack.

    Attributes:
        dst: Memory operand receiving the 10-byte (80-bit) value.
    """

    dst: Operand

    def emit(self) -> str:
        """Return a ``fstpt`` instruction in AT&T syntax."""
        return _asm_line("fstpt", self.dst.emit())


@dataclass(frozen=True, slots=True)
class Faddp(Instruction):
    """Add ST(0) and ST(1), store result in ST(1), and pop ST(0).

    In AT&T syntax: ``faddp``.  Implements ``long double`` addition.
    """

    def emit(self) -> str:
        """Return a ``faddp`` instruction."""
        return _asm_line("faddp")


@dataclass(frozen=True, slots=True)
class Fsubp(Instruction):
    """Subtract ST(0) from ST(1) (i.e. ST(1) -= ST(0)), store in ST(1), pop ST(0).

    Implements ``long double`` subtraction when operands are loaded as: push left, push right.
    AT&T: ``fsubrp`` (reversed operand convention in AT&T syntax means ST(1) = ST(1) - ST(0)).
    """

    def emit(self) -> str:
        """Return a ``fsubrp`` instruction (AT&T convention: ST(1) = ST(1) - ST(0))."""
        return _asm_line("fsubrp")


@dataclass(frozen=True, slots=True)
class Fmulp(Instruction):
    """Multiply ST(0) and ST(1), store result in ST(1), and pop ST(0).

    Implements ``long double`` multiplication.  AT&T: ``fmulp``.
    """

    def emit(self) -> str:
        """Return a ``fmulp`` instruction."""
        return _asm_line("fmulp")


@dataclass(frozen=True, slots=True)
class Fdivp(Instruction):
    """Divide ST(1) by ST(0) (i.e. ST(1) /= ST(0)), store in ST(1), pop ST(0).

    Implements ``long double`` division.  In AT&T syntax ``fdivrp`` means ST(1) = ST(1)/ST(0).
    """

    def emit(self) -> str:
        """Return a ``fdivrp`` instruction (AT&T convention: ST(1) = ST(1) / ST(0))."""
        return _asm_line("fdivrp")


@dataclass(frozen=True, slots=True)
class Fchs(Instruction):
    """Change the sign of ST(0) (negate).  AT&T: ``fchs``."""

    def emit(self) -> str:
        """Return a ``fchs`` instruction."""
        return _asm_line("fchs")


@dataclass(frozen=True, slots=True)
class Fucomip(Instruction):
    """Unordered compare ST(0) with ST(1), set EFLAGS, and pop ST(0).

    After this instruction: CF=1 → ST(0) < ST(1); ZF=1 → equal; CF=0,ZF=0 → ST(0) > ST(1).
    Emits ``fucomip %st, %st(1)`` in AT&T syntax.
    """

    def emit(self) -> str:
        """Return a ``fucomip %st, %st(1)`` instruction in AT&T syntax."""
        return _asm_line("fucomip", "%st", "%st(1)")


@dataclass(frozen=True, slots=True)
class Fstp0(Instruction):
    """Pop and discard ST(0) from the x87 stack.  Emits ``fstp %st(0)``."""

    def emit(self) -> str:
        """Return a ``fstp %st(0)`` instruction to pop the x87 stack top."""
        return _asm_line("fstp", "%st(0)")


@dataclass(frozen=True, slots=True)
class Fildl(Instruction):
    """Load a 32-bit signed integer from memory onto the x87 stack as extended precision.

    Attributes:
        src: Memory operand containing the 4-byte signed integer.
    """

    src: Operand

    def emit(self) -> str:
        """Return a ``fildl`` instruction in AT&T syntax."""
        return _asm_line("fildl", self.src.emit())


@dataclass(frozen=True, slots=True)
class Fildq(Instruction):
    """Load a 64-bit signed integer from memory onto the x87 stack as extended precision.

    Attributes:
        src: Memory operand containing the 8-byte signed integer.
    """

    src: Operand

    def emit(self) -> str:
        """Return a ``fildq`` instruction in AT&T syntax."""
        return _asm_line("fildq", self.src.emit())


@dataclass(frozen=True, slots=True)
class Fisttpl(Instruction):
    """Store ST(0) as a 32-bit signed integer (truncate toward zero) to memory, pop.

    Uses SSE2 truncation semantics (matches C cast behaviour).

    Attributes:
        dst: Memory operand receiving the 4-byte integer result.
    """

    dst: Operand

    def emit(self) -> str:
        """Return a ``fisttpl`` instruction in AT&T syntax."""
        return _asm_line("fisttpl", self.dst.emit())


@dataclass(frozen=True, slots=True)
class Fisttpq(Instruction):
    """Store ST(0) as a 64-bit signed integer (truncate toward zero) to memory, pop.

    Uses SSE2 truncation semantics (matches C cast behaviour).

    Attributes:
        dst: Memory operand receiving the 8-byte integer result.
    """

    dst: Operand

    def emit(self) -> str:
        """Return a ``fisttpq`` instruction in AT&T syntax."""
        return _asm_line("fisttpq", self.dst.emit())


@dataclass(frozen=True, slots=True)
class Fldl(Instruction):
    """Load a 64-bit double-precision float from memory onto the x87 stack (extends to 80-bit).

    Attributes:
        src: Memory operand containing the 8-byte double value.
    """

    src: Operand

    def emit(self) -> str:
        """Return a ``fldl`` instruction in AT&T syntax."""
        return _asm_line("fldl", self.src.emit())


@dataclass(frozen=True, slots=True)
class Fstpl(Instruction):
    """Store ST(0) as a 64-bit double-precision float to memory, pop (narrows from 80-bit).

    Attributes:
        dst: Memory operand receiving the 8-byte double result.
    """

    dst: Operand

    def emit(self) -> str:
        """Return a ``fstpl`` instruction in AT&T syntax."""
        return _asm_line("fstpl", self.dst.emit())


@dataclass(frozen=True, slots=True)
class Flds(Instruction):
    """Load a 32-bit single-precision float from memory onto the x87 stack (extends to 80-bit).

    Attributes:
        src: Memory operand containing the 4-byte float value.
    """

    src: Operand

    def emit(self) -> str:
        """Return a ``flds`` instruction in AT&T syntax."""
        return _asm_line("flds", self.src.emit())


@dataclass(frozen=True, slots=True)
class Fstps(Instruction):
    """Store ST(0) as a 32-bit single-precision float to memory, pop (narrows from 80-bit).

    Attributes:
        dst: Memory operand receiving the 4-byte float result.
    """

    dst: Operand

    def emit(self) -> str:
        """Return a ``fstps`` instruction in AT&T syntax."""
        return _asm_line("fstps", self.dst.emit())


@dataclass(frozen=True, slots=True)
class Ud2(Instruction):
    """The ``ud2`` instruction — guaranteed to raise ``SIGILL`` when executed.

    Emitted for ``__builtin_unreachable()`` to trap if control reaches a
    path the programmer declared unreachable.
    """

    def emit(self) -> str:
        """Return a ``ud2`` instruction."""
        return "  ud2"
