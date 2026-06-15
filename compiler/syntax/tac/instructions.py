"""Instruction nodes for the tac AST.

Defines the ``Instruction`` abstract base class and its concrete variants.
Each instruction form supported by the code generator is represented as a
subclass of ``Instruction``.
"""

from abc import ABC
from dataclasses import dataclass, field

from compiler.syntax.tac.operators import BinaryOperator, UnaryOperator
from compiler.syntax.tac.values import Val


class Instruction(ABC):
    """Abstract base for all tac instruction nodes.

    Each instruction form is represented as a concrete subclass. Consumers
    should use structural pattern matching (``match``/``case``) to dispatch on
    the concrete variant rather than inspecting fields directly.
    """


@dataclass(frozen=True, slots=True)
class Return(Instruction):
    """Represents a return from a function.

    Attributes:
        val: The value being returned, or ``None`` for a void return.
    """

    val: Val | None = field(default=None)


@dataclass(frozen=True, slots=True)
class Unary(Instruction):
    """Represents a unary expression within the tac

    Attributes:
        operator: The type of the operation
        src:  The source value for the expression
        dst:  The destination for the result of the expression
    """

    operator: UnaryOperator
    src: Val
    dst: Val


@dataclass(frozen=True, slots=True)
class Binary(Instruction):
    """Represents a binary expression within the tac

    Attributes:
        operator: The type of the operation
        src1: The first source value for the expression
        src2: The second source value for the expression
        dst:  The destination for the result of the expression
    """

    operator: BinaryOperator
    src1: Val
    src2: Val
    dst: Val


@dataclass(frozen=True, slots=True)
class SignExtend(Instruction):
    """Sign-extend a value from C ``int`` to C ``long``.

    Attributes:
        src: The ``int``-typed source value.
        dst: The ``long``-typed destination value.
    """

    src: Val
    dst: Val


@dataclass(frozen=True, slots=True)
class Truncate(Instruction):
    """Truncate a value from C ``long`` to C ``int``.

    Attributes:
        src: The ``long``-typed source value.
        dst: The ``int``-typed destination value.
    """

    src: Val
    dst: Val


@dataclass(frozen=True, slots=True)
class ZeroExtend(Instruction):
    """Widen a value from C ``int`` to C ``long``.

    Attributes:
        src: The ``long``-typed source value.
        dst: The ``int``-typed destination value.
    """

    src: Val
    dst: Val


@dataclass(frozen=True, slots=True)
class Copy(Instruction):
    """Represents a copy operation with the tac

    Attributes:
        src: The source location to copy from
        dst: The destination location to copy to
    """

    src: Val
    dst: Val


@dataclass(frozen=True, slots=True)
class Jump(Instruction):
    """Represents an unconditional jump within the tac

    Attributes:
        target: The labelled location to jump to
    """

    target: str


@dataclass(frozen=True, slots=True)
class JumpIfZero(Instruction):
    """Represents a conditional jump if the condition evaluates
    to zero.

    Attributes:
        condition: The condition to evaluate
        target: The labelled location to jump to
    """

    condition: Val
    target: str


@dataclass(frozen=True, slots=True)
class JumpIfNotZero(Instruction):
    """Represents a conditional jump if the condition evaluates
    to something other than zero

    Attributes:
        condition: The condition to evaluate
        target: The labelled location to jump to
    """

    condition: Val
    target: str


@dataclass(frozen=True, slots=True)
class Label(Instruction):
    """Represents a named location within the code.  Generally used
    as a jump target.

    Attributes:
        name: The name of the label
    """

    name: str


@dataclass(frozen=True, slots=True)
class FunCall(Instruction):
    """Call site for a function.

    Attributes:
        fun_val: Value that identifies the callable - a ``Var`` naming a function for a
            direct call, or a ``Var`` holding a function pointer for an indirect call.
        args: Ordered sequence of arguments to pass to the function.
        dst: The destination that receives the return value, or ``None`` for a void call.
    """

    fun_val: Val
    args: tuple[Val, ...]
    dst: Val | None = field(default=None)


@dataclass(frozen=True, slots=True)
class DoubleToInt(Instruction):
    """Cast a double to an integer

    Attributes:
        src: The source `double` value
        dst: The destination `int` or `long` value
    """

    src: Val
    dst: Val


@dataclass(frozen=True, slots=True)
class DoubleToUInt(Instruction):
    """Cast a double to an unsigned integer

    Attributes:
        src: The source `double` value
        dst: The destination `unsigned int` or `unsigned long` value
    """

    src: Val
    dst: Val


@dataclass(frozen=True, slots=True)
class IntToDouble(Instruction):
    """Cast an `int` or `long` to a double

    Attributes:
        src: The source integer value
        dst: The destination `double` value
    """

    src: Val
    dst: Val


@dataclass(frozen=True, slots=True)
class UIntToDouble(Instruction):
    """Cast an unsigned integer value to a double

    Attributes:
        src: The source unsigned integer value
        dst: The destination `double` value
    """

    src: Val
    dst: Val


@dataclass(frozen=True, slots=True)
class GetAddress(Instruction):
    """Obtain the memory address of a variable.

    Attributes:
        src: The variable whose address is taken.
        dst: The pointer-typed destination that receives the address.
    """

    src: Val
    dst: Val


@dataclass(frozen=True, slots=True)
class Load(Instruction):
    """Dereference a pointer and copy the value it points to into a destination.

    Attributes:
        src_ptr: A pointer-typed source value.
        dst: The destination that receives the loaded value.
    """

    src_ptr: Val
    dst: Val


@dataclass(frozen=True, slots=True)
class Store(Instruction):
    """Write a value through a pointer to its target memory location.

    Attributes:
        src: The value to store.
        dst_ptr: A pointer-typed destination that identifies where to write.
    """

    src: Val
    dst_ptr: Val


@dataclass(frozen=True, slots=True)
class AddPtr(Instruction):
    """Compute a pointer offset: ``dst = ptr + index * scale``.

    Used for pointer arithmetic and array element addressing.

    Attributes:
        ptr: The base pointer value.
        index: The signed integer index (number of elements to advance).
        scale: The size in bytes of one pointed-to element (e.g. 1 for ``char``,
            4 for ``int``, 8 for ``long``/``double``/pointer).
        dst: The destination that receives the resulting pointer value.
    """

    ptr: Val
    index: Val
    scale: int
    dst: Val


@dataclass(frozen=True, slots=True)
class CopyToOffset(Instruction):
    """Copy a scalar value into an aggregate variable at a byte offset.

    Writes ``src`` into the memory of the aggregate named by ``dst``,
    starting ``offset`` bytes from the aggregate's base address.  Used
    to initialise individual elements of an array or struct.

    Attributes:
        src: The scalar value to copy.
        dst: The name of the aggregate destination variable (a string
            identifier, not a ``Val``).
        offset: Byte offset from the start of ``dst`` at which to write.
    """

    src: Val
    dst: str
    offset: int


@dataclass(frozen=True, slots=True)
class CopyFromOffset(Instruction):
    """Copy an aggregate variable's member stored at ``offset`` into the
    scalar variable stored at ``dst``.

    Attributes:
        src: Name of the structure/aggregate to copy from
        offset: Byte offset to the value to extract.
        dst: Destination for the value extracted.
    """

    src: str
    offset: int
    dst: Val
