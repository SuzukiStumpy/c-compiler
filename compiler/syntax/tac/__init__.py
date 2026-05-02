"""AST node definitions for the intermediate Three Address Code representation.

This package defines the Abstract Syntax Tree (AST) node types produced by the
initial code generation pass (after the parser has done it's thing).  The grammar
mirrors the ASDL grammar for the supported assembly subset:

    program             = Program(function_definition)
    function_definition = Function(identifier, instruction* body)
    instruction         = Return(val)
                        | Unary(unary_operator, val src, val dst)
                        | Binary(binary_operator, val src1, val src2, val dst)
    val                 = Constant(int) | Var(identifier)
    unary_operator      = Complement | Negate
    binary_operator     = Add | Subtract | Multiply | Divide | Remainder
                        | LeftShift | RightShift | BitwiseAnd | BitwiseXor | BitwiseOr

Nodes are organised into submodules by logical role:

    declarations    - Program structure nodes (Function, Program)
    instructions    - Instruction nodes (Return, Unary)
    operators       - Operator types (Complement, Negate)
    values          - Value types (Constant, Var)

All node types are re-exported from this package so callers can import directly
from ``compiler.syntax.tac`` without knowing which submodule a type lives in::

    from compiler.tax import Program, Function, Return, Unary ... etc

Sum types (``Operator``, ``Instruction``, etc) are abstract base classes; each
variant is a concrete frozen dataclass subclassing the appropriate base.  All
nodes are immutable - once constructed a tac AST node cannot be modified.
"""

from compiler.syntax.tac.declarations import Function, Program
from compiler.syntax.tac.instructions import Binary, Instruction, Return, Unary
from compiler.syntax.tac.operators import (
    Add,
    BinaryOperator,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Complement,
    Divide,
    LeftShift,
    Multiply,
    Negate,
    Remainder,
    RightShift,
    Subtract,
    UnaryOperator,
)
from compiler.syntax.tac.values import Constant, Val, Var

__all__ = [
    "Program",
    "Function",
    "Instruction",
    "Return",
    "Unary",
    "Binary",
    "UnaryOperator",
    "Complement",
    "Negate",
    "BinaryOperator",
    "Add",
    "Subtract",
    "Multiply",
    "Divide",
    "Remainder",
    "LeftShift",
    "RightShift",
    "BitwiseAnd",
    "BitwiseXor",
    "BitwiseOr",
    "Val",
    "Constant",
    "Var",
]
