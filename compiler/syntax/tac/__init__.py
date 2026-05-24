"""AST node definitions for the intermediate Three Address Code representation.

This package defines the Abstract Syntax Tree (AST) node types produced by the
initial code generation pass (after the parser has done it's thing).  The grammar
mirrors the ASDL grammar for the supported assembly subset:

    program             = Program(top_level*)
    top_level           = Function(identifier, bool global, identifier* params, instruction* body)
                        | StaticVariable(identifier, bool global, type t,static_init* init)
    instruction         = Return(val)
                        | SignExtend(val src, val dst)
                        | Truncate(val src, val dst)
                        | ZeroExtend(val src, val dst)
                        | DoubleToInt(val src, val dst)
                        | DoubleToUInt(val src, val dst)
                        | IntToDouble(val src, val dst)
                        | UIntToDouble(val src, val dst)
                        | Unary(unary_operator, val src, val dst)
                        | Binary(binary_operator, val src1, val src2, val dst)
                        | Copy(val src, val dst)
                        | Jump(identifier target)
                        | JumpIfZero(val condition, identifier target)
                        | JumpIfNotZero(val condition, identifier target)
                        | Label(identifier)
                        | FunCall(val fun_val, val* args, val dst)
                        | GetAddress(val src, val dst)
                        | Load(val src_ptr, val dst)
                        | Store(val src, val dst_ptr)
                        | AddPtr(val ptr, val index, int scale, val dst)
                        | CopyToOffset(val src, identifier dst, int offset)
    val                 = Constant(const) | Var(identifier)
    static_init         = IntInit(int) | LongInit(int) | DoubleInit(float) | ZeroInit(int bytes)
    unary_operator      = Complement | Negate | Not
    binary_operator     = Add | Subtract | Multiply | Divide | Remainder
                        | LeftShift | RightShift | BitwiseAnd | BitwiseXor | BitwiseOr
                        | Equal | NotEqual | LessThan | LessOrEqual | GreaterThan
                        | GreaterOrEqual

Nodes are organised into submodules by logical role:

    declarations    - Program structure nodes (Function, Program)
    instructions    - Instruction nodes (Return, Unary)
    operators       - Operator types (Complement, Negate)
    values          - Value types (Constant, Var)

All node types are re-exported from this package so callers can import directly
from ``compiler.syntax.tac`` without knowing which submodule a type lives in::

    from compiler.tac import Program, Function, Return, Unary ... etc

Sum types (``Operator``, ``Instruction``, etc) are abstract base classes; each
variant is a concrete frozen dataclass subclassing the appropriate base.  All
nodes are immutable - once constructed a tac AST node cannot be modified.
"""

from compiler.syntax.tac.declarations import Function, Program, StaticVariable, TopLevel
from compiler.syntax.tac.instructions import (
    AddPtr,
    Binary,
    Copy,
    CopyToOffset,
    DoubleToInt,
    DoubleToUInt,
    FunCall,
    GetAddress,
    Instruction,
    IntToDouble,
    Jump,
    JumpIfNotZero,
    JumpIfZero,
    Label,
    Load,
    Return,
    SignExtend,
    Store,
    Truncate,
    UIntToDouble,
    Unary,
    ZeroExtend,
)
from compiler.syntax.tac.operators import (
    Add,
    BinaryOperator,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Complement,
    Divide,
    Equal,
    GreaterOrEqual,
    GreaterThan,
    LeftShift,
    LessOrEqual,
    LessThan,
    Multiply,
    Negate,
    Not,
    NotEqual,
    Remainder,
    RightShift,
    Subtract,
    UnaryOperator,
)
from compiler.syntax.tac.values import Constant, DoubleInit, IntInit, LongInit, StaticInit, Val, Var, ZeroInit

__all__ = [
    "AddPtr",
    "CopyToOffset",
    "DoubleToInt",
    "DoubleToUInt",
    "IntToDouble",
    "UIntToDouble",
    "ZeroExtend",
    "StaticVariable",
    "TopLevel",
    "FunCall",
    "GetAddress",
    "Load",
    "Store",
    "Equal",
    "GreaterOrEqual",
    "GreaterThan",
    "LessOrEqual",
    "LessThan",
    "NotEqual",
    "Not",
    "Copy",
    "Jump",
    "JumpIfNotZero",
    "JumpIfZero",
    "Label",
    "Program",
    "Function",
    "Instruction",
    "Return",
    "SignExtend",
    "Truncate",
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
    "StaticInit",
    "IntInit",
    "LongInit",
    "DoubleInit",
    "ZeroInit",
]
