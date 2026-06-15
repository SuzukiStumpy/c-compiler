"""Assembly AST node definitions for the C compiler.

This package defines the Abstract Syntax Tree (AST) node types that represent
an assembly language program, produced by the code generator from the TAC AST
and consumed by the assembly emitter. The hierarchy mirrors the ASDL grammar
for the supported assembly subset:

    program             = Program(top_level*)
    assembly_type       = Byte | Longword | Quadword | Double | Float | ByteArray(int size, int alignment)
    top_level           = Function(identifier name, bool globl, instruction* instructions)
                        | StaticVariable(identifier name, bool globl, int init, int alignment, static_init* init_list)
                        | StaticConstant(identifier name, int alignment, static_init init)
    instruction         = Mov(assembly_type, operand src, operand dst)
                        | Movsx(assembly_type src_type, assembly_type dst_type, operand src, operand dst)
                        | MoveZeroExtend(assembly_type src_type, assembly_type, dst_type, operand src, operand dst)
                        | Unary(unary_operator, assembly_type, operand)
                        | Binary(binary_operator, assembly_type, operand src1, operand src2, operand dst)
                        | Cmp(assembly_type, operand, operand)
                        | Idiv(assembly_type, operand)
                        | Div(assembly_type, operand)
                        | Cdq(assembly_type)
                        | Shl(operand, assembly_type)
                        | Sar(operand, assembly_type)
                        | Shr(operand, assembly_type)
                        | Jmp(identifier)
                        | JmpCC(cond_code, identifier)
                        | SetCC(cond_code, operand)
                        | Label(identifier)
                        | AllocateStack(int)
                        | DeallocateStack(int)
                        | Push(operand)
                        | Pop(reg)
                        | Call(identifier)
                        | Ret
                        | Cvttsd2si(assembly_type dst_type, operand src, operand dst)
                        | Cvtsi2sd(assembly_type src_type, operand src, operand dst)
                        | Lea(operand src, operand dst)
    unary_operator      = Neg | Not
    binary_operator     = Add | Sub | Mult | And | Or | Xor
                        | AddDouble | SubDouble | MultDouble | DivDouble | XorDouble
                        | AddFloat | SubFloat | MultFloat | DivFloat | XorFloat
    operand             = Imm(int)
                        | Register(reg)
                        | Pseudo(identifier)
                        | Memory(reg, int)
                        | Data(identifier, int)
                        | Indexed(reg base, reg index, int scale)
                        | PseudoMem(identifier, int)
    cond_code           = E | NE | G | GE | L | LE | A | AE | B | BE
    reg                 = AX | BX | CX | DX | DI | SI | R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 | SP | BP
                        | XMM0 | XMM1 | XMM2 | XMM3 | XMM4 | XMM5 | XMM6 | XMM7
                        | XMM8 | XMM9 | XMM10 | XMM11 | XMM12 | XMM13 | XMM14 | XMM15

Nodes are organised into submodules by logical role:

    operators     - Unary operator nodes (UnaryOperator, Neg, Not)
    operands      - Operand nodes (Operand, Imm, Register, Pseudo, Memory)
    instructions  - Instruction nodes (Instruction, Mov, Ret, Unary, Binary, Cdq, Idiv, AllocateStack)
    declarations  - Program structure nodes (Function, Program)

All node types are re-exported from this package so callers can import directly
from ``compiler.syntax.asm`` without knowing which submodule a type lives in::

    from compiler.syntax.asm import Program, Function, Mov, Ret, Imm, Register, ...

Sum types (``Operand``, ``Instruction``, ``UnaryOperator``) are abstract base
classes; each variant is a concrete frozen dataclass subclassing the appropriate
base.  All nodes are immutable - once constructed an assembly AST node cannot be
modified.

``Pseudo`` operands are an intermediate form used during code generation and must
be replaced by ``Memory`` operands before the assembly is emitted.
"""

from compiler.syntax.asm.declarations import Function, Program, StaticConstant, StaticVariable, TopLevel
from compiler.syntax.asm.instructions import (
    AllocateStack,
    Binary,
    Call,
    Cdq,
    Cmp,
    CondCodes,
    Cvtsd2ss,
    Cvtsi2sd,
    Cvtsi2ss,
    Cvtss2sd,
    Cvttsd2si,
    Cvttss2si,
    DeallocateStack,
    Div,
    Idiv,
    IndirectCall,
    Instruction,
    Jmp,
    JmpCC,
    Label,
    Lea,
    Mov,
    MoveZeroExtend,
    Movsx,
    Pop,
    Push,
    Ret,
    Sar,
    SetCC,
    Shl,
    Shr,
    Unary,
)
from compiler.syntax.asm.operands import (
    ALLOCATABLE_FP,
    ALLOCATABLE_GP,
    Data,
    FPRegisters,
    GPRegisters,
    Imm,
    Indexed,
    Memory,
    Operand,
    Pseudo,
    PseudoMem,
    Register,
    Registers,
    Width,
)
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
    Neg,
    Not,
    Or,
    Sub,
    SubDouble,
    SubFloat,
    UnaryOperator,
    Xor,
    XorDouble,
    XorFloat,
)
from compiler.syntax.asm.values import (
    AssemblyType,
    Byte,
    ByteArray,
    CharInit,
    Double,
    DoubleInit,
    Float,
    FloatInit,
    Int128Init,
    IntInit,
    LongInit,
    Longword,
    PointerInit,
    Quadword,
    ShortInit,
    StaticInit,
    StringInit,
    UInt128Init,
    UShortInit,
    Word,
    ZeroInit,
)

__all__ = [
    # instructions
    "AllocateStack",
    "Binary",
    "Call",
    "Cdq",
    "Cmp",
    "Cvtsd2ss",
    "Cvtsi2sd",
    "Cvtsi2ss",
    "Cvtss2sd",
    "Cvttsd2si",
    "Cvttss2si",
    "DeallocateStack",
    "Div",
    "Idiv",
    "Instruction",
    "IndirectCall",
    "Jmp",
    "JmpCC",
    "Label",
    "Lea",
    "Mov",
    "MoveZeroExtend",
    "Movsx",
    "Push",
    "Pop",
    "Ret",
    "Sar",
    "SetCC",
    "Shl",
    "Shr",
    "Unary",
    "CondCodes",
    # operators
    "BinaryOperator",
    "UnaryOperator",
    "Add",
    "AddDouble",
    "AddFloat",
    "And",
    "DivDouble",
    "DivFloat",
    "Mult",
    "MultDouble",
    "MultFloat",
    "Neg",
    "Not",
    "Or",
    "Sub",
    "SubDouble",
    "SubFloat",
    "Xor",
    "XorDouble",
    "XorFloat",
    # operands
    "ALLOCATABLE_FP",
    "ALLOCATABLE_GP",
    "Data",
    "FPRegisters",
    "GPRegisters",
    "Imm",
    "Indexed",
    "Operand",
    "Pseudo",
    "PseudoMem",
    "Register",
    "Registers",
    "Memory",
    "Width",
    # declarations
    "Function",
    "Program",
    "StaticConstant",
    "StaticVariable",
    "TopLevel",
    # values
    "AssemblyType",
    "Byte",
    "Word",
    "ByteArray",
    "Double",
    "DoubleInit",
    "Float",
    "FloatInit",
    "CharInit",
    "ShortInit",
    "UShortInit",
    "Int128Init",
    "UInt128Init",
    "IntInit",
    "LongInit",
    "Longword",
    "Quadword",
    "StaticInit",
    "ZeroInit",
    "StringInit",
    "PointerInit",
]
