"""Assembly AST node definitions for the C compiler.

This package defines the Abstract Syntax Tree (AST) node types that represent
an assembly language program, produced by the code generator from the TAC AST
and consumed by the assembly emitter. The hierarchy mirrors the ASDL grammar
for the supported assembly subset:

    program             = Program(function_definition)
    function_definition = Function(identifier name, instruction* instructions)
    instruction         = Mov(operand src, operand dst)
                        | Unary(unary_operator, operand)
                        | Binary(binary_operator, operand src1, operand src2, operand dst)
                        | Cmp(operand, operand)
                        | Idiv(operand)
                        | Shl(operand)
                        | Sar(operand)
                        | Cdq
                        | Jmp(identifier)
                        | JmpCC(cond_code, identifier)
                        | SetCC(cond_code, operand)
                        | Label(identifier)
                        | AllocateStack(int)
                        | Ret
    unary_operator      = Neg | Not
    binary_operator     = Add | Sub | Mult | And | Or | Xor
    operand             = Imm(int)
                        | Register(reg)
                        | Pseudo(identifier)
                        | Stack(int)
    cond_code           = E | NE | G | GE | L | LE
    reg                 = AX | CX | DX | R10 | R11

Nodes are organised into submodules by logical role:

    operators     — Unary operator nodes (UnaryOperator, Neg, Not)
    operands      — Operand nodes (Operand, Imm, Register, Pseudo, Stack)
    instructions  — Instruction nodes (Instruction, Mov, Ret, Unary, Binary, Cdq, Idiv, AllocateStack)
    declarations  — Program structure nodes (Function, Program)

All node types are re-exported from this package so callers can import directly
from ``compiler.syntax.asm`` without knowing which submodule a type lives in::

    from compiler.syntax.asm import Program, Function, Mov, Ret, Imm, Register, ...

Sum types (``Operand``, ``Instruction``, ``UnaryOperator``) are abstract base
classes; each variant is a concrete frozen dataclass subclassing the appropriate
base.  All nodes are immutable — once constructed an assembly AST node cannot be
modified.

``Pseudo`` operands are an intermediate form used during code generation and must
be replaced by ``Stack`` operands before the assembly is emitted.
"""

from compiler.syntax.asm.declarations import Function, Program
from compiler.syntax.asm.instructions import (
    AllocateStack,
    Binary,
    Cdq,
    Cmp,
    CondCodes,
    Idiv,
    Instruction,
    Jmp,
    JmpCC,
    Label,
    Mov,
    Ret,
    Sar,
    SetCC,
    Shl,
    Unary,
)
from compiler.syntax.asm.operands import Imm, Operand, Pseudo, Register, Registers, Stack, Width
from compiler.syntax.asm.operators import Add, And, BinaryOperator, Mult, Neg, Not, Or, Sub, UnaryOperator, Xor

__all__ = [
    "UnaryOperator",
    "Neg",
    "Not",
    "BinaryOperator",
    "Add",
    "Sub",
    "Mult",
    "And",
    "Or",
    "Xor",
    "Operand",
    "Imm",
    "Register",
    "Registers",
    "Width",
    "Pseudo",
    "Stack",
    "CondCodes",
    "Instruction",
    "Mov",
    "Ret",
    "Unary",
    "Binary",
    "Cdq",
    "Idiv",
    "Shl",
    "Sar",
    "AllocateStack",
    "Cmp",
    "Jmp",
    "JmpCC",
    "SetCC",
    "Label",
    "Function",
    "Program",
]
