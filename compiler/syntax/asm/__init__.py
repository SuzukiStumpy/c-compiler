"""Assembly AST node definitions for the C compiler.

This package defines the Abstract Syntax Tree (AST) node types that represent
an assembly language program, produced by the code generator from the main AST
and consumed by the assembly emitter. The hierarchy mirrors the ASDL grammar
for the supported assembly subset:

    program             = Program(function_definition)
    function_definition = Function(identifier name, instruction* instructions)
    instruction         = Mov(operand src, operand dst)
                        | Ret
    operand             = Imm(int)
                        | Register

Nodes are organised into submodules by logical role:

    operands      — Operand nodes (Operand, Imm, Register)
    instructions  — Instruction nodes (Instruction, Mov, Ret)
    declarations  — Program structure nodes (Function, Program)

All node types are re-exported from this package so callers can import directly
from ``compiler.syntax.asm`` without knowing which submodule a type lives in::

    from compiler.syntax.asm import Program, Function, Mov, Ret, Imm, Register

Sum types (``Operand``, ``Instruction``) are abstract base classes; each
variant is a concrete frozen dataclass subclassing the appropriate base. All
nodes are immutable — once constructed an assembly AST node cannot be modified.
"""

from compiler.syntax.asm.declarations import Function, Program
from compiler.syntax.asm.instructions import Instruction, Mov, Ret
from compiler.syntax.asm.operands import Imm, Operand, Register

__all__ = [
    "Operand",
    "Imm",
    "Register",
    "Instruction",
    "Mov",
    "Ret",
    "Function",
    "Program",
]
