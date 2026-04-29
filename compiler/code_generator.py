"""Code generation stage: translates the main AST into an assembly AST.

Performs a structural translation from ``compiler.ast`` nodes to
``compiler.asm`` nodes, following the mapping below:

    ast.Program(functions)          → asm.Program(function)
    ast.Function(name, body)        → asm.Function(name, instructions)
    ast.Return(exp)                 → Mov(translate(exp), Register()), Ret()
    ast.Constant(value)             → Imm(value)

The resulting ``asm.Program`` is a complete assembly AST ready for the
emission stage.
"""

import compiler.asm as asm
import compiler.ast as ast
from compiler.utils.exceptions import CodeGenerationError


def generate(program: ast.Program) -> asm.Program:
    """Translate a main-AST ``Program`` into an assembly-AST ``Program``.

    The current grammar restricts a translation unit to a single function,
    matching the ``asm.Program`` node's single ``function`` field.

    Args:
        program: The ``Program`` node produced by the parser.

    Returns:
        An ``asm.Program`` node representing the equivalent assembly program.

    Raises:
        CodeGenerationError: If ``program`` contains more than one function,
            or if any statement or expression cannot be translated.
    """
    if len(program.functions) != 1:
        raise CodeGenerationError(f"expected exactly one function definition, got {len(program.functions)}")
    return asm.Program(function=_translate_function(program.functions[0]))


def _translate_function(function: ast.Function) -> asm.Function:
    """Translate a main-AST ``Function`` into an assembly-AST ``Function``.

    Args:
        function: The function definition node to translate.

    Returns:
        An ``asm.Function`` node with the same name and translated instructions.

    Raises:
        CodeGenerationError: If any statement in the body cannot be translated.
    """
    instructions: list[asm.Instruction] = []
    for stmt in function.body:
        instructions.extend(_translate_statement(stmt))
    return asm.Function(name=function.name, instructions=tuple(instructions))


def _translate_statement(stmt: ast.Statement) -> list[asm.Instruction]:
    """Translate a single statement into a sequence of assembly instructions.

    Args:
        stmt: The statement node to translate.

    Returns:
        A list of ``asm.Instruction`` nodes equivalent to the statement.

    Raises:
        CodeGenerationError: If the statement variant is not supported.
    """
    match stmt:
        case ast.Return(exp=exp):
            return [asm.Mov(src=_translate_exp(exp), dst=asm.Register()), asm.Ret()]
        case _:
            raise CodeGenerationError(f"unsupported statement type: {type(stmt).__name__}")


def _translate_exp(exp: ast.Exp) -> asm.Operand:
    """Translate an expression into an assembly operand.

    Args:
        exp: The expression node to translate.

    Returns:
        An ``asm.Operand`` node equivalent to the expression.

    Raises:
        CodeGenerationError: If the expression variant is not supported.
    """
    match exp:
        case ast.Constant(value=value):
            return asm.Imm(value=value)
        case _:
            raise CodeGenerationError(f"unsupported expression type: {type(exp).__name__}")
