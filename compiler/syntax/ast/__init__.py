"""AST node definitions for the C compiler.

This package defines the Abstract Syntax Tree (AST) node types produced by the
parser and consumed by later compiler phases. The hierarchy mirrors the Zephyr
Abstract Syntax Description Language (ASDL) grammar for the supported subset of C:

    program             = Program(function_definition+)
    function_definition = Function(type return_type, identifier name,
                                   parameter* params, statement+ body)
    parameter           = Parameter(type param_type, identifier name)
    statement           = Return(exp)
    exp                 = Constant(int) | Unary(unary_operator, exp)
    unary_operator      = Complement | Negate
    type                = IntType

Nodes are organised into submodules by logical role:

    types        — C type nodes (Type, IntType)
    expressions  — Expression nodes (Exp, Constant)
    statements   — Statement nodes (Statement, Return)
    declarations — Program structure nodes (Parameter, Function, Program)
    operators    - Operators which act upon expressions (Complement, Negate)

All node types are re-exported from this package so callers can import directly
from ``compiler.syntax.ast`` without knowing which submodule a type lives in::

    from compiler.syntax.ast import Program, Function, Return, Constant, IntType

Sum types (``Type``, ``Exp``, ``Statement``) are abstract base classes; each
variant is a concrete frozen dataclass subclassing the appropriate base. All
nodes are immutable — once constructed an AST node cannot be modified.
"""

from compiler.syntax.ast.declarations import Function, Parameter, Program
from compiler.syntax.ast.expressions import Constant, Exp
from compiler.syntax.ast.operators import Complement, Negate
from compiler.syntax.ast.pretty_print import pretty_print
from compiler.syntax.ast.statements import Return, Statement
from compiler.syntax.ast.types import IntType, Type

__all__ = [
    "Type",
    "IntType",
    "Exp",
    "Constant",
    "Statement",
    "Return",
    "Parameter",
    "Function",
    "Program",
    "pretty_print",
    "Complement",
    "Negate",
]
