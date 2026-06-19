"""AST node definitions for the C compiler.

This package defines the Abstract Syntax Tree (AST) node types produced by the
parser and consumed by later compiler phases. The hierarchy mirrors the Zephyr
Abstract Syntax Description Language (ASDL) grammar for the supported subset of C:

    program              = Program(declaration*)
    statement            = Return(exp?)
                         | Expression(exp)
                         | If(exp condition, statement then, statement? else_)
                         | Goto(identifier target)
                         | Label(identifier name, statement)
                         | Compound(block)
                         | Break(identifier)
                         | Continue(identifier)
                         | While(exp condition, statement body, identifier)
                         | DoWhile(statement body, exp condition, identifier)
                         | For(for_init init, exp? condition, exp? post, statement body, identifier)
                         | Switch(exp control, statement body, identifier)
                         | Case(exp value, statement body, identifier)
                         | Default(statement body, identifier)
                         | Null
    declaration          = FunDecl(function_declaration)
                         | VarDecl(variable_declaration)
                         | StructDecl(struct_declaration)
                         | UnionDecl(union_declaration)
                         | TypedefDecl(identifier name, type aliased_type)
                         | StaticAssert(exp condition, string message)
    struct_declaration   = (identifier tag, member_declaration* members)
    union_declaration    = (identifier tag, member_declaration* members)
    member_declaration   = (identifier member_name, type member_type)
    variable_declaration = (identifier name, initializer? init, type var_type, storage_class?)
    function_declaration = (identifier name, parameter* params, block? body, type fun_type, storage_class?)
    initializer          = SingleInit(exp) | CompoundInit(intializer*)
                         | MemberDesignatorInit(str, initializer) | IndexDesignatorInit(exp, initializer)
    type                 = Int | Long | UInt | ULong | Double | Char | SChar | UChar | Void
                         | FunType(type* params, type ret)
                         | Pointer(type referenced)
                         | Array(type element, int size)
                         | Structure(identifier tag)
                         | Union(identifier tag)
    storage_class        = Static | Extern
    parameter            = Parameter(type param_type, identifier name)
    for_init             = InitDecl(declaration) | InitExp(exp?)
    exp                  = Constant(const)
                         | String(bytes string)
                         | Var(identifier)
                         | Cast(type target_type, exp)
                         | Unary(unary_operator, exp)
                         | Binary(binary_operator, exp, exp)
                         | Assignment(exp, exp)
                         | CompoundAssignment(exp, binary_operator, exp)
                         | Conditional(exp condition, exp exp_true, exp exp_false)
                         | FunctionCall(exp callee, exp* args)
                         | Dereference(exp)
                         | AddrOf(exp)
                         | Subscript(exp, exp)
                         | SizeOf(exp)
                         | SizeOfT(type)
                         | AlignOf(exp)
                         | AlignOfT(type)
                         | Dot(exp structure, identifier member)
                         | Arrow(exp pointer, identifier member)
    unary_operator       = Complement | Negate | Not
                         | PreIncrement | PreDecrement
                         | PostIncrement | PostDecrement
    binary_operator      = Add | Subtract | Multiply | Divide | Remainder
                         | LeftShift | RightShift | BitwiseAnd | BitwiseXor | BitwiseOr
                         | And | Or | Equal | NotEqual | LessThan | LessOrEqual
                         | GreaterThan | GreaterOrEqual
    linkage              = Linkage(NONE | EXTERNAL)
    block                = Block(block_item*)
    block_item           = S(statement) | D(declaration)
    const                = ConstInt(int)
                         | ConstLong(int)
                         | ConstUInt(int)
                         | ConstULong(int)
                         | ConstDouble(double)
                         | ConstFloat(float)
                         | ConstChar(int)
                         | ConstUChar(int)

Nodes are organised into submodules by logical role:

    types        - C type nodes (Type, IntType, VoidType, ...)
    expressions  - Expression nodes (Exp, Constant, Unary, Binary, Assignment,
                   Conditional, FunctionCall, SizeOf, SizeOfT, AlignOf, AlignOfT, Dot, Arrow)
    statements   - Statement nodes (Statement, Return, Expression, If, Null,
                   Goto, Label, Break, Continue, While, DoWhile)
    declarations - Program structure nodes (VariableDeclaration, Declaration,
                   FunDecl, VarDecl, StructDecl, MemberDeclaration,
                   StructDeclaration, TypedefDecl, StaticAssert, BlockItem,
                   S, D, Block, Compound, ForInit, InitDecl, InitExp, For,
                   Parameter, Function, Program)
    operators    - Operators which act upon expressions
                   (Complement, Negate, Not,
                   PreIncrement, PreDecrement, PostIncrement, PostDecrement,
                   Add, Subtract, Multiply, Divide, Remainder,
                   LeftShift, RightShift, BitwiseAnd, BitwiseXor, BitwiseOr,
                   And, Or, Equal, NotEqual, LessThan, LessOrEqual,
                   GreaterThan, GreaterOrEqual)

All node types are re-exported from this package so callers can import directly
from ``compiler.syntax.ast`` without knowing which submodule a type lives in::

    from compiler.syntax.ast import Program, Function, Return, Constant, IntType

Sum types (``Type``, ``Exp``, ``Statement``, ``Declaration``, ``BlockItem``,
``ForInit``) are abstract base classes; each variant is a concrete frozen
dataclass subclassing the appropriate base. All nodes are immutable - once
constructed an AST node cannot be modified.
"""

from compiler.syntax.ast.constants import (
    Const,
    ConstBool,
    ConstChar,
    ConstDouble,
    ConstFloat,
    ConstInt,
    ConstInt128,
    ConstLong,
    ConstLongDouble,
    ConstLongLong,
    ConstShort,
    ConstUChar,
    ConstUInt,
    ConstUInt128,
    ConstULong,
    ConstULongLong,
    ConstUShort,
)
from compiler.syntax.ast.declarations import (
    Block,
    BlockItem,
    Compound,
    D,
    Declaration,
    EnumDecl,
    EnumDeclaration,
    EnumeratorDeclaration,
    For,
    ForInit,
    Function,
    FunDecl,
    InitDecl,
    InitExp,
    MemberDeclaration,
    Parameter,
    Program,
    S,
    StorageClass,
    StructDecl,
    StructDeclaration,
    UnionDecl,
    UnionDeclaration,
    VarDecl,
    VariableDeclaration,
)
from compiler.syntax.ast.declarations import (
    StaticAssert as StaticAssert,
)
from compiler.syntax.ast.declarations import (
    TypedefDecl as TypedefDecl,
)
from compiler.syntax.ast.expressions import (
    AddrOf,
    AlignOf,
    AlignOfT,
    Arrow,
    Assignment,
    Binary,
    Cast,
    CompoundAssignment,
    CompoundInit,
    CompoundLiteral,
    Conditional,
    Constant,
    Dereference,
    Dot,
    Exp,
    FunctionCall,
    IndexDesignatorInit,
    Initializer,
    MemberDesignatorInit,
    SingleInit,
    SizeOf,
    SizeOfT,
    String,
    Subscript,
    Unary,
    VaArg,
    VaCopy,
    VaEnd,
    Var,
    VaStart,
)
from compiler.syntax.ast.operators import (
    Add,
    And,
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
    Or,
    PostDecrement,
    PostIncrement,
    PreDecrement,
    PreIncrement,
    Remainder,
    RightShift,
    Subtract,
    UnaryOperator,
)
from compiler.syntax.ast.pretty_print import pretty_print
from compiler.syntax.ast.statements import (
    Break,
    Case,
    Continue,
    Default,
    DoWhile,
    Expression,
    Goto,
    If,
    Label,
    Null,
    Return,
    Statement,
    Switch,
    While,
)
from compiler.syntax.ast.types import (
    Array,
    BoolType,
    Char,
    DoubleType,
    EnumType,
    FloatType,
    FunType,
    Int128Type,
    IntType,
    Linkage,
    LongDoubleType,
    LongLongType,
    LongType,
    Pointer,
    QualifiedType,
    SChar,
    ShortType,
    Structure,
    Type,
    UChar,
    UInt128Type,
    UIntType,
    ULongLongType,
    ULongType,
    Union,
    UShortType,
    VoidType,
    is_const,
    is_restrict,
    is_volatile,
    unqualify,
)

__all__ = [
    "Arrow",
    "CompoundAssignment",
    "Dot",
    "BoolType",
    "EnumDecl",
    "EnumDeclaration",
    "StaticAssert",
    "EnumeratorDeclaration",
    "EnumType",
    "MemberDeclaration",
    "Structure",
    "StructDecl",
    "StructDeclaration",
    "Union",
    "UnionDecl",
    "UnionDeclaration",
    "AlignOf",
    "AlignOfT",
    "SizeOf",
    "SizeOfT",
    "VoidType",
    "String",
    "ConstBool",
    "ConstChar",
    "ConstUChar",
    "ConstShort",
    "ConstUShort",
    "ConstLongLong",
    "ConstULongLong",
    "ConstInt128",
    "ConstUInt128",
    "Char",
    "SChar",
    "UChar",
    "ShortType",
    "UShortType",
    "Int128Type",
    "UInt128Type",
    "Subscript",
    "Array",
    "Initializer",
    "SingleInit",
    "CompoundInit",
    "CompoundLiteral",
    "MemberDesignatorInit",
    "IndexDesignatorInit",
    "Dereference",
    "AddrOf",
    "Pointer",
    "ConstDouble",
    "ConstFloat",
    "ConstLongDouble",
    "DoubleType",
    "FloatType",
    "LongDoubleType",
    "ConstUInt",
    "ConstULong",
    "UIntType",
    "ULongType",
    "LongLongType",
    "ULongLongType",
    "Const",
    "ConstLong",
    "ConstInt",
    "Cast",
    "LongType",
    "FunType",
    "StorageClass",
    "Switch",
    "Case",
    "Default",
    "Break",
    "Continue",
    "For",
    "While",
    "DoWhile",
    "InitExp",
    "InitDecl",
    "ForInit",
    "Block",
    "Compound",
    "Goto",
    "Label",
    "Conditional",
    "If",
    "PostDecrement",
    "PostIncrement",
    "PreIncrement",
    "PreDecrement",
    "Expression",
    "Null",
    "Var",
    "VaStart",
    "VaArg",
    "VaEnd",
    "VaCopy",
    "Assignment",
    "Type",
    "IntType",
    "Linkage",
    "Exp",
    "Constant",
    "Unary",
    "Binary",
    "UnaryOperator",
    "BinaryOperator",
    "Statement",
    "Return",
    "BlockItem",
    "S",
    "D",
    "VariableDeclaration",
    "Declaration",
    "FunDecl",
    "VarDecl",
    "FunctionCall",
    "Parameter",
    "Function",
    "Program",
    "QualifiedType",
    "is_const",
    "is_restrict",
    "is_volatile",
    "unqualify",
    "pretty_print",
    "Complement",
    "Negate",
    "Not",
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
    "And",
    "Or",
    "Equal",
    "NotEqual",
    "LessThan",
    "LessOrEqual",
    "GreaterThan",
    "GreaterOrEqual",
]
