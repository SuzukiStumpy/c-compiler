"""Semantic analysis stage: variable and function resolution over the AST.

Walks the ``compiler.syntax.ast`` tree produced by the parser and performs
semantic checks that cannot be expressed by the grammar alone.  Responsibilities
are *variable resolution*, *function declaration validation*, and *type checking*:

* Every variable reference is verified to have a corresponding declaration in
  scope, and each declared name is assigned a unique internal identifier so that
  later pipeline stages (TAC generation, code generation) need not reason about
  shadowing or scoping rules.  File-scope variables keep their original names;
  block-scope variables (including ``static`` locals) are renamed to unique
  internal identifiers.
* Every function name is tracked in a program-wide function map.  Multiple
  *declarations* of the same function are permitted; a function *definition*
  (one with a body) may appear at most once per program.  Top-level functions
  are processed in source order; a function must be declared or defined before
  any call site that precedes it.
* File-scope variable declarations are added to the identifier table unaltered.
  ``static`` file-scope variables receive ``Linkage.INTERNAL``; all others
  receive ``Linkage.EXTERNAL``.
* Block-scope ``extern`` declarations are checked to ensure they carry no
  initialiser and are not mixed with a non-``extern`` declaration of the same
  name in the same scope.  Block-scope ``static`` declarations are renamed
  exactly like regular local variables but carry ``Linkage.INTERNAL``.
* ``FunctionCall`` expressions are validated to ensure the callee has been
  declared or defined at an earlier point in the source, and that the callee
  name is not a local variable.
* ``Var`` expressions are checked to ensure the name was declared as a variable
  rather than as a function.
* Linkage is recorded for each binding: ``Linkage.NONE`` for local variables;
  ``Linkage.INTERNAL`` for ``static`` variables and for ``extern`` declarations
  that inherit a prior internal linkage (C11 sec. 6.2.2p4); ``Linkage.EXTERNAL``
  for functions and file-scope variables that have not established internal
  linkage.

The main entry point for external callers is ``resolve``, which constructs a
``SemanticAnalyser``, runs it over the program, and returns a semantically
validated, uniquely-renamed ``ast.Program``.  The ``SemanticAnalyser`` class
is also exposed directly for testing.

Raises:
    SemanticError: If the program contains a semantic violation such as
        a duplicate variable declaration, a reference to an undeclared variable
        or function, a redefinition of a function, an invalid lvalue, a function
        name used as a variable, a variable called as a function, mixed linkage
        on a block-scope variable, or a block-scope ``extern`` declaration with
        an initialiser.
"""

import math
from itertools import count
from typing import NamedTuple

import compiler.syntax.ast as ast
from compiler.syntax.ast.declarations import MemberDeclaration, StorageClass, StructDeclaration, UnionDeclaration
from compiler.syntax.ast.operators import (
    And,
    Equal,
    GreaterOrEqual,
    GreaterThan,
    LeftShift,
    LessOrEqual,
    LessThan,
    Not,
    NotEqual,
    Or,
    RightShift,
)
from compiler.syntax.ast.types import (
    Array,
    Char,
    DoubleType,
    FunType,
    Int128Type,
    IntType,
    Linkage,
    LongLongType,
    LongType,
    Pointer,
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
)
from compiler.types.symbols import (
    CharInit,
    ConstantAttr,
    DoubleInit,
    FunAttr,
    Initial,
    InitialValue,
    Int128Init,
    IntInit,
    LocalAttr,
    LongInit,
    MemberEntry,
    NoInitializer,
    PointerInit,
    ShortInit,
    StaticAttr,
    StaticInit,
    StringInit,
    StructEntry,
    StructTypeTable,
    SymbolTable,
    SymbolTableEntry,
    Tentative,
    UCharInit,
    UInt128Init,
    ULongInit,
    UShortInit,
    ZeroInit,
)
from compiler.utils.exceptions import SemanticError

_TYPE_SIZE: dict[type, int] = {
    IntType: 4,
    UIntType: 4,
    LongType: 8,
    ULongType: 8,
    LongLongType: 8,
    ULongLongType: 8,
    DoubleType: 8,
    Char: 1,
    SChar: 1,
    UChar: 1,
    ShortType: 2,
    UShortType: 2,
    Int128Type: 16,
    UInt128Type: 16,
}
_SIGNED_TYPES: frozenset[type] = frozenset({IntType, LongType, LongLongType, Int128Type})
_CHAR_TYPES: tuple[type, ...] = (Char, SChar, UChar)
# Types that undergo integer promotion to IntType before arithmetic (C11 §6.3.1.1).
_PROMOTE_TO_INT: tuple[type, ...] = (Char, SChar, UChar, ShortType, UShortType)


def _round_up(n: int, align: int) -> int:
    """Round ``n`` up to the nearest multiple of ``align``.

    Args:
        n: The value to round up.
        align: The alignment boundary (must be positive).

    Returns:
        The smallest integer ``>= n`` that is a multiple of ``align``.
    """
    return (n + align - 1) // align * align


def type_alignment(t: Type, struct_type_table: StructTypeTable) -> int:
    """Return the required alignment in bytes for a C type.

    Scalar and pointer types have fixed alignments (matching their sizes).
    Array alignment equals element alignment.  Struct alignment is looked up
    from ``struct_type_table``.

    Args:
        t: The resolved C type whose alignment is needed.
        struct_type_table: Mapping from resolved struct tags to ``StructEntry``
            layout records; used to determine struct alignment.

    Returns:
        The alignment requirement in bytes.

    Raises:
        SemanticError: If ``t`` is a ``Structure`` type not yet present in
            ``struct_type_table`` (incomplete struct type).
    """
    match t:
        case IntType() | UIntType():
            return 4
        case LongType() | ULongType() | LongLongType() | ULongLongType() | DoubleType() | Pointer():
            return 8
        case Char() | SChar() | UChar():
            return 1
        case ShortType() | UShortType():
            return 2
        case Int128Type() | UInt128Type():
            return 16
        case Array(element=elem_t):
            return type_alignment(elem_t, struct_type_table)
        case Structure(tag=tag) | Union(tag=tag):
            entry = struct_type_table.get(tag)
            if entry is None:
                raise SemanticError(f"Cannot determine alignment of incomplete aggregate type '{tag}'")
            return entry.alignment
        case _:
            raise SemanticError(f"Cannot determine alignment of type '{t}'")


def type_size(t: Type, struct_type_table: StructTypeTable) -> int:
    """Return the size in bytes of a C type.

    Args:
        t: The resolved C type whose size is needed.
        struct_type_table: Mapping from resolved struct tags to ``StructEntry``
            layout records; used to determine struct size.

    Returns:
        The size in bytes.

    Raises:
        SemanticError: If ``t`` is a ``Structure`` type not yet present in
            ``struct_type_table`` (incomplete struct type).
    """
    match t:
        case IntType() | UIntType():
            return 4
        case LongType() | ULongType() | LongLongType() | ULongLongType() | DoubleType() | Pointer():
            return 8
        case Char() | SChar() | UChar():
            return 1
        case ShortType() | UShortType():
            return 2
        case Int128Type() | UInt128Type():
            return 16
        case Array(element=elem_t, size=n):
            return type_size(elem_t, struct_type_table) * n
        case Structure(tag=tag) | Union(tag=tag):
            entry = struct_type_table.get(tag)
            if entry is None:
                raise SemanticError(f"Cannot determine size of incomplete aggregate type '{tag}'")
            return entry.size
        case _:
            raise SemanticError(f"Cannot determine size of type '{t}'")


def get_common_type(type1: Type, type2: Type) -> Type:
    """Return the common type to which both operands of a binary expression should be converted.

    Implements C's usual arithmetic conversions for the integer types supported by this compiler.
    Character types (``Char``, ``SChar``, ``UChar``) are first promoted to ``IntType`` per
    C11 sec. 6.3.1.1.  When both operands already share the same type the result is that type
    unchanged.  When the sizes differ the wider type is chosen.  When sizes are equal but
    signedness differs the unsigned type is chosen (C11 sec. 6.3.1.8, rule 4).

    Args:
        type1: The type of the first operand.
        type2: The type of the second operand.

    Returns:
        The common ``Type`` to which both operands must be converted before the operation is
        performed.
    """
    if isinstance(type1, _PROMOTE_TO_INT):
        type1 = IntType()
    if isinstance(type2, _PROMOTE_TO_INT):
        type2 = IntType()
    if type1 == type2:
        return type1
    if isinstance(type1, DoubleType) or isinstance(type2, DoubleType):
        return DoubleType()
    size1 = _TYPE_SIZE.get(type(type1), 0)
    size2 = _TYPE_SIZE.get(type(type2), 0)
    if size1 == size2:
        return type2 if type(type1) in _SIGNED_TYPES else type1
    return type1 if size1 > size2 else type2


def get_common_pointer_type(e1: ast.Exp, e2: ast.Exp) -> Type:
    """Return the common pointer type for two expressions where at least one operand is a pointer.

    Implements C's pointer-compatibility rules (C11 sec. 6.5.15, sec. 6.3.2.3): two pointer operands are
    compatible when they share the same type, when one of them is a null pointer constant (an
    integer constant expression with value zero) in which case the other operand's type is used,
    or when one operand is ``void *`` and the other is any pointer type, in which case the common
    type is ``void *``.

    Args:
        e1: The first operand expression, already resolved and typed by the semantic analysis pass.
        e2: The second operand expression, already resolved and typed by the semantic analysis pass.

    Returns:
        The common ``Type`` to which both operands are compatible.

    Raises:
        SemanticError: If the two expressions have incompatible pointer types and neither is a null
            pointer constant nor a ``void *``.
    """
    e1_t = e1.exp_type
    e2_t = e2.exp_type
    assert e1_t is not None and e2_t is not None
    if e1_t == e2_t:
        return e1_t
    if _is_null_pointer_constant(e1):
        return e2_t
    if _is_null_pointer_constant(e2):
        return e1_t
    if isinstance(e1_t, Pointer) and isinstance(e2_t, Pointer):
        if isinstance(e1_t.referenced, VoidType) or isinstance(e2_t.referenced, VoidType):
            return Pointer(VoidType())
    raise SemanticError("Expressions have incompatible types")


def _convert_by_assignment(e: ast.Exp, target_type: Type) -> ast.Exp:
    """Convert an expression to a target type using the rules for assignment compatibility (C11 sec. 6.5.16.1).

    The legal cases are: the expression already has the target type (covers struct-to-same-struct
    assignment - no ``Cast`` node is inserted); both the expression and the target are arithmetic
    types (usual arithmetic conversions apply); the expression is a null pointer constant and the
    target is a pointer type; or one side is ``void *`` and the other is any pointer type
    (C11 sec. 6.3.2.3p1) - the non-``void *`` operand is converted to the target type.
    All other combinations are rejected as incompatible, including assignments between two struct
    types that do not share the same resolved tag.

    Args:
        e: The source expression, already typed by the semantic analysis pass.
        target_type: The type the expression must be converted to.

    Returns:
        ``e`` unchanged if it is already of ``target_type``, otherwise an ``ast.Cast`` node
        wrapping ``e`` with ``exp_type`` set to ``target_type``.

    Raises:
        SemanticError: If the conversion is not permitted under the assignment-compatibility rules.
    """
    e_type = e.exp_type
    assert e_type is not None
    if e_type == target_type:
        return e
    if type(e_type) in _TYPE_SIZE and type(target_type) in _TYPE_SIZE:
        return ast.Cast(target_type, e, exp_type=target_type)
    if _is_null_pointer_constant(e) and isinstance(target_type, Pointer):
        return ast.Cast(target_type, e, exp_type=target_type)
    if isinstance(e_type, Pointer) and isinstance(target_type, Pointer):
        if isinstance(e_type.referenced, VoidType) or isinstance(target_type.referenced, VoidType):
            return ast.Cast(target_type, e, exp_type=target_type)
    if isinstance(e_type, Pointer) or isinstance(target_type, Pointer):
        raise SemanticError("Cannot implicitly convert between pointer and non-pointer types in assignment.")
    raise SemanticError(f"Cannot convert expression of type '{e_type}' to '{target_type}' in assignment.")


def _static_pointer_init(raw_init: ast.Exp, variable_map: "dict[str, VariableBinding]") -> InitialValue:
    """Return the symbol-table ``InitialValue`` for a static pointer variable initialiser.

    Must be called with the *unresolved* initialiser expression so that the original
    source-level variable name is available for a linkage check.  Value/type compatibility
    is enforced separately by ``_convert_by_assignment``; this function only classifies
    the initialiser form and verifies static storage duration for address-of operands.

    Args:
        raw_init: The unresolved initialiser expression straight from the AST.
        variable_map: The identifier resolution map at the point of the declaration, used
            to verify that a ``&var`` operand has static storage duration.

    Returns:
        ``ULongInit(0)`` when *raw_init* is a constant (null pointer constant).
        ``PointerInit(unique_name)`` when *raw_init* is ``&var`` for a static variable,
        where *unique_name* is the variable's resolved identifier.

    Raises:
        SemanticError: If a ``&var`` operand is not in *variable_map* or has automatic
            storage duration (``Linkage.NONE``).
    """
    if isinstance(raw_init, ast.Constant):
        return ULongInit(0)
    assert isinstance(raw_init, ast.AddrOf) and isinstance(raw_init.exp, ast.Var)
    orig_name = raw_init.exp.name
    binding = variable_map.get(orig_name)
    if binding is None or binding.linkage == Linkage.NONE:
        raise SemanticError(
            f"Cannot use '&{orig_name}' as a static pointer initialiser: "
            f"'{orig_name}' does not have static storage duration."
        )
    return PointerInit(binding.unique_name)


def _truncate_case_value(val: int, switch_type: Type) -> int:
    """Truncate an integer case constant to the switch expression's type.

    C converts case label values to the type of the switch control expression
    before comparing them.  For example, ``case 34359738768l`` in an ``int``
    switch equals ``case 400`` because ``34359738768 mod 2**32 == 400``.

    Args:
        val: The raw integer value of the case constant.
        switch_type: The C type of the switch control expression.

    Returns:
        The value after truncation and (for signed types) sign extension.
    """
    match switch_type:
        case IntType():
            v = val & 0xFFFFFFFF
            return v - (1 << 32) if v >= (1 << 31) else v
        case LongType() | LongLongType():
            v = val & 0xFFFFFFFFFFFFFFFF
            return v - (1 << 64) if v >= (1 << 63) else v
        case UIntType():
            return val & 0xFFFFFFFF
        case ULongType() | ULongLongType():
            return val & 0xFFFFFFFFFFFFFFFF
        case ShortType():
            v = val & 0xFFFF
            return v - (1 << 16) if v >= (1 << 15) else v
        case UShortType():
            return val & 0xFFFF
        case Int128Type():
            v = val & ((1 << 128) - 1)
            return v - (1 << 128) if v >= (1 << 127) else v
        case UInt128Type():
            return val & ((1 << 128) - 1)
        case _:
            return val


def _is_null_pointer_constant(e: ast.Exp) -> bool:
    """Check to see if an expression node evaluates to a constant zero value

    Args:
        e: The expression to check

    Returns:
        True if the expression is a zero constant, False otherwise
    """
    match e:
        case ast.Constant(value=value):
            match value:
                case ast.ConstInt(0) | ast.ConstUInt(0) | ast.ConstLong(0) | ast.ConstULong(0):
                    return True
                case _:
                    return False
        case _:
            return False


def is_scalar(t: Type | None) -> bool:
    """Return True when ``t`` is a scalar type (arithmetic or pointer).

    Void, array, and function types are not scalar per C11 sec. 6.2.5.

    Args:
        t: The C type to test, or ``None`` if the type is not yet known.

    Returns:
        ``True`` for arithmetic types and pointer types; ``False`` for
        ``VoidType``, ``Array``, ``FunType``, ``Structure``, and ``None``.
    """
    match t:
        case None | VoidType() | Array() | FunType() | Structure() | Union():
            return False
        case _:
            return True


def is_arithmetic(t: Type | None) -> bool:
    """Return True when ``t`` is an arithmetic type (integer or floating-point).

    Arithmetic types are the numeric types recognised by this compiler: all
    integer types (``IntType``, ``UIntType``, ``LongType``, ``ULongType``,
    ``Char``, ``SChar``, ``UChar``) and the floating-point type (``DoubleType``).
    Pointer, array, function, and void types are not arithmetic.

    Args:
        t: The C type to test, or ``None`` if the type is not yet known.

    Returns:
        ``True`` for all types recorded in ``_TYPE_SIZE``; ``False`` otherwise.
    """
    return t is not None and type(t) in _TYPE_SIZE


def is_integer(t: Type | None) -> bool:
    """Return True when ``t`` is an integer type (arithmetic and not floating-point).

    Integer types are the arithmetic types that are not ``DoubleType``:
    ``IntType``, ``UIntType``, ``LongType``, ``ULongType``, ``Char``, ``SChar``,
    and ``UChar``.  Only integer types are valid operands for bitwise operators,
    shift operators, and the modulus operator.

    Args:
        t: The C type to test, or ``None`` if the type is not yet known.

    Returns:
        ``True`` when ``t`` is in ``_TYPE_SIZE`` and is not ``DoubleType``;
        ``False`` otherwise.
    """
    return is_arithmetic(t) and not isinstance(t, DoubleType)


def is_lvalue(e: ast.Exp) -> bool:
    """Return True when ``e`` is an lvalue expression.

    An lvalue is an expression that designates a memory location and may appear
    on the left-hand side of an assignment or as the operand of ``&``.

    * ``Var``, ``Dereference``, ``Subscript``, and ``String`` are unconditionally
      lvalues.
    * ``Arrow`` (``ptr->member``) is always an lvalue because ``->`` implicitly
      dereferences a pointer.
    * ``Dot`` (``s.member``) is an lvalue only when its left operand is itself an
      lvalue - the check recurses until a non-lvalue or a base lvalue is found.
    * All other expression forms are rvalues.

    Args:
        e: The typed expression to test.

    Returns:
        ``True`` when ``e`` designates a modifiable memory location.
    """
    match e:
        case ast.Var() | ast.Dereference() | ast.Subscript() | ast.String():
            return True
        case ast.Arrow():
            return True
        case ast.Dot(structure=structure):
            return is_lvalue(structure)
        case _:
            return False


def is_complete(t: Type, struct_type_table: StructTypeTable) -> bool:
    """Return True when ``t`` is a complete type.

    ``VoidType`` is always incomplete.  A ``Structure`` type is complete only
    when its tag has an entry in ``struct_type_table`` (i.e. a definition with
    a member list has been seen).  All other types are complete.

    Args:
        t: The C type to test.
        struct_type_table: Mapping from resolved struct tags to ``StructEntry``
            layout records, used to decide whether a struct type is complete.

    Returns:
        ``False`` for ``VoidType`` and for ``Structure`` types whose tag is not
        yet in ``struct_type_table``; ``True`` for all other types.
    """
    match t:
        case VoidType():
            return False
        case Structure(tag=tag) | Union(tag=tag):
            return tag in struct_type_table
        case _:
            return True


def is_pointer_to_complete(t: Type | None, struct_type_table: StructTypeTable) -> bool:
    """Return True when ``t`` is a pointer to a complete type.

    Args:
        t: The C type to test, or ``None`` if the type is not yet known.
        struct_type_table: Forwarded to ``is_complete`` for struct completeness
            checks.

    Returns:
        ``True`` when ``t`` is a ``Pointer`` whose referenced type satisfies
        ``is_complete``; ``False`` otherwise.
    """
    match t:
        case Pointer(referenced=inner):
            return is_complete(inner, struct_type_table)
        case _:
            return False


def validate_type_specifier(t: Type, struct_type_table: StructTypeTable) -> None:
    """Recursively validate a type specifier for structural correctness.

    Checks that no array element type is incomplete (C11 sec. 6.7.6.2p1 - arrays
    of incomplete type are not allowed), and recurses into pointer, array, and
    function component types.

    Args:
        t: The C type to validate.
        struct_type_table: Forwarded to ``is_complete`` for struct completeness
            checks on array element types.

    Raises:
        SemanticError: If an array element type is incomplete.
    """
    match t:
        case Array(element=elem_t):
            if not is_complete(elem_t, struct_type_table):
                raise SemanticError("Illegal array of incomplete type")
            validate_type_specifier(elem_t, struct_type_table)
        case Pointer(referenced=referenced_t):
            validate_type_specifier(referenced_t, struct_type_table)
        case FunType(params=param_ts, ret=ret_t):
            for param_t in param_ts:
                validate_type_specifier(param_t, struct_type_table)
            validate_type_specifier(ret_t, struct_type_table)
        case _:
            return


def _zero_initializer(t: Type, struct_type_table: StructTypeTable) -> ast.Initializer:
    """Create a zero-valued, fully type-annotated ``Initializer`` for the given type.

    Used to pad aggregate initialisers that are shorter than their declared size
    (C11 sec. 6.7.9p21: elements not explicitly initialised receive zero-initialisation).

    Args:
        t: The C type for which a zero initializer is needed.
        struct_type_table: Mapping from resolved struct tags to ``StructEntry`` layout
            records; required to enumerate members when zero-initialising a struct.

    Returns:
        A ``SingleInit`` wrapping a zero-valued ``Constant`` (or ``Cast`` for pointers)
        for scalar types, a ``CompoundInit`` recursively filled with zero initialisers
        for ``Array`` types, or a ``CompoundInit`` of per-member zero initialisers for
        ``Structure`` types.

    Raises:
        SemanticError: If ``t`` is a ``FunType`` or otherwise uninitializable type.
    """
    match t:
        case IntType():
            return ast.SingleInit(ast.Constant(ast.ConstInt(0), exp_type=IntType()))
        case LongType():
            return ast.SingleInit(ast.Constant(ast.ConstLong(0), exp_type=LongType()))
        case LongLongType():
            return ast.SingleInit(ast.Constant(ast.ConstLongLong(0), exp_type=LongLongType()))
        case UIntType():
            return ast.SingleInit(ast.Constant(ast.ConstUInt(0), exp_type=UIntType()))
        case ULongType():
            return ast.SingleInit(ast.Constant(ast.ConstULong(0), exp_type=ULongType()))
        case ULongLongType():
            return ast.SingleInit(ast.Constant(ast.ConstULongLong(0), exp_type=ULongLongType()))
        case DoubleType():
            return ast.SingleInit(ast.Constant(ast.ConstDouble(0.0), exp_type=DoubleType()))
        case Char() | SChar():
            return ast.SingleInit(ast.Constant(ast.ConstChar(0), exp_type=t))
        case UChar():
            return ast.SingleInit(ast.Constant(ast.ConstUChar(0), exp_type=UChar()))
        case ShortType():
            return ast.SingleInit(ast.Constant(ast.ConstShort(0), exp_type=ShortType()))
        case UShortType():
            return ast.SingleInit(ast.Constant(ast.ConstUShort(0), exp_type=UShortType()))
        case Int128Type():
            return ast.SingleInit(ast.Constant(ast.ConstInt128(0), exp_type=Int128Type()))
        case UInt128Type():
            return ast.SingleInit(ast.Constant(ast.ConstUInt128(0), exp_type=UInt128Type()))
        case Pointer():
            null = ast.Constant(ast.ConstULong(0), exp_type=ULongType())
            return ast.SingleInit(ast.Cast(t, null, exp_type=t))
        case Array(element=elem_t, size=size):
            return ast.CompoundInit([_zero_initializer(elem_t, struct_type_table) for _ in range(size)])
        case Structure(tag=tag):
            struct_def = struct_type_table[tag]
            return ast.CompoundInit(
                [_zero_initializer(m.member_type, struct_type_table) for m in struct_def.members],
                init_type=Structure(tag=tag),
            )
        case Union(tag=tag):
            # C11 sec. 6.7.9p10: union zero-initialisation initialises the first named member.
            union_def = struct_type_table[tag]
            return ast.CompoundInit(
                [_zero_initializer(union_def.members[0].member_type, struct_type_table)],
                init_type=Union(tag=tag),
            )
        case _:
            raise SemanticError(f"Cannot create a zero initializer for type '{t}'.")


def _is_constant_init(init: ast.Initializer) -> bool:
    """Return ``True`` if every scalar leaf in ``init`` is an ``ast.Constant`` node.

    Used to validate that static-storage array initialisers contain only
    compile-time constant expressions before calling ``_typecheck_init``.

    Args:
        init: The initialiser to inspect - either a ``SingleInit`` or a
            ``CompoundInit``.

    Returns:
        ``True`` when every ``SingleInit`` expression is an ``ast.Constant``;
        ``False`` as soon as any scalar expression is not a bare constant.
    """
    match init:
        case ast.SingleInit(exp=ast.Constant()):
            return True
        case ast.SingleInit(exp=ast.String()):
            return True
        case ast.SingleInit():
            return False
        case ast.CompoundInit(initializers=sub_inits):
            return all(_is_constant_init(i) for i in sub_inits)
        case _:
            return False


def _make_static_init(t: Type | None, v: int | float) -> StaticInit:
    """Return the ``StaticInit`` element for a scalar value of the given C type.

    Zero-valued scalars are encoded as ``ZeroInit``; non-zero values become the
    typed init variant that matches the byte width of ``t``.

    Args:
        t: The C type determining the byte width and variant - ``IntType``,
            ``UIntType``, ``LongType``, ``ULongType``, ``DoubleType``,
            ``Char``, ``SChar``, or ``UChar``.
            ``None`` or an unrecognised type defaults to a 4-byte ``IntInit``
            / ``ZeroInit``.
        v: The numeric value (``int`` or ``float``) from the constant expression.

    Returns:
        A ``ZeroInit(n)``, ``IntInit``, ``LongInit``, ``DoubleInit``,
        ``CharInit``, or ``UCharInit`` node.
    """
    match t:
        case DoubleType():
            fv = float(v)
            return ZeroInit(8) if fv == 0.0 and math.copysign(1.0, fv) > 0 else DoubleInit(fv)
        case LongType() | ULongType() | LongLongType() | ULongLongType():
            return ZeroInit(8) if int(v) == 0 else LongInit(int(v))
        case Char() | SChar():
            return ZeroInit(1) if int(v) == 0 else CharInit(int(v))
        case UChar():
            return ZeroInit(1) if int(v) == 0 else UCharInit(int(v))
        case ShortType():
            return ZeroInit(2) if int(v) == 0 else ShortInit(int(v))
        case UShortType():
            return ZeroInit(2) if int(v) == 0 else UShortInit(int(v))
        case Int128Type():
            return ZeroInit(16) if int(v) == 0 else Int128Init(int(v))
        case UInt128Type():
            return ZeroInit(16) if int(v) == 0 else UInt128Init(int(v))
        case _:
            return ZeroInit(4) if int(v) == 0 else IntInit(int(v))


def _append_zero_pad(result: list[StaticInit], n: int) -> None:
    """Append ``n`` zero bytes to ``result``, coalescing with a trailing ``ZeroInit`` if present.

    Args:
        result: The accumulator list being built by ``_flatten_helper``.
        n: Number of zero bytes to append.
    """
    if result and isinstance(result[-1], ZeroInit):
        result[-1:] = [ZeroInit(result[-1].bytes + n)]
    else:
        result.append(ZeroInit(n))


def _string_array_static_init(s: bytes, array_size: int) -> list[StaticInit]:
    """Build the ``StaticInit`` list for a character array initialised from a string literal.

    The list always starts with a ``StringInit``.  When the array is large enough to hold
    the NUL terminator, ``null_terminated`` is ``True`` and the terminator byte is included
    in the ``StringInit``; any remaining unused bytes are represented by a trailing
    ``ZeroInit``.  When the array is exactly the length of the string (no room for NUL),
    ``null_terminated`` is ``False`` and the bytes are emitted as-is.

    Args:
        s: Raw byte content of the string literal, NUL terminator NOT included.
        array_size: Declared size of the target character array.

    Returns:
        A list of one or two ``StaticInit`` elements: ``[StringInit(...), ZeroInit(n)]``
        when padding is needed, or ``[StringInit(...)]`` when the array is exactly full.
    """
    null_terminated = len(s) + 1 <= array_size
    result: list[StaticInit] = [StringInit(s, null_terminated)]
    bytes_used = len(s) + (1 if null_terminated else 0)
    remaining = array_size - bytes_used
    if remaining > 0:
        result.append(ZeroInit(remaining))
    return result


class VariableBinding(NamedTuple):
    """A single entry in the variable map, recording a resolved name and its scope origin.

    Attributes:
        unique_name: The unique internal identifier assigned to the source-level variable
            (e.g. ``".Vx.0"``).  File-scope variables use their original source name.
        from_current_block: ``True`` if this binding was declared in the block currently
            being analysed; ``False`` if it was inherited from an enclosing scope or the
            file scope.  Used by ``resolve_variable_declaration`` to distinguish shadowing
            (permitted) from a same-block duplicate (an error).
        var_type: The declared C type of this variable.  Carried forward so that
            ``resolve_exp`` can annotate ``Var`` expression nodes with their type.
        linkage: The linkage classification for this variable.  Local variables carry
            ``Linkage.NONE``; ``static`` variables carry ``Linkage.INTERNAL``; file-scope
            and ``extern``-declared variables carry ``Linkage.EXTERNAL``.
    """

    unique_name: str
    from_current_block: bool
    var_type: Type
    linkage: Linkage = Linkage.NONE


class FunctionBinding(NamedTuple):
    """A single entry in the function map, tracking whether a definition has been seen.

    Functions are globally scoped and their names are never renamed.  Unlike
    variables, multiple declarations of the same function are permitted; only a
    second *definition* (a ``Function`` node with a non-``None`` body) is an
    error.  All declarations of the same function must agree on parameter count
    and types; a mismatch is a conflict error.

    Attributes:
        has_definition: ``True`` once a definition (body present) has been
            registered for this function name; ``False`` if only forward
            declarations have been seen so far.
        param_types: The ordered tuple of parameter types established by the
            first declaration seen.  Every subsequent declaration must match
            this tuple exactly.
        ret_type: The declared return type of this function.  Used by
            ``resolve_exp`` to annotate ``FunctionCall`` expression nodes.
        linkage: The linkage classification for this function.  Functions
            declared without ``static`` carry ``Linkage.EXTERNAL``; those
            declared with ``static`` carry ``Linkage.INTERNAL``.
    """

    has_definition: bool
    param_types: tuple[ast.Type, ...]
    ret_type: ast.Type
    linkage: Linkage = Linkage.EXTERNAL


class StructureBinding(NamedTuple):
    """A single entry in the structure map, recording a resolved tag and its scope origin.

    Attributes:
        new_tag: The unique internal identifier assigned to the source-level struct tag
            (e.g. ``".Sfoo.0"``).
        from_current_scope: ``True`` if this binding was declared in the block currently
            being analysed; ``False`` if it was inherited from an enclosing scope or the
            file scope.  Used to distinguish same-scope redeclarations (permitted, reuse
            existing ``new_tag``) from shadowing declarations in a nested scope.
    """

    new_tag: str
    from_current_scope: bool


class SemanticAnalyser:
    """Resolves variable references and validates function declarations across a single AST program.

    Each instance owns a fresh counter and should be used for exactly one
    ``_resolve_program`` call.  Constructing a new instance per program
    guarantees that unique suffixes start from zero regardless of how many
    programs are analysed within the same process, which keeps test output
    deterministic.

    Attributes:
        _counter: Monotonically increasing integer source for unique variable
            name suffixes.
        _loop_counter: Monotonically increasing integer source for unique loop
            label suffixes, kept separate from ``_counter`` so that loop
            identifiers are predictable in tests regardless of how many
            variables have been declared.
        _variable_map: Mapping from source-level variable names to
            ``VariableBinding`` values.  Each binding records the unique
            internal name, whether it was declared in the current block, and
            the variable's linkage.  File-scope variables accumulate in this
            map (with ``from_current_block=False``) as top-level ``VarDecl``
            nodes are processed; when a ``FunDecl`` definition is resolved the
            map is snapshotted before entering the body and restored afterward
            so that local variables declared inside the function are not visible
            to subsequent top-level declarations.  Outer-scope bindings are
            retained in the map with ``from_current_block=False`` so that
            variable lookups can still resolve them; when a compound block exits
            the map is restored to its saved snapshot.
        _function_map: Program-wide mapping from function names to
            ``FunctionBinding`` values.  Each binding records whether a
            definition has been seen and the function's linkage
            (``Linkage.INTERNAL`` for ``static`` functions; ``Linkage.EXTERNAL``
            otherwise).  Populated by a pre-pass in ``_resolve_program`` and
            extended when ``FunDecl`` block items are processed.  Never reset
            between functions.
        _symbol_table: Program-wide mapping from every resolved identifier to a
            ``SymbolTableEntry`` describing its storage attributes.  Functions
            receive ``FunAttr``; variables with static storage duration receive
            ``StaticAttr``; plain local variables receive no entry.  Consumed by
            downstream stages (TAC generation) to emit ``StaticVariable``
            constructs.
        _type_table: Mapping from every resolved identifier name to its declared
            C type.  Keyed by the *unique* name for local and static-local
            variables (e.g. ``".Vx.0"``), and by the original source name for
            file-scope variables and block-scope ``extern`` declarations.
            Populated by ``resolve_variable_declaration`` and
            ``_resolve_file_scope_variable_declaration`` so that
            ``_type_check_exp`` can annotate ``Var`` expression nodes with their
            type without needing to trace back through ``_variable_map``.
        _label_set: Set of all label names declared in the function currently
            being analysed.  Populated by a first pass before resolution begins
            so that forward ``goto`` targets can be validated.
        _loop_stack: Stack of identifiers for the loops currently enclosing the
            statement being resolved.  Used exclusively by ``Continue``:
            ``continue`` is only valid inside a loop, never a bare switch.
        _break_stack: Stack of identifiers for the loops *and* switches
            currently enclosing the statement being resolved.  The top entry is
            the target for any ``Break`` encountered.  An empty stack indicates
            that ``break`` would be invalid at the current position.
        _switch_stack: Stack of identifiers for the switches currently
            enclosing the statement being resolved.  The top entry is the
            enclosing switch identifier, which is attached to every ``Case``
            and ``Default`` encountered during resolution.  An empty stack
            indicates that ``case``/``default`` appear outside any switch,
            which is a semantic error.
        _case_values_stack: Parallel stack to ``_switch_stack``.  Each entry
            is the set of integer values already seen by ``Case`` labels within
            the corresponding switch, after truncating each constant to the type
            of the switch control expression.  Truncation is required because C
            compares case labels against the switch expression using the type of
            that expression - so ``case 34359738768l`` in an ``int`` switch is
            duplicate with ``case 400`` (both truncate to 400).
            Used to detect duplicate case values within a single switch statement.
        _switch_type_stack: Parallel stack to ``_switch_stack``.  Each entry is
            the C type of the corresponding switch control expression, used to
            truncate case constants before duplicate checking.
        _default_seen_stack: Parallel stack to ``_switch_stack``.  Each entry
            is ``True`` once a ``Default`` label has been encountered within the
            corresponding switch, and ``False`` until then.  Used to enforce the
            C rule that at most one ``default`` clause may appear per switch.
        _current_block_funs: Set of function names declared as block items in
            the innermost enclosing block scope.  Saved and cleared on every
            ``_enter_scope`` call and restored when that scope exits, mirroring
            the ``from_current_block`` flag on ``VariableBinding``.  Used to
            detect within-scope conflicts between a function declaration and a
            variable declaration sharing the same name.
        _file_scope_linkage: Persistent mapping from file-scope names to their
            established linkage.  Written by both file-scope variable declarations
            and block-scope ``extern`` declarations; never reset between functions.
            Enables cross-boundary linkage-conflict detection regardless of
            declaration order.
        _file_scope_definitions: Set of file-scope variable names for which a
            definition (a declaration carrying an initialiser) has already been
            seen.  A second definition of the same name is a constraint violation
            in C.  Tentative definitions (declarations without initialisers) do
            not add entries here and are always permitted to repeat.
        _all_unique_struct_tags: Program-wide set of every unique internal struct tag
            ever generated by ``_make_struct_tag`` (e.g. ``".Souter.1"``).  Used by
            ``_resolve_type`` to detect types that have already been resolved so that
            double-resolution (which occurs when ``_register_fun_declaration`` resolves
            param types and ``_resolve_function`` then passes those resolved types back
            through ``resolve_variable_declaration``) is treated as a no-op rather than
            an error.  Never cleared between scopes.
        _struct_type_table: Program-wide mapping from resolved struct tags to
            ``StructEntry`` layout records.  Populated by ``_typecheck_struct_declaration``
            whenever a struct definition with a non-empty member list is encountered.
            Consulted by ``type_size`` and ``type_alignment`` to compute the
            layout of struct-typed members and variables.  Consumed by downstream
            stages (TAC generation, code generation) via ``resolve_with_symbols``.
        _structure_map: Mapping from source-level struct tag names to
            ``StructureBinding`` values.  Each binding records the unique internal
            tag (e.g. ``".Sfoo.0"``) and whether the struct was declared in the
            current scope.  File-scope struct declarations are stored with
            ``from_current_scope=False``; block-scope declarations with ``True``.
            When a new scope is entered (``_enter_scope``) all existing bindings
            are marked ``from_current_scope=False`` so that inner scopes may shadow
            them.  When a declaration is seen for a tag that is already bound in
            the current scope the existing ``new_tag`` is reused (compatible
            redeclaration); otherwise a fresh unique tag is generated.
    """

    def __init__(self) -> None:
        """Initialise the analyser with fresh counters and empty symbol tables."""
        self._counter: count[int] = count()
        self._loop_counter: count[int] = count()
        self._variable_map: dict[str, VariableBinding] = {}
        self._function_map: dict[str, FunctionBinding] = {}
        self._structure_map: dict[str, StructureBinding] = {}
        self._union_map: dict[str, StructureBinding] = {}
        self._symbol_table: SymbolTable = {}
        self._type_table: dict[str, Type] = {}
        self._struct_type_table: StructTypeTable = {}
        self._all_unique_struct_tags: set[str] = set()
        self._all_unique_union_tags: set[str] = set()
        self._file_scope_linkage: dict[str, Linkage] = {}
        self._file_scope_definitions: set[str] = set()
        self._current_block_funs: set[str] = set()
        self._current_return_type: Type | None = None
        self._label_set: set[str] = set()
        self._loop_stack: list[str] = []
        self._break_stack: list[str] = []
        self._switch_stack: list[str] = []
        self._case_values_stack: list[set[int]] = []
        self._switch_type_stack: list[Type] = []
        self._default_seen_stack: list[bool] = []

    def _make_unique(self, name: str) -> str:
        """Generate a unique internal name for a source-level variable.

        Produces identifiers of the form ``.Vx.0``, ``.Vx.1``, etc.  The
        leading ``.V`` prefix ensures the name cannot collide with any valid C
        identifier (C identifiers may not begin with a dot).

        Args:
            name: The source-level variable name to uniquify.

        Returns:
            A string of the form ``".V<name>.<n>"`` where ``<n>`` is the next
            value drawn from ``_counter``.
        """
        return f".V{name}.{next(self._counter)}"

    def _make_loop_label(self, kind: str) -> str:
        """Generate a unique identifier for a loop construct.

        Produces identifiers of the form ``.L.while.0``, ``.L.for.1``,
        ``.L.do.2``, etc.  The ``.L.`` prefix distinguishes loop labels from
        the ``.V`` prefix used for renamed variables.  The ``kind`` segment
        names the construct for readability in TAC and assembly output.

        Args:
            kind: A short string identifying the loop type, e.g. ``"while"``,
                ``"for"``, or ``"do"``.

        Returns:
            A string of the form ``".L.<kind>.<n>"`` where ``<n>`` is the next
            value drawn from ``_loop_counter``.
        """
        return f".L.{kind}.{next(self._loop_counter)}"

    def _make_string_name(self) -> str:
        """Generate a globally unique, assembler-safe name for a hidden string-literal constant.

        Produces identifiers of the form ``.Lstr.0``, ``.Lstr.1``, etc., using the
        same counter as ``_make_unique`` to guarantee uniqueness across the whole
        translation unit regardless of how many other symbols have been named.

        Returns:
            A string of the form ``".Lstr.<n>"`` where ``<n>`` is the next value
            drawn from ``_counter``.
        """
        return f".Lstr.{next(self._counter)}"

    def _make_struct_tag(self, tag: str) -> str:
        """Generate a unique internal tag name for a source-level struct tag.

        Produces identifiers of the form ``.Sfoo.0``, ``.Sfoo.1``, etc.  The
        leading ``.S`` prefix ensures the name cannot collide with any valid C
        identifier.

        Args:
            tag: The source-level struct tag to uniquify.

        Returns:
            A string of the form ``".S<tag>.<n>"`` where ``<n>`` is the next
            value drawn from ``_counter``.
        """
        new_tag = f".S{tag}.{next(self._counter)}"
        self._all_unique_struct_tags.add(new_tag)
        return new_tag

    def _make_union_tag(self, tag: str) -> str:
        """Generate a unique internal tag name for a source-level union tag.

        Produces identifiers of the form ``.Ufoo.0``, ``.Ufoo.1``, etc.  The
        leading ``.U`` prefix ensures the name cannot collide with any valid C
        identifier.

        Args:
            tag: The source-level union tag to uniquify.

        Returns:
            A string of the form ``".U<tag>.<n>"`` where ``<n>`` is the next
            value drawn from ``_counter``.
        """
        new_tag = f".U{tag}.{next(self._counter)}"
        self._all_unique_union_tags.add(new_tag)
        return new_tag

    def _resolve_type(self, t: Type) -> Type:
        """Recursively resolve a type specifier, replacing struct tags with their unique internal tags.

        ``Structure`` nodes whose tag is present in ``_structure_map`` are replaced
        by ``Structure(unique_tag)``; a tag absent from the map raises ``SemanticError``.
        ``Pointer``, ``Array``, and ``FunType`` nodes are resolved recursively.
        All other types are returned unchanged.

        Args:
            t: The raw C type as produced by the parser (possibly containing
                source-level struct tags), or an already-resolved type (idempotent
                when the tag is a previously generated unique internal tag).

        Returns:
            A new ``Type`` node with every ``Structure`` tag replaced by its
            unique internal identifier, or ``t`` unchanged when ``t`` contains
            only already-resolved unique tags.

        Raises:
            SemanticError: If a ``Structure`` tag has not been declared before
                use and is not a known unique internal tag (in the algorithm).
        """
        match t:
            case Structure(tag=tag):
                u_binding = self._union_map.get(tag)
                if u_binding is not None and u_binding.from_current_scope:
                    raise SemanticError(f"Tag '{tag}' already declared as a union in this scope.")
                binding = self._structure_map.get(tag)
                if binding is not None:
                    return Structure(binding.new_tag)
                if tag in self._all_unique_struct_tags:
                    return t  # already resolved - idempotent
                raise SemanticError(f"Use of undeclared struct tag '{tag}'")
            case Union(tag=tag):
                s_binding = self._structure_map.get(tag)
                if s_binding is not None and s_binding.from_current_scope:
                    raise SemanticError(f"Tag '{tag}' already declared as a struct in this scope.")
                binding = self._union_map.get(tag)
                if binding is not None:
                    return Union(binding.new_tag)
                if tag in self._all_unique_union_tags:
                    return t  # already resolved - idempotent
                raise SemanticError(f"Use of undeclared union tag '{tag}'")
            case Pointer(referenced=ref_t):
                return Pointer(self._resolve_type(ref_t))
            case Array(element=elem_t, size=size):
                return Array(self._resolve_type(elem_t), size)
            case FunType(params=param_ts, ret=ret_t):
                return FunType(tuple(self._resolve_type(pt) for pt in param_ts), self._resolve_type(ret_t))
            case _:
                return t

    def _is_member_type_complete(self, t: Type) -> bool:
        """Return ``True`` when ``t`` is a type that may be used as a struct member.

        A type is eligible when it is not void, not a function type, not an
        incomplete struct type (not yet in ``_struct_type_table``), and - for
        array types - when the element type is recursively eligible.

        Args:
            t: The resolved C type to test.

        Returns:
            ``False`` for ``VoidType``, ``FunType``, incomplete ``Structure``,
            or arrays thereof; ``True`` for all other types.
        """
        match t:
            case VoidType() | FunType():
                return False
            case Structure(tag=tag) | Union(tag=tag):
                return tag in self._struct_type_table
            case Array(element=elem_t):
                return self._is_member_type_complete(elem_t)
            case _:
                return True

    def _validate_struct_definition(self, struct_decl: StructDeclaration) -> None:
        """Validate the member list of a struct definition.

        Checks that no two members share the same name and that every
        member's type is eligible (complete, not void, not a function type).

        Args:
            struct_decl: The resolved ``StructDeclaration`` whose members are
                to be validated.  ``struct_decl.members`` must be non-empty.

        Raises:
            SemanticError: If any member name is duplicated within the struct,
                or if any member has an incomplete or invalid type.
        """
        seen: set[str] = set()
        for member in struct_decl.members:
            if member.member_name in seen:
                raise SemanticError(f"Duplicate member '{member.member_name}' in struct '{struct_decl.tag}'")
            seen.add(member.member_name)
            if not self._is_member_type_complete(member.member_type):
                raise SemanticError(
                    f"Member '{member.member_name}' of struct '{struct_decl.tag}' has an incomplete or invalid type"
                )
            validate_type_specifier(member.member_type, self._struct_type_table)

    def _typecheck_struct_declaration(self, struct_decl: StructDeclaration) -> None:
        """Validate a struct definition and record its layout in ``_struct_type_table``.

        Implements the type-checking algorithm for struct declarations:

        * Declarations without a member list are silently ignored - they either
          forward-declare an incomplete type or redundantly name a type that was
          already defined.
        * ``_validate_struct_definition`` checks for duplicate member names and
          incomplete member types.
        * Member offsets are computed by rounding the running size up to the
          member's alignment before placing each member.
        * The final struct size is rounded up to the struct's own alignment
          (C11 sec. 6.7.2.1p15 - trailing padding).
        * The completed ``StructEntry`` is stored in ``_struct_type_table``
          keyed by the unique internal tag.

        Args:
            struct_decl: The resolved ``StructDeclaration`` (tag is a unique
                internal identifier; member types are fully resolved).

        Raises:
            SemanticError: If the struct tag already has a complete definition in
                the current scope; if ``_validate_struct_definition`` finds a
                violation; or if any member has a type whose size or alignment
                cannot be determined (e.g. an incomplete struct type used as a
                value member).
        """
        if not struct_decl.members:  # forward declaration - nothing to do
            return
        if struct_decl.tag in self._struct_type_table:
            raise SemanticError(f"Redefinition of struct '{struct_decl.tag}'.")
        self._validate_struct_definition(struct_decl)  #
        member_entries: list[MemberEntry] = []  #
        struct_size = 0
        struct_alignment = 1
        for member in struct_decl.members:
            member_align = type_alignment(member.member_type, self._struct_type_table)
            member_offset = _round_up(struct_size, member_align)  #
            member_entries.append(MemberEntry(member.member_name, member.member_type, member_offset))  #
            struct_alignment = max(struct_alignment, member_align)
            struct_size = member_offset + type_size(member.member_type, self._struct_type_table)
        struct_size = _round_up(struct_size, struct_alignment)  #
        struct_def = StructEntry(alignment=struct_alignment, size=struct_size, members=tuple(member_entries))
        self._struct_type_table[struct_decl.tag] = struct_def  #

    def _validate_union_definition(self, union_decl: UnionDeclaration) -> None:
        """Validate the member list of a union definition.

        Checks that no two members share the same name and that every
        member's type is eligible (complete, not void, not a function type).

        Args:
            union_decl: The resolved ``UnionDeclaration`` whose members are
                to be validated.  ``union_decl.members`` must be non-empty.

        Raises:
            SemanticError: If any member name is duplicated within the union,
                or if any member has an incomplete or invalid type.
        """
        seen: set[str] = set()
        for member in union_decl.members:
            if member.member_name in seen:
                raise SemanticError(f"Duplicate member '{member.member_name}' in union '{union_decl.tag}'")
            seen.add(member.member_name)
            if not self._is_member_type_complete(member.member_type):
                raise SemanticError(
                    f"Member '{member.member_name}' of union '{union_decl.tag}' has an incomplete or invalid type"
                )
            validate_type_specifier(member.member_type, self._struct_type_table)

    def _typecheck_union_declaration(self, union_decl: UnionDeclaration) -> None:
        """Validate a union definition and record its layout in ``_struct_type_table``.

        Union layout (C11 sec. 6.7.2.1p16): all members start at offset 0; size
        equals the largest member size rounded up to the union's alignment;
        alignment equals the maximum member alignment.

        * Declarations without a member list are silently ignored.
        * ``_validate_union_definition`` checks for duplicate names and bad types.
        * Every member entry records offset 0.
        * Track max alignment and max raw member size.
        * Round final size up to alignment (trailing padding).
        * Store in ``_struct_type_table`` keyed by the unique internal tag.

        Args:
            union_decl: The resolved ``UnionDeclaration`` (tag is a unique
                internal identifier; member types are fully resolved).

        Raises:
            SemanticError: If the union tag already has a complete definition;
                if ``_validate_union_definition`` finds a violation.
        """
        if not union_decl.members:  # forward declaration - nothing to do
            return
        if union_decl.tag in self._struct_type_table:
            raise SemanticError(f"Redefinition of union '{union_decl.tag}'.")
        self._validate_union_definition(union_decl)  #
        member_entries: list[MemberEntry] = []
        union_alignment = 1
        union_size = 0
        for member in union_decl.members:
            member_align = type_alignment(member.member_type, self._struct_type_table)
            member_size = type_size(member.member_type, self._struct_type_table)
            member_entries.append(MemberEntry(member.member_name, member.member_type, 0))  #
            union_alignment = max(union_alignment, member_align)  #
            union_size = max(union_size, member_size)
        union_size = _round_up(union_size, union_alignment)  #
        union_def = StructEntry(alignment=union_alignment, size=union_size, members=tuple(member_entries))
        self._struct_type_table[union_decl.tag] = union_def  #

    def _collect_labels_in_statement(self, statement: ast.Statement) -> None:
        """Recursively collect label names from a statement into ``_label_set``.

        Raises ``SemanticError`` immediately upon finding a duplicate label name.
        Recurses into ``If``, ``Label``, ``Compound``, and all loop body nodes
        so that labels nested at any depth are captured.

        Args:
            statement: The statement node to walk.

        Raises:
            SemanticError: If a label name already present in ``_label_set`` is
                encountered, indicating a duplicate within the current function.
        """
        match statement:
            case ast.Label(name=name, statement=inner):
                if name in self._label_set:
                    raise SemanticError(f"Duplicate label '{name}' in function.")
                self._label_set.add(name)
                self._collect_labels_in_statement(inner)
            case ast.If(then=then_stmt, else_=else_stmt):
                self._collect_labels_in_statement(then_stmt)
                if else_stmt is not None:
                    self._collect_labels_in_statement(else_stmt)
            case ast.Compound(body=body):
                for item in body:
                    match item:
                        case ast.S(statement=inner_stmt):
                            self._collect_labels_in_statement(inner_stmt)
            case ast.While(body=loop_body) | ast.DoWhile(body=loop_body):
                self._collect_labels_in_statement(loop_body)
            case ast.For(body=loop_body):
                self._collect_labels_in_statement(loop_body)
            case ast.Switch(body=switch_body):
                self._collect_labels_in_statement(switch_body)
            case ast.Case(statement=inner_stmt) | ast.Default(statement=inner_stmt):
                self._collect_labels_in_statement(inner_stmt)
            case _:
                pass

    def _collect_labels(self, body: tuple[ast.BlockItem, ...]) -> None:
        """First-pass walk over a function body to populate ``_label_set``.

        Only statement block items are walked; declaration items contain no
        labels.  Delegates to ``_collect_labels_in_statement`` for each
        statement found.

        Args:
            body: The sequence of block items forming the function body.

        Raises:
            SemanticError: If any duplicate label name is found within ``body``.
        """
        for item in body:
            match item:
                case ast.S(statement=statement):
                    self._collect_labels_in_statement(statement)

    def resolve_variable_declaration(self, var_decl: ast.VariableDeclaration) -> ast.VariableDeclaration:  # noqa: C901
        """Resolve a block-scope variable declaration against the current symbol table.

        ``extern`` declarations keep their original source name and are recorded
        with ``Linkage.EXTERNAL``; an initialiser is forbidden.  All other
        declarations (plain or ``static``) are renamed to a unique internal
        identifier.  ``static`` declarations are recorded with
        ``Linkage.INTERNAL``; plain declarations with ``Linkage.NONE``.

        If the same name has already been declared in the *current* block (i.e.
        the existing binding has ``from_current_block=True``), a
        ``SemanticError`` is raised.  When the existing binding has linkage and
        the new declaration does not (or vice versa), the error message
        identifies the linkage conflict specifically; otherwise it reports a
        plain duplicate.

        This method is *not* used for file-scope variable declarations; those
        are handled by ``_resolve_file_scope_variable_declaration``.

        Args:
            var_decl: The ``VariableDeclaration`` AST node to resolve.

        Returns:
            ``var_decl`` unchanged for ``extern`` declarations.  A new
            ``VariableDeclaration`` with a unique internal name (and resolved
            ``init``) for all other declarations.

        Raises:
            SemanticError: If ``var_decl.var_type`` is ``VoidType`` (variables
                may not be declared with type ``void``); if ``var_decl`` is an
                ``extern`` declaration with an initialiser; if the same name was
                declared with conflicting linkage in the same block; if the same
                name was already declared in this block; or if a non-``extern``
                name conflicts with a same-scope function declaration.
        """
        var_decl = ast.VariableDeclaration(
            var_decl.name, var_decl.init, self._resolve_type(var_decl.var_type), var_decl.storage_class
        )
        validate_type_specifier(var_decl.var_type, self._struct_type_table)
        if isinstance(var_decl.var_type, VoidType):
            raise SemanticError(f"Variable '{var_decl.name}' declared with incomplete type 'void'.")
        if (
            var_decl.storage_class != StorageClass.Extern
            and isinstance(var_decl.var_type, (Structure, Union))
            and not is_complete(var_decl.var_type, self._struct_type_table)
        ):
            raise SemanticError(f"Variable '{var_decl.name}' declared with incomplete aggregate type.")
        binding = self._variable_map.get(var_decl.name)
        if var_decl.storage_class == StorageClass.Extern:
            if var_decl.init is not None:
                raise SemanticError(f"Block-scope 'extern' variable '{var_decl.name}' may not have an initialiser.")
            if var_decl.name in self._function_map:
                raise SemanticError(
                    f"'{var_decl.name}' redeclared as a variable but was previously declared as a function."
                )
            if binding is not None and binding.from_current_block:
                if binding.unique_name == var_decl.name:
                    return var_decl  # repeated extern declaration (EXTERNAL or INTERNAL) - permitted
                # Prior declaration was a renamed local or static local - block-scope extern conflicts.
                raise SemanticError(f"'{var_decl.name}' previously declared without linkage in this scope.")
            prior_linkage = self._file_scope_linkage.get(var_decl.name)
            # Inherit whichever linkage was already established; default to EXTERNAL when unseen.
            effective_linkage = prior_linkage if prior_linkage is not None else Linkage.EXTERNAL
            existing_type = self._type_table.get(var_decl.name)
            if existing_type is not None and existing_type != var_decl.var_type:
                raise SemanticError(f"'{var_decl.name}' redeclared with a conflicting type.")
            self._file_scope_linkage[var_decl.name] = effective_linkage
            self._variable_map[var_decl.name] = VariableBinding(
                var_decl.name,
                from_current_block=True,
                var_type=var_decl.var_type,
                linkage=effective_linkage,
            )
            self._type_table[var_decl.name] = var_decl.var_type
            # Only add a symbol table entry when the name hasn't been seen yet; file-scope
            # declarations take precedence and should not be downgraded to NoInitializer.
            if var_decl.name not in self._symbol_table:
                self._symbol_table[var_decl.name] = SymbolTableEntry(
                    t=var_decl.var_type,
                    attrs=StaticAttr(init=NoInitializer(), globl=True),
                )
            return var_decl
        else:
            if binding is not None and binding.from_current_block:
                if binding.linkage == Linkage.EXTERNAL:
                    raise SemanticError(f"'{var_decl.name}' previously declared with linkage in this scope.")
                raise SemanticError(f"Duplicate variable declaration: {var_decl.name}!")
            if var_decl.name in self._current_block_funs:
                raise SemanticError(
                    f"'{var_decl.name}' redeclared as a variable but was previously declared as a function "
                    "in this scope."
                )
            unique_name = self._make_unique(var_decl.name)
            self._type_table[unique_name] = var_decl.var_type
            linkage = Linkage.INTERNAL if var_decl.storage_class == StorageClass.Static else Linkage.NONE
            self._variable_map[var_decl.name] = VariableBinding(
                unique_name,
                from_current_block=True,
                var_type=var_decl.var_type,
                linkage=linkage,
            )
            raw_init_exp: ast.Exp | None = var_decl.init.exp if isinstance(var_decl.init, ast.SingleInit) else None
            if var_decl.init is not None and var_decl.storage_class == StorageClass.Static:
                if isinstance(var_decl.var_type, Pointer):
                    if isinstance(raw_init_exp, ast.String):
                        if not isinstance(var_decl.var_type.referenced, Char):
                            raise SemanticError(
                                f"Cannot initialize pointer to '{var_decl.var_type.referenced}' with a string "
                                "literal; string literals have type 'char *'."
                            )
                    elif not (
                        isinstance(raw_init_exp, ast.Constant)
                        or (isinstance(raw_init_exp, ast.AddrOf) and isinstance(raw_init_exp.exp, ast.Var))
                    ):
                        raise SemanticError(
                            f"Initialiser of static pointer '{var_decl.name}' must be a null pointer constant "
                            "or the address of a static variable."
                        )
                elif isinstance(var_decl.var_type, Array):
                    if not _is_constant_init(var_decl.init):
                        raise SemanticError(
                            f"Initialiser of static array '{var_decl.name}' must consist of constant expressions."
                        )
                elif isinstance(var_decl.var_type, (Structure, Union)):
                    if not _is_constant_init(var_decl.init):
                        raise SemanticError(
                            f"Initialiser of static aggregate '{var_decl.name}' must consist of constant expressions."
                        )
                elif not isinstance(raw_init_exp, ast.Constant):
                    raise SemanticError(
                        f"Initialiser of 'static' variable '{var_decl.name}' must be a constant expression."
                    )
            init: ast.Initializer | None = (
                self._typecheck_init(var_decl.var_type, var_decl.init) if var_decl.init is not None else None
            )
            if var_decl.storage_class == StorageClass.Static:
                if var_decl.init is None:
                    static_init: InitialValue = Tentative()
                elif isinstance(var_decl.var_type, Pointer):
                    assert raw_init_exp is not None
                    if isinstance(raw_init_exp, ast.String):
                        str_name = self._make_string_name()
                        self._symbol_table[str_name] = SymbolTableEntry(
                            t=Array(element=Char(), size=len(raw_init_exp.string) + 1),
                            attrs=ConstantAttr(StringInit(raw_init_exp.string, True)),
                        )
                        static_init = PointerInit(str_name)
                    else:
                        static_init = _static_pointer_init(raw_init_exp, self._variable_map)
                elif isinstance(var_decl.var_type, Array):
                    if isinstance(init, ast.SingleInit) and isinstance(init.exp, ast.String):
                        static_init = Initial(tuple(_string_array_static_init(init.exp.string, var_decl.var_type.size)))
                    else:
                        assert isinstance(init, ast.CompoundInit)
                        static_init = Initial(tuple(self._flatten_static_init(init)))
                elif isinstance(var_decl.var_type, (Structure, Union)):
                    assert isinstance(init, ast.CompoundInit)
                    static_init = Initial(tuple(self._flatten_static_init(init)))
                else:
                    assert isinstance(init, ast.SingleInit)
                    static_init = Initial((self._exp_to_static_init(init.exp),))
                self._symbol_table[unique_name] = SymbolTableEntry(
                    t=var_decl.var_type, attrs=StaticAttr(init=static_init, globl=False)
                )
            else:
                self._symbol_table[unique_name] = SymbolTableEntry(t=var_decl.var_type, attrs=LocalAttr())
            return ast.VariableDeclaration(unique_name, init, var_decl.var_type, var_decl.storage_class)

    def _resolve_file_scope_variable_declaration(self, var_decl: ast.VariableDeclaration) -> ast.VariableDeclaration:  # noqa: C901
        """Record a file-scope variable declaration in the identifier table.

        File-scope variables are *not* renamed - they keep their original source
        names throughout the compilation pipeline.  The binding is added to
        ``_variable_map`` with ``from_current_block=False`` so that subsequent
        function bodies can reference it.

        Linkage is determined by storage class: ``static`` yields
        ``Linkage.INTERNAL``; everything else (``extern`` or no specifier) yields
        ``Linkage.EXTERNAL``.

        Args:
            var_decl: The ``VariableDeclaration`` AST node from a top-level
                ``VarDecl`` block.

        Returns:
            ``var_decl`` unchanged - file-scope variables are not renamed.

        Raises:
            SemanticError: If ``var_decl.var_type`` is ``VoidType`` (variables
                may not be declared with type ``void``); if ``var_decl.name``
                was previously declared as a function (conflicting entity kind on
                the same external name); if the initialiser is not a constant
                expression (file-scope variables have static storage duration and
                must be initialised at compile time); if a prior declaration of
                the same name carried a different C type; or if linkage conflicts
                with a prior declaration.
        """
        var_decl = ast.VariableDeclaration(
            var_decl.name, var_decl.init, self._resolve_type(var_decl.var_type), var_decl.storage_class
        )
        validate_type_specifier(var_decl.var_type, self._struct_type_table)
        if isinstance(var_decl.var_type, VoidType):
            raise SemanticError(f"Variable '{var_decl.name}' declared with incomplete type 'void'.")
        if (
            var_decl.storage_class != StorageClass.Extern
            and isinstance(var_decl.var_type, (Structure, Union))
            and not is_complete(var_decl.var_type, self._struct_type_table)
        ):
            raise SemanticError(f"Variable '{var_decl.name}' declared with incomplete aggregate type.")
        if var_decl.name in self._function_map:
            raise SemanticError(
                f"'{var_decl.name}' redeclared as a variable but was previously declared as a function."
            )
        file_raw_exp: ast.Exp | None = var_decl.init.exp if isinstance(var_decl.init, ast.SingleInit) else None
        if var_decl.init is not None:
            if isinstance(var_decl.var_type, Pointer):
                if isinstance(file_raw_exp, ast.String):
                    if not isinstance(var_decl.var_type.referenced, Char):
                        raise SemanticError(
                            f"Cannot initialize pointer to '{var_decl.var_type.referenced}' with a string "
                            "literal; string literals have type 'char *'."
                        )
                elif not (
                    isinstance(file_raw_exp, ast.Constant)
                    or (isinstance(file_raw_exp, ast.AddrOf) and isinstance(file_raw_exp.exp, ast.Var))
                ):
                    raise SemanticError(
                        f"Initialiser of file-scope pointer '{var_decl.name}' must be a null pointer constant "
                        "or the address of a static variable."
                    )
            elif isinstance(var_decl.var_type, Array):
                if not _is_constant_init(var_decl.init):
                    raise SemanticError(
                        f"Initialiser of file-scope array '{var_decl.name}' must consist of constant expressions."
                    )
            elif isinstance(var_decl.var_type, (Structure, Union)):
                if not _is_constant_init(var_decl.init):
                    raise SemanticError(
                        f"Initialiser of file-scope aggregate '{var_decl.name}' must consist of constant expressions."
                    )
            elif not isinstance(file_raw_exp, ast.Constant):
                raise SemanticError(
                    f"Initialiser of file-scope variable '{var_decl.name}' must be a constant expression."
                )
            if file_raw_exp is not None and not isinstance(file_raw_exp, ast.String):
                _convert_by_assignment(self._type_check_exp(file_raw_exp), var_decl.var_type)
        if var_decl.init is not None:
            if var_decl.name in self._file_scope_definitions:
                raise SemanticError(f"Duplicate definition of file-scope variable '{var_decl.name}'.")
            self._file_scope_definitions.add(var_decl.name)
        prior_linkage = self._file_scope_linkage.get(var_decl.name)
        if var_decl.storage_class == StorageClass.Static:
            new_linkage = Linkage.INTERNAL
        elif var_decl.storage_class == StorageClass.Extern and prior_linkage is not None:
            new_linkage = prior_linkage  # extern inherits whatever linkage was already established
        else:
            new_linkage = Linkage.EXTERNAL
        if prior_linkage is not None and prior_linkage != new_linkage:
            raise SemanticError(
                f"'{var_decl.name}' previously declared with "
                f"{'external' if prior_linkage == Linkage.EXTERNAL else 'internal'} linkage; "
                f"cannot redeclare with {'internal' if new_linkage == Linkage.INTERNAL else 'external'} linkage."
            )
        existing_type = self._type_table.get(var_decl.name)
        if existing_type is not None and existing_type != var_decl.var_type:
            raise SemanticError(f"'{var_decl.name}' redeclared with a conflicting type.")
        self._file_scope_linkage[var_decl.name] = new_linkage
        self._variable_map[var_decl.name] = VariableBinding(
            var_decl.name,
            from_current_block=False,
            var_type=var_decl.var_type,
            linkage=new_linkage,
        )
        self._type_table[var_decl.name] = var_decl.var_type
        globl = new_linkage == Linkage.EXTERNAL
        existing_entry = self._symbol_table.get(var_decl.name)
        if var_decl.init is not None:
            if isinstance(var_decl.var_type, Pointer):
                assert file_raw_exp is not None
                if isinstance(file_raw_exp, ast.String):
                    str_name = self._make_string_name()
                    self._symbol_table[str_name] = SymbolTableEntry(
                        t=Array(element=Char(), size=len(file_raw_exp.string) + 1),
                        attrs=ConstantAttr(StringInit(file_raw_exp.string, True)),
                    )
                    file_init: InitialValue = PointerInit(str_name)
                else:
                    file_init = _static_pointer_init(file_raw_exp, self._variable_map)
            elif isinstance(var_decl.var_type, Array):
                file_typed_init = self._typecheck_init(var_decl.var_type, var_decl.init)
                if isinstance(file_typed_init, ast.SingleInit) and isinstance(file_typed_init.exp, ast.String):
                    file_init = Initial(
                        tuple(_string_array_static_init(file_typed_init.exp.string, var_decl.var_type.size))
                    )
                else:
                    assert isinstance(file_typed_init, ast.CompoundInit)
                    file_init = Initial(tuple(self._flatten_static_init(file_typed_init)))
            elif isinstance(var_decl.var_type, (Structure, Union)):
                file_typed_init = self._typecheck_init(var_decl.var_type, var_decl.init)
                assert isinstance(file_typed_init, ast.CompoundInit)
                file_init = Initial(tuple(self._flatten_static_init(file_typed_init)))
            else:
                file_typed_init = self._typecheck_init(var_decl.var_type, var_decl.init)
                assert isinstance(file_typed_init, ast.SingleInit)
                file_init = Initial((self._exp_to_static_init(file_typed_init.exp),))
            self._symbol_table[var_decl.name] = SymbolTableEntry(
                t=var_decl.var_type,
                attrs=StaticAttr(init=file_init, globl=globl),
            )
        elif var_decl.storage_class == StorageClass.Extern:
            # A non-defining extern declaration - add a NoInitializer entry only when the name
            # is entirely new; any existing Initial or Tentative entry takes precedence.
            if existing_entry is None:
                self._symbol_table[var_decl.name] = SymbolTableEntry(
                    t=var_decl.var_type,
                    attrs=StaticAttr(init=NoInitializer(), globl=globl),
                )
        else:
            # Tentative definition - upgrade from NoInitializer but never downgrade from Initial.
            if (
                existing_entry is None
                or isinstance(existing_entry.attrs, StaticAttr)
                and isinstance(existing_entry.attrs.init, NoInitializer)
            ):
                self._symbol_table[var_decl.name] = SymbolTableEntry(
                    t=var_decl.var_type, attrs=StaticAttr(init=Tentative(), globl=globl)
                )
        return var_decl

    def _validate_params_unique(self, fun: ast.Function) -> None:
        """Verify that no two parameters in a function declaration share the same name.

        Parameter names in a declaration need not be identical to those in the
        matching definition, but within any single declaration or definition they
        must be distinct.  This check applies equally to forward declarations
        (``body is None``) and full definitions.

        Args:
            fun: The ``Function`` node whose parameter list is to be validated.

        Raises:
            SemanticError: If two or more parameters in ``fun.params`` carry the
                same source-level name.
        """
        seen: set[str] = set()
        for param in fun.params:
            if param.name in seen:
                raise SemanticError(f"Duplicate parameter name '{param.name}' in declaration of '{fun.name}'.")
            seen.add(param.name)

    def _register_fun_declaration(self, fun: ast.Function) -> ast.Function:
        """Register a function declaration or definition in the program-wide function map.

        Records the function name in ``_function_map``.  A function may be
        declared any number of times; a definition (``fun.body is not None``)
        may appear at most once - a second definition raises ``SemanticError``.

        Args:
            fun: The ``Function`` node whose name is to be registered.

        Returns:
            The (possibly adjusted) ``Function`` node, with any ``Array``-typed
            parameters replaced by their equivalent ``Pointer`` types.

        Raises:
            SemanticError: If the function's declared return type is an ``Array``
                (functions may not return arrays in C); if any parameter has type
                ``void`` (parameters may not be declared with type ``void``);
                if ``fun`` is a definition and a definition for the same name has
                already been registered; if the parameter types of this
                declaration conflict with a prior declaration of the same name;
                if ``fun.name`` was already declared as a variable in the current
                block scope; or if ``fun.name`` was previously declared as an
                ``extern`` variable at any scope level (conflicting entity kind on
                the same external name).
        """
        fun = ast.Function(
            name=fun.name,
            params=tuple(ast.Parameter(self._resolve_type(p.param_type), p.name) for p in fun.params),
            body=fun.body,
            fun_type=self._resolve_type(fun.fun_type),
            globl=fun.globl,
            storage_class=fun.storage_class,
            source_file=fun.source_file,
            source_line=fun.source_line,
        )
        if isinstance(fun.fun_type, FunType) and isinstance(fun.fun_type.ret, Array):
            raise SemanticError(f"Function '{fun.name}' cannot return an array type.")
        validate_type_specifier(fun.fun_type, self._struct_type_table)
        adjusted_params: list[ast.Parameter] = []
        any_adjusted = False
        for param in fun.params:
            if isinstance(param.param_type, Array):
                adjusted_params.append(ast.Parameter(param_type=Pointer(param.param_type.element), name=param.name))
                any_adjusted = True
            else:
                adjusted_params.append(param)
        if any_adjusted:
            adjusted_param_types = tuple(p.param_type for p in adjusted_params)
            new_fun_type: Type = (
                FunType(params=adjusted_param_types, ret=fun.fun_type.ret)
                if isinstance(fun.fun_type, FunType)
                else fun.fun_type
            )
            fun = ast.Function(
                name=fun.name,
                params=tuple(adjusted_params),
                body=fun.body,
                fun_type=new_fun_type,
                globl=fun.globl,
                storage_class=fun.storage_class,
                source_file=fun.source_file,
                source_line=fun.source_line,
            )
        for param in fun.params:
            if isinstance(param.param_type, VoidType):
                raise SemanticError(f"Parameter '{param.name}' of '{fun.name}' has type 'void'.")
        if fun.body is not None:
            fn_ret = fun.fun_type.ret if isinstance(fun.fun_type, FunType) else IntType()
            if isinstance(fn_ret, (Structure, Union)) and not is_complete(fn_ret, self._struct_type_table):
                raise SemanticError(f"Function '{fun.name}' cannot be defined with incomplete aggregate return type.")
            for param in fun.params:
                if isinstance(param.param_type, (Structure, Union)) and not is_complete(
                    param.param_type, self._struct_type_table
                ):
                    raise SemanticError(
                        f"Parameter '{param.name}' of '{fun.name}' cannot be defined with incomplete aggregate type."
                    )
        var_binding = self._variable_map.get(fun.name)
        if var_binding is not None and (var_binding.from_current_block or var_binding.linkage == Linkage.EXTERNAL):
            raise SemanticError(f"'{fun.name}' redeclared as a function but was previously declared as a variable.")
        new_param_types = tuple(p.param_type for p in fun.params)
        ret_type: Type = fun.fun_type.ret if isinstance(fun.fun_type, FunType) else IntType()
        existing = self._function_map.get(fun.name)
        if fun.storage_class == StorageClass.Static:
            new_linkage = Linkage.INTERNAL
        elif existing is not None:
            new_linkage = existing.linkage  # inherit whatever linkage was already established
        else:
            new_linkage = Linkage.EXTERNAL
        if existing is not None:
            if existing.param_types != new_param_types:
                raise SemanticError(
                    f"Conflicting declarations of '{fun.name}': parameter types do not match prior declaration."
                )
            if existing.ret_type != ret_type:
                raise SemanticError(
                    f"Conflicting declarations of '{fun.name}': return type does not match prior declaration."
                )
            if existing.linkage != new_linkage:
                raise SemanticError(
                    f"'{fun.name}' previously declared with "
                    f"{'external' if existing.linkage == Linkage.EXTERNAL else 'internal'} linkage; "
                    f"cannot redeclare with {'internal' if new_linkage == Linkage.INTERNAL else 'external'} linkage."
                )
            if fun.body is not None and existing.has_definition:
                raise SemanticError(f"Redefinition of function '{fun.name}'.")
            if fun.body is not None:
                self._function_map[fun.name] = FunctionBinding(
                    has_definition=True,
                    param_types=new_param_types,
                    ret_type=ret_type,
                    linkage=new_linkage,
                )
        else:
            self._function_map[fun.name] = FunctionBinding(
                has_definition=fun.body is not None,
                param_types=new_param_types,
                ret_type=ret_type,
                linkage=new_linkage,
            )
        fun_globl = new_linkage == Linkage.EXTERNAL
        prior_sym = self._symbol_table.get(fun.name)
        prior_defined = (
            isinstance(prior_sym.attrs, FunAttr) and prior_sym.attrs.defined if prior_sym is not None else False
        )
        self._symbol_table[fun.name] = SymbolTableEntry(
            t=FunType(params=new_param_types, ret=ret_type),
            attrs=FunAttr(defined=prior_defined or fun.body is not None, globl=fun_globl),
        )
        self._current_block_funs.add(fun.name)
        return fun

    def _enter_scope(
        self,
    ) -> tuple[dict[str, VariableBinding], set[str], dict[str, StructureBinding], dict[str, StructureBinding]]:
        """Open a new nested scope and return snapshots of the current block state.

        Saves a copy of ``_variable_map``, ``_current_block_funs``, ``_structure_map``, and
        ``_union_map``, then marks every existing binding as not-from-the-current-scope and
        clears ``_current_block_funs`` so that the new scope starts fresh.

        Returns:
            A 4-tuple ``(saved_variable_map, saved_block_funs, saved_structure_map,
            saved_union_map)``.  Assign all four back to the corresponding attributes
            once the nested scope has been fully processed to restore the enclosing bindings.
        """
        saved_vars = dict(self._variable_map)
        saved_funs = self._current_block_funs
        saved_structs = dict(self._structure_map)
        saved_unions = dict(self._union_map)
        self._variable_map = {k: v._replace(from_current_block=False) for k, v in self._variable_map.items()}
        self._current_block_funs = set()
        self._structure_map = {k: v._replace(from_current_scope=False) for k, v in self._structure_map.items()}
        self._union_map = {k: v._replace(from_current_scope=False) for k, v in self._union_map.items()}
        return saved_vars, saved_funs, saved_structs, saved_unions

    def _resolve_block(self, body: ast.Block) -> ast.Block:
        """Resolve a sequence of block items in a new nested scope.

        Opens a nested scope via ``_enter_scope``, processes ``body``, then
        restores the enclosing bindings.  Any variable declared inside ``body``
        is invisible after this call returns; outer-scope bindings remain
        accessible throughout.

        Args:
            body: The ordered sequence of block items to resolve.

        Returns:
            A new tuple of resolved block items.

        Raises:
            SemanticError: If any block item within ``body`` contains a
                semantic violation.
        """
        saved_vars, saved_funs, saved_structs, saved_unions = self._enter_scope()
        result = tuple(self._resolve_block_item(item) for item in body)
        self._variable_map = saved_vars
        self._current_block_funs = saved_funs
        self._structure_map = saved_structs
        self._union_map = saved_unions
        return result

    def _type_check_exp(self, e: ast.Exp) -> ast.Exp:  # noqa: C901 - ignore complexity warning
        """Annotate an already-name-resolved expression with its C type.

        If ``e.exp_type`` is already set, the expression is returned unchanged.
        Otherwise, the type is determined by the expression form:

        * ``Var`` - looks up the variable's type from ``_type_table`` by the
          resolved unique name.  If the name maps to a ``FunAttr`` entry in
          ``_symbol_table``, a ``SemanticError`` is raised (function name used
          as a variable).
        * ``Constant`` - ``ConstLong`` constants receive ``LongType``; all
          others receive ``IntType``.
        * ``Unary`` - ``Not`` always produces ``IntType``; all other unary
          operators propagate the operand's type.
        * ``Binary`` - logical (``And``, ``Or``) and comparison operators
          always produce ``IntType``; arithmetic and bitwise operators produce
          ``LongType`` when either operand is ``LongType``, otherwise the type
          of the left operand.
        * ``Assignment`` - the type of the lvalue operand.  The lvalue must be a
          ``Var``, ``Dereference``, or ``Subscript``; any other form raises ``SemanticError``.
          The lvalue type must not be ``Array``; array objects are not modifiable lvalues in C.
          Sub-expressions of the lvalue are type-checked without array decay so that an array
          ``Var`` retains its ``Array`` type and can be rejected with a specific error.
        * ``Conditional`` - ``LongType`` if either branch is ``LongType``;
          otherwise the type of the true branch.
        * ``FunctionCall`` - the declared return type of the callee, looked up
          from ``_function_map``.  Validates that the callee is a known
          function, that the name does not refer to a local variable, and that
          the argument count matches the declared arity.
        * ``Cast`` - the explicit target type.
        * ``Subscript`` - the referenced type of whichever operand is a pointer.  Exactly one
          operand must be a ``Pointer`` and the other must be an integer type (``IntType``,
          ``LongType``, ``UIntType``, or ``ULongType``); the integer operand is cast to
          ``LongType`` when it is not already that type.
        * ``String`` - ``Array(Char(), len(bytes) + 1)``; the ``+1`` accounts for the implicit
          NUL terminator.  Array decay in ``_typecheck_and_convert`` will then convert the result
          to ``Pointer(Char())`` at every call site that expects an expression value.

        Child expressions that are not yet typed are recursively type-checked
        before the parent's type is determined, so this method may be called on
        expressions at any point after name resolution.

        Args:
            e: An expression node, already name-resolved, to type-check.

        Returns:
            A copy of ``e`` with ``exp_type`` set to the correct C type, or
            ``e`` itself when ``exp_type`` was already present.

        Raises:
            SemanticError: If a ``Var`` maps to a ``FunAttr`` in
                ``_symbol_table`` (function name used as a variable); if a
                ``Var`` name is absent from ``_type_table`` (undeclared); if a
                ``FunctionCall`` names a local variable (variable called as a
                function); if a ``FunctionCall`` names an undeclared function;
                if the argument count does not match the declared arity;
                if an ``Assignment`` lvalue is not a ``Var``, ``Dereference``, or
                ``Subscript``; or if an ``Assignment`` lvalue has an array type.
        """
        if e.exp_type is not None:
            return e
        match e:
            case ast.Var(name=name):
                sym_entry = self._symbol_table.get(name)
                if sym_entry is not None and isinstance(sym_entry.attrs, FunAttr):
                    return ast.Var(name, exp_type=Pointer(sym_entry.t))
                var_type = self._type_table.get(name)
                if var_type is None:
                    raise SemanticError(f"Undeclared variable {name}!")
                return ast.Var(name, exp_type=var_type)
            case ast.Constant(value=ast.ConstLong()):
                return ast.Constant(e.value, exp_type=LongType())
            case ast.Constant(value=ast.ConstLongLong()):
                return ast.Constant(e.value, exp_type=LongLongType())
            case ast.Constant(value=ast.ConstUInt()):
                return ast.Constant(e.value, exp_type=UIntType())
            case ast.Constant(value=ast.ConstULong()):
                return ast.Constant(e.value, exp_type=ULongType())
            case ast.Constant(value=ast.ConstULongLong()):
                return ast.Constant(e.value, exp_type=ULongLongType())
            case ast.Constant(value=ast.ConstDouble()):
                return ast.Constant(e.value, exp_type=DoubleType())
            case ast.Constant():
                return ast.Constant(e.value, exp_type=IntType())
            case ast.Unary(operator=op, exp=exp):
                typed_exp = self._typecheck_and_convert(exp)
                if isinstance(op, ast.Complement) and not is_integer(typed_exp.exp_type):
                    raise SemanticError("Bitwise complement requires an integer operand.")
                if isinstance(op, ast.Negate) and not is_arithmetic(typed_exp.exp_type):
                    raise SemanticError("Arithmetic negation requires an arithmetic operand.")
                if isinstance(typed_exp.exp_type, (Char, SChar, UChar)) and isinstance(
                    op, (ast.Negate, ast.Complement)
                ):
                    typed_exp = ast.Cast(IntType(), typed_exp, exp_type=IntType())
                if isinstance(op, Not) and not is_scalar(typed_exp.exp_type):
                    raise SemanticError("Logical operators only apply to scalar expressions")
                if isinstance(op, (ast.PreIncrement, ast.PreDecrement, ast.PostIncrement, ast.PostDecrement)):
                    if not is_lvalue(typed_exp):
                        raise SemanticError("Operand of increment/decrement must be a modifiable lvalue.")
                    if isinstance(typed_exp.exp_type, (Structure, Union)):
                        raise SemanticError("Increment/decrement requires a scalar or pointer operand.")
                    if isinstance(typed_exp.exp_type, Pointer) and isinstance(typed_exp.exp_type.referenced, VoidType):
                        raise SemanticError("Cannot increment/decrement a pointer to void.")
                    if isinstance(typed_exp.exp_type, FunType) or (
                        isinstance(typed_exp.exp_type, Pointer) and isinstance(typed_exp.exp_type.referenced, FunType)
                    ):
                        raise SemanticError("Cannot increment/decrement a function or function pointer.")
                exp_type: Type | None = IntType() if isinstance(op, Not) else typed_exp.exp_type
                return ast.Unary(op, typed_exp, exp_type=exp_type)
            case ast.Binary(operator=op, left=left, right=right):
                typed_left = self._typecheck_and_convert(left)
                typed_right = self._typecheck_and_convert(right)
                left_is_ptr = isinstance(typed_left.exp_type, Pointer)
                right_is_ptr = isinstance(typed_right.exp_type, Pointer)
                either_is_ptr = left_is_ptr or right_is_ptr
                if isinstance(op, (And, Or)):
                    if not is_scalar(typed_left.exp_type) or not is_scalar(typed_right.exp_type):
                        raise SemanticError("Logical operators only apply to scalar expressions")
                    bin_type: Type | None = IntType()
                elif isinstance(op, (Equal, NotEqual, LessThan, LessOrEqual, GreaterThan, GreaterOrEqual)):
                    bin_type = IntType()
                    if typed_left.exp_type is not None and typed_right.exp_type is not None:
                        if either_is_ptr:
                            if isinstance(op, (Equal, NotEqual)):
                                # Equality allows null pointer constants on either side (C11 sec. 6.5.9).
                                common = get_common_pointer_type(typed_left, typed_right)
                                if typed_left.exp_type != common:
                                    typed_left = ast.Cast(common, typed_left, exp_type=common)
                                if typed_right.exp_type != common:
                                    typed_right = ast.Cast(common, typed_right, exp_type=common)
                            else:
                                if typed_left.exp_type != typed_right.exp_type:
                                    raise SemanticError(
                                        "Cannot compare values of incompatible types: "
                                        "pointer types are not implicitly convertible to or from arithmetic types."
                                    )
                        else:
                            if not is_arithmetic(typed_left.exp_type) or not is_arithmetic(typed_right.exp_type):
                                raise SemanticError("Comparison operators require arithmetic operands.")
                            common = get_common_type(typed_left.exp_type, typed_right.exp_type)
                            if typed_left.exp_type != common:
                                typed_left = ast.Cast(common, typed_left, exp_type=common)
                            if typed_right.exp_type != common:
                                typed_right = ast.Cast(common, typed_right, exp_type=common)
                elif isinstance(op, (LeftShift, RightShift)):
                    if not is_integer(typed_left.exp_type) or not is_integer(typed_right.exp_type):
                        raise SemanticError("Shift operands must be integer types.")
                    # C sec. 6.5.7: integer promotions applied to each operand;
                    # result type is that of the promoted left operand.
                    if isinstance(typed_left.exp_type, (Char, SChar, UChar)):
                        typed_left = ast.Cast(IntType(), typed_left, exp_type=IntType())
                    if isinstance(typed_right.exp_type, (Char, SChar, UChar)):
                        typed_right = ast.Cast(IntType(), typed_right, exp_type=IntType())
                    bin_type = typed_left.exp_type
                else:
                    if either_is_ptr:
                        if (
                            left_is_ptr
                            and isinstance(typed_left.exp_type, Pointer)
                            and isinstance(typed_left.exp_type.referenced, FunType)
                        ) or (
                            right_is_ptr
                            and isinstance(typed_right.exp_type, Pointer)
                            and isinstance(typed_right.exp_type.referenced, FunType)
                        ):
                            raise SemanticError("Cannot perform arithmetic on a function pointer type.")
                        tbl = self._struct_type_table
                        if (left_is_ptr and not is_pointer_to_complete(typed_left.exp_type, tbl)) or (
                            right_is_ptr and not is_pointer_to_complete(typed_right.exp_type, tbl)
                        ):
                            raise SemanticError("Cannot perform pointer arithmetic on a pointer to an incomplete type.")
                        if isinstance(op, ast.Add):
                            if left_is_ptr and right_is_ptr:
                                raise SemanticError("Cannot add two pointers.")
                            if left_is_ptr:
                                if isinstance(typed_right.exp_type, DoubleType):
                                    raise SemanticError(
                                        "Cannot perform pointer arithmetic with a floating-point operand."
                                    )
                                if not isinstance(typed_right.exp_type, (LongType, LongLongType)):
                                    typed_right = ast.Cast(LongType(), typed_right, exp_type=LongType())
                                bin_type = typed_left.exp_type
                            else:
                                if isinstance(typed_left.exp_type, DoubleType):
                                    raise SemanticError(
                                        "Cannot perform pointer arithmetic with a floating-point operand."
                                    )
                                if not isinstance(typed_left.exp_type, (LongType, LongLongType)):
                                    typed_left = ast.Cast(LongType(), typed_left, exp_type=LongType())
                                bin_type = typed_right.exp_type
                        elif isinstance(op, ast.Subtract):
                            if left_is_ptr and right_is_ptr:
                                if typed_left.exp_type != typed_right.exp_type:
                                    raise SemanticError("Cannot subtract pointers of different types.")
                                bin_type = LongType()
                            elif left_is_ptr:
                                if isinstance(typed_right.exp_type, DoubleType):
                                    raise SemanticError(
                                        "Cannot perform pointer arithmetic with a floating-point operand."
                                    )
                                if not isinstance(typed_right.exp_type, (LongType, LongLongType)):
                                    typed_right = ast.Cast(LongType(), typed_right, exp_type=LongType())
                                bin_type = typed_left.exp_type
                            else:
                                raise SemanticError("Cannot subtract a pointer from a non-pointer value.")
                        else:
                            raise SemanticError("Cannot apply this operator to a pointer type.")
                    else:
                        if isinstance(op, (ast.Remainder, ast.BitwiseAnd, ast.BitwiseOr, ast.BitwiseXor)):
                            if not is_integer(typed_left.exp_type) or not is_integer(typed_right.exp_type):
                                raise SemanticError("Operator requires integer operands.")
                        else:
                            if not is_arithmetic(typed_left.exp_type) or not is_arithmetic(typed_right.exp_type):
                                raise SemanticError("Operator requires arithmetic operands.")
                        assert typed_left.exp_type is not None and typed_right.exp_type is not None
                        bin_type = get_common_type(typed_left.exp_type, typed_right.exp_type)
                        if typed_left.exp_type != bin_type:
                            typed_left = ast.Cast(bin_type, typed_left, exp_type=bin_type)
                        if typed_right.exp_type != bin_type:
                            typed_right = ast.Cast(bin_type, typed_right, exp_type=bin_type)
                return ast.Binary(op, typed_left, typed_right, exp_type=bin_type)
            case ast.Assignment(lvalue=lvalue, rvalue=rvalue):
                typed_lv = self._type_check_exp(lvalue)
                if not is_lvalue(typed_lv):
                    raise SemanticError("Invalid lvalue in assignment expression.")
                if isinstance(typed_lv.exp_type, Array):
                    raise SemanticError("Cannot assign to an expression of array type.")
                if isinstance(typed_lv.exp_type, (Structure, Union)) and not is_complete(
                    typed_lv.exp_type, self._struct_type_table
                ):
                    raise SemanticError("Cannot assign to an expression of incomplete aggregate type.")
                typed_rv = self._typecheck_and_convert(rvalue)
                lv_type = typed_lv.exp_type
                if lv_type is not None:
                    typed_rv = _convert_by_assignment(typed_rv, lv_type)
                return ast.Assignment(typed_lv, typed_rv, exp_type=lv_type)
            case ast.CompoundAssignment(lvalue=lvalue, op=op, rvalue=rvalue):
                typed_lv = self._type_check_exp(lvalue)
                if not is_lvalue(typed_lv):
                    raise SemanticError("Invalid lvalue in compound assignment expression.")
                if isinstance(typed_lv.exp_type, (Array, Structure, Union, VoidType, FunType)):
                    raise SemanticError("Invalid type for compound assignment target.")
                typed_rv = self._typecheck_and_convert(rvalue)
                lv_type = typed_lv.exp_type
                assert lv_type is not None
                if isinstance(lv_type, Pointer) and isinstance(op, (ast.Add, ast.Subtract)):
                    if isinstance(typed_rv.exp_type, Pointer):
                        raise SemanticError("Cannot add or subtract two pointers via compound assignment.")
                    if isinstance(typed_rv.exp_type, DoubleType):
                        raise SemanticError("Cannot perform pointer arithmetic with a floating-point operand.")
                    if not isinstance(lv_type.referenced, (FunType, VoidType)):
                        tbl = self._struct_type_table
                        if not is_pointer_to_complete(lv_type, tbl):
                            raise SemanticError("Cannot perform pointer arithmetic on a pointer to an incomplete type.")
                    if not isinstance(typed_rv.exp_type, (LongType, LongLongType)):
                        typed_rv = ast.Cast(LongType(), typed_rv, exp_type=LongType())
                elif isinstance(op, (LeftShift, RightShift)):
                    if not is_integer(lv_type) or not is_integer(typed_rv.exp_type):
                        raise SemanticError("Shift operator requires integer operands.")
                    if isinstance(typed_rv.exp_type, (Char, SChar, UChar)):
                        typed_rv = ast.Cast(IntType(), typed_rv, exp_type=IntType())
                elif isinstance(op, (ast.Remainder, ast.BitwiseAnd, ast.BitwiseOr, ast.BitwiseXor)):
                    if not is_integer(lv_type) or not is_integer(typed_rv.exp_type):
                        raise SemanticError("Operator requires integer operands.")
                    assert typed_rv.exp_type is not None
                    common = get_common_type(lv_type, typed_rv.exp_type)
                    if typed_rv.exp_type != common:
                        typed_rv = ast.Cast(common, typed_rv, exp_type=common)
                else:
                    if not is_arithmetic(lv_type) or not is_arithmetic(typed_rv.exp_type):
                        raise SemanticError("Operator requires arithmetic operands.")
                    assert typed_rv.exp_type is not None
                    common = get_common_type(lv_type, typed_rv.exp_type)
                    if typed_rv.exp_type != common:
                        typed_rv = ast.Cast(common, typed_rv, exp_type=common)
                return ast.CompoundAssignment(typed_lv, op, typed_rv, exp_type=lv_type)
            case ast.Conditional(condition=cond, exp_true=exp_true, exp_false=exp_false):
                typed_cond = self._typecheck_and_convert(cond)
                typed_true = self._typecheck_and_convert(exp_true)
                typed_false = self._typecheck_and_convert(exp_false)
                if not is_scalar(typed_cond.exp_type):
                    raise SemanticError("Condition in conditional operator must be scalar")
                if typed_true.exp_type is not None and typed_false.exp_type is not None:
                    t1 = typed_true.exp_type
                    t2 = typed_false.exp_type
                    if isinstance(t1, VoidType) and isinstance(t2, VoidType):
                        cond_type: Type | None = VoidType()
                    elif isinstance(t1, Pointer) or isinstance(t2, Pointer):
                        common_cond = get_common_pointer_type(typed_true, typed_false)
                        cond_type = common_cond
                        if t1 != common_cond:
                            typed_true = ast.Cast(common_cond, typed_true, exp_type=common_cond)
                        if t2 != common_cond:
                            typed_false = ast.Cast(common_cond, typed_false, exp_type=common_cond)
                    elif is_scalar(t1) and is_scalar(t2):
                        common_cond = get_common_type(t1, t2)
                        cond_type = common_cond
                        if t1 != common_cond:
                            typed_true = ast.Cast(common_cond, typed_true, exp_type=common_cond)
                        if t2 != common_cond:
                            typed_false = ast.Cast(common_cond, typed_false, exp_type=common_cond)
                    elif isinstance(t1, (Structure, Union)) and isinstance(t2, (Structure, Union)):
                        if t1 != t2:
                            raise SemanticError("Branches of conditional expression have incompatible aggregate types.")
                        cond_type = t1
                    else:
                        raise SemanticError("Cannot convert branches of conditional to a common type")
                else:
                    cond_type = typed_true.exp_type
                return ast.Conditional(typed_cond, typed_true, typed_false, exp_type=cond_type)
            case ast.FunctionCall(callee=callee, args=args):
                if isinstance(callee, ast.Var) and callee.exp_type is None:
                    fn_name = callee.name
                    if fn_name not in self._function_map:
                        raise SemanticError(f"Call to undeclared function '{fn_name}'.")
                    fn_binding = self._function_map[fn_name]
                    typed_callee: ast.Exp = ast.Var(
                        fn_name,
                        exp_type=FunType(params=fn_binding.param_types, ret=fn_binding.ret_type),
                    )
                    param_types = fn_binding.param_types
                    fn_ret_type: Type = fn_binding.ret_type
                else:
                    typed_callee = self._typecheck_and_convert(callee)
                    callee_type = typed_callee.exp_type
                    if isinstance(callee_type, FunType):
                        param_types = callee_type.params
                        fn_ret_type = callee_type.ret
                    elif isinstance(callee_type, Pointer) and isinstance(callee_type.referenced, FunType):
                        param_types = callee_type.referenced.params
                        fn_ret_type = callee_type.referenced.ret
                    else:
                        raise SemanticError("Callee expression is not a function or function pointer.")
                expected = len(param_types)
                if len(args) != expected:
                    raise SemanticError(f"Function called with {len(args)} argument(s) but declared with {expected}.")
                typed_args = [
                    _convert_by_assignment(self._typecheck_and_convert(arg), param_type)
                    for arg, param_type in zip(args, param_types)
                ]
                return ast.FunctionCall(typed_callee, typed_args, exp_type=fn_ret_type)
            case ast.Cast(target_type=target_type, exp=exp):
                target_type = self._resolve_type(target_type)
                validate_type_specifier(target_type, self._struct_type_table)
                typed_inner = self._typecheck_and_convert(exp)
                if isinstance(target_type, VoidType):
                    return ast.Cast(target_type, typed_inner, exp_type=target_type)
                if not is_scalar(target_type):
                    raise SemanticError("Can only cast to scalar type or void")
                if not is_scalar(typed_inner.exp_type):
                    raise SemanticError("Cannot cast non-scalar expression to scalar type")
                if isinstance(target_type, Pointer) and isinstance(typed_inner.exp_type, DoubleType):
                    raise SemanticError("Cannot cast a double to a pointer type.")
                if isinstance(target_type, DoubleType) and isinstance(typed_inner.exp_type, Pointer):
                    raise SemanticError("Cannot cast a pointer to a double.")
                return ast.Cast(target_type, typed_inner, exp_type=target_type)
            case ast.Dereference(exp=exp):
                typed_exp = self._typecheck_and_convert(exp)
                if not isinstance(typed_exp.exp_type, Pointer):
                    raise SemanticError("Cannot dereference a non-pointer type.")
                if isinstance(typed_exp.exp_type.referenced, VoidType):
                    raise SemanticError("Cannot dereference a pointer to void.")
                return ast.Dereference(typed_exp, exp_type=typed_exp.exp_type.referenced)
            case ast.AddrOf(exp=exp):
                typed_exp = self._type_check_exp(exp)
                if not is_lvalue(typed_exp):
                    raise SemanticError("Operand of '&' must be an lvalue.")
                if typed_exp.exp_type is None:
                    raise SemanticError("Cannot take address of expression with unknown type.")
                return ast.AddrOf(typed_exp, exp_type=Pointer(typed_exp.exp_type))
            case ast.Subscript(exp1=exp1, exp2=exp2):
                typed_e1 = self._typecheck_and_convert(exp1)
                typed_e2 = self._typecheck_and_convert(exp2)
                t1 = typed_e1.exp_type
                t2 = typed_e2.exp_type
                _int_types = (IntType, LongType, LongLongType, UIntType, ULongType, ULongLongType, Char, SChar, UChar)
                if isinstance(t1, Pointer) and isinstance(t2, _int_types):
                    if not is_pointer_to_complete(t1, self._struct_type_table):
                        raise SemanticError("Cannot subscript a pointer to an incomplete type.")
                    if not isinstance(t2, (LongType, LongLongType)):
                        typed_e2 = ast.Cast(LongType(), typed_e2, exp_type=LongType())
                    subscript_type: Type = t1.referenced
                elif isinstance(t1, _int_types) and isinstance(t2, Pointer):
                    if not is_pointer_to_complete(t2, self._struct_type_table):
                        raise SemanticError("Cannot subscript a pointer to an incomplete type.")
                    if not isinstance(t1, (LongType, LongLongType)):
                        typed_e1 = ast.Cast(LongType(), typed_e1, exp_type=LongType())
                    subscript_type = t2.referenced
                else:
                    raise SemanticError("Subscript must have integer and pointer operands.")
                return ast.Subscript(typed_e1, typed_e2, exp_type=subscript_type)
            case ast.String(string=s):
                return ast.String(s, exp_type=Array(element=Char(), size=len(s) + 1))
            case ast.SizeOfT(t=t):
                t = self._resolve_type(t)
                validate_type_specifier(t, self._struct_type_table)
                if not is_complete(t, self._struct_type_table):
                    raise SemanticError("Cannot get the size of an incomplete type")
                return ast.SizeOfT(t, exp_type=ULongType())
            case ast.SizeOf(exp=exp):
                typed_inner = self._type_check_exp(exp)
                if (
                    isinstance(typed_inner, ast.Var)
                    and isinstance(self._symbol_table.get(typed_inner.name), SymbolTableEntry)
                    and isinstance(self._symbol_table[typed_inner.name].attrs, FunAttr)
                ):
                    raise SemanticError("Cannot apply sizeof to a function type")
                if typed_inner.exp_type is not None and not is_complete(typed_inner.exp_type, self._struct_type_table):
                    raise SemanticError("Cannot get the size of an incomplete type")
                return ast.SizeOf(typed_inner, exp_type=ULongType())
            case ast.Dot(structure=structure, member=member):
                typed_structure = self._typecheck_and_convert(structure)  #
                match typed_structure.exp_type:
                    case Structure(tag=tag) | Union(tag=tag):
                        aggregate_def = self._struct_type_table.get(tag)  #
                        if aggregate_def is None:
                            raise SemanticError("Cannot access member of incomplete aggregate type.")
                        member_def = next(  #
                            (m for m in aggregate_def.members if m.name == member), None
                        )
                        if member_def is None:
                            raise SemanticError(f"Aggregate has no member named '{member}'.")
                        return ast.Dot(typed_structure, member, exp_type=member_def.member_type)  #
                    case _:
                        raise SemanticError("Operand of '.' is not a struct or union.")
            case ast.Arrow(pointer=pointer, member=member):
                typed_pointer = self._typecheck_and_convert(pointer)
                match typed_pointer.exp_type:
                    case Pointer(referenced=Structure(tag=tag)) | Pointer(referenced=Union(tag=tag)):
                        aggregate_def = self._struct_type_table.get(tag)
                        if aggregate_def is None:
                            raise SemanticError("Cannot access member through pointer to incomplete aggregate type.")
                        member_def = next((m for m in aggregate_def.members if m.name == member), None)
                        if member_def is None:
                            raise SemanticError(f"Aggregate has no member named '{member}'.")
                        return ast.Arrow(typed_pointer, member, exp_type=member_def.member_type)
                    case _:
                        raise SemanticError("Operand of '->' is not a pointer to a struct or union.")
            case _:
                raise SemanticError("Unhandled expression type in type checker")

    def _typecheck_and_convert(self, e: ast.Exp) -> ast.Exp:
        """Type-check an expression and convert any array result to a pointer (array decay).

        Wraps ``_type_check_exp`` with the C implicit array-to-pointer decay rule (C11 sec. 6.3.2.1p3):
        if the typed result has an ``Array`` type, it is wrapped in an ``AddrOf`` node whose
        type is ``Pointer(element_type)``.  All other types are returned unchanged.

        This method should be used in place of ``_type_check_exp`` everywhere except when
        type-checking the operand of an ``AddrOf`` expression, where array decay must be
        suppressed so that ``&arr`` receives the correct type ``Pointer(Array(...))``.

        Args:
            e: An expression node, already name-resolved, to type-check and convert.

        Returns:
            The typed expression, with an ``AddrOf`` wrapper added when the expression has
            an array type; otherwise the expression returned by ``_type_check_exp`` unchanged.

        Raises:
            SemanticError: If ``_type_check_exp`` raises for any reason.
        """
        typed_e = self._type_check_exp(e)
        match typed_e.exp_type:
            case Array(element=elem_t):
                return ast.AddrOf(typed_e, exp_type=Pointer(elem_t))
            case (Structure() | Union()) as agg_t if not is_complete(agg_t, self._struct_type_table):
                raise SemanticError("Expression has incomplete aggregate type; only its address may be taken.")
            case _:
                return typed_e

    def _pad_to_size(self, inits: list[ast.Initializer], elem_t: Type, size: int) -> list[ast.Initializer]:
        """Pad a type-checked initializer list to ``size`` elements with zero initialisers.

        Appends zero-valued ``SingleInit`` nodes (using ``_zero_initializer``) until the list
        reaches ``size``.  Character element types use ``ConstChar`` / ``ConstUChar`` zero
        constants; all other types use their natural zero constant (C11 sec. 6.7.9p21).

        Args:
            inits: Already type-checked initializer nodes for the explicitly-given elements.
                Modified in place and returned.
            elem_t: The C element type of the array being padded.
            size: The total declared array size to pad up to.

        Returns:
            ``inits`` extended with zero initialisers so that ``len(inits) == size``.
        """
        while len(inits) < size:
            inits.append(_zero_initializer(elem_t, self._struct_type_table))
        return inits

    def _exp_to_static_init(self, e: ast.Exp) -> StaticInit:
        """Convert a type-annotated constant expression to its ``StaticInit`` equivalent.

        Handles the expression forms that appear as leaves after ``_typecheck_init``
        has processed a static-storage initialiser:

        * ``Constant`` nodes with any numeric ``Const`` variant.
        * ``Cast(Pointer, Constant)`` - the null-pointer form produced by
          ``_zero_initializer`` for pointer-typed array elements.
        * ``Cast(scalar_type, Constant)`` - widening casts inserted by
          ``_convert_by_assignment`` when an integer literal is used to initialise
          a wider type (e.g. ``long arr[2] = {1, 2}``).
        * ``AddrOf(String(s))`` - a string literal decayed to a pointer, used to
          initialise a ``char *`` (or compatible) struct member or array element.
          Registers the string content as a hidden static constant in
          ``_symbol_table`` and returns ``PointerInit(str_name)``.

        Zero-valued scalars are encoded as ``ZeroInit``; adjacent ``ZeroInit`` nodes
        are coalesced by ``_flatten_static_init``.

        Args:
            e: A fully type-annotated scalar expression produced by ``_typecheck_init``.

        Returns:
            The corresponding ``StaticInit`` element.

        Raises:
            SemanticError: If ``e`` is not a recognised constant form.
        """
        match e:
            case ast.AddrOf(exp=ast.String(string=s)):
                str_name = self._make_string_name()
                self._symbol_table[str_name] = SymbolTableEntry(
                    t=Array(element=Char(), size=len(s) + 1),
                    attrs=ConstantAttr(StringInit(s, True)),
                )
                return PointerInit(str_name)
            case ast.Cast(target_type=Pointer()):
                return ZeroInit(8)
            case ast.Cast(target_type=target, exp=ast.Constant(value=const)):
                return _make_static_init(target, const.val)
            case ast.Constant(value=const):
                return _make_static_init(e.exp_type, const.val)
            case _:
                raise SemanticError(f"Cannot convert expression to a static initialiser element: {type(e).__name__}")

    def _flatten_static_init(self, init: ast.Initializer) -> list[StaticInit]:
        """Flatten a fully type-checked ``Initializer`` tree into a list of ``StaticInit`` elements.

        Recursively traverses ``CompoundInit`` nodes and converts each leaf
        ``SingleInit`` to its ``StaticInit`` equivalent.  Adjacent ``ZeroInit``
        nodes are coalesced into a single ``ZeroInit`` with the combined byte count
        so that sparse array regions are represented compactly.  Struct
        ``CompoundInit`` nodes (identified by a non-``None`` ``init_type``) emit
        inter-member padding and trailing padding according to the layout recorded
        in ``_struct_type_table``.

        Args:
            init: A fully type-checked (and zero-padded) ``Initializer`` node, as
                returned by ``_typecheck_init``.

        Returns:
            A flat list of ``StaticInit`` elements with consecutive zero regions merged.

        Raises:
            SemanticError: If any leaf expression cannot be converted to a static
                initialiser element.
        """
        result: list[StaticInit] = []
        self._flatten_helper(init, result)
        return result

    def _flatten_helper(self, init: ast.Initializer, result: list[StaticInit]) -> None:
        """Recursive worker for ``_flatten_static_init``.

        Traverses the ``Initializer`` tree depth-first and appends ``StaticInit``
        elements to ``result``, merging any ``ZeroInit`` that is adjacent to a
        trailing ``ZeroInit`` already in the list.  Struct ``CompoundInit`` nodes
        (``init_type = Structure(tag)``) are expanded with the inter-member and
        trailing padding dictated by the struct's layout in ``_struct_type_table``.

        Args:
            init: The current ``Initializer`` node being processed.
            result: Accumulator list modified in place.
        """
        match init:
            case ast.SingleInit(exp=ast.String(string=s, exp_type=Array(size=size))):
                for elem in _string_array_static_init(s, size):
                    if isinstance(elem, ZeroInit) and result and isinstance(result[-1], ZeroInit):
                        result[-1:] = [ZeroInit(result[-1].bytes + elem.bytes)]
                    else:
                        result.append(elem)
            case ast.SingleInit(exp=e):
                elem = self._exp_to_static_init(e)
                if isinstance(elem, ZeroInit) and result and isinstance(result[-1], ZeroInit):
                    result[-1:] = [ZeroInit(result[-1].bytes + elem.bytes)]
                else:
                    result.append(elem)
            case ast.CompoundInit(initializers=sub_inits, init_type=Structure(tag=tag)):
                struct_def = self._struct_type_table[tag]
                current_offset = 0
                for i, member in enumerate(struct_def.members):
                    if member.offset > current_offset:
                        _append_zero_pad(result, member.offset - current_offset)
                    self._flatten_helper(sub_inits[i], result)
                    current_offset = member.offset + type_size(member.member_type, self._struct_type_table)
                trailing = struct_def.size - current_offset
                if trailing > 0:
                    _append_zero_pad(result, trailing)
            case ast.CompoundInit(initializers=sub_inits, init_type=Union(tag=tag)):
                # First (and only) member is initialised; remainder is zero padding.
                union_def = self._struct_type_table[tag]
                first_member_t = union_def.members[0].member_type
                self._flatten_helper(sub_inits[0], result)
                first_size = type_size(first_member_t, self._struct_type_table)
                trailing = union_def.size - first_size
                if trailing > 0:
                    _append_zero_pad(result, trailing)
            case ast.CompoundInit(initializers=sub_inits):
                for sub_init in sub_inits:
                    self._flatten_helper(sub_init, result)

    def _typecheck_init(self, target_type: Type, init: ast.Initializer) -> ast.Initializer:
        """Type-check a variable initializer against its declared type, applying assignment conversion.

        Handles both scalar (``SingleInit``) and aggregate (``CompoundInit``) initialisers.
        For scalar initialisers the wrapped expression is fully resolved, array-decay-converted,
        and then converted via assignment-compatibility rules to the target type.  For array
        initialisers each element is recursively checked against the array's element type; if
        the initializer list is shorter than the declared size, trailing elements are
        zero-initialised (C11 sec. 6.7.9p21).

        Args:
            target_type: The declared C type of the variable or array element being initialised.
            init: The initialiser node to type-check - either a ``SingleInit`` or ``CompoundInit``.

        Returns:
            A new ``Initializer`` with all sub-expressions fully resolved, type-annotated, and
            converted to the target type.  ``CompoundInit`` nodes are padded to the array's
            declared size with zero initialisers.

        Raises:
            SemanticError: If a ``CompoundInit`` is used with a non-array, non-struct target
                type; if the initializer list is longer than the declared array size or member
                count; if a string literal is used to initialise a non-character array or the
                string (including NUL terminator) exceeds the declared array size; or if any
                sub-expression fails type-checking or assignment conversion.
        """
        match target_type, init:
            case Array(element=elem_t, size=size), ast.SingleInit(exp=ast.String(string=s)):
                if not isinstance(elem_t, _CHAR_TYPES):
                    raise SemanticError("Cannot initialize a non-character array with a string literal.")
                if len(s) > size:
                    raise SemanticError(
                        f"Too many characters in string literal: string has {len(s)} bytes but array has size {size}."
                    )
                return ast.SingleInit(ast.String(s, exp_type=target_type))
            case _, ast.SingleInit(exp=e):
                typechecked_exp = self._typecheck_and_convert(self.resolve_exp(e))
                cast_exp = _convert_by_assignment(typechecked_exp, target_type)
                return ast.SingleInit(cast_exp)
            case Array(element=elem_t, size=size), ast.CompoundInit(initializers=init_list):
                if len(init_list) > size:
                    raise SemanticError(
                        f"Too many values in initializer: expected at most {size}, got {len(init_list)}."
                    )
                typechecked_list: list[ast.Initializer] = [
                    self._typecheck_init(elem_t, elem_init) for elem_init in init_list
                ]
                return ast.CompoundInit(self._pad_to_size(typechecked_list, elem_t, size))
            case Structure(tag=tag), ast.CompoundInit(initializers=init_list):
                struct_def = self._struct_type_table.get(tag)  #
                if struct_def is None:
                    raise SemanticError("Cannot initialize an incomplete struct type.")
                if len(init_list) > len(struct_def.members):  #
                    raise SemanticError(
                        f"Too many elements in struct initializer: "
                        f"expected at most {len(struct_def.members)}, got {len(init_list)}."
                    )
                typechecked_list = [  #
                    self._typecheck_init(struct_def.members[i].member_type, elem) for i, elem in enumerate(init_list)
                ]
                for i in range(len(init_list), len(struct_def.members)):  #
                    typechecked_list.append(
                        _zero_initializer(struct_def.members[i].member_type, self._struct_type_table)
                    )
                return ast.CompoundInit(typechecked_list, init_type=target_type)
            case Union(tag=tag), ast.CompoundInit(initializers=init_list):
                # C11 sec. 6.7.9p16: union initializer initialises the first named member only.
                union_def = self._struct_type_table.get(tag)
                if union_def is None:
                    raise SemanticError("Cannot initialize an incomplete union type.")
                if len(init_list) > 1:
                    raise SemanticError(
                        f"Too many elements in union initializer: expected at most 1, got {len(init_list)}."
                    )
                first_member_t = union_def.members[0].member_type
                typechecked_list = (
                    [self._typecheck_init(first_member_t, init_list[0])]
                    if init_list
                    else [_zero_initializer(first_member_t, self._struct_type_table)]
                )
                return ast.CompoundInit(typechecked_list, init_type=target_type)
            case _:
                raise SemanticError("Cannot initialize a scalar object with a compound initializer.")

    def resolve_exp(self, e: ast.Exp) -> ast.Exp:  # noqa: C901 - ignore complexity warning
        """Resolve all variable references within an expression and annotate it with its C type.

        Recursively renames every ``ast.Var`` node from its source-level name to
        the unique internal identifier recorded in ``_variable_map``, then
        delegates type annotation to ``_type_check_exp``.  Assignment
        expressions are validated here to ensure their left-hand side is an
        ``ast.Var`` (simple lvalue); ``FunctionCall`` validation (declared
        function, correct arity, not a local variable) is performed by
        ``_type_check_exp``.

        Args:
            e: The expression node to resolve.

        Returns:
            A new expression node with all ``ast.Var`` references rewritten to
            unique internal names and ``exp_type`` set to the correct C type.

        Raises:
            SemanticError: If a ``Var`` node names a function rather than a
                variable; if a variable reference is undeclared; if the
                left-hand side of an assignment is not a plain ``ast.Var``; or
                if any sub-expression contains a semantic violation detected by
                ``_type_check_exp``.
        """
        match e:
            case ast.Assignment(lvalue=lvalue, rvalue=rvalue):
                resolved_lvalue = self.resolve_exp(lvalue)
                resolved_rvalue = self.resolve_exp(rvalue)
                return self._type_check_exp(ast.Assignment(resolved_lvalue, resolved_rvalue))
            case ast.CompoundAssignment(lvalue=lvalue, op=op, rvalue=rvalue):
                resolved_lvalue = self.resolve_exp(lvalue)
                resolved_rvalue = self.resolve_exp(rvalue)
                return self._type_check_exp(ast.CompoundAssignment(resolved_lvalue, op, resolved_rvalue))
            case ast.Var(name=name):
                if name in self._current_block_funs:
                    raise SemanticError(f"Function name '{name}' used as a variable.")
                if name in self._variable_map:
                    unique_name = self._variable_map[name].unique_name
                elif name in self._function_map:
                    fn_binding = self._function_map[name]
                    fun_type = FunType(params=fn_binding.param_types, ret=fn_binding.ret_type)
                    return ast.Var(name, exp_type=Pointer(fun_type))
                else:
                    raise SemanticError(f"Undeclared variable {name}!")
                return self._type_check_exp(ast.Var(unique_name))
            case ast.Constant():
                return self._type_check_exp(e)
            case ast.Unary(operator=op, exp=exp):
                return self._type_check_exp(ast.Unary(op, self.resolve_exp(exp)))
            case ast.Binary(operator=op, left=left, right=right):
                return self._type_check_exp(ast.Binary(op, self.resolve_exp(left), self.resolve_exp(right)))
            case ast.Conditional(condition=cond, exp_true=exp_true, exp_false=exp_false):
                return self._type_check_exp(
                    ast.Conditional(
                        self.resolve_exp(cond),
                        self.resolve_exp(exp_true),
                        self.resolve_exp(exp_false),
                    )
                )
            case ast.FunctionCall(callee=callee, args=args):
                if isinstance(callee, ast.Var):
                    name = callee.name
                    if name in self._variable_map and name not in self._current_block_funs:
                        binding = self._variable_map[name]
                        ptr_to_fun = isinstance(binding.var_type, Pointer) and isinstance(
                            binding.var_type.referenced, FunType
                        )
                        if not ptr_to_fun:
                            raise SemanticError(f"Variable '{name}' called as a function.")
                        resolved_callee: ast.Exp = self.resolve_exp(callee)
                    elif name in self._function_map or name in self._current_block_funs:
                        resolved_callee = callee
                    else:
                        raise SemanticError(f"Call to undeclared function '{name}'.")
                else:
                    resolved_callee = self.resolve_exp(callee)
                return self._type_check_exp(ast.FunctionCall(resolved_callee, [self.resolve_exp(arg) for arg in args]))
            case ast.Cast(target_type=target_type, exp=exp):
                return self._type_check_exp(ast.Cast(target_type, self.resolve_exp(exp)))
            case ast.Dereference(exp=exp):
                return self._type_check_exp(ast.Dereference(self.resolve_exp(exp)))
            case ast.AddrOf(exp=exp):
                return self._type_check_exp(ast.AddrOf(self.resolve_exp(exp)))
            case ast.Subscript(exp1=exp1, exp2=exp2):
                return self._type_check_exp(ast.Subscript(self.resolve_exp(exp1), self.resolve_exp(exp2)))
            case ast.String():
                return self._type_check_exp(e)
            case ast.SizeOfT():
                return self._type_check_exp(e)
            case ast.SizeOf(exp=exp):
                return self._type_check_exp(ast.SizeOf(self.resolve_exp(exp)))
            case ast.Dot(structure=structure, member=member):
                return self._type_check_exp(ast.Dot(self.resolve_exp(structure), member))
            case ast.Arrow(pointer=pointer, member=member):
                return self._type_check_exp(ast.Arrow(self.resolve_exp(pointer), member))
            case _:
                raise SemanticError("Unhandled expression type in semantic analyser")

    def _resolve_statement(self, statement: ast.Statement) -> ast.Statement:  # noqa: C901 - ignore complexity warning
        """Resolve all variable references within a statement and assign loop identifiers.

        Recursively resolves any expressions contained in the statement,
        returning a new statement node with all ``ast.Var`` references rewritten
        to their unique internal names.  Additionally:

        * Each loop construct (``While``, ``DoWhile``, ``For``) is assigned a
          fresh unique identifier via ``_make_loop_label`` and pushes that
          identifier onto both ``_loop_stack`` and ``_break_stack``.
        * Each ``Switch`` is assigned a fresh identifier and pushes it onto
          ``_break_stack`` and ``_switch_stack``.
        * ``Break`` is tagged with the top of ``_break_stack`` (the nearest
          enclosing loop or switch).
        * ``Continue`` is tagged with the top of ``_loop_stack`` (the nearest
          enclosing loop; a bare switch does not satisfy ``continue``).
        * ``Case`` and ``Default`` are tagged with the top of ``_switch_stack``
          (the identifier of their enclosing switch).

        Args:
            statement: The statement node to resolve.

        Returns:
            A new statement node with all variable references resolved and all
            control-flow statements tagged with their enclosing construct's identifier.

        Raises:
            SemanticError: If any expression contains an undeclared variable or
                an invalid lvalue; if ``Break`` appears outside a loop or switch;
                if ``Continue`` appears outside a loop; or if ``Case``/``Default``
                appear outside a switch.
        """
        match statement:
            case ast.Compound(body=block):
                return ast.Compound(self._resolve_block(block))
            case ast.Return(exp=exp):
                return_type = self._current_return_type
                if isinstance(return_type, VoidType):
                    if exp is not None:
                        raise SemanticError("void function must not return a value.")
                    return ast.Return(None)
                if exp is None:
                    raise SemanticError("non-void function must return a value.")
                resolved = self._typecheck_and_convert(self.resolve_exp(exp))
                if return_type is not None and resolved.exp_type is not None:
                    resolved = _convert_by_assignment(resolved, return_type)
                return ast.Return(resolved)
            case ast.Expression(exp=exp):
                return ast.Expression(self._typecheck_and_convert(self.resolve_exp(exp)))
            case ast.Null():
                return statement
            case ast.Break():
                if not self._break_stack:
                    raise SemanticError("'break' statement not within a loop or switch.")
                return ast.Break(self._break_stack[-1])
            case ast.Continue():
                if not self._loop_stack:
                    raise SemanticError("'continue' statement not within a loop.")
                return ast.Continue(self._loop_stack[-1])
            case ast.If(condition=cond, then=then_stmt, else_=None):
                return ast.If(self._resolve_condition(cond), self._resolve_statement(then_stmt), None)
            case ast.If(condition=cond, then=then_stmt, else_=else_stmt):
                assert else_stmt is not None
                return ast.If(
                    self._resolve_condition(cond),
                    self._resolve_statement(then_stmt),
                    self._resolve_statement(else_stmt),
                )
            case ast.While(condition=condition, body=body):
                label = self._make_loop_label("while")
                self._loop_stack.append(label)
                self._break_stack.append(label)
                result = ast.While(self._resolve_condition(condition), self._resolve_statement(body), label)
                self._loop_stack.pop()
                self._break_stack.pop()
                return result
            case ast.DoWhile(body=body, condition=condition):
                label = self._make_loop_label("do")
                self._loop_stack.append(label)
                self._break_stack.append(label)
                result = ast.DoWhile(self._resolve_statement(body), self._resolve_condition(condition), label)
                self._loop_stack.pop()
                self._break_stack.pop()
                return result
            case ast.For(init=init, condition=condition, post=post, body=body):
                saved_vars, saved_funs, saved_structs, saved_unions = self._enter_scope()
                label = self._make_loop_label("for")
                self._loop_stack.append(label)
                self._break_stack.append(label)
                init = self._resolve_for_init(init)
                condition = self._resolve_optional_exp(condition)
                if condition is not None and not is_scalar(condition.exp_type):
                    raise SemanticError("Controlling expression must be scalar; void expression is not permitted here.")
                post = self._resolve_optional_exp(post)
                body = self._resolve_statement(body)
                self._loop_stack.pop()
                self._break_stack.pop()
                self._variable_map = saved_vars
                self._current_block_funs = saved_funs
                self._structure_map = saved_structs
                self._union_map = saved_unions
                return ast.For(init, condition, post, body, label)
            case ast.Switch(control=control, body=body):
                label = self._make_loop_label("switch")
                self._break_stack.append(label)
                self._switch_stack.append(label)
                self._case_values_stack.append(set())
                self._default_seen_stack.append(False)
                resolved_control = self._typecheck_and_convert(self.resolve_exp(control))
                if not is_integer(resolved_control.exp_type):
                    raise SemanticError("Switch controlling expression must be an integer type")
                # C11 sec. 6.8.4.2p1: integer promotions applied to the controlling expression
                if isinstance(resolved_control.exp_type, (Char, SChar, UChar)):
                    resolved_control = ast.Cast(IntType(), resolved_control, exp_type=IntType())
                self._switch_type_stack.append(resolved_control.exp_type or IntType())
                result = ast.Switch(resolved_control, self._resolve_statement(body), label)
                self._break_stack.pop()
                self._switch_stack.pop()
                self._case_values_stack.pop()
                self._switch_type_stack.pop()
                self._default_seen_stack.pop()
                return result
            case ast.Case(value=value, statement=inner):
                if not self._switch_stack:
                    raise SemanticError("'case' label not within a switch statement.")
                raw_const = value.value
                if isinstance(raw_const, ast.ConstDouble):
                    raise SemanticError("Case expression must be an integer constant, not a floating-point value.")
                raw_int = int(raw_const.val)
                truncated = _truncate_case_value(raw_int, self._switch_type_stack[-1])
                if truncated in self._case_values_stack[-1]:
                    raise SemanticError(f"Duplicate case value {raw_int} in switch statement.")
                self._case_values_stack[-1].add(truncated)
                return ast.Case(value, self._resolve_statement(inner), self._switch_stack[-1])
            case ast.Default(statement=inner):
                if not self._switch_stack:
                    raise SemanticError("'default' label not within a switch statement.")
                if self._default_seen_stack[-1]:
                    raise SemanticError("Multiple 'default' labels in one switch statement.")
                self._default_seen_stack[-1] = True
                return ast.Default(self._resolve_statement(inner), self._switch_stack[-1])
            case ast.Goto(target=target):
                if target not in self._label_set:
                    raise SemanticError(f"goto targets undefined label '{target}'.")
                return statement
            case ast.Label(name=name, statement=inner):
                return ast.Label(name, self._resolve_statement(inner))
            case _:
                raise SemanticError("Unhandled statement type in semantic analyser")

    def _resolve_for_init(self, init: ast.ForInit) -> ast.ForInit:
        """Resolve the initialiser clause of a ``for`` loop.

        Dispatches to ``_resolve_optional_exp`` for an expression initialiser
        (``InitExp``) and to ``resolve_variable_declaration`` for a declaration
        initialiser (``InitDecl``).

        Args:
            init: The ``ForInit`` node from the parser - either an ``InitExp``
                wrapping an optional expression, or an ``InitDecl`` wrapping a
                variable declaration.

        Returns:
            A new ``ForInit`` node with all variable references resolved.

        Raises:
            SemanticError: If a declaration within ``init`` carries a storage
                class specifier; if a declaration within ``init`` duplicates a
                name already declared in the same scope; or if an expression
                within ``init`` contains an undeclared variable or an invalid
                lvalue.
        """
        match init:
            case ast.InitExp(exp=exp):
                return ast.InitExp(self._resolve_optional_exp(exp))
            case ast.InitDecl(declaration=ast.VarDecl(variable_declaration=var_decl)):
                if var_decl.storage_class is not None:
                    raise SemanticError(
                        f"Variable '{var_decl.name}' in 'for' loop initialiser may not have a storage class specifier."
                    )
                return ast.InitDecl(ast.VarDecl(self.resolve_variable_declaration(var_decl)))
            case _:
                raise SemanticError("Unhandled for-init type in semantic analyser")

    def _resolve_condition(self, cond: ast.Exp) -> ast.Exp:
        """Resolve a controlling condition expression and verify it is scalar.

        Conditions in ``if``, ``while``, ``do``/``while``, and ``for`` statements must be scalar
        expressions (C11 sec. 6.8.4.1, sec. 6.8.5).  ``void`` expressions are non-scalar and are
        therefore illegal as controlling conditions.

        Args:
            cond: The condition expression to resolve.

        Returns:
            The fully resolved and type-converted condition expression.

        Raises:
            SemanticError: If the resolved condition has a non-scalar type (e.g. ``void``).
        """
        resolved = self._typecheck_and_convert(self.resolve_exp(cond))
        if not is_scalar(resolved.exp_type):
            raise SemanticError("Controlling expression must be scalar; void expression is not permitted here.")
        return resolved

    def _resolve_optional_exp(self, exp: ast.Exp | None) -> ast.Exp | None:
        """Resolve an optional expression, returning ``None`` when absent.

        A convenience wrapper around ``resolve_exp`` for contexts where an
        expression is syntactically optional, such as the condition and post
        expressions of a ``for`` loop.

        Args:
            exp: The expression to resolve, or ``None`` if absent.

        Returns:
            The resolved expression, or ``None`` if ``exp`` is ``None``.

        Raises:
            SemanticError: If ``exp`` is not ``None`` and contains an
                undeclared variable or an invalid lvalue.
        """
        if exp is not None:
            return self._typecheck_and_convert(self.resolve_exp(exp))
        return None

    def _resolve_block_item(self, item: ast.BlockItem) -> ast.BlockItem:
        """Resolve a single block item (statement, variable declaration, or function declaration).

        Dispatches to ``resolve_variable_declaration`` for ``ast.D`` items wrapping
        a ``VarDecl``, to ``_register_fun_declaration`` for ``ast.D`` items wrapping a
        ``FunDecl`` (forward declarations only - definitions raise immediately), and to
        ``_resolve_statement`` for ``ast.S`` items.

        Args:
            item: The block item to resolve.

        Returns:
            A new block item with all variable references resolved.

        Raises:
            SemanticError: If the item contains a semantic violation, including a
                ``FunDecl`` whose ``body`` is not ``None`` (nested definition), or
                a ``FunDecl`` with a ``static`` storage class specifier (``static``
                is not permitted on block-scope function declarations).
        """
        match item:
            case ast.D(declaration=ast.VarDecl(variable_declaration=var_decl)):
                return ast.D(ast.VarDecl(self.resolve_variable_declaration(var_decl)))
            case ast.D(declaration=ast.StructDecl(struct_declaration=struct_decl)):
                tag = struct_decl.tag
                if self._union_map.get(tag, StructureBinding("", False)).from_current_scope:
                    raise SemanticError(f"Tag '{tag}' already declared as a union in this scope.")
                existing = self._structure_map.get(tag)
                if existing is not None and existing.from_current_scope:
                    unique_tag = existing.new_tag
                else:
                    unique_tag = self._make_struct_tag(tag)
                    self._structure_map[tag] = StructureBinding(new_tag=unique_tag, from_current_scope=True)
                resolved_members = tuple(
                    MemberDeclaration(m.member_name, self._resolve_type(m.member_type)) for m in struct_decl.members
                )
                resolved_struct_decl = StructDeclaration(tag=unique_tag, members=resolved_members)
                self._typecheck_struct_declaration(resolved_struct_decl)
                return ast.D(ast.StructDecl(resolved_struct_decl))
            case ast.D(declaration=ast.UnionDecl(union_declaration=union_decl)):
                tag = union_decl.tag
                if self._structure_map.get(tag, StructureBinding("", False)).from_current_scope:
                    raise SemanticError(f"Tag '{tag}' already declared as a struct in this scope.")
                existing = self._union_map.get(tag)
                if existing is not None and existing.from_current_scope:
                    unique_tag = existing.new_tag
                else:
                    unique_tag = self._make_union_tag(tag)
                    self._union_map[tag] = StructureBinding(new_tag=unique_tag, from_current_scope=True)
                resolved_members = tuple(
                    MemberDeclaration(m.member_name, self._resolve_type(m.member_type)) for m in union_decl.members
                )
                resolved_union_decl = UnionDeclaration(tag=unique_tag, members=resolved_members)
                self._typecheck_union_declaration(resolved_union_decl)
                return ast.D(ast.UnionDecl(resolved_union_decl))
            case ast.D(declaration=ast.FunDecl(function_declaration=fun)):
                if fun.body is not None:
                    raise SemanticError(f"Nested function definition of '{fun.name}' is not permitted.")
                if fun.storage_class == StorageClass.Static:
                    raise SemanticError(
                        f"Block-scope function declaration '{fun.name}' may not have 'static' storage class."
                    )
                self._validate_params_unique(fun)
                adjusted_fun = self._register_fun_declaration(fun)
                return ast.D(ast.FunDecl(adjusted_fun))
            case ast.S(statement=statement):
                return ast.S(self._resolve_statement(statement))
            case _:
                raise SemanticError("Unhandled block item type in semantic analyser")

    def _resolve_function(self, function: ast.Function) -> ast.Function:
        """Resolve all variable references within a function definition.

        Validates that parameter names are unique (via ``_validate_params_unique``)
        for both declarations and definitions.  If ``function.body`` is ``None``
        this is a forward declaration and is returned immediately after the
        parameter check (the name has already been registered by the pre-pass in
        ``_resolve_program``).  Otherwise:

        1. Resets per-function state; marks all entries in ``_variable_map`` as
           ``from_current_block=False`` so that file-scope variables accumulated
           in ``_resolve_program`` are visible inside the function body.
        2. Adds each parameter to ``_variable_map`` under a unique internal name.
        3. Collects all labels in a first pass.
        4. Resolves every block item in the body.

        Args:
            function: The ``Function`` AST node to resolve.

        Returns:
            A new ``Function`` node with all variable references resolved,
            parameter names replaced by unique internal identifiers, and
            all declared variable names replaced by unique internal identifiers.
            Returns ``function`` unchanged if ``body`` is ``None``.

        Raises:
            SemanticError: If the function body contains any semantic violation,
                or if two parameters share the same name.
        """
        self._validate_params_unique(function)
        if function.body is None:
            return function

        self._variable_map = {k: v._replace(from_current_block=False) for k, v in self._variable_map.items()}
        self._structure_map = {k: v._replace(from_current_scope=False) for k, v in self._structure_map.items()}
        self._union_map = {k: v._replace(from_current_scope=False) for k, v in self._union_map.items()}
        self._current_block_funs = set()
        self._label_set = set()
        self._loop_stack = []
        self._break_stack = []
        self._switch_stack = []
        self._case_values_stack = []
        self._switch_type_stack = []
        self._default_seen_stack = []

        resolved_params: list[ast.Parameter] = []
        for param in function.params:
            resolved_var = self.resolve_variable_declaration(
                ast.VariableDeclaration(param.name, None, param.param_type)
            )
            resolved_params.append(ast.Parameter(param_type=param.param_type, name=resolved_var.name))

        self._current_return_type = function.fun_type.ret if isinstance(function.fun_type, FunType) else IntType()
        self._collect_labels(function.body)
        binding = self._function_map.get(function.name)
        globl = binding.linkage == Linkage.EXTERNAL if binding is not None else True
        return ast.Function(
            name=function.name,
            params=tuple(resolved_params),
            body=tuple(self._resolve_block_item(item) for item in function.body),
            fun_type=function.fun_type,
            globl=globl,
            storage_class=function.storage_class,
            source_file=function.source_file,
            source_line=function.source_line,
        )

    def _resolve_program(self, program: ast.Program) -> ast.Program:
        """Perform semantic analysis on a parsed program.

        Processes top-level declarations in source order.  For each ``FunDecl``,
        ``_register_fun_declaration`` is called first (making the name visible to
        the bodies that follow), then ``_resolve_function`` resolves the body if
        present.  For each ``VarDecl``, ``_resolve_file_scope_variable_declaration``
        validates linkage consistency and records the variable in
        ``_variable_map`` and ``_file_scope_linkage`` so that subsequent function
        bodies can reference it.

        A function must therefore be declared or defined before any subsequent
        function body may call it; mutual recursion requires a forward declaration.
        A file-scope variable must be declared before any function body that
        references it.

        Args:
            program: The ``Program`` AST node produced by the parser.

        Returns:
            A semantically validated ``Program`` node with uniquely renamed
            variable references throughout.

        Raises:
            SemanticError: If the program contains any semantic violation such
                as a duplicate variable declaration, a reference to an undeclared
                variable or function, a redefinition of a function, a call to a
                function whose declaration has not yet appeared in source order,
                conflicting linkage on a file-scope variable, or a duplicate
                file-scope definition.
        """
        resolved: list[ast.Declaration] = []
        for decl in program.declarations:
            match decl:
                case ast.FunDecl(function_declaration=fn):
                    file_scope_state = dict(self._variable_map)
                    file_scope_struct_state = dict(self._structure_map)
                    fn = self._register_fun_declaration(fn)
                    resolved.append(ast.FunDecl(self._resolve_function(fn)))
                    self._variable_map = file_scope_state
                    self._structure_map = file_scope_struct_state
                    self._current_block_funs = set()
                case ast.VarDecl(variable_declaration=var_decl):
                    resolved.append(ast.VarDecl(self._resolve_file_scope_variable_declaration(var_decl)))
                case ast.StructDecl(struct_declaration=struct_decl):
                    tag = struct_decl.tag
                    if tag in self._union_map:
                        raise SemanticError(f"Tag '{tag}' already declared as a union in this scope.")
                    existing = self._structure_map.get(tag)
                    if existing is not None and not existing.from_current_scope:
                        unique_tag = existing.new_tag
                    else:
                        unique_tag = self._make_struct_tag(tag)
                        self._structure_map[tag] = StructureBinding(new_tag=unique_tag, from_current_scope=False)
                    resolved_members = tuple(
                        MemberDeclaration(m.member_name, self._resolve_type(m.member_type)) for m in struct_decl.members
                    )
                    resolved_struct_decl = StructDeclaration(tag=unique_tag, members=resolved_members)
                    self._typecheck_struct_declaration(resolved_struct_decl)
                    resolved.append(ast.StructDecl(resolved_struct_decl))
                case ast.UnionDecl(union_declaration=union_decl):
                    tag = union_decl.tag
                    if tag in self._structure_map:
                        raise SemanticError(f"Tag '{tag}' already declared as a struct in this scope.")
                    existing = self._union_map.get(tag)
                    if existing is not None and not existing.from_current_scope:
                        unique_tag = existing.new_tag
                    else:
                        unique_tag = self._make_union_tag(tag)
                        self._union_map[tag] = StructureBinding(new_tag=unique_tag, from_current_scope=False)
                    resolved_members = tuple(
                        MemberDeclaration(m.member_name, self._resolve_type(m.member_type)) for m in union_decl.members
                    )
                    resolved_union_decl = UnionDeclaration(tag=unique_tag, members=resolved_members)
                    self._typecheck_union_declaration(resolved_union_decl)
                    resolved.append(ast.UnionDecl(resolved_union_decl))
                case _:
                    raise SemanticError("Unhandled top-level declaration type in semantic analyser")
        return ast.Program(declarations=tuple(resolved))


def resolve(program: ast.Program) -> ast.Program:
    """Convenience entry point for the semantic analysis stage.

    Constructs a fresh ``SemanticAnalyser`` and immediately runs it over
    ``program``.  Callers that need access to the analyser instance (e.g. for
    testing individual helpers) should instantiate ``SemanticAnalyser``
    directly.

    For pipeline use (where the symbol table is needed by downstream stages)
    prefer ``resolve_with_symbols``.

    Args:
        program: The ``Program`` AST node produced by the parser.

    Returns:
        A semantically validated ``Program`` node with uniquely renamed
        variable references throughout.

    Raises:
        SemanticError: If the program contains any semantic violation.
    """
    return SemanticAnalyser()._resolve_program(program)


def resolve_with_symbols(program: ast.Program) -> tuple[ast.Program, SymbolTable, StructTypeTable]:
    """Semantic analysis stage entry point that also returns the populated symbol and type tables.

    Identical to ``resolve`` but additionally exposes the ``SymbolTable`` and
    ``StructTypeTable`` built during analysis.  Downstream stages (TAC generation,
    code generation) require the symbol table to emit ``StaticVariable`` constructs
    and the struct type table to determine member offsets and struct sizes.

    Args:
        program: The ``Program`` AST node produced by the parser.

    Returns:
        A tuple ``(resolved_program, symbol_table, struct_type_table)`` where
        ``resolved_program`` is the semantically validated AST, ``symbol_table``
        maps every resolved identifier to its ``SymbolTableEntry``, and
        ``struct_type_table`` maps every resolved struct tag to its
        ``StructEntry`` layout record.

    Raises:
        SemanticError: If the program contains any semantic violation.
    """
    analyser = SemanticAnalyser()
    resolved = analyser._resolve_program(program)
    return resolved, analyser._symbol_table, analyser._struct_type_table
