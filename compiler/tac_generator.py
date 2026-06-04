"""TAC generation stage: translates the main AST into a TAC intermediate representation.

Performs a structural translation from ``compiler.syntax.ast`` nodes to
``compiler.syntax.tac`` nodes, flattening nested expression trees into sequences
of three-address instructions.  Temporary variables (``Var`` nodes) are created
on demand and named ``tmp.0``, ``tmp.1``, etc.

The translation follows this mapping:

    ast.Program(decls)                    → tac.Program(top_level=[...])
    ast.FunDecl(fn) [with body]           → tac.Function(name, globl, params, instructions)
    ast.VarDecl(v)  [file-scope/static]   → tac.StaticVariable(name, globl, init=(StaticInit, ...))
    ast.Binary(Add, ptr, int)  [ptr result] → <emit(ptr)> + <emit(int)>
                                               + [AddPtr(ptr_val, int_val, sizeof(*ptr))]
    ast.Binary(Sub, ptr, int)  [ptr result] → <emit(ptr)> + <emit(int)>
                                               + [Unary(Negate, int_val, neg_tmp)]
                                               + [AddPtr(ptr_val, neg_tmp, sizeof(*ptr))]
    ast.Binary(Sub, ptr, ptr)  [long result]→ <emit(p1)> + <emit(p2)>
                                               + [Binary(Subtract, p1, p2, diff_tmp)]
                                               + [Binary(Divide, diff_tmp, sizeof(*ptr), result)]
    ast.CompoundInit({e0,e1,...})         → [CopyToOffset(v0, name, 0),
                                             CopyToOffset(v1, name, sizeof(e0)),
                                             CopyToOffset(v2, name, sizeof(e0)+sizeof(e1)), ...]
    ast.SingleInit(String(s))             → [CopyToOffset(ConstLong(le8), name, 0), ...
      [auto char array]                      CopyToOffset(ConstInt(le4), name, n), ...
                                             CopyToOffset(ConstChar(b), name, m), ...]
                                            (greedy packing, little-endian, NUL+zero-pad if len<size)
    ast.Return(exp)                       → <emit(exp)> + [tac.Return(result_val)]
    ast.Constant(v)                       → tac.Constant(v)   (no extra instructions)
    ast.Unary(op, inner)                  → <emit(inner)> + [tac.Unary(op, inner_val, tmp)]
                                            returns tmp as the result val
    ast.Unary(PreIncrement, x)            → [Binary(Add, x, 1, tmp), Copy(tmp, x)]
                                            returns x (new value)
    ast.Unary(PreDecrement, x)            → [Binary(Sub, x, 1, tmp), Copy(tmp, x)]
                                            returns x (new value)
    ast.Unary(PostIncrement,x)            → [Copy(x, old), Binary(Add, x, 1, tmp), Copy(tmp, x)]
                                            returns old (original value)
    ast.Unary(PostDecrement,x)            → [Copy(x, old), Binary(Sub, x, 1, tmp), Copy(tmp, x)]
                                            returns old (original value)
    ast.Binary(op, l, r)                  → <emit(l)> + <emit(r)>
      (arithmetic / bitwise /               + [tac.Binary(op, l_val, r_val, tmp)]
       comparison operators)                returns tmp as the result val
    ast.Binary(And, l, r)                 → <emit(l)>
                                            + [JumpIfZero(l_val, false_lbl)]
                                            + <emit(r)>
                                            + [JumpIfZero(r_val, false_lbl),
                                               Copy(1, tmp), Jump(end_lbl),
                                               false_lbl, Copy(0, tmp), end_lbl]
                                            returns tmp as the result val
    ast.AddrOf(String(s))                 → sym[".Lstr.N"] = ConstantAttr(StringInit(s, True))
                                            + [GetAddress(Var(".Lstr.N"), tmp)]
                                            returns tmp (pointer-to-char-typed)
    ast.AddrOf(Dereference(inner))        → <emit(inner)>   (&* cancels out)
    ast.AddrOf(Subscript(ptr, idx))       → <ptr+idx AddPtr, no Load>  (&arr[i] ≡ arr+i)
    ast.AddrOf(Var(name))                 → [GetAddress(Var(name), tmp)]
                                            returns tmp (pointer-typed)
    ast.AddrOf(Dot(structure, member))    → _addr_of_lvalue(structure)
                                            + [AddPtr(base_ptr, member_offset, 1, tmp)]  (if offset > 0)
                                            returns tmp (pointer-to-member-typed)
    ast.AddrOf(Arrow(pointer, member))    → <emit(pointer)>   (&ptr->m ≡ ptr + offset)
                                            + [AddPtr(ptr_val, member_offset, 1, tmp)]  (if offset > 0)
                                            returns tmp (pointer-to-member-typed)
    ast.Dot(structure, member)            → <emit(structure)>
                                            + [CopyFromOffset(struct_var, member_offset, tmp)]
                                            returns tmp
    ast.Arrow(pointer, member)            → <emit(pointer)>
                                            + [AddPtr(ptr_val, member_offset, 1, member_ptr)]  (if offset > 0)
                                            + [Load(member_ptr, tmp)]
                                            returns tmp
    ast.Assignment(Dot(Var(n), member),   → <emit(rvalue)>
                   rvalue)                  + [CopyToOffset(rval, n, member_offset)]
    ast.Assignment(Dot(structure, member),→ <emit(rvalue)> + _addr_of_lvalue(structure)
                   rvalue)                  + [AddPtr(base, member_offset, 1, ptr)]  (if offset > 0)
                                            + [Store(rval, ptr)]
    ast.Assignment(Arrow(ptr, member),    → <emit(rvalue)> + <emit(ptr)>
                   rvalue)                  + [AddPtr(ptr_val, member_offset, 1, member_ptr)]  (if offset > 0)
                                            + [Store(rval, member_ptr)]
    ast.Dereference(inner)                → <emit(inner)>
                                            + [Load(inner_val, tmp)]
                                            returns tmp
    ast.Assignment(Dereference(ptr_exp),  → <emit(rvalue)> + <emit(ptr_exp)>
                   rvalue)                  + [Store(rval, ptr_val)]
                                            returns rval
    ast.Binary(Or, l, r)                  → <emit(l)>
                                            + [JumpIfNotZero(l_val, true_lbl)]
                                            + <emit(r)>
                                            + [JumpIfNotZero(r_val, true_lbl),
                                               Copy(0, tmp), Jump(end_lbl),
                                               true_lbl, Copy(1, tmp), end_lbl]
                                            returns tmp as the result val
    ast.If(cond, then, None)              → <emit(cond)>
                                            + [JumpIfZero(cond_val, end_lbl)]
                                            + <emit(then)>
                                            + [end_lbl]
    ast.Conditional(c, e1, e2)            → <emit(c)>
                                            + [JumpIfZero(c_val, false_lbl)]
                                            + <emit(e1)>
                                            + [Copy(e1_val, tmp), Jump(end_lbl), false_lbl]
                                            + <emit(e2)>
                                            + [Copy(e2_val, tmp), end_lbl]
                                            returns tmp as the result val
    ast.If(cond, then, else_)             → <emit(cond)>
                                            + [JumpIfZero(cond_val, else_lbl)]
                                            + <emit(then)>
                                            + [Jump(end_lbl), else_lbl]
                                            + <emit(else_)>
                                            + [end_lbl]
    ast.Goto(target)                      → [Jump(target)]
    ast.Label(name, stmt)                 → [Label(name)] + <emit(stmt)>
    ast.Compound(body)                    → <emit(item_0)> + <emit(item_1)> + ...
    ast.While(cond,body,id)                → [continue_<id>, <emit(cond)>,
                                             JumpIfZero(cond_val, break_<id>),
                                             <emit(body)>, Jump(continue_<id>), break_<id>]
    ast.DoWhile(body,cond,id)             → [start_<id>, <emit(body)>, continue_<id>,
                                             <emit(cond)>,
                                             JumpIfNotZero(cond_val, start_<id>), break_<id>]
    ast.For(init,cond,post,               → <emit(init)>
            body,id)                        + [start_<id>, <emit(cond)>,
                                               JumpIfZero(cond_val, break_<id>),
                                               <emit(body)>, continue_<id>,
                                               <emit(post)>, Jump(start_<id>), break_<id>]
    ast.Break(target)                     → [Jump("break_"+ target)]
    ast.Continue(target)                  → [Jump("continue_" + target)]
    ast.Switch(ctrl,body,id)              → <emit(ctrl)>
                                             + for each Case(v) in body (in order):
                                                 [tmp = Binary(Equal, ctrl_val, v),
                                                  JumpIfNotZero(tmp, "case_<id>_<v>")]
                                             + [Jump("default_<id>")] if Default present
                                             + [Jump("break_<id>")] otherwise
                                             + <emit(body)>
                                             + [break_<id>]
    ast.Case(v, stmt, id)                 → [Label("case_<id>_<v>")] + <emit(stmt)>
    ast.Default(stmt, id)                 → [Label("default_<id>")] + <emit(stmt)>

The resulting ``tac.Program`` is a complete TAC AST containing all ``TopLevel``
definitions and is ready for the assembly code generation stage.
"""

from dataclasses import dataclass
from itertools import count

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from compiler.types.symbols import (
    CharInit as SymCharInit,
)
from compiler.types.symbols import (
    ConstantAttr,
    Initial,
    LocalAttr,
    NoInitializer,
    PointerInit,
    StaticAttr,
    StructTypeTable,
    SymbolTable,
    SymbolTableEntry,
    Tentative,
    ULongInit,
)
from compiler.types.symbols import (
    DoubleInit as SymDoubleInit,
)
from compiler.types.symbols import (
    IntInit as SymIntInit,
)
from compiler.types.symbols import (
    LongInit as SymLongInit,
)
from compiler.types.symbols import (
    StaticInit as SymStaticInit,
)
from compiler.types.symbols import (
    StringInit as SymStringInit,
)
from compiler.types.symbols import (
    UCharInit as SymUCharInit,
)
from compiler.types.symbols import (
    ZeroInit as SymZeroInit,
)
from compiler.utils.exceptions import CodeGenerationError


def generate(
    program: ast.Program,
    symbol_table: SymbolTable,
    struct_type_table: StructTypeTable | None = None,
) -> tac.Program:
    """Translate an AST ``Program`` into a TAC ``Program``.

    File-scope variable declarations and block-scope ``static``/``extern``
    declarations are not translated directly from the AST.  Instead, after all
    function bodies have been processed, ``_convert_symbols_to_tacky`` walks the
    symbol table and emits a ``tac.StaticVariable`` for each entry that carries a
    ``StaticAttr`` with an ``Initial`` or ``Tentative`` initial value.

    Args:
        program: The ``Program`` node produced by the semantic analysis stage.
        symbol_table: The symbol table populated by the semantic analysis stage.
        struct_type_table: Mapping from struct tag to layout, used to compute member
            offsets for ``Dot`` and ``Arrow`` expressions.  Defaults to empty (safe
            for programs that contain no struct member access).

    Returns:
        A ``tac.Program`` node representing the equivalent TAC program.
        Forward declarations (functions without a body) are skipped.

    Raises:
        CodeGenerationError: If any construct cannot be translated.
    """
    gen = _TacGenerator(symbol_table, struct_type_table or {})
    functions = [
        gen.translate_function(decl.function_declaration)
        for decl in program.declarations
        if isinstance(decl, ast.FunDecl) and decl.function_declaration.body is not None
    ]
    static_vars = _convert_symbols_to_tacky(symbol_table, struct_type_table or {})
    return tac.Program(top_level=tuple(static_vars) + tuple(functions))


def _sym_const_init_to_tac(sym_init: SymStaticInit) -> tac.StaticInit:
    """Convert a symbol-table ``StaticInit`` to its TAC-layer equivalent.

    Used when lowering ``ConstantAttr`` entries to ``tac.StaticConstant`` nodes.
    ``CharInit`` and ``UCharInit`` are both mapped to ``tac.IntInit`` because the
    TAC layer carries no byte-width init variant.

    Args:
        sym_init: A ``compiler.types.symbols.StaticInit`` instance.

    Returns:
        The corresponding ``tac.StaticInit`` node.

    Raises:
        CodeGenerationError: If the init variant has no TAC equivalent.
    """
    match sym_init:
        case SymIntInit(value=v):
            return tac.IntInit(v)
        case SymLongInit(value=v):
            return tac.LongInit(v)
        case SymDoubleInit(value=v):
            return tac.DoubleInit(v)
        case SymCharInit(value=v) | SymUCharInit(value=v):
            return tac.IntInit(v)
        case SymZeroInit(bytes=n):
            return tac.ZeroInit(n)
        case SymStringInit(string=s, null_terminated=nt):
            return tac.StringInit(s, nt)
        case _:
            raise CodeGenerationError(f"Cannot lower symbol StaticInit {sym_init!r} to TAC.")


def _convert_symbols_to_tacky(
    symbol_table: SymbolTable,
    struct_type_table: StructTypeTable | None = None,
) -> list[tac.TopLevel]:
    """Emit TAC top-level nodes for every static-storage symbol in the symbol table.

    Walks all entries in ``symbol_table``.  Entries that carry a ``StaticAttr``
    with an ``Initial`` or ``Tentative`` initial value produce a
    ``tac.StaticVariable``; ``ConstantAttr`` entries (compiler-generated
    read-only constants such as string literals and double constants) produce a
    ``tac.StaticConstant``; entries with ``NoInitializer`` (non-defining
    ``extern`` declarations) or ``FunAttr`` entries are silently skipped.

    Args:
        symbol_table: The symbol table returned by the semantic analysis stage.
        struct_type_table: Mapping from struct tag to layout, used to compute the
            zero-initialisation size of tentative struct-typed static variables.

    Returns:
        A list of ``tac.TopLevel`` nodes, in symbol-table insertion order.
    """
    result: list[tac.TopLevel] = []
    for name, entry in symbol_table.items():
        is_double = isinstance(entry.t, ast.DoubleType)
        match entry.attrs:
            case StaticAttr(init=Initial(initializers=inits), globl=globl):
                static_inits: list[tac.StaticInit] = []
                skip = False
                for elem in inits:
                    match elem:
                        case SymIntInit(value=v):
                            static_inits.append(tac.IntInit(v))
                        case SymLongInit(value=v):
                            static_inits.append(tac.LongInit(v))
                        case SymCharInit(value=v) | SymUCharInit(value=v):
                            static_inits.append(tac.CharInit(v))
                        case SymDoubleInit(value=v):
                            static_inits.append(tac.DoubleInit(v))
                        case SymZeroInit(bytes=n):
                            # Doubles: preserve DoubleInit so -0.0 vs +0.0 is not collapsed.
                            static_inits.append(tac.DoubleInit(0.0) if is_double else tac.ZeroInit(n))
                        case SymStringInit(string=s, null_terminated=nt):
                            static_inits.append(tac.StringInit(s, nt))
                        case PointerInit(name=target):
                            static_inits.append(tac.PointerInit(target))
                        case _:
                            skip = True
                            break
                if not skip:
                    result.append(tac.StaticVariable(name=name, globl=globl, t=entry.t, init=tuple(static_inits)))
            case StaticAttr(init=Tentative(), globl=globl):
                n_bytes = _type_size(entry.t, struct_type_table)
                result.append(tac.StaticVariable(name=name, globl=globl, t=entry.t, init=(tac.ZeroInit(n_bytes),)))
            case StaticAttr(init=ULongInit(), globl=globl):
                # Null pointer constant: pointer-sized (8-byte) zero.
                result.append(tac.StaticVariable(name=name, globl=globl, t=entry.t, init=(tac.LongInit(0),)))
            case StaticAttr(init=PointerInit(name=target), globl=globl):
                result.append(tac.StaticVariable(name=name, globl=globl, t=entry.t, init=(tac.PointerInit(target),)))
            case StaticAttr(init=NoInitializer()):
                pass
            case ConstantAttr(init=sym_init):
                result.append(tac.StaticConstant(name=name, t=entry.t, init=_sym_const_init_to_tac(sym_init)))
            case _:
                pass
    return result


def _translate_binary_op(op: ast.BinaryOperator) -> tac.BinaryOperator:
    """Translate an AST binary operator into its TAC equivalent.

    Args:
        op: The AST binary operator node.

    Returns:
        The corresponding ``tac.BinaryOperator`` node.

    Raises:
        CodeGenerationError: If the operator variant is not supported.
    """
    match op:
        case ast.Add():
            return tac.Add()
        case ast.Subtract():
            return tac.Subtract()
        case ast.Multiply():
            return tac.Multiply()
        case ast.Divide():
            return tac.Divide()
        case ast.Remainder():
            return tac.Remainder()
        case ast.LeftShift():
            return tac.LeftShift()
        case ast.RightShift():
            return tac.RightShift()
        case ast.BitwiseAnd():
            return tac.BitwiseAnd()
        case ast.BitwiseXor():
            return tac.BitwiseXor()
        case ast.BitwiseOr():
            return tac.BitwiseOr()
        case ast.Equal():
            return tac.Equal()
        case ast.NotEqual():
            return tac.NotEqual()
        case ast.LessThan():
            return tac.LessThan()
        case ast.LessOrEqual():
            return tac.LessOrEqual()
        case ast.GreaterThan():
            return tac.GreaterThan()
        case ast.GreaterOrEqual():
            return tac.GreaterOrEqual()
        case _:
            raise CodeGenerationError(f"unsupported binary operator: {type(op).__name__}")


def _translate_unary_op(op: ast.UnaryOperator) -> tac.UnaryOperator:
    """Translate an AST unary operator into its TAC equivalent.

    Args:
        op: The AST unary operator node.

    Returns:
        The corresponding ``tac.UnaryOperator`` node.

    Raises:
        CodeGenerationError: If the operator variant is not supported.
    """
    match op:
        case ast.Complement():
            return tac.Complement()
        case ast.Negate():
            return tac.Negate()
        case ast.Not():
            return tac.Not()
        case _:
            raise CodeGenerationError(f"unsupported unary operator: {type(op).__name__}")


def _type_size(t: ast.Type, struct_type_table: StructTypeTable | None = None) -> int:
    """Return the size in bytes of a C type.

    Pointer types are treated as pointer-sized (8 bytes), identical to ``unsigned long``.
    ``Structure`` types require ``struct_type_table`` to be supplied.

    Args:
        t: The C type whose size is required.
        struct_type_table: Mapping from struct tag to layout.  Required when ``t`` is or
            contains a ``Structure``; may be omitted for purely scalar/pointer types.

    Returns:
        1 for ``char``/``signed char``/``unsigned char``; 4 for ``int``/``unsigned int``;
        8 for ``long``/``unsigned long``/any pointer type; the padded struct size for
        ``Structure``; ``element_size * count`` for ``Array``.

    Raises:
        CodeGenerationError: If the type is not recognised, or a struct tag is unknown.
    """
    if isinstance(t, (ast.Char, ast.SChar, ast.UChar)):
        return 1
    if isinstance(t, (ast.IntType, ast.UIntType)):
        return 4
    if isinstance(t, (ast.LongType, ast.ULongType, ast.Pointer, ast.DoubleType)):
        return 8
    if isinstance(t, ast.Array):
        return t.size * _type_size(t.element, struct_type_table)
    if isinstance(t, ast.Structure):
        if struct_type_table is None:
            raise CodeGenerationError(f"struct type table required to size struct '{t.tag}'")
        entry = struct_type_table.get(t.tag)
        if entry is None:
            raise CodeGenerationError(f"unknown struct tag '{t.tag}' in size calculation")
        return entry.size
    raise CodeGenerationError(f"unsupported type in size calculation: {type(t).__name__}")


def _is_signed(t: ast.Type) -> bool:
    """Return whether a C integer or pointer type is signed.

    Pointer types are treated as unsigned (equivalent to ``unsigned long``).

    Args:
        t: The C type to inspect.

    Returns:
        ``True`` for ``int``, ``long``, ``char``, and ``signed char``; ``False`` for
        ``unsigned int``, ``unsigned long``, ``unsigned char``, and any pointer type.

    Raises:
        CodeGenerationError: If the type is not a scalar integer or pointer type.
    """
    if isinstance(t, (ast.IntType, ast.LongType, ast.Char, ast.SChar)):
        return True
    if isinstance(t, (ast.UIntType, ast.ULongType, ast.Pointer, ast.UChar)):
        return False
    raise CodeGenerationError(f"unsupported type in signed check: {type(t).__name__}")


# ---------------------------------------------------------------------------
# Abstract lvalue representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _PlainOperand:
    """A concrete TAC value — a scalar variable or constant."""

    val: tac.Val


@dataclass(frozen=True, slots=True)
class _SubObject:
    """A sub-region of a named aggregate variable at a compile-time byte offset.

    Produced by member-access on a ``Var``-based struct lvalue.  The offset
    accumulates across nested ``Dot`` expressions so that ``s.a.b`` becomes a
    single ``SubObject("s", offset_a + offset_b)`` without emitting any
    intermediate ``CopyFromOffset``.

    Attributes:
        base: Name of the aggregate variable in the symbol table.
        offset: Byte offset from the start of ``base`` to this sub-object.
    """

    base: str
    offset: int


@dataclass(frozen=True, slots=True)
class _DereferencedPointer:
    """A value reached by dereferencing a pointer.

    Produced by ``Arrow`` or ``Dot`` on a pointer-backed struct.  Writes become
    ``Store``, reads become ``Load``, and taking the address cancels the
    dereference.

    Attributes:
        ptr: TAC value that holds the address of the object.
    """

    ptr: tac.Val


_AbstractObject = _PlainOperand | _SubObject | _DereferencedPointer


class _TacGenerator:
    """Stateful TAC generator for a single function.

    Maintains a temporary-variable counter and a label counter, both used to
    assign unique names throughout the translation.  Instantiate once per
    function to ensure fresh temporaries and labels for each translation unit.
    """

    def __init__(self, symbol_table: SymbolTable, struct_type_table: StructTypeTable | None = None) -> None:
        """Initialise the generator with zeroed temporary-variable and label counters.

        Args:
            symbol_table: The program-wide symbol table, mutated in-place to record
                compiler-generated temporaries (e.g. cast destinations).
            struct_type_table: Mapping from struct tag to layout, used to compute member
                offsets for ``Dot`` and ``Arrow`` expressions.  Defaults to empty.
        """
        self._counter = count()
        self._label_counter = count()
        self._function_name: str = ""
        self._symbol_table = symbol_table
        self._struct_type_table: StructTypeTable = struct_type_table or {}

    def _make_temp(self, t: ast.Type = ast.IntType()) -> tac.Var:
        """Create a fresh temporary variable, register it in the symbol table, and advance the counter.

        Every temporary is registered under ``LocalAttr`` so that the assembly
        generation stage can determine operand size and stack allocation without
        re-deriving types from the instruction stream.

        Args:
            t: The C type of the temporary.  Defaults to ``ast.IntType()``; callers that
                produce a ``long``-typed result (e.g. ``Cast`` to ``ast.LongType``) must
                pass the correct type explicitly.

        Returns:
            A ``tac.Var`` named ``tmp.<n>`` where ``n`` is the next counter value.
        """
        name = f"tmp.{next(self._counter)}"
        self._symbol_table[name] = SymbolTableEntry(t=t, attrs=LocalAttr())
        return tac.Var(value=name)

    def _make_label(self, base: str) -> tac.Label:
        """Create a fresh label and advance the counter.

        Args:
            base: The prefix for the label name.

        Returns:
            A ``tac.Label`` named ``<base>.<n>`` where ``n`` is the next counter value.
        """
        return tac.Label(name=f"{base}.{next(self._label_counter)}")

    def _make_string_name(self) -> str:
        """Generate a unique name for a hidden string-literal constant.

        Produces identifiers of the form ``.Lstr.<n>``, mirroring the naming convention
        used by the semantic analysis stage.  Uses ``_counter`` to guarantee uniqueness
        across temporaries and string names within the translation unit.

        Returns:
            A string of the form ``".Lstr.<n>"`` where ``n`` is the next value drawn from
            ``_counter``.
        """
        return f".Lstr.{next(self._counter)}"

    def _size_of(self, t: ast.Type) -> int:
        """Return the byte size of ``t``, forwarding the struct type table for struct types.

        Args:
            t: The C type to measure.

        Returns:
            The size in bytes as reported by ``_type_size``.
        """
        return _type_size(t, self._struct_type_table)

    def _get_member(self, tag: str, member_name: str) -> tuple[int, ast.Type]:
        """Return the byte offset and declared type of a named member within a struct.

        Args:
            tag: The struct tag to look up in the struct type table.
            member_name: The name of the member to locate.

        Returns:
            A ``(offset, member_type)`` tuple where ``offset`` is the byte offset from
            the start of the struct and ``member_type`` is the C type of the member.

        Raises:
            CodeGenerationError: If ``tag`` is absent from the struct type table or
                ``member_name`` is not declared on that struct.
        """
        entry = self._struct_type_table.get(tag)
        if entry is None:
            raise CodeGenerationError(f"unknown struct tag '{tag}'")
        for m in entry.members:
            if m.name == member_name:
                return m.offset, m.member_type
        raise CodeGenerationError(f"struct '{tag}' has no member '{member_name}'")

    def _eval_exp(self, exp: ast.Exp) -> tuple[_AbstractObject, list[tac.Instruction]]:
        """Lazily evaluate an expression into an abstract lvalue object.

        Unlike ``translate_exp``, this method does not immediately materialise
        struct-typed sub-expressions.  Consecutive ``Dot`` accesses accumulate
        into a single ``_SubObject`` offset; pointer-based accesses fold into a
        ``_DereferencedPointer``.  The caller converts the result to a concrete
        TAC value with ``_force_val``, to an address with ``_addr_of_obj``, or
        to an assignment instruction with ``_emit_assign``.

        Args:
            exp: The expression to evaluate lazily.

        Returns:
            A ``(_AbstractObject, instructions)`` tuple.  ``instructions`` must be
            emitted before the abstract object is used.

        Raises:
            CodeGenerationError: If any sub-expression cannot be translated.
        """
        match exp:
            case ast.Var(name=name):
                return _PlainOperand(tac.Var(value=name)), []
            case ast.Dereference(exp=inner):
                ptr_val, ptr_instrs = self.translate_exp(inner)
                return _DereferencedPointer(ptr=ptr_val), ptr_instrs
            case ast.Subscript(exp1=exp1, exp2=exp2):
                if isinstance(exp1.exp_type, ast.Pointer):
                    ptr_node, int_node = exp1, exp2
                else:
                    ptr_node, int_node = exp2, exp1
                ptr_val, ptr_instrs = self.translate_exp(ptr_node)
                int_val, int_instrs = self.translate_exp(int_node)
                assert isinstance(ptr_node.exp_type, ast.Pointer)
                scale = self._size_of(ptr_node.exp_type.referenced)
                elem_ptr = self._make_temp(ptr_node.exp_type)
                return _DereferencedPointer(ptr=elem_ptr), [
                    *ptr_instrs,
                    *int_instrs,
                    tac.AddPtr(ptr=ptr_val, index=int_val, scale=scale, dst=elem_ptr),
                ]
            case ast.Dot(structure=structure, member=member):
                assert isinstance(structure.exp_type, ast.Structure)
                offset, _ = self._get_member(structure.exp_type.tag, member)
                inner_obj, inner_instrs = self._eval_exp(structure)
                match inner_obj:
                    case _PlainOperand(val=tac.Var(value=v)):
                        return _SubObject(base=v, offset=offset), inner_instrs  # ❶
                    case _SubObject(base=base, offset=inner_offset):
                        return _SubObject(base=base, offset=inner_offset + offset), inner_instrs  # ❷
                    case _DereferencedPointer(ptr=ptr):  # ❸
                        if offset == 0:
                            return _DereferencedPointer(ptr=ptr), inner_instrs
                        dst_ptr = self._make_temp(ast.Pointer(exp.exp_type or ast.IntType()))
                        return _DereferencedPointer(ptr=dst_ptr), [
                            *inner_instrs,
                            tac.AddPtr(ptr=ptr, index=tac.Constant(ast.ConstLong(offset)), scale=1, dst=dst_ptr),
                        ]
            case ast.Arrow(pointer=pointer, member=member):
                assert isinstance(pointer.exp_type, ast.Pointer)
                assert isinstance(pointer.exp_type.referenced, ast.Structure)
                offset, _ = self._get_member(pointer.exp_type.referenced.tag, member)
                ptr_val, ptr_instrs = self.translate_exp(pointer)
                if offset == 0:
                    return _DereferencedPointer(ptr=ptr_val), ptr_instrs
                dst_ptr = self._make_temp(ast.Pointer(exp.exp_type or ast.IntType()))
                return _DereferencedPointer(ptr=dst_ptr), [
                    *ptr_instrs,
                    tac.AddPtr(ptr=ptr_val, index=tac.Constant(ast.ConstLong(offset)), scale=1, dst=dst_ptr),
                ]
            case _:
                val, instrs = self.translate_exp(exp)
                return _PlainOperand(val=val), instrs

    def _force_val(self, obj: _AbstractObject, t: ast.Type) -> tuple[tac.Val, list[tac.Instruction]]:
        """Convert an abstract lvalue object into a concrete TAC value (read).

        Args:
            obj: The abstract object to materialise.
            t: The C type of the resulting value (used when allocating a fresh temporary).

        Returns:
            A ``(val, instructions)`` tuple.  ``instructions`` must be emitted before
            ``val`` is valid.
        """
        match obj:
            case _PlainOperand(val=val):
                return val, []
            case _SubObject(base=base, offset=offset):
                dst = self._make_temp(t)
                return dst, [tac.CopyFromOffset(src=base, offset=offset, dst=dst)]
            case _DereferencedPointer(ptr=ptr):
                dst = self._make_temp(t)
                return dst, [tac.Load(src_ptr=ptr, dst=dst)]

    def _addr_of_obj(self, obj: _AbstractObject, t: ast.Type) -> tuple[tac.Val, list[tac.Instruction]]:
        """Compute the address of an abstract lvalue object.

        For ``_DereferencedPointer`` the dereference and address-of cancel out and
        no instructions are emitted.  For ``_SubObject`` a ``GetAddress`` instruction
        plus an optional ``AddPtr`` are emitted.

        Args:
            obj: The abstract lvalue whose address is required.
            t: The C pointer type of the resulting address.

        Returns:
            A ``(address_val, instructions)`` tuple.

        Raises:
            CodeGenerationError: If ``obj`` cannot have its address taken.
        """
        match obj:
            case _PlainOperand(val=tac.Var(value=name)):
                dst = self._make_temp(t)
                return dst, [tac.GetAddress(src=tac.Var(value=name), dst=dst)]
            case _SubObject(base=base, offset=offset):
                dst = self._make_temp(t)
                get_addr = tac.GetAddress(src=tac.Var(value=base), dst=dst)
                if offset == 0:
                    return dst, [get_addr]
                return dst, [
                    get_addr,
                    tac.AddPtr(ptr=dst, index=tac.Constant(ast.ConstLong(offset)), scale=1, dst=dst),
                ]
            case _DereferencedPointer(ptr=ptr):
                return ptr, []  # & and * cancel
            case _:
                raise CodeGenerationError("cannot take address of non-lvalue abstract object")

    def _emit_assign(self, obj: _AbstractObject, val: tac.Val) -> list[tac.Instruction]:
        """Emit the assignment instruction(s) for writing ``val`` into ``obj``.

        Args:
            obj: The abstract lvalue to write to.
            val: The TAC value to store.

        Returns:
            A list of instructions that perform the assignment.

        Raises:
            CodeGenerationError: If ``obj`` is not a writable lvalue form.
        """
        match obj:
            case _PlainOperand(val=tac.Var() as dst_var):
                return [tac.Copy(src=val, dst=dst_var)]
            case _SubObject(base=base, offset=offset):
                return [tac.CopyToOffset(src=val, dst=base, offset=offset)]
            case _DereferencedPointer(ptr=ptr):
                return [tac.Store(src=val, dst_ptr=ptr)]
            case _:
                raise CodeGenerationError("cannot assign to non-lvalue abstract object")

    def translate_function(self, function: ast.Function) -> tac.Function:
        """Translate an AST ``Function`` into a TAC ``Function``.

        Args:
            function: The function definition node to translate.

        Returns:
            A ``tac.Function`` with the same name and a flattened instruction body.

        Raises:
            CodeGenerationError: If any block item in the body cannot be translated.
        """
        self._function_name = function.name
        instructions = self.translate_block(function.body)
        # Guard return for functions that don't supply a return value; omit if
        # the last emitted instruction is already a Return.
        if not instructions or not isinstance(instructions[-1], tac.Return):
            instructions.append(tac.Return(tac.Constant(ast.ConstInt(0))))
        return tac.Function(
            name=function.name,
            globl=function.globl,
            params=tuple(p.name for p in function.params),
            body=tuple(instructions),
            source_file=function.source_file,
            source_line=function.source_line,
        )

    def translate_block(self, block: ast.Block) -> list[tac.Instruction]:
        """Translate a sequence of block items into a flat list of TAC instructions.

        Iterates over every item in ``block`` and concatenates the instruction
        sequences produced by ``_translate_block_item``.  Used both for function
        bodies and for ``Compound`` statement bodies.

        Args:
            block: The ordered sequence of block items to translate.

        Returns:
            A list of ``tac.Instruction`` nodes equivalent to the entire block.

        Raises:
            CodeGenerationError: If any block item variant is not supported.
        """
        instructions: list[tac.Instruction] = []
        for item in block:
            instructions.extend(self._translate_block_item(item))
        return instructions

    def _translate_block_item(self, item: ast.BlockItem) -> list[tac.Instruction]:
        """Translate a single block item into a sequence of TAC instructions.

        Args:
            item: The block item node to translate.

        Returns:
            A list of ``tac.Instruction`` nodes equivalent to the block item.

        Raises:
            CodeGenerationError: If the block item variant is not supported.
        """
        match item:
            case ast.S(statement=stmt):
                return self.translate_statement(stmt)
            case ast.D(
                declaration=ast.VarDecl(
                    variable_declaration=ast.VariableDeclaration(name=name, init=init, storage_class=sc)
                )
            ):
                if sc in (ast.StorageClass.Static, ast.StorageClass.Extern):
                    return []  # emitted as StaticVariable via symbol table; no inline instructions
                if init is None:
                    return []
                match init:
                    case ast.SingleInit(exp=ast.String(string=s, exp_type=ast.Array(size=array_size))):
                        return self._translate_string_init(name, s, array_size)
                    case ast.SingleInit(exp=e):
                        val, instrs = self.translate_exp(e)
                        return [*instrs, tac.Copy(src=val, dst=tac.Var(value=name))]
                    case ast.CompoundInit():
                        return self._translate_compound_init(name, init)
                    case _:
                        raise CodeGenerationError(f"unsupported initializer variant: {type(init).__name__}")
            case ast.D(declaration=ast.FunDecl()):
                return []
            case ast.D(declaration=ast.StructDecl()):
                return []  # local struct type definition — no code, pure type metadata
            case _:
                raise CodeGenerationError(f"unsupported block item type: {type(item).__name__}")

    def _init_byte_size(self, init: ast.Initializer) -> int:
        """Return the total byte size covered by an initializer.

        For ``CompoundInit`` nodes targeting a struct the padded ``StructEntry.size`` is
        returned rather than summing sub-initializer sizes, which would under-count when
        the struct has inter-member or trailing alignment padding.

        Args:
            init: A ``SingleInit`` or ``CompoundInit`` node whose size is needed.

        Returns:
            For ``SingleInit``: the size of the wrapped expression's type.
            For struct ``CompoundInit``: the padded size of the struct from the type table.
            For array ``CompoundInit``: the sum of sizes of all sub-initializers.

        Raises:
            CodeGenerationError: If ``init`` is an unrecognised ``Initializer`` variant, contains
                an expression with no type annotation, or references an unknown struct tag.
        """
        match init:
            case ast.SingleInit(exp=e):
                return self._size_of(e.exp_type or ast.IntType())
            case ast.CompoundInit(init_type=ast.Structure(tag=tag)):
                entry = self._struct_type_table.get(tag)
                if entry is None:
                    raise CodeGenerationError(f"unknown struct tag '{tag}' in initializer")
                return entry.size
            case ast.CompoundInit(initializers=sub_inits):
                return sum(self._init_byte_size(sub) for sub in sub_inits)
            case _:
                raise CodeGenerationError(f"unsupported initializer type: {type(init).__name__}")

    def _emit_copy_to_offset(
        self,
        var_name: str,
        byte_offset: int,
        init: ast.Initializer,
        instructions: list[tac.Instruction],
    ) -> None:
        """Recursively emit ``CopyToOffset`` instructions for an initializer at a byte offset.

        Each scalar leaf emits one ``CopyToOffset`` with a compile-time offset.  Struct
        ``CompoundInit`` nodes use the member offsets from the struct type table so that
        alignment padding is respected.  Array ``CompoundInit`` nodes use a running size
        accumulator.

        Args:
            var_name: The resolved name of the aggregate variable (e.g. ``".Varr.0"``).
            byte_offset: Byte offset from the variable's base at which to write ``init``'s value.
            init: The initializer to emit — either a ``SingleInit`` (scalar leaf) or
                ``CompoundInit`` (nested aggregate).
            instructions: The instruction list to which new instructions are appended in-place.
        """
        match init:
            case ast.SingleInit(exp=ast.String(string=s, exp_type=ast.Array(size=array_size))):
                instructions.extend(self._translate_string_init(var_name, s, array_size, base_offset=byte_offset))
            case ast.SingleInit(exp=e):
                val, elem_instrs = self.translate_exp(e)
                instructions.extend(elem_instrs)
                instructions.append(tac.CopyToOffset(src=val, dst=var_name, offset=byte_offset))
            case ast.CompoundInit(initializers=sub_inits, init_type=ast.Structure(tag=tag)):
                entry = self._struct_type_table.get(tag)
                if entry is None:
                    raise CodeGenerationError(f"unknown struct tag '{tag}' in initializer")
                # Zero-fill the entire struct region first so that padding bytes are always 0.
                remaining = entry.size
                fill_off = byte_offset
                while remaining > 0:
                    if remaining >= 8:
                        zero: ast.Const = ast.ConstLong(0)
                        chunk = 8
                    elif remaining >= 4:
                        zero = ast.ConstInt(0)
                        chunk = 4
                    else:
                        zero = ast.ConstChar(0)
                        chunk = 1
                    instructions.append(tac.CopyToOffset(src=tac.Constant(zero), dst=var_name, offset=fill_off))
                    fill_off += chunk
                    remaining -= chunk
                # Write member values on top using each member's precise byte offset. ❷
                for sub_init, member in zip(sub_inits, entry.members):
                    self._emit_copy_to_offset(var_name, byte_offset + member.offset, sub_init, instructions)
            case ast.CompoundInit(initializers=sub_inits):
                offset = byte_offset
                for sub_init in sub_inits:
                    self._emit_copy_to_offset(var_name, offset, sub_init, instructions)
                    offset += self._init_byte_size(sub_init)

    def _translate_compound_init(self, name: str, init: ast.CompoundInit) -> list[tac.Instruction]:
        """Translate a ``CompoundInit`` for a local aggregate variable into ``CopyToOffset`` instructions.

        Emits one ``CopyToOffset`` per scalar element.  For struct targets, padding bytes are zeroed
        at each nesting level inside ``_emit_copy_to_offset`` before member values are written on top.
        The byte offset for each element is computed at compile time, so no runtime pointer arithmetic
        or ``GetAddress`` is needed.

        Args:
            name: The unique resolved name of the aggregate variable (e.g. ``".Varr.0"``).
            init: The ``CompoundInit`` node carrying the element or member initialisers.

        Returns:
            A list of ``tac.Instruction`` nodes that initialise every element/member of the aggregate,
            with padding bytes zeroed for struct targets.
        """
        instructions: list[tac.Instruction] = []
        self._emit_copy_to_offset(name, 0, init, instructions)
        return instructions

    def _translate_string_init(
        self, var_name: str, string: bytes, array_size: int, base_offset: int = 0
    ) -> list[tac.Instruction]:
        """Emit ``CopyToOffset`` instructions to initialise a local char array from a string literal.

        Bytes are packed greedily into the largest available constant type to minimise the number
        of instructions: 8-byte aligned chunks become ``ConstLong``, 4-byte chunks become
        ``ConstInt``, and individual bytes become ``ConstChar``.  Multi-byte values are packed
        in little-endian order to match the x86-64 C ABI.

        When ``len(string) < array_size``, a NUL terminator and zero padding are appended to fill
        the array.  When ``len(string) == array_size``, the bytes are written as-is with no NUL
        (C11 §6.7.9p15 — the NUL is omitted when the array is exactly the string length).

        Args:
            var_name: Resolved name of the local array variable (e.g. ``".Varr.0"``).
            string: Raw byte content of the string literal, NUL terminator not included.
            array_size: Declared size of the target character array in bytes.
            base_offset: Byte offset from ``var_name``'s base to start writing at.  Defaults to
                0 for top-level array declarations; set to the element's byte position when
                initialising a sub-array inside a ``CompoundInit``.

        Returns:
            A list of ``CopyToOffset`` instructions that write the string content into the array,
            using as few instructions as possible.
        """
        if len(string) < array_size:
            full = string + b"\x00" * (array_size - len(string))
        else:
            full = string

        instructions: list[tac.Instruction] = []
        rel_offset = 0
        remaining = array_size

        while remaining >= 8:
            val = int.from_bytes(full[rel_offset : rel_offset + 8], byteorder="little")
            instructions.append(
                tac.CopyToOffset(src=tac.Constant(ast.ConstLong(val)), dst=var_name, offset=base_offset + rel_offset)
            )
            rel_offset += 8
            remaining -= 8

        if remaining >= 4:
            val = int.from_bytes(full[rel_offset : rel_offset + 4], byteorder="little")
            instructions.append(
                tac.CopyToOffset(src=tac.Constant(ast.ConstInt(val)), dst=var_name, offset=base_offset + rel_offset)
            )
            rel_offset += 4
            remaining -= 4

        while remaining > 0:
            instructions.append(
                tac.CopyToOffset(
                    src=tac.Constant(ast.ConstChar(full[rel_offset])),
                    dst=var_name,
                    offset=base_offset + rel_offset,
                )
            )
            rel_offset += 1
            remaining -= 1

        return instructions

    def translate_statement(self, stmt: ast.Statement) -> list[tac.Instruction]:  # noqa: C901 - suppress function too complicated
        """Translate a single AST statement into a sequence of TAC instructions.

        Handles all statement forms supported by the compiler: ``Return``,
        ``Expression``, ``Null``, ``If``, ``Goto``, ``Label``, ``Compound``,
        ``Break``, ``Continue``, ``While``, ``DoWhile``, ``For``, ``Switch``,
        ``Case``, and ``Default``.

        Loop constructs use the identifier already assigned by the semantic
        analysis stage to derive consistent ``start_<id>``, ``continue_<id>``,
        and ``break_<id>`` label names.  ``Break`` and ``Continue`` statements
        carry those same identifiers and translate to unconditional jumps to the
        corresponding label.

        Args:
            stmt: The statement node to translate.

        Returns:
            A list of ``tac.Instruction`` nodes equivalent to the statement.

        Raises:
            CodeGenerationError: If the statement variant is not supported.
        """
        match stmt:
            case ast.Return(exp=None):
                return [tac.Return()]
            case ast.Return(exp=exp):
                val, extra = self.translate_exp(exp)
                return [*extra, tac.Return(val=val)]
            case ast.Expression(exp=exp):
                _, instr = self.translate_exp(exp)
                return instr
            case ast.Null():
                return []
            case ast.If(condition=cond, then=then_stmt, else_=None):
                cond_val, cond_instrs = self.translate_exp(cond)
                label_end = self._make_label("if_end")
                return [
                    *cond_instrs,
                    tac.JumpIfZero(condition=cond_val, target=label_end.name),
                    *self.translate_statement(then_stmt),
                    label_end,
                ]
            case ast.If(condition=cond, then=then_stmt, else_=else_stmt):
                cond_val, cond_instrs = self.translate_exp(cond)
                label_else = self._make_label("if_else")
                label_end = self._make_label("if_end")
                return [
                    *cond_instrs,
                    tac.JumpIfZero(condition=cond_val, target=label_else.name),
                    *self.translate_statement(then_stmt),
                    tac.Jump(target=label_end.name),
                    label_else,
                    *self.translate_statement(else_stmt),
                    label_end,
                ]
            case ast.Goto(target=target):
                return [tac.Jump(target=f"{self._function_name}.{target}")]
            case ast.Label(name=name, statement=inner):
                return [tac.Label(name=f"{self._function_name}.{name}"), *self.translate_statement(inner)]
            case ast.Compound(body=body):
                return self.translate_block(body)
            case ast.Break(identifier=identifier):
                return [tac.Jump(f"break_{identifier}")]
            case ast.Continue(identifier=identifier):
                return [tac.Jump(f"continue_{identifier}")]
            case ast.DoWhile(body=body, condition=condition, identifier=identifier):
                cond_val, cond_instrs = self.translate_exp(condition)
                label_start = tac.Label(f"start_{identifier}")
                label_continue = tac.Label(f"continue_{identifier}")
                label_break = tac.Label(f"break_{identifier}")
                return [
                    label_start,
                    *self.translate_statement(body),
                    label_continue,
                    *cond_instrs,
                    tac.JumpIfNotZero(condition=cond_val, target=label_start.name),
                    label_break,
                ]
            case ast.While(condition=condition, body=body, identifier=identifier):
                cond_val, cond_instrs = self.translate_exp(condition)
                label_continue = tac.Label(f"continue_{identifier}")
                label_break = tac.Label(f"break_{identifier}")
                return [
                    label_continue,
                    *cond_instrs,
                    tac.JumpIfZero(condition=cond_val, target=label_break.name),
                    *self.translate_statement(body),
                    tac.Jump(target=label_continue.name),
                    label_break,
                ]
            case ast.For(init=init, condition=condition, post=post, body=body, identifier=identifier):
                init_instrs = self._translate_for_init(init)
                cond_val, cond_instrs = self._translate_optional_exp(condition)
                _, post_instrs = self._translate_optional_exp(post)
                label_start = tac.Label(f"start_{identifier}")
                label_break = tac.Label(f"break_{identifier}")
                label_continue = tac.Label(f"continue_{identifier}")
                # When no condition is present the loop is unconditional; omit
                # the guard entirely rather than emitting JumpIfZero(Constant(1)).
                cond_guard = (
                    []
                    if condition is None
                    else [*cond_instrs, tac.JumpIfZero(condition=cond_val, target=label_break.name)]
                )
                return [
                    *init_instrs,
                    label_start,
                    *cond_guard,
                    *self.translate_statement(body),
                    label_continue,
                    *post_instrs,
                    tac.Jump(target=label_start.name),
                    label_break,
                ]
            case ast.Switch(control=control, body=body, identifier=identifier):
                ctrl_val, ctrl_instrs = self.translate_exp(control)
                cases = self._collect_switch_cases(body)
                dispatch: list[tac.Instruction] = list(ctrl_instrs)
                has_default = False
                for c in cases:
                    match c:
                        case ast.Case(value=ast.Constant(value=v)):
                            tmp = self._make_temp()
                            dispatch.append(
                                tac.Binary(operator=tac.Equal(), src1=ctrl_val, src2=tac.Constant(value=v), dst=tmp)
                            )
                            dispatch.append(tac.JumpIfNotZero(condition=tmp, target=f"case_{identifier}_{v.val}"))
                        case ast.Default():
                            has_default = True
                if has_default:
                    dispatch.append(tac.Jump(target=f"default_{identifier}"))
                else:
                    dispatch.append(tac.Jump(target=f"break_{identifier}"))
                return [
                    *dispatch,
                    *self.translate_statement(body),
                    tac.Label(name=f"break_{identifier}"),
                ]
            case ast.Case(value=ast.Constant(value=v), statement=inner, identifier=identifier):
                return [tac.Label(name=f"case_{identifier}_{v.val}"), *self.translate_statement(inner)]
            case ast.Default(statement=inner, identifier=identifier):
                return [tac.Label(name=f"default_{identifier}"), *self.translate_statement(inner)]
            case _:
                raise CodeGenerationError(f"unsupported statement type: {type(stmt).__name__}")

    def _translate_for_init(self, init: ast.ForInit) -> list[tac.Instruction]:
        """Translate the initialiser clause of a ``for`` loop into TAC instructions.

        An ``InitExp`` clause is treated as a plain expression statement: its
        instructions are emitted and the result value is discarded.  An absent
        expression (``InitExp(exp=None)``) produces no instructions.

        An ``InitDecl`` clause is treated identically to a top-level declaration:
        if the declaration has an initialiser the expression is evaluated and
        copied into the variable; if not, no instructions are emitted.

        Args:
            init: The ``ForInit`` node — either an ``InitExp`` or an ``InitDecl``.

        Returns:
            A flat list of ``tac.Instruction`` nodes that implement the
            initialiser.  The caller is responsible for emitting these before
            the loop's start label.

        Raises:
            CodeGenerationError: If the ``ForInit`` variant is not supported.
        """
        match init:
            case ast.InitExp(exp=exp):
                if exp is None:
                    return []
                _, instrs = self.translate_exp(exp)
                return instrs
            case ast.InitDecl(
                declaration=ast.VarDecl(
                    variable_declaration=ast.VariableDeclaration(
                        name=name,
                        init=decl_init,
                    )
                )
            ):
                if decl_init is None:
                    return []
                match decl_init:
                    case ast.SingleInit(exp=e):
                        val, instrs = self.translate_exp(e)
                        return [*instrs, tac.Copy(src=val, dst=tac.Var(value=name))]
                    case ast.CompoundInit():
                        return self._translate_compound_init(name, decl_init)
                    case _:
                        raise CodeGenerationError(f"unsupported initializer variant: {type(decl_init).__name__}")
            case _:
                raise CodeGenerationError(f"unsupported for-init type: {type(init).__name__}")

    def _translate_optional_exp(self, exp: ast.Exp | None) -> tuple[tac.Val, list[tac.Instruction]]:
        """Translate an optional expression, substituting a non-zero constant when absent.

        Used for the condition and post-expression slots of a ``for`` loop, both
        of which may be omitted.  An absent condition must be treated as always
        true so that ``for (;;)`` loops indefinitely; returning ``Constant(1)``
        ensures the ``JumpIfZero`` that guards the loop body never fires.  An
        absent post expression simply produces no instructions; the returned
        value is ignored by the caller in that case.

        Args:
            exp: The expression to translate, or ``None`` if the slot is absent.

        Returns:
            A tuple ``(val, instructions)`` where ``instructions`` must be
            emitted before ``val`` is used.  When ``exp`` is ``None``, returns
            ``(Constant(1), [])`` so that a missing condition is unconditionally
            true.
        """
        if exp is None:
            return tac.Constant(ast.ConstInt(1)), []
        return self.translate_exp(exp)

    def _collect_switch_cases(self, stmt: ast.Statement) -> list[ast.Case | ast.Default]:
        """Collect all ``Case`` and ``Default`` nodes from a switch body in source order.

        Walks the statement tree recursively, yielding every ``Case`` and
        ``Default`` encountered.  Recursion stops at nested ``Switch`` nodes so
        that inner-switch labels are not included in the outer jump table.

        Args:
            stmt: The statement to walk (typically the ``body`` of a ``Switch``).

        Returns:
            A list of ``ast.Case`` and ``ast.Default`` nodes in the order they
            appear in the source.
        """
        result: list[ast.Case | ast.Default] = []
        match stmt:
            case ast.Case() as c:
                result.append(c)
                result.extend(self._collect_switch_cases(c.statement))
            case ast.Default() as d:
                result.append(d)
                result.extend(self._collect_switch_cases(d.statement))
            case ast.Compound(body=body):
                for item in body:
                    match item:
                        case ast.S(statement=inner):
                            result.extend(self._collect_switch_cases(inner))
            case ast.Switch():
                pass
            case ast.If(then=then_stmt, else_=None):
                result.extend(self._collect_switch_cases(then_stmt))
            case ast.If(then=then_stmt, else_=else_stmt) if else_stmt is not None:
                result.extend(self._collect_switch_cases(then_stmt))
                result.extend(self._collect_switch_cases(else_stmt))
            case ast.While(body=body):
                result.extend(self._collect_switch_cases(body))
            case ast.DoWhile(body=body):
                result.extend(self._collect_switch_cases(body))
            case ast.For(body=body):
                result.extend(self._collect_switch_cases(body))
            case ast.Label(statement=inner):
                result.extend(self._collect_switch_cases(inner))
        return result

    def translate_exp(self, exp: ast.Exp) -> tuple[tac.Val, list[tac.Instruction]]:  # noqa: C901 - suppress function too complicated
        """Translate an AST expression into a TAC value and any preceding instructions.

        Compound expressions (unary and binary operations) are broken down into a
        sequence of TAC instructions that write their result to a fresh temporary
        variable, which is returned as the result value.  Leaf expressions
        (constants) produce no instructions.

        Args:
            exp: The expression node to translate.

        Returns:
            A tuple ``(val, instructions)`` where ``instructions`` must be emitted
            before ``val`` is valid, and ``val`` holds the computed result.

        Raises:
            CodeGenerationError: If the expression variant is not supported.
        """
        match exp:
            case ast.Cast(target_type=ast.VoidType(), exp=inner):
                _, inner_instrs = self.translate_exp(inner)
                return tac.Var(value="DUMMY"), inner_instrs
            case ast.Cast(target_type=t, exp=inner):
                result, inner_instrs = self.translate_exp(inner)
                inner_type = inner.exp_type
                if t == inner_type:
                    return result, inner_instrs
                dst = self._make_temp(t)
                if isinstance(t, ast.DoubleType):
                    if _is_signed(inner_type):  # type: ignore[arg-type]
                        conv: tac.Instruction = tac.IntToDouble(src=result, dst=dst)
                    else:
                        conv = tac.UIntToDouble(src=result, dst=dst)
                elif isinstance(inner_type, ast.DoubleType):
                    if _is_signed(t):  # type: ignore[arg-type]
                        conv = tac.DoubleToInt(src=result, dst=dst)
                    else:
                        conv = tac.DoubleToUInt(src=result, dst=dst)
                else:
                    dst_size = self._size_of(t)
                    src_size = self._size_of(inner_type)  # type: ignore[arg-type]
                    if dst_size == src_size:
                        conv = tac.Copy(src=result, dst=dst)
                    elif dst_size < src_size:
                        conv = tac.Truncate(src=result, dst=dst)
                    elif _is_signed(inner_type):  # type: ignore[arg-type]
                        conv = tac.SignExtend(src=result, dst=dst)
                    else:
                        conv = tac.ZeroExtend(src=result, dst=dst)
                return dst, [*inner_instrs, conv]
            case ast.SizeOf(exp=inner):
                return tac.Constant(ast.ConstULong(self._size_of(inner.exp_type or ast.IntType()))), []
            case ast.SizeOfT(t=t):
                return tac.Constant(ast.ConstULong(self._size_of(t))), []
            case ast.Constant(value=value):
                return tac.Constant(value=value), []
            case ast.Unary(operator=ast.PreIncrement(), exp=ast.Var(name=name)):
                var = tac.Var(value=name)
                t = exp.exp_type or ast.IntType()
                tmp = self._make_temp(t)
                one: ast.Const = ast.ConstDouble(1.0) if isinstance(t, ast.DoubleType) else ast.ConstInt(1)
                return var, [
                    tac.Binary(operator=tac.Add(), src1=var, src2=tac.Constant(one), dst=tmp),
                    tac.Copy(src=tmp, dst=var),
                ]
            case ast.Unary(operator=ast.PreDecrement(), exp=ast.Var(name=name)):
                var = tac.Var(value=name)
                t = exp.exp_type or ast.IntType()
                tmp = self._make_temp(t)
                one = ast.ConstDouble(1.0) if isinstance(t, ast.DoubleType) else ast.ConstInt(1)
                return var, [
                    tac.Binary(operator=tac.Subtract(), src1=var, src2=tac.Constant(one), dst=tmp),
                    tac.Copy(src=tmp, dst=var),
                ]
            case ast.Unary(operator=ast.PostIncrement(), exp=ast.Var(name=name)):
                var = tac.Var(value=name)
                t = exp.exp_type or ast.IntType()
                old = self._make_temp(t)
                tmp = self._make_temp(t)
                one = ast.ConstDouble(1.0) if isinstance(t, ast.DoubleType) else ast.ConstInt(1)
                return old, [
                    tac.Copy(src=var, dst=old),
                    tac.Binary(operator=tac.Add(), src1=var, src2=tac.Constant(one), dst=tmp),
                    tac.Copy(src=tmp, dst=var),
                ]
            case ast.Unary(operator=ast.PostDecrement(), exp=ast.Var(name=name)):
                var = tac.Var(value=name)
                t = exp.exp_type or ast.IntType()
                old = self._make_temp(t)
                tmp = self._make_temp(t)
                one = ast.ConstDouble(1.0) if isinstance(t, ast.DoubleType) else ast.ConstInt(1)
                return old, [
                    tac.Copy(src=var, dst=old),
                    tac.Binary(operator=tac.Subtract(), src1=var, src2=tac.Constant(one), dst=tmp),
                    tac.Copy(src=tmp, dst=var),
                ]
            case ast.Unary(operator=op, exp=inner):
                src, inner_instrs = self.translate_exp(inner)
                dst = self._make_temp(exp.exp_type or ast.IntType())
                return dst, [
                    *inner_instrs,
                    tac.Unary(operator=_translate_unary_op(op), src=src, dst=dst),
                ]
            case ast.Binary(operator=ast.And(), left=left, right=right):
                label_false = self._make_label("and_false")
                label_end = self._make_label("and_end")
                dst = self._make_temp()
                left_val, left_instrs = self.translate_exp(left)
                right_val, right_instrs = self.translate_exp(right)
                return dst, [
                    *left_instrs,
                    tac.JumpIfZero(condition=left_val, target=label_false.name),
                    *right_instrs,
                    tac.JumpIfZero(condition=right_val, target=label_false.name),
                    tac.Copy(src=tac.Constant(ast.ConstInt(1)), dst=dst),
                    tac.Jump(target=label_end.name),
                    label_false,
                    tac.Copy(src=tac.Constant(ast.ConstInt(0)), dst=dst),
                    label_end,
                ]
            case ast.Binary(operator=ast.Or(), left=left, right=right):
                label_true = self._make_label("or_true")
                label_end = self._make_label("or_end")
                dst = self._make_temp()
                left_val, left_instrs = self.translate_exp(left)
                right_val, right_instrs = self.translate_exp(right)
                return dst, [
                    *left_instrs,
                    tac.JumpIfNotZero(condition=left_val, target=label_true.name),
                    *right_instrs,
                    tac.JumpIfNotZero(condition=right_val, target=label_true.name),
                    tac.Copy(src=tac.Constant(ast.ConstInt(0)), dst=dst),
                    tac.Jump(target=label_end.name),
                    label_true,
                    tac.Copy(src=tac.Constant(ast.ConstInt(1)), dst=dst),
                    label_end,
                ]
            case ast.Binary(operator=ast.Add(), left=left, right=right) if isinstance(exp.exp_type, ast.Pointer):
                # Pointer addition: ptr + int OR int + ptr.  Pointer is always the AddPtr base.
                if isinstance(left.exp_type, ast.Pointer):
                    ptr_node, int_node = left, right
                else:
                    ptr_node, int_node = right, left
                ptr_val, ptr_instrs = self.translate_exp(ptr_node)
                int_val, int_instrs = self.translate_exp(int_node)
                assert isinstance(ptr_node.exp_type, ast.Pointer)
                scale = self._size_of(ptr_node.exp_type.referenced)
                dst = self._make_temp(exp.exp_type)
                return dst, [
                    *ptr_instrs,
                    *int_instrs,
                    tac.AddPtr(ptr=ptr_val, index=int_val, scale=scale, dst=dst),
                ]
            case ast.Binary(operator=ast.Subtract(), left=left, right=right) if isinstance(exp.exp_type, ast.Pointer):
                # Pointer subtraction: ptr - int.  Negate the index then use AddPtr.
                ptr_val, ptr_instrs = self.translate_exp(left)
                int_val, int_instrs = self.translate_exp(right)
                assert isinstance(left.exp_type, ast.Pointer)
                scale = self._size_of(left.exp_type.referenced)
                neg_tmp = self._make_temp(right.exp_type or ast.LongType())
                dst = self._make_temp(exp.exp_type)
                return dst, [
                    *ptr_instrs,
                    *int_instrs,
                    tac.Unary(operator=tac.Negate(), src=int_val, dst=neg_tmp),
                    tac.AddPtr(ptr=ptr_val, index=neg_tmp, scale=scale, dst=dst),
                ]
            case ast.Binary(operator=ast.Subtract(), left=left, right=right) if isinstance(
                left.exp_type, ast.Pointer
            ) and isinstance(right.exp_type, ast.Pointer):
                # Pointer-to-pointer subtraction: (p1 - p2) / sizeof(*p1).
                if left.exp_type != right.exp_type:
                    raise CodeGenerationError(f"Pointer subtraction type mismatch: {left.exp_type} vs {right.exp_type}")
                assert isinstance(left.exp_type, ast.Pointer)
                scale = self._size_of(left.exp_type.referenced)
                p1_val, p1_instrs = self.translate_exp(left)
                p2_val, p2_instrs = self.translate_exp(right)
                diff_tmp = self._make_temp(ast.LongType())
                dst = self._make_temp(exp.exp_type or ast.LongType())
                return dst, [
                    *p1_instrs,
                    *p2_instrs,
                    tac.Binary(operator=tac.Subtract(), src1=p1_val, src2=p2_val, dst=diff_tmp),
                    tac.Binary(operator=tac.Divide(), src1=diff_tmp, src2=tac.Constant(ast.ConstLong(scale)), dst=dst),
                ]
            case ast.Binary(operator=op, left=left, right=right):
                src1, left_instrs = self.translate_exp(left)
                src2, right_instrs = self.translate_exp(right)
                dst = self._make_temp(exp.exp_type or ast.IntType())
                binary = tac.Binary(operator=_translate_binary_op(op), src1=src1, src2=src2, dst=dst)
                return dst, [*left_instrs, *right_instrs, binary]
            case ast.Var(name=name):
                return tac.Var(value=name), []
            case ast.Assignment(lvalue=ast.Dereference(exp=ptr_exp), rvalue=rvalue):
                rval, rval_instrs = self.translate_exp(rvalue)
                ptr_val, ptr_instrs = self.translate_exp(ptr_exp)
                return rval, [*rval_instrs, *ptr_instrs, tac.Store(src=rval, dst_ptr=ptr_val)]
            case ast.Assignment(lvalue=ast.Subscript(exp1=exp1, exp2=exp2), rvalue=rvalue):
                rval, rval_instrs = self.translate_exp(rvalue)
                if isinstance(exp1.exp_type, ast.Pointer):
                    ptr_node, int_node = exp1, exp2
                else:
                    ptr_node, int_node = exp2, exp1
                ptr_val, ptr_instrs = self.translate_exp(ptr_node)
                int_val, int_instrs = self.translate_exp(int_node)
                assert isinstance(ptr_node.exp_type, ast.Pointer)
                scale = self._size_of(ptr_node.exp_type.referenced)
                elem_ptr = self._make_temp(ptr_node.exp_type)
                return rval, [
                    *rval_instrs,
                    *ptr_instrs,
                    *int_instrs,
                    tac.AddPtr(ptr=ptr_val, index=int_val, scale=scale, dst=elem_ptr),
                    tac.Store(src=rval, dst_ptr=elem_ptr),
                ]
            case ast.Assignment(lvalue=ast.Var(name=name), rvalue=rvalue):
                val, instrs = self.translate_exp(rvalue)
                dst = tac.Var(value=name)
                return dst, [*instrs, tac.Copy(src=val, dst=dst)]
            case ast.Assignment(lvalue=ast.Dot() | ast.Arrow() as lvalue_node, rvalue=rvalue):
                lval_obj, lval_instrs = self._eval_exp(lvalue_node)
                rval, rval_instrs = self.translate_exp(rvalue)
                return rval, [*rval_instrs, *lval_instrs, *self._emit_assign(lval_obj, rval)]
            case ast.Conditional(condition=cond, exp_true=exp_true, exp_false=exp_false):
                cond_val, cond_instrs = self.translate_exp(cond)
                label_false = self._make_label("cond_false")
                label_end = self._make_label("cond_end")
                if isinstance(exp.exp_type, ast.VoidType):
                    _, e1_instrs = self.translate_exp(exp_true)
                    _, e2_instrs = self.translate_exp(exp_false)
                    return tac.Var(value="DUMMY"), [
                        *cond_instrs,
                        tac.JumpIfZero(condition=cond_val, target=label_false.name),
                        *e1_instrs,
                        tac.Jump(target=label_end.name),
                        label_false,
                        *e2_instrs,
                        label_end,
                    ]
                dst = self._make_temp(exp.exp_type or ast.IntType())
                e1_val, e1_instrs = self.translate_exp(exp_true)
                e2_val, e2_instrs = self.translate_exp(exp_false)
                return dst, [
                    *cond_instrs,
                    tac.JumpIfZero(condition=cond_val, target=label_false.name),
                    *e1_instrs,
                    tac.Copy(src=e1_val, dst=dst),
                    tac.Jump(target=label_end.name),
                    label_false,
                    *e2_instrs,
                    tac.Copy(src=e2_val, dst=dst),
                    label_end,
                ]
            case ast.FunctionCall(callee=callee, args=args):
                # (*fp)(args) — dereference on a function pointer cancels out, same as &*
                inner_callee = callee.exp if isinstance(callee, ast.Dereference) else callee
                callee_val, callee_instrs = self.translate_exp(inner_callee)
                arg_vals: list[tac.Val] = []
                arg_instrs: list[tac.Instruction] = []
                for arg in args:
                    val, instrs = self.translate_exp(arg)
                    arg_instrs.extend(instrs)
                    arg_vals.append(val)
                dst = self._make_temp(exp.exp_type or ast.IntType())
                return dst, [
                    *callee_instrs,
                    *arg_instrs,
                    tac.FunCall(fun_val=callee_val, args=tuple(arg_vals), dst=dst),
                ]
            case ast.AddrOf(exp=ast.Subscript(exp1=exp1, exp2=exp2)):
                # &arr[i] ≡ &*(arr+i) ≡ arr+i — AddPtr only, no Load.
                if isinstance(exp1.exp_type, ast.Pointer):
                    ptr_node, int_node = exp1, exp2
                else:
                    ptr_node, int_node = exp2, exp1
                ptr_val, ptr_instrs = self.translate_exp(ptr_node)
                int_val, int_instrs = self.translate_exp(int_node)
                assert isinstance(ptr_node.exp_type, ast.Pointer)
                scale = self._size_of(ptr_node.exp_type.referenced)
                dst = self._make_temp(exp.exp_type or ptr_node.exp_type)
                return dst, [*ptr_instrs, *int_instrs, tac.AddPtr(ptr=ptr_val, index=int_val, scale=scale, dst=dst)]
            case ast.AddrOf(exp=ast.String(string=s)):
                str_name = self._make_string_name()
                self._symbol_table[str_name] = SymbolTableEntry(
                    t=ast.Array(element=ast.Char(), size=len(s) + 1),
                    attrs=ConstantAttr(SymStringInit(s, True)),
                )
                dst = self._make_temp(exp.exp_type or ast.Pointer(ast.Char()))
                return dst, [tac.GetAddress(src=tac.Var(value=str_name), dst=dst)]
            case ast.AddrOf(exp=ast.Dereference(exp=inner)):
                return self.translate_exp(inner)
            case ast.AddrOf(exp=ast.Var(name=name)):
                dst = self._make_temp(exp.exp_type or ast.Pointer(ast.IntType()))
                return dst, [tac.GetAddress(src=tac.Var(value=name), dst=dst)]
            case ast.AddrOf(exp=ast.Dot() | ast.Arrow() as inner):
                obj, obj_instrs = self._eval_exp(inner)
                addr_val, addr_instrs = self._addr_of_obj(obj, exp.exp_type or ast.Pointer(ast.IntType()))
                return addr_val, [*obj_instrs, *addr_instrs]
            case ast.Dereference(exp=inner):
                ptr_val, ptr_instrs = self.translate_exp(inner)
                dst = self._make_temp(exp.exp_type or ast.IntType())
                return dst, [*ptr_instrs, tac.Load(src_ptr=ptr_val, dst=dst)]
            case ast.Subscript(exp1=exp1, exp2=exp2):
                if isinstance(exp1.exp_type, ast.Pointer):
                    ptr_node, int_node = exp1, exp2
                else:
                    ptr_node, int_node = exp2, exp1
                ptr_val, ptr_instrs = self.translate_exp(ptr_node)
                int_val, int_instrs = self.translate_exp(int_node)
                assert isinstance(ptr_node.exp_type, ast.Pointer)
                scale = self._size_of(ptr_node.exp_type.referenced)
                elem_ptr = self._make_temp(ptr_node.exp_type)
                result = self._make_temp(exp.exp_type or ast.IntType())
                return result, [
                    *ptr_instrs,
                    *int_instrs,
                    tac.AddPtr(ptr=ptr_val, index=int_val, scale=scale, dst=elem_ptr),
                    tac.Load(src_ptr=elem_ptr, dst=result),
                ]
            case ast.Dot() | ast.Arrow():
                obj, obj_instrs = self._eval_exp(exp)
                val, force_instrs = self._force_val(obj, exp.exp_type or ast.IntType())
                return val, [*obj_instrs, *force_instrs]
            case _:
                raise CodeGenerationError(f"unsupported expression type: {type(exp).__name__}")
