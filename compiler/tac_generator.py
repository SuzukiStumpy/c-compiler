"""TAC generation stage: translates the main AST into a TAC intermediate representation.

Performs a structural translation from ``compiler.syntax.ast`` nodes to
``compiler.syntax.tac`` nodes, flattening nested expression trees into sequences
of three-address instructions.  Temporary variables (``Var`` nodes) are created
on demand and named ``tmp.0``, ``tmp.1``, etc.

The translation follows this mapping:

    ast.Program(decls)                    → tac.Program(top_level=[...])
    ast.FunDecl(fn) [with body]           → tac.Function(name, globl, params, instructions)
    ast.VarDecl(v)  [file-scope/static]   → tac.StaticVariable(name, globl, init)
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

from itertools import count

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from compiler.types.symbols import (
    Initial,
    LocalAttr,
    NoInitializer,
    PointerInit,
    StaticAttr,
    SymbolTable,
    SymbolTableEntry,
    Tentative,
    ULongInit,
)
from compiler.utils.exceptions import CodeGenerationError


def generate(program: ast.Program, symbol_table: SymbolTable) -> tac.Program:
    """Translate an AST ``Program`` into a TAC ``Program``.

    File-scope variable declarations and block-scope ``static``/``extern``
    declarations are not translated directly from the AST.  Instead, after all
    function bodies have been processed, ``_convert_symbols_to_tacky`` walks the
    symbol table and emits a ``tac.StaticVariable`` for each entry that carries a
    ``StaticAttr`` with an ``Initial`` or ``Tentative`` initial value.

    Args:
        program: The ``Program`` node produced by the semantic analysis stage.
        symbol_table: The symbol table populated by the semantic analysis stage.

    Returns:
        A ``tac.Program`` node representing the equivalent TAC program.
        Forward declarations (functions without a body) are skipped.

    Raises:
        CodeGenerationError: If any construct cannot be translated.
    """
    gen = _TacGenerator(symbol_table)
    functions = [
        gen.translate_function(decl.function_declaration)
        for decl in program.declarations
        if isinstance(decl, ast.FunDecl) and decl.function_declaration.body is not None
    ]
    static_vars = _convert_symbols_to_tacky(symbol_table)
    return tac.Program(top_level=tuple(static_vars) + tuple(functions))


def _convert_symbols_to_tacky(symbol_table: SymbolTable) -> list[tac.StaticVariable]:
    """Emit a ``tac.StaticVariable`` for every static-storage variable in the symbol table.

    Walks all entries in ``symbol_table``.  Entries that carry a ``StaticAttr``
    with an ``Initial`` or ``Tentative`` initial value produce a
    ``tac.StaticVariable``; entries with ``NoInitializer`` (non-defining
    ``extern`` declarations) or ``FunAttr`` entries are silently skipped.

    Args:
        symbol_table: The symbol table returned by the semantic analysis stage.

    Returns:
        A list of ``tac.StaticVariable`` nodes, in symbol-table insertion order.
    """
    result: list[tac.StaticVariable] = []
    for name, entry in symbol_table.items():
        is_long = isinstance(entry.t, ast.LongType | ast.ULongType)
        is_double = isinstance(entry.t, ast.DoubleType)
        match entry.attrs:
            case StaticAttr(init=Initial(value=i), globl=globl):
                if is_double:
                    static_init: tac.StaticInit = tac.DoubleInit(float(i))
                elif is_long:
                    static_init = tac.LongInit(int(i))
                else:
                    static_init = tac.IntInit(int(i))
                result.append(tac.StaticVariable(name=name, globl=globl, t=entry.t, init=static_init))
            case StaticAttr(init=Tentative(), globl=globl):
                if is_double:
                    static_init = tac.DoubleInit(0.0)
                elif is_long:
                    static_init = tac.LongInit(0)
                else:
                    static_init = tac.IntInit(0)
                result.append(tac.StaticVariable(name=name, globl=globl, t=entry.t, init=static_init))
            case StaticAttr(init=ULongInit(), globl=globl):
                # Null pointer constant: pointer-sized (8-byte) zero.
                result.append(tac.StaticVariable(name=name, globl=globl, t=entry.t, init=tac.LongInit(0)))
            case StaticAttr(init=PointerInit(name=target), globl=globl):
                raise CodeGenerationError(
                    f"Static pointer initialiser '&{target}' cannot yet be lowered to TAC: "
                    "data-section pointer relocations are not yet implemented."
                )
            case StaticAttr(init=NoInitializer()):
                pass
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


def _type_size(t: ast.Type) -> int:
    """Return the size in bytes of a C type.

    Args:
        t: The C type whose size is required.

    Returns:
        4 for ``int``/``unsigned int``; 8 for ``long``/``unsigned long``.

    Raises:
        CodeGenerationError: If the type is not a scalar integer type.
    """
    if isinstance(t, (ast.IntType, ast.UIntType)):
        return 4
    if isinstance(t, (ast.LongType, ast.ULongType)):
        return 8
    raise CodeGenerationError(f"unsupported type in size calculation: {type(t).__name__}")


def _is_signed(t: ast.Type) -> bool:
    """Return whether a C integer type is signed.

    Args:
        t: The C type to inspect.

    Returns:
        ``True`` for ``int`` and ``long``; ``False`` for ``unsigned int`` and ``unsigned long``.

    Raises:
        CodeGenerationError: If the type is not a scalar integer type.
    """
    if isinstance(t, (ast.IntType, ast.LongType)):
        return True
    if isinstance(t, (ast.UIntType, ast.ULongType)):
        return False
    raise CodeGenerationError(f"unsupported type in signed check: {type(t).__name__}")


class _TacGenerator:
    """Stateful TAC generator for a single function.

    Maintains a temporary-variable counter and a label counter, both used to
    assign unique names throughout the translation.  Instantiate once per
    function to ensure fresh temporaries and labels for each translation unit.
    """

    def __init__(self, symbol_table: SymbolTable) -> None:
        """Initialise the generator with zeroed temporary-variable and label counters.

        Args:
            symbol_table: The program-wide symbol table, mutated in-place to record
                compiler-generated temporaries (e.g. cast destinations).
        """
        self._counter = count()
        self._label_counter = count()
        self._function_name: str = ""
        self._symbol_table = symbol_table

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
        # Guard instruction for functions that don't supply a return value
        instructions.append(tac.Return(tac.Constant(ast.ConstInt(0))))
        return tac.Function(
            name=function.name,
            globl=function.globl,
            params=tuple(p.name for p in function.params),
            body=tuple(instructions),
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
                val, instrs = self.translate_exp(init)
                return [*instrs, tac.Copy(src=val, dst=tac.Var(value=name))]
            case ast.D(declaration=ast.FunDecl()):
                return []
            case _:
                raise CodeGenerationError(f"unsupported block item type: {type(item).__name__}")

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
                val, instrs = self.translate_exp(decl_init)
                return [*instrs, tac.Copy(src=val, dst=tac.Var(value=name))]
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
            case ast.Cast(target_type=t, exp=inner):
                result, inner_instrs = self.translate_exp(inner)
                inner_type = inner.exp_type
                if t == inner_type:
                    return result, inner_instrs
                dst = self._make_temp(t)
                if isinstance(t, ast.DoubleType):
                    if isinstance(inner_type, (ast.IntType, ast.LongType)):
                        conv: tac.Instruction = tac.IntToDouble(src=result, dst=dst)
                    else:
                        conv = tac.UIntToDouble(src=result, dst=dst)
                elif isinstance(inner_type, ast.DoubleType):
                    if isinstance(t, (ast.IntType, ast.LongType)):
                        conv = tac.DoubleToInt(src=result, dst=dst)
                    else:
                        conv = tac.DoubleToUInt(src=result, dst=dst)
                else:
                    dst_size = _type_size(t)
                    src_size = _type_size(inner_type)  # type: ignore[arg-type]
                    if dst_size == src_size:
                        conv = tac.Copy(src=result, dst=dst)
                    elif dst_size < src_size:
                        conv = tac.Truncate(src=result, dst=dst)
                    elif _is_signed(inner_type):  # type: ignore[arg-type]
                        conv = tac.SignExtend(src=result, dst=dst)
                    else:
                        conv = tac.ZeroExtend(src=result, dst=dst)
                return dst, [*inner_instrs, conv]
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
            case ast.Binary(operator=op, left=left, right=right):
                src1, left_instrs = self.translate_exp(left)
                src2, right_instrs = self.translate_exp(right)
                dst = self._make_temp(exp.exp_type or ast.IntType())
                binary = tac.Binary(operator=_translate_binary_op(op), src1=src1, src2=src2, dst=dst)
                return dst, [*left_instrs, *right_instrs, binary]
            case ast.Var(name=name):
                return tac.Var(value=name), []
            case ast.Assignment(lvalue=ast.Var(name=name), rvalue=rvalue):
                val, instrs = self.translate_exp(rvalue)
                dst = tac.Var(value=name)
                return dst, [*instrs, tac.Copy(src=val, dst=dst)]
            case ast.Conditional(condition=cond, exp_true=exp_true, exp_false=exp_false):
                cond_val, cond_instrs = self.translate_exp(cond)
                label_false = self._make_label("cond_false")
                label_end = self._make_label("cond_end")
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
                arg_vals: list[tac.Val] = []
                all_instrs: list[tac.Instruction] = []
                for arg in args:
                    val, instrs = self.translate_exp(arg)
                    all_instrs.extend(instrs)
                    arg_vals.append(val)
                dst = self._make_temp(exp.exp_type or ast.IntType())
                if not isinstance(callee, ast.Var):
                    raise CodeGenerationError("Indirect function calls through non-Var callees are not yet supported.")
                return dst, [*all_instrs, tac.FunCall(name=callee.name, args=tuple(arg_vals), dst=dst)]
            case _:
                raise CodeGenerationError(f"unsupported expression type: {type(exp).__name__}")
