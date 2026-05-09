"""TAC generation stage: translates the main AST into a TAC intermediate representation.

Performs a structural translation from ``compiler.syntax.ast`` nodes to
``compiler.syntax.tac`` nodes, flattening nested expression trees into sequences
of three-address instructions.  Temporary variables (``Var`` nodes) are created
on demand and named ``tmp.0``, ``tmp.1``, etc.

The translation follows this mapping:

    ast.Program([fn])          → tac.Program(fn)
    ast.Function(name, body)   → tac.Function(name, instructions)
    ast.Return(exp)            → <emit(exp)> + [tac.Return(result_val)]
    ast.Constant(v)            → tac.Constant(v)   (no extra instructions)
    ast.Unary(op, inner)       → <emit(inner)> + [tac.Unary(op, inner_val, tmp)]
                                 returns tmp as the result val
    ast.Unary(PreIncrement, x) → [Binary(Add, x, 1, tmp), Copy(tmp, x)]
                                 returns x (new value)
    ast.Unary(PreDecrement, x) → [Binary(Sub, x, 1, tmp), Copy(tmp, x)]
                                 returns x (new value)
    ast.Unary(PostIncrement,x) → [Copy(x, old), Binary(Add, x, 1, tmp), Copy(tmp, x)]
                                 returns old (original value)
    ast.Unary(PostDecrement,x) → [Copy(x, old), Binary(Sub, x, 1, tmp), Copy(tmp, x)]
                                 returns old (original value)
    ast.Binary(op, l, r)       → <emit(l)> + <emit(r)>
      (arithmetic / bitwise /    + [tac.Binary(op, l_val, r_val, tmp)]
       comparison operators)     returns tmp as the result val
    ast.Binary(And, l, r)      → <emit(l)>
                                 + [JumpIfZero(l_val, false_lbl)]
                                 + <emit(r)>
                                 + [JumpIfZero(r_val, false_lbl),
                                    Copy(1, tmp), Jump(end_lbl),
                                    false_lbl, Copy(0, tmp), end_lbl]
                                 returns tmp as the result val
    ast.Binary(Or, l, r)       → <emit(l)>
                                 + [JumpIfNotZero(l_val, true_lbl)]
                                 + <emit(r)>
                                 + [JumpIfNotZero(r_val, true_lbl),
                                    Copy(0, tmp), Jump(end_lbl),
                                    true_lbl, Copy(1, tmp), end_lbl]
                                 returns tmp as the result val
    ast.If(cond, then, None)   → <emit(cond)>
                                 + [JumpIfZero(cond_val, end_lbl)]
                                 + <emit(then)>
                                 + [end_lbl]
    ast.Conditional(c, e1, e2) → <emit(c)>
                                 + [JumpIfZero(c_val, false_lbl)]
                                 + <emit(e1)>
                                 + [Copy(e1_val, tmp), Jump(end_lbl), false_lbl]
                                 + <emit(e2)>
                                 + [Copy(e2_val, tmp), end_lbl]
                                 returns tmp as the result val
    ast.If(cond, then, else_)  → <emit(cond)>
                                 + [JumpIfZero(cond_val, else_lbl)]
                                 + <emit(then)>
                                 + [Jump(end_lbl), else_lbl]
                                 + <emit(else_)>
                                 + [end_lbl]

The resulting ``tac.Program`` is a complete TAC AST ready for the assembly code
generation stage.
"""

from itertools import count

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from compiler.utils.exceptions import CodeGenerationError


def generate(program: ast.Program) -> tac.Program:
    """Translate an AST ``Program`` into a TAC ``Program``.

    Args:
        program: The ``Program`` node produced by the parser.

    Returns:
        A ``tac.Program`` node representing the equivalent TAC program.

    Raises:
        CodeGenerationError: If ``program`` contains more than one function,
            or if any construct cannot be translated.
    """
    if len(program.functions) != 1:
        raise CodeGenerationError(f"expected exactly one function definition, got {len(program.functions)}")
    return tac.Program(function=_TacGenerator().translate_function(program.functions[0]))


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


class _TacGenerator:
    """Stateful TAC generator for a single function.

    Maintains a counter used to assign unique names to temporary variables.
    Instantiate once per function to ensure fresh temporaries for each
    translation unit.
    """

    def __init__(self) -> None:
        """Initialise the generator with zeroed temporary-variable and label counters."""
        self._counter = count()
        self._label_counter = count()

    def _make_temp(self) -> tac.Var:
        """Create a fresh temporary variable and advance the counter.

        Returns:
            A ``tac.Var`` named ``tmp.<n>`` where ``n`` is the next counter value.
        """
        return tac.Var(value=f"tmp.{next(self._counter)}")

    def _make_label(self, base: str) -> tac.Label:
        """Create a fresh label and avance the counter.

        Returns:
            A ``tac.Label`` named ``${base}.<n>`` where ``n`` is the next counter value.
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
        instructions: list[tac.Instruction] = []
        for item in function.body:
            instructions.extend(self.translate_block_item(item))

        # Guard instruction for functions that don't supply a return value
        instructions.append(tac.Return(tac.Constant(0)))
        return tac.Function(name=function.name, body=tuple(instructions))

    def translate_block_item(self, item: ast.BlockItem) -> list[tac.Instruction]:
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
            case ast.D(declaration=ast.Declaration(name=name, init=init)):
                if init is None:
                    return []
                val, instrs = self.translate_exp(init)
                return [*instrs, tac.Copy(src=val, dst=tac.Var(value=name))]
            case _:
                raise CodeGenerationError(f"unsupported block item type: {type(item).__name__}")

    def translate_statement(self, stmt: ast.Statement) -> list[tac.Instruction]:
        """Translate a single AST statement into a sequence of TAC instructions.

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
            case _:
                raise CodeGenerationError(f"unsupported statement type: {type(stmt).__name__}")

    def translate_exp(self, exp: ast.Exp) -> tuple[tac.Val, list[tac.Instruction]]:
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
            case ast.Constant(value=value):
                return tac.Constant(value=value), []
            case ast.Unary(operator=ast.PreIncrement(), exp=ast.Var(name=name)):
                var = tac.Var(value=name)
                tmp = self._make_temp()
                return var, [
                    tac.Binary(operator=tac.Add(), src1=var, src2=tac.Constant(value=1), dst=tmp),
                    tac.Copy(src=tmp, dst=var),
                ]
            case ast.Unary(operator=ast.PreDecrement(), exp=ast.Var(name=name)):
                var = tac.Var(value=name)
                tmp = self._make_temp()
                return var, [
                    tac.Binary(operator=tac.Subtract(), src1=var, src2=tac.Constant(value=1), dst=tmp),
                    tac.Copy(src=tmp, dst=var),
                ]
            case ast.Unary(operator=ast.PostIncrement(), exp=ast.Var(name=name)):
                var = tac.Var(value=name)
                old = self._make_temp()
                tmp = self._make_temp()
                return old, [
                    tac.Copy(src=var, dst=old),
                    tac.Binary(operator=tac.Add(), src1=var, src2=tac.Constant(value=1), dst=tmp),
                    tac.Copy(src=tmp, dst=var),
                ]
            case ast.Unary(operator=ast.PostDecrement(), exp=ast.Var(name=name)):
                var = tac.Var(value=name)
                old = self._make_temp()
                tmp = self._make_temp()
                return old, [
                    tac.Copy(src=var, dst=old),
                    tac.Binary(operator=tac.Subtract(), src1=var, src2=tac.Constant(value=1), dst=tmp),
                    tac.Copy(src=tmp, dst=var),
                ]
            case ast.Unary(operator=op, exp=inner):
                src, inner_instrs = self.translate_exp(inner)
                dst = self._make_temp()
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
                    tac.Copy(src=tac.Constant(value=1), dst=dst),
                    tac.Jump(target=label_end.name),
                    label_false,
                    tac.Copy(src=tac.Constant(value=0), dst=dst),
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
                    tac.Copy(src=tac.Constant(value=0), dst=dst),
                    tac.Jump(target=label_end.name),
                    label_true,
                    tac.Copy(src=tac.Constant(value=1), dst=dst),
                    label_end,
                ]
            case ast.Binary(operator=op, left=left, right=right):
                src1, left_instrs = self.translate_exp(left)
                src2, right_instrs = self.translate_exp(right)
                dst = self._make_temp()
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
                dst = self._make_temp()
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
            case _:
                raise CodeGenerationError(f"unsupported expression type: {type(exp).__name__}")
