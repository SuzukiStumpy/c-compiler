"""Tests for type-checking rules enforced by ``SemanticAnalyser``.

Verifies that operations illegal on ``double`` or pointer operands are rejected
at the semantic analysis stage with a ``SemanticError``, and that
``Dereference`` and ``AddrOf`` expressions are correctly typed and validated.
"""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import _make_prog, _program, _vd


def _double_const() -> ast.Constant:
    """Return a typed ``double`` constant (5.0)."""
    return ast.Constant(value=ast.ConstDouble(5.0))


def _int_const() -> ast.Constant:
    """Return a typed ``int`` constant (3)."""
    return ast.Constant(value=ast.ConstInt(3))


def _return_exp(exp: ast.Exp) -> tuple[ast.BlockItem, ...]:
    """Wrap *exp* in a ``Return`` statement body suitable for ``_program``."""
    return (ast.S(ast.Return(exp)),)


def _expr_stmt(exp: ast.Exp) -> tuple[ast.BlockItem, ...]:
    """Wrap *exp* in an ``Expression`` statement body suitable for ``_program``.

    Args:
        exp: The expression to wrap.

    Returns:
        A single-element block-item tuple containing the expression statement.
    """
    return (ast.S(ast.Expression(exp)),)


class TestDoubleOperatorRestrictions:
    """Operators that are undefined for ``double`` must raise ``SemanticError``."""

    def test_bitwise_complement_of_double_raises(self) -> None:
        """Applying ``~`` to a ``double`` operand raises ``SemanticError``."""
        prog = _program(_return_exp(ast.Unary(ast.Complement(), _double_const())))
        with pytest.raises(SemanticError, match="bitwise complement"):
            resolve(prog)

    def test_remainder_with_double_left_raises(self) -> None:
        """Applying ``%`` with a ``double`` left operand raises ``SemanticError``."""
        prog = _program(_return_exp(ast.Binary(ast.Remainder(), _double_const(), _int_const())))
        with pytest.raises(SemanticError, match="modulus"):
            resolve(prog)

    def test_remainder_with_double_right_raises(self) -> None:
        """Applying ``%`` with a ``double`` right operand raises ``SemanticError``."""
        prog = _program(_return_exp(ast.Binary(ast.Remainder(), _int_const(), _double_const())))
        with pytest.raises(SemanticError, match="modulus"):
            resolve(prog)

    def test_left_shift_with_double_left_raises(self) -> None:
        """Applying ``<<`` with a ``double`` left operand raises ``SemanticError``."""
        prog = _program(_return_exp(ast.Binary(ast.LeftShift(), _double_const(), _int_const())))
        with pytest.raises(SemanticError, match="shift"):
            resolve(prog)

    def test_left_shift_with_double_right_raises(self) -> None:
        """Applying ``<<`` with a ``double`` right operand raises ``SemanticError``."""
        prog = _program(_return_exp(ast.Binary(ast.LeftShift(), _int_const(), _double_const())))
        with pytest.raises(SemanticError, match="shift"):
            resolve(prog)

    def test_right_shift_with_double_left_raises(self) -> None:
        """Applying ``>>`` with a ``double`` left operand raises ``SemanticError``."""
        prog = _program(_return_exp(ast.Binary(ast.RightShift(), _double_const(), _int_const())))
        with pytest.raises(SemanticError, match="shift"):
            resolve(prog)

    def test_right_shift_with_double_right_raises(self) -> None:
        """Applying ``>>`` with a ``double`` right operand raises ``SemanticError``."""
        prog = _program(_return_exp(ast.Binary(ast.RightShift(), _int_const(), _double_const())))
        with pytest.raises(SemanticError, match="shift"):
            resolve(prog)


class TestSwitchTypeRestrictions:
    """The controlling expression of a ``switch`` must be an integer type."""

    def test_switch_with_double_control_raises(self) -> None:
        """A ``switch`` whose control expression is ``double`` raises ``SemanticError``."""
        switch = ast.Switch(control=_double_const(), body=ast.Null())
        prog = _program((ast.S(statement=switch),))
        with pytest.raises(SemanticError, match="integer"):
            resolve(prog)


def _ptr_var(name: str) -> tuple[ast.BlockItem, ...]:
    """Declare a local ``int *`` variable named *name* with no initialiser.

    Args:
        name: The source-level variable name.

    Returns:
        A single-element block-item tuple containing the declaration.
    """
    return (_vd(name, var_type=ast.Pointer(ast.IntType())),)


class TestPointerOperatorRestrictions:
    """Operators that are undefined for pointer types must raise ``SemanticError``."""

    def test_multiply_pointer_raises(self) -> None:
        """Multiplying a pointer operand raises ``SemanticError``."""
        body = _ptr_var("p") + _return_exp(ast.Binary(ast.Multiply(), ast.Var("p"), _int_const()))
        with pytest.raises(SemanticError, match="multiplication"):
            resolve(_program(body))

    def test_divide_pointer_raises(self) -> None:
        """Dividing a pointer operand raises ``SemanticError``."""
        body = _ptr_var("p") + _return_exp(ast.Binary(ast.Divide(), _int_const(), ast.Var("p")))
        with pytest.raises(SemanticError, match="division"):
            resolve(_program(body))

    def test_remainder_pointer_raises(self) -> None:
        """Applying ``%`` to a pointer operand raises ``SemanticError``."""
        body = _ptr_var("p") + _return_exp(ast.Binary(ast.Remainder(), ast.Var("p"), _int_const()))
        with pytest.raises(SemanticError, match="remainder"):
            resolve(_program(body))

    def test_complement_pointer_raises(self) -> None:
        """Applying ``~`` to a pointer operand raises ``SemanticError``."""
        body = _ptr_var("p") + _return_exp(ast.Unary(ast.Complement(), ast.Var("p")))
        with pytest.raises(SemanticError, match="pointer type"):
            resolve(_program(body))

    def test_negate_pointer_raises(self) -> None:
        """Applying unary ``-`` to a pointer operand raises ``SemanticError``."""
        body = _ptr_var("p") + _return_exp(ast.Unary(ast.Negate(), ast.Var("p")))
        with pytest.raises(SemanticError, match="pointer type"):
            resolve(_program(body))

    def test_shift_pointer_raises(self) -> None:
        """Applying ``<<`` with a pointer operand raises ``SemanticError``."""
        body = _ptr_var("p") + _return_exp(ast.Binary(ast.LeftShift(), ast.Var("p"), _int_const()))
        with pytest.raises(SemanticError, match="pointer type"):
            resolve(_program(body))

    def test_logical_not_pointer_succeeds(self) -> None:
        """Applying ``!`` to a pointer operand is valid: pointers are scalar types (C11 §6.5.3.3 ¶5)."""
        body = _ptr_var("p") + _return_exp(ast.Unary(ast.Not(), ast.Var("p")))
        resolve(_program(body))


class TestPointerIncompatibleTypes:
    """Comparing or assigning incompatible pointer and arithmetic types must raise ``SemanticError``."""

    def test_compare_pointer_to_double_raises(self) -> None:
        """Comparing a pointer to a ``double`` operand raises ``SemanticError``."""
        body = _ptr_var("p") + _return_exp(ast.Binary(ast.Equal(), ast.Var("p"), _double_const()))
        with pytest.raises(SemanticError, match="incompatible"):
            resolve(_program(body))

    def test_compare_pointer_to_int_raises(self) -> None:
        """Comparing a pointer to an ``int`` operand raises ``SemanticError``."""
        body = _ptr_var("p") + _return_exp(ast.Binary(ast.NotEqual(), ast.Var("p"), _int_const()))
        with pytest.raises(SemanticError, match="incompatible"):
            resolve(_program(body))

    def test_assign_double_to_pointer_raises(self) -> None:
        """Assigning a ``double`` value to a pointer lvalue raises ``SemanticError``."""
        body = (
            _ptr_var("p")
            + (ast.S(ast.Expression(ast.Assignment(ast.Var("p"), _double_const()))),)
            + _return_exp(_int_const())
        )
        with pytest.raises(SemanticError, match="pointer"):
            resolve(_program(body))


class TestDereferenceTypeChecking:
    """``Dereference`` expressions are correctly typed and validated."""

    def test_dereference_int_pointer_has_int_type(self) -> None:
        """Dereferencing an ``int *`` variable yields an expression of type ``IntType``."""
        body = _ptr_var("p") + _expr_stmt(ast.Dereference(ast.Var("p"))) + _return_exp(_int_const())
        resolved = resolve(_program(body))
        stmt = resolved.declarations[0].function_declaration.body[1].statement
        assert isinstance(stmt, ast.Expression)
        assert stmt.exp.exp_type == ast.IntType()

    def test_dereference_non_pointer_raises(self) -> None:
        """Dereferencing a non-pointer (e.g., ``int``) operand raises ``SemanticError``."""
        body = (_vd("x", ast.Constant(ast.ConstInt(1))),) + _return_exp(ast.Dereference(ast.Var("x")))
        with pytest.raises(SemanticError, match="non-pointer"):
            resolve(_program(body))


class TestAddrOfTypeChecking:
    """``AddrOf`` expressions are correctly typed and validated."""

    def test_addr_of_variable_has_pointer_type(self) -> None:
        """Taking the address of an ``int`` variable yields an ``int *`` expression."""
        body = (
            (_vd("x", ast.Constant(ast.ConstInt(1))),)
            + _expr_stmt(ast.AddrOf(ast.Var("x")))
            + _return_exp(_int_const())
        )
        resolved = resolve(_program(body))
        stmt = resolved.declarations[0].function_declaration.body[1].statement
        assert isinstance(stmt, ast.Expression)
        assert stmt.exp.exp_type == ast.Pointer(ast.IntType())

    def test_addr_of_dereference_is_valid_lvalue(self) -> None:
        """Taking the address of a dereferenced pointer (``&*p``) is a valid lvalue operation."""
        body = _ptr_var("p") + _expr_stmt(ast.AddrOf(ast.Dereference(ast.Var("p")))) + _return_exp(_int_const())
        resolve(_program(body))

    def test_addr_of_constant_raises(self) -> None:
        """Taking the address of a constant (not an lvalue) raises ``SemanticError``."""
        with pytest.raises(SemanticError, match="lvalue"):
            resolve(_program(_return_exp(ast.AddrOf(_int_const()))))


def _fun_ptr_type() -> ast.Pointer:
    """Return a simple function pointer type ``int (*)(void)``."""
    return ast.Pointer(ast.FunType(params=(), ret=ast.IntType()))


def _null_const() -> ast.Constant:
    """Return a null pointer constant (integer zero)."""
    return ast.Constant(value=ast.ConstInt(0))


class TestFunctionPointerTypeChecking:
    """Type-checking rules specific to function pointer types.

    Verifies that the pointer-compatibility helpers (``_convert_by_assignment``,
    ``get_common_pointer_type``) correctly treat ``Pointer(FunType(...))`` as a
    pointer type, so that null pointer constants are accepted wherever a function
    pointer is expected and incompatible types are rejected.
    """

    def test_compare_fun_ptr_to_null_equal_succeeds(self) -> None:
        """Comparing a function pointer to a null pointer constant with ``==`` is valid."""
        body = (_vd("fp", var_type=_fun_ptr_type()),) + _return_exp(
            ast.Binary(ast.Equal(), ast.Var("fp"), _null_const())
        )
        resolve(_program(body))

    def test_compare_fun_ptr_to_null_not_equal_succeeds(self) -> None:
        """Comparing a function pointer to a null pointer constant with ``!=`` is valid."""
        body = (_vd("fp", var_type=_fun_ptr_type()),) + _return_exp(
            ast.Binary(ast.NotEqual(), ast.Var("fp"), _null_const())
        )
        resolve(_program(body))

    def test_assign_null_to_fun_ptr_succeeds(self) -> None:
        """Assigning a null pointer constant to a function pointer variable is valid."""
        body = (
            (_vd("fp", var_type=_fun_ptr_type()),)
            + (ast.S(ast.Expression(ast.Assignment(ast.Var("fp"), _null_const()))),)
            + _return_exp(_int_const())
        )
        resolve(_program(body))

    def test_conditional_fun_ptr_and_null_succeeds(self) -> None:
        """A conditional expression whose branches are a function pointer and null is valid."""
        body = (
            (_vd("fp", var_type=_fun_ptr_type()),)
            + _expr_stmt(ast.Conditional(_int_const(), ast.Var("fp"), _null_const()))
            + _return_exp(_int_const())
        )
        resolve(_program(body))

    def test_return_null_from_fun_ptr_function_succeeds(self) -> None:
        """Returning a null pointer constant from a function whose return type is a function pointer is valid."""
        fn = ast.Function(
            name="get_fp",
            params=(),
            body=(ast.S(ast.Return(_null_const())),),
            fun_type=ast.FunType(params=(), ret=_fun_ptr_type()),
        )
        resolve(_make_prog(fn))

    def test_compare_fun_ptr_to_double_raises(self) -> None:
        """Comparing a function pointer to a ``double`` raises ``SemanticError``."""
        body = (_vd("fp", var_type=_fun_ptr_type()),) + _return_exp(
            ast.Binary(ast.Equal(), ast.Var("fp"), _double_const())
        )
        with pytest.raises(SemanticError, match="incompatible"):
            resolve(_program(body))

    def test_compare_incompatible_function_pointers_raises(self) -> None:
        """Comparing two function pointers with different signatures raises ``SemanticError``."""
        fp1_type = ast.Pointer(ast.FunType(params=(), ret=ast.IntType()))
        fp2_type = ast.Pointer(ast.FunType(params=(ast.IntType(),), ret=ast.IntType()))
        body = (
            (_vd("fp1", var_type=fp1_type),)
            + (_vd("fp2", var_type=fp2_type),)
            + _return_exp(ast.Binary(ast.Equal(), ast.Var("fp1"), ast.Var("fp2")))
        )
        with pytest.raises(SemanticError, match="incompatible"):
            resolve(_program(body))

    def test_assign_double_to_fun_ptr_raises(self) -> None:
        """Assigning a ``double`` to a function pointer variable raises ``SemanticError``."""
        body = (
            (_vd("fp", var_type=_fun_ptr_type()),)
            + (ast.S(ast.Expression(ast.Assignment(ast.Var("fp"), _double_const()))),)
            + _return_exp(_int_const())
        )
        with pytest.raises(SemanticError, match="pointer"):
            resolve(_program(body))

    def test_return_double_from_fun_ptr_function_raises(self) -> None:
        """Returning a ``double`` from a function whose return type is a function pointer raises ``SemanticError``."""
        fn = ast.Function(
            name="get_fp",
            params=(),
            body=(ast.S(ast.Return(_double_const())),),
            fun_type=ast.FunType(params=(), ret=_fun_ptr_type()),
        )
        with pytest.raises(SemanticError, match="pointer"):
            resolve(_make_prog(fn))


class TestCastTypeRestrictions:
    """Explicit cast expressions are validated for pointer/double incompatibility."""

    def test_cast_double_to_pointer_raises(self) -> None:
        """Casting a ``double`` to a pointer type raises ``SemanticError``."""
        with pytest.raises(SemanticError, match="pointer"):
            resolve(
                _program(_expr_stmt(ast.Cast(ast.Pointer(ast.IntType()), _double_const())) + _return_exp(_int_const()))
            )

    def test_cast_pointer_to_double_raises(self) -> None:
        """Casting a pointer to ``double`` raises ``SemanticError``."""
        body = _ptr_var("p") + _expr_stmt(ast.Cast(ast.DoubleType(), ast.Var("p"))) + _return_exp(_int_const())
        with pytest.raises(SemanticError, match="double"):
            resolve(_program(body))

    def test_cast_int_to_pointer_succeeds(self) -> None:
        """Casting an ``int`` to a pointer type is a valid explicit cast."""
        resolve(_program(_expr_stmt(ast.Cast(ast.Pointer(ast.IntType()), _int_const())) + _return_exp(_int_const())))

    def test_cast_pointer_to_int_succeeds(self) -> None:
        """Casting a pointer to ``int`` is a valid explicit cast."""
        body = _ptr_var("p") + _return_exp(ast.Cast(ast.IntType(), ast.Var("p")))
        resolve(_program(body))


def _char_ptr_var(name: str) -> tuple[ast.BlockItem, ...]:
    """Declare a local ``char *`` variable named *name* with no initialiser.

    Args:
        name: The source-level variable name.

    Returns:
        A single-element block-item tuple containing the declaration.
    """
    return (_vd(name, var_type=ast.Pointer(ast.Char())),)


def _char_var(name: str) -> tuple[ast.BlockItem, ...]:
    """Declare a local ``char`` variable named *name* with no initialiser.

    Args:
        name: The source-level variable name.

    Returns:
        A single-element block-item tuple containing the declaration.
    """
    return (_vd(name, var_type=ast.Char()),)


class TestCharTypeChecking:
    """Type-checking rules involving ``char``, ``signed char``, and ``unsigned char`` operands."""

    def test_subscript_with_char_index_succeeds(self) -> None:
        """``ptr[c]`` where ``c`` has type ``char`` is a valid subscript expression."""
        body = _char_ptr_var("ptr") + _char_var("c") + _return_exp(ast.Subscript(ast.Var("ptr"), ast.Var("c")))
        resolve(_program(body))

    def test_subscript_with_schar_index_succeeds(self) -> None:
        """``ptr[c]`` where ``c`` has type ``signed char`` is a valid subscript expression."""
        body = (
            _char_ptr_var("ptr")
            + (_vd("c", var_type=ast.SChar()),)
            + _return_exp(ast.Subscript(ast.Var("ptr"), ast.Var("c")))
        )
        resolve(_program(body))

    def test_subscript_with_uchar_index_succeeds(self) -> None:
        """``ptr[c]`` where ``c`` has type ``unsigned char`` is a valid subscript expression."""
        body = (
            _char_ptr_var("ptr")
            + (_vd("c", var_type=ast.UChar()),)
            + _return_exp(ast.Subscript(ast.Var("ptr"), ast.Var("c")))
        )
        resolve(_program(body))

    def test_return_char_from_int_function_succeeds(self) -> None:
        """Returning a ``char`` variable from an ``int``-returning function is valid (implicit conversion)."""
        body = _char_var("c") + _return_exp(ast.Var("c"))
        resolve(_program(body))

    def test_assign_char_to_int_succeeds(self) -> None:
        """Assigning a ``char`` value to an ``int`` variable is a valid implicit conversion."""
        body = _char_var("c") + (_vd("x"),) + (ast.S(ast.Expression(ast.Assignment(ast.Var("x"), ast.Var("c")))),)
        resolve(_program(body + _return_exp(_int_const())))
