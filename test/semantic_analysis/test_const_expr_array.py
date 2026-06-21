"""Semantic analysis tests for constant-expression array dimensions.

After ``resolve()``, every ``Array.size`` that was an ``ast.Exp`` must be
evaluated to a concrete ``int``.  Non-constant or non-positive expressions
must raise ``SemanticError``.
"""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import _prog_with_decls


def _array_var(name: str, elem: ast.Type, size_exp: ast.Exp) -> ast.VarDecl:
    """Return a top-level VarDecl whose type is ``Array(elem, size_exp)``.

    Args:
        name: Variable name.
        elem: Element type of the array.
        size_exp: Unevaluated size expression (as the parser would produce).

    Returns:
        An ``ast.VarDecl`` with the given array type.
    """
    return ast.VarDecl(ast.VariableDeclaration(name, None, ast.Array(elem, size_exp)))


def _add(a: ast.Exp, b: ast.Exp) -> ast.Binary:
    """Return ``a + b``."""
    return ast.Binary(ast.Add(), a, b)


def _mul(a: ast.Exp, b: ast.Exp) -> ast.Binary:
    """Return ``a * b``."""
    return ast.Binary(ast.Multiply(), a, b)


def _sub(a: ast.Exp, b: ast.Exp) -> ast.Binary:
    """Return ``a - b``."""
    return ast.Binary(ast.Subtract(), a, b)


def _int_const(n: int) -> ast.Constant:
    """Return an integer constant expression node.

    Args:
        n: The integer value.

    Returns:
        An ``ast.Constant`` wrapping ``ConstInt(n)``.
    """
    return ast.Constant(ast.ConstInt(n))


class TestArithmeticConstExprEvaluation:
    """Arithmetic constant expressions in array dimensions are folded to int."""

    def test_addition_evaluates_correctly(self) -> None:
        """``int a[2 + 3];`` resolves to Array(IntType, 5)."""
        prog = _prog_with_decls(_array_var("a", ast.IntType(), _add(_int_const(2), _int_const(3))))
        resolved = resolve(prog)
        decl = resolved.declarations[0]
        assert isinstance(decl, ast.VarDecl)
        assert decl.variable_declaration.var_type == ast.Array(ast.IntType(), 5)

    def test_multiplication_evaluates_correctly(self) -> None:
        """``int a[3 * 4];`` resolves to Array(IntType, 12)."""
        prog = _prog_with_decls(_array_var("a", ast.IntType(), _mul(_int_const(3), _int_const(4))))
        resolved = resolve(prog)
        decl = resolved.declarations[0]
        assert decl.variable_declaration.var_type == ast.Array(ast.IntType(), 12)

    def test_plain_constant_evaluates_correctly(self) -> None:
        """``int a[7];`` (stored as Constant(7)) resolves to Array(IntType, 7)."""
        prog = _prog_with_decls(_array_var("a", ast.IntType(), _int_const(7)))
        resolved = resolve(prog)
        decl = resolved.declarations[0]
        assert decl.variable_declaration.var_type == ast.Array(ast.IntType(), 7)


class TestSizeofConstExprEvaluation:
    """``sizeof`` expressions are evaluated to their machine-level byte counts."""

    def test_sizeof_int(self) -> None:
        """``int a[sizeof(int)];`` resolves to Array(IntType, 4) on x86-64."""
        size_exp = ast.SizeOfT(ast.IntType())
        prog = _prog_with_decls(_array_var("a", ast.IntType(), size_exp))
        resolved = resolve(prog)
        decl = resolved.declarations[0]
        assert decl.variable_declaration.var_type == ast.Array(ast.IntType(), 4)

    def test_sizeof_pointer(self) -> None:
        """``int a[sizeof(void *)];`` resolves to Array(IntType, 8) on x86-64."""
        size_exp = ast.SizeOfT(ast.Pointer(ast.VoidType()))
        prog = _prog_with_decls(_array_var("a", ast.IntType(), size_exp))
        resolved = resolve(prog)
        decl = resolved.declarations[0]
        assert decl.variable_declaration.var_type == ast.Array(ast.IntType(), 8)

    def test_complex_sizeof_expression(self) -> None:
        """``char a[12 * sizeof(int) - 5 * sizeof(void *)];`` → size 8."""
        # 12 * sizeof(int)     = 12 * 4 = 48
        # 5  * sizeof(void *)  = 5  * 8 = 40
        # 48 - 40              = 8
        size_exp = _sub(
            _mul(_int_const(12), ast.SizeOfT(ast.IntType())),
            _mul(_int_const(5), ast.SizeOfT(ast.Pointer(ast.VoidType()))),
        )
        prog = _prog_with_decls(_array_var("a", ast.Char(), size_exp))
        resolved = resolve(prog)
        decl = resolved.declarations[0]
        assert decl.variable_declaration.var_type == ast.Array(ast.Char(), 8)

    def test_sizeof_in_struct_member(self) -> None:
        """``struct { char buf[sizeof(int)]; };`` resolves buf to Array(Char, 4)."""
        member = ast.MemberDeclaration(
            member_name="buf",
            member_type=ast.Array(ast.Char(), ast.SizeOfT(ast.IntType())),
        )
        struct_decl = ast.StructDecl(ast.StructDeclaration(tag="S", members=(member,)))
        prog = _prog_with_decls(struct_decl)
        resolved = resolve(prog)
        resolved_struct = resolved.declarations[0]
        assert isinstance(resolved_struct, ast.StructDecl)
        resolved_member = resolved_struct.struct_declaration.members[0]
        assert isinstance(resolved_member.member_type, ast.Array)
        assert resolved_member.member_type.size == 4


class TestInvalidArrayDimensions:
    """Non-constant or non-positive array dimensions raise SemanticError."""

    def test_zero_size_raises(self) -> None:
        """``int a[0];`` raises SemanticError (zero-length array)."""
        prog = _prog_with_decls(_array_var("a", ast.IntType(), _int_const(0)))
        with pytest.raises(SemanticError, match="positive"):
            resolve(prog)

    def test_negative_size_raises(self) -> None:
        """``int a[-1];`` raises SemanticError."""
        prog = _prog_with_decls(_array_var("a", ast.IntType(), ast.Unary(ast.Negate(), _int_const(1))))
        with pytest.raises(SemanticError, match="positive"):
            resolve(prog)
