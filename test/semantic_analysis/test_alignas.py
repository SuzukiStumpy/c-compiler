"""Tests for ``_Alignas`` alignment specifier in semantic analysis."""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import _resolved_body, _vd


def _alignas_var(name: str, align_exp: ast.Exp, var_type: ast.Type | None = None) -> ast.D:
    """Return a ``D`` block item wrapping a variable declaration with an alignment specifier.

    Args:
        name: The variable name.
        align_exp: The raw alignment expression (``AlignOfT`` or ``Constant``).
        var_type: The declared type; defaults to ``IntType``.

    Returns:
        An ``ast.D`` wrapping ``ast.VarDecl(ast.VariableDeclaration(..., align_as_exp=align_exp))``.
    """
    return ast.D(ast.VarDecl(ast.VariableDeclaration(name, None, var_type or ast.IntType(), align_as_exp=align_exp)))


def _return_zero() -> tuple[ast.BlockItem, ...]:
    """Return a ``return 0;`` body item."""
    return (ast.S(ast.Return(ast.Constant(ast.ConstInt(0)))),)


class TestAlignAsEvaluation:
    """SA evaluates ``_Alignas`` expressions to integer alignment values."""

    def test_alignas_integer_constant_stored_on_var_decl(self) -> None:
        """``_Alignas(16) int x;`` stores ``align_as=16`` on the resolved declaration."""
        body = (_alignas_var("x", ast.Constant(ast.ConstInt(16))),) + _return_zero()
        resolved_body = _resolved_body(body)
        item = resolved_body[0]
        assert isinstance(item, ast.D)
        assert isinstance(item.declaration, ast.VarDecl)
        var = item.declaration.variable_declaration
        assert var.align_as == 16

    def test_alignas_type_int_stored_as_4(self) -> None:
        """``_Alignas(int) double x;`` stores ``align_as=4`` (alignment of int)."""
        body = (_alignas_var("x", ast.AlignOfT(ast.IntType()), ast.DoubleType()),) + _return_zero()
        resolved_body = _resolved_body(body)
        item = resolved_body[0]
        assert isinstance(item, ast.D)
        var = item.declaration.variable_declaration
        assert var.align_as == 4

    def test_alignas_type_double_stored_as_8(self) -> None:
        """``_Alignas(double) int x;`` stores ``align_as=8`` (alignment of double)."""
        body = (_alignas_var("x", ast.AlignOfT(ast.DoubleType())),) + _return_zero()
        resolved_body = _resolved_body(body)
        item = resolved_body[0]
        assert isinstance(item, ast.D)
        var = item.declaration.variable_declaration
        assert var.align_as == 8

    def test_no_alignas_gives_none(self) -> None:
        """A plain ``int x;`` has ``align_as=None`` after resolution."""
        body = (_vd("x", ast.Constant(ast.ConstInt(0))),) + _return_zero()
        resolved_body = _resolved_body(body)
        item = resolved_body[0]
        assert isinstance(item, ast.D)
        var = item.declaration.variable_declaration
        assert var.align_as is None

    def test_alignas_non_power_of_two_raises(self) -> None:
        """``_Alignas(3) int x;`` raises ``SemanticError`` (not a power of two)."""
        body = (_alignas_var("x", ast.Constant(ast.ConstInt(3))),) + _return_zero()
        with pytest.raises(SemanticError, match="power of two"):
            _resolved_body(body)

    def test_alignas_zero_raises(self) -> None:
        """``_Alignas(0) int x;`` raises ``SemanticError``."""
        body = (_alignas_var("x", ast.Constant(ast.ConstInt(0))),) + _return_zero()
        with pytest.raises(SemanticError):
            _resolved_body(body)


class TestAlignAsFileScope:
    """SA resolves ``_Alignas`` on file-scope variable declarations."""

    def test_alignas_file_scope_integer(self) -> None:
        """``_Alignas(16) int x;`` at file scope stores ``align_as=16``."""
        from test.semantic_analysis.helpers import _prog_with_decls

        prog = _prog_with_decls(
            ast.VarDecl(ast.VariableDeclaration("x", None, ast.IntType(), align_as_exp=ast.Constant(ast.ConstInt(16))))
        )
        resolved = resolve(prog)
        decl = resolved.declarations[0]
        assert isinstance(decl, ast.VarDecl)
        assert decl.variable_declaration.align_as == 16

    def test_alignas_file_scope_type(self) -> None:
        """``_Alignas(double) int x;`` at file scope stores ``align_as=8``."""
        from test.semantic_analysis.helpers import _prog_with_decls

        prog = _prog_with_decls(
            ast.VarDecl(ast.VariableDeclaration("x", None, ast.IntType(), align_as_exp=ast.AlignOfT(ast.DoubleType())))
        )
        resolved = resolve(prog)
        decl = resolved.declarations[0]
        assert isinstance(decl, ast.VarDecl)
        assert decl.variable_declaration.align_as == 8
