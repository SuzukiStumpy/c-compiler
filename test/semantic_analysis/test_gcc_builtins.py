"""Semantic analysis tests for GCC built-in expressions.

Covers ``__builtin_unreachable`` and ``__builtin_offsetof``.
``__builtin_expect`` needs no SA coverage because it is folded to its first
argument at parse time and carries no new AST node.
"""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import _program

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _return_exp(exp: ast.Exp) -> tuple[ast.BlockItem, ...]:
    """Return a ``(S(Return(exp)),)`` block-item tuple."""
    return (ast.S(ast.Return(exp)),)


def _expr_stmt(exp: ast.Exp) -> tuple[ast.BlockItem, ...]:
    """Return a ``(S(Expression(exp)),)`` block-item tuple."""
    return (ast.S(ast.Expression(exp)),)


def _return_zero() -> tuple[ast.BlockItem, ...]:
    return _return_exp(ast.Constant(ast.ConstInt(0)))


def _struct_prog(
    tag: str,
    members: list[tuple[str, ast.Type]],
    body: tuple[ast.BlockItem, ...],
) -> ast.Program:
    """Return a program containing a struct declaration followed by main."""
    member_decls = tuple(ast.MemberDeclaration(name, t) for name, t in members)
    struct_decl = ast.StructDecl(ast.StructDeclaration(tag=tag, members=member_decls))
    main_fn = ast.Function(
        name="main",
        params=(),
        body=body,
        fun_type=ast.FunType(params=(), ret=ast.IntType()),
    )
    return ast.Program(declarations=(struct_decl, ast.FunDecl(main_fn)))


# ---------------------------------------------------------------------------
# __builtin_unreachable
# ---------------------------------------------------------------------------


class TestBuiltinUnreachable:
    """SA correctly annotates ``BuiltinUnreachable`` as ``VoidType``."""

    def test_unreachable_has_void_type(self) -> None:
        """``__builtin_unreachable()`` is given ``exp_type=VoidType()`` by SA."""
        body = _expr_stmt(ast.BuiltinUnreachable()) + _return_zero()
        resolved = resolve(_program(body))
        stmt = resolved.declarations[0].function_declaration.body[0].statement
        assert isinstance(stmt, ast.Expression)
        assert stmt.exp.exp_type == ast.VoidType()
        assert isinstance(stmt.exp, ast.BuiltinUnreachable)

    def test_unreachable_as_statement_before_return(self) -> None:
        """``__builtin_unreachable(); return 0;`` resolves without error."""
        body = _expr_stmt(ast.BuiltinUnreachable()) + _return_zero()
        resolved = resolve(_program(body))
        assert len(resolved.declarations[0].function_declaration.body) == 2


# ---------------------------------------------------------------------------
# __builtin_offsetof
# ---------------------------------------------------------------------------


class TestBuiltinOffsetof:
    """SA resolves ``OffsetOf`` to ``ULongType`` and validates struct membership."""

    def test_offsetof_simple_member_has_ulong_type(self) -> None:
        """``offsetof(struct s, x)`` is annotated with ``ULongType`` before return-cast."""
        exp = ast.OffsetOf(t=ast.Structure("s"), member="x")
        body = _expr_stmt(exp) + _return_zero()
        prog = _struct_prog("s", [("x", ast.IntType()), ("y", ast.IntType())], body)
        resolved = resolve(prog)
        fn_body = resolved.declarations[1].function_declaration.body
        stmt = fn_body[0].statement
        assert isinstance(stmt, ast.Expression)
        assert isinstance(stmt.exp, ast.OffsetOf)
        assert stmt.exp.exp_type == ast.ULongType()

    def test_offsetof_resolved_type_is_structure(self) -> None:
        """``OffsetOf`` retains the resolved ``Structure`` type after SA."""
        exp = ast.OffsetOf(t=ast.Structure("s"), member="x")
        body = _expr_stmt(exp) + _return_zero()
        prog = _struct_prog("s", [("x", ast.IntType()), ("y", ast.IntType())], body)
        resolved = resolve(prog)
        fn_body = resolved.declarations[1].function_declaration.body
        stmt = fn_body[0].statement
        assert isinstance(stmt, ast.Expression)
        assert isinstance(stmt.exp, ast.OffsetOf)
        assert isinstance(stmt.exp.t, ast.Structure)

    def test_offsetof_second_member_has_ulong_type(self) -> None:
        """``offsetof(struct {int x; int y;}, y)`` has ``ULongType`` after SA."""
        exp = ast.OffsetOf(t=ast.Structure("pt"), member="y")
        body = _expr_stmt(exp) + _return_zero()
        prog = _struct_prog("pt", [("x", ast.IntType()), ("y", ast.IntType())], body)
        resolved = resolve(prog)
        fn_body = resolved.declarations[1].function_declaration.body
        stmt = fn_body[0].statement
        assert isinstance(stmt, ast.Expression)
        assert isinstance(stmt.exp, ast.OffsetOf)
        assert stmt.exp.exp_type == ast.ULongType()

    def test_offsetof_unknown_member_raises(self) -> None:
        """``offsetof(struct s, missing)`` raises ``SemanticError``."""
        exp = ast.OffsetOf(t=ast.Structure("s"), member="missing")
        body = _return_exp(exp)
        prog = _struct_prog("s", [("x", ast.IntType())], body)
        with pytest.raises(SemanticError, match="member"):
            resolve(prog)

    def test_offsetof_non_struct_raises(self) -> None:
        """``offsetof(int, x)`` raises ``SemanticError`` (not a struct/union type)."""
        exp = ast.OffsetOf(t=ast.IntType(), member="x")
        body = _return_exp(exp)
        with pytest.raises(SemanticError):
            resolve(_program(body))

    def test_offsetof_incomplete_struct_raises(self) -> None:
        """``offsetof`` on a forward-declared struct raises ``SemanticError``."""
        exp = ast.OffsetOf(t=ast.Structure("incomplete"), member="x")
        body = _return_exp(exp)
        # No struct declaration in the program — struct is incomplete
        with pytest.raises(SemanticError, match="incomplete"):
            resolve(_program(body))
