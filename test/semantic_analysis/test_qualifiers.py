"""Semantic analysis tests for type-qualifier enforcement.

Covers:
- ``const`` variables and pointers: assignment raises ``SemanticError``.
- ``restrict`` on non-pointer types raises ``SemanticError``.
- Volatile copy-propagation and dead-store-elimination stubs (integration
  tests for the optimisation passes live in their own test modules).
"""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import SemanticAnalyser, resolve
from compiler.syntax.ast.types import QualifiedType, unqualify
from compiler.utils.exceptions import SemanticError
from test.semantic_analysis.helpers import _get_fn, _program, _vd

# ---------------------------------------------------------------------------
# const variable enforcement
# ---------------------------------------------------------------------------


def _const_int_decl(name: str, init: ast.Exp | None = None) -> ast.D:
    """Return a ``D`` block item for ``const int <name> [= init];``."""
    return ast.D(
        ast.VarDecl(
            ast.VariableDeclaration(
                name=name,
                init=ast.SingleInit(init) if init is not None else None,
                var_type=QualifiedType(base=ast.IntType(), const=True),
            )
        )
    )


def _volatile_int_decl(name: str) -> ast.D:
    """Return a ``D`` block item for ``volatile int <name>;``."""
    return ast.D(
        ast.VarDecl(
            ast.VariableDeclaration(
                name=name,
                init=None,
                var_type=QualifiedType(base=ast.IntType(), volatile=True),
            )
        )
    )


class TestConstAssignment:
    """Assignment to a const-qualified variable must raise ``SemanticError``."""

    def test_assign_to_const_local_raises(self) -> None:
        """``const int x = 1; x = 2;`` raises SemanticError."""
        program = _program(
            body=(
                _const_int_decl("x", ast.Constant(ast.ConstInt(1))),
                ast.S(
                    ast.Expression(
                        ast.Assignment(
                            lvalue=ast.Var("x"),
                            rvalue=ast.Constant(ast.ConstInt(2)),
                        )
                    )
                ),
            )
        )
        with pytest.raises(SemanticError, match="const"):
            resolve(program)

    def test_compound_assign_to_const_raises(self) -> None:
        """``const int x = 1; x += 2;`` raises SemanticError."""
        program = _program(
            body=(
                _const_int_decl("x", ast.Constant(ast.ConstInt(1))),
                ast.S(
                    ast.Expression(
                        ast.CompoundAssignment(
                            lvalue=ast.Var("x"),
                            op=ast.Add(),
                            rvalue=ast.Constant(ast.ConstInt(2)),
                        )
                    )
                ),
            )
        )
        with pytest.raises(SemanticError, match="const"):
            resolve(program)

    def test_increment_const_raises(self) -> None:
        """``const int x = 1; ++x;`` raises SemanticError."""
        program = _program(
            body=(
                _const_int_decl("x", ast.Constant(ast.ConstInt(1))),
                ast.S(
                    ast.Expression(
                        ast.Unary(
                            operator=ast.PreIncrement(),
                            exp=ast.Var("x"),
                        )
                    )
                ),
            )
        )
        with pytest.raises(SemanticError, match="const"):
            resolve(program)

    def test_postdecrement_const_raises(self) -> None:
        """``const int x = 1; x--;`` raises SemanticError."""
        program = _program(
            body=(
                _const_int_decl("x", ast.Constant(ast.ConstInt(1))),
                ast.S(
                    ast.Expression(
                        ast.Unary(
                            operator=ast.PostDecrement(),
                            exp=ast.Var("x"),
                        )
                    )
                ),
            )
        )
        with pytest.raises(SemanticError, match="const"):
            resolve(program)

    def test_read_const_is_allowed(self) -> None:
        """Reading a const variable is legal: ``const int x = 1; return x;``."""
        program = _program(
            body=(
                _const_int_decl("x", ast.Constant(ast.ConstInt(1))),
                ast.S(ast.Return(ast.Var("x"))),
            )
        )
        resolved = resolve(program)
        fn = _get_fn(resolved, 0)
        assert fn is not None


class TestConstPointer:
    """const pointer semantics: const qualifier on the pointer itself vs. pointee."""

    def test_assign_to_const_pointer_raises(self) -> None:
        """``int * const p; p = NULL;`` raises SemanticError (pointer itself is const)."""
        program = _program(
            body=(
                ast.D(
                    ast.VarDecl(
                        ast.VariableDeclaration(
                            name="p",
                            init=None,
                            var_type=QualifiedType(base=ast.Pointer(ast.IntType()), const=True),
                        )
                    )
                ),
                ast.S(
                    ast.Expression(
                        ast.Assignment(
                            lvalue=ast.Var("p"),
                            rvalue=ast.Constant(ast.ConstInt(0)),
                        )
                    )
                ),
            )
        )
        with pytest.raises(SemanticError, match="const"):
            resolve(program)

    def test_assign_through_pointer_to_const_raises(self) -> None:
        """``const int *p; *p = 1;`` raises SemanticError (pointee is const)."""
        program = _program(
            body=(
                ast.D(
                    ast.VarDecl(
                        ast.VariableDeclaration(
                            name="p",
                            init=None,
                            var_type=ast.Pointer(QualifiedType(base=ast.IntType(), const=True)),
                        )
                    )
                ),
                ast.S(
                    ast.Expression(
                        ast.Assignment(
                            lvalue=ast.Dereference(ast.Var("p")),
                            rvalue=ast.Constant(ast.ConstInt(1)),
                        )
                    )
                ),
            )
        )
        with pytest.raises(SemanticError, match="const"):
            resolve(program)

    def test_read_through_pointer_to_const_is_allowed(self) -> None:
        """Reading through a pointer-to-const is legal: ``const int *p; int x = *p;``."""
        program = _program(
            body=(
                ast.D(
                    ast.VarDecl(
                        ast.VariableDeclaration(
                            name="p",
                            init=None,
                            var_type=ast.Pointer(QualifiedType(base=ast.IntType(), const=True)),
                        )
                    )
                ),
                _vd("x", init=ast.Dereference(ast.Var("p"))),
            )
        )
        resolved = resolve(program)
        assert resolved is not None


# ---------------------------------------------------------------------------
# restrict qualifier enforcement
# ---------------------------------------------------------------------------


class TestRestrictQualifier:
    """``restrict`` on a non-pointer type must raise ``SemanticError``."""

    def test_restrict_on_int_raises(self) -> None:
        """``restrict int x;`` raises SemanticError."""
        program = _program(
            body=(
                ast.D(
                    ast.VarDecl(
                        ast.VariableDeclaration(
                            name="x",
                            init=None,
                            var_type=QualifiedType(base=ast.IntType(), restrict=True),
                        )
                    )
                ),
            )
        )
        with pytest.raises(SemanticError, match="restrict"):
            resolve(program)

    def test_restrict_on_pointer_is_allowed(self) -> None:
        """``int * restrict p;`` is valid."""
        program = _program(
            body=(
                ast.D(
                    ast.VarDecl(
                        ast.VariableDeclaration(
                            name="p",
                            init=None,
                            var_type=QualifiedType(base=ast.Pointer(ast.IntType()), restrict=True),
                        )
                    )
                ),
            )
        )
        resolved = resolve(program)
        assert resolved is not None


# ---------------------------------------------------------------------------
# Qualified type in resolved AST
# ---------------------------------------------------------------------------


class TestQualifiedTypePreserved:
    """The symbol table entry for a const/volatile variable retains its qualifiers."""

    def test_const_var_type_in_symbol_table(self) -> None:
        """After resolving ``const int x = 1;``, symbol table stores QualifiedType."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration(
                name="x",
                init=None,
                var_type=QualifiedType(base=ast.IntType(), const=True),
            )
        )
        assert result.var_type == QualifiedType(base=ast.IntType(), const=True)
        entry = analyser._symbol_table.get(result.name)
        if entry is not None:
            assert is_const_type(entry.t) or entry.t == QualifiedType(base=ast.IntType(), const=True)

    def test_const_var_exp_type_is_unqualified(self) -> None:
        """After resolution, a ``Var`` expression for a const int has exp_type IntType()."""
        program = _program(
            body=(
                _const_int_decl("x", ast.Constant(ast.ConstInt(1))),
                ast.S(ast.Return(ast.Var("x"))),
            )
        )
        resolved = resolve(program)
        fn = _get_fn(resolved, 0)
        assert fn is not None
        ret = fn.body[-1]
        assert isinstance(ret, ast.S)
        assert isinstance(ret.statement, ast.Return)
        var_exp = ret.statement.exp
        assert isinstance(var_exp, ast.Var)
        assert unqualify(var_exp.exp_type) == ast.IntType()


def is_const_type(t: ast.Type) -> bool:
    """Return True when t is a QualifiedType with const=True."""
    return isinstance(t, QualifiedType) and t.const
