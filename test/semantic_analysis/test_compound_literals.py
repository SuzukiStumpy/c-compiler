"""Semantic analysis tests for compound literal expressions: ``(Type){initializer}``."""

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve
from test.semantic_analysis.helpers import _resolved_body


def _lit(target_type: ast.Type, *exps: ast.Exp) -> ast.CompoundLiteral:
    """Build a ``CompoundLiteral`` with positional ``SingleInit`` members.

    Args:
        target_type: The declared type of the compound literal.
        *exps: The initialiser expressions, each wrapped in a ``SingleInit``.

    Returns:
        A ``CompoundLiteral`` node with a ``CompoundInit`` initialiser.
    """
    inits: list[ast.Initializer] = [ast.SingleInit(e) for e in exps]
    return ast.CompoundLiteral(
        target_type=target_type,
        init=ast.CompoundInit(initializers=inits),
    )


class TestScalarCompoundLiteralSA:
    """Scalar compound literals are type-checked and acquire the correct exp_type."""

    def test_int_compound_literal_has_int_type(self) -> None:
        """``(int){42}`` resolves with ``exp_type = IntType``."""
        lit = _lit(ast.IntType(), ast.Constant(ast.ConstInt(42)))
        body: tuple[ast.BlockItem, ...] = (ast.S(ast.Return(exp=lit)),)
        resolved_body = _resolved_body(body)
        ret = resolved_body[0]
        assert isinstance(ret, ast.S)
        assert isinstance(ret.statement, ast.Return)
        resolved_exp = ret.statement.exp
        assert isinstance(resolved_exp, ast.CompoundLiteral)
        assert resolved_exp.exp_type == ast.IntType()

    def test_long_compound_literal_has_long_type(self) -> None:
        """``(long){1L}`` resolves with ``exp_type = LongType``."""
        lit = _lit(ast.LongType(), ast.Constant(ast.ConstLong(1)))
        # Store in a variable to avoid implicit return-type cast.
        body: tuple[ast.BlockItem, ...] = (
            ast.D(ast.VarDecl(ast.VariableDeclaration("v", ast.SingleInit(lit), ast.LongType()))),
            ast.S(ast.Return(exp=ast.Constant(ast.ConstInt(0)))),
        )
        resolved_body = _resolved_body(body)
        decl_item = resolved_body[0]
        assert isinstance(decl_item, ast.D)
        assert isinstance(decl_item.declaration, ast.VarDecl)
        init = decl_item.declaration.variable_declaration.init
        assert isinstance(init, ast.SingleInit)
        assert isinstance(init.exp, ast.CompoundLiteral)
        assert init.exp.exp_type == ast.LongType()


class TestStructCompoundLiteralSA:
    """Struct compound literals initialise struct members correctly."""

    def test_struct_compound_literal_resolves(self) -> None:
        """``(struct Point){1, 2}`` resolves without error."""
        struct_decl = ast.StructDecl(
            ast.StructDeclaration(
                tag="Point",
                members=(
                    ast.MemberDeclaration(member_name="x", member_type=ast.IntType()),
                    ast.MemberDeclaration(member_name="y", member_type=ast.IntType()),
                ),
            )
        )
        prog = ast.Program(
            declarations=(
                struct_decl,
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=(ast.S(ast.Return(exp=ast.Constant(ast.ConstInt(0)))),),
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )
        # Register the struct, then resolve a compound literal expression
        resolved = resolve(prog)
        # The struct tag must exist after resolution
        assert any(isinstance(d, ast.StructDecl) for d in resolved.declarations)

    def test_struct_compound_literal_has_struct_type(self) -> None:
        """``(struct Point){...}`` resolves with ``exp_type = Structure(resolved_tag)``."""
        struct_decl = ast.StructDecl(
            ast.StructDeclaration(
                tag="Point",
                members=(
                    ast.MemberDeclaration(member_name="x", member_type=ast.IntType()),
                    ast.MemberDeclaration(member_name="y", member_type=ast.IntType()),
                ),
            )
        )
        lit = ast.CompoundLiteral(
            target_type=ast.Structure(tag="Point"),
            init=ast.CompoundInit(
                initializers=[
                    ast.SingleInit(ast.Constant(ast.ConstInt(1))),
                    ast.SingleInit(ast.Constant(ast.ConstInt(2))),
                ]
            ),
        )
        prog = ast.Program(
            declarations=(
                struct_decl,
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=(ast.S(ast.Return(exp=ast.Constant(ast.ConstInt(0)))),),
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
                ast.FunDecl(
                    ast.Function(
                        name="use_lit",
                        params=(),
                        body=(ast.S(ast.Return(exp=lit)),),
                        fun_type=ast.FunType(params=(), ret=ast.Structure(tag="Point")),
                    )
                ),
            )
        )
        resolved = resolve(prog)
        use_fn = next(
            d.function_declaration
            for d in resolved.declarations
            if isinstance(d, ast.FunDecl) and d.function_declaration.name == "use_lit"
        )
        assert use_fn.body is not None
        ret = use_fn.body[0]
        assert isinstance(ret, ast.S)
        assert isinstance(ret.statement, ast.Return)
        resolved_exp = ret.statement.exp
        assert isinstance(resolved_exp, ast.CompoundLiteral)
        assert isinstance(resolved_exp.exp_type, ast.Structure)
