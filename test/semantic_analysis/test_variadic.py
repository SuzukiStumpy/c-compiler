"""Semantic analysis tests for variadic function support (va_list / va_arg etc.)."""

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve
from test.semantic_analysis.helpers import _make_prog


def _variadic_fn(
    *body: ast.BlockItem,
    name: str = "f",
    named_params: tuple[ast.Parameter, ...] = (ast.Parameter(param_type=ast.IntType(), name="x"),),
) -> ast.Function:
    """Return a variadic ``int f(int x, ...) { *body }`` function AST node.

    Args:
        *body: Block items for the function body.
        name: Function name.
        named_params: Named parameters before the ellipsis.

    Returns:
        An ``ast.Function`` with ``fun_type.variadic=True``.
    """
    return ast.Function(
        name=name,
        params=named_params,
        body=tuple(body),
        fun_type=ast.FunType(
            params=tuple(p.param_type for p in named_params),
            ret=ast.IntType(),
            variadic=True,
        ),
    )


def _va_list_decl(var_name: str) -> ast.D:
    """Return ``__va_list_tag var_name;`` as a ``D`` block item.

    Args:
        var_name: Name of the va_list variable.

    Returns:
        An ``ast.D`` wrapping a ``VarDecl`` of type ``Structure("__va_list_tag")``.
    """
    return ast.D(
        ast.VarDecl(
            ast.VariableDeclaration(
                name=var_name,
                init=None,
                var_type=ast.Structure("__va_list_tag"),
            )
        )
    )


def _ret_zero() -> ast.S:
    """Return ``return 0;`` as an ``S`` block item.

    Returns:
        An ``ast.S`` wrapping a ``Return`` of ``Constant(ConstInt(0))``.
    """
    return ast.S(ast.Return(ast.Constant(ast.ConstInt(0))))


# ---------------------------------------------------------------------------
# __va_list_tag pre-populated in struct_type_table
# ---------------------------------------------------------------------------


class TestVaListTagPrePopulated:
    """``__va_list_tag`` is available without a source-level struct definition."""

    def test_va_list_struct_variable_resolves(self) -> None:
        """Declaring ``__va_list_tag ap;`` in a variadic function resolves cleanly."""
        prog = _make_prog(_variadic_fn(_va_list_decl("ap"), _ret_zero()))
        resolved = resolve(prog)
        fn = resolved.declarations[0]
        assert isinstance(fn, ast.FunDecl)
        body = fn.function_declaration.body
        assert body is not None
        ap_decl = body[0]
        assert isinstance(ap_decl, ast.D)
        vd = ap_decl.declaration
        assert isinstance(vd, ast.VarDecl)
        # Resolved name carries the unique suffix but type must be Structure with __va_list_tag.
        assert isinstance(vd.variable_declaration.var_type, ast.Structure)
        assert vd.variable_declaration.var_type.tag == "__va_list_tag"


# ---------------------------------------------------------------------------
# VaStart type-check
# ---------------------------------------------------------------------------


class TestVaStartTypeCheck:
    """``VaStart`` nodes resolve and get ``exp_type=VoidType()``."""

    def test_va_start_has_void_type(self) -> None:
        """``__builtin_va_start(ap, x)`` resolves with ``exp_type=VoidType()``."""
        va_start = ast.VaStart(
            ap=ast.Var("ap"),
            last=ast.Var("x"),
        )
        prog = _make_prog(
            _variadic_fn(
                _va_list_decl("ap"),
                ast.S(ast.Expression(va_start)),
                _ret_zero(),
            )
        )
        resolved = resolve(prog)
        fn = resolved.declarations[0].function_declaration  # type: ignore[union-attr]
        assert fn.body is not None
        stmt = fn.body[1]
        assert isinstance(stmt, ast.S)
        assert isinstance(stmt.statement, ast.Expression)
        node = stmt.statement.exp
        assert isinstance(node, ast.VaStart)
        assert node.exp_type == ast.VoidType()


# ---------------------------------------------------------------------------
# VaArg type-check
# ---------------------------------------------------------------------------


class TestVaArgTypeCheck:
    """``VaArg`` nodes resolve and get ``exp_type`` equal to the requested type."""

    def test_va_arg_int_type(self) -> None:
        """``__builtin_va_arg(ap, int)`` has ``exp_type=IntType()``."""
        va_arg = ast.VaArg(ap=ast.Var("ap"), arg_type=ast.IntType())
        prog = _make_prog(
            _variadic_fn(
                _va_list_decl("ap"),
                ast.S(ast.Return(va_arg)),
            )
        )
        resolved = resolve(prog)
        fn = resolved.declarations[0].function_declaration  # type: ignore[union-attr]
        assert fn.body is not None
        ret = fn.body[1]
        assert isinstance(ret, ast.S)
        assert isinstance(ret.statement, ast.Return)
        node = ret.statement.exp
        assert isinstance(node, ast.VaArg)
        assert node.exp_type == ast.IntType()

    def test_va_arg_long_type(self) -> None:
        """``__builtin_va_arg(ap, long)`` has ``exp_type=LongType()``."""
        va_arg = ast.VaArg(ap=ast.Var("ap"), arg_type=ast.LongType())
        # Use a long-returning function so no implicit cast wraps the VaArg node.
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    ast.Function(
                        name="f",
                        params=(ast.Parameter(param_type=ast.LongType(), name="x"),),
                        body=(
                            _va_list_decl("ap"),
                            ast.S(ast.Return(va_arg)),
                        ),
                        fun_type=ast.FunType(
                            params=(ast.LongType(),),
                            ret=ast.LongType(),
                            variadic=True,
                        ),
                    )
                ),
            )
        )
        resolved = resolve(prog)
        fn = resolved.declarations[0].function_declaration  # type: ignore[union-attr]
        assert fn.body is not None
        ret = fn.body[1]
        assert isinstance(ret, ast.S)
        assert isinstance(ret.statement, ast.Return)
        node = ret.statement.exp
        assert isinstance(node, ast.VaArg)
        assert node.exp_type == ast.LongType()


# ---------------------------------------------------------------------------
# VaEnd type-check
# ---------------------------------------------------------------------------


class TestVaEndTypeCheck:
    """``VaEnd`` nodes resolve and get ``exp_type=VoidType()``."""

    def test_va_end_has_void_type(self) -> None:
        """``__builtin_va_end(ap)`` has ``exp_type=VoidType()``."""
        va_end = ast.VaEnd(ap=ast.Var("ap"))
        prog = _make_prog(
            _variadic_fn(
                _va_list_decl("ap"),
                ast.S(ast.Expression(va_end)),
                _ret_zero(),
            )
        )
        resolved = resolve(prog)
        fn = resolved.declarations[0].function_declaration  # type: ignore[union-attr]
        assert fn.body is not None
        stmt = fn.body[1]
        assert isinstance(stmt, ast.S)
        assert isinstance(stmt.statement, ast.Expression)
        node = stmt.statement.exp
        assert isinstance(node, ast.VaEnd)
        assert node.exp_type == ast.VoidType()


# ---------------------------------------------------------------------------
# VaCopy type-check
# ---------------------------------------------------------------------------


class TestVaCopyTypeCheck:
    """``VaCopy`` nodes resolve and get ``exp_type=VoidType()``."""

    def test_va_copy_has_void_type(self) -> None:
        """``__builtin_va_copy(dest, src)`` has ``exp_type=VoidType()``."""
        va_copy = ast.VaCopy(dest=ast.Var("dest"), src=ast.Var("src"))
        prog = _make_prog(
            _variadic_fn(
                _va_list_decl("dest"),
                _va_list_decl("src"),
                ast.S(ast.Expression(va_copy)),
                _ret_zero(),
            )
        )
        resolved = resolve(prog)
        fn = resolved.declarations[0].function_declaration  # type: ignore[union-attr]
        assert fn.body is not None
        stmt = fn.body[2]
        assert isinstance(stmt, ast.S)
        assert isinstance(stmt.statement, ast.Expression)
        node = stmt.statement.exp
        assert isinstance(node, ast.VaCopy)
        assert node.exp_type == ast.VoidType()
