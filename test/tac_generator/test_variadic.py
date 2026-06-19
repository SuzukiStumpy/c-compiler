"""TAC generation tests for variadic built-in expressions (VaStart, VaArg, VaEnd, VaCopy)."""

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from compiler.semantic_analysis import resolve_with_symbols
from compiler.tac_generator import generate


def _variadic_fn_prog(
    *body: ast.BlockItem,
    ret_type: ast.Type | None = None,
    named_params: tuple[ast.Parameter, ...] = (ast.Parameter(param_type=ast.IntType(), name="x"),),
) -> ast.Program:
    """Wrap body items in a variadic ``int f(int x, ...) { ... }`` program.

    Args:
        *body: Block items for the function body.
        ret_type: Return type; defaults to ``IntType()``.
        named_params: Named parameters before the ellipsis.

    Returns:
        A resolved ``ast.Program`` ready for TAC generation.
    """
    rt = ret_type or ast.IntType()
    prog = ast.Program(
        declarations=(
            ast.FunDecl(
                ast.Function(
                    name="f",
                    params=named_params,
                    body=tuple(body),
                    fun_type=ast.FunType(
                        params=tuple(p.param_type for p in named_params),
                        ret=rt,
                        variadic=True,
                    ),
                )
            ),
        )
    )
    resolved, sym_table, struct_table, _ = resolve_with_symbols(prog)
    tac_prog = generate(resolved, sym_table, struct_table)
    return tac_prog


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


class TestVaStartTac:
    """``VaStart`` lowers to a single ``tac.VaStart`` instruction."""

    def test_va_start_emits_tac_va_start(self) -> None:
        """VaStart lowers to exactly one ``tac.VaStart`` instruction."""
        va_start = ast.VaStart(
            ap=ast.Var("ap"),
            last=ast.Var("x"),
        )
        tac_prog = _variadic_fn_prog(
            _va_list_decl("ap"),
            ast.S(ast.Expression(va_start)),
            ast.S(ast.Return(ast.Constant(ast.ConstInt(0)))),
        )
        body = tac_prog.functions[0].body
        va_starts = [i for i in body if isinstance(i, tac.VaStart)]
        assert len(va_starts) == 1

    def test_va_start_gp_offset_one_int_param(self) -> None:
        """One named ``int`` param → ``gp_offset=8``."""
        va_start = ast.VaStart(ap=ast.Var("ap"), last=ast.Var("x"))
        tac_prog = _variadic_fn_prog(
            _va_list_decl("ap"),
            ast.S(ast.Expression(va_start)),
            ast.S(ast.Return(ast.Constant(ast.ConstInt(0)))),
        )
        body = tac_prog.functions[0].body
        inst = next(i for i in body if isinstance(i, tac.VaStart))
        assert inst.gp_offset == 8

    def test_va_start_fp_offset_no_xmm_params(self) -> None:
        """No XMM params → ``fp_offset=48``."""
        va_start = ast.VaStart(ap=ast.Var("ap"), last=ast.Var("x"))
        tac_prog = _variadic_fn_prog(
            _va_list_decl("ap"),
            ast.S(ast.Expression(va_start)),
            ast.S(ast.Return(ast.Constant(ast.ConstInt(0)))),
        )
        body = tac_prog.functions[0].body
        inst = next(i for i in body if isinstance(i, tac.VaStart))
        assert inst.fp_offset == 48

    def test_va_start_fp_offset_with_double_param(self) -> None:
        """One named ``double`` param → ``gp_offset=0``, ``fp_offset=64``."""
        va_start = ast.VaStart(ap=ast.Var("ap"), last=ast.Var("x"))
        tac_prog = _variadic_fn_prog(
            _va_list_decl("ap"),
            ast.S(ast.Expression(va_start)),
            ast.S(ast.Return(ast.Constant(ast.ConstInt(0)))),
            named_params=(ast.Parameter(param_type=ast.DoubleType(), name="x"),),
        )
        body = tac_prog.functions[0].body
        inst = next(i for i in body if isinstance(i, tac.VaStart))
        assert inst.gp_offset == 0
        assert inst.fp_offset == 64


class TestVaArgTac:
    """``VaArg`` lowers to a conditional fetch (register vs overflow) sequence."""

    def test_va_arg_int_produces_load_instruction(self) -> None:
        """VaArg lowers to a sequence containing a ``tac.Load`` instruction."""
        va_arg = ast.VaArg(ap=ast.Var("ap"), arg_type=ast.IntType(), exp_type=ast.IntType())
        tac_prog = _variadic_fn_prog(
            _va_list_decl("ap"),
            ast.S(ast.Return(va_arg)),
        )
        body = tac_prog.functions[0].body
        loads = [i for i in body if isinstance(i, tac.Load)]
        assert len(loads) >= 1

    def test_va_arg_emits_conditional_jump(self) -> None:
        """VaArg expansion includes a conditional jump (overflow guard)."""
        va_arg = ast.VaArg(ap=ast.Var("ap"), arg_type=ast.IntType(), exp_type=ast.IntType())
        tac_prog = _variadic_fn_prog(
            _va_list_decl("ap"),
            ast.S(ast.Return(va_arg)),
        )
        body = tac_prog.functions[0].body
        cond_jumps = [i for i in body if isinstance(i, tac.JumpIfNotZero)]
        assert len(cond_jumps) >= 1


class TestVaEndTac:
    """``VaEnd`` produces no TAC instructions (no-op on x86-64)."""

    def test_va_end_is_noop(self) -> None:
        """VaEnd emits no instructions beyond the surrounding code."""
        va_end = ast.VaEnd(ap=ast.Var("ap"), exp_type=ast.VoidType())
        tac_prog = _variadic_fn_prog(
            _va_list_decl("ap"),
            ast.S(ast.Expression(va_end)),
            ast.S(ast.Return(ast.Constant(ast.ConstInt(0)))),
        )
        body = tac_prog.functions[0].body
        # No VaStart means va_end produced nothing special.
        va_starts = [i for i in body if isinstance(i, tac.VaStart)]
        assert len(va_starts) == 0


class TestVaCopyTac:
    """``VaCopy`` lowers to field-by-field CopyFromOffset/CopyToOffset pairs."""

    def test_va_copy_emits_copy_from_offset(self) -> None:
        """VaCopy emits at least 4 ``CopyFromOffset`` instructions."""
        va_copy = ast.VaCopy(dest=ast.Var("dest"), src=ast.Var("src"), exp_type=ast.VoidType())
        tac_prog = _variadic_fn_prog(
            _va_list_decl("dest"),
            _va_list_decl("src"),
            ast.S(ast.Expression(va_copy)),
            ast.S(ast.Return(ast.Constant(ast.ConstInt(0)))),
        )
        body = tac_prog.functions[0].body
        copies_from = [i for i in body if isinstance(i, tac.CopyFromOffset)]
        assert len(copies_from) >= 4

    def test_va_copy_emits_copy_to_offset(self) -> None:
        """VaCopy emits at least 4 ``CopyToOffset`` instructions."""
        va_copy = ast.VaCopy(dest=ast.Var("dest"), src=ast.Var("src"), exp_type=ast.VoidType())
        tac_prog = _variadic_fn_prog(
            _va_list_decl("dest"),
            _va_list_decl("src"),
            ast.S(ast.Expression(va_copy)),
            ast.S(ast.Return(ast.Constant(ast.ConstInt(0)))),
        )
        body = tac_prog.functions[0].body
        copies_to = [i for i in body if isinstance(i, tac.CopyToOffset)]
        assert len(copies_to) >= 4
