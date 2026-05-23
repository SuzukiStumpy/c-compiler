"""Tests for function-level TAC generation."""

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from compiler.tac_generator import generate
from test.tac_generator.helpers import _body, _instrs_for_body

# ---------------------------------------------------------------------------
# Multiple functions and forward declarations
# ---------------------------------------------------------------------------


def _make_function(name: str, body: ast.Block, params: tuple[ast.Parameter, ...] = ()) -> ast.Function:
    """Build a minimal ``ast.Function`` definition.

    Args:
        name: The function name.
        body: The block of statements forming the function body.
        params: Optional parameter list; defaults to empty.

    Returns:
        An ``ast.Function`` node with the given name, params, and body.
    """
    return ast.Function(name=name, params=params, body=body, fun_type=ast.FunType(params=(), ret=ast.IntType()))


class TestTacGeneratorMultipleFunctions:
    """``generate`` correctly handles programs with more than one function definition."""

    def test_two_functions_produce_two_tac_functions(self) -> None:
        """A program with two definitions yields two entries in ``tac.Program.functions``."""
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    _make_function("foo", (ast.S(statement=ast.Return(exp=ast.Constant(value=ast.ConstInt(1)))),))
                ),
                ast.FunDecl(
                    _make_function("bar", (ast.S(statement=ast.Return(exp=ast.Constant(value=ast.ConstInt(2)))),))
                ),
            )
        )
        result = generate(prog, {})
        assert len(result.functions) == 2

    def test_function_names_are_preserved_in_order(self) -> None:
        """The TAC functions appear in the same order as the source definitions."""
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    _make_function("alpha", (ast.S(statement=ast.Return(exp=ast.Constant(value=ast.ConstInt(0)))),))
                ),
                ast.FunDecl(
                    _make_function("beta", (ast.S(statement=ast.Return(exp=ast.Constant(value=ast.ConstInt(0)))),))
                ),
            )
        )
        result = generate(prog, {})
        assert result.functions[0].name == "alpha"
        assert result.functions[1].name == "beta"

    def test_forward_declaration_is_skipped(self) -> None:
        """A function with ``body=None`` (forward declaration) is excluded from the TAC output."""
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    ast.Function(
                        name="declared_only", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType())
                    )
                ),
                ast.FunDecl(
                    _make_function("defined", (ast.S(statement=ast.Return(exp=ast.Constant(value=ast.ConstInt(0)))),))
                ),
            )
        )
        result = generate(prog, {})
        assert len(result.functions) == 1
        assert result.functions[0].name == "defined"

    def test_all_forward_declarations_produces_empty_function_list(self) -> None:
        """A program consisting solely of forward declarations yields an empty ``functions`` tuple."""
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    ast.Function(name="f", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
                ),
                ast.FunDecl(
                    ast.Function(name="g", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType()))
                ),
            )
        )
        result = generate(prog, {})
        assert result.functions == ()

    def test_temporaries_are_unique_across_functions(self) -> None:
        """Temporaries allocated in different functions have distinct names."""
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    _make_function(
                        "f",
                        (
                            ast.S(
                                statement=ast.Return(
                                    exp=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(1)))
                                )
                            ),
                        ),
                    )
                ),
                ast.FunDecl(
                    _make_function(
                        "g",
                        (
                            ast.S(
                                statement=ast.Return(
                                    exp=ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(2)))
                                )
                            ),
                        ),
                    )
                ),
            )
        )
        result = generate(prog, {})
        f_temps = {i.dst for i in result.functions[0].body if isinstance(i, tac.Unary)}
        g_temps = {i.dst for i in result.functions[1].body if isinstance(i, tac.Unary)}
        assert len(f_temps) == 1
        assert len(g_temps) == 1
        assert f_temps.isdisjoint(g_temps)


# ---------------------------------------------------------------------------
# Function parameters
# ---------------------------------------------------------------------------


class TestTacGeneratorFunctionParameters:
    """Parameter names are extracted from ``ast.Parameter`` nodes and stored in ``tac.Function.params``."""

    def test_single_parameter_name_preserved(self) -> None:
        """A function with one parameter stores its name in ``tac.Function.params``."""
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    _make_function(
                        "f",
                        (ast.S(statement=ast.Return(exp=ast.Constant(value=ast.ConstInt(0)))),),
                        params=(ast.Parameter(param_type=ast.IntType(), name="x"),),
                    )
                ),
            )
        )
        result = generate(prog, {})
        assert result.functions[0].params == ("x",)

    def test_multiple_parameters_names_preserved_in_order(self) -> None:
        """Multiple parameters are stored in declaration order."""
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    _make_function(
                        "add",
                        (ast.S(statement=ast.Return(exp=ast.Constant(value=ast.ConstInt(0)))),),
                        params=(
                            ast.Parameter(param_type=ast.IntType(), name="a"),
                            ast.Parameter(param_type=ast.IntType(), name="b"),
                        ),
                    )
                ),
            )
        )
        result = generate(prog, {})
        assert result.functions[0].params == ("a", "b")

    def test_no_parameters_yields_empty_params(self) -> None:
        """A function with no parameters has an empty ``params`` tuple."""
        prog = ast.Program(
            declarations=(
                ast.FunDecl(
                    _make_function("main", (ast.S(statement=ast.Return(exp=ast.Constant(value=ast.ConstInt(0)))),))
                ),
            )
        )
        result = generate(prog, {})
        assert result.functions[0].params == ()


# ---------------------------------------------------------------------------
# FunDecl as block item
# ---------------------------------------------------------------------------


class TestTacGeneratorFunDeclBlockItem:
    """A ``FunDecl`` inside a function body is silently ignored (no instructions emitted)."""

    def test_fun_decl_block_item_produces_no_instructions(self) -> None:
        """A forward declaration inside a block contributes no TAC instructions."""
        forward = ast.FunDecl(
            function_declaration=ast.Function(
                name="helper", params=(), body=None, fun_type=ast.FunType(params=(), ret=ast.IntType())
            )
        )
        instrs = _instrs_for_body((ast.D(declaration=forward),))
        assert instrs == (tac.Return(val=tac.Constant(value=ast.ConstInt(0))),)


# ---------------------------------------------------------------------------
# Function call expression
# ---------------------------------------------------------------------------


class TestTacGeneratorFunctionCall:
    """``ast.FunctionCall`` lowers to a ``tac.FunCall`` instruction with evaluated arguments."""

    def test_no_arg_call_produces_funcall(self) -> None:
        """``foo()`` emits a single ``FunCall`` before the ``Return``."""
        instrs = _body(ast.FunctionCall(callee=ast.Var("foo"), args=[]))
        funcalls = [i for i in instrs if isinstance(i, tac.FunCall)]
        assert len(funcalls) == 1
        assert funcalls[0].fun_val == tac.Var(value="foo")

    def test_no_arg_call_has_empty_args(self) -> None:
        """A zero-argument call stores an empty args tuple in ``FunCall``."""
        instrs = _body(ast.FunctionCall(callee=ast.Var("foo"), args=[]))
        funcall = next(i for i in instrs if isinstance(i, tac.FunCall))
        assert funcall.args == ()

    def test_call_result_stored_in_temp(self) -> None:
        """The return value of a call is stored in a temporary that is then returned."""
        instrs = _body(ast.FunctionCall(callee=ast.Var("foo"), args=[]))
        funcall = next(i for i in instrs if isinstance(i, tac.FunCall))
        ret = instrs[-2]  # penultimate; last is the function guard
        assert isinstance(ret, tac.Return)
        assert ret.val == funcall.dst

    def test_constant_arg_becomes_tac_constant(self) -> None:
        """A constant argument is passed directly as a ``tac.Constant``."""
        instrs = _body(ast.FunctionCall(callee=ast.Var("foo"), args=[ast.Constant(value=ast.ConstInt(42))]))
        funcall = next(i for i in instrs if isinstance(i, tac.FunCall))
        assert funcall.args == (tac.Constant(value=ast.ConstInt(42)),)

    def test_multiple_constant_args_preserved_in_order(self) -> None:
        """Multiple constant arguments appear in the ``FunCall.args`` tuple in declaration order."""
        instrs = _body(
            ast.FunctionCall(
                callee=ast.Var("add"), args=[ast.Constant(value=ast.ConstInt(1)), ast.Constant(value=ast.ConstInt(2))]
            )
        )
        funcall = next(i for i in instrs if isinstance(i, tac.FunCall))
        assert funcall.args == (tac.Constant(value=ast.ConstInt(1)), tac.Constant(value=ast.ConstInt(2)))

    def test_expression_arg_emitted_before_funcall(self) -> None:
        """A non-trivial argument expression is fully evaluated before the ``FunCall`` is emitted."""
        instrs = _body(
            ast.FunctionCall(
                callee=ast.Var("f"),
                args=[ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(1)))],
            )
        )
        unary_idx = next(i for i, x in enumerate(instrs) if isinstance(x, tac.Unary))
        funcall_idx = next(i for i, x in enumerate(instrs) if isinstance(x, tac.FunCall))
        assert unary_idx < funcall_idx

    def test_expression_arg_result_passed_to_funcall(self) -> None:
        """The temporary produced by evaluating an argument expression is passed as the arg value."""
        instrs = _body(
            ast.FunctionCall(
                callee=ast.Var("f"),
                args=[ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(1)))],
            )
        )
        unary = next(i for i in instrs if isinstance(i, tac.Unary))
        funcall = next(i for i in instrs if isinstance(i, tac.FunCall))
        assert funcall.args == (unary.dst,)

    def test_multiple_expression_args_evaluated_left_to_right(self) -> None:
        """Arguments with side-effects are evaluated strictly left to right."""
        instrs = _body(
            ast.FunctionCall(
                callee=ast.Var("f"),
                args=[
                    ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(1))),
                    ast.Unary(operator=ast.Negate(), exp=ast.Constant(value=ast.ConstInt(2))),
                ],
            )
        )
        unaries = [i for i in instrs if isinstance(i, tac.Unary)]
        funcall = next(i for i in instrs if isinstance(i, tac.FunCall))
        assert len(unaries) == 2
        assert unaries[0].src == tac.Constant(value=ast.ConstInt(1))
        assert unaries[1].src == tac.Constant(value=ast.ConstInt(2))
        assert funcall.args == (unaries[0].dst, unaries[1].dst)
