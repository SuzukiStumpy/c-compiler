"""Tests for the AST pretty-printer.

Verifies that ``pretty_print`` produces correctly indented, human-readable
output for the full range of AST node shapes.
"""

from compiler.syntax.ast import (
    Break,
    Constant,
    Continue,
    DoWhile,
    For,
    Function,
    FunDecl,
    FunType,
    InitExp,
    IntType,
    Null,
    Parameter,
    Program,
    Return,
    While,
    pretty_print,
)


def test_leaf_node_rendered_inline() -> None:
    """Nodes with only scalar fields are rendered on a single line."""
    assert pretty_print(Constant(value=42)) == "Constant(value=42)"


def test_no_field_node_rendered_inline() -> None:
    """Nodes with no fields (e.g. ``IntType``) are rendered as ``Name()``."""
    assert pretty_print(IntType()) == "IntType()"


def test_parameter_rendered_inline() -> None:
    """``Parameter`` renders inline because all its fields are scalar or leaf nodes."""
    param = Parameter(param_type=IntType(), name="x")
    assert pretty_print(param) == 'Parameter(param_type=IntType(), name="x")'


def test_return_with_constant_is_inline() -> None:
    """``Return`` whose sole field renders inline is itself rendered inline."""
    node = Return(exp=Constant(value=0))
    assert pretty_print(node) == "Return(exp=Constant(value=0))"


def test_minimal_program_indentation() -> None:
    """Full ``Program`` renders with correctly nested indentation."""
    program = Program(
        declarations=(
            FunDecl(
                function_declaration=Function(
                    name="main",
                    params=(),
                    body=(Return(exp=Constant(value=2)),),
                    fun_type=FunType(params=(), ret=IntType()),
                ),
            ),
        )
    )
    result = pretty_print(program)
    expected = (
        "Program(\n"
        "    declarations=(\n"
        "        FunDecl(\n"
        "            function_declaration=Function(\n"
        '                name="main"\n'
        "                params=()\n"
        "                body=(\n"
        "                    Return(exp=Constant(value=2))\n"
        "                )\n"
        "                fun_type=FunType(params=(), ret=IntType())\n"
        "                globl=False\n"
        "                storage_class=None\n"
        '                source_file=""\n'
        "                source_line=0\n"
        "            )\n"
        "        )\n"
        "    )\n"
        ")"
    )
    assert result == expected


def test_empty_params_tuple_inline() -> None:
    """An empty tuple renders as ``()`` on the same line as the field name."""
    program = Program(
        declarations=(
            FunDecl(
                function_declaration=Function(
                    name="main",
                    params=(),
                    body=(Return(exp=Constant(value=0)),),
                    fun_type=FunType(params=(), ret=IntType()),
                ),
            ),
        )
    )
    assert "params=()" in pretty_print(program)


def test_single_param_inline() -> None:
    """A single ``Parameter`` inside a params tuple renders inline."""
    program = Program(
        declarations=(
            FunDecl(
                function_declaration=Function(
                    name="foo",
                    params=(Parameter(param_type=IntType(), name="n"),),
                    body=(Return(exp=Constant(value=0)),),
                    fun_type=FunType(params=(IntType(),), ret=IntType()),
                ),
            ),
        )
    )
    result = pretty_print(program)
    assert 'Parameter(param_type=IntType(), name="n")' in result


def test_multiple_functions_each_indented() -> None:
    """Multiple functions in a program each appear correctly indented."""

    def _fn(name: str) -> FunDecl:
        return FunDecl(
            function_declaration=Function(
                name=name,
                params=(),
                body=(Return(exp=Constant(value=0)),),
                fun_type=FunType(params=(), ret=IntType()),
            )
        )

    program = Program(declarations=(_fn("foo"), _fn("bar")))
    result = pretty_print(program)
    assert '"foo"' in result
    assert '"bar"' in result
    # Both Function nodes start at the same indent level
    lines = result.splitlines()
    fn_lines = [ln for ln in lines if "function_declaration=Function(" in ln]
    assert len(fn_lines) == 2
    assert fn_lines[0].index("Function(") == fn_lines[1].index("Function(")


def test_multiple_statements_in_body() -> None:
    """Multiple statements in a body all appear in the rendered output."""
    program = Program(
        declarations=(
            FunDecl(
                function_declaration=Function(
                    name="main",
                    params=(),
                    body=(
                        Return(exp=Constant(value=1)),
                        Return(exp=Constant(value=2)),
                    ),
                    fun_type=FunType(params=(), ret=IntType()),
                ),
            ),
        )
    )
    result = pretty_print(program)
    assert result.count("Return(") == 2
    assert result.count("Constant(value=1)") == 1
    assert result.count("Constant(value=2)") == 1


# ---------------------------------------------------------------------------
# Loop nodes
# ---------------------------------------------------------------------------


def test_break_renders_inline() -> None:
    """``Break`` with an empty identifier renders on a single line."""
    assert pretty_print(Break()) == 'Break(identifier="")'


def test_break_with_identifier_renders_inline() -> None:
    """``Break`` with a populated identifier renders on a single line."""
    assert pretty_print(Break(identifier="loop_0")) == 'Break(identifier="loop_0")'


def test_continue_renders_inline() -> None:
    """``Continue`` with an empty identifier renders on a single line."""
    assert pretty_print(Continue()) == 'Continue(identifier="")'


def test_continue_with_identifier_renders_inline() -> None:
    """``Continue`` with a populated identifier renders on a single line."""
    assert pretty_print(Continue(identifier="loop_0")) == 'Continue(identifier="loop_0")'


def test_while_with_simple_body_renders_inline() -> None:
    """``While`` whose condition and body both render inline is itself inline."""
    node = While(condition=Constant(value=1), body=Null())
    assert pretty_print(node) == 'While(condition=Constant(value=1), body=Null(), identifier="")'


def test_while_identifier_appears_in_output() -> None:
    """A labelled ``While`` node includes its identifier in the rendered output."""
    node = While(condition=Constant(value=1), body=Null(), identifier="loop_0")
    assert 'identifier="loop_0"' in pretty_print(node)


def test_dowhile_with_simple_body_renders_inline() -> None:
    """``DoWhile`` whose body and condition both render inline is itself inline."""
    node = DoWhile(body=Null(), condition=Constant(value=1))
    assert pretty_print(node) == 'DoWhile(body=Null(), condition=Constant(value=1), identifier="")'


def test_dowhile_identifier_appears_in_output() -> None:
    """A labelled ``DoWhile`` node includes its identifier in the rendered output."""
    node = DoWhile(body=Null(), condition=Constant(value=1), identifier="loop_1")
    assert 'identifier="loop_1"' in pretty_print(node)


def test_for_with_simple_fields_renders_inline() -> None:
    """``For`` whose fields all render inline is itself inline."""
    node = For(init=InitExp(exp=None), condition=None, post=None, body=Null())
    assert pretty_print(node) == ('For(init=InitExp(exp=None), condition=None, post=None, body=Null(), identifier="")')


def test_for_identifier_appears_in_output() -> None:
    """A labelled ``For`` node includes its identifier in the rendered output."""
    node = For(init=InitExp(exp=None), condition=None, post=None, body=Null(), identifier="loop_2")
    assert 'identifier="loop_2"' in pretty_print(node)


# ---------------------------------------------------------------------------
# Function forward declarations (body=None)
# ---------------------------------------------------------------------------


def test_function_with_none_body_renders_inline() -> None:
    """A forward-declaration ``Function`` with ``body=None`` renders inline."""
    node = Function(name="foo", params=(), body=None, fun_type=FunType(params=(), ret=IntType()))
    assert pretty_print(node) == (
        'Function(name="foo", params=(), body=None, '
        'fun_type=FunType(params=(), ret=IntType()), globl=False, storage_class=None, source_file="", source_line=0)'
    )


def test_function_none_body_in_program() -> None:
    """A ``Program`` containing a forward-declaration function renders ``body=None``."""
    program = Program(
        declarations=(
            FunDecl(
                function_declaration=Function(
                    name="foo", params=(), body=None, fun_type=FunType(params=(), ret=IntType())
                )
            ),
        )
    )
    result = pretty_print(program)
    assert "body=None" in result
