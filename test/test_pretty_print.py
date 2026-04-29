"""Tests for the AST pretty-printer.

Verifies that ``pretty_print`` produces correctly indented, human-readable
output for the full range of AST node shapes.
"""

from compiler.syntax.ast import Constant, Function, IntType, Parameter, Program, Return, pretty_print


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
        functions=(
            Function(
                return_type=IntType(),
                name="main",
                params=(),
                body=(Return(exp=Constant(value=2)),),
            ),
        )
    )
    result = pretty_print(program)
    expected = (
        "Program(\n"
        "    functions=(\n"
        "        Function(\n"
        "            return_type=IntType()\n"
        '            name="main"\n'
        "            params=()\n"
        "            body=(\n"
        "                Return(exp=Constant(value=2))\n"
        "            )\n"
        "        )\n"
        "    )\n"
        ")"
    )
    assert result == expected


def test_empty_params_tuple_inline() -> None:
    """An empty tuple renders as ``()`` on the same line as the field name."""
    program = Program(
        functions=(
            Function(
                return_type=IntType(),
                name="main",
                params=(),
                body=(Return(exp=Constant(value=0)),),
            ),
        )
    )
    assert "params=()" in pretty_print(program)


def test_single_param_inline() -> None:
    """A single ``Parameter`` inside a params tuple renders inline."""
    program = Program(
        functions=(
            Function(
                return_type=IntType(),
                name="foo",
                params=(Parameter(param_type=IntType(), name="n"),),
                body=(Return(exp=Constant(value=0)),),
            ),
        )
    )
    result = pretty_print(program)
    assert 'Parameter(param_type=IntType(), name="n")' in result


def test_multiple_functions_each_indented() -> None:
    """Multiple functions in a program each appear correctly indented."""

    def _fn(name: str) -> Function:
        return Function(
            return_type=IntType(),
            name=name,
            params=(),
            body=(Return(exp=Constant(value=0)),),
        )

    program = Program(functions=(_fn("foo"), _fn("bar")))
    result = pretty_print(program)
    assert '"foo"' in result
    assert '"bar"' in result
    # Both functions start at the same indent level
    lines = result.splitlines()
    fn_lines = [ln for ln in lines if "Function(" in ln]
    assert len(fn_lines) == 2
    assert fn_lines[0].index("Function(") == fn_lines[1].index("Function(")


def test_multiple_statements_in_body() -> None:
    """Multiple statements in a body all appear in the rendered output."""
    program = Program(
        functions=(
            Function(
                return_type=IntType(),
                name="main",
                params=(),
                body=(
                    Return(exp=Constant(value=1)),
                    Return(exp=Constant(value=2)),
                ),
            ),
        )
    )
    result = pretty_print(program)
    assert result.count("Return(") == 2
    assert result.count("Constant(value=1)") == 1
    assert result.count("Constant(value=2)") == 1
