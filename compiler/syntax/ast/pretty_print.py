"""Pretty-printer for AST nodes.

Converts any AST node into a human-readable, indented string suitable for
console output.  Nodes whose fields are all scalar values (``int``, ``str``,
or empty tuples) are rendered on a single line; all others expand into
indented multi-line form.

Typical output for ``int main(void) { return 2; }``::

    Program(
        functions=(
            Function(
                return_type=IntType()
                name="main"
                params=()
                body=(
                    Return(
                        exp=Constant(value=2)
                    )
                )
            )
        )
    )
"""

import dataclasses

_INDENT = "    "


def pretty_print(node: object) -> str:
    """Render an AST node as a human-readable indented string.

    Produces a tree-shaped text representation of *node* and all of its
    children.  Leaf nodes (those whose fields contain only scalar values) are
    collapsed onto a single line; container nodes expand into indented
    multi-line form.

    Args:
        node: Any AST node or primitive value to render.

    Returns:
        A multi-line string representation of the node, ready for printing.
    """
    return _render(node, 0)


def _render(value: object, depth: int) -> str:
    """Recursively render a value at the given indentation depth.

    The returned string never starts with leading whitespace — the caller is
    responsible for prepending the appropriate indent to the first line.
    Subsequent lines, however, carry their own absolute indentation so that
    the whole string can be embedded directly into a parent render.

    Args:
        value: The value to render.
        depth: Current nesting depth, used to compute indent widths.

    Returns:
        A string representation of the value, possibly spanning multiple lines.
    """
    pad = _INDENT * depth
    inner = _INDENT * (depth + 1)

    if isinstance(value, int):
        return str(value)

    if isinstance(value, str):
        return f'"{value}"'

    if isinstance(value, tuple):
        if not value:
            return "()"
        lines = ["("]
        for item in value:
            lines.append(f"{inner}{_render(item, depth + 1)}")
        lines.append(f"{pad})")
        return "\n".join(lines)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls_name = type(value).__name__
        fields = dataclasses.fields(value)
        if not fields:
            return f"{cls_name}()"
        field_renders = [(f.name, _render(getattr(value, f.name), depth + 1)) for f in fields]
        if all("\n" not in rendered for _, rendered in field_renders):
            parts = ", ".join(f"{fname}={rendered}" for fname, rendered in field_renders)
            return f"{cls_name}({parts})"
        lines = [f"{cls_name}("]
        for fname, rendered in field_renders:
            lines.append(f"{inner}{fname}={rendered}")
        lines.append(f"{pad})")
        return "\n".join(lines)

    return repr(value)
