"""Terminal formatting for compiler diagnostics."""

import click

from compiler.utils.exceptions import LexerError


def format_for_terminal(error: LexerError) -> str:
    """Format a ``LexerError`` for display in a colour-capable terminal.

    Produces the same structure as ``LexerError.__str__`` but with ANSI colour
    codes applied via ``click.style``: the error label and offending token are
    rendered in bold red, the source location in bold white, and gutter line
    numbers in bold. ``click`` automatically strips colour codes when the output
    is not a TTY.

    Args:
        error: The lexer error to format.

    Returns:
        A multi-line string with ANSI colour codes suitable for passing to
        ``click.echo``. Structure matches the plain-text format from
        ``LexerError.__str__``.
    """
    location = click.style(f"{error.filename}:{error.line}:{error.col}", bold=True)
    label = click.style("LexerError", fg="red", bold=True)
    message = click.style(error.message, bold=True)
    header = f"{label} at {location}: {message}"

    error_idx = max(0, error.line - 1)
    start = max(0, error_idx - 2)
    end = min(len(error.source_lines), error_idx + 3)

    max_width = len(str(end))
    context: list[str] = []
    for i in range(start, end):
        line_num = str(i + 1).rjust(max_width)
        gutter = click.style(f"{line_num} | ", bold=True)

        if i == error_idx:
            source_line = error.source_lines[i]
            token_start = error.col - 1
            token_end = min(token_start + len(error.token_value), len(source_line))
            before = source_line[:token_start]
            highlighted = click.style(source_line[token_start:token_end], fg="red", bold=True)
            after = source_line[token_end:]
            context.append(gutter + before + highlighted + after)

            prefix = " " * max_width + " | " + " " * (error.col - 1)
            underline = click.style("^" + "~" * max(0, len(error.token_value) - 1), fg="red", bold=True)
            context.append(prefix + underline)
        else:
            context.append(gutter + error.source_lines[i])

    return header + "\n\n" + "\n".join(context)
