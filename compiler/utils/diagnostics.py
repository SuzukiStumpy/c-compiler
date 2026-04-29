"""Terminal formatting for compiler diagnostics."""

import click

from compiler.utils.exceptions import CodeGenerationError, LexerError, ParseError


def format_for_terminal(error: LexerError | ParseError | CodeGenerationError) -> str:
    """Format a compiler diagnostic for display in a colour-capable terminal.

    Applies ANSI colour codes via ``click.style``: the error label and
    offending token are rendered in bold red, the source location in bold
    white, and gutter line numbers in bold. ``click`` automatically strips
    colour codes when the output is not a TTY.

    ``LexerError`` and ``ParseError`` produce a multi-line output with a
    location header, ±2 lines of source context in a numbered gutter, and a
    ``^~~~~`` underline beneath the offending token.  When source lines are
    unavailable the output degrades to the header line only.

    ``CodeGenerationError`` has no source position, so only the bold-red
    label and message are emitted.

    Args:
        error: The diagnostic to format.

    Returns:
        A string with ANSI colour codes suitable for passing to ``click.echo``.
    """
    if isinstance(error, CodeGenerationError):
        label = click.style("CodeGenerationError", fg="red", bold=True)
        return f"{label}: {click.style(error.message, bold=True)}"

    return _format_source_diagnostic(error)


def _format_source_diagnostic(error: LexerError | ParseError) -> str:
    """Format a source-positioned diagnostic with gutter and underline.

    Used for both ``LexerError`` and ``ParseError``, which share identical
    fields (``filename``, ``line``, ``col``, ``token_value``, ``source_lines``,
    ``message``).

    Args:
        error: A ``LexerError`` or ``ParseError`` to format.

    Returns:
        A multi-line string with ANSI colour codes, or a single header line
        if ``error.source_lines`` is empty.
    """
    label = click.style(type(error).__name__, fg="red", bold=True)
    location = click.style(f"{error.filename}:{error.line}:{error.col}", bold=True)
    message = click.style(error.message, bold=True)
    header = f"{label} at {location}: {message}"

    if not error.source_lines:
        return header

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
