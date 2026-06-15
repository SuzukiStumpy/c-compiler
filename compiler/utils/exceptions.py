"""Custom exceptions raised by the compiler pipeline."""

from subprocess import CalledProcessError


class SubprocessError(CalledProcessError):
    """A ``CalledProcessError`` that includes subprocess output in its message.

    Wraps the standard library ``CalledProcessError`` to ensure that any
    captured stdout or stderr from a failed subprocess is visible in the
    exception message, aiding diagnosis of GCC failures.
    """

    def __str__(self) -> str:
        """Return the error message with captured subprocess output appended.

        Returns:
            The base ``CalledProcessError`` message followed by the captured
            stdout or stderr, stripped of leading and trailing whitespace.
            If no output was captured, returns the base message unchanged.
        """
        base = super().__str__()
        output = self.output or self.stderr
        return f"{base}\n{output.strip()}" if output else base


class LexerError(Exception):
    """Raised when the lexer encounters input that cannot be tokenised.

    Carries structured information about the error location and surrounding
    source context so that callers can produce rich diagnostic output.

    Attributes:
        message: Human-readable description of what went wrong.
        filename: Path to the original C source file where the error occurred.
        line: 1-indexed line number in the original C source file.
        col: 1-indexed column number at which the invalid character begins.
        token_value: The text of the offending character or token.
        source_lines: Lines of the source file used to display error context.
    """

    def __init__(
        self,
        message: str,
        filename: str,
        line: int,
        col: int,
        token_value: str,
        source_lines: list[str],
    ) -> None:
        """Initialise a LexerError with full location and context information.

        Args:
            message: Human-readable description of what went wrong.
            filename: Path to the original C source file.
            line: 1-indexed line number in the original C source file.
            col: 1-indexed column number at which the invalid character begins.
            token_value: The text of the offending character or token.
            source_lines: Lines of the source file for context display.
        """
        self.message = message
        self.filename = filename
        self.line = line
        self.col = col
        self.token_value = token_value
        self.source_lines = source_lines
        super().__init__(message)

    def __str__(self) -> str:
        """Format the error as plain text with source context.

        Produces a multi-line string containing the error location header,
        +/-2 lines of source context in a numbered gutter, and a clang-style
        ``^~~~~`` underline beneath the offending token. Suitable for
        non-terminal output and test assertions.

        Returns:
            A formatted error string with location header, gutter, context
            lines, and underline. ANSI colour codes are never included.
        """
        location = f"{self.filename}:{self.line}:{self.col}"
        header = f"LexerError at {location}: {self.message}"

        error_idx = max(0, self.line - 1)
        start = max(0, error_idx - 2)
        end = min(len(self.source_lines), error_idx + 3)

        max_width = len(str(end))
        context: list[str] = []
        for i in range(start, end):
            line_num = str(i + 1).rjust(max_width)
            context.append(f"{line_num} | {self.source_lines[i]}")
            if i == error_idx:
                prefix = " " * max_width + " | " + " " * (self.col - 1)
                underline = "^" + "~" * max(0, len(self.token_value) - 1)
                context.append(prefix + underline)

        return header + "\n\n" + "\n".join(context)


class CodeGenerationError(Exception):
    """Raised when the code generator encounters an AST construct it cannot translate.

    This should not occur for any AST produced by the parser from a valid
    grammar, but guards against future grammar extensions that outpace the
    code generator.

    Attributes:
        message: Human-readable description of what went wrong.
    """

    def __init__(self, message: str) -> None:
        """Initialise a CodeGenerationError with a descriptive message.

        Args:
            message: Human-readable description of the unsupported construct.
        """
        self.message = message
        super().__init__(message)


class SemanticError(Exception):
    """Raised when the semantic analyser encounters a violation it cannot resolve.

    Covers errors that are grammatically valid but semantically illegal, such as
    duplicate variable declarations within the same scope or references to
    undeclared identifiers.

    Attributes:
        message: Human-readable description of what went wrong.
    """

    def __init__(self, message: str) -> None:
        """Initialise a SemanticError with a descriptive message.

        Args:
            message: Human-readable description of the semantic violation.
        """
        self.message = message
        super().__init__(message)


class ParseError(Exception):
    """Raised when the parser encounters an unexpected token or end of input.

    Carries the source position of the offending token and the surrounding
    source lines so that callers can produce a rich diagnostic pointing at the
    exact location in the C source file - formatted the same way as
    ``LexerError``.

    Attributes:
        message: Human-readable description of what went wrong.
        filename: Path to the original C source file where the error occurred.
        line: 1-indexed line number in the original C source file.
        col: 1-indexed column number at which the offending token begins.
        token_value: The exact source text of the offending token, or ``''``
            when the error is triggered by unexpected end of input.
        source_lines: Lines of the source file used to display error context.
            May be empty if the file cannot be read, in which case only the
            header line is emitted.
    """

    def __init__(
        self,
        message: str,
        filename: str,
        line: int,
        col: int,
        token_value: str,
        source_lines: list[str],
    ) -> None:
        """Initialise a ParseError with full location and context information.

        Args:
            message: Human-readable description of what went wrong.
            filename: Path to the original C source file.
            line: 1-indexed line number in the original C source file.
            col: 1-indexed column number at which the offending token begins.
            token_value: The exact source text of the offending token, or
                ``''`` for an unexpected end-of-input error.
            source_lines: Lines of the source file for context display.
        """
        self.message = message
        self.filename = filename
        self.line = line
        self.col = col
        self.token_value = token_value
        self.source_lines = source_lines
        super().__init__(message)

    def __str__(self) -> str:
        """Format the error as plain text with source context.

        Produces the same structure as ``LexerError.__str__``: an error
        location header, +/-2 lines of source context in a numbered gutter, and
        a clang-style ``^~~~~`` underline beneath the offending token.  When
        ``source_lines`` is empty (e.g. the source file could not be read),
        only the header line is returned.

        Returns:
            A formatted error string with location header and, when source
            lines are available, gutter, context lines, and underline. ANSI
            colour codes are never included.
        """
        location = f"{self.filename}:{self.line}:{self.col}"
        header = f"ParseError at {location}: {self.message}"

        if not self.source_lines:
            return header

        error_idx = max(0, self.line - 1)
        start = max(0, error_idx - 2)
        end = min(len(self.source_lines), error_idx + 3)

        max_width = len(str(end))
        context: list[str] = []
        for i in range(start, end):
            line_num = str(i + 1).rjust(max_width)
            context.append(f"{line_num} | {self.source_lines[i]}")
            if i == error_idx:
                prefix = " " * max_width + " | " + " " * (self.col - 1)
                underline = "^" + "~" * max(0, len(self.token_value) - 1)
                context.append(prefix + underline)

        return header + "\n\n" + "\n".join(context)
