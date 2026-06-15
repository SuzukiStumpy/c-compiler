"""Tests for LexerError.__str__() - the plain-text error formatting.

Tests construct LexerError instances directly so they are independent of the
lexer algorithm. Only external behaviour (the formatted string) is asserted.
"""

import re

from compiler.utils.exceptions import LexerError


def _make_error(
    line: int,
    col: int,
    token_value: str,
    source_lines: list[str],
    filename: str = "test.c",
) -> LexerError:
    return LexerError(
        message=f"unexpected character {token_value!r}",
        filename=filename,
        line=line,
        col=col,
        token_value=token_value,
        source_lines=source_lines,
    )


def _source_display_lines(output: str) -> list[str]:
    """Lines from the formatted output that show numbered source lines (not the underline)."""
    return [line for line in output.split("\n") if re.match(r"\s*\d+\s*\|", line)]


def _underline_line(output: str) -> str:
    """The line containing the caret from the formatted output."""
    for line in output.split("\n"):
        if "^" in line:
            return line
    raise AssertionError("No caret line found in error output")


def _caret_col_offset(output: str) -> int:
    """Zero-based offset of '^' from the start of the source content (after the gutter)."""
    line = _underline_line(output)
    gutter_end = line.index("|") + 2
    return line.index("^") - gutter_end


# ---------------------------------------------------------------------------
# Header content
# ---------------------------------------------------------------------------


def test_header_includes_filename_line_and_column() -> None:
    err = _make_error(2, 5, "$", ["int x;", "int $y;", "int z;"], filename="src.c")
    assert "src.c:2:5" in str(err)


def test_header_includes_error_message() -> None:
    err = _make_error(1, 1, "$", ["$"])
    assert "unexpected character '$'" in str(err)


# ---------------------------------------------------------------------------
# Context line count
# ---------------------------------------------------------------------------


def test_error_in_middle_of_file_shows_two_lines_above_and_below() -> None:
    source = [f"line {i}" for i in range(1, 8)]  # 7 lines
    err = _make_error(4, 1, "$", source)
    displayed = _source_display_lines(str(err))
    # Should show lines 2, 3, 4, 5, 6 - five source lines
    assert len(displayed) == 5


def test_error_on_line_1_shows_no_lines_above() -> None:
    source = ["$line", "line 2", "line 3", "line 4"]
    err = _make_error(1, 1, "$", source)
    output = str(err)
    displayed = _source_display_lines(output)
    # Lines 1, 2, 3 - three lines (no line 0)
    assert len(displayed) == 3
    assert "0 |" not in output


def test_error_on_last_line_shows_no_lines_below() -> None:
    source = ["line 1", "line 2", "line 3", "$end"]
    err = _make_error(4, 1, "$", source)
    output = str(err)
    displayed = _source_display_lines(output)
    # Lines 2, 3, 4 - three lines (no line 5)
    assert len(displayed) == 3
    assert "5 |" not in output


def test_error_near_start_of_file_shows_only_available_lines() -> None:
    source = ["$"]
    err = _make_error(1, 1, "$", source)
    displayed = _source_display_lines(str(err))
    assert len(displayed) == 1


def test_error_on_line_2_of_short_file_shows_one_line_above() -> None:
    source = ["line 1", "line $2"]
    err = _make_error(2, 8, "$", source)
    displayed = _source_display_lines(str(err))
    # Lines 1 and 2 - only one line above, no lines below (EOF)
    assert len(displayed) == 2


# ---------------------------------------------------------------------------
# Caret position and span
# ---------------------------------------------------------------------------


def test_caret_appears_at_correct_column_for_col_1() -> None:
    err = _make_error(1, 1, "$", ["$rest"])
    assert _caret_col_offset(str(err)) == 0


def test_caret_appears_at_correct_column_for_mid_line_error() -> None:
    # "    return $2;" - col 12 means 11 spaces before "^"
    source = ["int main(void) {", "    return $2;", "}"]
    err = _make_error(2, 12, "$", source)
    assert _caret_col_offset(str(err)) == 11


def test_single_character_token_produces_caret_only() -> None:
    err = _make_error(1, 1, "$", ["$"])
    underline = _underline_line(str(err))
    assert "^" in underline
    assert "~" not in underline


def test_multi_character_token_produces_correct_number_of_tildes() -> None:
    # 4-character token: "^~~~" (1 caret + 3 tildes)
    err = _make_error(1, 1, "$$$$", ["$$$$"])
    underline = _underline_line(str(err))
    caret_section = underline[underline.index("^") :]
    assert caret_section == "^~~~"


# ---------------------------------------------------------------------------
# Gutter alignment
# ---------------------------------------------------------------------------


def test_pipe_characters_are_vertically_aligned() -> None:
    # Error on line 10 of an 11-line file forces 2-digit line numbers.
    # The underline gutter must match the source-line gutter width.
    source = [f"line {i}" for i in range(1, 12)]
    err = _make_error(10, 1, "$", source)
    output = str(err)
    lines_with_pipe = [line for line in output.split("\n") if "|" in line]
    pipe_positions = {line.index("|") for line in lines_with_pipe}
    assert len(pipe_positions) == 1, f"Pipe characters not aligned: positions {pipe_positions}"


def test_single_digit_line_numbers_are_right_padded_when_context_exceeds_9_lines() -> None:
    source = [f"line {i}" for i in range(1, 12)]
    err = _make_error(10, 1, "$", source)
    output = str(err)
    # Line 8 and 9 are in context; they must be padded to 2 digits
    assert " 8 |" in output
    assert " 9 |" in output
    assert "10 |" in output
