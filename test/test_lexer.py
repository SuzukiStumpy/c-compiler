"""Tests for the lexer() function — external behaviour only.

All tests preprocess real .c fixture files via GCC so that linemarker handling
is exercised against actual preprocessor output.
"""

from pathlib import Path
from typing import Callable

import pytest

from compiler.lexer import lexer
from compiler.types.tokens import TokenType
from compiler.utils.exceptions import LexerError

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Type alias for the preprocess fixture
Preprocessor = Callable[[str], Path]


# ---------------------------------------------------------------------------
# Token types and values
# ---------------------------------------------------------------------------


def test_minimal_program_produces_correct_token_types(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/minimal.c"))
    assert [t.type for t in tokens] == [
        TokenType.INT,
        TokenType.IDENTIFIER,
        TokenType.LPAREN,
        TokenType.VOID,
        TokenType.RPAREN,
        TokenType.LBRACE,
        TokenType.RETURN,
        TokenType.NUMBER,
        TokenType.SEMICOLON,
        TokenType.RBRACE,
    ]


def test_minimal_program_produces_correct_token_values(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/minimal.c"))
    assert [t.value for t in tokens] == ["int", "main", "(", "void", ")", "{", "return", "2", ";", "}"]


def test_keywords_produce_distinct_types_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/minimal.c"))
    by_value = {t.value: t.type for t in tokens}
    assert by_value["int"] == TokenType.INT
    assert by_value["void"] == TokenType.VOID
    assert by_value["return"] == TokenType.RETURN


def test_identifier_sharing_keyword_prefix_is_not_a_keyword(preprocess_c_file: Preprocessor) -> None:
    # "returning" starts with the keyword "return" — must not be classified as RETURN
    tokens = lexer(preprocess_c_file("valid/keyword_prefix.c"))
    returning_token = next(t for t in tokens if t.value == "returning")
    assert returning_token.type == TokenType.IDENTIFIER


# ---------------------------------------------------------------------------
# Source positions
# ---------------------------------------------------------------------------


def test_first_token_on_first_line_reports_line_1(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/minimal.c"))
    assert tokens[0].line == 1


def test_column_numbers_advance_correctly_across_first_line(preprocess_c_file: Preprocessor) -> None:
    # "int main(void) {"
    #  ^   ^   ^^   ^^
    #  1   5   910  14 16
    tokens = lexer(preprocess_c_file("valid/minimal.c"))
    line1 = [t for t in tokens if t.line == 1]
    assert line1[0].col == 1   # int
    assert line1[1].col == 5   # main
    assert line1[2].col == 9   # (
    assert line1[3].col == 10  # void
    assert line1[4].col == 14  # )
    assert line1[5].col == 16  # {


def test_line_numbers_increment_correctly_across_newlines(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/minimal.c"))
    return_token = next(t for t in tokens if t.type == TokenType.RETURN)
    rbrace_token = next(t for t in tokens if t.type == TokenType.RBRACE)
    assert return_token.line == 2
    assert rbrace_token.line == 3


def test_column_numbers_on_second_line(preprocess_c_file: Preprocessor) -> None:
    # "    return 2;"
    #      ^      ^^
    #      5      12 13
    tokens = lexer(preprocess_c_file("valid/minimal.c"))
    line2 = [t for t in tokens if t.line == 2]
    assert line2[0].col == 5   # return
    assert line2[1].col == 12  # 2
    assert line2[2].col == 13  # ;


# ---------------------------------------------------------------------------
# Linemarker handling
# ---------------------------------------------------------------------------


def test_tokens_carry_original_c_filename_not_preprocessed_filename(preprocess_c_file: Preprocessor) -> None:
    src = (FIXTURES_DIR / "valid" / "minimal.c").resolve()
    tokens = lexer(preprocess_c_file("valid/minimal.c"))
    assert all(t.filename == str(src) for t in tokens)


def test_no_linemarker_tokens_in_output(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/minimal.c"))
    # Linemarker lines start with "#" — none should appear as token values
    assert not any(t.value.startswith("#") for t in tokens)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_file_with_only_whitespace_returns_empty_token_list(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/whitespace_only.c"))
    assert tokens == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_character_raises_lexer_error(preprocess_c_file: Preprocessor) -> None:
    with pytest.raises(LexerError):
        lexer(preprocess_c_file("invalid/bad_char.c"))


def test_lexer_error_identifies_the_offending_character(preprocess_c_file: Preprocessor) -> None:
    with pytest.raises(LexerError) as exc_info:
        lexer(preprocess_c_file("invalid/bad_char.c"))
    assert exc_info.value.token_value == "$"


def test_lexer_error_reports_original_c_filename(preprocess_c_file: Preprocessor) -> None:
    src = (FIXTURES_DIR / "invalid" / "bad_char.c").resolve()
    with pytest.raises(LexerError) as exc_info:
        lexer(preprocess_c_file("invalid/bad_char.c"))
    assert exc_info.value.filename == str(src)


def test_lexer_error_reports_correct_line_number(preprocess_c_file: Preprocessor) -> None:
    # "$" appears on line 2 of bad_char.c
    with pytest.raises(LexerError) as exc_info:
        lexer(preprocess_c_file("invalid/bad_char.c"))
    assert exc_info.value.line == 2


def test_lexer_error_reports_correct_column_number(preprocess_c_file: Preprocessor) -> None:
    # "    return $2;" — "$" is at column 12
    with pytest.raises(LexerError) as exc_info:
        lexer(preprocess_c_file("invalid/bad_char.c"))
    assert exc_info.value.col == 12
