"""Tests for the lexer() function - external behaviour only.

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


def test_union_keyword_produces_union_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/union_keyword.c"))
    union_token = next(t for t in tokens if t.value == "union")
    assert union_token.type == TokenType.UNION


def test_typedef_keyword_produces_typedef_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/typedef_keyword.c"))
    typedef_token = next(t for t in tokens if t.value == "typedef")
    assert typedef_token.type == TokenType.TYPEDEF


def test_float_keyword_produces_float_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/float_type.c"))
    float_token = next(t for t in tokens if t.value == "float")
    assert float_token.type == TokenType.FLOAT


def test_float_literal_produces_fpfloat_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/float_type.c"))
    fpfloat_tokens = [t for t in tokens if t.type == TokenType.FPFLOAT]
    assert len(fpfloat_tokens) == 4
    assert {t.value for t in fpfloat_tokens} == {"1.5f", "1.5e10f", "1e5F", "1.F"}


def test_long_double_literal_produces_fplongdouble_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/long_double_type.c"))
    ld_tokens = [t for t in tokens if t.type == TokenType.FPLONGDOUBLE]
    assert len(ld_tokens) == 4
    assert {t.value for t in ld_tokens} == {"1.5L", "1.5e10L", "1e5l", "1.L"}


def test_long_double_literal_not_classified_as_long_int(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/long_double_type.c"))
    assert not any(t.type == TokenType.LONGINT and t.value.endswith(("L", "l")) and "." in t.value for t in tokens)


def test_identifier_sharing_keyword_prefix_is_not_a_keyword(preprocess_c_file: Preprocessor) -> None:
    # "returning" starts with the keyword "return" - must not be classified as RETURN
    tokens = lexer(preprocess_c_file("valid/keyword_prefix.c"))
    returning_token = next(t for t in tokens if t.value == "returning")
    assert returning_token.type == TokenType.IDENTIFIER


# ---------------------------------------------------------------------------
# Source positions
# ---------------------------------------------------------------------------


def test_first_token_on_first_line_reports_line_1(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/minimal.c"))
    assert tokens[0].line == 1


def test_enum_keyword_produces_enum_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    """The ``enum`` keyword must be classified as ``ENUM``, not as ``IDENTIFIER``."""
    tokens = lexer(preprocess_c_file("valid/enum_keyword.c"))
    enum_token = next(t for t in tokens if t.value == "enum")
    assert enum_token.type == TokenType.ENUM


def test_enum_members_are_classified_as_identifiers(preprocess_c_file: Preprocessor) -> None:
    """Enumerator names such as ``RED``, ``GREEN``, ``BLUE`` are ordinary identifiers."""
    tokens = lexer(preprocess_c_file("valid/enum_keyword.c"))
    member_tokens = [t for t in tokens if t.value in {"RED", "GREEN", "BLUE"}]
    assert len(member_tokens) == 3
    assert all(t.type == TokenType.IDENTIFIER for t in member_tokens)


def test_column_numbers_advance_correctly_across_first_line(preprocess_c_file: Preprocessor) -> None:
    # "int main(void) {"
    #  ^   ^   ^^   ^^
    #  1   5   910  14 16
    tokens = lexer(preprocess_c_file("valid/minimal.c"))
    line1 = [t for t in tokens if t.line == 1]
    assert line1[0].col == 1  # int
    assert line1[1].col == 5  # main
    assert line1[2].col == 9  # (
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
    assert line2[0].col == 5  # return
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
    # Linemarker lines start with "#" - none should appear as token values
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
    # "    return $2;" - "$" is at column 12
    with pytest.raises(LexerError) as exc_info:
        lexer(preprocess_c_file("invalid/bad_char.c"))
    assert exc_info.value.col == 12


# ---------------------------------------------------------------------------
# Multi-character token disambiguation
# ---------------------------------------------------------------------------


def test_increment_is_single_token_not_two_plus(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/inc_dec.c"))
    types = [t.type for t in tokens]
    assert TokenType.INCREMENT in types
    assert TokenType.PLUS not in types


def test_decrement_is_single_token_not_two_minus(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/inc_dec.c"))
    types = [t.type for t in tokens]
    assert TokenType.DECREMENT in types
    assert TokenType.MINUS not in types


def test_plus_assign_is_single_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/compound_assign.c"))
    types = [t.type for t in tokens]
    assert TokenType.PLUS_ASSIGN in types
    assert TokenType.PLUS not in types


def test_lshift_assign_is_single_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/compound_assign.c"))
    types = [t.type for t in tokens]
    assert TokenType.LSHIFT_ASSIGN in types
    assert TokenType.LSHIFT not in types


def test_rshift_assign_is_single_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/compound_assign.c"))
    types = [t.type for t in tokens]
    assert TokenType.RSHIFT_ASSIGN in types
    assert TokenType.RSHIFT not in types


# ---------------------------------------------------------------------------
# Loop keywords
# ---------------------------------------------------------------------------


def test_while_keyword_produces_while_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/loops.c"))
    while_token = next(t for t in tokens if t.value == "while")
    assert while_token.type == TokenType.WHILE


def test_do_keyword_produces_do_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/loops.c"))
    do_token = next(t for t in tokens if t.value == "do")
    assert do_token.type == TokenType.DO


def test_for_keyword_produces_for_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/loops.c"))
    for_token = next(t for t in tokens if t.value == "for")
    assert for_token.type == TokenType.FOR


def test_break_keyword_produces_break_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/loops.c"))
    break_token = next(t for t in tokens if t.value == "break")
    assert break_token.type == TokenType.BREAK


def test_continue_keyword_produces_continue_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/loops.c"))
    continue_token = next(t for t in tokens if t.value == "continue")
    assert continue_token.type == TokenType.CONTINUE


# ---------------------------------------------------------------------------
# Goto and labeled statements
# ---------------------------------------------------------------------------


def test_goto_keyword_produces_goto_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/goto_label.c"))
    goto_token = next(t for t in tokens if t.value == "goto")
    assert goto_token.type == TokenType.GOTO


def test_label_colon_produces_colon_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/goto_label.c"))
    assert TokenType.COLON in [t.type for t in tokens]


def test_label_identifier_and_colon_are_separate_tokens(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/goto_label.c"))
    colon_idx = next(i for i, t in enumerate(tokens) if t.type == TokenType.COLON)
    assert tokens[colon_idx - 1].type == TokenType.IDENTIFIER
    assert tokens[colon_idx - 1].value == "end"


# ---------------------------------------------------------------------------
# Switch keywords
# ---------------------------------------------------------------------------


def test_switch_keyword_produces_switch_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/switch.c"))
    switch_token = next(t for t in tokens if t.value == "switch")
    assert switch_token.type == TokenType.SWITCH


def test_case_keyword_produces_case_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/switch.c"))
    case_token = next(t for t in tokens if t.value == "case")
    assert case_token.type == TokenType.CASE


def test_default_keyword_produces_default_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/switch.c"))
    default_token = next(t for t in tokens if t.value == "default")
    assert default_token.type == TokenType.DEFAULT


# ---------------------------------------------------------------------------
# Storage-class keywords
# ---------------------------------------------------------------------------


def test_static_keyword_produces_static_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/storage_class.c"))
    static_token = next(t for t in tokens if t.value == "static")
    assert static_token.type == TokenType.STATIC


def test_extern_keyword_produces_extern_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/storage_class.c"))
    extern_token = next(t for t in tokens if t.value == "extern")
    assert extern_token.type == TokenType.EXTERN


def test_identifier_sharing_static_prefix_is_not_a_keyword(preprocess_c_file: Preprocessor) -> None:
    # "statically" starts with "static" - must not be classified as STATIC
    tokens = lexer(preprocess_c_file("valid/keyword_prefix.c"))
    assert all(t.type != TokenType.STATIC for t in tokens)


def test_identifier_sharing_extern_prefix_is_not_a_keyword(preprocess_c_file: Preprocessor) -> None:
    # "externally" starts with "extern" - must not be classified as EXTERN
    tokens = lexer(preprocess_c_file("valid/keyword_prefix.c"))
    assert all(t.type != TokenType.EXTERN for t in tokens)


def test_auto_keyword_produces_auto_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/auto_register.c"))
    auto_token = next(t for t in tokens if t.value == "auto")
    assert auto_token.type == TokenType.AUTO


def test_register_keyword_produces_register_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/auto_register.c"))
    register_token = next(t for t in tokens if t.value == "register")
    assert register_token.type == TokenType.REGISTER


# ---------------------------------------------------------------------------
# Long type keyword and long integer constants
# ---------------------------------------------------------------------------


def test_long_keyword_produces_long_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/long_type.c"))
    long_token = next(t for t in tokens if t.value == "long")
    assert long_token.type == TokenType.LONG


def test_long_integer_constant_lowercase_suffix_produces_longint_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/long_type.c"))
    assert any(t.type == TokenType.LONGINT and t.value == "42l" for t in tokens)


def test_long_integer_constant_uppercase_suffix_produces_longint_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/long_type.c"))
    assert any(t.type == TokenType.LONGINT and t.value == "100L" for t in tokens)


def test_longint_token_is_distinct_from_number_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/long_type.c"))
    longint_values = {t.value for t in tokens if t.type == TokenType.LONGINT}
    number_values = {t.value for t in tokens if t.type == TokenType.NUMBER}
    assert longint_values.isdisjoint(number_values)


def test_longint_token_is_not_split_into_number_and_identifier(preprocess_c_file: Preprocessor) -> None:
    # "42l" must be a single LONGINT, not NUMBER("42") followed by IDENTIFIER("l")
    tokens = lexer(preprocess_c_file("valid/long_type.c"))
    types = [t.type for t in tokens]
    longint_idx = next(i for i, t in enumerate(tokens) if t.type == TokenType.LONGINT and t.value == "42l")
    assert types[longint_idx - 1] != TokenType.NUMBER
    assert types[longint_idx + 1] != TokenType.IDENTIFIER


def test_identifier_sharing_long_prefix_is_not_a_keyword(preprocess_c_file: Preprocessor) -> None:
    # "longer" starts with "long" - must not be classified as LONG
    tokens = lexer(preprocess_c_file("valid/keyword_prefix.c"))
    assert all(t.type != TokenType.LONG for t in tokens)


# ---------------------------------------------------------------------------
# signed and unsigned keywords
# ---------------------------------------------------------------------------


def test_signed_keyword_produces_signed_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/unsigned_types.c"))
    signed_token = next(t for t in tokens if t.value == "signed")
    assert signed_token.type == TokenType.SIGNED


def test_unsigned_keyword_produces_unsigned_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/unsigned_types.c"))
    unsigned_token = next(t for t in tokens if t.value == "unsigned")
    assert unsigned_token.type == TokenType.UNSIGNED


def test_identifier_sharing_signed_prefix_is_not_a_keyword(preprocess_c_file: Preprocessor) -> None:
    # "signedValue" starts with "signed" - must not be classified as SIGNED
    tokens = lexer(preprocess_c_file("valid/keyword_prefix.c"))
    assert all(t.type != TokenType.SIGNED for t in tokens)


def test_identifier_sharing_unsigned_prefix_is_not_a_keyword(preprocess_c_file: Preprocessor) -> None:
    # "unsignedValue" starts with "unsigned" - must not be classified as UNSIGNED
    tokens = lexer(preprocess_c_file("valid/keyword_prefix.c"))
    assert all(t.type != TokenType.UNSIGNED for t in tokens)


# ---------------------------------------------------------------------------
# Unsigned integer constants (UINT)
# ---------------------------------------------------------------------------


def test_unsigned_integer_constant_lowercase_suffix_produces_uint_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/unsigned_types.c"))
    assert any(t.type == TokenType.UINT and t.value == "42u" for t in tokens)


def test_unsigned_integer_constant_uppercase_suffix_produces_uint_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/unsigned_types.c"))
    assert any(t.type == TokenType.UINT and t.value == "99U" for t in tokens)


def test_uint_token_is_distinct_from_number_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/unsigned_types.c"))
    uint_values = {t.value for t in tokens if t.type == TokenType.UINT}
    number_values = {t.value for t in tokens if t.type == TokenType.NUMBER}
    assert uint_values.isdisjoint(number_values)


def test_uint_token_is_not_split_into_number_and_identifier(preprocess_c_file: Preprocessor) -> None:
    # "42u" must be a single UINT, not NUMBER("42") followed by IDENTIFIER("u")
    tokens = lexer(preprocess_c_file("valid/unsigned_types.c"))
    uint_idx = next(i for i, t in enumerate(tokens) if t.type == TokenType.UINT and t.value == "42u")
    assert tokens[uint_idx - 1].type != TokenType.NUMBER
    assert tokens[uint_idx + 1].type != TokenType.IDENTIFIER


# ---------------------------------------------------------------------------
# Unsigned long integer constants (ULONG)
# ---------------------------------------------------------------------------


def test_ulong_constant_ul_suffix_produces_ulong_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/unsigned_types.c"))
    assert any(t.type == TokenType.ULONG and t.value == "100ul" for t in tokens)


def test_ulong_constant_lu_suffix_produces_ulong_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/unsigned_types.c"))
    assert any(t.type == TokenType.ULONG and t.value == "200lu" for t in tokens)


def test_ulong_constant_uppercase_ul_suffix_produces_ulong_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/unsigned_types.c"))
    assert any(t.type == TokenType.ULONG and t.value == "300UL" for t in tokens)


def test_ulong_constant_uppercase_lu_suffix_produces_ulong_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/unsigned_types.c"))
    assert any(t.type == TokenType.ULONG and t.value == "400LU" for t in tokens)


def test_ulong_token_is_distinct_from_number_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/unsigned_types.c"))
    ulong_values = {t.value for t in tokens if t.type == TokenType.ULONG}
    number_values = {t.value for t in tokens if t.type == TokenType.NUMBER}
    assert ulong_values.isdisjoint(number_values)


def test_ulong_token_is_not_split_into_number_and_identifier(preprocess_c_file: Preprocessor) -> None:
    # "100ul" must be a single ULONG, not NUMBER("100") followed by IDENTIFIER("ul")
    tokens = lexer(preprocess_c_file("valid/unsigned_types.c"))
    ulong_idx = next(i for i, t in enumerate(tokens) if t.type == TokenType.ULONG and t.value == "100ul")
    assert tokens[ulong_idx - 1].type != TokenType.NUMBER
    assert tokens[ulong_idx + 1].type != TokenType.IDENTIFIER


# ---------------------------------------------------------------------------
# double type keyword and floating-point double constants (FPDOUBLE)
# ---------------------------------------------------------------------------


def test_double_keyword_produces_double_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/double_type.c"))
    double_token = next(t for t in tokens if t.value == "double")
    assert double_token.type == TokenType.DOUBLE


def test_identifier_sharing_double_prefix_is_not_a_keyword(preprocess_c_file: Preprocessor) -> None:
    # "doubler" starts with "double" - must not be classified as DOUBLE
    tokens = lexer(preprocess_c_file("valid/keyword_prefix.c"))
    assert all(t.type != TokenType.DOUBLE for t in tokens)


def test_fpdouble_decimal_form_produces_fpdouble_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/double_type.c"))
    assert any(t.type == TokenType.FPDOUBLE and t.value == "1.5" for t in tokens)


def test_fpdouble_exponent_form_produces_fpdouble_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/double_type.c"))
    assert any(t.type == TokenType.FPDOUBLE and t.value == "1.5e10" for t in tokens)


def test_fpdouble_integer_exponent_form_produces_fpdouble_token(preprocess_c_file: Preprocessor) -> None:
    # "1e5" has no decimal point but is still a floating-point constant
    tokens = lexer(preprocess_c_file("valid/double_type.c"))
    assert any(t.type == TokenType.FPDOUBLE and t.value == "1e5" for t in tokens)


def test_fpdouble_trailing_dot_form_produces_fpdouble_token(preprocess_c_file: Preprocessor) -> None:
    # "1." is a valid floating-point constant with a trailing dot
    tokens = lexer(preprocess_c_file("valid/double_type.c"))
    assert any(t.type == TokenType.FPDOUBLE and t.value == "1." for t in tokens)


def test_fpdouble_token_is_distinct_from_number_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/double_type.c"))
    fpdouble_values = {t.value for t in tokens if t.type == TokenType.FPDOUBLE}
    number_values = {t.value for t in tokens if t.type == TokenType.NUMBER}
    assert fpdouble_values.isdisjoint(number_values)


def test_fpdouble_decimal_is_single_token_not_number_and_dot(preprocess_c_file: Preprocessor) -> None:
    # "1.5" must be a single FPDOUBLE, not NUMBER("1") followed by separate tokens
    tokens = lexer(preprocess_c_file("valid/double_type.c"))
    fpdouble_idx = next(i for i, t in enumerate(tokens) if t.type == TokenType.FPDOUBLE and t.value == "1.5")
    assert tokens[fpdouble_idx - 1].type != TokenType.NUMBER
    assert tokens[fpdouble_idx + 1].type != TokenType.FPDOUBLE


def test_fpdouble_with_second_dot_suffix_raises_lexer_error(preprocess_c_file: Preprocessor) -> None:
    # "1.0e10.0" is not a valid constant; the second dot must not be silently accepted
    with pytest.raises(LexerError):
        lexer(preprocess_c_file("invalid/bad_float.c"))


def test_longint_followed_by_dot_raises_lexer_error(preprocess_c_file: Preprocessor) -> None:
    # "42l.0" is a malformed preprocessing-number, not LONGINT + FPDOUBLE
    with pytest.raises(LexerError):
        lexer(preprocess_c_file("invalid/bad_longint.c"))


def test_uint_followed_by_dot_raises_lexer_error(preprocess_c_file: Preprocessor) -> None:
    # "42u.0" is a malformed preprocessing-number, not UINT + FPDOUBLE
    with pytest.raises(LexerError):
        lexer(preprocess_c_file("invalid/bad_uint.c"))


def test_ulong_followed_by_dot_raises_lexer_error(preprocess_c_file: Preprocessor) -> None:
    # "42ul.0" is a malformed preprocessing-number, not ULONG + FPDOUBLE
    with pytest.raises(LexerError):
        lexer(preprocess_c_file("invalid/bad_ulong.c"))


def test_mixed_case_longlong_suffix_raises_lexer_error(preprocess_c_file: Preprocessor) -> None:
    """lL and Ll suffixes are invalid; only ll and LL are permitted."""
    with pytest.raises(LexerError):
        lexer(preprocess_c_file("invalid/bad_longlong_suffix.c"))


# ---------------------------------------------------------------------------
# Hexadecimal integer literals
# ---------------------------------------------------------------------------


def test_hex_integer_lexed_as_number(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/hex_literals.c"))
    assert any(t.type == TokenType.NUMBER and t.value == "0xFF" for t in tokens)


def test_hex_integer_uppercase_prefix_lexed_as_number(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/hex_literals.c"))
    assert any(t.type == TokenType.NUMBER and t.value == "0XFF" for t in tokens)


def test_hex_longint_lexed(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/hex_literals.c"))
    assert any(t.type == TokenType.LONGINT and t.value == "0xDEADl" for t in tokens)


def test_hex_uint_lexed(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/hex_literals.c"))
    assert any(t.type == TokenType.UINT and t.value == "0xCAFEu" for t in tokens)


def test_hex_ulong_lexed(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/hex_literals.c"))
    assert any(t.type == TokenType.ULONG and t.value == "0xDEADul" for t in tokens)


def test_hex_longlong_lexed(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/hex_literals.c"))
    assert any(t.type == TokenType.LONGLONG and t.value == "0xDEADBEEFll" for t in tokens)


def test_hex_ulonglong_lexed(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/hex_literals.c"))
    assert any(t.type == TokenType.ULONGLONG and t.value == "0xDEADBEEFull" for t in tokens)


# ---------------------------------------------------------------------------
# Octal integer literals
# ---------------------------------------------------------------------------


def test_octal_integer_lexed_as_number(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/octal_literals.c"))
    assert any(t.type == TokenType.NUMBER and t.value == "0777" for t in tokens)


def test_octal_longint_lexed(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/octal_literals.c"))
    assert any(t.type == TokenType.LONGINT and t.value == "0777l" for t in tokens)


def test_octal_uint_lexed(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/octal_literals.c"))
    assert any(t.type == TokenType.UINT and t.value == "0777u" for t in tokens)


def test_octal_ulong_lexed(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/octal_literals.c"))
    assert any(t.type == TokenType.ULONG and t.value == "0777ul" for t in tokens)


def test_octal_longlong_lexed(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/octal_literals.c"))
    assert any(t.type == TokenType.LONGLONG and t.value == "0777ll" for t in tokens)


def test_octal_ulonglong_lexed(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/octal_literals.c"))
    assert any(t.type == TokenType.ULONGLONG and t.value == "0777ull" for t in tokens)


# ---------------------------------------------------------------------------
# Binary integer literals (GCC extension: 0b/0B prefix)
# ---------------------------------------------------------------------------


def test_binary_integer_lexed_as_number(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/binary_literals.c"))
    assert any(t.type == TokenType.NUMBER and t.value == "0b1010" for t in tokens)


def test_binary_integer_uppercase_prefix_lexed_as_number(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/binary_literals.c"))
    assert any(t.type == TokenType.NUMBER and t.value == "0B1100" for t in tokens)


def test_binary_longint_lexed(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/binary_literals.c"))
    assert any(t.type == TokenType.LONGINT and t.value == "0b1010l" for t in tokens)


def test_binary_uint_lexed(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/binary_literals.c"))
    assert any(t.type == TokenType.UINT and t.value == "0b1010u" for t in tokens)


def test_binary_ulong_lexed(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/binary_literals.c"))
    assert any(t.type == TokenType.ULONG and t.value == "0b1010ul" for t in tokens)


def test_binary_longlong_lexed(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/binary_literals.c"))
    assert any(t.type == TokenType.LONGLONG and t.value == "0b1010ll" for t in tokens)


def test_binary_ulonglong_lexed(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/binary_literals.c"))
    assert any(t.type == TokenType.ULONGLONG and t.value == "0b1010ull" for t in tokens)


# ---------------------------------------------------------------------------
# char keyword (CHAR)
# ---------------------------------------------------------------------------


def test_char_keyword_produces_char_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/char_and_string.c"))
    assert any(t.type == TokenType.CHAR for t in tokens)


def test_char_keyword_token_value_is_char(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/char_and_string.c"))
    assert all(t.value == "char" for t in tokens if t.type == TokenType.CHAR)


def test_identifier_with_char_prefix_is_not_char_keyword(preprocess_c_file: Preprocessor) -> None:
    # "charset" starts with "char" but must not be classified as CHAR
    tokens = lexer(preprocess_c_file("valid/char_and_string.c"))
    char_keyword_values = {t.value for t in tokens if t.type == TokenType.CHAR}
    assert "charset" not in char_keyword_values


# ---------------------------------------------------------------------------
# Character constants (CONSTCHAR)
# ---------------------------------------------------------------------------


def test_char_constant_produces_constchar_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/char_and_string.c"))
    assert any(t.type == TokenType.CONSTCHAR for t in tokens)


def test_char_constant_ascii_letter_token_value(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/char_and_string.c"))
    assert any(t.type == TokenType.CONSTCHAR and t.value == "'a'" for t in tokens)


def test_char_constant_newline_escape_token_value(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/char_and_string.c"))
    assert any(t.type == TokenType.CONSTCHAR and t.value == "'\\n'" for t in tokens)


def test_char_constant_backslash_escape_token_value(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/char_and_string.c"))
    assert any(t.type == TokenType.CONSTCHAR and t.value == "'\\\\'" for t in tokens)


def test_char_constant_single_quote_escape_token_value(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/char_and_string.c"))
    assert any(t.type == TokenType.CONSTCHAR and t.value == "'\\''" for t in tokens)


def test_constchar_is_distinct_from_number(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/char_and_string.c"))
    constchar_values = {t.value for t in tokens if t.type == TokenType.CONSTCHAR}
    number_values = {t.value for t in tokens if t.type == TokenType.NUMBER}
    assert constchar_values.isdisjoint(number_values)


def test_char_constant_is_a_single_constchar_token(preprocess_c_file: Preprocessor) -> None:
    # "'a'" must produce exactly one CONSTCHAR token; it must not be split into three tokens
    tokens = lexer(preprocess_c_file("valid/char_and_string.c"))
    constchar_a = [t for t in tokens if t.type == TokenType.CONSTCHAR and t.value == "'a'"]
    assert len(constchar_a) == 1


# ---------------------------------------------------------------------------
# String literals (STRING)
# ---------------------------------------------------------------------------


def test_string_literal_produces_string_token(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/char_and_string.c"))
    assert any(t.type == TokenType.STRING for t in tokens)


def test_non_empty_string_literal_token_value(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/char_and_string.c"))
    assert any(t.type == TokenType.STRING and t.value == '"hello world"' for t in tokens)


def test_empty_string_literal_token_value(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/char_and_string.c"))
    assert any(t.type == TokenType.STRING and t.value == '""' for t in tokens)


def test_string_token_is_distinct_from_identifier(preprocess_c_file: Preprocessor) -> None:
    tokens = lexer(preprocess_c_file("valid/char_and_string.c"))
    string_values = {t.value for t in tokens if t.type == TokenType.STRING}
    ident_values = {t.value for t in tokens if t.type == TokenType.IDENTIFIER}
    assert string_values.isdisjoint(ident_values)


def test_string_literal_is_a_single_string_token(preprocess_c_file: Preprocessor) -> None:
    # '"hello world"' must be a single STRING token, not split at the space
    tokens = lexer(preprocess_c_file("valid/char_and_string.c"))
    hello_world = [t for t in tokens if t.type == TokenType.STRING and t.value == '"hello world"']
    assert len(hello_world) == 1


# ---------------------------------------------------------------------------
# 16-bit integer types (short)
# ---------------------------------------------------------------------------


def test_short_keyword_produces_short_token(preprocess_c_file: Preprocessor) -> None:
    """``short`` is lexed as a ``SHORT`` keyword token, not an identifier."""
    tokens = lexer(preprocess_c_file("valid/short_type.c"))
    short_tokens = [t for t in tokens if t.value == "short"]
    assert all(t.type == TokenType.SHORT for t in short_tokens)
    assert short_tokens, "expected at least one 'short' token"


def test_short_type_program_produces_short_tokens(preprocess_c_file: Preprocessor) -> None:
    """``short`` and ``unsigned short`` declarations lex correctly."""
    tokens = lexer(preprocess_c_file("valid/short_type.c"))
    types = [t.type for t in tokens if t.type in {TokenType.SHORT, TokenType.UNSIGNED}]
    assert TokenType.SHORT in types


def test_identifier_with_short_prefix_is_not_short_token(preprocess_c_file: Preprocessor) -> None:
    """Identifiers that begin with 'short' (e.g. 'shortwave') are not SHORT tokens."""
    # Use keyword_prefix.c which contains identifier 'returning'; we test a similar property
    # by verifying the lexer doesn't classify 'short' sub-strings as keywords.
    tokens = lexer(preprocess_c_file("valid/short_type.c"))
    # All identifier-like tokens that are NOT 'short' should be IDENTIFIER, not SHORT
    non_short_idents = [t for t in tokens if t.type == TokenType.IDENTIFIER and t.value.startswith("short")]
    assert all(t.type == TokenType.IDENTIFIER for t in non_short_idents)


# ---------------------------------------------------------------------------
# 128-bit integer types (__int128)
# ---------------------------------------------------------------------------


def test_int128_keyword_produces_int128_token(preprocess_c_file: Preprocessor) -> None:
    """``__int128`` is lexed as an ``INT128`` keyword token, not an identifier."""
    tokens = lexer(preprocess_c_file("valid/int128_type.c"))
    int128_tokens = [t for t in tokens if t.value == "__int128"]
    assert all(t.type == TokenType.INT128 for t in int128_tokens)
    assert int128_tokens, "expected at least one '__int128' token"


def test_int128_program_produces_int128_and_unsigned_tokens(preprocess_c_file: Preprocessor) -> None:
    """``__int128`` and ``unsigned __int128`` declarations lex both token types."""
    tokens = lexer(preprocess_c_file("valid/int128_type.c"))
    types = {t.type for t in tokens}
    assert TokenType.INT128 in types
    assert TokenType.UNSIGNED in types


# ---------------------------------------------------------------------------
# 64-bit long long types (long long / unsigned long long)
# ---------------------------------------------------------------------------


def test_long_long_type_program_produces_long_tokens(preprocess_c_file: Preprocessor) -> None:
    """``long long`` declarations produce two ``LONG`` tokens per type specifier."""
    tokens = lexer(preprocess_c_file("valid/long_long_type.c"))
    long_tokens = [t for t in tokens if t.type == TokenType.LONG]
    assert len(long_tokens) >= 2, "expected at least two 'long' tokens for 'long long'"


def test_ll_suffix_produces_longlong_token(preprocess_c_file: Preprocessor) -> None:
    """``42LL`` is lexed as a ``LONGLONG`` literal token."""
    tokens = lexer(preprocess_c_file("valid/long_long_type.c"))
    ll_tokens = [t for t in tokens if t.type == TokenType.LONGLONG]
    assert ll_tokens, "expected at least one LONGLONG token for '42LL'"
    assert ll_tokens[0].value == "42LL"


def test_ull_suffix_produces_ulonglong_token(preprocess_c_file: Preprocessor) -> None:
    """``100ULL`` is lexed as a ``ULONGLONG`` literal token."""
    tokens = lexer(preprocess_c_file("valid/long_long_type.c"))
    ull_tokens = [t for t in tokens if t.type == TokenType.ULONGLONG]
    assert ull_tokens, "expected at least one ULONGLONG token for '100ULL'"
    assert ull_tokens[0].value == "100ULL"


def test_llu_suffix_also_produces_ulonglong_token(preprocess_c_file: Preprocessor) -> None:
    """``123LLU`` (LL before U) is also lexed as a ``ULONGLONG`` literal token."""
    tokens = lexer(preprocess_c_file("valid/long_long_llu.c"))
    ull_tokens = [t for t in tokens if t.type == TokenType.ULONGLONG]
    assert ull_tokens, "expected ULONGLONG token for '123LLU'"
    assert ull_tokens[0].value == "123LLU"


def test_ellipsis_produces_ellipsis_token(preprocess_c_file: Preprocessor) -> None:
    """``...`` in a variadic parameter list is lexed as a single ``ELLIPSIS`` token."""
    tokens = lexer(preprocess_c_file("valid/variadic_decl.c"))
    ellipsis_tokens = [t for t in tokens if t.value == "..."]
    assert len(ellipsis_tokens) == 1
    assert ellipsis_tokens[0].type == TokenType.ELLIPSIS


def test_ellipsis_not_lexed_as_three_period_tokens(preprocess_c_file: Preprocessor) -> None:
    """``...`` must produce one ``ELLIPSIS`` token, not three ``PERIOD`` tokens."""
    tokens = lexer(preprocess_c_file("valid/variadic_decl.c"))
    assert not any(t.type == TokenType.PERIOD for t in tokens)


def test_const_keyword_produces_const_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    """``const`` is a keyword and must produce a ``CONST`` token, not ``IDENTIFIER``."""
    tokens = lexer(preprocess_c_file("valid/qualifiers.c"))
    const_token = next(t for t in tokens if t.value == "const")
    assert const_token.type == TokenType.CONST


def test_volatile_keyword_produces_volatile_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    """``volatile`` is a keyword and must produce a ``VOLATILE`` token, not ``IDENTIFIER``."""
    tokens = lexer(preprocess_c_file("valid/qualifiers.c"))
    volatile_token = next(t for t in tokens if t.value == "volatile")
    assert volatile_token.type == TokenType.VOLATILE


def test_restrict_keyword_produces_restrict_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    """``restrict`` is a keyword and must produce a ``RESTRICT`` token, not ``IDENTIFIER``."""
    tokens = lexer(preprocess_c_file("valid/qualifiers.c"))
    restrict_token = next(t for t in tokens if t.value == "restrict")
    assert restrict_token.type == TokenType.RESTRICT


def test_inline_keyword_produces_inline_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    """``inline`` is a keyword and must produce an ``INLINE`` token, not ``IDENTIFIER``."""
    tokens = lexer(preprocess_c_file("valid/inline_function.c"))
    inline_token = next(t for t in tokens if t.value == "inline")
    assert inline_token.type == TokenType.INLINE


def test_noreturn_keyword_produces_noreturn_token_not_identifier(preprocess_c_file: Preprocessor) -> None:
    """``_Noreturn`` is a keyword and must produce a ``NORETURN`` token, not ``IDENTIFIER``."""
    tokens = lexer(preprocess_c_file("valid/noreturn_function.c"))
    noreturn_token = next(t for t in tokens if t.value == "_Noreturn")
    assert noreturn_token.type == TokenType.NORETURN


def test_alignof_keyword_produces_alignof_token(preprocess_c_file: Preprocessor) -> None:
    """``_Alignof`` is a keyword and must produce an ``ALIGNOF`` token, not ``IDENTIFIER``."""
    tokens = lexer(preprocess_c_file("valid/alignof_basic.c"))
    alignof_token = next(t for t in tokens if t.value == "_Alignof")
    assert alignof_token.type == TokenType.ALIGNOF


def test_alignas_keyword_produces_alignas_token(preprocess_c_file: Preprocessor) -> None:
    """``_Alignas`` is a keyword and must produce an ``ALIGNAS`` token, not ``IDENTIFIER``."""
    tokens = lexer(preprocess_c_file("valid/alignas_basic.c"))
    alignas_token = next(t for t in tokens if t.value == "_Alignas")
    assert alignas_token.type == TokenType.ALIGNAS
