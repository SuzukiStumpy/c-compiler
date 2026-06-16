"""Parser tests for enum declarations and enum type specifiers."""

import pytest

from compiler.parser import parse
from compiler.syntax.ast import EnumDecl, EnumType, Program, VarDecl
from compiler.types.tokens import Token, TokenType
from compiler.utils.exceptions import ParseError


def _tok(token_type: TokenType, value: str) -> Token:
    """Construct a Token with default positional information.

    Args:
        token_type: The classification of the token.
        value: The source text of the token.

    Returns:
        An immutable ``Token`` instance.
    """
    return Token(type=token_type, value=value, line=1, col=1, filename="test.c")


_ENUM = _tok(TokenType.ENUM, "enum")
_LBRACE = _tok(TokenType.LBRACE, "{")
_RBRACE = _tok(TokenType.RBRACE, "}")
_SEMICOLON = _tok(TokenType.SEMICOLON, ";")
_COMMA = _tok(TokenType.COMMA, ",")
_INT = _tok(TokenType.INT, "int")
_VOID = _tok(TokenType.VOID, "void")
_LPAREN = _tok(TokenType.LPAREN, "(")
_RPAREN = _tok(TokenType.RPAREN, ")")
_ASSIGNMENT = _tok(TokenType.ASSIGNMENT, "=")
_MINUS = _tok(TokenType.MINUS, "-")


def _ident(name: str) -> Token:
    """Return an IDENTIFIER token with the given name.

    Args:
        name: The identifier text.

    Returns:
        A ``Token`` of type ``IDENTIFIER``.
    """
    return _tok(TokenType.IDENTIFIER, name)


def _number(value: str) -> Token:
    """Return a NUMBER token with the given literal text.

    Args:
        value: The integer literal text.

    Returns:
        A ``Token`` of type ``NUMBER``.
    """
    return _tok(TokenType.NUMBER, value)


def _parse_top(*tokens: Token) -> Program:
    """Parse a token stream as a complete program and return the result.

    Args:
        *tokens: The complete token stream forming the program.

    Returns:
        The parsed ``Program`` node.
    """
    return parse(list(tokens))


class TestEnumDefinition:
    """``enum Tag { enumerator-list };`` produces an ``EnumDecl`` node."""

    def test_single_enumerator_default_value(self) -> None:
        """The first enumerator in an enum has value 0 by default."""
        prog = _parse_top(_ENUM, _ident("Color"), _LBRACE, _ident("RED"), _RBRACE, _SEMICOLON)
        assert len(prog.declarations) == 1
        decl = prog.declarations[0]
        assert isinstance(decl, EnumDecl)
        assert decl.enum_declaration.tag == "Color"
        assert len(decl.enum_declaration.members) == 1
        assert decl.enum_declaration.members[0].name == "RED"
        assert decl.enum_declaration.members[0].value == 0

    def test_three_enumerators_increment_by_one(self) -> None:
        """Three enumerators without explicit values get 0, 1, 2."""
        prog = _parse_top(
            _ENUM,
            _ident("Color"),
            _LBRACE,
            _ident("RED"),
            _COMMA,
            _ident("GREEN"),
            _COMMA,
            _ident("BLUE"),
            _RBRACE,
            _SEMICOLON,
        )
        decl = prog.declarations[0]
        assert isinstance(decl, EnumDecl)
        members = decl.enum_declaration.members
        assert len(members) == 3
        assert members[0].name == "RED" and members[0].value == 0
        assert members[1].name == "GREEN" and members[1].value == 1
        assert members[2].name == "BLUE" and members[2].value == 2

    def test_explicit_value_on_first_enumerator(self) -> None:
        """An explicit value overrides the running counter."""
        prog = _parse_top(
            _ENUM,
            _ident("Direction"),
            _LBRACE,
            _ident("NORTH"),
            _ASSIGNMENT,
            _number("10"),
            _COMMA,
            _ident("SOUTH"),
            _RBRACE,
            _SEMICOLON,
        )
        decl = prog.declarations[0]
        assert isinstance(decl, EnumDecl)
        members = decl.enum_declaration.members
        assert members[0].value == 10
        assert members[1].value == 11  # auto-incremented from 10

    def test_negative_explicit_value(self) -> None:
        """Explicit values may be negative using a unary minus."""
        prog = _parse_top(
            _ENUM,
            _ident("E"),
            _LBRACE,
            _ident("A"),
            _ASSIGNMENT,
            _MINUS,
            _number("1"),
            _COMMA,
            _ident("B"),
            _RBRACE,
            _SEMICOLON,
        )
        decl = prog.declarations[0]
        assert isinstance(decl, EnumDecl)
        members = decl.enum_declaration.members
        assert members[0].value == -1
        assert members[1].value == 0  # -1 + 1

    def test_trailing_comma_is_permitted(self) -> None:
        """A trailing comma after the last enumerator is accepted."""
        prog = _parse_top(_ENUM, _ident("E"), _LBRACE, _ident("A"), _COMMA, _RBRACE, _SEMICOLON)
        decl = prog.declarations[0]
        assert isinstance(decl, EnumDecl)
        assert len(decl.enum_declaration.members) == 1

    def test_enum_tag_is_stored(self) -> None:
        """The parsed tag name is preserved on the ``EnumDeclaration`` node."""
        prog = _parse_top(_ENUM, _ident("MyEnum"), _LBRACE, _ident("X"), _RBRACE, _SEMICOLON)
        decl = prog.declarations[0]
        assert isinstance(decl, EnumDecl)
        assert decl.enum_declaration.tag == "MyEnum"


class TestEnumAsTypeSpecifier:
    """``enum Tag`` used as a type specifier in a variable or function declaration."""

    def test_enum_type_variable_declaration(self) -> None:
        """``enum Color x;`` parses to a ``VarDecl`` with ``EnumType("Color")``."""
        prog = _parse_top(
            _ENUM,
            _ident("Color"),
            _LBRACE,
            _ident("RED"),
            _RBRACE,
            _SEMICOLON,
            _ENUM,
            _ident("Color"),
            _ident("x"),
            _SEMICOLON,
        )
        assert len(prog.declarations) == 2
        var_decl = prog.declarations[1]
        assert isinstance(var_decl, VarDecl)
        assert isinstance(var_decl.variable_declaration.var_type, EnumType)
        assert var_decl.variable_declaration.var_type.tag == "Color"

    def test_enum_type_specifier_after_separate_definition(self) -> None:
        """An enum defined first and then used as a type specifier produces ``EnumType``."""
        prog = _parse_top(
            _ENUM,
            _ident("E"),
            _LBRACE,
            _ident("A"),
            _RBRACE,
            _SEMICOLON,
            _ENUM,
            _ident("E"),
            _ident("v"),
            _SEMICOLON,
        )
        var_decl = prog.declarations[1]
        assert isinstance(var_decl, VarDecl)
        assert isinstance(var_decl.variable_declaration.var_type, EnumType)


class TestEnumParseErrors:
    """Invalid enum constructs raise ``ParseError``."""

    def test_enum_without_body_or_tag_fails(self) -> None:
        """``enum;`` is not valid syntax."""
        with pytest.raises(ParseError):
            _parse_top(_ENUM, _SEMICOLON)

    def test_empty_enum_body_fails(self) -> None:
        """``enum E {}`` is not valid — at least one enumerator is required."""
        with pytest.raises(ParseError):
            _parse_top(_ENUM, _ident("E"), _LBRACE, _RBRACE, _SEMICOLON)
