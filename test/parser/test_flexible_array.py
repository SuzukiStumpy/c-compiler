"""Parser tests for flexible array members: ``struct { ...; T arr[]; }``."""

from compiler.syntax.ast import Array, Char, IntType, StructDecl
from compiler.types.tokens import TokenType
from test.parser.helpers import (
    _LBRACE,
    _LBRACKET,
    _LPAREN,
    _RBRACE,
    _RBRACKET,
    _RPAREN,
    _SEMICOLON,
    _ident,
    _tok,
    parse,
)

_STRUCT = _tok(TokenType.STRUCT, "struct")
_CHAR = _tok(TokenType.CHAR, "char")
_INT = _tok(TokenType.INT, "int")


class TestFlexibleArrayMemberParsing:
    """``T name[];`` as a struct member is parsed with ``Array(T, size=None)``."""

    def test_flexible_char_array_member(self) -> None:
        """``struct S { int n; char data[]; };`` gives last member type Array(Char, None)."""
        tokens = [
            _STRUCT,
            _ident("S"),
            _LBRACE,
            _INT,
            _ident("n"),
            _SEMICOLON,
            _CHAR,
            _ident("data"),
            _LBRACKET,
            _RBRACKET,
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
        ]
        prog = parse(tokens)
        decl = prog.declarations[0]
        assert isinstance(decl, StructDecl)
        members = decl.struct_declaration.members
        assert len(members) == 2
        data_member = members[1]
        assert data_member.member_name == "data"
        assert isinstance(data_member.member_type, Array)
        assert isinstance(data_member.member_type.element, Char)
        assert data_member.member_type.size is None

    def test_flexible_int_array_member(self) -> None:
        """``struct S { int n; int vals[]; };`` gives last member type Array(Int, None)."""
        tokens = [
            _STRUCT,
            _ident("S"),
            _LBRACE,
            _INT,
            _ident("n"),
            _SEMICOLON,
            _INT,
            _ident("vals"),
            _LBRACKET,
            _RBRACKET,
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
        ]
        prog = parse(tokens)
        decl = prog.declarations[0]
        assert isinstance(decl, StructDecl)
        last_member = decl.struct_declaration.members[-1]
        assert isinstance(last_member.member_type, Array)
        assert last_member.member_type.size is None
        assert isinstance(last_member.member_type.element, IntType)


class TestArrayCompoundLiteralSizeInference:
    """``(T[]){...}`` compound literal infers its size from the initializer."""

    def test_empty_brackets_in_abstract_declarator(self) -> None:
        """``(int[])`` in cast position should parse ``Array(IntType, None)``."""
        from compiler.syntax.ast.expressions import CompoundLiteral

        # (int[]){1, 2, 3} in return position
        _PLUS = _tok(TokenType.PLUS, "+")
        _COMMA = _tok(TokenType.COMMA, ",")
        from test.parser.helpers import _number, _parsed_exp

        exp = _parsed_exp(
            _LPAREN,
            _INT,
            _LBRACKET,
            _RBRACKET,
            _RPAREN,
            _LBRACE,
            _number("1"),
            _COMMA,
            _number("2"),
            _COMMA,
            _number("3"),
            _RBRACE,
        )
        assert isinstance(exp, CompoundLiteral)
        assert isinstance(exp.target_type, Array)
        assert exp.target_type.size is None  # parser doesn't resolve size
        assert isinstance(exp.target_type.element, IntType)
