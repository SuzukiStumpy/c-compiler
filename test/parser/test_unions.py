"""Parser tests for union declarations and member-access expressions."""

import pytest

from compiler.parser import parse
from compiler.syntax.ast import (
    Arrow,
    Dot,
    IntType,
    LongType,
    MemberDeclaration,
    Pointer,
    Program,
    Union,
    UnionDecl,
    VarDecl,
)
from compiler.types.tokens import Token, TokenType
from compiler.utils.exceptions import ParseError
from test.parser.helpers import (
    _INT,
    _LONG,
    _RBRACE,
    _SEMICOLON,
    _ident,
    _parsed_exp,
)


def _tok(token_type: TokenType, value: str) -> Token:
    """Construct a Token with default positional information."""
    return Token(type=token_type, value=value, line=1, col=1, filename="test.c")


_UNION = _tok(TokenType.UNION, "union")
_LBRACE = _tok(TokenType.LBRACE, "{")
_PERIOD = _tok(TokenType.PERIOD, ".")
_ARROW = _tok(TokenType.ARROW, "->")
_ASTERISK = _tok(TokenType.ASTERISK, "*")


def _parse_top(*tokens: Token) -> Program:
    """Parse a token stream as a complete program and return the result."""
    return parse(list(tokens))


class TestUnionForwardDeclaration:
    """A bare ``union Tag;`` produces an empty-member ``UnionDecl``."""

    def test_forward_declaration_tag_name(self) -> None:
        """The ``UnionDeclaration`` carries the correct tag."""
        prog = _parse_top(_UNION, _ident("Foo"), _SEMICOLON)
        assert len(prog.declarations) == 1
        decl = prog.declarations[0]
        assert isinstance(decl, UnionDecl)
        assert decl.union_declaration.tag == "Foo"

    def test_forward_declaration_has_empty_member_list(self) -> None:
        """A forward declaration produces an empty ``members`` tuple."""
        prog = _parse_top(_UNION, _ident("Foo"), _SEMICOLON)
        decl = prog.declarations[0]
        assert isinstance(decl, UnionDecl)
        assert decl.union_declaration.members == ()


class TestUnionDefinition:
    """A ``union Tag { ... };`` produces a ``UnionDecl`` with populated members."""

    def test_single_int_member(self) -> None:
        """``union Data { int x; };`` yields one ``MemberDeclaration`` of ``IntType``."""
        prog = _parse_top(
            _UNION,
            _ident("Data"),
            _LBRACE,
            _INT,
            _ident("x"),
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
        )
        decl = prog.declarations[0]
        assert isinstance(decl, UnionDecl)
        ud = decl.union_declaration
        assert ud.tag == "Data"
        assert len(ud.members) == 1
        assert ud.members[0] == MemberDeclaration(member_name="x", member_type=IntType())

    def test_multiple_members(self) -> None:
        """``union U { int x; long y; };`` produces two members in declaration order."""
        prog = _parse_top(
            _UNION,
            _ident("U"),
            _LBRACE,
            _INT,
            _ident("x"),
            _SEMICOLON,
            _LONG,
            _ident("y"),
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
        )
        decl = prog.declarations[0]
        assert isinstance(decl, UnionDecl)
        members = decl.union_declaration.members
        assert len(members) == 2
        assert members[0] == MemberDeclaration(member_name="x", member_type=IntType())
        assert members[1] == MemberDeclaration(member_name="y", member_type=LongType())

    def test_empty_brace_body_is_a_parse_error(self) -> None:
        """``union Tag {};`` must raise ``ParseError`` - at least one member required."""
        with pytest.raises(ParseError):
            _parse_top(_UNION, _ident("Bad"), _LBRACE, _RBRACE, _SEMICOLON)


class TestUnionVariable:
    """A ``union Tag varname;`` produces a ``VarDecl`` with a ``Union`` type."""

    def test_variable_has_union_type(self) -> None:
        """``union U u;`` yields a ``VarDecl`` whose ``var_type`` is ``Union(tag='U')``."""
        prog = _parse_top(_UNION, _ident("U"), _ident("u"), _SEMICOLON)
        assert len(prog.declarations) == 1
        decl = prog.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == Union(tag="U")

    def test_pointer_to_union_variable(self) -> None:
        """``union U *p;`` yields a ``VarDecl`` with type ``Pointer(Union(tag='U'))``."""
        prog = _parse_top(_UNION, _ident("U"), _ASTERISK, _ident("p"), _SEMICOLON)
        decl = prog.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == Pointer(Union(tag="U"))


class TestUnionMemberAccess:
    """Dot and Arrow work on union-typed expressions."""

    def test_dot_access_on_union(self) -> None:
        """``u.x`` parses as ``Dot(Var('u'), 'x')`` regardless of aggregate kind."""
        exp = _parsed_exp(_ident("u"), _PERIOD, _ident("x"))
        assert isinstance(exp, Dot)
        assert exp.member == "x"

    def test_arrow_access_on_union_pointer(self) -> None:
        """``p->x`` parses as ``Arrow(Var('p'), 'x')``."""
        exp = _parsed_exp(_ident("p"), _ARROW, _ident("x"))
        assert isinstance(exp, Arrow)
        assert exp.member == "x"
