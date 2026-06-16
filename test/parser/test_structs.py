"""Parser tests for struct declarations and member-access expressions."""

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
    StructDecl,
    Structure,
    UIntType,
    VarDecl,
)
from compiler.types.tokens import Token, TokenType
from compiler.utils.exceptions import ParseError
from test.parser.helpers import (
    _COLON,
    _INT,
    _LONG,
    _LPAREN,
    _RBRACE,
    _RPAREN,
    _SEMICOLON,
    _UNSIGNED,
    _VOID,
    _ident,
    _number,
    _parsed_exp,
)


def _tok(token_type: TokenType, value: str) -> Token:
    """Construct a Token with default positional information."""
    return Token(type=token_type, value=value, line=1, col=1, filename="test.c")


_STRUCT = _tok(TokenType.STRUCT, "struct")
_LBRACE = _tok(TokenType.LBRACE, "{")
_PERIOD = _tok(TokenType.PERIOD, ".")
_ARROW = _tok(TokenType.ARROW, "->")
_ASTERISK = _tok(TokenType.ASTERISK, "*")


def _parse_top(*tokens: Token) -> Program:
    """Parse a token stream as a complete program and return the result."""
    return parse(list(tokens))


class TestStructForwardDeclaration:
    """A bare ``struct Tag;`` produces an empty-member ``StructDecl``."""

    def test_forward_declaration_tag_name(self) -> None:
        """The ``StructDeclaration`` carries the correct tag."""
        prog = _parse_top(_STRUCT, _ident("Foo"), _SEMICOLON)
        assert len(prog.declarations) == 1
        decl = prog.declarations[0]
        assert isinstance(decl, StructDecl)
        assert decl.struct_declaration.tag == "Foo"

    def test_forward_declaration_has_empty_member_list(self) -> None:
        """A forward declaration produces an empty ``members`` tuple."""
        prog = _parse_top(_STRUCT, _ident("Foo"), _SEMICOLON)
        decl = prog.declarations[0]
        assert isinstance(decl, StructDecl)
        assert decl.struct_declaration.members == ()


class TestStructDefinition:
    """A ``struct Tag { ... };`` produces a ``StructDecl`` with populated members."""

    def test_single_int_member(self) -> None:
        """``struct Point { int x; };`` yields one ``MemberDeclaration`` of ``IntType``."""
        prog = _parse_top(
            _STRUCT,
            _ident("Point"),
            _LBRACE,
            _INT,
            _ident("x"),
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
        )
        decl = prog.declarations[0]
        assert isinstance(decl, StructDecl)
        sd = decl.struct_declaration
        assert sd.tag == "Point"
        assert len(sd.members) == 1
        assert sd.members[0] == MemberDeclaration(member_name="x", member_type=IntType())

    def test_multiple_members(self) -> None:
        """``struct P { int x; long y; };`` produces two members in declaration order."""
        prog = _parse_top(
            _STRUCT,
            _ident("P"),
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
        assert isinstance(decl, StructDecl)
        members = decl.struct_declaration.members
        assert len(members) == 2
        assert members[0] == MemberDeclaration(member_name="x", member_type=IntType())
        assert members[1] == MemberDeclaration(member_name="y", member_type=LongType())

    def test_pointer_member(self) -> None:
        """A pointer member ``int *p`` is captured as ``Pointer(IntType())``."""
        prog = _parse_top(
            _STRUCT,
            _ident("Node"),
            _LBRACE,
            _INT,
            _ASTERISK,
            _ident("next"),
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
        )
        decl = prog.declarations[0]
        assert isinstance(decl, StructDecl)
        members = decl.struct_declaration.members
        assert len(members) == 1
        assert members[0] == MemberDeclaration(member_name="next", member_type=Pointer(IntType()))

    def test_struct_member_type(self) -> None:
        """A member whose type is another struct is captured as ``Structure(tag)``."""
        prog = _parse_top(
            _STRUCT,
            _ident("Inner"),
            _LBRACE,
            _INT,
            _ident("val"),
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
            _STRUCT,
            _ident("Outer"),
            _LBRACE,
            _STRUCT,
            _ident("Inner"),
            _ident("inner"),
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
        )
        outer = prog.declarations[1]
        assert isinstance(outer, StructDecl)
        members = outer.struct_declaration.members
        assert len(members) == 1
        assert members[0] == MemberDeclaration(member_name="inner", member_type=Structure(tag="Inner"))


class TestStructDistinguishedFromVarDecl:
    """The parser correctly distinguishes ``struct Tag;`` from ``struct Tag varname;``."""

    def test_struct_tag_then_semicolon_is_struct_decl(self) -> None:
        """``struct Foo;`` at file scope is a ``StructDecl``, not a ``VarDecl``."""
        prog = _parse_top(_STRUCT, _ident("Foo"), _SEMICOLON)
        assert isinstance(prog.declarations[0], StructDecl)

    def test_struct_tag_then_identifier_is_var_decl(self) -> None:
        """``struct Foo s;`` is a ``VarDecl`` with type ``Structure("Foo")``."""
        prog = _parse_top(_STRUCT, _ident("Foo"), _ident("s"), _SEMICOLON)
        decl = prog.declarations[0]
        assert isinstance(decl, VarDecl)
        vd = decl.variable_declaration
        assert isinstance(vd.var_type, Structure)
        assert vd.var_type.tag == "Foo"
        assert vd.name == "s"

    def test_struct_tag_then_brace_is_struct_decl(self) -> None:
        """``struct Foo { int x; };`` is a ``StructDecl``."""
        prog = _parse_top(
            _STRUCT,
            _ident("Foo"),
            _LBRACE,
            _INT,
            _ident("x"),
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
        )
        assert isinstance(prog.declarations[0], StructDecl)


class TestDotExpression:
    """``s.member`` parses to a ``Dot`` node."""

    def test_simple_dot_access(self) -> None:
        """``return s.x;`` produces ``Dot(Var("s"), "x")``."""
        exp = _parsed_exp(_ident("s"), _PERIOD, _ident("x"))
        assert isinstance(exp, Dot)
        assert exp.member == "x"

    def test_dot_structure_operand(self) -> None:
        """The structure sub-expression of a ``Dot`` is the left operand variable."""
        from compiler.syntax.ast.expressions import Var  # noqa: PLC0415

        exp = _parsed_exp(_ident("s"), _PERIOD, _ident("x"))
        assert isinstance(exp, Dot)
        assert isinstance(exp.structure, Var)
        assert exp.structure.name == "s"

    def test_chained_dot_access(self) -> None:
        """``s.a.b`` parses as ``Dot(Dot(Var("s"), "a"), "b")`` (left-associative)."""
        exp = _parsed_exp(_ident("s"), _PERIOD, _ident("a"), _PERIOD, _ident("b"))
        assert isinstance(exp, Dot)
        assert exp.member == "b"
        assert isinstance(exp.structure, Dot)
        assert exp.structure.member == "a"

    def test_dot_member_name(self) -> None:
        """The ``member`` field carries the identifier text exactly."""
        exp = _parsed_exp(_ident("s"), _PERIOD, _ident("value"))
        assert isinstance(exp, Dot)
        assert exp.member == "value"


class TestArrowExpression:
    """``p->member`` parses to an ``Arrow`` node."""

    def test_simple_arrow_access(self) -> None:
        """``return p->x;`` produces ``Arrow(Var("p"), "x")``."""
        exp = _parsed_exp(_ident("p"), _ARROW, _ident("x"))
        assert isinstance(exp, Arrow)
        assert exp.member == "x"

    def test_arrow_pointer_operand(self) -> None:
        """The pointer sub-expression of an ``Arrow`` is the left operand variable."""
        from compiler.syntax.ast.expressions import Var  # noqa: PLC0415

        exp = _parsed_exp(_ident("p"), _ARROW, _ident("x"))
        assert isinstance(exp, Arrow)
        assert isinstance(exp.pointer, Var)
        assert exp.pointer.name == "p"

    def test_arrow_then_dot(self) -> None:
        """``p->a.b`` parses as ``Dot(Arrow(Var("p"), "a"), "b")``."""
        exp = _parsed_exp(_ident("p"), _ARROW, _ident("a"), _PERIOD, _ident("b"))
        assert isinstance(exp, Dot)
        assert exp.member == "b"
        assert isinstance(exp.structure, Arrow)
        assert exp.structure.member == "a"

    def test_arrow_member_name(self) -> None:
        """The ``member`` field carries the identifier text exactly."""
        exp = _parsed_exp(_ident("p"), _ARROW, _ident("next"))
        assert isinstance(exp, Arrow)
        assert exp.member == "next"


class TestStructParseErrors:
    """The parser raises ``ParseError`` on malformed struct constructs."""

    def test_empty_brace_body_raises(self) -> None:
        """``struct Foo {};`` is rejected: a definition must contain at least one member."""
        tokens = [_STRUCT, _ident("Foo"), _LBRACE, _RBRACE, _SEMICOLON]
        with pytest.raises(ParseError):
            parse(tokens)

    def test_missing_semicolon_after_struct_decl_raises(self) -> None:
        """``struct Foo`` without a trailing semicolon is rejected."""
        tokens = [_STRUCT, _ident("Foo")]
        with pytest.raises(ParseError):
            parse(tokens)

    def test_function_type_member_raises(self) -> None:
        """A member with a function-type declarator is rejected."""
        tokens = [
            _STRUCT,
            _ident("Bad"),
            _LBRACE,
            _INT,
            _ident("fn"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
        ]
        with pytest.raises(ParseError):
            parse(tokens)


class TestBitFieldMemberDeclaration:
    """Struct member declarations with bit-field width specifiers (C11 §6.7.2.1p3)."""

    def test_named_bit_field(self) -> None:
        """``unsigned int x : 3;`` yields MemberDeclaration with bit_width=3."""
        prog = _parse_top(
            _STRUCT,
            _ident("Flags"),
            _LBRACE,
            _UNSIGNED,
            _INT,
            _ident("x"),
            _COLON,
            _number("3"),
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
        )
        members = prog.declarations[0].struct_declaration.members  # type: ignore[union-attr]
        assert len(members) == 1
        assert members[0] == MemberDeclaration(member_name="x", member_type=UIntType(), bit_width=3)

    def test_unnamed_bit_field(self) -> None:
        """``int : 5;`` yields MemberDeclaration with member_name=None, bit_width=5."""
        prog = _parse_top(
            _STRUCT,
            _ident("Flags"),
            _LBRACE,
            _INT,
            _COLON,
            _number("5"),
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
        )
        members = prog.declarations[0].struct_declaration.members  # type: ignore[union-attr]
        assert len(members) == 1
        assert members[0] == MemberDeclaration(member_name=None, member_type=IntType(), bit_width=5)

    def test_zero_width_unnamed_bit_field(self) -> None:
        """``int : 0;`` yields MemberDeclaration with bit_width=0 (forces unit boundary)."""
        prog = _parse_top(
            _STRUCT,
            _ident("Flags"),
            _LBRACE,
            _INT,
            _COLON,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
        )
        members = prog.declarations[0].struct_declaration.members  # type: ignore[union-attr]
        assert len(members) == 1
        assert members[0] == MemberDeclaration(member_name=None, member_type=IntType(), bit_width=0)

    def test_zero_width_named_bit_field_is_parse_error(self) -> None:
        """``int x : 0;`` is a parse error (C11 §6.7.2.1p3)."""
        with pytest.raises(ParseError):
            _parse_top(
                _STRUCT,
                _ident("Flags"),
                _LBRACE,
                _INT,
                _ident("x"),
                _COLON,
                _number("0"),
                _SEMICOLON,
                _RBRACE,
                _SEMICOLON,
            )

    def test_struct_member_no_declarator_is_parse_error(self) -> None:
        """``struct s { int; };`` is rejected: a non-bit-field member must have a declarator."""
        with pytest.raises(ParseError):
            _parse_top(
                _STRUCT,
                _ident("S"),
                _LBRACE,
                _INT,
                _SEMICOLON,
                _RBRACE,
                _SEMICOLON,
            )

    def test_union_member_no_declarator_is_parse_error(self) -> None:
        """``union u { int; };`` is rejected: a non-bit-field member must have a declarator."""
        with pytest.raises(ParseError):
            _parse_top(
                _tok(TokenType.UNION, "union"),
                _ident("U"),
                _LBRACE,
                _INT,
                _SEMICOLON,
                _RBRACE,
                _SEMICOLON,
            )

    def test_regular_member_has_no_bit_width(self) -> None:
        """Regular members have bit_width=None."""
        prog = _parse_top(
            _STRUCT,
            _ident("S"),
            _LBRACE,
            _INT,
            _ident("x"),
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
        )
        members = prog.declarations[0].struct_declaration.members  # type: ignore[union-attr]
        assert members[0].bit_width is None

    def test_bit_field_mixed_with_regular_members(self) -> None:
        """Bit fields and regular members can coexist in the same struct."""
        prog = _parse_top(
            _STRUCT,
            _ident("Mixed"),
            _LBRACE,
            _INT,
            _ident("x"),
            _SEMICOLON,
            _UNSIGNED,
            _INT,
            _ident("flags"),
            _COLON,
            _number("3"),
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
        )
        members = prog.declarations[0].struct_declaration.members  # type: ignore[union-attr]
        assert len(members) == 2
        assert members[0] == MemberDeclaration(member_name="x", member_type=IntType(), bit_width=None)
        assert members[1] == MemberDeclaration(member_name="flags", member_type=UIntType(), bit_width=3)
