"""Parser tests for GCC __attribute__((...)) skipping.

All four common positions tested:
  1. Before the type specifiers:    ``__attribute__((x)) int foo(void);``
  2. Between specifiers/declarator: ``int __attribute__((x)) foo(void);``
  3. After the declarator:          ``int foo(void) __attribute__((x));``
  4. In struct member:              ``struct { int x __attribute__((packed)); };``
  5. In typedef (between type and name): ``typedef int __attribute__((mode(__SI__))) myint;``
  6. Multiple consecutive attributes, nested parens.
"""

from compiler.parser import parse
from compiler.syntax.ast import (
    FunDecl,
    IntType,
    MemberDeclaration,
    StructDecl,
    TypedefDecl,
    UIntType,
    VarDecl,
)
from compiler.types.tokens import TokenType
from test.parser.helpers import (
    _INT,
    _LPAREN,
    _RPAREN,
    _SEMICOLON,
    _UNSIGNED,
    _VOID,
    _ident,
    _tok,
)

_ATTRIBUTE = _tok(TokenType.ATTRIBUTE, "__attribute__")
_LBRACE = _tok(TokenType.LBRACE, "{")
_RBRACE = _tok(TokenType.RBRACE, "}")
_LPAREN2 = _tok(TokenType.LPAREN, "(")
_RPAREN2 = _tok(TokenType.RPAREN, ")")
_TYPEDEF = _tok(TokenType.TYPEDEF, "typedef")
_STRUCT = _tok(TokenType.STRUCT, "struct")
_COMMA = _tok(TokenType.COMMA, ",")
_EXTERN = _tok(TokenType.EXTERN, "extern")


def _attr(*content_toks: object) -> list[object]:
    """Produce ``__attribute__ (( content ))`` token list."""
    return [_ATTRIBUTE, _LPAREN2, _LPAREN2, *content_toks, _RPAREN2, _RPAREN2]


def _nothrow() -> list[object]:
    """``__attribute__((__nothrow__))`` token list."""
    return _attr(_ident("__nothrow__"))


def _unused() -> list[object]:
    """``__attribute__((unused))`` token list."""
    return _attr(_ident("unused"))


# ---------------------------------------------------------------------------
# Position 3: attribute AFTER the declarator
# ---------------------------------------------------------------------------


class TestAttributeAfterDeclarator:
    """``int foo(void) __attribute__((nothrow));`` — attribute after function declarator."""

    def test_function_decl_parsed(self) -> None:
        """Declaration is produced with correct name and type."""
        tokens = [_INT, _ident("foo"), _LPAREN, _VOID, _RPAREN, *_nothrow(), _SEMICOLON]
        prog = parse(tokens)
        assert len(prog.declarations) == 1
        decl = prog.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.name == "foo"

    def test_variable_decl_with_attribute(self) -> None:
        """``int x __attribute__((unused));`` parses as VarDecl of IntType."""
        tokens = [_INT, _ident("x"), *_unused(), _SEMICOLON]
        prog = parse(tokens)
        assert len(prog.declarations) == 1
        decl = prog.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == IntType()

    def test_multiple_attributes(self) -> None:
        """Two consecutive ``__attribute__`` blocks are both skipped."""
        tokens = [_INT, _ident("foo"), _LPAREN, _VOID, _RPAREN, *_nothrow(), *_unused(), _SEMICOLON]
        prog = parse(tokens)
        assert isinstance(prog.declarations[0], FunDecl)

    def test_nested_parens_in_attribute(self) -> None:
        """``__attribute__((format(__printf__, 1, 2)))`` — nested parens skipped correctly."""
        _number = _tok(TokenType.NUMBER, "1")
        _number2 = _tok(TokenType.NUMBER, "2")
        nested_attr = _attr(
            _ident("format"),
            _LPAREN2,
            _ident("__printf__"),
            _COMMA,
            _number,
            _COMMA,
            _number2,
            _RPAREN2,
        )
        tokens = [_INT, _ident("printf"), _LPAREN, _VOID, _RPAREN, *nested_attr, _SEMICOLON]
        prog = parse(tokens)
        assert isinstance(prog.declarations[0], FunDecl)

    def test_extern_function_with_attribute(self) -> None:
        """``extern int foo(void) __attribute__((nothrow));`` parses."""
        tokens = [_EXTERN, _INT, _ident("foo"), _LPAREN, _VOID, _RPAREN, *_nothrow(), _SEMICOLON]
        prog = parse(tokens)
        assert isinstance(prog.declarations[0], FunDecl)


# ---------------------------------------------------------------------------
# Position 2: attribute BETWEEN type specifiers and declarator
# ---------------------------------------------------------------------------


class TestAttributeBetweenTypeAndDeclarator:
    """``int __attribute__((nothrow)) foo(void);`` — attribute between type and name."""

    def test_function_decl_parsed(self) -> None:
        """Function declaration is produced correctly."""
        tokens = [_INT, *_nothrow(), _ident("foo"), _LPAREN, _VOID, _RPAREN, _SEMICOLON]
        prog = parse(tokens)
        assert len(prog.declarations) == 1
        decl = prog.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.name == "foo"

    def test_variable_decl_parsed(self) -> None:
        """``int __attribute__((unused)) x;`` parses as VarDecl."""
        tokens = [_INT, *_unused(), _ident("x"), _SEMICOLON]
        prog = parse(tokens)
        assert isinstance(prog.declarations[0], VarDecl)


# ---------------------------------------------------------------------------
# Position 1: attribute BEFORE specifiers
# ---------------------------------------------------------------------------


class TestAttributeBeforeSpecifiers:
    """``__attribute__((unused)) int foo(void);`` — attribute before type specifiers."""

    def test_function_decl_parsed(self) -> None:
        tokens = [*_unused(), _INT, _ident("foo"), _LPAREN, _VOID, _RPAREN, _SEMICOLON]
        prog = parse(tokens)
        assert isinstance(prog.declarations[0], FunDecl)


# ---------------------------------------------------------------------------
# Position 4: attribute in struct member
# ---------------------------------------------------------------------------


class TestAttributeInStructMember:
    """``struct S { int x __attribute__((packed)); };`` — attribute on member."""

    def test_member_name_and_type_preserved(self) -> None:
        """Member is parsed with correct name and type; attribute is discarded."""
        tokens = [
            _STRUCT,
            _ident("S"),
            _LBRACE,
            _INT,
            _ident("x"),
            *_unused(),
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
        ]
        prog = parse(tokens)
        assert len(prog.declarations) == 1
        sd = prog.declarations[0]
        assert isinstance(sd, StructDecl)
        members = sd.struct_declaration.members
        assert len(members) == 1
        m = members[0]
        assert isinstance(m, MemberDeclaration)
        assert m.member_name == "x"
        assert m.member_type == IntType()


# ---------------------------------------------------------------------------
# Position 5: attribute in typedef (between type and alias name)
# ---------------------------------------------------------------------------


class TestAttributeInTypedef:
    """``typedef unsigned int __attribute__((mode(__QI__))) uint8_t;`` — attribute in typedef."""

    def test_typedef_with_attribute_between_type_and_name(self) -> None:
        """Attribute between type and alias name is skipped; TypedefDecl is produced."""
        mode_attr = _attr(
            _ident("mode"),
            _LPAREN2,
            _ident("__QI__"),
            _RPAREN2,
        )
        tokens = [_TYPEDEF, _UNSIGNED, _INT, *mode_attr, _ident("uint8_t"), _SEMICOLON]
        prog = parse(tokens)
        assert len(prog.declarations) == 1
        td = prog.declarations[0]
        assert isinstance(td, TypedefDecl)
        assert td.name == "uint8_t"
        assert td.aliased_type == UIntType()

    def test_typedef_struct_with_attribute_after_alias(self) -> None:
        """Attribute after typedef alias name is skipped."""
        tokens = [
            _TYPEDEF,
            _INT,
            _ident("myint"),
            *_unused(),
            _SEMICOLON,
        ]
        prog = parse(tokens)
        assert len(prog.declarations) == 1
        td = prog.declarations[0]
        assert isinstance(td, TypedefDecl)
        assert td.name == "myint"


# ---------------------------------------------------------------------------
# Regression: normal declarations unaffected
# ---------------------------------------------------------------------------


class TestAttributeRegression:
    """Declarations without any __attribute__ still parse correctly."""

    def test_plain_function_decl(self) -> None:
        tokens = [_INT, _ident("foo"), _LPAREN, _VOID, _RPAREN, _SEMICOLON]
        prog = parse(tokens)
        assert isinstance(prog.declarations[0], FunDecl)

    def test_plain_var_decl(self) -> None:
        tokens = [_INT, _ident("x"), _SEMICOLON]
        prog = parse(tokens)
        assert isinstance(prog.declarations[0], VarDecl)

    def test_plain_typedef(self) -> None:
        tokens = [_TYPEDEF, _INT, _ident("myint"), _SEMICOLON]
        prog = parse(tokens)
        assert isinstance(prog.declarations[0], TypedefDecl)
