"""Parser tests for inline struct/union body inside typedef declarations.

Covers:
  typedef struct { ... } Alias;           -- anonymous struct
  typedef struct Tag { ... } Alias;       -- named struct with body
  typedef union { ... } Alias;            -- anonymous union
  typedef union Tag { ... } Alias;        -- named union with body
  typedef struct Tag *Alias;              -- ptr to forward-declared struct (regression)
"""

from compiler.parser import parse
from compiler.syntax.ast import (
    IntType,
    MemberDeclaration,
    StructDecl,
    Structure,
    TypedefDecl,
    Union,
    UnionDecl,
    VarDecl,
)
from compiler.types.tokens import TokenType
from test.parser.helpers import (
    _INT,
    _SEMICOLON,
    _ident,
    _tok,
)

_TYPEDEF = _tok(TokenType.TYPEDEF, "typedef")
_STRUCT = _tok(TokenType.STRUCT, "struct")
_UNION = _tok(TokenType.UNION, "union")
_LBRACE = _tok(TokenType.LBRACE, "{")
_RBRACE = _tok(TokenType.RBRACE, "}")
_ASTERISK = _tok(TokenType.ASTERISK, "*")


# ---------------------------------------------------------------------------
# Anonymous struct: typedef struct { int x; } Point;
# ---------------------------------------------------------------------------


class TestTypedefAnonStruct:
    """``typedef struct { ... } Alias;`` emits [StructDecl, TypedefDecl]."""

    def _tokens(self) -> list[object]:
        # typedef struct { int x; } Point;
        return [
            _TYPEDEF,
            _STRUCT,
            _LBRACE,
            _INT,
            _ident("x"),
            _SEMICOLON,
            _RBRACE,
            _ident("Point"),
            _SEMICOLON,
        ]

    def test_two_declarations_emitted(self) -> None:
        """Two top-level declarations are produced: StructDecl then TypedefDecl."""
        prog = parse(self._tokens())
        assert len(prog.declarations) == 2

    def test_first_is_struct_decl(self) -> None:
        """First declaration is a StructDecl."""
        prog = parse(self._tokens())
        assert isinstance(prog.declarations[0], StructDecl)

    def test_struct_has_one_int_member(self) -> None:
        """The anonymous struct has a single ``int x`` member."""
        prog = parse(self._tokens())
        decl = prog.declarations[0]
        assert isinstance(decl, StructDecl)
        members = decl.struct_declaration.members
        assert len(members) == 1
        m = members[0]
        assert isinstance(m, MemberDeclaration)
        assert m.member_name == "x"
        assert m.member_type == IntType()

    def test_second_is_typedef_decl(self) -> None:
        """Second declaration is a TypedefDecl aliasing ``Point``."""
        prog = parse(self._tokens())
        assert isinstance(prog.declarations[1], TypedefDecl)

    def test_typedef_alias_name(self) -> None:
        """TypedefDecl.name is 'Point'."""
        prog = parse(self._tokens())
        td = prog.declarations[1]
        assert isinstance(td, TypedefDecl)
        assert td.name == "Point"

    def test_typedef_aliased_type_is_structure(self) -> None:
        """TypedefDecl.aliased_type is the Structure whose tag matches the StructDecl."""
        prog = parse(self._tokens())
        sd = prog.declarations[0]
        td = prog.declarations[1]
        assert isinstance(sd, StructDecl)
        assert isinstance(td, TypedefDecl)
        assert isinstance(td.aliased_type, Structure)
        assert td.aliased_type.tag == sd.struct_declaration.tag

    def test_typedef_name_usable_as_type(self) -> None:
        """After the typedef, ``Point x;`` resolves to the struct type."""
        tokens = [
            _TYPEDEF,
            _STRUCT,
            _LBRACE,
            _INT,
            _ident("x"),
            _SEMICOLON,
            _RBRACE,
            _ident("Point"),
            _SEMICOLON,
            _ident("Point"),
            _ident("p"),
            _SEMICOLON,
        ]
        prog = parse(tokens)
        assert len(prog.declarations) == 3
        vd = prog.declarations[2]
        assert isinstance(vd, VarDecl)
        assert isinstance(vd.variable_declaration.var_type, Structure)


# ---------------------------------------------------------------------------
# Named struct: typedef struct Tag { int x; int y; } Tag;
# ---------------------------------------------------------------------------


class TestTypedefNamedStruct:
    """``typedef struct Tag { ... } Alias;`` emits [StructDecl, TypedefDecl]."""

    def _tokens(self) -> list[object]:
        # typedef struct Point { int x; int y; } Point;
        return [
            _TYPEDEF,
            _STRUCT,
            _ident("Point"),
            _LBRACE,
            _INT,
            _ident("x"),
            _SEMICOLON,
            _INT,
            _ident("y"),
            _SEMICOLON,
            _RBRACE,
            _ident("Point"),
            _SEMICOLON,
        ]

    def test_two_declarations_emitted(self) -> None:
        """Two top-level declarations are produced."""
        prog = parse(self._tokens())
        assert len(prog.declarations) == 2

    def test_struct_tag_correct(self) -> None:
        """The StructDecl carries the explicit tag 'Point'."""
        prog = parse(self._tokens())
        sd = prog.declarations[0]
        assert isinstance(sd, StructDecl)
        assert sd.struct_declaration.tag == "Point"

    def test_struct_has_two_int_members(self) -> None:
        """The struct body contains 'int x' and 'int y'."""
        prog = parse(self._tokens())
        sd = prog.declarations[0]
        assert isinstance(sd, StructDecl)
        members = sd.struct_declaration.members
        assert len(members) == 2
        assert members[0].member_name == "x"
        assert members[1].member_name == "y"

    def test_typedef_aliased_type_uses_explicit_tag(self) -> None:
        """TypedefDecl.aliased_type is Structure(tag='Point')."""
        prog = parse(self._tokens())
        td = prog.declarations[1]
        assert isinstance(td, TypedefDecl)
        assert td.aliased_type == Structure(tag="Point")

    def test_alias_different_from_tag(self) -> None:
        """Alias name can differ from the struct tag."""
        tokens = [
            _TYPEDEF,
            _STRUCT,
            _ident("Point"),
            _LBRACE,
            _INT,
            _ident("x"),
            _SEMICOLON,
            _RBRACE,
            _ident("Pt"),
            _SEMICOLON,
        ]
        prog = parse(tokens)
        sd = prog.declarations[0]
        td = prog.declarations[1]
        assert isinstance(sd, StructDecl)
        assert sd.struct_declaration.tag == "Point"
        assert isinstance(td, TypedefDecl)
        assert td.name == "Pt"
        assert td.aliased_type == Structure(tag="Point")


# ---------------------------------------------------------------------------
# Anonymous union: typedef union { int i; float f; } Value;
# ---------------------------------------------------------------------------


class TestTypedefAnonUnion:
    """``typedef union { ... } Alias;`` emits [UnionDecl, TypedefDecl]."""

    def _tokens(self) -> list[object]:
        _FLOAT = _tok(TokenType.FLOAT, "float")
        return [
            _TYPEDEF,
            _UNION,
            _LBRACE,
            _INT,
            _ident("i"),
            _SEMICOLON,
            _FLOAT,
            _ident("f"),
            _SEMICOLON,
            _RBRACE,
            _ident("Value"),
            _SEMICOLON,
        ]

    def test_two_declarations_emitted(self) -> None:
        """Two top-level declarations: UnionDecl then TypedefDecl."""
        prog = parse(self._tokens())
        assert len(prog.declarations) == 2

    def test_first_is_union_decl(self) -> None:
        """First declaration is a UnionDecl."""
        prog = parse(self._tokens())
        assert isinstance(prog.declarations[0], UnionDecl)

    def test_typedef_aliased_type_is_union(self) -> None:
        """TypedefDecl.aliased_type is the Union whose tag matches the UnionDecl."""
        prog = parse(self._tokens())
        ud = prog.declarations[0]
        td = prog.declarations[1]
        assert isinstance(ud, UnionDecl)
        assert isinstance(td, TypedefDecl)
        assert isinstance(td.aliased_type, Union)
        assert td.aliased_type.tag == ud.union_declaration.tag


# ---------------------------------------------------------------------------
# Named union: typedef union Value { int i; float f; } Value;
# ---------------------------------------------------------------------------


class TestTypedefNamedUnion:
    """``typedef union Tag { ... } Alias;`` emits [UnionDecl, TypedefDecl]."""

    def _tokens(self) -> list[object]:
        _FLOAT = _tok(TokenType.FLOAT, "float")
        return [
            _TYPEDEF,
            _UNION,
            _ident("Value"),
            _LBRACE,
            _INT,
            _ident("i"),
            _SEMICOLON,
            _FLOAT,
            _ident("f"),
            _SEMICOLON,
            _RBRACE,
            _ident("Value"),
            _SEMICOLON,
        ]

    def test_two_declarations_emitted(self) -> None:
        prog = parse(self._tokens())
        assert len(prog.declarations) == 2

    def test_union_tag_correct(self) -> None:
        prog = parse(self._tokens())
        ud = prog.declarations[0]
        assert isinstance(ud, UnionDecl)
        assert ud.union_declaration.tag == "Value"

    def test_typedef_aliased_type_uses_explicit_tag(self) -> None:
        prog = parse(self._tokens())
        td = prog.declarations[1]
        assert isinstance(td, TypedefDecl)
        assert td.aliased_type == Union(tag="Value")


# ---------------------------------------------------------------------------
# Regression: typedef struct Tag *Alias; still works (no body)
# ---------------------------------------------------------------------------


class TestTypedefStructPointerRegression:
    """``typedef struct Tag *Alias;`` — pointer-to-forward-declared struct — still parses."""

    def test_pointer_typedef_still_works(self) -> None:
        tokens = [
            _TYPEDEF,
            _STRUCT,
            _ident("Node"),
            _ASTERISK,
            _ident("NodePtr"),
            _SEMICOLON,
        ]
        prog = parse(tokens)
        assert len(prog.declarations) == 1
        td = prog.declarations[0]
        assert isinstance(td, TypedefDecl)
        assert td.name == "NodePtr"

    def test_simple_typedef_still_works(self) -> None:
        """``typedef int myint;`` is unaffected."""
        tokens = [
            _TYPEDEF,
            _INT,
            _ident("myint"),
            _SEMICOLON,
        ]
        prog = parse(tokens)
        assert len(prog.declarations) == 1
        td = prog.declarations[0]
        assert isinstance(td, TypedefDecl)
        assert td.aliased_type == IntType()
