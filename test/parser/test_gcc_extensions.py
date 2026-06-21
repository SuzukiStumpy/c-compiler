"""Parser tests for GCC extension syntax handled by the compiler.

Covers:
  1. ``__restrict`` / ``__restrict__`` as type qualifiers in function parameters
     and pointer declarators.
  2. ``__asm__("..." "...")`` symbol-renaming suffix on function declarations,
     including combinations with ``__attribute__``.
  3. Qualified typedef names in ``_parse_type()`` — ``const my_typedef_t *p``
     where the qualifier precedes the typedef-name identifier.
"""

from compiler.parser import parse
from compiler.syntax.ast import (
    D,
    FunDecl,
    Parameter,
    Pointer,
    VarDecl,
)
from compiler.syntax.ast.types import QualifiedType, Structure
from compiler.types.tokens import TokenType
from test.parser.helpers import (
    _ASTERISK,
    _CHAR,
    _CONST,
    _EXTERN,
    _INT,
    _LPAREN,
    _RPAREN,
    _SEMICOLON,
    _VOID,
    _ident,
    _tok,
)

_RESTRICT = _tok(TokenType.RESTRICT, "restrict")
_RESTRICT_GCC = _tok(TokenType.RESTRICT, "__restrict")
_RESTRICT_GCC2 = _tok(TokenType.RESTRICT, "__restrict__")
_TYPEDEF = _tok(TokenType.TYPEDEF, "typedef")
_ASM = _tok(TokenType.ASM, "__asm__")
_STRING = _tok(TokenType.STRING, '"scanf"')
_STRUCT = _tok(TokenType.STRUCT, "struct")
_LBRACE = _tok(TokenType.LBRACE, "{")
_RBRACE = _tok(TokenType.RBRACE, "}")
_ATTRIBUTE = _tok(TokenType.ATTRIBUTE, "__attribute__")
_ELLIPSIS = _tok(TokenType.ELLIPSIS, "...")
_COMMA = _tok(TokenType.COMMA, ",")


def _asm_rename(*string_toks: object) -> list[object]:
    """Produce ``__asm__( string_toks )`` token list."""
    return [_ASM, _tok(TokenType.LPAREN, "("), *string_toks, _tok(TokenType.RPAREN, ")")]


def _attr(*content_toks: object) -> list[object]:
    """Produce ``__attribute__(( content ))`` token list."""
    lp = _tok(TokenType.LPAREN, "(")
    rp = _tok(TokenType.RPAREN, ")")
    return [_ATTRIBUTE, lp, lp, *content_toks, rp, rp]


# ---------------------------------------------------------------------------
# __restrict / __restrict__ in parameter declarations
# ---------------------------------------------------------------------------


class TestRestrictGCCExtension:
    """``__restrict`` and ``__restrict__`` are treated as type qualifiers."""

    def test_dunder_restrict_on_pointer_param(self) -> None:
        """``char * __restrict p`` in a parameter list parses without error."""
        tokens = [
            _EXTERN,
            _INT,
            _ident("memcpy"),
            _LPAREN,
            _CHAR,
            _ASTERISK,
            _RESTRICT_GCC,
            _ident("dst"),
            _COMMA,
            _CHAR,
            _ASTERISK,
            _RESTRICT_GCC,
            _ident("src"),
            _RPAREN,
            _SEMICOLON,
        ]
        prog = parse(tokens)
        assert len(prog.declarations) == 1
        decl = prog.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.name == "memcpy"
        assert len(decl.function_declaration.params) == 2

    def test_dunder_restrict_dunder_on_pointer_param(self) -> None:
        """``char * __restrict__ p`` in a parameter list parses without error."""
        tokens = [
            _EXTERN,
            _INT,
            _ident("foo"),
            _LPAREN,
            _CHAR,
            _ASTERISK,
            _RESTRICT_GCC2,
            _ident("p"),
            _RPAREN,
            _SEMICOLON,
        ]
        prog = parse(tokens)
        decl = prog.declarations[0]
        assert isinstance(decl, FunDecl)
        assert len(decl.function_declaration.params) == 1

    def test_all_three_restrict_spellings_equivalent(self) -> None:
        """``restrict``, ``__restrict``, and ``__restrict__`` each produce a RESTRICT-qualified pointer."""
        for restrict_tok in (_RESTRICT, _RESTRICT_GCC, _RESTRICT_GCC2):
            tokens = [
                _INT,
                _ident("f"),
                _LPAREN,
                _INT,
                _ASTERISK,
                restrict_tok,
                _ident("p"),
                _RPAREN,
                _SEMICOLON,
            ]
            prog = parse(tokens)
            decl = prog.declarations[0]
            assert isinstance(decl, FunDecl)
            param = decl.function_declaration.params[0]
            assert isinstance(param, Parameter)
            ptr = param.param_type
            assert isinstance(ptr, QualifiedType)
            assert ptr.restrict is True


# ---------------------------------------------------------------------------
# __asm__("symbol") after function declarations
# ---------------------------------------------------------------------------


class TestAsmSymbolRename:
    """``__asm__("alias")`` suffix is silently skipped on function declarations."""

    def test_simple_asm_after_function_decl(self) -> None:
        """``extern int foo(void) __asm__("bar");`` parses as a FunDecl."""
        tokens = [
            _EXTERN,
            _INT,
            _ident("foo"),
            _LPAREN,
            _VOID,
            _RPAREN,
            *_asm_rename(_STRING),
            _SEMICOLON,
        ]
        prog = parse(tokens)
        assert len(prog.declarations) == 1
        decl = prog.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.name == "foo"

    def test_asm_with_two_string_args(self) -> None:
        """``__asm__("" "isoc23_scanf")`` (two concatenated strings) is skipped."""
        string2 = _tok(TokenType.STRING, '"isoc23_scanf"')
        tokens = [
            _EXTERN,
            _INT,
            _ident("scanf"),
            _LPAREN,
            _CHAR,
            _ASTERISK,
            _RESTRICT_GCC,
            _ident("fmt"),
            _COMMA,
            _ELLIPSIS,
            _RPAREN,
            *_asm_rename(_STRING, string2),
            _SEMICOLON,
        ]
        prog = parse(tokens)
        decl = prog.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.name == "scanf"

    def test_asm_followed_by_attribute(self) -> None:
        """``__asm__(…) __attribute__((nothrow))`` — both suffixes are skipped."""
        nothrow_attr = _attr(_ident("__nothrow__"))
        tokens = [
            _EXTERN,
            _INT,
            _ident("foo"),
            _LPAREN,
            _VOID,
            _RPAREN,
            *_asm_rename(_STRING),
            *nothrow_attr,
            _SEMICOLON,
        ]
        prog = parse(tokens)
        assert isinstance(prog.declarations[0], FunDecl)

    def test_attribute_followed_by_asm(self) -> None:
        """``__attribute__((nothrow)) __asm__(…)`` — order reversed, both skipped."""
        nothrow_attr = _attr(_ident("__nothrow__"))
        tokens = [
            _EXTERN,
            _INT,
            _ident("bar"),
            _LPAREN,
            _VOID,
            _RPAREN,
            *nothrow_attr,
            *_asm_rename(_STRING),
            _SEMICOLON,
        ]
        prog = parse(tokens)
        assert isinstance(prog.declarations[0], FunDecl)


# ---------------------------------------------------------------------------
# const <typedef_name> in _parse_type()
# ---------------------------------------------------------------------------


class TestQualifiedTypedefName:
    """A qualifier (const/volatile/restrict) before a typedef name in _parse_type()."""

    def _typedef_struct(self, alias: str) -> list[object]:
        """Token list for ``typedef struct Tag { int x; } alias;``."""
        return [
            _TYPEDEF,
            _STRUCT,
            _ident(f"_{alias}_tag"),
            _LBRACE,
            _INT,
            _ident("x"),
            _SEMICOLON,
            _RBRACE,
            _ident(alias),
            _SEMICOLON,
        ]

    def test_const_typedef_in_function_param(self) -> None:
        """``const fpos_t *p`` as a parameter type resolves the typedef and applies const."""
        tokens = [
            *self._typedef_struct("fpos_t"),
            _EXTERN,
            _INT,
            _ident("fsetpos"),
            _LPAREN,
            _CONST,
            _ident("fpos_t"),
            _ASTERISK,
            _ident("pos"),
            _RPAREN,
            _SEMICOLON,
        ]
        prog = parse(tokens)
        decls = prog.declarations
        assert len(decls) == 3  # StructDecl + TypedefDecl + FunDecl
        fun = decls[2]
        assert isinstance(fun, FunDecl)
        assert fun.function_declaration.name == "fsetpos"
        param = fun.function_declaration.params[0]
        ptr = param.param_type
        assert isinstance(ptr, Pointer)
        assert isinstance(ptr.referenced, QualifiedType)
        assert ptr.referenced.const is True
        assert isinstance(ptr.referenced.base, Structure)

    def test_const_typedef_unnamed_param(self) -> None:
        """``const mytype_t *`` as an abstract parameter (no name) parses correctly."""
        tokens = [
            *self._typedef_struct("mytype_t"),
            _EXTERN,
            _INT,
            _ident("foo"),
            _LPAREN,
            _CONST,
            _ident("mytype_t"),
            _ASTERISK,
            _RPAREN,
            _SEMICOLON,
        ]
        prog = parse(tokens)
        fun = prog.declarations[2]
        assert isinstance(fun, FunDecl)
        param = fun.function_declaration.params[0]
        assert isinstance(param.param_type, Pointer)

    def test_const_typedef_in_var_decl(self) -> None:
        """``const myint_t x`` at block level resolves typedef and applies const."""
        _TYPEDEF_INT = [_TYPEDEF, _INT, _ident("myint_t"), _SEMICOLON]
        _LBRACE = _tok(TokenType.LBRACE, "{")
        _RBRACE = _tok(TokenType.RBRACE, "}")
        _RETURN = _tok(TokenType.RETURN, "return")
        _ZERO = _tok(TokenType.NUMBER, "0")
        tokens = [
            *_TYPEDEF_INT,
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _CONST,
            _ident("myint_t"),
            _ident("x"),
            _SEMICOLON,
            _RETURN,
            _ZERO,
            _SEMICOLON,
            _RBRACE,
        ]
        prog = parse(tokens)
        fun = prog.declarations[1]
        assert isinstance(fun, FunDecl)
        body = fun.function_declaration.body
        assert body is not None
        first = body[0]
        assert isinstance(first, D)
        assert isinstance(first.declaration, VarDecl)
        vt = first.declaration.variable_declaration.var_type
        assert isinstance(vt, QualifiedType)
        assert vt.const is True
