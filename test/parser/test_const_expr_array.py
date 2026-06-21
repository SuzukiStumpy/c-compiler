"""Parser tests for constant-expression array dimensions.

C11 §6.7.6 — array declarators may use any integer constant expression as the
dimension, not just plain integer literals.  The parser must now produce an
``ast.Exp`` node for the size rather than a bare ``int``, leaving evaluation to
semantic analysis.
"""

import compiler.syntax.ast as ast
from compiler.syntax.ast.expressions import Binary, Constant, SizeOfT
from compiler.syntax.ast.operators import Add, Multiply, Subtract
from compiler.types.tokens import TokenType
from test.parser.helpers import (
    _ASTERISK,
    _INT,
    _LBRACE,
    _LBRACKET,
    _LPAREN,
    _MINUS,
    _PLUS,
    _RBRACE,
    _RBRACKET,
    _RPAREN,
    _SEMICOLON,
    _ident,
    _number,
    _tok,
    parse,
)

_CHAR = _tok(TokenType.CHAR, "char")
_VOID = _tok(TokenType.VOID, "void")
_SIZEOF = _tok(TokenType.SIZEOF, "sizeof")
_STRUCT = _tok(TokenType.STRUCT, "struct")


def _file_var_tokens(*type_and_decl: object) -> list:
    """Build a file-scope ``int``-prefixed variable declaration token list ending with ``;``."""
    return list(type_and_decl) + [_SEMICOLON]


class TestArithmeticExpressionArrayDimension:
    """Array dimensions may be arithmetic constant expressions."""

    def test_addition_in_array_dimension(self) -> None:
        """``int a[2 + 3];`` produces Array with a Binary(Add, ...) size expression."""
        tokens = _file_var_tokens(_INT, _ident("a"), _LBRACKET, _number("2"), _PLUS, _number("3"), _RBRACKET)
        prog = parse(tokens)
        decl = prog.declarations[0]
        assert isinstance(decl, ast.VarDecl)
        t = decl.variable_declaration.var_type
        assert isinstance(t, ast.Array)
        assert isinstance(t.size, Binary)
        assert isinstance(t.size.operator, Add)

    def test_multiplication_in_array_dimension(self) -> None:
        """``int a[3 * 4];`` produces Array with Binary(Multiply, ...) size."""
        tokens = _file_var_tokens(_INT, _ident("a"), _LBRACKET, _number("3"), _ASTERISK, _number("4"), _RBRACKET)
        prog = parse(tokens)
        t = prog.declarations[0].variable_declaration.var_type
        assert isinstance(t, ast.Array)
        assert isinstance(t.size, Binary)
        assert isinstance(t.size.operator, Multiply)

    def test_plain_integer_literal_becomes_constant_expression(self) -> None:
        """``int a[5];`` produces Array with Constant(...) size (not bare int)."""
        tokens = _file_var_tokens(_INT, _ident("a"), _LBRACKET, _number("5"), _RBRACKET)
        prog = parse(tokens)
        t = prog.declarations[0].variable_declaration.var_type
        assert isinstance(t, ast.Array)
        assert isinstance(t.size, Constant)


class TestSizeofInArrayDimension:
    """``sizeof`` expressions are valid constant-expression array dimensions."""

    def test_sizeof_type_in_array_dimension(self) -> None:
        """``int a[sizeof(int)];`` produces Array with SizeOfT(IntType()) size."""
        tokens = _file_var_tokens(_INT, _ident("a"), _LBRACKET, _SIZEOF, _LPAREN, _INT, _RPAREN, _RBRACKET)
        prog = parse(tokens)
        t = prog.declarations[0].variable_declaration.var_type
        assert isinstance(t, ast.Array)
        assert isinstance(t.size, SizeOfT)
        assert t.size.t == ast.IntType()

    def test_complex_sizeof_expression(self) -> None:
        """``char a[12 * sizeof(int) - 5 * sizeof(void *)];`` parses as nested Binary."""
        # 12 * sizeof(int)  -  5 * sizeof(void *)
        tokens = _file_var_tokens(
            _CHAR,
            _ident("a"),
            _LBRACKET,
            _number("12"),
            _ASTERISK,
            _SIZEOF,
            _LPAREN,
            _INT,
            _RPAREN,
            _MINUS,
            _number("5"),
            _ASTERISK,
            _SIZEOF,
            _LPAREN,
            _tok(TokenType.VOID, "void"),
            _ASTERISK,
            _RPAREN,
            _RBRACKET,
        )
        prog = parse(tokens)
        t = prog.declarations[0].variable_declaration.var_type
        assert isinstance(t, ast.Array)
        assert isinstance(t.size, Binary)
        assert isinstance(t.size.operator, Subtract)

    def test_sizeof_in_struct_member_array(self) -> None:
        """``struct S { char buf[sizeof(int)]; };`` parses member as Array with SizeOfT."""
        tokens = [
            _STRUCT,
            _ident("S"),
            _LBRACE,
            _CHAR,
            _ident("buf"),
            _LBRACKET,
            _SIZEOF,
            _LPAREN,
            _INT,
            _RPAREN,
            _RBRACKET,
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
        ]
        prog = parse(tokens)
        decl = prog.declarations[0]
        assert isinstance(decl, ast.StructDecl)
        member = decl.struct_declaration.members[0]
        assert isinstance(member.member_type, ast.Array)
        assert isinstance(member.member_type.size, SizeOfT)


class TestFlexibleArrayUnchanged:
    """Empty ``[]`` still produces ``Array(T, size=None)``."""

    def test_empty_brackets_give_none_size(self) -> None:
        """``int a[];`` produces Array with size=None."""
        tokens = _file_var_tokens(_INT, _ident("a"), _LBRACKET, _RBRACKET)
        prog = parse(tokens)
        t = prog.declarations[0].variable_declaration.var_type
        assert isinstance(t, ast.Array)
        assert t.size is None
