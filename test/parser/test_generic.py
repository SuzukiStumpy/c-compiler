"""Parser tests for ``_Generic`` selection expressions (C11 §6.5.1.1)."""

import pytest

import compiler.syntax.ast as ast
from compiler.syntax.ast import (
    Constant,
    ConstInt,
    DoubleType,
    FunDecl,
    Generic,
    IntType,
    LongType,
    Pointer,
    S,
    Var,
    VoidType,
)
from compiler.syntax.ast.statements import Return
from compiler.types.tokens import TokenType
from compiler.utils.exceptions import ParseError
from test.parser.helpers import (
    _COLON,
    _COMMA,
    _INT,
    _LONG,
    _LPAREN,
    _RETURN,
    _RPAREN,
    _SEMICOLON,
    _VOID,
    _ident,
    _number,
    _tok,
    parse,
)

_LBRACE = _tok(TokenType.LBRACE, "{")
_RBRACE = _tok(TokenType.RBRACE, "}")
_DOUBLE = _tok(TokenType.DOUBLE, "double")
_DEFAULT = _tok(TokenType.DEFAULT, "default")
_GENERIC = _tok(TokenType.GENERIC, "_Generic")
_ASTERISK = _tok(TokenType.ASTERISK, "*")


def _parse_return_exp(*exp_tokens: object) -> ast.Exp:
    """Parse ``int main(void) { return <exp>; }`` and return the return expression.

    Args:
        *exp_tokens: Token stream for the expression following ``return``.

    Returns:
        The expression node from the parsed ``Return`` statement.
    """
    tokens = [
        _INT,
        _ident("main"),
        _LPAREN,
        _VOID,
        _RPAREN,
        _LBRACE,
        _RETURN,
        *exp_tokens,
        _SEMICOLON,
        _RBRACE,
    ]
    program = parse(list(tokens))
    decl = program.declarations[0]
    assert isinstance(decl, FunDecl)
    body = decl.function_declaration.body
    assert body is not None
    stmt = body[0]
    assert isinstance(stmt, S)
    assert isinstance(stmt.statement, Return)
    exp = stmt.statement.exp
    assert exp is not None
    return exp


class TestGenericParsing:
    """``_Generic(ctrl, type: expr, ..., default: expr)`` parses to a ``Generic`` node."""

    def test_generic_single_type_and_default(self) -> None:
        """``_Generic(x, int: 1, default: 0)`` → ``Generic`` with two associations."""
        exp = _parse_return_exp(
            _GENERIC,
            _LPAREN,
            _ident("x"),
            _COMMA,
            _INT,
            _COLON,
            _number("1"),
            _COMMA,
            _DEFAULT,
            _COLON,
            _number("0"),
            _RPAREN,
        )
        assert isinstance(exp, Generic)
        assert isinstance(exp.controlling_exp, Var)
        assert exp.controlling_exp.name == "x"
        assert len(exp.associations) == 2
        int_type, int_exp = exp.associations[0]
        assert int_type == IntType()
        assert isinstance(int_exp, Constant) and int_exp.value == ConstInt(1)
        default_type, default_exp = exp.associations[1]
        assert default_type is None
        assert isinstance(default_exp, Constant) and default_exp.value == ConstInt(0)

    def test_generic_multiple_types(self) -> None:
        """``_Generic(x, int: 1, double: 2, long: 3)`` → three type associations."""
        exp = _parse_return_exp(
            _GENERIC,
            _LPAREN,
            _ident("x"),
            _COMMA,
            _INT,
            _COLON,
            _number("1"),
            _COMMA,
            _DOUBLE,
            _COLON,
            _number("2"),
            _COMMA,
            _LONG,
            _COLON,
            _number("3"),
            _RPAREN,
        )
        assert isinstance(exp, Generic)
        assert len(exp.associations) == 3
        types = [t for t, _ in exp.associations]
        assert IntType() in types
        assert DoubleType() in types
        assert LongType() in types

    def test_generic_pointer_type_association(self) -> None:
        """``_Generic(p, int *: 1, default: 0)`` → association with ``Pointer(IntType())``."""
        exp = _parse_return_exp(
            _GENERIC,
            _LPAREN,
            _ident("p"),
            _COMMA,
            _INT,
            _ASTERISK,
            _COLON,
            _number("1"),
            _COMMA,
            _DEFAULT,
            _COLON,
            _number("0"),
            _RPAREN,
        )
        assert isinstance(exp, Generic)
        ptr_assoc = next((t, e) for t, e in exp.associations if t is not None)
        assert ptr_assoc[0] == Pointer(IntType())

    def test_generic_no_default(self) -> None:
        """``_Generic(x, int: 1)`` → single association, no default."""
        exp = _parse_return_exp(
            _GENERIC,
            _LPAREN,
            _ident("x"),
            _COMMA,
            _INT,
            _COLON,
            _number("1"),
            _RPAREN,
        )
        assert isinstance(exp, Generic)
        assert len(exp.associations) == 1
        assoc_type, _ = exp.associations[0]
        assert assoc_type == IntType()

    def test_generic_controlling_exp_is_constant(self) -> None:
        """Controlling expression can be any expression, including a constant."""
        exp = _parse_return_exp(
            _GENERIC,
            _LPAREN,
            _number("0"),
            _COMMA,
            _INT,
            _COLON,
            _number("1"),
            _COMMA,
            _DEFAULT,
            _COLON,
            _number("0"),
            _RPAREN,
        )
        assert isinstance(exp, Generic)
        assert isinstance(exp.controlling_exp, Constant)

    def test_generic_void_pointer_association(self) -> None:
        """``_Generic(p, void *: 1, default: 0)`` → ``Pointer(VoidType())`` association."""
        _VOID_KW = _tok(TokenType.VOID, "void")
        exp = _parse_return_exp(
            _GENERIC,
            _LPAREN,
            _ident("p"),
            _COMMA,
            _VOID_KW,
            _ASTERISK,
            _COLON,
            _number("1"),
            _COMMA,
            _DEFAULT,
            _COLON,
            _number("0"),
            _RPAREN,
        )
        assert isinstance(exp, Generic)
        void_ptr_assoc = next((t, e) for t, e in exp.associations if t == Pointer(VoidType()))
        assert void_ptr_assoc is not None

    def test_generic_default_only_raises_parse_error(self) -> None:
        """``_Generic(x)`` with no associations raises ``ParseError``."""
        with pytest.raises(ParseError):
            _parse_return_exp(
                _GENERIC,
                _LPAREN,
                _ident("x"),
                _RPAREN,
            )
