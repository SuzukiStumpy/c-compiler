"""Parser tests for compound literal expressions: ``(Type){initializer}``."""

from compiler.types.tokens import TokenType
from test.parser.helpers import (
    _ASSIGNMENT,
    _COMMA,
    _LBRACE,
    _LBRACKET,
    _LPAREN,
    _RBRACE,
    _RBRACKET,
    _RPAREN,
    _SEMICOLON,
    CompoundInit,
    IntType,
    SingleInit,
    _ident,
    _number,
    _parsed_body,
    _parsed_exp,
    _tok,
)

_STRUCT = _tok(TokenType.STRUCT, "struct")
_COLON = _tok(TokenType.COLON, ":")


class TestScalarCompoundLiteral:
    """``(scalar-type){value}`` parses to a ``CompoundLiteral`` node."""

    def test_int_compound_literal(self) -> None:
        """``(int){42}`` parses to CompoundLiteral with a CompoundInit containing one SingleInit."""
        from compiler.syntax.ast import Constant, ConstInt
        from compiler.syntax.ast.expressions import CompoundLiteral

        exp = _parsed_exp(_LPAREN, _tok(TokenType.INT, "int"), _RPAREN, _LBRACE, _number("42"), _RBRACE)
        assert isinstance(exp, CompoundLiteral)
        assert exp.target_type == IntType()
        assert isinstance(exp.init, CompoundInit)
        assert len(exp.init.initializers) == 1
        assert isinstance(exp.init.initializers[0], SingleInit)
        assert exp.init.initializers[0].exp == Constant(ConstInt(42))

    def test_int_compound_literal_is_distinct_from_cast(self) -> None:
        """``(int){1}`` is a CompoundLiteral, not a Cast followed by Constant."""
        from compiler.syntax.ast import Cast
        from compiler.syntax.ast.expressions import CompoundLiteral

        exp = _parsed_exp(_LPAREN, _tok(TokenType.INT, "int"), _RPAREN, _LBRACE, _number("1"), _RBRACE)
        assert isinstance(exp, CompoundLiteral)
        assert not isinstance(exp, Cast)


class TestStructCompoundLiteral:
    """``(struct Tag){...}`` parses with a compound initializer."""

    def test_struct_compound_literal_with_positional_members(self) -> None:
        """``(struct Point){1, 2}`` parses to CompoundLiteral with CompoundInit."""
        from compiler.syntax.ast.expressions import CompoundLiteral

        exp = _parsed_exp(
            _LPAREN,
            _STRUCT,
            _ident("Point"),
            _RPAREN,
            _LBRACE,
            _number("1"),
            _COMMA,
            _number("2"),
            _RBRACE,
        )
        assert isinstance(exp, CompoundLiteral)
        assert isinstance(exp.init, CompoundInit)


class TestArrayCompoundLiteral:
    """``(type[]){...}`` and ``(type[N]){...}`` compound literals."""

    def test_array_compound_literal(self) -> None:
        """``(int[3]){1, 2, 3}`` parses with an array target type and compound init."""
        from compiler.syntax.ast import Array
        from compiler.syntax.ast.expressions import CompoundLiteral

        exp = _parsed_exp(
            _LPAREN,
            _tok(TokenType.INT, "int"),
            _LBRACKET,
            _number("3"),
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
        assert exp.target_type.size == 3
        assert isinstance(exp.init, CompoundInit)


class TestCompoundLiteralInStatement:
    """Compound literals can appear in initializers and assignments."""

    def test_compound_literal_in_variable_init(self) -> None:
        """``int x = (int){5};`` parses correctly."""
        from compiler.syntax.ast import Constant, ConstInt, D, SingleInit, VarDecl
        from compiler.syntax.ast.expressions import CompoundLiteral

        body = _parsed_body(
            _tok(TokenType.INT, "int"),
            _ident("x"),
            _ASSIGNMENT,
            _LPAREN,
            _tok(TokenType.INT, "int"),
            _RPAREN,
            _LBRACE,
            _number("5"),
            _RBRACE,
            _SEMICOLON,
        )
        item = body[0]
        assert isinstance(item, D)
        assert isinstance(item.declaration, VarDecl)
        init = item.declaration.variable_declaration.init
        assert isinstance(init, SingleInit)
        assert isinstance(init.exp, CompoundLiteral)
        assert isinstance(init.exp.init, CompoundInit)
        assert len(init.exp.init.initializers) == 1
        assert init.exp.init.initializers[0].exp == Constant(ConstInt(5))
