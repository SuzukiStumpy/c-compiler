"""Parser tests for C11 designated initialisers (struct member and array index designators)."""

from compiler.parser import parse
from compiler.syntax.ast import (
    CompoundInit,
    Constant,
    ConstInt,
    D,
    FunDecl,
    IndexDesignatorInit,
    Initializer,
    MemberDesignatorInit,
    SingleInit,
    VarDecl,
)
from compiler.types.tokens import Token, TokenType
from test.parser.helpers import (
    _ASSIGNMENT,
    _INT,
    _LBRACE,
    _LBRACKET,
    _LPAREN,
    _RBRACE,
    _RBRACKET,
    _RPAREN,
    _SEMICOLON,
    _VOID,
    _ident,
    _number,
    _tok,
)

_COMMA = _tok(TokenType.COMMA, ",")
_PERIOD = _tok(TokenType.PERIOD, ".")
_STRUCT = _tok(TokenType.STRUCT, "struct")


def _parse_init_in_decl(*tokens: Token) -> Initializer:
    """Parse ``int main(void){ struct S s = <tokens>; }`` and return the init node.

    Args:
        *tokens: The tokens forming the initialiser (including surrounding braces).

    Returns:
        The ``Initializer`` node on the ``VariableDeclaration``.
    """
    prog = parse(
        [
            _STRUCT,
            _ident("S"),
            _LBRACE,
            _INT,
            _ident("x"),
            _SEMICOLON,
            _INT,
            _ident("y"),
            _SEMICOLON,
            _RBRACE,
            _SEMICOLON,
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _STRUCT,
            _ident("S"),
            _ident("s"),
            _ASSIGNMENT,
            *tokens,
            _SEMICOLON,
            _RBRACE,
        ]
    )
    fn = next(d.function_declaration for d in prog.declarations if isinstance(d, FunDecl))
    assert fn.body is not None
    decl_item = fn.body[0]
    assert isinstance(decl_item, D)
    vd = decl_item.declaration
    assert isinstance(vd, VarDecl)
    init = vd.variable_declaration.init
    assert init is not None
    return init


def _parse_array_init(*tokens: Token) -> Initializer:
    """Parse ``int main(void){ int arr[5] = <tokens>; }`` and return the init node.

    Args:
        *tokens: The tokens forming the initialiser.

    Returns:
        The ``Initializer`` node.
    """
    prog = parse(
        [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _INT,
            _ident("arr"),
            _LBRACKET,
            _number("5"),
            _RBRACKET,
            _ASSIGNMENT,
            *tokens,
            _SEMICOLON,
            _RBRACE,
        ]
    )
    fn = next(d.function_declaration for d in prog.declarations if isinstance(d, FunDecl))
    assert fn.body is not None
    decl_item = fn.body[0]
    assert isinstance(decl_item, D)
    vd = decl_item.declaration
    assert isinstance(vd, VarDecl)
    init = vd.variable_declaration.init
    assert init is not None
    return init


class TestMemberDesignatorInit:
    """Parser produces ``MemberDesignatorInit`` nodes for ``.member = value`` syntax."""

    def test_single_member_designator(self) -> None:
        """``{.x = 1}`` produces a ``CompoundInit`` containing a ``MemberDesignatorInit``."""
        init = _parse_init_in_decl(
            _LBRACE,
            _PERIOD,
            _ident("x"),
            _ASSIGNMENT,
            _number("1"),
            _RBRACE,
        )
        assert isinstance(init, CompoundInit)
        assert len(init.initializers) == 1
        item = init.initializers[0]
        assert isinstance(item, MemberDesignatorInit)
        assert item.member == "x"
        assert isinstance(item.init, SingleInit)
        assert item.init.exp == Constant(ConstInt(1))

    def test_two_member_designators(self) -> None:
        """``{.x = 1, .y = 2}`` produces two ``MemberDesignatorInit`` nodes."""
        init = _parse_init_in_decl(
            _LBRACE,
            _PERIOD,
            _ident("x"),
            _ASSIGNMENT,
            _number("1"),
            _COMMA,
            _PERIOD,
            _ident("y"),
            _ASSIGNMENT,
            _number("2"),
            _RBRACE,
        )
        assert isinstance(init, CompoundInit)
        assert len(init.initializers) == 2
        first, second = init.initializers
        assert isinstance(first, MemberDesignatorInit)
        assert first.member == "x"
        assert isinstance(second, MemberDesignatorInit)
        assert second.member == "y"

    def test_mixed_positional_and_designated(self) -> None:
        """``{1, .y = 2}`` mixes a positional ``SingleInit`` with a ``MemberDesignatorInit``."""
        init = _parse_init_in_decl(
            _LBRACE,
            _number("1"),
            _COMMA,
            _PERIOD,
            _ident("y"),
            _ASSIGNMENT,
            _number("2"),
            _RBRACE,
        )
        assert isinstance(init, CompoundInit)
        assert len(init.initializers) == 2
        first, second = init.initializers
        assert isinstance(first, SingleInit)
        assert isinstance(second, MemberDesignatorInit)
        assert second.member == "y"

    def test_designated_with_nested_compound(self) -> None:
        """``{.x = {1, 2}}`` wraps a nested ``CompoundInit`` inside a ``MemberDesignatorInit``."""
        init = _parse_init_in_decl(
            _LBRACE,
            _PERIOD,
            _ident("x"),
            _ASSIGNMENT,
            _LBRACE,
            _number("1"),
            _COMMA,
            _number("2"),
            _RBRACE,
            _RBRACE,
        )
        assert isinstance(init, CompoundInit)
        assert len(init.initializers) == 1
        item = init.initializers[0]
        assert isinstance(item, MemberDesignatorInit)
        assert item.member == "x"
        assert isinstance(item.init, CompoundInit)

    def test_trailing_comma_after_designator(self) -> None:
        """``{.x = 1,}`` with trailing comma is accepted."""
        init = _parse_init_in_decl(
            _LBRACE,
            _PERIOD,
            _ident("x"),
            _ASSIGNMENT,
            _number("1"),
            _COMMA,
            _RBRACE,
        )
        assert isinstance(init, CompoundInit)
        assert len(init.initializers) == 1


class TestIndexDesignatorInit:
    """Parser produces ``IndexDesignatorInit`` nodes for ``[index] = value`` syntax."""

    def test_single_index_designator(self) -> None:
        """``{[2] = 7}`` produces a ``CompoundInit`` with one ``IndexDesignatorInit``."""
        init = _parse_array_init(
            _LBRACE,
            _LBRACKET,
            _number("2"),
            _RBRACKET,
            _ASSIGNMENT,
            _number("7"),
            _RBRACE,
        )
        assert isinstance(init, CompoundInit)
        assert len(init.initializers) == 1
        item = init.initializers[0]
        assert isinstance(item, IndexDesignatorInit)
        assert item.index == Constant(ConstInt(2))
        assert isinstance(item.init, SingleInit)
        assert item.init.exp == Constant(ConstInt(7))

    def test_two_index_designators(self) -> None:
        """``{[0] = 1, [4] = 5}`` produces two ``IndexDesignatorInit`` nodes."""
        init = _parse_array_init(
            _LBRACE,
            _LBRACKET,
            _number("0"),
            _RBRACKET,
            _ASSIGNMENT,
            _number("1"),
            _COMMA,
            _LBRACKET,
            _number("4"),
            _RBRACKET,
            _ASSIGNMENT,
            _number("5"),
            _RBRACE,
        )
        assert isinstance(init, CompoundInit)
        assert len(init.initializers) == 2
        first, second = init.initializers
        assert isinstance(first, IndexDesignatorInit)
        assert first.index == Constant(ConstInt(0))
        assert isinstance(second, IndexDesignatorInit)
        assert second.index == Constant(ConstInt(4))

    def test_mixed_positional_and_index_designator(self) -> None:
        """``{1, [3] = 9}`` mixes a positional element with an ``IndexDesignatorInit``."""
        init = _parse_array_init(
            _LBRACE,
            _number("1"),
            _COMMA,
            _LBRACKET,
            _number("3"),
            _RBRACKET,
            _ASSIGNMENT,
            _number("9"),
            _RBRACE,
        )
        assert isinstance(init, CompoundInit)
        assert len(init.initializers) == 2
        first, second = init.initializers
        assert isinstance(first, SingleInit)
        assert isinstance(second, IndexDesignatorInit)
        assert second.index == Constant(ConstInt(3))
