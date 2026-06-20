"""Parser tests for GCC built-in expressions.

Covers ``__builtin_expect``, ``__builtin_unreachable``, and ``__builtin_offsetof``.
"""

import compiler.syntax.ast as ast
from compiler.syntax.ast import (
    BuiltinUnreachable,
    Constant,
    ConstInt,
    FunDecl,
    OffsetOf,
    S,
    Structure,
    Var,
)
from compiler.types.tokens import TokenType
from test.parser.helpers import (
    _COMMA,
    _INT,
    _LPAREN,
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
_RETURN = _tok(TokenType.RETURN, "return")
_STRUCT = _tok(TokenType.STRUCT, "struct")


def _simple_fn_body(*body_tokens: object) -> tuple:
    """Parse ``int main(void) { <body> }`` and return the body tuple."""
    tokens = [
        _INT,
        _ident("main"),
        _LPAREN,
        _VOID,
        _RPAREN,
        _LBRACE,
        *body_tokens,
        _RBRACE,
    ]
    program = parse(list(tokens))
    decl = program.declarations[0]
    assert isinstance(decl, FunDecl)
    body = decl.function_declaration.body
    assert body is not None
    return body


def _parse_return_exp(*exp_tokens: object) -> ast.Exp:
    """Parse ``int main(void) { return <exp>; }`` and return the return expression."""
    body = _simple_fn_body(_RETURN, *exp_tokens, _SEMICOLON)
    from compiler.syntax.ast.statements import Return

    stmt = body[0]
    assert isinstance(stmt, S)
    assert isinstance(stmt.statement, Return)
    return stmt.statement.exp


# ---------------------------------------------------------------------------
# __builtin_expect
# ---------------------------------------------------------------------------


class TestBuiltinExpect:
    """``__builtin_expect(expr, hint)`` is lowered to ``expr`` at parse time."""

    def test_expect_returns_inner_var(self) -> None:
        """``__builtin_expect(x, 1)`` parses as ``Var("x")``."""
        exp = _parse_return_exp(
            _ident("__builtin_expect"),
            _LPAREN,
            _ident("x"),
            _COMMA,
            _number("1"),
            _RPAREN,
        )
        assert isinstance(exp, Var)
        assert exp.name == "x"

    def test_expect_returns_inner_constant(self) -> None:
        """``__builtin_expect(0, 0)`` parses as ``Constant(ConstInt(0))``."""
        exp = _parse_return_exp(
            _ident("__builtin_expect"),
            _LPAREN,
            _number("0"),
            _COMMA,
            _number("0"),
            _RPAREN,
        )
        assert isinstance(exp, Constant)
        assert exp.value == ConstInt(0)

    def test_expect_hint_is_discarded(self) -> None:
        """The hint argument (second) is discarded; only the first arg is returned."""
        # __builtin_expect(x, some_other_var) -> Var("x"), not Var("some_other_var")
        exp = _parse_return_exp(
            _ident("__builtin_expect"),
            _LPAREN,
            _ident("x"),
            _COMMA,
            _ident("likely"),
            _RPAREN,
        )
        assert isinstance(exp, Var)
        assert exp.name == "x"


# ---------------------------------------------------------------------------
# __builtin_unreachable
# ---------------------------------------------------------------------------


class TestBuiltinUnreachable:
    """``__builtin_unreachable()`` is parsed as a ``BuiltinUnreachable`` node."""

    def test_unreachable_as_expression_statement(self) -> None:
        """``__builtin_unreachable();`` in a function body parses as ``BuiltinUnreachable``."""
        from compiler.syntax.ast.statements import Expression

        body = _simple_fn_body(
            _ident("__builtin_unreachable"),
            _LPAREN,
            _RPAREN,
            _SEMICOLON,
            _RETURN,
            _number("0"),
            _SEMICOLON,
        )
        stmt_item = body[0]
        assert isinstance(stmt_item, S)
        assert isinstance(stmt_item.statement, Expression)
        assert isinstance(stmt_item.statement.exp, BuiltinUnreachable)

    def test_unreachable_in_return(self) -> None:
        """``return __builtin_unreachable();`` parses; inner exp is ``BuiltinUnreachable``."""
        from compiler.syntax.ast.statements import Return

        body = _simple_fn_body(
            _RETURN,
            _ident("__builtin_unreachable"),
            _LPAREN,
            _RPAREN,
            _SEMICOLON,
        )
        stmt_item = body[0]
        assert isinstance(stmt_item, S)
        assert isinstance(stmt_item.statement, Return)
        assert isinstance(stmt_item.statement.exp, BuiltinUnreachable)


# ---------------------------------------------------------------------------
# __builtin_offsetof
# ---------------------------------------------------------------------------


class TestBuiltinOffsetof:
    """``__builtin_offsetof(type, member)`` is parsed as an ``OffsetOf`` node."""

    def test_offsetof_struct_tag_member(self) -> None:
        """``__builtin_offsetof(struct s, m)`` → ``OffsetOf(Structure("s"), "m")``."""
        exp = _parse_return_exp(
            _ident("__builtin_offsetof"),
            _LPAREN,
            _STRUCT,
            _ident("s"),
            _COMMA,
            _ident("m"),
            _RPAREN,
        )
        assert isinstance(exp, OffsetOf)
        assert isinstance(exp.t, Structure)
        assert exp.t.tag == "s"
        assert exp.member == "m"

    def test_offsetof_typedef_name(self) -> None:
        """``__builtin_offsetof(MyStruct, x)`` with typedef → ``OffsetOf`` with resolved type."""
        # typedef struct s MyStruct; then __builtin_offsetof(MyStruct, x)

        _TYPEDEF = _tok(TokenType.TYPEDEF, "typedef")
        tokens = [
            _TYPEDEF,
            _STRUCT,
            _ident("s"),
            _ident("MyStruct"),
            _SEMICOLON,
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _tok(TokenType.LBRACE, "{"),
            _RETURN,
            _ident("__builtin_offsetof"),
            _LPAREN,
            _ident("MyStruct"),
            _COMMA,
            _ident("x"),
            _RPAREN,
            _SEMICOLON,
            _tok(TokenType.RBRACE, "}"),
        ]
        program = parse(tokens)
        fn = program.declarations[1]
        assert isinstance(fn, FunDecl)
        body = fn.function_declaration.body
        assert body is not None
        from compiler.syntax.ast.statements import Return

        ret = body[0]
        assert isinstance(ret, S)
        assert isinstance(ret.statement, Return)
        exp = ret.statement.exp
        assert isinstance(exp, OffsetOf)
        assert isinstance(exp.t, Structure)
        assert exp.t.tag == "s"
        assert exp.member == "x"

    def test_offsetof_different_member_names(self) -> None:
        """``__builtin_offsetof`` preserves the member name exactly."""
        exp = _parse_return_exp(
            _ident("__builtin_offsetof"),
            _LPAREN,
            _STRUCT,
            _ident("point"),
            _COMMA,
            _ident("y"),
            _RPAREN,
        )
        assert isinstance(exp, OffsetOf)
        assert exp.member == "y"
