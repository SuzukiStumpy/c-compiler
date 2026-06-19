"""Parser tests for ``__builtin_va_*`` variadic built-in expressions."""

from compiler.syntax.ast import (
    D,
    FunDecl,
    IntType,
    S,
    Structure,
    VaArg,
    VaCopy,
    VaEnd,
    Var,
    VarDecl,
    VaStart,
)
from compiler.types.tokens import TokenType
from test.parser.helpers import (
    _COMMA,
    _INT,
    _LBRACE,
    _LPAREN,
    _RBRACE,
    _RPAREN,
    _SEMICOLON,
    _VOID,
    _ident,
    _number,
    _tok,
    parse,
)

_ELLIPSIS = _tok(TokenType.ELLIPSIS, "...")


def _variadic_fn_tokens(
    *body_tokens: object,
) -> list[object]:
    """Wrap body tokens in ``int f(int x, ...) { <body> }``."""
    return [
        _INT,
        _ident("f"),
        _LPAREN,
        _INT,
        _ident("x"),
        _COMMA,
        _ELLIPSIS,
        _RPAREN,
        _LBRACE,
        *body_tokens,
        _RBRACE,
    ]


def _parse_fn_body(*body_tokens: object) -> tuple:
    """Parse a variadic function and return its body block-item tuple."""
    tokens = _variadic_fn_tokens(*body_tokens)
    program = parse(list(tokens))
    decl = program.declarations[0]
    assert isinstance(decl, FunDecl)
    body = decl.function_declaration.body
    assert body is not None
    return body


# ---------------------------------------------------------------------------
# __builtin_va_list pre-registered typedef
# ---------------------------------------------------------------------------


class TestBuiltinVaListTypedef:
    """``__builtin_va_list`` is pre-registered as a typedef in the parser."""

    def test_builtin_va_list_usable_as_type(self) -> None:
        """``__builtin_va_list ap;`` parses as a struct-typed variable declaration."""
        # int main(void) { __builtin_va_list ap; return 0; }
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _ident("__builtin_va_list"),
            _ident("ap"),
            _SEMICOLON,
            _tok(TokenType.RETURN, "return"),
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        ]
        program = parse(tokens)
        decl = program.declarations[0]
        assert isinstance(decl, FunDecl)
        body = decl.function_declaration.body
        assert body is not None
        first_item = body[0]
        assert isinstance(first_item, D)
        assert isinstance(first_item.declaration, VarDecl)
        vd = first_item.declaration.variable_declaration
        assert vd.name == "ap"
        assert isinstance(vd.var_type, Structure)
        assert vd.var_type.tag == "__va_list_tag"

    def test_gnuc_va_list_typedef_chain(self) -> None:
        """``typedef __builtin_va_list __gnuc_va_list; __gnuc_va_list ap;`` parses correctly."""
        tokens = [
            _tok(TokenType.TYPEDEF, "typedef"),
            _ident("__builtin_va_list"),
            _ident("__gnuc_va_list"),
            _SEMICOLON,
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _ident("__gnuc_va_list"),
            _ident("ap"),
            _SEMICOLON,
            _tok(TokenType.RETURN, "return"),
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        ]
        program = parse(tokens)
        decl = program.declarations[1]
        assert isinstance(decl, FunDecl)
        body = decl.function_declaration.body
        assert body is not None
        first_item = body[0]
        assert isinstance(first_item, D)
        assert isinstance(first_item.declaration, VarDecl)
        vd = first_item.declaration.variable_declaration
        assert isinstance(vd.var_type, Structure)
        assert vd.var_type.tag == "__va_list_tag"


# ---------------------------------------------------------------------------
# __builtin_va_start
# ---------------------------------------------------------------------------


class TestBuiltinVaStart:
    """``__builtin_va_start(ap, last)`` is parsed as a ``VaStart`` expression."""

    def test_va_start_parsed_as_va_start_node(self) -> None:
        """``__builtin_va_start(ap, x);`` becomes ``VaStart(Var("ap"), Var("x"))``."""
        body = _parse_fn_body(
            _ident("__builtin_va_list"),
            _ident("ap"),
            _SEMICOLON,
            _ident("__builtin_va_start"),
            _LPAREN,
            _ident("ap"),
            _COMMA,
            _ident("x"),
            _RPAREN,
            _SEMICOLON,
            _tok(TokenType.RETURN, "return"),
            _number("0"),
            _SEMICOLON,
        )
        from compiler.syntax.ast.statements import Expression

        stmt_item = body[1]
        assert isinstance(stmt_item, S)
        assert isinstance(stmt_item.statement, Expression)
        va_exp = stmt_item.statement.exp
        assert isinstance(va_exp, VaStart)
        assert isinstance(va_exp.ap, Var)
        assert va_exp.ap.name == "ap"
        assert isinstance(va_exp.last, Var)
        assert va_exp.last.name == "x"

    def test_c23_va_start_parsed_as_va_start_node(self) -> None:
        """``__builtin_c23_va_start(ap, x);`` also becomes a ``VaStart`` node."""
        body = _parse_fn_body(
            _ident("__builtin_va_list"),
            _ident("ap"),
            _SEMICOLON,
            _ident("__builtin_c23_va_start"),
            _LPAREN,
            _ident("ap"),
            _COMMA,
            _ident("x"),
            _RPAREN,
            _SEMICOLON,
            _tok(TokenType.RETURN, "return"),
            _number("0"),
            _SEMICOLON,
        )
        from compiler.syntax.ast.statements import Expression

        stmt_item = body[1]
        assert isinstance(stmt_item, S)
        assert isinstance(stmt_item.statement, Expression)
        va_exp = stmt_item.statement.exp
        assert isinstance(va_exp, VaStart)


# ---------------------------------------------------------------------------
# __builtin_va_arg
# ---------------------------------------------------------------------------


class TestBuiltinVaArg:
    """``__builtin_va_arg(ap, T)`` is parsed as a ``VaArg`` node with type ``T``."""

    def test_va_arg_int_parsed(self) -> None:
        """``__builtin_va_arg(ap, int)`` → ``VaArg(ap=Var("ap"), arg_type=IntType())``."""
        body = _parse_fn_body(
            _ident("__builtin_va_list"),
            _ident("ap"),
            _SEMICOLON,
            _tok(TokenType.RETURN, "return"),
            _ident("__builtin_va_arg"),
            _LPAREN,
            _ident("ap"),
            _COMMA,
            _INT,
            _RPAREN,
            _SEMICOLON,
        )
        from compiler.syntax.ast.statements import Return

        ret_item = body[1]
        assert isinstance(ret_item, S)
        assert isinstance(ret_item.statement, Return)
        va_exp = ret_item.statement.exp
        assert isinstance(va_exp, VaArg)
        assert isinstance(va_exp.ap, Var)
        assert va_exp.ap.name == "ap"
        assert isinstance(va_exp.arg_type, IntType)

    def test_va_arg_long_parsed(self) -> None:
        """``__builtin_va_arg(ap, long)`` → ``VaArg`` with ``LongType()``."""
        from compiler.syntax.ast import LongType

        body = _parse_fn_body(
            _ident("__builtin_va_list"),
            _ident("ap"),
            _SEMICOLON,
            _tok(TokenType.RETURN, "return"),
            _ident("__builtin_va_arg"),
            _LPAREN,
            _ident("ap"),
            _COMMA,
            _tok(TokenType.LONG, "long"),
            _RPAREN,
            _SEMICOLON,
        )
        ret_item = body[1]
        assert isinstance(ret_item, S)
        va_exp = ret_item.statement.exp
        assert isinstance(va_exp, VaArg)
        assert isinstance(va_exp.arg_type, LongType)

    def test_va_arg_double_parsed(self) -> None:
        """``__builtin_va_arg(ap, double)`` → ``VaArg`` with ``DoubleType()``."""
        from compiler.syntax.ast import DoubleType

        body = _parse_fn_body(
            _ident("__builtin_va_list"),
            _ident("ap"),
            _SEMICOLON,
            _tok(TokenType.RETURN, "return"),
            _ident("__builtin_va_arg"),
            _LPAREN,
            _ident("ap"),
            _COMMA,
            _tok(TokenType.DOUBLE, "double"),
            _RPAREN,
            _SEMICOLON,
        )
        ret_item = body[1]
        assert isinstance(ret_item, S)
        va_exp = ret_item.statement.exp
        assert isinstance(va_exp, VaArg)
        assert isinstance(va_exp.arg_type, DoubleType)


# ---------------------------------------------------------------------------
# __builtin_va_end
# ---------------------------------------------------------------------------


class TestBuiltinVaEnd:
    """``__builtin_va_end(ap)`` is parsed as a ``VaEnd`` expression."""

    def test_va_end_parsed(self) -> None:
        """``__builtin_va_end(ap);`` → ``VaEnd(ap=Var("ap"))``."""
        from compiler.syntax.ast.statements import Expression

        body = _parse_fn_body(
            _ident("__builtin_va_list"),
            _ident("ap"),
            _SEMICOLON,
            _ident("__builtin_va_end"),
            _LPAREN,
            _ident("ap"),
            _RPAREN,
            _SEMICOLON,
            _tok(TokenType.RETURN, "return"),
            _number("0"),
            _SEMICOLON,
        )
        stmt_item = body[1]
        assert isinstance(stmt_item, S)
        assert isinstance(stmt_item.statement, Expression)
        va_exp = stmt_item.statement.exp
        assert isinstance(va_exp, VaEnd)
        assert isinstance(va_exp.ap, Var)
        assert va_exp.ap.name == "ap"


# ---------------------------------------------------------------------------
# __builtin_va_copy
# ---------------------------------------------------------------------------


class TestBuiltinVaCopy:
    """``__builtin_va_copy(dest, src)`` is parsed as a ``VaCopy`` expression."""

    def test_va_copy_parsed(self) -> None:
        """``__builtin_va_copy(dest, src);`` → ``VaCopy(dest=Var("dest"), src=Var("src"))``."""
        from compiler.syntax.ast.statements import Expression

        body = _parse_fn_body(
            _ident("__builtin_va_list"),
            _ident("dest"),
            _SEMICOLON,
            _ident("__builtin_va_list"),
            _ident("src"),
            _SEMICOLON,
            _ident("__builtin_va_copy"),
            _LPAREN,
            _ident("dest"),
            _COMMA,
            _ident("src"),
            _RPAREN,
            _SEMICOLON,
            _tok(TokenType.RETURN, "return"),
            _number("0"),
            _SEMICOLON,
        )
        stmt_item = body[2]
        assert isinstance(stmt_item, S)
        assert isinstance(stmt_item.statement, Expression)
        va_exp = stmt_item.statement.exp
        assert isinstance(va_exp, VaCopy)
        assert isinstance(va_exp.dest, Var)
        assert va_exp.dest.name == "dest"
        assert isinstance(va_exp.src, Var)
        assert va_exp.src.name == "src"
