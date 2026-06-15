"""Parser tests for typedef declarations and typedef-name resolution."""

from compiler.syntax.ast import (
    D,
    FunDecl,
    IntType,
    LongType,
    TypedefDecl,
    ULongType,
    VarDecl,
)
from compiler.types.tokens import TokenType
from test.parser.helpers import (
    _ASSIGNMENT,
    _INT,
    _LBRACE,
    _LONG,
    _LPAREN,
    _RBRACE,
    _RETURN,
    _RPAREN,
    _SEMICOLON,
    _UNSIGNED,
    _VOID,
    _ident,
    _number,
    _tok,
    parse,
)


def _TYPEDEF() -> object:
    """Return a TYPEDEF keyword token."""
    return _tok(TokenType.TYPEDEF, "typedef")


# ---------------------------------------------------------------------------
# Cycle 2: typedef declaration produces TypedefDecl node
# ---------------------------------------------------------------------------


def test_typedef_declaration_produces_typedef_decl_node() -> None:
    # typedef int myint;
    tokens = [_TYPEDEF(), _INT, _ident("myint"), _SEMICOLON]
    program = parse(tokens)
    assert len(program.declarations) == 1
    decl = program.declarations[0]
    assert isinstance(decl, TypedefDecl)
    assert decl.name == "myint"
    assert decl.aliased_type == IntType()


# ---------------------------------------------------------------------------
# Cycle 3: typedef name resolves to aliased type in variable declaration
# ---------------------------------------------------------------------------


def test_typedef_name_resolves_to_aliased_type_in_var_decl() -> None:
    # typedef unsigned long size_t;
    # int main(void) { size_t x = 0; return 0; }
    tokens = [
        _TYPEDEF(),
        _UNSIGNED,
        _LONG,
        _ident("size_t"),
        _SEMICOLON,
        _INT,
        _ident("main"),
        _LPAREN,
        _VOID,
        _RPAREN,
        _LBRACE,
        _ident("size_t"),
        _ident("x"),
        _ASSIGNMENT,
        _number("0"),
        _SEMICOLON,
        _RETURN,
        _number("0"),
        _SEMICOLON,
        _RBRACE,
    ]
    program = parse(tokens)
    fun_decl = program.declarations[1]
    assert isinstance(fun_decl, FunDecl)
    body = fun_decl.function_declaration.body
    assert body is not None
    first_item = body[0]
    assert isinstance(first_item, D)
    var_decl = first_item.declaration
    assert isinstance(var_decl, VarDecl)
    assert var_decl.variable_declaration.var_type == ULongType()


# ---------------------------------------------------------------------------
# Cycle 4: inner typedef shadows outer typedef
# ---------------------------------------------------------------------------


def test_inner_typedef_shadows_outer_typedef() -> None:
    # typedef int T;
    # int main(void) {
    #   typedef long T;
    #   T x = 0;
    #   return 0;
    # }
    tokens = [
        _TYPEDEF(),
        _INT,
        _ident("T"),
        _SEMICOLON,
        _INT,
        _ident("main"),
        _LPAREN,
        _VOID,
        _RPAREN,
        _LBRACE,
        _TYPEDEF(),
        _LONG,
        _ident("T"),
        _SEMICOLON,
        _ident("T"),
        _ident("x"),
        _ASSIGNMENT,
        _number("0"),
        _SEMICOLON,
        _RETURN,
        _number("0"),
        _SEMICOLON,
        _RBRACE,
    ]
    program = parse(tokens)
    fun_decl = program.declarations[1]
    assert isinstance(fun_decl, FunDecl)
    body = fun_decl.function_declaration.body
    assert body is not None
    # body[0] is the inner TypedefDecl, body[1] is the var decl
    var_item = body[1]
    assert isinstance(var_item, D)
    var_decl = var_item.declaration
    assert isinstance(var_decl, VarDecl)
    assert var_decl.variable_declaration.var_type == LongType()


# ---------------------------------------------------------------------------
# Cycle 5: inner typedef not visible after block exits
# ---------------------------------------------------------------------------


def test_inner_typedef_not_visible_after_block_exits() -> None:
    # typedef int T;
    # int main(void) {
    #   { typedef long T; }
    #   T x = 0;   <-- T must still be int here
    #   return 0;
    # }
    tokens = [
        _TYPEDEF(),
        _INT,
        _ident("T"),
        _SEMICOLON,
        _INT,
        _ident("main"),
        _LPAREN,
        _VOID,
        _RPAREN,
        _LBRACE,
        _LBRACE,
        _TYPEDEF(),
        _LONG,
        _ident("T"),
        _SEMICOLON,
        _RBRACE,
        _ident("T"),
        _ident("x"),
        _ASSIGNMENT,
        _number("0"),
        _SEMICOLON,
        _RETURN,
        _number("0"),
        _SEMICOLON,
        _RBRACE,
    ]
    program = parse(tokens)
    fun_decl = program.declarations[1]
    assert isinstance(fun_decl, FunDecl)
    body = fun_decl.function_declaration.body
    assert body is not None
    # body[0] is the inner Compound block, body[1] is the var decl
    var_item = body[1]
    assert isinstance(var_item, D)
    var_decl = var_item.declaration
    assert isinstance(var_decl, VarDecl)
    assert var_decl.variable_declaration.var_type == IntType()
