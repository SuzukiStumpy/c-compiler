"""Parser tests for type qualifiers: ``const``, ``volatile``, ``restrict``.

Qualifiers are preserved in the AST as ``QualifiedType`` wrapper nodes. Each
test verifies that a qualified declaration or expression parses to a node that
carries the correct qualifier flags.
"""

from compiler.parser import parse
from compiler.syntax.ast import (
    Constant,
    D,
    FunDecl,
    IntType,
    LongType,
    Pointer,
    SingleInit,
    VarDecl,
    VariableDeclaration,
)
from compiler.syntax.ast.constants import ConstInt
from compiler.syntax.ast.expressions import Cast
from compiler.syntax.ast.types import QualifiedType, UIntType
from compiler.types.tokens import Token, TokenType
from test.parser.helpers import (
    _ASSIGNMENT,
    _ASTERISK,
    _INT,
    _LBRACE,
    _LONG,
    _LPAREN,
    _RBRACE,
    _RETURN,
    _RPAREN,
    _SEMICOLON,
    _UNSIGNED,
    _ident,
    _number,
    _parsed_body,
    _parsed_exp,
    _tok,
)


def _CONST() -> Token:
    """Return a CONST token."""
    return _tok(TokenType.CONST, "const")


def _VOLATILE() -> Token:
    """Return a VOLATILE token."""
    return _tok(TokenType.VOLATILE, "volatile")


def _RESTRICT() -> Token:
    """Return a RESTRICT token."""
    return _tok(TokenType.RESTRICT, "restrict")


# ---------------------------------------------------------------------------
# const qualifier
# ---------------------------------------------------------------------------


def test_const_before_int_in_local_declaration() -> None:
    """``const int x = 5;`` produces ``var_type=QualifiedType(IntType(), const=True)``."""
    body = _parsed_body(_CONST(), _INT, _ident("x"), _ASSIGNMENT, _number("5"), _SEMICOLON)
    item = body[0]
    assert isinstance(item, D)
    assert isinstance(item.declaration, VarDecl)
    vd = item.declaration.variable_declaration
    assert vd == VariableDeclaration(
        name="x",
        var_type=QualifiedType(base=IntType(), const=True),
        init=SingleInit(Constant(ConstInt(5))),
        storage_class=None,
    )


def test_const_after_int_in_local_declaration() -> None:
    """``int const x = 5;`` (qualifier after specifier) also yields a const-qualified int."""
    body = _parsed_body(_INT, _CONST(), _ident("x"), _ASSIGNMENT, _number("5"), _SEMICOLON)
    item = body[0]
    assert isinstance(item, D)
    assert isinstance(item.declaration, VarDecl)
    vd = item.declaration.variable_declaration
    assert vd == VariableDeclaration(
        name="x",
        var_type=QualifiedType(base=IntType(), const=True),
        init=SingleInit(Constant(ConstInt(5))),
        storage_class=None,
    )


def test_const_on_global_variable_declaration() -> None:
    """``const int x = 0;`` at file scope produces a const-qualified int."""
    tokens = [_CONST(), _INT, _ident("x"), _ASSIGNMENT, _number("0"), _SEMICOLON]
    program = parse(tokens)
    assert len(program.declarations) == 1
    decl = program.declarations[0]
    assert isinstance(decl, VarDecl)
    assert decl.variable_declaration.var_type == QualifiedType(base=IntType(), const=True)


# ---------------------------------------------------------------------------
# volatile qualifier
# ---------------------------------------------------------------------------


def test_volatile_before_long_in_local_declaration() -> None:
    """``volatile long y;`` produces ``var_type=QualifiedType(LongType(), volatile=True)``."""
    body = _parsed_body(_VOLATILE(), _LONG, _ident("y"), _SEMICOLON)
    item = body[0]
    assert isinstance(item, D)
    assert isinstance(item.declaration, VarDecl)
    vd = item.declaration.variable_declaration
    assert vd.var_type == QualifiedType(base=LongType(), volatile=True)
    assert vd.name == "y"


# ---------------------------------------------------------------------------
# restrict qualifier
# ---------------------------------------------------------------------------


def test_restrict_on_pointer_parameter() -> None:
    """``int * restrict p`` as a parameter produces a restrict-qualified pointer."""
    tokens = [
        _INT,
        _ident("f"),
        _LPAREN,
        _INT,
        _ASTERISK,
        _RESTRICT(),
        _ident("p"),
        _RPAREN,
        _LBRACE,
        _RETURN,
        _number("0"),
        _SEMICOLON,
        _RBRACE,
    ]
    program = parse(tokens)
    decl = program.declarations[0]
    assert isinstance(decl, FunDecl)
    params = decl.function_declaration.params
    assert len(params) == 1
    assert params[0].param_type == QualifiedType(base=Pointer(IntType()), restrict=True)


# ---------------------------------------------------------------------------
# Multiple qualifiers and pointer qualifiers
# ---------------------------------------------------------------------------


def test_const_pointer_declarator() -> None:
    """``int * const p;`` (const pointer to int) produces a const-qualified pointer."""
    body = _parsed_body(_INT, _ASTERISK, _CONST(), _ident("p"), _SEMICOLON)
    item = body[0]
    assert isinstance(item, D)
    assert isinstance(item.declaration, VarDecl)
    vd = item.declaration.variable_declaration
    assert vd.var_type == QualifiedType(base=Pointer(IntType()), const=True)
    assert vd.name == "p"


def test_const_pointer_to_const_int() -> None:
    """``const int * const p;`` is a const pointer to const int."""
    body = _parsed_body(_CONST(), _INT, _ASTERISK, _CONST(), _ident("p"), _SEMICOLON)
    item = body[0]
    assert isinstance(item, D)
    assert isinstance(item.declaration, VarDecl)
    vd = item.declaration.variable_declaration
    assert vd.var_type == QualifiedType(
        base=Pointer(QualifiedType(base=IntType(), const=True)),
        const=True,
    )


def test_multiple_qualifiers_on_type() -> None:
    """``const volatile int x;`` produces a const+volatile qualified int."""
    body = _parsed_body(_CONST(), _VOLATILE(), _INT, _ident("x"), _SEMICOLON)
    item = body[0]
    assert isinstance(item, D)
    assert isinstance(item.declaration, VarDecl)
    vd = item.declaration.variable_declaration
    assert vd.var_type == QualifiedType(base=IntType(), const=True, volatile=True)


def test_restrict_and_const_on_pointer() -> None:
    """``const int * restrict p;`` is a restrict-qualified pointer to const int."""
    body = _parsed_body(_CONST(), _INT, _ASTERISK, _RESTRICT(), _ident("p"), _SEMICOLON)
    item = body[0]
    assert isinstance(item, D)
    assert isinstance(item.declaration, VarDecl)
    vd = item.declaration.variable_declaration
    assert vd.var_type == QualifiedType(
        base=Pointer(QualifiedType(base=IntType(), const=True)),
        restrict=True,
    )


# ---------------------------------------------------------------------------
# Qualifiers in cast expressions
# ---------------------------------------------------------------------------


def test_cast_to_const_int() -> None:
    """``(const int) x`` parses as a cast to ``QualifiedType(IntType(), const=True)``."""
    exp = _parsed_exp(_LPAREN, _CONST(), _INT, _RPAREN, _ident("x"))
    assert isinstance(exp, Cast)
    assert exp.target_type == QualifiedType(base=IntType(), const=True)


def test_cast_to_unsigned_const_int() -> None:
    """``(const unsigned int) x`` parses as a cast to ``QualifiedType(UIntType(), const=True)``."""
    exp = _parsed_exp(_LPAREN, _CONST(), _UNSIGNED, _INT, _RPAREN, _ident("x"))
    assert isinstance(exp, Cast)
    assert exp.target_type == QualifiedType(base=UIntType(), const=True)


# ---------------------------------------------------------------------------
# Qualifiers on function parameters
# ---------------------------------------------------------------------------


def test_const_int_parameter() -> None:
    """``const int n`` as a function parameter preserves the const qualifier."""
    tokens = [
        _INT,
        _ident("f"),
        _LPAREN,
        _CONST(),
        _INT,
        _ident("n"),
        _RPAREN,
        _LBRACE,
        _RETURN,
        _number("0"),
        _SEMICOLON,
        _RBRACE,
    ]
    program = parse(tokens)
    decl = program.declarations[0]
    assert isinstance(decl, FunDecl)
    params = decl.function_declaration.params
    assert len(params) == 1
    assert params[0].param_type == QualifiedType(base=IntType(), const=True)
    assert params[0].name == "n"
