"""Parser tests for ``_Alignas`` alignment specifier."""

from compiler.syntax.ast import (
    AlignOfT,
    Constant,
    ConstInt,
    D,
    DoubleType,
    FunDecl,
    VarDecl,
    VariableDeclaration,
)
from compiler.types.tokens import TokenType
from test.parser.helpers import (
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

_ALIGNAS = _tok(TokenType.ALIGNAS, "_Alignas")
_DOUBLE = _tok(TokenType.DOUBLE, "double")


def _parse_file_var(*prefix_tokens: object, name: str = "x") -> VariableDeclaration:
    """Parse ``<prefix> int <name>;`` at file scope and return the ``VariableDeclaration``."""
    tokens = [*prefix_tokens, _INT, _ident(name), _SEMICOLON]
    decl = parse(tokens).declarations[0]  # type: ignore[arg-type]
    assert isinstance(decl, VarDecl)
    return decl.variable_declaration


def _parse_body_var(*prefix_tokens: object, name: str = "x") -> VariableDeclaration:
    """Parse ``<prefix> int <name>;`` inside a function and return the ``VariableDeclaration``."""
    tokens = [
        _INT,
        _ident("main"),
        _LPAREN,
        _VOID,
        _RPAREN,
        _tok(TokenType.LBRACE, "{"),
        *prefix_tokens,
        _INT,
        _ident(name),
        _SEMICOLON,
        _tok(TokenType.RBRACE, "}"),
    ]
    decl = parse(tokens).declarations[0]  # type: ignore[arg-type]
    assert isinstance(decl, FunDecl)
    body = decl.function_declaration.body
    assert body is not None
    item = body[0]
    assert isinstance(item, D)
    assert isinstance(item.declaration, VarDecl)
    return item.declaration.variable_declaration


def test_alignas_integer_on_file_scope_var() -> None:
    """``_Alignas(16) int x;`` at file scope stores raw alignment expression."""
    var = _parse_file_var(_ALIGNAS, _LPAREN, _number("16"), _RPAREN)
    assert isinstance(var.align_as_exp, Constant)
    assert var.align_as_exp.value == ConstInt(16)


def test_alignas_type_on_file_scope_var() -> None:
    """``_Alignas(double) int x;`` stores an ``AlignOfT(DoubleType())`` expression."""
    var = _parse_file_var(_ALIGNAS, _LPAREN, _DOUBLE, _RPAREN)
    assert isinstance(var.align_as_exp, AlignOfT)
    assert var.align_as_exp.t == DoubleType()


def test_alignas_integer_on_local_var() -> None:
    """``_Alignas(8) int x;`` inside a function stores raw alignment expression."""
    var = _parse_body_var(_ALIGNAS, _LPAREN, _number("8"), _RPAREN)
    assert isinstance(var.align_as_exp, Constant)
    assert var.align_as_exp.value == ConstInt(8)


def test_no_alignas_gives_none_align_as_exp() -> None:
    """A plain ``int x;`` has ``align_as_exp=None``."""
    var = _parse_file_var()
    assert var.align_as_exp is None


def test_alignas_after_type_specifier() -> None:
    """``int _Alignas(16) x;`` (alignment specifier after type) stores raw alignment expression."""
    tokens = [_INT, _ALIGNAS, _LPAREN, _number("16"), _RPAREN, _ident("x"), _SEMICOLON]
    decl = parse(tokens).declarations[0]  # type: ignore[arg-type]
    assert isinstance(decl, VarDecl)
    var = decl.variable_declaration
    assert isinstance(var.align_as_exp, Constant)
    assert var.align_as_exp.value == ConstInt(16)
