"""Parser tests for the ``_Noreturn`` function specifier."""

from compiler.parser import parse
from compiler.syntax.ast import (
    Function,
    FunDecl,
    StorageClass,
)
from compiler.types.tokens import TokenType
from test.parser.helpers import (
    _EXTERN,
    _INT,
    _LBRACE,
    _LPAREN,
    _RBRACE,
    _RETURN,
    _RPAREN,
    _SEMICOLON,
    _STATIC,
    _VOID,
    _ident,
    _number,
    _tok,
)

_NORETURN = _tok(TokenType.NORETURN, "_Noreturn")

_MINIMAL_BODY = (
    _LBRACE,
    _RETURN,
    _number("0"),
    _SEMICOLON,
    _RBRACE,
)

_VOID_PARAMS = (_LPAREN, _VOID, _RPAREN)


def _parse_fn(*prefix_tokens: object) -> Function:
    """Parse ``<prefix> int foo(void) { return 0; }`` and return the Function node.

    Args:
        prefix_tokens: Tokens appearing before ``int`` in the declaration.

    Returns:
        The ``Function`` node from the first declaration.
    """
    tokens = [*prefix_tokens, _INT, _ident("foo"), *_VOID_PARAMS, *_MINIMAL_BODY]
    decl = parse(tokens).declarations[0]  # type: ignore[arg-type]
    assert isinstance(decl, FunDecl)
    return decl.function_declaration


def test_noreturn_function_sets_is_noreturn_true() -> None:
    """``_Noreturn int foo(void) { ... }`` parses with ``is_noreturn=True``."""
    fn = _parse_fn(_NORETURN)
    assert fn.is_noreturn is True


def test_non_noreturn_function_has_is_noreturn_false() -> None:
    """Plain function declaration parses with ``is_noreturn=False`` by default."""
    fn = _parse_fn()
    assert fn.is_noreturn is False


def test_static_noreturn_sets_both_flags() -> None:
    """``static _Noreturn int foo(void) { ... }`` sets ``storage_class=Static`` and ``is_noreturn=True``."""
    fn = _parse_fn(_STATIC, _NORETURN)
    assert fn.storage_class == StorageClass.Static
    assert fn.is_noreturn is True


def test_noreturn_static_order_independent() -> None:
    """``_Noreturn static int foo(void) { ... }`` (reversed order) also parses correctly."""
    fn = _parse_fn(_NORETURN, _STATIC)
    assert fn.storage_class == StorageClass.Static
    assert fn.is_noreturn is True


def test_extern_noreturn_sets_both_flags() -> None:
    """``extern _Noreturn void foo(void);`` sets ``storage_class=Extern`` and ``is_noreturn=True``."""
    tokens = [_EXTERN, _NORETURN, _VOID, _ident("foo"), *_VOID_PARAMS, _SEMICOLON]
    decl = parse(tokens).declarations[0]  # type: ignore[arg-type]
    assert isinstance(decl, FunDecl)
    fn = decl.function_declaration
    assert fn.storage_class == StorageClass.Extern
    assert fn.is_noreturn is True
    assert fn.body is None


def test_noreturn_forward_declaration() -> None:
    """``_Noreturn void foo(void);`` parses as a prototype with ``is_noreturn=True``."""
    tokens = [_NORETURN, _VOID, _ident("foo"), *_VOID_PARAMS, _SEMICOLON]
    decl = parse(tokens).declarations[0]  # type: ignore[arg-type]
    assert isinstance(decl, FunDecl)
    fn = decl.function_declaration
    assert fn.is_noreturn is True
    assert fn.body is None


def test_noreturn_does_not_affect_is_inline() -> None:
    """``_Noreturn`` does not set ``is_inline``."""
    fn = _parse_fn(_NORETURN)
    assert fn.is_inline is False
