"""Parser tests for the ``inline`` function specifier."""

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

_INLINE = _tok(TokenType.INLINE, "inline")

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


def test_inline_function_sets_is_inline_true() -> None:
    """``inline int foo(void) { ... }`` parses with ``is_inline=True``."""
    fn = _parse_fn(_INLINE)
    assert fn.is_inline is True


def test_non_inline_function_has_is_inline_false() -> None:
    """Plain function declaration parses with ``is_inline=False`` by default."""
    fn = _parse_fn()
    assert fn.is_inline is False


def test_static_inline_sets_both_flags() -> None:
    """``static inline int foo(void) { ... }`` sets ``storage_class=Static`` and ``is_inline=True``."""
    fn = _parse_fn(_STATIC, _INLINE)
    assert fn.storage_class == StorageClass.Static
    assert fn.is_inline is True


def test_inline_static_order_independent() -> None:
    """``inline static int foo(void) { ... }`` (reversed order) also parses correctly."""
    fn = _parse_fn(_INLINE, _STATIC)
    assert fn.storage_class == StorageClass.Static
    assert fn.is_inline is True


def test_extern_inline_sets_both_flags() -> None:
    """``extern inline int foo(void);`` sets ``storage_class=Extern`` and ``is_inline=True``."""
    tokens = [_EXTERN, _INLINE, _INT, _ident("foo"), *_VOID_PARAMS, _SEMICOLON]
    decl = parse(tokens).declarations[0]  # type: ignore[arg-type]
    assert isinstance(decl, FunDecl)
    fn = decl.function_declaration
    assert fn.storage_class == StorageClass.Extern
    assert fn.is_inline is True
    assert fn.body is None


def test_inline_forward_declaration() -> None:
    """``inline int foo(void);`` parses as a prototype with ``is_inline=True``."""
    tokens = [_INLINE, _INT, _ident("foo"), *_VOID_PARAMS, _SEMICOLON]
    decl = parse(tokens).declarations[0]  # type: ignore[arg-type]
    assert isinstance(decl, FunDecl)
    fn = decl.function_declaration
    assert fn.is_inline is True
    assert fn.body is None
