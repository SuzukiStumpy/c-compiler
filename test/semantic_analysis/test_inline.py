"""Semantic-analysis tests for C11 §6.7.4 p6-7 inline function external-linkage rules.

An inline function with external linkage is an *inline definition* (not an external
definition) when every file-scope declaration in the translation unit uses the
``inline`` specifier without ``extern``.  Such functions must not receive a ``.globl``
directive — they are compiled for intra-TU calls only.

The cases tested below match the grid in C11 §6.7.4 p6-7:

  +-----------------------+----------+---------+---------+
  | Declaration form      | linkage  | defined | globl   |
  +-----------------------+----------+---------+---------+
  | inline f() {...}      | external | yes     | False   |
  | extern inline f(){..} | external | yes     | True    |
  | static inline f(){..} | internal | yes     | False   |
  | inline f(); + f(){..} | external | yes     | False   |
  | f(); + inline f(){..} | external | yes     | True    |
  | inline f(){..} + extern f(); | external | yes | True |
  +-----------------------+----------+---------+---------+
"""

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve_with_symbols
from compiler.types.symbols import FunAttr
from test.semantic_analysis.helpers import _prog_with_decls

_VOID_FUN_TYPE = ast.FunType(params=(), ret=ast.IntType())
_RETURN_ZERO: tuple[ast.BlockItem, ...] = (ast.S(ast.Return(ast.Constant(0))),)


def _fn(
    name: str,
    body: tuple[ast.BlockItem, ...] | None = None,
    sc: ast.StorageClass | None = None,
    is_inline: bool = False,
) -> ast.FunDecl:
    """Build a top-level ``FunDecl`` with optional storage class and inline flag.

    Args:
        name: Function name.
        body: Block items for the definition body, or ``None`` for a prototype.
        sc: Optional storage class specifier.
        is_inline: Whether the ``inline`` specifier is present.

    Returns:
        An ``ast.FunDecl`` wrapping the constructed ``ast.Function``.
    """
    return ast.FunDecl(
        ast.Function(
            name=name,
            params=(),
            body=body,
            fun_type=_VOID_FUN_TYPE,
            storage_class=sc,
            is_inline=is_inline,
        )
    )


_MAIN = _fn("main", body=_RETURN_ZERO)


class TestInlineExternalLinkageRules:
    """C11 §6.7.4 p6-7: inline definitions vs. external definitions."""

    def test_inline_definition_only_is_not_global(self) -> None:
        """``inline int foo(void) {...}`` is an inline definition — ``globl`` must be ``False``."""
        program = _prog_with_decls(_fn("foo", body=_RETURN_ZERO, is_inline=True), _MAIN)
        _, symbols, _, _ = resolve_with_symbols(program)
        entry = symbols["foo"]
        assert isinstance(entry.attrs, FunAttr)
        assert entry.attrs.defined is True
        assert entry.attrs.globl is False

    def test_extern_inline_definition_is_global(self) -> None:
        """``extern inline int foo(void) {...}`` provides an external definition — ``globl`` must be ``True``."""
        program = _prog_with_decls(_fn("foo", body=_RETURN_ZERO, sc=ast.StorageClass.Extern, is_inline=True), _MAIN)
        _, symbols, _, _ = resolve_with_symbols(program)
        entry = symbols["foo"]
        assert isinstance(entry.attrs, FunAttr)
        assert entry.attrs.defined is True
        assert entry.attrs.globl is True

    def test_static_inline_definition_is_not_global(self) -> None:
        """``static inline int foo(void) {...}`` has internal linkage — ``globl`` must be ``False``."""
        program = _prog_with_decls(_fn("foo", body=_RETURN_ZERO, sc=ast.StorageClass.Static, is_inline=True), _MAIN)
        _, symbols, _, _ = resolve_with_symbols(program)
        entry = symbols["foo"]
        assert isinstance(entry.attrs, FunAttr)
        assert entry.attrs.defined is True
        assert entry.attrs.globl is False

    def test_inline_prototype_then_inline_definition_is_not_global(self) -> None:
        """``inline foo();`` followed by ``inline foo() {...}`` — all inline, no extern → ``globl=False``."""
        program = _prog_with_decls(
            _fn("foo", body=None, is_inline=True),
            _fn("foo", body=_RETURN_ZERO, is_inline=True),
            _MAIN,
        )
        _, symbols, _, _ = resolve_with_symbols(program)
        entry = symbols["foo"]
        assert isinstance(entry.attrs, FunAttr)
        assert entry.attrs.globl is False

    def test_non_inline_prototype_then_inline_definition_is_global(self) -> None:
        """``foo();`` (no inline) followed by ``inline foo() {...}`` — mixed → ``globl=True``."""
        program = _prog_with_decls(
            _fn("foo", body=None, is_inline=False),
            _fn("foo", body=_RETURN_ZERO, is_inline=True),
            _MAIN,
        )
        _, symbols, _, _ = resolve_with_symbols(program)
        entry = symbols["foo"]
        assert isinstance(entry.attrs, FunAttr)
        assert entry.attrs.globl is True

    def test_inline_definition_then_extern_declaration_is_global(self) -> None:
        """``inline foo() {...}`` then ``extern foo();`` — extern decl in TU → ``globl=True``."""
        program = _prog_with_decls(
            _fn("foo", body=_RETURN_ZERO, is_inline=True),
            _fn("foo", body=None, sc=ast.StorageClass.Extern),
            _MAIN,
        )
        _, symbols, _, _ = resolve_with_symbols(program)
        entry = symbols["foo"]
        assert isinstance(entry.attrs, FunAttr)
        assert entry.attrs.globl is True
