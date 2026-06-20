"""Semantic analysis tests for wide and Unicode string / char literal handling.

Verifies that the semantic analyser assigns correct array element types to
wide string literals based on their :class:`StringEncoding` prefix and that
the resulting ``StringInit`` entries carry the correct ``element_size``.
"""

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve_with_symbols
from compiler.syntax.ast.expressions import StringEncoding
from compiler.types.symbols import ConstantAttr, StringInit
from test.semantic_analysis.helpers import _file_var, _prog_with_decls, _resolved_body

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wchar_str(chars: str) -> ast.String:
    """Build a ``WCHAR``-encoded ``String`` node from a plain ASCII string.

    Args:
        chars: The ASCII source characters (no NUL).

    Returns:
        A ``String`` node with ``WCHAR`` encoding and 4-byte LE per character.
    """
    raw = b"".join(ord(c).to_bytes(4, "little") for c in chars)
    return ast.String(raw, StringEncoding.WCHAR)


def _u16_str(chars: str) -> ast.String:
    """Build a ``UTF16``-encoded ``String`` node from a plain ASCII string."""
    raw = b"".join(ord(c).to_bytes(2, "little") for c in chars)
    return ast.String(raw, StringEncoding.UTF16)


def _u32_str(chars: str) -> ast.String:
    """Build a ``UTF32``-encoded ``String`` node from a plain ASCII string."""
    raw = b"".join(ord(c).to_bytes(4, "little") for c in chars)
    return ast.String(raw, StringEncoding.UTF32)


def _resolved_string_exp_type(string_node: ast.String) -> ast.Type:
    """Resolve a program evaluating *string_node* as an expression and return the ``String`` node's ``exp_type``.

    The ``String`` decays to a pointer in value context (SA wraps it in ``AddrOf``).
    This helper extracts the inner ``String`` node and returns its ``exp_type``, which
    is the array type (not the pointer type of the decayed expression).

    Args:
        string_node: An untyped ``String`` AST node.

    Returns:
        The ``exp_type`` assigned by semantic analysis to the ``String`` node.
    """
    body = _resolved_body((ast.S(ast.Expression(string_node)),))
    stmt = body[0]
    assert isinstance(stmt, ast.S)
    assert isinstance(stmt.statement, ast.Expression)
    resolved_exp = stmt.statement.exp
    # In value context the String decays: SA wraps it in AddrOf(String(...))
    if isinstance(resolved_exp, ast.AddrOf) and isinstance(resolved_exp.exp, ast.String):
        return resolved_exp.exp.exp_type  # type: ignore[return-value]
    assert isinstance(resolved_exp, ast.String)
    return resolved_exp.exp_type  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# _type_check_exp: String node type assignment
# ---------------------------------------------------------------------------


class TestWideStringExpType:
    """``_type_check_exp`` assigns the correct array element type based on string encoding."""

    def test_plain_string_gets_char_element_type(self) -> None:
        """Unencoded ``String`` -> ``Array(Char(), N+1)``."""
        t = _resolved_string_exp_type(ast.String(b"hi"))
        assert t == ast.Array(ast.Char(), 3)

    def test_wchar_string_gets_int_element_type(self) -> None:
        """``L"AB"`` (WCHAR) -> ``Array(IntType(), 3)`` (2 chars + NUL wchar_t)."""
        t = _resolved_string_exp_type(_wchar_str("AB"))
        assert t == ast.Array(ast.IntType(), 3)

    def test_utf16_string_gets_ushort_element_type(self) -> None:
        """``u"ABC"`` (UTF16) -> ``Array(UShortType(), 4)`` (3 chars + NUL)."""
        t = _resolved_string_exp_type(_u16_str("ABC"))
        assert t == ast.Array(ast.UShortType(), 4)

    def test_utf32_string_gets_uint_element_type(self) -> None:
        """``U"A"`` (UTF32) -> ``Array(UIntType(), 2)`` (1 char + NUL)."""
        t = _resolved_string_exp_type(_u32_str("A"))
        assert t == ast.Array(ast.UIntType(), 2)

    def test_wchar_empty_string_gets_size_one_array(self) -> None:
        """``L""`` (WCHAR) -> ``Array(IntType(), 1)`` (only the NUL wchar_t)."""
        t = _resolved_string_exp_type(ast.String(b"", StringEncoding.WCHAR))
        assert t == ast.Array(ast.IntType(), 1)


# ---------------------------------------------------------------------------
# Hidden symbol entry for wide pointer initialiser
# ---------------------------------------------------------------------------


class TestWideStringPointerInit:
    """Wide string pointer initialisers create a hidden symbol with the correct element type."""

    def _wchar_ptr_decl(self, chars: str) -> ast.VarDecl:
        """Build a file-scope ``int *p = L"<chars>";`` declaration (int = wchar_t on Linux).

        Args:
            chars: ASCII source characters.

        Returns:
            A ``VarDecl`` node whose initialiser is a ``SingleInit(String(WCHAR))``.
        """
        return _file_var(
            "p",
            ast.SingleInit(_wchar_str(chars)),
            var_type=ast.Pointer(ast.IntType()),
        )

    def test_wchar_pointer_init_creates_hidden_symbol(self) -> None:
        """``int *p = L"hi";`` creates a hidden ``ConstantAttr`` symbol entry."""
        prog = _prog_with_decls(self._wchar_ptr_decl("hi"))
        _, symbols, _, _ = resolve_with_symbols(prog)
        hidden = [name for name, e in symbols.items() if isinstance(e.attrs, ConstantAttr)]
        assert len(hidden) == 1

    def test_wchar_pointer_init_hidden_symbol_type_is_int_array(self) -> None:
        """The hidden symbol for ``L"hi"`` has type ``Array(IntType(), 3)``."""
        prog = _prog_with_decls(self._wchar_ptr_decl("hi"))
        _, symbols, _, _ = resolve_with_symbols(prog)
        hidden_entry = next(e for e in symbols.values() if isinstance(e.attrs, ConstantAttr))
        assert hidden_entry.t == ast.Array(ast.IntType(), 3)

    def test_wchar_pointer_init_stringinit_has_element_size_four(self) -> None:
        """The ``StringInit`` for a ``L""`` literal has ``element_size=4``."""
        prog = _prog_with_decls(self._wchar_ptr_decl("hi"))
        _, symbols, _, _ = resolve_with_symbols(prog)
        hidden_entry = next(e for e in symbols.values() if isinstance(e.attrs, ConstantAttr))
        assert isinstance(hidden_entry.attrs.init, StringInit)
        assert hidden_entry.attrs.init.element_size == 4
