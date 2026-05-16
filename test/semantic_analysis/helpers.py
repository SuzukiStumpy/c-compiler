"""Shared helpers for the semantic analysis test package.

Provides lightweight AST-building utilities used across the themed test
modules so that each test constructs ``ast`` nodes directly and exercises
the semantic analyser in isolation, without invoking the lexer or parser.
"""

import compiler.syntax.ast as ast
from compiler.semantic_analysis import resolve

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vd(name: str, init: ast.Exp | None = None) -> ast.D:
    """Return a ``D`` block item wrapping a variable declaration.

    Args:
        name: The source-level variable name.
        init: Optional initialiser expression.

    Returns:
        An ``ast.D`` wrapping ``ast.VarDecl(ast.VariableDeclaration(name, init))``.
    """
    return ast.D(ast.VarDecl(ast.VariableDeclaration(name, init)))


def _program(body: tuple[ast.BlockItem, ...]) -> ast.Program:
    """Wrap a sequence of block items in a minimal single-function ``ast.Program``.

    Args:
        body: The block items to place in the body of ``main``.

    Returns:
        An ``ast.Program`` containing a single ``main`` function with the
        given body.
    """
    return ast.Program(
        declarations=(
            ast.FunDecl(
                ast.Function(
                    name="main",
                    params=(),
                    body=body,
                )
            ),
        )
    )


def _make_prog(*fns: ast.Function) -> ast.Program:
    """Wrap one or more ``Function`` nodes in an ``ast.Program``.

    Args:
        *fns: The ``Function`` nodes to include in the program, in source order.

    Returns:
        An ``ast.Program`` whose ``declarations`` tuple contains each function
        wrapped in an ``ast.FunDecl``.
    """
    return ast.Program(declarations=tuple(ast.FunDecl(fn) for fn in fns))


def _get_fn(program: ast.Program, index: int = 0) -> ast.Function:
    """Return the *n*-th function declaration from a resolved program.

    Args:
        program: The resolved ``ast.Program``.
        index: Zero-based index into the list of ``FunDecl`` declarations.

    Returns:
        The ``ast.Function`` at the given position among the program's
        ``FunDecl`` entries.
    """
    fns = [d.function_declaration for d in program.declarations if isinstance(d, ast.FunDecl)]
    return fns[index]


def _resolved_body(body: tuple[ast.BlockItem, ...]) -> tuple[ast.BlockItem, ...]:
    """Resolve a function body and return the resolved block items.

    Args:
        body: The block items to resolve.

    Returns:
        The resolved block items from the single function in the resolved
        program.
    """
    return _get_fn(resolve(_program(body))).body


def _file_var(name: str, init: ast.Exp | None = None, sc: ast.StorageClass | None = None) -> ast.VarDecl:
    """Return a top-level ``VarDecl`` with the given name, optional initialiser, and optional storage class.

    Args:
        name: The variable name.
        init: Optional initialiser expression.
        sc: Optional storage class specifier.

    Returns:
        An ``ast.VarDecl`` wrapping a ``VariableDeclaration``.
    """
    return ast.VarDecl(ast.VariableDeclaration(name, init, sc))


def _prog_with_decls(*decls: ast.Declaration) -> ast.Program:
    """Build an ``ast.Program`` from an arbitrary mix of top-level declarations.

    Args:
        *decls: Any combination of ``ast.FunDecl`` and ``ast.VarDecl`` nodes.

    Returns:
        An ``ast.Program`` whose ``declarations`` tuple contains all given nodes.
    """
    return ast.Program(declarations=decls)
