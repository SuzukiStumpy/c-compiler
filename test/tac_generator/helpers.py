"""Tests for the TAC generation stage.

Verifies that ``generate`` correctly translates ``ast`` nodes into TAC
instruction sequences, with particular attention to binary operations,
instruction ordering, and temporary variable allocation.

Each test constructs ``ast`` nodes directly (without invoking the lexer or
parser) so that TAC generator behaviour is tested in isolation.
"""

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from compiler.tac_generator import generate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _program_returning(exp: ast.Exp) -> ast.Program:
    """Wrap an expression in a minimal ``ast.Program`` that returns it.

    Args:
        exp: The expression to return from the generated function.

    Returns:
        An ``ast.Program`` containing a single ``main`` function whose body
        is ``return <exp>;``.
    """
    return ast.Program(
        declarations=(
            ast.FunDecl(
                ast.Function(
                    name="main",
                    params=(),
                    body=(ast.S(statement=ast.Return(exp=exp)),),
                )
            ),
        )
    )


def _body(exp: ast.Exp) -> tuple[tac.Instruction, ...]:
    """Translate a single expression and return the resulting TAC instruction body.

    Args:
        exp: The expression to translate.

    Returns:
        The TAC instruction tuple from the generated function body.
    """
    return generate(_program_returning(exp), {}).functions[0].body


# ---------------------------------------------------------------------------
# Helpers for body-level tests (declarations + statements)
# ---------------------------------------------------------------------------


def _instrs_for_body(body: tuple[ast.BlockItem, ...]) -> tuple[tac.Instruction, ...]:
    """Translate a full function body and return the resulting TAC instructions.

    Args:
        body: The block items to place in the body of ``main``.

    Returns:
        The TAC instruction tuple from the generated function body.
    """
    return (
        generate(
            ast.Program(
                declarations=(
                    ast.FunDecl(
                        ast.Function(
                            name="main",
                            params=(),
                            body=body,
                        )
                    ),
                )
            ),
            {},
        )
        .functions[0]
        .body
    )
