"""Statement nodes for the AST.

Defines the ``Statement`` abstract base class and its concrete variants. Each
statement form supported by the compiler is represented as a subclass of
``Statement``. New statement types (if, while, for, assignments, etc.) are
added here as language support is extended.
"""

from abc import ABC
from dataclasses import dataclass

from compiler.ast.expressions import Exp


class Statement(ABC):
    """Abstract base for all statement nodes.

    Each statement form supported by the compiler is represented as a
    concrete subclass. Consumers should use structural pattern matching
    (``match``/``case``) to dispatch on the concrete variant rather than
    inspecting fields directly.
    """


@dataclass(frozen=True, slots=True)
class Return(Statement):
    """A ``return`` statement.

    Attributes:
        exp: The expression whose value is returned to the caller.
    """

    exp: Exp
