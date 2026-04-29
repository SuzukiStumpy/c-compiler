"""Compilation stage flags used to control pipeline execution."""

from enum import IntFlag, auto


class Stage(IntFlag):
    """Bitmask controlling which compilation stages are executed.

    Flags may be combined with ``|`` to run multiple stages. The ``ALL``
    member combines every stage flag for a complete compilation run. Omitting
    a flag causes the pipeline to stop before that stage.

    The ``LINK`` stage distinguishes a full compilation run (which assembles
    and links) from ``--codegen`` (which stops after generating the assembly
    AST).  Without this distinction the ``--codegen`` bitmask would be
    identical to ``ALL``, making it impossible to detect inside ``compile()``.
    """

    LEX = auto()
    PARSE = auto()
    CODEGEN = auto()
    LINK = auto()
    ALL = LEX | PARSE | CODEGEN | LINK
