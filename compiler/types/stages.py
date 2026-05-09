"""Compilation stage flags used to control pipeline execution."""

from enum import IntFlag, auto


class Stage(IntFlag):
    """Bitmask controlling which compilation stages are executed.

    Flags may be combined with ``|`` to run multiple stages. The ``ALL``
    member combines every stage flag for a complete compilation run. Omitting
    a flag causes the pipeline to stop before that stage.

    ``VALIDATE`` sits directly before the ``TACKY`` stage and performs semantic
    analysis on the generated AST.

    ``TACKY`` sits between ``VALIDATE`` and ``CODEGEN`` and represents the
    intermediate three-address-code (TAC) generation pass.

    The ``LINK`` stage distinguishes a full compilation run (which assembles
    and links) from ``--codegen`` (which stops after generating the assembly
    AST).  Without this distinction the ``--codegen`` bitmask would be
    identical to ``ALL``, making it impossible to detect inside ``compile()``.
    """

    LEX = auto()
    PARSE = auto()
    VALIDATE = auto()
    TACKY = auto()
    CODEGEN = auto()
    LINK = auto()
    ALL = LEX | PARSE | VALIDATE | TACKY | CODEGEN | LINK
