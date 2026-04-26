"""Compilation stage flags used to control pipeline execution."""

from enum import IntFlag, auto


class Stage(IntFlag):
    """Bitmask controlling which compilation stages are executed.

    Flags may be combined with ``|`` to run multiple stages. The ``ALL``
    member combines every stage flag for a complete compilation run. Omitting
    a flag causes the pipeline to stop before that stage.
    """

    LEX = auto()
    PARSE = auto()
    CODEGEN = auto()
    ALL = LEX | PARSE | CODEGEN
