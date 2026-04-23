from enum import IntFlag, auto


class Stage(IntFlag):
    """Compilation stages, used as a bitmask to control which stages are run."""

    LEX = auto()
    PARSE = auto()
    CODEGEN = auto()
    ALL = LEX | PARSE | CODEGEN
