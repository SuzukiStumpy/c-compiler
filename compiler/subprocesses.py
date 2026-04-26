"""Subprocess wrappers for external tools invoked by the compiler pipeline."""

import subprocess
from pathlib import Path

from compiler.utils.decorators import run_or_raise
from compiler.utils.exceptions import SubprocessError


@run_or_raise(
    lambda result: SubprocessError(result[0], "gcc", result[1]),
    ok=lambda result: isinstance(result, Path),
)
def preprocess(source: Path) -> tuple[int, str] | Path:
    """Run the GCC preprocessor on a C source file.

    Invokes ``gcc -E`` to expand macros and include directives, writing the
    result to a ``.i`` file in the same directory as the source. Linemarkers
    are preserved (no ``-P`` flag) so the lexer can resolve token positions
    back to the original source file.

    Args:
        source: Path to the ``.c`` file to preprocess.

    Returns:
        The path to the generated ``.i`` file on success.

    Raises:
        SubprocessError: If GCC exits with a non-zero return code.
    """
    file_path = source.parent
    file_name_no_ext = source.stem

    print(f"Preprocessing {source}...")

    r = subprocess.run(
        [
            "gcc",
            "-E",
            source.as_posix(),
            "-o",
            (Path(file_path) / f"{file_name_no_ext}.i").as_posix(),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if r.returncode != 0:
        return (r.returncode, r.stdout)
    else:
        return Path(file_path) / f"{file_name_no_ext}.i"


@run_or_raise(
    lambda result: SubprocessError(result[0], "gcc", result[1]),
    ok=lambda result: isinstance(result, Path),
)
def assemble_and_link(source: Path) -> tuple[int, str] | Path:
    """Assemble and link a generated assembly file into an executable.

    Invokes ``gcc`` on the given ``.s`` file to produce a native executable
    at the same path without the file extension.

    Args:
        source: Path to the ``.s`` assembly file to assemble and link.

    Returns:
        The path to the generated executable on success.

    Raises:
        SubprocessError: If GCC exits with a non-zero return code.
    """
    print(f"Assembling and linking {source}...")

    r = subprocess.run(
        [
            "gcc",
            source.as_posix(),
            "-o",
            (source.parent / source.stem).as_posix(),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if r.returncode != 0:
        return (r.returncode, r.stdout)
    else:
        # Path.unlink(source)
        return source.parent / source.stem
