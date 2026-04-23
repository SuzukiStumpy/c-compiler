"""Functions which instantiate subprocesses called from the main compiler"""

import subprocess
from pathlib import Path

from compiler.utils.decorators import run_or_raise
from compiler.utils.exceptions import SubprocessError


@run_or_raise(
    lambda result: SubprocessError(result[0], "gcc", result[1]),
    ok=lambda result: isinstance(result, Path),
)
def preprocess(source: Path) -> tuple[int, str] | Path:

    file_path = source.parent
    file_name_no_ext = source.stem

    print(f"Preprocessing {source}...")

    r = subprocess.run(
        [
            "gcc",
            "-E",
            "-P",
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
