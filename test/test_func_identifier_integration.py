"""End-to-end integration tests for the ``__func__`` predefined identifier.

Compiles C source files that use ``__func__``, runs the resulting binary, and
verifies the exit code.  Each fixture encodes the expected result as a ``return``
value that exits with 0 on success.
"""

import subprocess
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "valid"

COMPILER_ENTRY = ["uv", "run", "compiler"]


def _compile_and_run(src: Path, tmp_path: Path) -> int:
    """Compile ``src`` to an executable in ``tmp_path`` and return its exit code.

    Args:
        src: Path to the C source file.
        tmp_path: Temporary directory for build artefacts.

    Returns:
        The exit code of the compiled binary.

    Raises:
        pytest.fail: If compilation itself fails.
    """
    out_bin = tmp_path / src.stem
    result = subprocess.run(
        [*COMPILER_ENTRY, str(src), "--output", str(out_bin)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        result2 = subprocess.run(
            [*COMPILER_ENTRY, str(src)],
            capture_output=True,
            text=True,
        )
        if result2.returncode != 0:
            pytest.fail(f"Compilation failed:\n{result2.stderr}")
        out_bin = src.with_suffix("")

    run = subprocess.run([str(out_bin)])
    return run.returncode


@pytest.mark.integration
class TestFuncIdentifierIntegration:
    """Full compile-and-run tests for the ``__func__`` predefined identifier."""

    def test_func_equals_function_name(self, tmp_path: Path) -> None:
        """``strcmp(__func__, "named_function")`` returns 0; binary exits 0."""
        rc = _compile_and_run(FIXTURES / "func_identifier.c", tmp_path)
        assert rc == 0
