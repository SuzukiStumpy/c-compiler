"""End-to-end integration tests for variadic function support.

These tests compile C source files that use ``va_list``, ``va_start``, ``va_arg``,
and ``va_end``, run the resulting binary, and verify the exit code.  Each fixture
encodes the expected result as a ``return`` value that exits with 0 on success.
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
    # Fall back: if --output not supported, binary lands next to source.
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
class TestVariadicIntegration:
    """Full compile-and-run tests for variadic function definitions."""

    def test_int_sum_via_va_arg(self, tmp_path: Path) -> None:
        """``sum(3, 10, 20, 30)`` returns 60; binary exits 0 (60 - 60 = 0)."""
        rc = _compile_and_run(FIXTURES / "variadic_sum.c", tmp_path)
        assert rc == 0

    def test_double_sum_via_va_arg(self, tmp_path: Path) -> None:
        """``dsum(3, 1.5, 2.5, 6.0)`` returns 10.0; binary exits 0."""
        rc = _compile_and_run(FIXTURES / "variadic_double.c", tmp_path)
        assert rc == 0

    def test_overflow_area_va_arg(self, tmp_path: Path) -> None:
        """7th arg (past 6 GP registers) is fetched from overflow area; exits 0."""
        rc = _compile_and_run(FIXTURES / "variadic_overflow.c", tmp_path)
        assert rc == 0
