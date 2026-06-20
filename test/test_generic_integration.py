"""End-to-end integration tests for ``_Generic`` selection expressions (C11 §6.5.1.1).

Compiles and runs C programs that use ``_Generic``; verifies correct branch selection
by inspecting the binary exit code.
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
class TestGenericIntegration:
    """Full compile-and-run tests for ``_Generic`` expressions."""

    def test_int_branch_selected(self, tmp_path: Path) -> None:
        """``_Generic(x, int: 0, double: 1, default: 2)`` with int ``x`` returns 0."""
        rc = _compile_and_run(FIXTURES / "generic_select_int.c", tmp_path)
        assert rc == 0
