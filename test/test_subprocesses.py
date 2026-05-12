"""Tests for the subprocess wrappers in ``compiler.subprocesses``.

Covers ``assemble_and_link`` in both executable and object-file modes, including
the gcc argument list, output path, and error-path behaviour.
"""

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from compiler.subprocesses import assemble_and_link
from compiler.utils.exceptions import SubprocessError


def _make_run(returncode: int, stdout: str = "") -> MagicMock:
    """Return a mock that mimics ``subprocess.run`` for the given exit code.

    Args:
        returncode: The simulated process exit code.
        stdout: The simulated captured stdout/stderr text.

    Returns:
        A ``MagicMock`` whose return value looks like a ``CompletedProcess``.
    """
    result = MagicMock(spec=CompletedProcess)
    result.returncode = returncode
    result.stdout = stdout
    mock = MagicMock(return_value=result)
    return mock


class TestAssembleAndLink:
    """Tests for ``assemble_and_link``."""

    def test_exe_out_true_calls_gcc_without_dash_c(self, tmp_path: Path) -> None:
        """When ``exe_out=True``, gcc must not receive the ``-c`` flag."""
        asm_file = tmp_path / "prog.s"
        asm_file.touch()

        mock_run = _make_run(0)
        with patch("compiler.subprocesses.subprocess.run", mock_run):
            assemble_and_link(asm_file, exe_out=True)

        args = mock_run.call_args[0][0]
        assert "-c" not in args

    def test_exe_out_true_returns_path_without_extension(self, tmp_path: Path) -> None:
        """When ``exe_out=True``, the returned path has no file extension."""
        asm_file = tmp_path / "prog.s"
        asm_file.touch()

        mock_run = _make_run(0)
        with patch("compiler.subprocesses.subprocess.run", mock_run):
            result = assemble_and_link(asm_file, exe_out=True)

        assert result == tmp_path / "prog"
        assert result.suffix == ""

    def test_exe_out_false_calls_gcc_with_dash_c(self, tmp_path: Path) -> None:
        """When ``exe_out=False``, gcc must receive the ``-c`` flag."""
        asm_file = tmp_path / "prog.s"
        asm_file.touch()

        mock_run = _make_run(0)
        with patch("compiler.subprocesses.subprocess.run", mock_run):
            assemble_and_link(asm_file, exe_out=False)

        args = mock_run.call_args[0][0]
        assert "-c" in args

    def test_exe_out_false_returns_dot_o_path(self, tmp_path: Path) -> None:
        """When ``exe_out=False``, the returned path has a ``.o`` extension."""
        asm_file = tmp_path / "prog.s"
        asm_file.touch()

        mock_run = _make_run(0)
        with patch("compiler.subprocesses.subprocess.run", mock_run):
            result = assemble_and_link(asm_file, exe_out=False)

        assert result == tmp_path / "prog.o"
        assert result.suffix == ".o"

    def test_exe_out_default_is_true(self, tmp_path: Path) -> None:
        """Calling ``assemble_and_link`` without ``exe_out`` defaults to executable mode."""
        asm_file = tmp_path / "prog.s"
        asm_file.touch()

        mock_run = _make_run(0)
        with patch("compiler.subprocesses.subprocess.run", mock_run):
            result = assemble_and_link(asm_file)

        args = mock_run.call_args[0][0]
        assert "-c" not in args
        assert result.suffix == ""

    def test_gcc_failure_raises_subprocess_error(self, tmp_path: Path) -> None:
        """A non-zero gcc exit code must raise ``SubprocessError``."""
        asm_file = tmp_path / "prog.s"
        asm_file.touch()

        mock_run = _make_run(1, stdout="prog.s: Assembler messages: Error")
        with patch("compiler.subprocesses.subprocess.run", mock_run):
            with pytest.raises(SubprocessError):
                assemble_and_link(asm_file)

    def test_no_empty_string_args_in_exe_mode(self, tmp_path: Path) -> None:
        """The gcc argument list must contain no empty strings in executable mode."""
        asm_file = tmp_path / "prog.s"
        asm_file.touch()

        mock_run = _make_run(0)
        with patch("compiler.subprocesses.subprocess.run", mock_run):
            assemble_and_link(asm_file, exe_out=True)

        args = mock_run.call_args[0][0]
        assert "" not in args

    def test_no_empty_string_args_in_object_mode(self, tmp_path: Path) -> None:
        """The gcc argument list must contain no empty strings in object-file mode."""
        asm_file = tmp_path / "prog.s"
        asm_file.touch()

        mock_run = _make_run(0)
        with patch("compiler.subprocesses.subprocess.run", mock_run):
            assemble_and_link(asm_file, exe_out=False)

        args = mock_run.call_args[0][0]
        assert "" not in args
