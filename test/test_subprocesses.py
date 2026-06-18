"""Tests for the subprocess wrappers in ``compiler.subprocesses``.

Covers ``preprocess`` (include/define/undefine passthrough) and
``assemble_and_link`` in both executable and object-file modes, including
the gcc argument list, output path, and error-path behaviour.
"""

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from compiler.subprocesses import assemble_and_link, preprocess
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


class TestPreprocess:
    """Tests for ``preprocess`` passthrough flags."""

    def test_no_flags_calls_gcc_e(self, tmp_path: Path) -> None:
        """Without extra flags, gcc receives ``-E`` and nothing else preprocessor-specific."""
        src = tmp_path / "prog.c"
        src.touch()
        mock_run = _make_run(0)

        with patch("compiler.subprocesses.subprocess.run", mock_run):
            preprocess(src)

        args = mock_run.call_args[0][0]
        assert "gcc" in args
        assert "-E" in args

    def test_single_include_dir_appears_in_args(self, tmp_path: Path) -> None:
        """A single ``include_dirs`` entry must produce a ``-I<path>`` gcc argument."""
        src = tmp_path / "prog.c"
        src.touch()
        mock_run = _make_run(0)

        with patch("compiler.subprocesses.subprocess.run", mock_run):
            preprocess(src, include_dirs=("/usr/local/include",))

        args = mock_run.call_args[0][0]
        assert "-I/usr/local/include" in args

    def test_multiple_include_dirs_all_appear(self, tmp_path: Path) -> None:
        """Every entry in ``include_dirs`` must produce its own ``-I<path>`` argument."""
        src = tmp_path / "prog.c"
        src.touch()
        mock_run = _make_run(0)

        with patch("compiler.subprocesses.subprocess.run", mock_run):
            preprocess(src, include_dirs=("/usr/include", "/opt/include"))

        args = mock_run.call_args[0][0]
        assert "-I/usr/include" in args
        assert "-I/opt/include" in args

    def test_single_define_appears_in_args(self, tmp_path: Path) -> None:
        """A single ``defines`` entry must produce a ``-D<name>`` gcc argument."""
        src = tmp_path / "prog.c"
        src.touch()
        mock_run = _make_run(0)

        with patch("compiler.subprocesses.subprocess.run", mock_run):
            preprocess(src, defines=("NDEBUG",))

        args = mock_run.call_args[0][0]
        assert "-DNDEBUG" in args

    def test_define_with_value_appears_in_args(self, tmp_path: Path) -> None:
        """A ``defines`` entry of the form ``NAME=VALUE`` must produce ``-DNAME=VALUE``."""
        src = tmp_path / "prog.c"
        src.touch()
        mock_run = _make_run(0)

        with patch("compiler.subprocesses.subprocess.run", mock_run):
            preprocess(src, defines=("FOO=1",))

        args = mock_run.call_args[0][0]
        assert "-DFOO=1" in args

    def test_multiple_defines_all_appear(self, tmp_path: Path) -> None:
        """Every entry in ``defines`` must produce its own ``-D<name>`` argument."""
        src = tmp_path / "prog.c"
        src.touch()
        mock_run = _make_run(0)

        with patch("compiler.subprocesses.subprocess.run", mock_run):
            preprocess(src, defines=("NDEBUG", "FOO=1"))

        args = mock_run.call_args[0][0]
        assert "-DNDEBUG" in args
        assert "-DFOO=1" in args

    def test_single_undefine_appears_in_args(self, tmp_path: Path) -> None:
        """A single ``undefines`` entry must produce a ``-U<name>`` gcc argument."""
        src = tmp_path / "prog.c"
        src.touch()
        mock_run = _make_run(0)

        with patch("compiler.subprocesses.subprocess.run", mock_run):
            preprocess(src, undefines=("NDEBUG",))

        args = mock_run.call_args[0][0]
        assert "-UNDEBUG" in args

    def test_multiple_undefines_all_appear(self, tmp_path: Path) -> None:
        """Every entry in ``undefines`` must produce its own ``-U<name>`` argument."""
        src = tmp_path / "prog.c"
        src.touch()
        mock_run = _make_run(0)

        with patch("compiler.subprocesses.subprocess.run", mock_run):
            preprocess(src, undefines=("FOO", "BAR"))

        args = mock_run.call_args[0][0]
        assert "-UFOO" in args
        assert "-UBAR" in args

    def test_no_flags_produces_no_extra_args(self, tmp_path: Path) -> None:
        """With no passthrough flags, no ``-I``, ``-D``, or ``-U`` args appear."""
        src = tmp_path / "prog.c"
        src.touch()
        mock_run = _make_run(0)

        with patch("compiler.subprocesses.subprocess.run", mock_run):
            preprocess(src)

        args = mock_run.call_args[0][0]
        assert not any(a.startswith("-I") for a in args)
        assert not any(a.startswith("-D") for a in args)
        assert not any(a.startswith("-U") for a in args)

    def test_all_flags_combined(self, tmp_path: Path) -> None:
        """``include_dirs``, ``defines``, and ``undefines`` may all be used together."""
        src = tmp_path / "prog.c"
        src.touch()
        mock_run = _make_run(0)

        with patch("compiler.subprocesses.subprocess.run", mock_run):
            preprocess(src, include_dirs=("/inc",), defines=("FOO",), undefines=("BAR",))

        args = mock_run.call_args[0][0]
        assert "-I/inc" in args
        assert "-DFOO" in args
        assert "-UBAR" in args

    def test_failure_raises_subprocess_error(self, tmp_path: Path) -> None:
        """A non-zero gcc exit code must raise ``SubprocessError``."""
        src = tmp_path / "prog.c"
        src.touch()
        mock_run = _make_run(1, stdout="prog.c:1: error: something")

        with patch("compiler.subprocesses.subprocess.run", mock_run):
            with pytest.raises(SubprocessError):
                preprocess(src)

    def test_output_path_is_dot_i_file(self, tmp_path: Path) -> None:
        """On success, the returned path has a ``.i`` extension in the same directory."""
        src = tmp_path / "prog.c"
        src.touch()
        mock_run = _make_run(0)

        with patch("compiler.subprocesses.subprocess.run", mock_run):
            result = preprocess(src)

        assert result == tmp_path / "prog.i"
        assert result.suffix == ".i"


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

    def test_single_lib_appears_in_gcc_args(self, tmp_path: Path) -> None:
        """A single entry in ``libs`` must produce a ``-l<name>`` gcc argument."""
        asm_file = tmp_path / "prog.s"
        asm_file.touch()

        mock_run = _make_run(0)
        with patch("compiler.subprocesses.subprocess.run", mock_run):
            assemble_and_link(asm_file, exe_out=True, libs=("m",))

        args = mock_run.call_args[0][0]
        assert "-lm" in args

    def test_multiple_libs_all_appear_in_gcc_args(self, tmp_path: Path) -> None:
        """Every entry in ``libs`` must produce its own ``-l<name>`` gcc argument."""
        asm_file = tmp_path / "prog.s"
        asm_file.touch()

        mock_run = _make_run(0)
        with patch("compiler.subprocesses.subprocess.run", mock_run):
            assemble_and_link(asm_file, exe_out=True, libs=("m", "pthread"))

        args = mock_run.call_args[0][0]
        assert "-lm" in args
        assert "-lpthread" in args

    def test_no_libs_produces_no_dash_l_args(self, tmp_path: Path) -> None:
        """When ``libs`` is empty, no ``-l`` argument must appear in the gcc call."""
        asm_file = tmp_path / "prog.s"
        asm_file.touch()

        mock_run = _make_run(0)
        with patch("compiler.subprocesses.subprocess.run", mock_run):
            assemble_and_link(asm_file, exe_out=True, libs=())

        args = mock_run.call_args[0][0]
        assert not any(a.startswith("-l") for a in args)

    def test_libs_ignored_in_object_file_mode(self, tmp_path: Path) -> None:
        """``libs`` must be silently ignored when ``exe_out=False``; no ``-l`` args emitted."""
        asm_file = tmp_path / "prog.s"
        asm_file.touch()

        mock_run = _make_run(0)
        with patch("compiler.subprocesses.subprocess.run", mock_run):
            assemble_and_link(asm_file, exe_out=False, libs=("m",))

        args = mock_run.call_args[0][0]
        assert not any(a.startswith("-l") for a in args)
