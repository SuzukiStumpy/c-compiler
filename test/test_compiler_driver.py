"""Tests for the compiler CLI driver (``compiler.compiler``).

Covers option parsing and argument forwarding for the ``-l`` library flag,
the ``-c`` object-file flag, and their interaction with pipeline stage flags.
The subprocess layer and internal pipeline are mocked so these tests exercise
only the Click layer and the ``compiler()`` command function.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from compiler.compiler import compiler


def _make_mocks(tmp_path: Path) -> tuple[MagicMock, MagicMock]:
    """Return (mock_preprocess, mock_compile) pre-wired for a successful run.

    Args:
        tmp_path: A temporary directory; the mock preprocessor returns a ``.i``
            file inside it so downstream path operations do not fail.

    Returns:
        A 2-tuple of ``(mock_preprocess, mock_compile)`` ``MagicMock`` objects.
    """
    preprocessed = tmp_path / "prog.i"
    preprocessed.touch()
    mock_preprocess = MagicMock(return_value=preprocessed)
    mock_compile = MagicMock()
    return mock_preprocess, mock_compile


class TestLibraryFlag:
    """Tests for the ``-l`` / ``libraries`` CLI option."""

    def test_single_lib_concatenated_form(self, tmp_path: Path) -> None:
        """``-lm`` must forward ``('m',)`` as the ``libs`` argument to ``compile``."""
        source = tmp_path / "prog.c"
        source.touch()
        mock_pre, mock_compile = _make_mocks(tmp_path)

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, ["-lm", str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_compile.call_args
        assert kwargs["libs"] == ("m",)

    def test_single_lib_spaced_form(self, tmp_path: Path) -> None:
        """``-l m`` (space-separated) must also forward ``('m',)``."""
        source = tmp_path / "prog.c"
        source.touch()
        mock_pre, mock_compile = _make_mocks(tmp_path)

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, ["-l", "m", str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_compile.call_args
        assert kwargs["libs"] == ("m",)

    def test_multiple_libs_forwarded(self, tmp_path: Path) -> None:
        """Repeating ``-l`` must accumulate all names in order."""
        source = tmp_path / "prog.c"
        source.touch()
        mock_pre, mock_compile = _make_mocks(tmp_path)

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, ["-lm", "-lpthread", str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_compile.call_args
        assert kwargs["libs"] == ("m", "pthread")

    def test_no_lib_flag_passes_empty_tuple(self, tmp_path: Path) -> None:
        """Omitting ``-l`` entirely must pass an empty tuple as ``libs``."""
        source = tmp_path / "prog.c"
        source.touch()
        mock_pre, mock_compile = _make_mocks(tmp_path)

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, [str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_compile.call_args
        assert kwargs["libs"] == ()

    def test_lib_flag_with_codegen_stage(self, tmp_path: Path) -> None:
        """``-lm --codegen`` must still parse without error (libs are unused but accepted)."""
        source = tmp_path / "prog.c"
        source.touch()
        mock_pre, mock_compile = _make_mocks(tmp_path)

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, ["-lm", "--codegen", str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_compile.call_args
        assert kwargs["libs"] == ("m",)


class TestObjectFileFlag:
    """Tests for the ``-c`` flag (object-file mode)."""

    def test_c_flag_sets_exe_out_false(self, tmp_path: Path) -> None:
        """``-c`` must pass ``exe_out=False`` to ``compile``."""
        source = tmp_path / "prog.c"
        source.touch()
        mock_pre, mock_compile = _make_mocks(tmp_path)

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, ["-c", str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_compile.call_args
        assert kwargs["exe_out"] is False

    def test_no_c_flag_sets_exe_out_true(self, tmp_path: Path) -> None:
        """Omitting ``-c`` must pass ``exe_out=True`` to ``compile``."""
        source = tmp_path / "prog.c"
        source.touch()
        mock_pre, mock_compile = _make_mocks(tmp_path)

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, [str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_compile.call_args
        assert kwargs["exe_out"] is True

    def test_c_and_lib_flags_combined(self, tmp_path: Path) -> None:
        """``-c`` and ``-lm`` may coexist; both must be forwarded correctly."""
        source = tmp_path / "prog.c"
        source.touch()
        mock_pre, mock_compile = _make_mocks(tmp_path)

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, ["-c", "-lm", str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_compile.call_args
        assert kwargs["exe_out"] is False
        assert kwargs["libs"] == ("m",)
