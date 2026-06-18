"""Tests for the compiler CLI driver (``compiler.compiler``).

Covers option parsing and argument forwarding for the ``-l`` library flag,
the ``-c`` object-file flag, the ``-I``/``-D``/``-U`` preprocessor passthrough
flags, and their interaction with pipeline stage flags.
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


class TestPreprocessorPassthroughFlags:
    """Tests for ``-I``, ``-D``, and ``-U`` CLI options forwarded to ``preprocess``."""

    def test_single_include_dir_forwarded(self, tmp_path: Path) -> None:
        """``-I/usr/include`` must forward ``include_dirs=('/usr/include',)`` to ``preprocess``."""
        source = tmp_path / "prog.c"
        source.touch()
        preprocessed = tmp_path / "prog.i"
        preprocessed.touch()
        mock_pre = MagicMock(return_value=preprocessed)
        mock_compile = MagicMock()

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, ["-I/usr/include", str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_pre.call_args
        assert kwargs["include_dirs"] == ("/usr/include",)

    def test_multiple_include_dirs_forwarded(self, tmp_path: Path) -> None:
        """Multiple ``-I`` flags must accumulate all paths in order."""
        source = tmp_path / "prog.c"
        source.touch()
        preprocessed = tmp_path / "prog.i"
        preprocessed.touch()
        mock_pre = MagicMock(return_value=preprocessed)
        mock_compile = MagicMock()

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, ["-I/usr/include", "-I/opt/include", str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_pre.call_args
        assert kwargs["include_dirs"] == ("/usr/include", "/opt/include")

    def test_no_include_flag_passes_empty_tuple(self, tmp_path: Path) -> None:
        """Omitting ``-I`` must pass an empty tuple as ``include_dirs``."""
        source = tmp_path / "prog.c"
        source.touch()
        preprocessed = tmp_path / "prog.i"
        preprocessed.touch()
        mock_pre = MagicMock(return_value=preprocessed)
        mock_compile = MagicMock()

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, [str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_pre.call_args
        assert kwargs["include_dirs"] == ()

    def test_single_define_forwarded(self, tmp_path: Path) -> None:
        """``-DNDEBUG`` must forward ``defines=('NDEBUG',)`` to ``preprocess``."""
        source = tmp_path / "prog.c"
        source.touch()
        preprocessed = tmp_path / "prog.i"
        preprocessed.touch()
        mock_pre = MagicMock(return_value=preprocessed)
        mock_compile = MagicMock()

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, ["-DNDEBUG", str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_pre.call_args
        assert kwargs["defines"] == ("NDEBUG",)

    def test_define_with_value_forwarded(self, tmp_path: Path) -> None:
        """``-DFOO=1`` must forward ``defines=('FOO=1',)`` to ``preprocess``."""
        source = tmp_path / "prog.c"
        source.touch()
        preprocessed = tmp_path / "prog.i"
        preprocessed.touch()
        mock_pre = MagicMock(return_value=preprocessed)
        mock_compile = MagicMock()

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, ["-DFOO=1", str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_pre.call_args
        assert kwargs["defines"] == ("FOO=1",)

    def test_multiple_defines_forwarded(self, tmp_path: Path) -> None:
        """Multiple ``-D`` flags must accumulate all names in order."""
        source = tmp_path / "prog.c"
        source.touch()
        preprocessed = tmp_path / "prog.i"
        preprocessed.touch()
        mock_pre = MagicMock(return_value=preprocessed)
        mock_compile = MagicMock()

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, ["-DNDEBUG", "-DFOO=1", str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_pre.call_args
        assert kwargs["defines"] == ("NDEBUG", "FOO=1")

    def test_no_define_flag_passes_empty_tuple(self, tmp_path: Path) -> None:
        """Omitting ``-D`` must pass an empty tuple as ``defines``."""
        source = tmp_path / "prog.c"
        source.touch()
        preprocessed = tmp_path / "prog.i"
        preprocessed.touch()
        mock_pre = MagicMock(return_value=preprocessed)
        mock_compile = MagicMock()

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, [str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_pre.call_args
        assert kwargs["defines"] == ()

    def test_single_undefine_forwarded(self, tmp_path: Path) -> None:
        """``-UNDEBUG`` must forward ``undefines=('NDEBUG',)`` to ``preprocess``."""
        source = tmp_path / "prog.c"
        source.touch()
        preprocessed = tmp_path / "prog.i"
        preprocessed.touch()
        mock_pre = MagicMock(return_value=preprocessed)
        mock_compile = MagicMock()

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, ["-UNDEBUG", str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_pre.call_args
        assert kwargs["undefines"] == ("NDEBUG",)

    def test_multiple_undefines_forwarded(self, tmp_path: Path) -> None:
        """Multiple ``-U`` flags must accumulate all names in order."""
        source = tmp_path / "prog.c"
        source.touch()
        preprocessed = tmp_path / "prog.i"
        preprocessed.touch()
        mock_pre = MagicMock(return_value=preprocessed)
        mock_compile = MagicMock()

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, ["-UFOO", "-UBAR", str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_pre.call_args
        assert kwargs["undefines"] == ("FOO", "BAR")

    def test_no_undefine_flag_passes_empty_tuple(self, tmp_path: Path) -> None:
        """Omitting ``-U`` must pass an empty tuple as ``undefines``."""
        source = tmp_path / "prog.c"
        source.touch()
        preprocessed = tmp_path / "prog.i"
        preprocessed.touch()
        mock_pre = MagicMock(return_value=preprocessed)
        mock_compile = MagicMock()

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, [str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_pre.call_args
        assert kwargs["undefines"] == ()

    def test_all_passthrough_flags_combined(self, tmp_path: Path) -> None:
        """``-I``, ``-D``, and ``-U`` may all be used together."""
        source = tmp_path / "prog.c"
        source.touch()
        preprocessed = tmp_path / "prog.i"
        preprocessed.touch()
        mock_pre = MagicMock(return_value=preprocessed)
        mock_compile = MagicMock()

        with patch("compiler.compiler.preprocess", mock_pre), patch("compiler.compiler.compile", mock_compile):
            result = CliRunner().invoke(compiler, ["-I/inc", "-DFOO", "-UBAR", str(source)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_pre.call_args
        assert kwargs["include_dirs"] == ("/inc",)
        assert kwargs["defines"] == ("FOO",)
        assert kwargs["undefines"] == ("BAR",)
