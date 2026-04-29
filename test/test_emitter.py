"""Tests for the assembly emission stage.

Covers ``emit()`` on every assembly AST node and the platform-specific
behaviour of ``emit_program``.
"""

from pathlib import Path

import pytest

from compiler.asm import Function, Imm, Mov, Program, Register, Ret
from compiler.emitter import emit_program

# ---------------------------------------------------------------------------
# Operand emit()
# ---------------------------------------------------------------------------


def test_imm_emit() -> None:
    """``Imm(42).emit()`` produces ``$42``."""
    assert Imm(value=42).emit() == "$42"


def test_imm_emit_zero() -> None:
    """``Imm(0).emit()`` produces ``$0``."""
    assert Imm(value=0).emit() == "$0"


def test_register_emit() -> None:
    """``Register().emit()`` produces ``%eax``."""
    assert Register().emit() == "%eax"


# ---------------------------------------------------------------------------
# Instruction emit()
# ---------------------------------------------------------------------------


def test_mov_emit() -> None:
    """``Mov`` emits ``  movl <src>, <dst>``."""
    assert Mov(src=Imm(value=2), dst=Register()).emit() == "  movl $2, %eax"


def test_ret_emit() -> None:
    """``Ret`` emits ``  ret``."""
    assert Ret().emit() == "  ret"


# ---------------------------------------------------------------------------
# Function emit()
# ---------------------------------------------------------------------------


def test_function_emit_globl_directive() -> None:
    """Function emission begins with a ``.globl`` directive."""
    fn = Function(name="main", instructions=(Ret(),))
    assert fn.emit().startswith("  .globl main\n")


def test_function_emit_label() -> None:
    """Function emission includes the label on the line after ``.globl``."""
    fn = Function(name="main", instructions=(Ret(),))
    lines = fn.emit().splitlines()
    assert lines[1] == "main:"


def test_function_emit_instructions() -> None:
    """Instruction lines follow the label in order."""
    fn = Function(name="main", instructions=(Mov(src=Imm(value=2), dst=Register()), Ret()))
    lines = fn.emit().splitlines()
    assert lines[2] == "  movl $2, %eax"
    assert lines[3] == "  ret"


def test_function_emit_full() -> None:
    """Full function emission matches the expected template exactly."""
    fn = Function(name="main", instructions=(Mov(src=Imm(value=0), dst=Register()), Ret()))
    assert fn.emit() == "  .globl main\nmain:\n  movl $0, %eax\n  ret"


# ---------------------------------------------------------------------------
# emit_program — Linux
# ---------------------------------------------------------------------------


def test_emit_program_linux_content(tmp_path: Path) -> None:
    """On Linux the file contains the expected instructions."""
    program = Program(function=Function(name="main", instructions=(Mov(src=Imm(value=3), dst=Register()), Ret())))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="linux")
    content = out.read_text()
    assert "  .globl main" in content
    assert "main:" in content
    assert "  movl $3, %eax" in content
    assert "  ret" in content


def test_emit_program_linux_gnu_stack(tmp_path: Path) -> None:
    """On Linux the GNU-stack section directive is appended."""
    program = Program(function=Function(name="main", instructions=(Ret(),)))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="linux")
    content = out.read_text()
    assert '.section .note.GNU-stack,"",@progbits' in content


def test_emit_program_linux_gnu_stack_is_last_line(tmp_path: Path) -> None:
    """The GNU-stack directive is the last non-empty line of the file."""
    program = Program(function=Function(name="main", instructions=(Ret(),)))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="linux")
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert lines[-1] == '  .section .note.GNU-stack,"",@progbits'


def test_emit_program_linux_no_underscore_prefix(tmp_path: Path) -> None:
    """On Linux function names are NOT prefixed with an underscore."""
    program = Program(function=Function(name="main", instructions=(Ret(),)))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="linux")
    content = out.read_text()
    assert "_main" not in content


def test_emit_program_linux_trailing_newline(tmp_path: Path) -> None:
    """The emitted file ends with a newline character."""
    program = Program(function=Function(name="main", instructions=(Ret(),)))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="linux")
    assert out.read_text().endswith("\n")


# ---------------------------------------------------------------------------
# emit_program — macOS
# ---------------------------------------------------------------------------


def test_emit_program_macos_underscore_prefix(tmp_path: Path) -> None:
    """On macOS the function name is prefixed with ``_``."""
    program = Program(function=Function(name="main", instructions=(Ret(),)))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="darwin")
    content = out.read_text()
    assert "  .globl _main" in content
    assert "_main:" in content


def test_emit_program_macos_no_gnu_stack(tmp_path: Path) -> None:
    """On macOS the GNU-stack section directive is NOT emitted."""
    program = Program(function=Function(name="main", instructions=(Ret(),)))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="darwin")
    assert ".note.GNU-stack" not in out.read_text()


def test_emit_program_macos_original_node_unchanged(tmp_path: Path) -> None:
    """``emit_program`` does not mutate the original ``Program`` node."""
    program = Program(function=Function(name="main", instructions=(Ret(),)))
    emit_program(program, tmp_path / "main.s", platform="darwin")
    assert program.function.name == "main"


def test_emit_program_writes_file(tmp_path: Path) -> None:
    """``emit_program`` creates the output file."""
    program = Program(function=Function(name="main", instructions=(Ret(),)))
    out = tmp_path / "out.s"
    assert not out.exists()
    emit_program(program, out, platform="linux")
    assert out.exists()


def test_emit_program_bad_path_raises(tmp_path: Path) -> None:
    """``emit_program`` raises ``OSError`` if the output path is not writable."""
    program = Program(function=Function(name="main", instructions=(Ret(),)))
    with pytest.raises(OSError):
        emit_program(program, tmp_path / "nonexistent_dir" / "out.s", platform="linux")
