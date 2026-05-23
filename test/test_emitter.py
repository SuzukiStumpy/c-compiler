"""Tests for the assembly emission stage.

Covers ``emit()`` on every assembly AST node and the platform-specific
behaviour of ``emit_program``.
"""

from pathlib import Path

import pytest

from compiler.emitter import emit_program
from compiler.syntax.asm import (
    AllocateStack,
    Call,
    Cmp,
    CondCodes,
    DeallocateStack,
    Function,
    Imm,
    Jmp,
    JmpCC,
    Label,
    Memory,
    Mov,
    Program,
    Push,
    Register,
    Registers,
    Ret,
    SetCC,
    Width,
)

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
    """``Register(Registers.AX).emit()`` produces ``%eax``."""
    assert Register(Registers.AX).emit() == "%eax"


@pytest.mark.parametrize(
    ("reg", "width", "expected"),
    [
        (Registers.AX, Width.DWORD, "%eax"),
        (Registers.AX, Width.BYTE, "%al"),
        (Registers.CX, Width.DWORD, "%ecx"),
        (Registers.CX, Width.BYTE, "%cl"),
        (Registers.DX, Width.DWORD, "%edx"),
        (Registers.DX, Width.BYTE, "%dl"),
        (Registers.R10, Width.DWORD, "%r10d"),
        (Registers.R10, Width.BYTE, "%r10b"),
        (Registers.R11, Width.DWORD, "%r11d"),
        (Registers.R11, Width.BYTE, "%r11b"),
    ],
    ids=[
        "ax-dword",
        "ax-byte",
        "cx-dword",
        "cx-byte",
        "dx-dword",
        "dx-byte",
        "r10-dword",
        "r10-byte",
        "r11-dword",
        "r11-byte",
    ],
)
def test_register_emit_all_widths(reg: Registers, width: Width, expected: str) -> None:
    """``Register.emit()`` returns the correct AT&T name for every register/width combination."""
    assert Register(reg, width).emit() == expected


# ---------------------------------------------------------------------------
# Instruction emit()
# ---------------------------------------------------------------------------


def test_mov_emit() -> None:
    """``Mov`` emits ``  movl <src>, <dst>``."""
    assert Mov(src=Imm(value=2), dst=Register(Registers.AX)).emit() == "  movl $2, %eax"


def test_ret_emit() -> None:
    """``Ret`` emits ``  ret``."""
    assert Ret().emit() == "  movq  %rbp, %rsp\n  popq  %rbp\n  ret"


# ---------------------------------------------------------------------------
# Function emit()
# ---------------------------------------------------------------------------


def test_function_emit_globl_directive() -> None:
    """Function emission begins with a ``.globl`` directive."""
    fn = Function(name="main", globl=True, instructions=(Ret(),))
    assert fn.emit().startswith("  .globl main\n")


def test_function_emit_label() -> None:
    """Function emission includes the label after the ``.globl`` and ``.text`` directives."""
    fn = Function(name="main", globl=True, instructions=(Ret(),))
    lines = fn.emit().splitlines()
    assert lines[2] == "main:"


def test_function_emit_instructions() -> None:
    """Instruction lines follow the prologue in order."""
    fn = Function(name="main", globl=True, instructions=(Mov(src=Imm(value=2), dst=Register(Registers.AX)), Ret()))
    lines = fn.emit().splitlines()
    # lines[0] = .globl, [1] = .text, [2] = label, [3] = pushq, [4] = movq (prologue)
    assert lines[5] == "  movl $2, %eax"
    assert lines[8] == "  ret"


def test_function_emit_full() -> None:
    """Full function emission matches the expected template exactly."""
    fn = Function(name="main", globl=True, instructions=(Mov(src=Imm(value=0), dst=Register(Registers.AX)), Ret()))
    expected = (
        "  .globl main\n"
        "   .text\n"
        "main:\n"
        "  pushq %rbp\n"
        "  movq %rsp, %rbp\n"
        "  movl $0, %eax\n"
        "  movq  %rbp, %rsp\n"
        "  popq  %rbp\n"
        "  ret"
    )
    assert fn.emit() == expected


# ---------------------------------------------------------------------------
# emit_program — Linux
# ---------------------------------------------------------------------------


def test_emit_program_linux_content(tmp_path: Path) -> None:
    """On Linux the file contains the expected instructions."""
    mov = Mov(src=Imm(value=3), dst=Register(Registers.AX))
    program = Program(definitions=(Function(name="main", globl=True, instructions=(mov, Ret())),))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="linux")
    content = out.read_text()
    assert "  .globl main" in content
    assert "main:" in content
    assert "  movl $3, %eax" in content
    assert "  ret" in content


def test_emit_program_linux_gnu_stack(tmp_path: Path) -> None:
    """On Linux the GNU-stack section directive is appended."""
    program = Program(definitions=(Function(name="main", globl=True, instructions=(Ret(),)),))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="linux")
    content = out.read_text()
    assert '.section .note.GNU-stack,"",@progbits' in content


def test_emit_program_linux_gnu_stack_is_last_line(tmp_path: Path) -> None:
    """The GNU-stack directive is the last non-empty line of the file."""
    program = Program(definitions=(Function(name="main", globl=True, instructions=(Ret(),)),))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="linux")
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert lines[-1] == '  .section .note.GNU-stack,"",@progbits'


def test_emit_program_linux_no_underscore_prefix(tmp_path: Path) -> None:
    """On Linux function names are NOT prefixed with an underscore."""
    program = Program(definitions=(Function(name="main", globl=True, instructions=(Ret(),)),))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="linux")
    content = out.read_text()
    assert "_main" not in content


def test_emit_program_linux_trailing_newline(tmp_path: Path) -> None:
    """The emitted file ends with a newline character."""
    program = Program(definitions=(Function(name="main", globl=True, instructions=(Ret(),)),))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="linux")
    assert out.read_text().endswith("\n")


# ---------------------------------------------------------------------------
# emit_program — macOS
# ---------------------------------------------------------------------------


def test_emit_program_macos_underscore_prefix(tmp_path: Path) -> None:
    """On macOS the function name is prefixed with ``_``."""
    program = Program(definitions=(Function(name="main", globl=True, instructions=(Ret(),)),))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="darwin")
    content = out.read_text()
    assert "  .globl _main" in content
    assert "_main:" in content


def test_emit_program_macos_no_gnu_stack(tmp_path: Path) -> None:
    """On macOS the GNU-stack section directive is NOT emitted."""
    program = Program(definitions=(Function(name="main", globl=True, instructions=(Ret(),)),))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="darwin")
    assert ".note.GNU-stack" not in out.read_text()


def test_emit_program_macos_original_node_unchanged(tmp_path: Path) -> None:
    """``emit_program`` does not mutate the original ``Program`` node."""
    program = Program(definitions=(Function(name="main", globl=True, instructions=(Ret(),)),))
    emit_program(program, tmp_path / "main.s", platform="darwin")
    assert program.definitions[0].name == "main"


def test_emit_program_writes_file(tmp_path: Path) -> None:
    """``emit_program`` creates the output file."""
    program = Program(definitions=(Function(name="main", globl=True, instructions=(Ret(),)),))
    out = tmp_path / "out.s"
    assert not out.exists()
    emit_program(program, out, platform="linux")
    assert out.exists()


def test_emit_program_bad_path_raises(tmp_path: Path) -> None:
    """``emit_program`` raises ``OSError`` if the output path is not writable."""
    program = Program(definitions=(Function(name="main", globl=True, instructions=(Ret(),)),))
    with pytest.raises(OSError):
        emit_program(program, tmp_path / "nonexistent_dir" / "out.s", platform="linux")


# ---------------------------------------------------------------------------
# New instruction emit()
# ---------------------------------------------------------------------------


def test_cmp_emit() -> None:
    """``Cmp`` emits ``  cmpl <op1>, <op2>``."""
    assert Cmp(op1=Imm(value=1), op2=Register(Registers.AX)).emit() == "  cmpl $1, %eax"


def test_cmp_emit_memory_operands() -> None:
    """``Cmp`` with a stack op2 emits the correct memory reference."""
    assert (
        Cmp(op1=Imm(value=0), op2=Memory(reg=Register(Registers.BP, Width.QWORD), offset=-4)).emit()
        == "  cmpl $0, -4(%rbp)"
    )


def test_jmp_emit() -> None:
    """``Jmp`` emits ``  jmp .L<label>``."""
    assert Jmp(label="my_label").emit() == "  jmp .Lmy_label"


@pytest.mark.parametrize(
    ("cond_code", "expected"),
    [
        (CondCodes.E, "  je .Lfoo"),
        (CondCodes.NE, "  jne .Lfoo"),
        (CondCodes.L, "  jl .Lfoo"),
        (CondCodes.LE, "  jle .Lfoo"),
        (CondCodes.G, "  jg .Lfoo"),
        (CondCodes.GE, "  jge .Lfoo"),
    ],
    ids=["e", "ne", "l", "le", "g", "ge"],
)
def test_jmpcc_emit(cond_code: CondCodes, expected: str) -> None:
    """``JmpCC`` emits ``  j<cc> .L<label>`` for every condition code."""
    assert JmpCC(cond_code=cond_code, label="foo").emit() == expected


@pytest.mark.parametrize(
    ("cond_code", "expected"),
    [
        (CondCodes.E, "  sete %al"),
        (CondCodes.NE, "  setne %al"),
        (CondCodes.L, "  setl %al"),
        (CondCodes.LE, "  setle %al"),
        (CondCodes.G, "  setg %al"),
        (CondCodes.GE, "  setge %al"),
    ],
    ids=["e", "ne", "l", "le", "g", "ge"],
)
def test_setcc_emit(cond_code: CondCodes, expected: str) -> None:
    """``SetCC`` emits ``  set<cc> <operand>`` for every condition code."""
    assert SetCC(cond_code=cond_code, operand=Register(Registers.AX, Width.BYTE)).emit() == expected


def test_setcc_emit_stack_operand() -> None:
    """``SetCC`` with a stack operand emits the memory reference correctly."""
    assert (
        SetCC(cond_code=CondCodes.E, operand=Memory(reg=Register(Registers.BP, Width.QWORD), offset=-4)).emit()
        == "  sete -4(%rbp)"
    )


def test_label_emit() -> None:
    """``Label`` emits ``.L<name>:`` flush with the left margin."""
    assert Label(name="loop_start").emit() == ".Lloop_start:"


def test_push_emit() -> None:
    """``Push`` emits ``  pushq <operand>``."""
    assert Push(operand=Register(Registers.AX, Width.QWORD)).emit() == "  pushq %rax"


def test_allocate_stack_emit() -> None:
    """``AllocateStack`` emits ``  subq $<size>, %rsp``."""
    assert AllocateStack(size=16).emit() == "  subq $16, %rsp"


def test_deallocate_stack_emit() -> None:
    """``DeallocateStack`` emits ``  addq $<size>, %rsp``."""
    assert DeallocateStack(size=16).emit() == "  addq $16, %rsp"


def test_call_emit() -> None:
    """``Call`` emits ``  call <name>``."""
    assert Call(name="foo").emit() == "  call foo"


# ---------------------------------------------------------------------------
# emit_program — macOS Call prefix
# ---------------------------------------------------------------------------


def test_emit_program_macos_call_prefix(tmp_path: Path) -> None:
    """On macOS, ``Call`` instruction targets are prefixed with ``_``."""
    fn = Function(name="main", globl=True, instructions=(Call(name="foo"), Ret()))
    program = Program(definitions=(fn,))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="darwin")
    content = out.read_text()
    assert "  call _foo" in content


def test_emit_program_linux_call_plt_suffix(tmp_path: Path) -> None:
    """On Linux, ``Call`` instruction targets are suffixed with ``@PLT``."""
    fn = Function(name="main", globl=True, instructions=(Call(name="foo"), Ret()))
    program = Program(definitions=(fn,))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="linux")
    content = out.read_text()
    assert "  call foo@PLT" in content


def test_emit_program_linux_call_no_underscore_prefix(tmp_path: Path) -> None:
    """On Linux, ``Call`` instruction targets are NOT prefixed with ``_``."""
    fn = Function(name="main", globl=True, instructions=(Call(name="foo"), Ret()))
    program = Program(definitions=(fn,))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="linux")
    content = out.read_text()
    assert "  call _foo" not in content


def test_emit_program_macos_call_no_plt(tmp_path: Path) -> None:
    """On macOS, ``Call`` instruction targets do NOT receive ``@PLT``."""
    fn = Function(name="main", globl=True, instructions=(Call(name="foo"), Ret()))
    program = Program(definitions=(fn,))
    out = tmp_path / "main.s"
    emit_program(program, out, platform="darwin")
    assert "@PLT" not in out.read_text()
