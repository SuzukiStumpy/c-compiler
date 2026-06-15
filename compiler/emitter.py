"""Assembly emission stage: writes an assembly AST to a ``.s`` file.

Converts an ``asm.Program`` into a text assembly file, applying three
platform-specific rules:

* **macOS** - all function definition names and ``call`` targets are prefixed
  with an underscore (``_``) because the macOS ABI requires a leading
  underscore on C symbols.
* **Linux** - every ``call`` target is suffixed with ``@PLT`` so the linker
  routes the call through the Procedure Linkage Table, enabling correct
  dynamic symbol resolution for external symbols.
* **Linux / WSL** - a ``.section .note.GNU-stack,"",@progbits`` directive
  is appended to mark the stack as non-executable, as required by modern
  Linux toolchains.

Platform detection uses ``sys.platform`` by default; callers may override
it via the ``platform`` parameter to facilitate testing without patching.
"""

import dataclasses
import sys
from pathlib import Path

import compiler.syntax.asm as asm

_GNU_STACK = '      .section .note.GNU-stack,"",@progbits'


def emit_program(program: asm.Program, output_path: Path, platform: str | None = None) -> None:
    """Emit an assembly AST as a ``.s`` file at ``output_path``.

    Applies platform-specific transformations before writing:

    * On macOS (``"darwin"``), function definition names and ``call`` targets
      are prefixed with ``_``.
    * On Linux (``"linux"``), ``call`` targets are suffixed with ``@PLT`` and
      a GNU-stack section directive is appended.

    Args:
        program: The assembly AST to emit.
        output_path: Destination path for the ``.s`` file.
        platform: Override for ``sys.platform``; pass ``"darwin"`` or
            ``"linux"`` in tests to avoid patching the global.  If ``None``
            the current platform is detected automatically.

    Raises:
        OSError: If ``output_path`` cannot be written.
    """
    if platform is None:
        platform = sys.platform

    adjusted = _apply_platform_conventions(program, platform)
    content = adjusted.emit()

    if platform.startswith("linux"):
        content = content + "\n" + _GNU_STACK

    output_path.write_text(content + "\n")


def _apply_platform_conventions(program: asm.Program, platform: str) -> asm.Program:
    """Return a copy of ``program`` with platform-specific name adjustments.

    On macOS, every function definition name and every ``Call`` instruction
    target receive a leading underscore.  On Linux, every ``Call`` target is
    suffixed with ``@PLT``.  On other platforms the program is returned
    unchanged.

    Args:
        program: The assembly AST to adjust.
        platform: The target platform string (e.g. ``sys.platform``).

    Returns:
        A new ``asm.Program`` with adjusted names, or the original if no
        adjustment is needed.
    """
    if platform.startswith("darwin"):

        def _adjust_call(instr: asm.Instruction) -> asm.Instruction:
            return dataclasses.replace(instr, name=f"_{instr.name}") if isinstance(instr, asm.Call) else instr

        def _adjust_darwin(item: asm.TopLevel) -> asm.TopLevel:
            if isinstance(item, asm.Function):
                return dataclasses.replace(
                    item, name=f"_{item.name}", instructions=tuple(_adjust_call(i) for i in item.instructions)
                )
            if isinstance(item, asm.StaticVariable):
                return dataclasses.replace(item, identifier=f"_{item.identifier}")
            if isinstance(item, asm.StaticConstant):
                return dataclasses.replace(item, identifier=f"_{item.identifier}", section="__TEXT,__const")
            return item

        return dataclasses.replace(program, definitions=tuple(_adjust_darwin(d) for d in program.definitions))

    if platform.startswith("linux"):

        def _adjust_call_plt(instr: asm.Instruction) -> asm.Instruction:
            return dataclasses.replace(instr, name=f"{instr.name}@PLT") if isinstance(instr, asm.Call) else instr

        def _adjust_linux(item: asm.TopLevel) -> asm.TopLevel:
            if isinstance(item, asm.Function):
                return dataclasses.replace(item, instructions=tuple(_adjust_call_plt(i) for i in item.instructions))
            return item

        return dataclasses.replace(program, definitions=tuple(_adjust_linux(d) for d in program.definitions))

    return program
