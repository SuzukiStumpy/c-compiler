"""Assembly emission stage: writes an assembly AST to a ``.s`` file.

Converts an ``asm.Program`` into a text assembly file, applying two
platform-specific rules:

* **macOS** — all function names are prefixed with an underscore (``_``)
  because the macOS ABI requires a leading underscore on C symbols.
* **Linux / WSL** — a ``.section .note.GNU-stack,"",@progbits`` directive
  is appended to mark the stack as non-executable, as required by modern
  Linux toolchains.

Platform detection uses ``sys.platform`` by default; callers may override
it via the ``platform`` parameter to facilitate testing without patching.
"""

import dataclasses
import sys
from pathlib import Path

import compiler.asm as asm

_GNU_STACK = '  .section .note.GNU-stack,"",@progbits'


def emit_program(program: asm.Program, output_path: Path, platform: str | None = None) -> None:
    """Emit an assembly AST as a ``.s`` file at ``output_path``.

    Applies platform-specific transformations before writing:

    * On macOS (``"darwin"``), the function name is prefixed with ``_``.
    * On Linux (``"linux"``), a GNU-stack section directive is appended.

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

    On macOS, every function name receives a leading underscore.  On other
    platforms the program is returned unchanged.

    Args:
        program: The assembly AST to adjust.
        platform: The target platform string (e.g. ``sys.platform``).

    Returns:
        A new ``asm.Program`` with adjusted names, or the original if no
        adjustment is needed.
    """
    if not platform.startswith("darwin"):
        return program
    prefixed = dataclasses.replace(program.function, name=f"_{program.function.name}")
    return dataclasses.replace(program, function=prefixed)
