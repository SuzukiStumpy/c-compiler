# c-compiler

A homegown C compiler written in Python (only works for a subset of the C language)

## Dev installation

The codebase requires the use of ``uv`` and ``ruff``.  ``pyrefly`` is assumed to be in place
as the linter.  These tools should all be installed under a Linux or WSL environment.

The project also requires the presence of ``gcc`` or ``clang`` (with the ``gcc`` command shim)
C compilers to handle the preprocessing and final assembly/linking stages of any programs.

To install, clone the repository into your filesystem, then launch ``VS Code``.  From a ``bash``
terminal in the IDE, issue the command:

```bash
$ uv sync
```

This should set up the virtual environment and ensure the correct dependencies are installed.

## Tool installation

From within the IDE, the compiler can be invoked directly from the command line.  If you want to
later install it for use outside the development environment then you'll need to issue the command

```bash
$ uv tool install compiler
```

## Usage

To call the compiler simply call ``compiler`` from the command line.  The command requires you to pass in
a ``C`` source file on the command line.  It also accepts three optional switches which will halt compilation
at the appropriate stage.  Note that these switches (``lex``, ``parse`` and ``codegen``) are mutually exclusive.

```
Usage: compiler [OPTIONS] SOURCE

  Compile SOURCE from C to a native executable.

  Pass a stage flag to stop the pipeline early and inspect intermediate
  output. Without a flag the full pipeline runs and an executable is produced.

Options:
  --lex      Stop after lexing.
  --parse    Stop after parsing.
  --codegen  Stop after code generation; emit assembly but do not link.
  --help     Show this message and exit.
```