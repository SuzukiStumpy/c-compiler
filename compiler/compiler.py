"""Main entry point and CLI for the compiler pipeline."""

import sys
from pathlib import Path

import click

from compiler.code_generator import generate
from compiler.lexer import lexer
from compiler.parser import parse
from compiler.subprocesses import assemble_and_link, preprocess
from compiler.types.stages import Stage
from compiler.utils.diagnostics import format_for_terminal
from compiler.utils.exceptions import LexerError


def compile(source: Path, stages: Stage = Stage.ALL) -> Path:
    """Run the selected compilation stages on a preprocessed source file.

    Executes each requested stage in pipeline order: lexing, parsing, and
    code generation. The ``stages`` bitmask controls which stages run; omitting
    a flag causes the pipeline to stop before that stage.

    Args:
        source: Path to the preprocessed ``.i`` file to compile.
        stages: Bitmask of stages to execute. Defaults to ``Stage.ALL``.

    Returns:
        Path to the assembly ``.s`` file that the code generation stage would
        produce (the file may not yet exist while generation is stubbed).

    Raises:
        LexerError: If the lexer stage is active and encounters an invalid
            character in the source.
    """
    token_stream = []

    if Stage.LEX in stages:
        token_stream = lexer(source)

    if Stage.PARSE in stages:
        parse(token_stream)

    if Stage.CODEGEN in stages:
        generate()

    # Path.unlink(source)
    return source.parent / f"{source.stem}.s"


@click.command()
@click.option("--lex", is_flag=True)
@click.option("--parse", is_flag=True)
@click.option("--codegen", is_flag=True)
@click.argument("source", type=click.Path(), required=True, nargs=1)
def compiler(
    source: str,
    lex: bool,
    parse: bool,
    codegen: bool,
) -> None:
    """CLI entry point: compile a C source file to an executable.

    Runs the full compilation pipeline from C source file to native executable.
    If a stage flag is provided (``--lex``, ``--parse``, ``--codegen``), the
    pipeline stops after that stage. With no flags, all stages run.

    Exits with code 0 on success and code 1 on any compilation failure.
    Note: Click's standalone mode ignores return values — ``sys.exit()`` is
    used directly to ensure the process exits with the correct code.

    Args:
        source: Path to the C source file to compile.
        lex: Stop the pipeline after lexing.
        parse: Stop the pipeline after parsing.
        codegen: Stop the pipeline after code generation (equivalent to no flags).
    """
    if codegen:
        stages = Stage.ALL
    elif parse:
        stages = Stage.LEX | Stage.PARSE
    elif lex:
        stages = Stage.LEX
    else:
        stages = Stage.ALL

    try:
        pre_processed = preprocess(Path(source))
        assembly = compile(pre_processed, stages)

        if Stage.CODEGEN in stages:
            executable = assemble_and_link(assembly)
            print(f"Compilation complete.  Output file is: {executable}")
    except LexerError as ex:
        click.echo(format_for_terminal(ex), err=True)
        sys.exit(1)
    except Exception as ex:
        print(f"Compiler failed with exception {ex}")
        sys.exit(1)
