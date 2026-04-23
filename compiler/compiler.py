"""Main entry point for the application.  For now, just an empty file"""

from pathlib import Path

import click

from compiler.code_generator import generate
from compiler.lexer import lexer
from compiler.parser import parse
from compiler.subprocesses import assemble_and_link, preprocess
from compiler.utils.types import Stage


def compile(source: Path, stages: Stage = Stage.ALL) -> Path:
    """Currently stubbed out.  Will eventually become the main compiler"""
    if Stage.LEX in stages:
        lexer(source)

    if Stage.PARSE in stages:
        parse()

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
) -> int:

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
        executable = assemble_and_link(assembly)

        print(f"Compilation complete.  Output file is: {executable}")
        return 0
    except Exception as ex:
        print(f"Compiler failed with exception {ex}")
        return 1
