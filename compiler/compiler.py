"""Main entry point and CLI for the compiler pipeline."""

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import click

from compiler.ast import Program, pretty_print
from compiler.code_generator import generate
from compiler.lexer import lexer
from compiler.parser import parse
from compiler.subprocesses import assemble_and_link, preprocess
from compiler.types.stages import Stage
from compiler.utils.diagnostics import format_for_terminal
from compiler.utils.exceptions import LexerError, ParseError


class MutuallyExclusiveOption(click.Option):
    """A Click option that rejects simultaneous use of any named peers.

    Specify peer parameter names (without leading dashes, using Click's
    underscore-normalised form) via ``mutually_exclusive`` at decoration time.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialise the option, extracting the ``mutually_exclusive`` peer list.

        Args:
            *args: Positional arguments forwarded to ``click.Option``.
            **kwargs: Keyword arguments forwarded to ``click.Option``.
                ``mutually_exclusive`` (list[str]) names the parameter names
                that conflict with this option.
        """
        self.mutually_exclusive: list[str] = kwargs.pop("mutually_exclusive", [])
        super().__init__(*args, **kwargs)

    def handle_parse_result(
        self,
        ctx: click.Context,
        opts: Mapping[str, Any],
        args: list[str],
    ) -> tuple[Any, list[str]]:
        """Raise UsageError if this option and any peer are both set.

        Args:
            ctx: The active Click context.
            opts: Parsed option values accumulated so far.
            args: Remaining unparsed command-line arguments.

        Returns:
            The parsed value and remaining args from the parent handler.

        Raises:
            click.UsageError: If this option is truthy and any peer in
                ``mutually_exclusive`` is also truthy.
        """
        name = self.name
        if name is not None and opts.get(name):
            for peer in self.mutually_exclusive:
                if opts.get(peer):
                    raise click.UsageError(f"--{name} and --{peer} are mutually exclusive.")
        return super().handle_parse_result(ctx, opts, args)


def compile(source: Path, stages: Stage = Stage.ALL) -> tuple[Program | None, Path]:
    """Run the selected compilation stages on a preprocessed source file.

    Executes each requested stage in pipeline order: lexing, parsing, and
    code generation. The ``stages`` bitmask controls which stages run; omitting
    a flag causes the pipeline to stop before that stage.

    Args:
        source: Path to the preprocessed ``.i`` file to compile.
        stages: Bitmask of stages to execute. Defaults to ``Stage.ALL``.

    Returns:
        A tuple of the parsed ``Program`` (or ``None`` if parsing did not run)
        and the path to the assembly ``.s`` file.

    Raises:
        LexerError: If the lexer stage is active and encounters an invalid
            character in the source.
        ParseError: If the parse stage is active and the token stream does not
            conform to the grammar.
    """
    token_stream = []
    program: Program | None = None

    if Stage.LEX in stages:
        token_stream = lexer(source)

    if Stage.PARSE in stages:
        program = parse(token_stream)

    if Stage.CODEGEN in stages:
        generate()

    # Path.unlink(source)
    return program, source.parent / f"{source.stem}.s"


@click.command(
    help=(
        "Compile SOURCE from C to a native executable.\n\n"
        "Pass a stage flag to stop the pipeline early and inspect intermediate output. "
        "Without a flag the full pipeline runs and an executable is produced."
    )
)
@click.option(
    "--lex",
    is_flag=True,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["parse", "codegen"],
    help="Stop after lexing.",
)
@click.option(
    "--parse",
    is_flag=True,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["lex", "codegen"],
    help="Stop after parsing.",
)
@click.option(
    "--codegen",
    is_flag=True,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["lex", "parse"],
    help="Stop after code generation; emit assembly but do not link.",
)
@click.argument("source", type=click.Path(), required=True, nargs=1)
def compiler(
    source: str,
    lex: bool,
    parse: bool,
    codegen: bool,
) -> None:
    """Invoke the compiler pipeline for a single C source file.

    Args:
        source: Path to the C source file to compile.
        lex: Stop the pipeline after lexing.
        parse: Stop the pipeline after parsing.
        codegen: Stop after code generation; emit assembly but do not link.
    """
    if codegen:
        stages = Stage.LEX | Stage.PARSE | Stage.CODEGEN
    elif parse:
        stages = Stage.LEX | Stage.PARSE
    elif lex:
        stages = Stage.LEX
    else:
        stages = Stage.ALL

    try:
        pre_processed = preprocess(Path(source))
        program, assembly = compile(pre_processed, stages)

        if parse and program is not None:
            click.echo(pretty_print(program))
        elif not (lex or parse or codegen):
            executable = assemble_and_link(assembly)
            print(f"Compilation complete.  Output file is: {executable}")
    except LexerError as ex:
        click.echo(format_for_terminal(ex), err=True)
        sys.exit(1)
    except ParseError as ex:
        click.echo(str(ex), err=True)
        sys.exit(1)
    except Exception as ex:
        print(f"Compiler failed with exception {ex}")
        sys.exit(1)
