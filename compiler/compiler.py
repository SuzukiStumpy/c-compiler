"""Main entry point and CLI for the compiler pipeline."""

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import click

import compiler.syntax.asm as asm
import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from compiler.code_generator import generate
from compiler.emitter import emit_program
from compiler.lexer import lexer
from compiler.parser import parse
from compiler.semantic_analysis import resolve
from compiler.subprocesses import assemble_and_link, preprocess
from compiler.syntax.ast import pretty_print
from compiler.tac_generator import generate as tac_generate
from compiler.types.stages import Stage
from compiler.utils.diagnostics import format_for_terminal
from compiler.utils.exceptions import CodeGenerationError, LexerError, ParseError


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


def compile(source: Path, stages: Stage = Stage.ALL, exe_out: bool = True) -> None:
    """Run the selected compilation stages on a preprocessed source file.

    Executes each requested stage in pipeline order: lexing, parsing, TAC
    generation, assembly code generation, and assembly/linking.  The ``stages``
    bitmask controls which stages run; omitting a flag causes the pipeline to
    stop before that stage.

    When the pipeline stops after ``PARSE``, the main AST is printed to stdout.
    When it stops after ``TACKY``, the TAC AST is printed.  When it stops after
    ``CODEGEN``, the assembly AST is printed.  When ``LINK`` is included, the
    executable is produced and a completion message is printed.

    Args:
        source: Path to the preprocessed ``.i`` file to compile.
        stages: Bitmask of stages to execute.  Defaults to ``Stage.ALL``.
        exe_out: When ``Stage.LINK`` is included, ``True`` produces a linked executable
            and ``False`` produces an object file only.

    Raises:
        LexerError: If the lexer stage encounters an invalid character.
        ParseError: If the token stream does not conform to the grammar.
        SemanticError: If the AST fails semantic analysis.
        CodeGenerationError: If the AST or TAC contains a construct that cannot
            be translated.
    """
    token_stream = []
    program: ast.Program | None = None
    tac_program: tac.Program | None = None
    asm_program: asm.Program | None = None

    if Stage.LEX in stages:
        token_stream = lexer(source)

    if Stage.PARSE in stages:
        program = parse(token_stream)
        if Stage.VALIDATE not in stages:
            click.echo(pretty_print(program))

    if Stage.VALIDATE in stages:
        program = resolve(program)  # type: ignore[bad-argument-type]
        if Stage.TACKY not in stages:
            click.echo(pretty_print(program))

    if Stage.TACKY in stages:
        assert program is not None
        tac_program = tac_generate(program)
        if Stage.CODEGEN not in stages:
            click.echo(pretty_print(tac_program))

    if Stage.CODEGEN in stages:
        assert tac_program is not None
        asm_program = generate(tac_program)
        if Stage.LINK not in stages:
            click.echo(pretty_print(asm_program))

    if Stage.LINK in stages:
        assert asm_program is not None
        assembly = source.parent / f"{source.stem}.s"
        emit_program(asm_program, assembly)
        executable = assemble_and_link(assembly, exe_out=exe_out)
        click.echo(f"Compilation complete.  Output file is: {executable}")


# ==============================================================================
# == Main Program Entry Point
# ==============================================================================
@click.command(
    help=(
        "Compile SOURCE from C to a native executable.\n\n"
        "Pass a stage flag to stop the pipeline early and inspect intermediate output. "
        "Without a flag the full pipeline runs and an executable is produced."
    )
)
@click.option(
    "-c",
    is_flag=True,
    help="Don't link, just produce an object file.",
)
@click.option(
    "--lex",
    is_flag=True,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["parse", "validate", "tacky", "codegen"],
    help="Stop after lexing.",
)
@click.option(
    "--parse",
    is_flag=True,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["lex", "validate", "tacky", "codegen"],
    help="Stop after parsing; print the AST.",
)
@click.option(
    "--validate",
    is_flag=True,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["lex", "parse", "tacky", "codegen"],
    help="Stop after performing semantic analysis; print the AST.",
)
@click.option(
    "--tacky",
    is_flag=True,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["lex", "parse", "validate", "codegen"],
    help="Stop after TAC generation; print the TAC AST.",
)
@click.option(
    "--codegen",
    is_flag=True,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["lex", "parse", "validate", "tacky"],
    help="Stop after code generation; emit assembly but do not link.",
)
@click.argument("source", type=click.Path(), required=True, nargs=1)
def compiler(
    source: str,
    c: bool,
    lex: bool,
    parse: bool,
    validate: bool,
    tacky: bool,
    codegen: bool,
) -> None:
    """Invoke the compiler pipeline for a single C source file.

    Args:
        source: Path to the C source file to compile.
        c: Don't link an executable file, just produce object file.
        lex: Stop the pipeline after lexing.
        parse: Stop the pipeline after parsing; print the AST.
        validate: Stop after semantic analysis; print the resolved AST.
        tacky: Stop the pipeline after TAC generation; print the TAC AST.
        codegen: Stop after code generation; emit assembly but do not link.
    """
    if codegen:
        stages = Stage.LEX | Stage.PARSE | Stage.VALIDATE | Stage.TACKY | Stage.CODEGEN
    elif tacky:
        stages = Stage.LEX | Stage.PARSE | Stage.VALIDATE | Stage.TACKY
    elif validate:
        stages = Stage.LEX | Stage.PARSE | Stage.VALIDATE
    elif parse:
        stages = Stage.LEX | Stage.PARSE
    elif lex:
        stages = Stage.LEX
    else:
        stages = Stage.ALL

    try:
        pre_processed = preprocess(Path(source))
        compile(pre_processed, stages, exe_out=not c)
    except (LexerError, ParseError, CodeGenerationError) as ex:
        click.echo(format_for_terminal(ex), err=True)
        sys.exit(1)
    except Exception as ex:
        click.echo(f"Compiler failed with exception {ex}", err=True)
        sys.exit(1)
