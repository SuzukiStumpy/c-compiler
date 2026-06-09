"""Top-level optimization loop for the TAC representation.

Applies a fixed-point loop over every function in a TAC program,
delegating each enabled pass to its own module.
"""

import compiler.syntax.tac as tac
from compiler.types.optimizations import Optimizations
from compiler.types.symbols import SymbolTable

from .cfg import cfg_to_instructions, make_control_flow_graph
from .constant_folding import constant_folding
from .copy_propagation import copy_propagation
from .dead_store_elimination import dead_store_elimination
from .unreachable_code_elimination import unreachable_code_elimination


def optimize_function(
    function: tac.Function,
    enabled_optimizations: Optimizations,
    symbol_table: SymbolTable | None = None,
) -> tac.Function:
    """Apply all enabled optimization passes to a single function in a fixed-point loop.

    Passes are applied in order on every iteration: constant folding, unreachable-code
    elimination, copy propagation, then dead-store elimination.  The loop repeats
    until the function body no longer changes or becomes empty.

    Args:
        function: The TAC function to optimize.
        enabled_optimizations: Flags controlling which passes are active.
        symbol_table: The program's symbol table, forwarded to ``constant_folding``
            so that type-conversion instructions can be resolved.

    Returns:
        The optimized ``tac.Function``.  May equal the input when no pass fires.
    """
    if not function.body:
        return function

    while True:
        if enabled_optimizations.fold_constants:
            post_constant_folding = constant_folding(function, symbol_table)
        else:
            post_constant_folding = function

        cfg = make_control_flow_graph(post_constant_folding)

        if enabled_optimizations.eliminate_unreachable_code:
            cfg = unreachable_code_elimination(cfg)

        if enabled_optimizations.propagate_copies:
            cfg = copy_propagation(cfg, symbol_table)

        if enabled_optimizations.eliminate_dead_stores:
            cfg = dead_store_elimination(cfg)

        optimized_function = cfg_to_instructions(cfg)

        if optimized_function == function or not optimized_function.body:
            return optimized_function

        function = optimized_function


def do_optimization(
    tac_program: tac.Program,
    enabled_optimisations: Optimizations,
    symbol_table: SymbolTable | None = None,
) -> tac.Program:
    """Apply optimization passes to every function in a TAC program.

    Non-function top-level items (``StaticVariable``, ``StaticConstant``) are
    passed through unchanged.

    Args:
        tac_program: The TAC program to optimize.
        enabled_optimisations: Flags controlling which passes are active.
        symbol_table: The program's symbol table.  Forwarded to each per-function
            pass so that type-conversion instructions can be resolved.

    Returns:
        A new ``tac.Program`` with all functions optimized.
    """
    optimized_top_level = tuple(
        optimize_function(item, enabled_optimisations, symbol_table) if isinstance(item, tac.Function) else item
        for item in tac_program.top_level
    )
    return tac.Program(top_level=optimized_top_level)
