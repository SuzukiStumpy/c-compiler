from dataclasses import dataclass


@dataclass
class Optimizations:
    """Flags controlling which optional compiler optimizations are enabled.

    Attributes:
        fold_constants: Enable TAC-level constant folding.
        propagate_copies: Enable TAC-level copy propagation.
        eliminate_unreachable_code: Enable TAC-level unreachable code elimination.
        eliminate_dead_stores: Enable TAC-level dead store elimination.
    """

    fold_constants: bool
    propagate_copies: bool
    eliminate_unreachable_code: bool
    eliminate_dead_stores: bool
