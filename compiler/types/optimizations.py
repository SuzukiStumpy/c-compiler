from dataclasses import dataclass


@dataclass
class Optimizations:
    fold_constants: bool
    propagate_copies: bool
    eliminate_unreachable_code: bool
    eliminate_dead_stores: bool
