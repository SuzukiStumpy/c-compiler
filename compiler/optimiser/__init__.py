"""Optimization pipeline package.

Re-exports the public entry points so callers can continue to write::

    from compiler.optimiser import do_optimization
"""

from compiler.optimiser.tac_main import do_optimization

__all__ = ["do_optimization"]
