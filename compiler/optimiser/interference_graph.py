"""Interference graph for register allocation.

An ``InterferenceGraph`` records which values (pseudo-registers and hardware
registers) cannot share a register because they are simultaneously live.  Each
node is a ``GraphNode`` - either a pseudo-register name (``str``) or a hardware
register enum member (``GPRegisters`` or ``FPRegisters``).

The graph is pre-populated with all allocatable hardware register nodes by
``build_graph``; pseudo-register nodes are added as they are discovered in the
instruction stream.
"""

from dataclasses import dataclass, field
from typing import TypeAlias

from compiler.syntax.asm.operands import FPRegisters, GPRegisters

GraphNode: TypeAlias = str | GPRegisters | FPRegisters
"""A node in the interference graph: a pseudo-register name or a hardware register."""


@dataclass
class InterferenceGraph:
    """Undirected interference graph used during register allocation.

    Each node represents a value that must occupy a register at some point in
    the function.  An edge between two nodes means the corresponding values are
    simultaneously live and therefore cannot be assigned the same register.

    Attributes:
        nodes: The set of all nodes in the graph.  Pre-populated with the
            allocatable hardware registers; pseudo nodes are added by
            ``_add_pseudoregisters``.
        edges: The set of undirected edges, each represented as a two-element
            ``frozenset`` so that ``{a, b}`` and ``{b, a}`` are the same edge.
        spill_costs: Maps each node to its spill cost.  Hardware registers carry
            ``math.inf`` (never spill); pseudo-register costs are set by
            ``add_spill_costs`` and reflect use/def frequency.
        pruned: Nodes temporarily removed from the graph during ``color_graph``
            simplification.  A node remaining in ``pruned`` after ``color_graph``
            completes has been spilled (no register could be assigned to it).
        colors: Maps each successfully coloured node to its assigned hardware
            register.  Populated by ``color_graph``.
    """

    nodes: set[GraphNode] = field(default_factory=set)
    edges: set[frozenset[GraphNode]] = field(default_factory=set)
    spill_costs: dict[GraphNode, float] = field(default_factory=dict)
    pruned: set[GraphNode] = field(default_factory=set)
    colors: dict[GraphNode, GPRegisters | FPRegisters] = field(default_factory=dict)

    def add_edge(self, a: GraphNode, b: GraphNode) -> None:
        """Add an undirected interference edge between nodes ``a`` and ``b``.

        Self-loops (``a == b``) are silently ignored because a value cannot
        interfere with itself.  Both ``a`` and ``b`` must already be present
        in ``self.nodes``; this is not enforced at runtime.

        Args:
            a: One endpoint of the new edge.
            b: The other endpoint of the new edge.
        """
        if a != b:
            self.edges.add(frozenset({a, b}))

    def neighbours(self, node: GraphNode) -> set[GraphNode]:
        """Return all nodes that share an interference edge with ``node``.

        Args:
            node: The node whose neighbours are requested.

        Returns:
            A set of all nodes adjacent to ``node`` in the graph.
        """
        return {next(iter(e - {node})) for e in self.edges if node in e}
