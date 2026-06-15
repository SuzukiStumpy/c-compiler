"""Generic disjoint-set (union-find) data structure."""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class DisjointSet(Generic[T]):
    """A disjoint-set (union-find) structure backed by a flat dict.

    Each call to ``union`` records that one element has been merged into another.
    ``find`` follows the chain of merges to the canonical representative of any
    element.  Truthiness reflects whether any unions have occurred.

    This is intentionally a simple recursive implementation without path
    compression; the sets in the register-coalescing pass are small enough that
    the overhead is irrelevant.

    Example::

        ds: DisjointSet[str] = DisjointSet()
        ds.union("a", "b")   # a -> b
        ds.union("b", "c")   # b -> c
        ds.find("a")         # returns "c"
        bool(ds)             # True - at least one union recorded
    """

    def __init__(self) -> None:
        """Initialise an empty disjoint-set."""
        self._map: dict[T, T] = {}

    def union(self, x: T, y: T) -> None:
        """Record that element *x* has been merged into element *y*.

        After this call, ``find(x)`` will return the same representative as
        ``find(y)``.

        Args:
            x: The element being eliminated (merged away).
            y: The element *x* is merged into (the merge target).
        """
        self._map[x] = y

    def find(self, r: T) -> T:
        """Return the canonical representative of the partition containing *r*.

        Follows the chain of merges recorded by ``union`` until an element is
        reached that has not itself been merged into anything else.

        Args:
            r: The element whose representative is sought.

        Returns:
            The canonical representative of *r*'s partition.  Returns *r*
            itself if *r* has never appeared as the first argument to ``union``.
        """
        if r in self._map:
            return self.find(self._map[r])
        return r

    def __bool__(self) -> bool:
        """Return ``True`` if at least one union has been recorded.

        Returns:
            ``True`` when the underlying map is non-empty, ``False`` otherwise.
        """
        return bool(self._map)

    def __repr__(self) -> str:
        """Return a developer-readable representation.

        Returns:
            A string showing the internal merge map.
        """
        return f"DisjointSet({self._map!r})"
