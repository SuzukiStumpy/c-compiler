"""Tests for semantic analysis of C11 designated initialisers.

Verifies that ``MemberDesignatorInit`` and ``IndexDesignatorInit`` nodes produced
by the parser are expanded into positional ``CompoundInit`` nodes during type-checking,
and that semantic errors are raised for invalid designators.
"""

import pytest

import compiler.syntax.ast as ast
from compiler.semantic_analysis import SemanticAnalyser, resolve
from compiler.syntax.ast.constants import ConstInt
from compiler.utils.exceptions import SemanticError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _struct_decl(tag: str, *members: ast.MemberDeclaration) -> ast.StructDecl:
    """Return a ``StructDecl`` for testing.

    Args:
        tag: Source-level struct tag name.
        *members: ``MemberDeclaration`` nodes.

    Returns:
        An ``ast.StructDecl`` node.
    """
    return ast.StructDecl(ast.StructDeclaration(tag=tag, members=members))


def _union_decl(tag: str, *members: ast.MemberDeclaration) -> ast.UnionDecl:
    """Return a ``UnionDecl`` for testing.

    Args:
        tag: Source-level union tag name.
        *members: ``MemberDeclaration`` nodes.

    Returns:
        An ``ast.UnionDecl`` node.
    """
    return ast.UnionDecl(ast.UnionDeclaration(tag=tag, members=members))


def _member(name: str, t: ast.Type) -> ast.MemberDeclaration:
    """Return a ``MemberDeclaration`` with the given name and type.

    Args:
        name: Member name.
        t: Member C type.

    Returns:
        An ``ast.MemberDeclaration`` node.
    """
    return ast.MemberDeclaration(member_name=name, member_type=t)


def _int_const(v: int) -> ast.Constant:
    """Return an ``int`` constant expression.

    Args:
        v: Integer value.

    Returns:
        An ``ast.Constant(ConstInt(v))`` node.
    """
    return ast.Constant(ConstInt(v))


def _resolved_local_init(
    type_decls: tuple[ast.Declaration, ...],
    var_name: str,
    var_type: ast.Type,
    var_init: ast.Initializer,
) -> ast.Initializer:
    """Resolve a block-scope variable declaration and return its type-checked initialiser.

    Places the variable inside a ``main()`` function body so the block-scope
    resolution path runs, which populates ``VariableDeclaration.init`` with the
    expanded, type-checked initialiser.

    Args:
        type_decls: Top-level type declarations (struct/union definitions).
        var_name: Variable name.
        var_type: Declared type.
        var_init: Raw (pre-resolution) initialiser containing designator nodes.

    Returns:
        The resolved, type-checked ``Initializer``.

    Raises:
        AssertionError: If the variable cannot be located in the resolved body.
    """
    prog = ast.Program(
        declarations=(
            *type_decls,
            ast.FunDecl(
                ast.Function(
                    name="main",
                    params=(),
                    body=(ast.D(ast.VarDecl(ast.VariableDeclaration(var_name, var_init, var_type))),),
                    fun_type=ast.FunType(params=(), ret=ast.IntType()),
                )
            ),
        )
    )
    resolved = resolve(prog)
    fn_decl = next(d for d in resolved.declarations if isinstance(d, ast.FunDecl))
    assert fn_decl.function_declaration.body is not None
    for item in fn_decl.function_declaration.body:
        if isinstance(item, ast.D) and isinstance(item.declaration, ast.VarDecl):
            vd = item.declaration.variable_declaration
            result_init = vd.init
            assert result_init is not None
            return result_init
    raise AssertionError(f"No variable '{var_name}' found in resolved function body")


def _get_const_val(init: ast.Initializer) -> int:
    """Extract the integer constant value from a ``SingleInit`` wrapping a ``Constant``.

    Follows optional ``Cast`` wrappers produced by assignment conversion.

    Args:
        init: A ``SingleInit`` whose expression is an (optionally cast) integer constant.

    Returns:
        The integer value of the constant.

    Raises:
        AssertionError: If the structure does not match expectations.
    """
    assert isinstance(init, ast.SingleInit)
    exp = init.exp
    while isinstance(exp, ast.Cast):
        exp = exp.exp
    assert isinstance(exp, ast.Constant)
    return int(exp.value.val)


# ---------------------------------------------------------------------------
# Struct designated initialisers
# ---------------------------------------------------------------------------


class TestStructDesignatedInit:
    """``MemberDesignatorInit`` nodes in a struct initialiser are expanded to positional form."""

    def test_reversed_designators_produce_correct_member_order(self) -> None:
        """``{.y=2, .x=1}`` for ``struct{int x; int y;}`` resolves to positional [x=1, y=2]."""
        init = _resolved_local_init(
            (_struct_decl("S", _member("x", ast.IntType()), _member("y", ast.IntType())),),
            "s",
            ast.Structure("S"),
            ast.CompoundInit(
                [
                    ast.MemberDesignatorInit("y", ast.SingleInit(_int_const(2))),
                    ast.MemberDesignatorInit("x", ast.SingleInit(_int_const(1))),
                ]
            ),
        )
        assert isinstance(init, ast.CompoundInit)
        assert len(init.initializers) == 2
        assert _get_const_val(init.initializers[0]) == 1  # x
        assert _get_const_val(init.initializers[1]) == 2  # y

    def test_partial_designator_zero_fills_omitted_member(self) -> None:
        """``{.x=5}`` for ``struct{int x; int y;}`` produces [x=5, y=0]."""
        init = _resolved_local_init(
            (_struct_decl("S", _member("x", ast.IntType()), _member("y", ast.IntType())),),
            "s",
            ast.Structure("S"),
            ast.CompoundInit(
                [
                    ast.MemberDesignatorInit("x", ast.SingleInit(_int_const(5))),
                ]
            ),
        )
        assert isinstance(init, ast.CompoundInit)
        assert len(init.initializers) == 2
        assert _get_const_val(init.initializers[0]) == 5  # x
        assert _get_const_val(init.initializers[1]) == 0  # y (zero-filled)

    def test_mix_positional_then_designator(self) -> None:
        """``{1, .y=2}`` for ``struct{int x; int y;}`` resolves to [x=1, y=2]."""
        init = _resolved_local_init(
            (_struct_decl("S", _member("x", ast.IntType()), _member("y", ast.IntType())),),
            "s",
            ast.Structure("S"),
            ast.CompoundInit(
                [
                    ast.SingleInit(_int_const(1)),
                    ast.MemberDesignatorInit("y", ast.SingleInit(_int_const(2))),
                ]
            ),
        )
        assert isinstance(init, ast.CompoundInit)
        assert len(init.initializers) == 2
        assert _get_const_val(init.initializers[0]) == 1  # x
        assert _get_const_val(init.initializers[1]) == 2  # y

    def test_designator_then_positional_fills_next_member(self) -> None:
        """``{.x=1, 2}`` for ``struct{int x; int y;}`` assigns y=2 (next after x)."""
        init = _resolved_local_init(
            (_struct_decl("S", _member("x", ast.IntType()), _member("y", ast.IntType())),),
            "s",
            ast.Structure("S"),
            ast.CompoundInit(
                [
                    ast.MemberDesignatorInit("x", ast.SingleInit(_int_const(1))),
                    ast.SingleInit(_int_const(2)),
                ]
            ),
        )
        assert isinstance(init, ast.CompoundInit)
        assert len(init.initializers) == 2
        assert _get_const_val(init.initializers[0]) == 1  # x
        assert _get_const_val(init.initializers[1]) == 2  # y

    def test_no_designated_nodes_in_resolved_init(self) -> None:
        """After resolution, no ``MemberDesignatorInit`` nodes remain in the output."""
        init = _resolved_local_init(
            (_struct_decl("S", _member("x", ast.IntType()), _member("y", ast.IntType())),),
            "s",
            ast.Structure("S"),
            ast.CompoundInit(
                [
                    ast.MemberDesignatorInit("y", ast.SingleInit(_int_const(2))),
                ]
            ),
        )
        assert isinstance(init, ast.CompoundInit)
        for item in init.initializers:
            assert not isinstance(item, ast.MemberDesignatorInit)

    def test_unknown_member_raises(self) -> None:
        """A designator naming a non-existent member raises ``SemanticError``."""
        prog = ast.Program(
            declarations=(
                _struct_decl("S", _member("x", ast.IntType())),
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=(
                            ast.D(
                                ast.VarDecl(
                                    ast.VariableDeclaration(
                                        "s",
                                        ast.CompoundInit(
                                            [
                                                ast.MemberDesignatorInit("z", ast.SingleInit(_int_const(1))),
                                            ]
                                        ),
                                        ast.Structure("S"),
                                    )
                                )
                            ),
                        ),
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )
        with pytest.raises(SemanticError, match="[Nn]o member"):
            resolve(prog)

    def test_index_designator_in_struct_raises(self) -> None:
        """An array-index designator ``[n] = ...`` inside a struct initialiser raises ``SemanticError``."""
        prog = ast.Program(
            declarations=(
                _struct_decl("S", _member("x", ast.IntType())),
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=(
                            ast.D(
                                ast.VarDecl(
                                    ast.VariableDeclaration(
                                        "s",
                                        ast.CompoundInit(
                                            [
                                                ast.IndexDesignatorInit(
                                                    ast.Constant(ConstInt(0)), ast.SingleInit(_int_const(1))
                                                ),
                                            ]
                                        ),
                                        ast.Structure("S"),
                                    )
                                )
                            ),
                        ),
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )
        with pytest.raises(SemanticError):
            resolve(prog)


# ---------------------------------------------------------------------------
# Union designated initialisers
# ---------------------------------------------------------------------------


class TestUnionDesignatedInit:
    """``MemberDesignatorInit`` in a union initialiser selects the correct member."""

    def test_first_member_designator_is_equivalent_to_non_designated(self) -> None:
        """``{.i=1}`` for ``union{int i; long l;}`` resolves to ``CompoundInit([SingleInit(1)])``."""
        init = _resolved_local_init(
            (_union_decl("U", _member("i", ast.IntType()), _member("l", ast.LongType())),),
            "u",
            ast.Union("U"),
            ast.CompoundInit(
                [
                    ast.MemberDesignatorInit("i", ast.SingleInit(_int_const(1))),
                ]
            ),
        )
        assert isinstance(init, ast.CompoundInit)
        assert len(init.initializers) == 1
        assert _get_const_val(init.initializers[0]) == 1

    def test_second_member_designator_selects_correct_member(self) -> None:
        """``{.l=42}`` for ``union{int i; long l;}`` selects the ``long`` member."""
        init = _resolved_local_init(
            (_union_decl("U", _member("i", ast.IntType()), _member("l", ast.LongType())),),
            "u",
            ast.Union("U"),
            ast.CompoundInit(
                [
                    ast.MemberDesignatorInit("l", ast.SingleInit(_int_const(42))),
                ]
            ),
        )
        assert isinstance(init, ast.CompoundInit)
        assert len(init.initializers) == 1
        assert init.union_member_index == 1
        assert _get_const_val(init.initializers[0]) == 42

    def test_unknown_union_member_raises(self) -> None:
        """A designator naming a non-existent union member raises ``SemanticError``."""
        prog = ast.Program(
            declarations=(
                _union_decl("U", _member("i", ast.IntType())),
                ast.FunDecl(
                    ast.Function(
                        name="main",
                        params=(),
                        body=(
                            ast.D(
                                ast.VarDecl(
                                    ast.VariableDeclaration(
                                        "u",
                                        ast.CompoundInit(
                                            [
                                                ast.MemberDesignatorInit("z", ast.SingleInit(_int_const(1))),
                                            ]
                                        ),
                                        ast.Union("U"),
                                    )
                                )
                            ),
                        ),
                        fun_type=ast.FunType(params=(), ret=ast.IntType()),
                    )
                ),
            )
        )
        with pytest.raises(SemanticError, match="[Nn]o member"):
            resolve(prog)


# ---------------------------------------------------------------------------
# Array designated initialisers
# ---------------------------------------------------------------------------


class TestArrayDesignatedInit:
    """``IndexDesignatorInit`` nodes in an array initialiser are expanded to positional form."""

    def test_single_index_designator_expands_with_zeros(self) -> None:
        """``int[5] = {[2]=7}`` expands to [0, 0, 7, 0, 0]."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration(
                "arr",
                ast.CompoundInit(
                    [
                        ast.IndexDesignatorInit(ast.Constant(ConstInt(2)), ast.SingleInit(_int_const(7))),
                    ]
                ),
                ast.Array(ast.IntType(), 5),
            )
        )
        init = result.init
        assert isinstance(init, ast.CompoundInit)
        assert len(init.initializers) == 5
        assert _get_const_val(init.initializers[0]) == 0
        assert _get_const_val(init.initializers[1]) == 0
        assert _get_const_val(init.initializers[2]) == 7
        assert _get_const_val(init.initializers[3]) == 0
        assert _get_const_val(init.initializers[4]) == 0

    def test_two_index_designators_out_of_order(self) -> None:
        """``int[5] = {[3]=9, [1]=4}`` places values at correct positions."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration(
                "arr",
                ast.CompoundInit(
                    [
                        ast.IndexDesignatorInit(ast.Constant(ConstInt(3)), ast.SingleInit(_int_const(9))),
                        ast.IndexDesignatorInit(ast.Constant(ConstInt(1)), ast.SingleInit(_int_const(4))),
                    ]
                ),
                ast.Array(ast.IntType(), 5),
            )
        )
        init = result.init
        assert isinstance(init, ast.CompoundInit)
        assert len(init.initializers) == 5
        assert _get_const_val(init.initializers[0]) == 0
        assert _get_const_val(init.initializers[1]) == 4
        assert _get_const_val(init.initializers[2]) == 0
        assert _get_const_val(init.initializers[3]) == 9
        assert _get_const_val(init.initializers[4]) == 0

    def test_index_designator_followed_by_positional_fills_next(self) -> None:
        """``{[1]=2, 3}`` assigns index 1 the value 2 and index 2 the value 3."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration(
                "arr",
                ast.CompoundInit(
                    [
                        ast.IndexDesignatorInit(ast.Constant(ConstInt(1)), ast.SingleInit(_int_const(2))),
                        ast.SingleInit(_int_const(3)),
                    ]
                ),
                ast.Array(ast.IntType(), 5),
            )
        )
        init = result.init
        assert isinstance(init, ast.CompoundInit)
        assert len(init.initializers) == 5
        assert _get_const_val(init.initializers[0]) == 0
        assert _get_const_val(init.initializers[1]) == 2
        assert _get_const_val(init.initializers[2]) == 3
        assert _get_const_val(init.initializers[3]) == 0
        assert _get_const_val(init.initializers[4]) == 0

    def test_out_of_bounds_index_raises(self) -> None:
        """``int[5] = {[5]=1}`` raises ``SemanticError`` (index 5 out of bounds)."""
        analyser = SemanticAnalyser()
        with pytest.raises(SemanticError, match="[Oo]ut of bounds|[Ii]ndex"):
            analyser.resolve_variable_declaration(
                ast.VariableDeclaration(
                    "arr",
                    ast.CompoundInit(
                        [
                            ast.IndexDesignatorInit(ast.Constant(ConstInt(5)), ast.SingleInit(_int_const(1))),
                        ]
                    ),
                    ast.Array(ast.IntType(), 5),
                )
            )

    def test_member_designator_in_array_raises(self) -> None:
        """A struct-member designator ``.x = ...`` inside an array initialiser raises ``SemanticError``."""
        analyser = SemanticAnalyser()
        with pytest.raises(SemanticError):
            analyser.resolve_variable_declaration(
                ast.VariableDeclaration(
                    "arr",
                    ast.CompoundInit(
                        [
                            ast.MemberDesignatorInit("x", ast.SingleInit(_int_const(1))),
                        ]
                    ),
                    ast.Array(ast.IntType(), 5),
                )
            )

    def test_no_designated_nodes_in_resolved_array_init(self) -> None:
        """After resolution, no ``IndexDesignatorInit`` nodes remain in the array init."""
        analyser = SemanticAnalyser()
        result = analyser.resolve_variable_declaration(
            ast.VariableDeclaration(
                "arr",
                ast.CompoundInit(
                    [
                        ast.IndexDesignatorInit(ast.Constant(ConstInt(0)), ast.SingleInit(_int_const(1))),
                    ]
                ),
                ast.Array(ast.IntType(), 3),
            )
        )
        init = result.init
        assert isinstance(init, ast.CompoundInit)
        for item in init.initializers:
            assert not isinstance(item, ast.IndexDesignatorInit)
