"""Tests for static variable and symbol table TAC generation."""

import compiler.syntax.ast as ast
import compiler.syntax.tac as tac
from compiler.tac_generator import _convert_symbols_to_tacky, generate
from compiler.types.symbols import (
    FunAttr,
    Initial,
    IntInit,
    LongInit,
    NoInitializer,
    StaticAttr,
    SymbolTableEntry,
    Tentative,
    ZeroInit,
)
from test.tac_generator.helpers import _instrs_for_body

# ---------------------------------------------------------------------------
# _convert_symbols_to_tacky
# ---------------------------------------------------------------------------


class TestConvertSymbolsToTacky:
    """``_convert_symbols_to_tacky`` emits ``StaticVariable`` nodes from symbol-table entries."""

    def test_initial_entry_produces_static_variable_with_value(self) -> None:
        """An ``Initial((IntInit(7),))`` entry produces ``StaticVariable(name, globl, (IntInit(7),))``."""
        symbols = {"x": SymbolTableEntry(t=ast.IntType(), attrs=StaticAttr(init=Initial((IntInit(7),)), globl=True))}
        result = _convert_symbols_to_tacky(symbols)
        assert result == [tac.StaticVariable(name="x", globl=True, t=ast.IntType(), init=(tac.IntInit(7),))]

    def test_tentative_entry_produces_static_variable_with_zero(self) -> None:
        """A ``Tentative`` entry produces ``StaticVariable`` with ``init=(ZeroInit(4),)`` for an int."""
        symbols = {"x": SymbolTableEntry(t=ast.IntType(), attrs=StaticAttr(init=Tentative(), globl=False))}
        result = _convert_symbols_to_tacky(symbols)
        assert result == [tac.StaticVariable(name="x", globl=False, t=ast.IntType(), init=(tac.ZeroInit(4),))]

    def test_no_initializer_entry_is_skipped(self) -> None:
        """A ``NoInitializer`` entry (non-defining ``extern`` declaration) produces nothing."""
        symbols = {"x": SymbolTableEntry(t=ast.IntType(), attrs=StaticAttr(init=NoInitializer(), globl=True))}
        result = _convert_symbols_to_tacky(symbols)
        assert result == []

    def test_fun_attr_entry_is_skipped(self) -> None:
        """A ``FunAttr`` entry produces no ``StaticVariable``."""
        symbols = {
            "foo": SymbolTableEntry(
                t=ast.FunType(params=(), ret=ast.IntType()), attrs=FunAttr(defined=True, globl=True)
            )
        }
        result = _convert_symbols_to_tacky(symbols)
        assert result == []

    def test_globl_true_is_preserved(self) -> None:
        """``globl=True`` on the entry is reflected in the emitted ``StaticVariable``."""
        symbols = {"g": SymbolTableEntry(t=ast.IntType(), attrs=StaticAttr(init=Initial((IntInit(0),)), globl=True))}
        result = _convert_symbols_to_tacky(symbols)
        assert result[0].globl is True

    def test_globl_false_is_preserved(self) -> None:
        """``globl=False`` on the entry is reflected in the emitted ``StaticVariable``."""
        symbols = {"s": SymbolTableEntry(t=ast.IntType(), attrs=StaticAttr(init=Initial((IntInit(0),)), globl=False))}
        result = _convert_symbols_to_tacky(symbols)
        assert result[0].globl is False

    def test_name_is_preserved(self) -> None:
        """The symbol name is used verbatim as the ``StaticVariable.name``."""
        symbols = {".Vstatic.3": SymbolTableEntry(t=ast.IntType(), attrs=StaticAttr(init=Tentative(), globl=False))}
        result = _convert_symbols_to_tacky(symbols)
        assert result[0].name == ".Vstatic.3"

    def test_multiple_static_entries_all_emitted(self) -> None:
        """Multiple ``StaticAttr`` entries all produce corresponding ``StaticVariable`` nodes."""
        symbols = {
            "a": SymbolTableEntry(t=ast.IntType(), attrs=StaticAttr(init=Initial((IntInit(1),)), globl=True)),
            "b": SymbolTableEntry(t=ast.IntType(), attrs=StaticAttr(init=Tentative(), globl=False)),
        }
        result = _convert_symbols_to_tacky(symbols)
        assert len(result) == 2
        assert result[0].name == "a"
        assert result[1].name == "b"

    def test_mixed_entries_only_static_attr_emitted(self) -> None:
        """Only ``StaticAttr`` entries with ``Initial`` or ``Tentative`` produce output."""
        symbols = {
            "g": SymbolTableEntry(t=ast.IntType(), attrs=StaticAttr(init=Initial((IntInit(5),)), globl=True)),
            "foo": SymbolTableEntry(
                t=ast.FunType(params=(), ret=ast.IntType()), attrs=FunAttr(defined=True, globl=True)
            ),
            "ext": SymbolTableEntry(t=ast.IntType(), attrs=StaticAttr(init=NoInitializer(), globl=True)),
        }
        result = _convert_symbols_to_tacky(symbols)
        assert len(result) == 1
        assert result[0].name == "g"

    def test_tentative_int_emits_zero_init_4(self) -> None:
        """A tentative ``int`` emits ``ZeroInit(4)``."""
        symbols = {"x": SymbolTableEntry(t=ast.IntType(), attrs=StaticAttr(init=Tentative(), globl=False))}
        result = _convert_symbols_to_tacky(symbols)
        assert result[0].init == (tac.ZeroInit(4),)

    def test_tentative_long_emits_zero_init_8(self) -> None:
        """A tentative ``long`` emits ``ZeroInit(8)``."""
        symbols = {"x": SymbolTableEntry(t=ast.LongType(), attrs=StaticAttr(init=Tentative(), globl=False))}
        result = _convert_symbols_to_tacky(symbols)
        assert result[0].init == (tac.ZeroInit(8),)

    def test_tentative_int_array_emits_zero_init_for_total_bytes(self) -> None:
        """A tentative ``int[3]`` emits a single ``ZeroInit(12)`` covering all 12 bytes."""
        t = ast.Array(element=ast.IntType(), size=3)
        symbols = {"arr": SymbolTableEntry(t=t, attrs=StaticAttr(init=Tentative(), globl=False))}
        result = _convert_symbols_to_tacky(symbols)
        assert result[0].init == (tac.ZeroInit(12),)

    def test_tentative_long_array_emits_zero_init_for_total_bytes(self) -> None:
        """A tentative ``long[3]`` emits ``ZeroInit(24)`` covering all 24 bytes."""
        t = ast.Array(element=ast.LongType(), size=3)
        symbols = {"arr": SymbolTableEntry(t=t, attrs=StaticAttr(init=Tentative(), globl=False))}
        result = _convert_symbols_to_tacky(symbols)
        assert result[0].init == (tac.ZeroInit(24),)

    def test_zero_init_in_initial_emits_tac_zero_init(self) -> None:
        """``Initial((ZeroInit(4),))`` (zero scalar) produces ``tac.ZeroInit(4)``."""
        symbols = {"x": SymbolTableEntry(t=ast.IntType(), attrs=StaticAttr(init=Initial((ZeroInit(4),)), globl=False))}
        result = _convert_symbols_to_tacky(symbols)
        assert result[0].init == (tac.ZeroInit(4),)

    def test_zero_init_in_initial_long_emits_tac_zero_init_8(self) -> None:
        """``Initial((ZeroInit(8),))`` (zero long) produces ``tac.ZeroInit(8)``."""
        symbols = {"x": SymbolTableEntry(t=ast.LongType(), attrs=StaticAttr(init=Initial((ZeroInit(8),)), globl=False))}
        result = _convert_symbols_to_tacky(symbols)
        assert result[0].init == (tac.ZeroInit(8),)

    def test_zero_init_double_emits_double_init_not_zero_init(self) -> None:
        """``Initial((ZeroInit(8),))`` for a ``double`` produces ``tac.DoubleInit(0.0)``, not ``ZeroInit``."""
        symbols = {
            "x": SymbolTableEntry(t=ast.DoubleType(), attrs=StaticAttr(init=Initial((ZeroInit(8),)), globl=False))
        }
        result = _convert_symbols_to_tacky(symbols)
        assert result[0].init == (tac.DoubleInit(0.0),)

    def test_initial_with_nonzero_long_emits_long_init(self) -> None:
        """``Initial((LongInit(42),))`` produces ``tac.LongInit(42)`` (not ``ZeroInit``)."""
        symbols = {
            "x": SymbolTableEntry(t=ast.LongType(), attrs=StaticAttr(init=Initial((LongInit(42),)), globl=False))
        }
        result = _convert_symbols_to_tacky(symbols)
        assert result[0].init == (tac.LongInit(42),)


# ---------------------------------------------------------------------------
# Static local declarations produce no inline instructions
# ---------------------------------------------------------------------------


class TestTacGeneratorStaticLocalDeclarations:
    """Block-scope ``static`` and ``extern`` declarations generate no inline TAC; they are
    emitted via the symbol table instead."""

    def test_static_local_without_init_produces_no_instructions(self) -> None:
        """``static int x;`` inside a function body contributes no instructions beyond the guard."""
        instrs = _instrs_for_body(
            (
                ast.D(
                    declaration=ast.VarDecl(
                        variable_declaration=ast.VariableDeclaration(
                            name=".Vx.0", init=None, var_type=ast.IntType(), storage_class=ast.StorageClass.Static
                        )
                    )
                ),
            )
        )
        assert instrs == (tac.Return(val=tac.Constant(value=ast.ConstInt(0))),)

    def test_static_local_with_init_produces_no_instructions(self) -> None:
        """``static int x = 5;`` inside a function body contributes no inline instructions."""
        instrs = _instrs_for_body(
            (
                ast.D(
                    declaration=ast.VarDecl(
                        variable_declaration=ast.VariableDeclaration(
                            name=".Vx.0",
                            init=ast.Constant(value=ast.ConstInt(5)),
                            var_type=ast.IntType(),
                            storage_class=ast.StorageClass.Static,
                        )
                    )
                ),
            )
        )
        assert instrs == (tac.Return(val=tac.Constant(value=ast.ConstInt(0))),)

    def test_extern_local_produces_no_instructions(self) -> None:
        """``extern int x;`` inside a function body contributes no instructions beyond the guard."""
        instrs = _instrs_for_body(
            (
                ast.D(
                    declaration=ast.VarDecl(
                        variable_declaration=ast.VariableDeclaration(
                            name="g", init=None, var_type=ast.IntType(), storage_class=ast.StorageClass.Extern
                        )
                    )
                ),
            )
        )
        assert instrs == (tac.Return(val=tac.Constant(value=ast.ConstInt(0))),)


# ---------------------------------------------------------------------------
# generate() integrates static variables from the symbol table
# ---------------------------------------------------------------------------


class TestTacGeneratorIntegrationWithSymbolTable:
    """``generate`` combines function definitions with ``StaticVariable`` nodes from the symbol table."""

    def _minimal_prog(self) -> ast.Program:
        """Return a minimal program with a single ``main`` function that returns 0."""
        fn = ast.Function(
            name="main",
            params=(),
            body=(ast.S(statement=ast.Return(exp=ast.Constant(value=ast.ConstInt(0)))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
        )
        return ast.Program(declarations=(ast.FunDecl(fn),))

    def test_initial_symbol_produces_static_variable_in_top_level(self) -> None:
        """A symbol table entry with ``Initial`` produces a ``StaticVariable`` in ``top_level``."""
        symbols = {"g": SymbolTableEntry(t=ast.IntType(), attrs=StaticAttr(init=Initial((IntInit(42),)), globl=True))}
        result = generate(self._minimal_prog(), symbols)
        static_vars = [t for t in result.top_level if isinstance(t, tac.StaticVariable)]
        assert len(static_vars) == 1
        assert static_vars[0] == tac.StaticVariable(name="g", globl=True, t=ast.IntType(), init=(tac.IntInit(42),))

    def test_tentative_symbol_produces_zero_initialized_static_variable(self) -> None:
        """A ``Tentative`` entry produces a ``StaticVariable`` with ``init=(ZeroInit(4),)`` for an int."""
        symbols = {"t": SymbolTableEntry(t=ast.IntType(), attrs=StaticAttr(init=Tentative(), globl=False))}
        result = generate(self._minimal_prog(), symbols)
        static_vars = [t for t in result.top_level if isinstance(t, tac.StaticVariable)]
        assert len(static_vars) == 1
        assert static_vars[0].init == (tac.ZeroInit(4),)

    def test_no_initializer_symbol_produces_no_static_variable(self) -> None:
        """A ``NoInitializer`` entry (non-defining ``extern``) produces no ``StaticVariable``."""
        symbols = {"ext": SymbolTableEntry(t=ast.IntType(), attrs=StaticAttr(init=NoInitializer(), globl=True))}
        result = generate(self._minimal_prog(), symbols)
        static_vars = [t for t in result.top_level if isinstance(t, tac.StaticVariable)]
        assert static_vars == []

    def test_static_variables_appear_before_functions_in_top_level(self) -> None:
        """``StaticVariable`` nodes precede ``Function`` nodes in ``top_level``."""
        symbols = {"g": SymbolTableEntry(t=ast.IntType(), attrs=StaticAttr(init=Initial((IntInit(1),)), globl=True))}
        result = generate(self._minimal_prog(), symbols)
        assert isinstance(result.top_level[0], tac.StaticVariable)
        assert isinstance(result.top_level[1], tac.Function)

    def test_empty_symbol_table_produces_only_functions(self) -> None:
        """An empty symbol table results in ``top_level`` containing only functions."""
        result = generate(self._minimal_prog(), {})
        assert all(isinstance(t, tac.Function) for t in result.top_level)


# ---------------------------------------------------------------------------
# tac.Program.functions property
# ---------------------------------------------------------------------------


class TestTacProgramFunctionsProperty:
    """``tac.Program.functions`` returns only the ``Function`` entries from ``top_level``."""

    def test_functions_returns_function_entries(self) -> None:
        """A program with one function returns a single-element tuple from ``.functions``."""
        program = tac.Program(top_level=(tac.Function(name="main"),))
        assert len(program.functions) == 1
        assert program.functions[0].name == "main"

    def test_functions_filters_out_static_variables(self) -> None:
        """``StaticVariable`` entries are excluded from ``.functions``."""
        program = tac.Program(
            top_level=(
                tac.StaticVariable(name="g", globl=True, t=ast.IntType(), init=(tac.IntInit(0),)),
                tac.Function(name="main"),
            )
        )
        assert len(program.functions) == 1
        assert program.functions[0].name == "main"

    def test_functions_empty_when_only_static_variables(self) -> None:
        """If ``top_level`` contains only ``StaticVariable`` nodes, ``.functions`` is empty."""
        program = tac.Program(
            top_level=(tac.StaticVariable(name="g", globl=True, t=ast.IntType(), init=(tac.IntInit(0),)),)
        )
        assert program.functions == ()

    def test_functions_preserves_order(self) -> None:
        """Functions appear in ``.functions`` in the same order as they appear in ``top_level``."""
        program = tac.Program(
            top_level=(
                tac.Function(name="alpha"),
                tac.Function(name="beta"),
            )
        )
        assert program.functions[0].name == "alpha"
        assert program.functions[1].name == "beta"

    def test_functions_empty_when_top_level_is_empty(self) -> None:
        """``.functions`` on a program with no ``top_level`` entries returns an empty tuple."""
        assert tac.Program(top_level=()).functions == ()


# ---------------------------------------------------------------------------
# tac.Function globl field
# ---------------------------------------------------------------------------


class TestTacFunctionGlobl:
    """The ``globl`` field on ``tac.Function`` is propagated from the AST function."""

    def _make_fn(self, name: str, globl: bool) -> ast.Function:
        return ast.Function(
            name=name,
            params=(),
            body=(ast.S(statement=ast.Return(exp=ast.Constant(value=ast.ConstInt(0)))),),
            fun_type=ast.FunType(params=(), ret=ast.IntType()),
            globl=globl,
        )

    def test_globl_true_propagated_to_tac_function(self) -> None:
        """An ``ast.Function`` with ``globl=True`` produces a ``tac.Function`` with ``globl=True``."""
        prog = ast.Program(declarations=(ast.FunDecl(self._make_fn("main", globl=True)),))
        result = generate(prog, {})
        assert result.functions[0].globl is True

    def test_globl_false_propagated_to_tac_function(self) -> None:
        """An ``ast.Function`` with ``globl=False`` produces a ``tac.Function`` with ``globl=False``."""
        prog = ast.Program(declarations=(ast.FunDecl(self._make_fn("helper", globl=False)),))
        result = generate(prog, {})
        assert result.functions[0].globl is False
