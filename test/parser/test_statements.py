"""Parser tests for statement constructs."""

import pytest

from compiler.parser import parse
from compiler.syntax.ast import (
    Block,
    Break,
    Case,
    Compound,
    Constant,
    ConstInt,
    Continue,
    D,
    Default,
    DoWhile,
    Expression,
    For,
    Goto,
    If,
    InitDecl,
    InitExp,
    IntType,
    Label,
    Null,
    Return,
    S,
    SingleInit,
    StorageClass,
    Switch,
    VarDecl,
    VariableDeclaration,
    While,
)
from compiler.syntax.ast.expressions import Binary, Unary, Var
from compiler.syntax.ast.operators import Add, GreaterThan, LessThan, PostIncrement
from compiler.utils.exceptions import ParseError
from test.parser.helpers import (
    _ASSIGNMENT,
    _BREAK,
    _CASE,
    _COLON,
    _CONTINUE,
    _DEFAULT,
    _DO,
    _ELSE,
    _EXTERN,
    _FOR,
    _GOTO,
    _GT,
    _IF,
    _INCREMENT,
    _INT,
    _LBRACE,
    _LPAREN,
    _LT,
    _PLUS,
    _RBRACE,
    _RETURN,
    _RPAREN,
    _SEMICOLON,
    _STATIC,
    _SWITCH,
    _VOID,
    _WHILE,
    _ident,
    _number,
    _parsed_body,
    _parsed_stmt,
)

# ---------------------------------------------------------------------------
# If / else statements
# ---------------------------------------------------------------------------


class TestIfStatements:
    """Parser correctly builds ``If`` nodes for if/else statements."""

    def test_if_without_else(self) -> None:
        """
        ``if (1) return 0;`` → ``If(condition=Constant(ConstInt(1)), then=Return(Constant(ConstInt(0))), else_=None)``.
        """
        result = _parsed_stmt(_IF, _LPAREN, _number("1"), _RPAREN, _RETURN, _number("0"), _SEMICOLON)
        assert result == If(condition=Constant(ConstInt(1)), then=Return(exp=Constant(ConstInt(0))), else_=None)

    def test_if_with_else(self) -> None:
        """``if (1) return 1; else return 0;`` → ``If`` with both branches populated."""
        result = _parsed_stmt(
            _IF,
            _LPAREN,
            _number("1"),
            _RPAREN,
            _RETURN,
            _number("1"),
            _SEMICOLON,
            _ELSE,
            _RETURN,
            _number("0"),
            _SEMICOLON,
        )
        assert result == If(
            condition=Constant(ConstInt(1)),
            then=Return(exp=Constant(ConstInt(1))),
            else_=Return(exp=Constant(ConstInt(0))),
        )

    def test_if_null_then_clause(self) -> None:
        """``if (1);`` → ``If`` whose ``then`` is a ``Null`` statement."""
        result = _parsed_stmt(_IF, _LPAREN, _number("1"), _RPAREN, _SEMICOLON)
        assert result == If(condition=Constant(ConstInt(1)), then=Null(), else_=None)

    def test_if_null_else_clause(self) -> None:
        """``if (1) return 1; else;`` → ``If`` whose ``else_`` is a ``Null`` statement."""
        result = _parsed_stmt(
            _IF,
            _LPAREN,
            _number("1"),
            _RPAREN,
            _RETURN,
            _number("1"),
            _SEMICOLON,
            _ELSE,
            _SEMICOLON,
        )
        assert result == If(
            condition=Constant(ConstInt(1)),
            then=Return(exp=Constant(ConstInt(1))),
            else_=Null(),
        )

    def test_if_without_else_does_not_consume_next_statement(self) -> None:
        """``if (1) return 1; return 0;`` — the ``return 0`` is not swallowed by the if."""
        body = _parsed_body(
            _IF,
            _LPAREN,
            _number("1"),
            _RPAREN,
            _RETURN,
            _number("1"),
            _SEMICOLON,
            _RETURN,
            _number("0"),
            _SEMICOLON,
        )
        assert len(body) == 2
        item = body[1]
        assert isinstance(item, S)
        assert item.statement == Return(exp=Constant(ConstInt(0)))

    def test_dangling_else_binds_to_nearest_if(self) -> None:
        """``if (1) if (2) return 2; else return 1;`` — else belongs to the inner if (C rule)."""
        result = _parsed_stmt(
            _IF,
            _LPAREN,
            _number("1"),
            _RPAREN,
            _IF,
            _LPAREN,
            _number("2"),
            _RPAREN,
            _RETURN,
            _number("2"),
            _SEMICOLON,
            _ELSE,
            _RETURN,
            _number("1"),
            _SEMICOLON,
        )
        assert result == If(
            condition=Constant(ConstInt(1)),
            then=If(
                condition=Constant(ConstInt(2)),
                then=Return(exp=Constant(ConstInt(2))),
                else_=Return(exp=Constant(ConstInt(1))),
            ),
            else_=None,
        )

    def test_if_condition_is_arbitrary_expression(self) -> None:
        """``if (a > 1) return a;`` — comparison expression as condition."""
        result = _parsed_stmt(
            _IF,
            _LPAREN,
            _ident("a"),
            _GT,
            _number("1"),
            _RPAREN,
            _RETURN,
            _ident("a"),
            _SEMICOLON,
        )
        assert result == If(
            condition=Binary(GreaterThan(), Var("a"), Constant(ConstInt(1))),
            then=Return(exp=Var("a")),
            else_=None,
        )

    def test_if_then_is_expression_statement(self) -> None:
        """``if (1) a;`` — expression statement (not return) in the then branch."""
        result = _parsed_stmt(_IF, _LPAREN, _number("1"), _RPAREN, _ident("a"), _SEMICOLON)
        assert result == If(condition=Constant(ConstInt(1)), then=Expression(exp=Var("a")), else_=None)

    def test_if_missing_lparen(self) -> None:
        """``if 1) return 0;`` — no opening parenthesis raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _IF,
            _number("1"),
            _RPAREN,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected LPAREN"):
            parse(tokens)

    def test_if_missing_rparen(self) -> None:
        """``if (1 return 0;`` — no closing parenthesis raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _IF,
            _LPAREN,
            _number("1"),
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected RPAREN"):
            parse(tokens)


# ---------------------------------------------------------------------------
# Goto statements
# ---------------------------------------------------------------------------


class TestGotoStatements:
    """Parser correctly builds ``Goto`` nodes for goto statements."""

    def test_goto_produces_goto_node(self) -> None:
        """``goto foo;`` → ``Goto(target="foo")``."""
        assert _parsed_stmt(_GOTO, _ident("foo"), _SEMICOLON) == Goto(target="foo")

    def test_goto_target_name_is_preserved(self) -> None:
        """The label name is stored verbatim in ``Goto.target``."""
        result = _parsed_stmt(_GOTO, _ident("my_label"), _SEMICOLON)
        assert isinstance(result, Goto)
        assert result.target == "my_label"

    def test_goto_missing_semicolon_raises_parse_error(self) -> None:
        """``goto foo`` without a trailing semicolon raises ``ParseError``."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _GOTO, _ident("foo"), _RBRACE]
        with pytest.raises(ParseError, match="expected SEMICOLON"):
            parse(tokens)

    def test_goto_missing_identifier_raises_parse_error(self) -> None:
        """``goto;`` without a label name raises ``ParseError``."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _GOTO, _SEMICOLON, _RBRACE]
        with pytest.raises(ParseError, match="expected IDENTIFIER"):
            parse(tokens)


# ---------------------------------------------------------------------------
# Labeled statements
# ---------------------------------------------------------------------------


class TestLabeledStatements:
    """Parser correctly builds ``Label`` nodes for labeled statements."""

    def test_label_wraps_return_statement(self) -> None:
        """``foo: return 0;`` → ``Label(name="foo", statement=Return(Constant(ConstInt(0))))``."""
        assert _parsed_stmt(_ident("foo"), _COLON, _RETURN, _number("0"), _SEMICOLON) == Label(
            name="foo", statement=Return(exp=Constant(ConstInt(0)))
        )

    def test_label_wraps_null_statement(self) -> None:
        """``foo:;`` → ``Label(name="foo", statement=Null())``."""
        assert _parsed_stmt(_ident("foo"), _COLON, _SEMICOLON) == Label(name="foo", statement=Null())

    def test_label_wraps_expression_statement(self) -> None:
        """``foo: a;`` → ``Label(name="foo", statement=Expression(Var("a")))``."""
        assert _parsed_stmt(_ident("foo"), _COLON, _ident("a"), _SEMICOLON) == Label(
            name="foo", statement=Expression(exp=Var("a"))
        )

    def test_nested_labels(self) -> None:
        """``foo: bar: return 0;`` → ``Label("foo", Label("bar", Return(Constant(ConstInt(0)))))``."""
        assert _parsed_stmt(_ident("foo"), _COLON, _ident("bar"), _COLON, _RETURN, _number("0"), _SEMICOLON) == Label(
            name="foo", statement=Label(name="bar", statement=Return(exp=Constant(ConstInt(0))))
        )

    def test_label_target_name_is_preserved(self) -> None:
        """The label identifier is stored verbatim in ``Label.name``."""
        result = _parsed_stmt(_ident("my_label"), _COLON, _SEMICOLON)
        assert isinstance(result, Label)
        assert result.name == "my_label"

    def test_identifier_without_colon_is_expression_statement(self) -> None:
        """``foo;`` — a bare identifier not followed by ``:`` is an expression statement."""
        assert _parsed_stmt(_ident("foo"), _SEMICOLON) == Expression(exp=Var("foo"))


# ---------------------------------------------------------------------------
# Compound (block) statements
# ---------------------------------------------------------------------------


class TestCompoundStatements:
    """Parser correctly builds ``Compound`` nodes for ``{ block_item* }`` statements."""

    def test_empty_compound(self) -> None:
        """``{}`` as a statement → ``Compound(body=())``."""
        assert _parsed_stmt(_LBRACE, _RBRACE) == Compound(body=())

    def test_compound_with_single_return(self) -> None:
        """``{ return 0; }`` → ``Compound`` wrapping a single ``Return``."""
        assert _parsed_stmt(_LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE) == Compound(
            body=(S(statement=Return(exp=Constant(ConstInt(0)))),)
        )

    def test_compound_with_declaration(self) -> None:
        """``{ int x; }`` → ``Compound`` whose body contains a single ``D`` item."""
        assert _parsed_stmt(_LBRACE, _INT, _ident("x"), _SEMICOLON, _RBRACE) == Compound(
            body=(
                D(
                    declaration=VarDecl(
                        variable_declaration=VariableDeclaration(name="x", init=None, var_type=IntType())
                    )
                ),
            )
        )

    def test_compound_with_initialised_declaration(self) -> None:
        """``{ int x = 1; }`` → ``Compound`` with a ``D`` whose declaration carries an initialiser."""
        assert _parsed_stmt(_LBRACE, _INT, _ident("x"), _ASSIGNMENT, _number("1"), _SEMICOLON, _RBRACE) == Compound(
            body=(
                D(
                    declaration=VarDecl(
                        variable_declaration=VariableDeclaration(
                            name="x", init=SingleInit(exp=Constant(ConstInt(1))), var_type=IntType()
                        )
                    )
                ),
            )
        )

    def test_compound_with_multiple_items(self) -> None:
        """``{ int x = 1; return x; }`` → body contains a declaration then a statement."""
        assert _parsed_stmt(
            _LBRACE,
            _INT,
            _ident("x"),
            _ASSIGNMENT,
            _number("1"),
            _SEMICOLON,
            _RETURN,
            _ident("x"),
            _SEMICOLON,
            _RBRACE,
        ) == Compound(
            body=(
                D(
                    declaration=VarDecl(
                        variable_declaration=VariableDeclaration(
                            name="x", init=SingleInit(exp=Constant(ConstInt(1))), var_type=IntType()
                        )
                    )
                ),
                S(statement=Return(exp=Var("x"))),
            )
        )

    def test_nested_compound(self) -> None:
        """``{ { return 0; } }`` → outer ``Compound`` wrapping an inner ``Compound``."""
        assert _parsed_stmt(_LBRACE, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE, _RBRACE) == Compound(
            body=(S(statement=Compound(body=(S(statement=Return(exp=Constant(ConstInt(0)))),))),)
        )

    def test_compound_does_not_consume_subsequent_block_items(self) -> None:
        """``{ {} return 0; }`` — the compound's ``}`` does not swallow the following ``return``."""
        body = _parsed_body(_LBRACE, _RBRACE, _RETURN, _number("0"), _SEMICOLON)
        assert body == (S(statement=Compound(body=())), S(statement=Return(exp=Constant(ConstInt(0)))))

    def test_compound_as_if_then_branch(self) -> None:
        """``if (1) { return 0; }`` → ``If`` with a ``Compound`` as the then-branch."""
        result = _parsed_stmt(
            _IF,
            _LPAREN,
            _number("1"),
            _RPAREN,
            _LBRACE,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        )
        assert result == If(
            condition=Constant(ConstInt(1)),
            then=Compound(body=(S(statement=Return(exp=Constant(ConstInt(0)))),)),
            else_=None,
        )

    def test_compound_as_else_branch(self) -> None:
        """``if (1) return 1; else { return 0; }`` → ``If`` with a ``Compound`` else-branch."""
        result = _parsed_stmt(
            _IF,
            _LPAREN,
            _number("1"),
            _RPAREN,
            _RETURN,
            _number("1"),
            _SEMICOLON,
            _ELSE,
            _LBRACE,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        )
        assert result == If(
            condition=Constant(ConstInt(1)),
            then=Return(exp=Constant(ConstInt(1))),
            else_=Compound(body=(S(statement=Return(exp=Constant(ConstInt(0)))),)),
        )

    def test_label_wraps_compound(self) -> None:
        """``foo: { return 0; }`` → ``Label`` whose governed statement is a ``Compound``."""
        result = _parsed_stmt(_ident("foo"), _COLON, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE)
        assert result == Label(
            name="foo",
            statement=Compound(body=(S(statement=Return(exp=Constant(ConstInt(0)))),)),
        )

    def test_compound_missing_closing_brace_raises_parse_error(self) -> None:
        """``int main(void) { { return 0; }`` — missing outer ``}`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _LBRACE,
            _RETURN,
            _number("0"),
            _SEMICOLON,
        ]
        with pytest.raises(ParseError):
            parse(tokens)

    def test_compound_body_is_block_type(self) -> None:
        """``Compound.body`` is an instance of ``Block`` (i.e., ``tuple[BlockItem, ...]``)."""
        result = _parsed_stmt(_LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE)
        assert isinstance(result, Compound)
        assert isinstance(result.body, tuple)
        # Block is a type alias — verify the runtime type satisfies it
        b: Block = result.body
        assert b == (S(statement=Return(exp=Constant(ConstInt(0)))),)


# ---------------------------------------------------------------------------
# Break and continue statements
# ---------------------------------------------------------------------------


class TestBreakContinueStatements:
    """Parser correctly builds ``Break`` and ``Continue`` nodes."""

    def test_break_produces_break_node(self) -> None:
        """``break;`` → ``Break()``."""
        assert _parsed_stmt(_BREAK, _SEMICOLON) == Break()

    def test_continue_produces_continue_node(self) -> None:
        """``continue;`` → ``Continue()``."""
        assert _parsed_stmt(_CONTINUE, _SEMICOLON) == Continue()

    def test_break_missing_semicolon_raises_parse_error(self) -> None:
        """``break`` without a trailing semicolon raises ``ParseError``."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _BREAK, _RBRACE]
        with pytest.raises(ParseError, match="expected SEMICOLON"):
            parse(tokens)

    def test_continue_missing_semicolon_raises_parse_error(self) -> None:
        """``continue`` without a trailing semicolon raises ``ParseError``."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _CONTINUE, _RBRACE]
        with pytest.raises(ParseError, match="expected SEMICOLON"):
            parse(tokens)


# ---------------------------------------------------------------------------
# While statements
# ---------------------------------------------------------------------------


class TestWhileStatements:
    """Parser correctly builds ``While`` nodes for while loops."""

    def test_while_with_constant_condition(self) -> None:
        """``while (1) return 0;`` → ``While(condition=Constant(ConstInt(1)), body=Return(Constant(ConstInt(0))))``."""
        result = _parsed_stmt(_WHILE, _LPAREN, _number("1"), _RPAREN, _RETURN, _number("0"), _SEMICOLON)
        assert result == While(condition=Constant(ConstInt(1)), body=Return(exp=Constant(ConstInt(0))))

    def test_while_with_compound_body(self) -> None:
        """``while (1) {}`` → ``While`` with an empty ``Compound`` body."""
        result = _parsed_stmt(_WHILE, _LPAREN, _number("1"), _RPAREN, _LBRACE, _RBRACE)
        assert result == While(condition=Constant(ConstInt(1)), body=Compound(body=()))

    def test_while_body_is_null_statement(self) -> None:
        """``while (1) ;`` → ``While`` whose body is a ``Null`` statement."""
        result = _parsed_stmt(_WHILE, _LPAREN, _number("1"), _RPAREN, _SEMICOLON)
        assert result == While(condition=Constant(ConstInt(1)), body=Null())

    def test_while_condition_is_arbitrary_expression(self) -> None:
        """``while (a < 10) ;`` — comparison expression as condition."""
        result = _parsed_stmt(_WHILE, _LPAREN, _ident("a"), _LT, _number("10"), _RPAREN, _SEMICOLON)
        assert result == While(condition=Binary(LessThan(), Var("a"), Constant(ConstInt(10))), body=Null())

    def test_while_missing_lparen_raises_parse_error(self) -> None:
        """``while 1) ;`` — missing opening parenthesis raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _WHILE,
            _number("1"),
            _RPAREN,
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected LPAREN"):
            parse(tokens)

    def test_while_missing_rparen_raises_parse_error(self) -> None:
        """``while (1 ;`` — missing closing parenthesis raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _WHILE,
            _LPAREN,
            _number("1"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected RPAREN"):
            parse(tokens)


# ---------------------------------------------------------------------------
# Do/while statements
# ---------------------------------------------------------------------------


class TestDoWhileStatements:
    """Parser correctly builds ``DoWhile`` nodes for do/while loops."""

    def test_do_while_basic(self) -> None:
        """
        ``do return 0; while (1);`` → ``DoWhile(body=Return(Constant(ConstInt(0))), condition=Constant(ConstInt(1)))``.
        """
        result = _parsed_stmt(
            _DO, _RETURN, _number("0"), _SEMICOLON, _WHILE, _LPAREN, _number("1"), _RPAREN, _SEMICOLON
        )
        assert result == DoWhile(body=Return(exp=Constant(ConstInt(0))), condition=Constant(ConstInt(1)))

    def test_do_while_with_compound_body(self) -> None:
        """``do {} while (1);`` → ``DoWhile`` with an empty ``Compound`` body."""
        result = _parsed_stmt(_DO, _LBRACE, _RBRACE, _WHILE, _LPAREN, _number("1"), _RPAREN, _SEMICOLON)
        assert result == DoWhile(body=Compound(body=()), condition=Constant(ConstInt(1)))

    def test_do_while_condition_is_arbitrary_expression(self) -> None:
        """``do ; while (a < 10);`` — comparison expression as condition."""
        result = _parsed_stmt(_DO, _SEMICOLON, _WHILE, _LPAREN, _ident("a"), _LT, _number("10"), _RPAREN, _SEMICOLON)
        assert result == DoWhile(body=Null(), condition=Binary(LessThan(), Var("a"), Constant(ConstInt(10))))

    def test_do_while_missing_while_keyword_raises_parse_error(self) -> None:
        """``do ; (1);`` — absent ``while`` keyword raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _DO,
            _SEMICOLON,
            _LPAREN,
            _number("1"),
            _RPAREN,
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected WHILE"):
            parse(tokens)

    def test_do_while_missing_closing_paren_raises_parse_error(self) -> None:
        """``do ; while (1;`` — missing ``)`` after condition raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _DO,
            _SEMICOLON,
            _WHILE,
            _LPAREN,
            _number("1"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected RPAREN"):
            parse(tokens)

    def test_do_while_missing_trailing_semicolon_raises_parse_error(self) -> None:
        """``do ; while (1)`` — missing trailing ``;`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _DO,
            _SEMICOLON,
            _WHILE,
            _LPAREN,
            _number("1"),
            _RPAREN,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected SEMICOLON"):
            parse(tokens)


# ---------------------------------------------------------------------------
# For statements
# ---------------------------------------------------------------------------


class TestForStatements:
    """Parser correctly builds ``For`` nodes for all for-loop clause combinations."""

    def test_for_with_declaration_init(self) -> None:
        """``for (int i = 0; i < 10; i++) return 0;`` — declaration init clause."""
        result = _parsed_stmt(
            _FOR,
            _LPAREN,
            _INT,
            _ident("i"),
            _ASSIGNMENT,
            _number("0"),
            _SEMICOLON,
            _ident("i"),
            _LT,
            _number("10"),
            _SEMICOLON,
            _ident("i"),
            _INCREMENT,
            _RPAREN,
            _RETURN,
            _number("0"),
            _SEMICOLON,
        )
        assert result == For(
            init=InitDecl(
                declaration=VarDecl(
                    variable_declaration=VariableDeclaration(
                        name="i", init=SingleInit(exp=Constant(ConstInt(0))), var_type=IntType()
                    )
                )
            ),
            condition=Binary(LessThan(), Var("i"), Constant(ConstInt(10))),
            post=Unary(PostIncrement(), Var("i")),
            body=Return(exp=Constant(ConstInt(0))),
        )

    def test_for_with_expression_init(self) -> None:
        """``for (i = 0; i < 10; i++) ;`` — expression init clause."""
        from compiler.syntax.ast.expressions import Assignment

        result = _parsed_stmt(
            _FOR,
            _LPAREN,
            _ident("i"),
            _ASSIGNMENT,
            _number("0"),
            _SEMICOLON,
            _ident("i"),
            _LT,
            _number("10"),
            _SEMICOLON,
            _ident("i"),
            _INCREMENT,
            _RPAREN,
            _SEMICOLON,
        )
        assert result == For(
            init=InitExp(exp=Assignment(lvalue=Var("i"), rvalue=Constant(ConstInt(0)))),
            condition=Binary(LessThan(), Var("i"), Constant(ConstInt(10))),
            post=Unary(PostIncrement(), Var("i")),
            body=Null(),
        )

    def test_for_with_empty_init(self) -> None:
        """``for (; i < 10; i++) ;`` — absent init clause yields ``InitExp(exp=None)``."""
        result = _parsed_stmt(
            _FOR,
            _LPAREN,
            _SEMICOLON,
            _ident("i"),
            _LT,
            _number("10"),
            _SEMICOLON,
            _ident("i"),
            _INCREMENT,
            _RPAREN,
            _SEMICOLON,
        )
        assert result == For(
            init=InitExp(exp=None),
            condition=Binary(LessThan(), Var("i"), Constant(ConstInt(10))),
            post=Unary(PostIncrement(), Var("i")),
            body=Null(),
        )

    def test_for_with_no_condition(self) -> None:
        """``for (int i = 0;; i++) ;`` — absent condition yields ``condition=None``."""
        result = _parsed_stmt(
            _FOR,
            _LPAREN,
            _INT,
            _ident("i"),
            _ASSIGNMENT,
            _number("0"),
            _SEMICOLON,
            _SEMICOLON,
            _ident("i"),
            _INCREMENT,
            _RPAREN,
            _SEMICOLON,
        )
        assert result == For(
            init=InitDecl(
                declaration=VarDecl(
                    variable_declaration=VariableDeclaration(
                        name="i", init=SingleInit(exp=Constant(ConstInt(0))), var_type=IntType()
                    )
                )
            ),
            condition=None,
            post=Unary(PostIncrement(), Var("i")),
            body=Null(),
        )

    def test_for_with_no_post(self) -> None:
        """``for (int i = 0; i < 10;) ;`` — absent post clause yields ``post=None``."""
        result = _parsed_stmt(
            _FOR,
            _LPAREN,
            _INT,
            _ident("i"),
            _ASSIGNMENT,
            _number("0"),
            _SEMICOLON,
            _ident("i"),
            _LT,
            _number("10"),
            _SEMICOLON,
            _RPAREN,
            _SEMICOLON,
        )
        assert result == For(
            init=InitDecl(
                declaration=VarDecl(
                    variable_declaration=VariableDeclaration(
                        name="i", init=SingleInit(exp=Constant(ConstInt(0))), var_type=IntType()
                    )
                )
            ),
            condition=Binary(LessThan(), Var("i"), Constant(ConstInt(10))),
            post=None,
            body=Null(),
        )

    def test_for_all_clauses_absent(self) -> None:
        """``for (;;) ;`` — all optional clauses absent."""
        result = _parsed_stmt(_FOR, _LPAREN, _SEMICOLON, _SEMICOLON, _RPAREN, _SEMICOLON)
        assert result == For(
            init=InitExp(exp=None),
            condition=None,
            post=None,
            body=Null(),
        )

    def test_for_body_is_compound(self) -> None:
        """``for (;;) { return 0; }`` — compound statement as the loop body."""
        result = _parsed_stmt(
            _FOR,
            _LPAREN,
            _SEMICOLON,
            _SEMICOLON,
            _RPAREN,
            _LBRACE,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        )
        assert result == For(
            init=InitExp(exp=None),
            condition=None,
            post=None,
            body=Compound(body=(S(statement=Return(exp=Constant(ConstInt(0)))),)),
        )

    def test_for_missing_lparen_raises_parse_error(self) -> None:
        """``for ;; ) ;`` — missing ``(`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _FOR,
            _SEMICOLON,
            _SEMICOLON,
            _RPAREN,
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected LPAREN"):
            parse(tokens)

    def test_for_missing_first_semicolon_raises_parse_error(self) -> None:
        """``for (i) ;`` — init clause expression not terminated with ``;`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _FOR,
            _LPAREN,
            _ident("i"),
            _RPAREN,
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected SEMICOLON"):
            parse(tokens)

    def test_for_missing_second_semicolon_raises_parse_error(self) -> None:
        """``for (; i) ;`` — condition expression not terminated with ``;`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _FOR,
            _LPAREN,
            _SEMICOLON,
            _ident("i"),
            _RPAREN,
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected SEMICOLON"):
            parse(tokens)

    def test_for_missing_rparen_raises_parse_error(self) -> None:
        """``for (;;`` — missing ``)`` after post expression raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _FOR,
            _LPAREN,
            _SEMICOLON,
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError):
            parse(tokens)

    def test_for_function_declaration_in_init_raises_parse_error(self) -> None:
        """``for (int f(void); ; ) {}`` — function declaration in for-init is rejected."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _FOR,
            _LPAREN,
            _INT,
            _ident("f"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _SEMICOLON,
            _SEMICOLON,
            _RPAREN,
            _LBRACE,
            _RBRACE,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="function declarations are not permitted"):
            parse(tokens)

    def test_for_static_init_records_storage_class(self) -> None:
        """``for (static int i = 0; ...) {}`` — parser accepts and records storage class."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _FOR,
            _LPAREN,
            _STATIC,
            _INT,
            _ident("i"),
            _ASSIGNMENT,
            _number("0"),
            _SEMICOLON,
            _SEMICOLON,
            _RPAREN,
            _LBRACE,
            _RBRACE,
            _RBRACE,
        ]
        result = parse(tokens)
        body = result.declarations[0].function_declaration.body
        for_stmt = body[0].statement
        assert isinstance(for_stmt, For)
        assert isinstance(for_stmt.init, InitDecl)
        assert for_stmt.init.declaration.variable_declaration.storage_class == StorageClass.Static

    def test_for_extern_init_records_storage_class(self) -> None:
        """``for (extern int i = 0; ...) {}`` — parser accepts and records storage class."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _FOR,
            _LPAREN,
            _EXTERN,
            _INT,
            _ident("i"),
            _ASSIGNMENT,
            _number("0"),
            _SEMICOLON,
            _SEMICOLON,
            _RPAREN,
            _LBRACE,
            _RBRACE,
            _RBRACE,
        ]
        result = parse(tokens)
        body = result.declarations[0].function_declaration.body
        for_stmt = body[0].statement
        assert isinstance(for_stmt, For)
        assert isinstance(for_stmt.init, InitDecl)
        assert for_stmt.init.declaration.variable_declaration.storage_class == StorageClass.Extern


# ---------------------------------------------------------------------------
# Switch / case / default statements
# ---------------------------------------------------------------------------


class TestSwitchStatements:
    """Parser correctly builds ``Switch``, ``Case``, and ``Default`` nodes."""

    def test_switch_with_null_body(self) -> None:
        """``switch (1) ;`` → ``Switch(control=Constant(ConstInt(1)), body=Null())``."""
        result = _parsed_stmt(_SWITCH, _LPAREN, _number("1"), _RPAREN, _SEMICOLON)
        assert result == Switch(control=Constant(ConstInt(1)), body=Null())

    def test_switch_with_empty_compound_body(self) -> None:
        """``switch (1) {}`` → ``Switch`` whose body is an empty ``Compound``."""
        result = _parsed_stmt(_SWITCH, _LPAREN, _number("1"), _RPAREN, _LBRACE, _RBRACE)
        assert result == Switch(control=Constant(ConstInt(1)), body=Compound(body=()))

    def test_switch_control_is_arbitrary_expression(self) -> None:
        """``switch (a + 1) ;`` — binary expression as the control value."""
        result = _parsed_stmt(_SWITCH, _LPAREN, _ident("a"), _PLUS, _number("1"), _RPAREN, _SEMICOLON)
        assert result == Switch(control=Binary(Add(), Var("a"), Constant(ConstInt(1))), body=Null())

    def test_switch_identifier_is_empty_before_semantic_analysis(self) -> None:
        """``Switch.identifier`` is the empty string until semantic analysis assigns one."""
        result = _parsed_stmt(_SWITCH, _LPAREN, _number("1"), _RPAREN, _SEMICOLON)
        assert isinstance(result, Switch)
        assert result.identifier == ""

    def test_case_governing_null_statement(self) -> None:
        """``case 1: ;`` → ``Case(value=Constant(ConstInt(1)), statement=Null())``."""
        result = _parsed_stmt(_CASE, _number("1"), _COLON, _SEMICOLON)
        assert result == Case(value=Constant(ConstInt(1)), statement=Null())

    def test_case_governing_return_statement(self) -> None:
        """``case 2: return 0;`` → ``Case(value=Constant(ConstInt(2)), statement=Return(Constant(ConstInt(0))))``."""
        result = _parsed_stmt(_CASE, _number("2"), _COLON, _RETURN, _number("0"), _SEMICOLON)
        assert result == Case(value=Constant(ConstInt(2)), statement=Return(exp=Constant(ConstInt(0))))

    def test_case_value_must_be_constant_expression_raises_parse_error(self) -> None:
        """``case a: ;`` — a non-constant case value raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _CASE,
            _ident("a"),
            _COLON,
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError):
            parse(tokens)

    def test_case_identifier_is_empty_before_semantic_analysis(self) -> None:
        """``Case.identifier`` is the empty string until semantic analysis assigns one."""
        result = _parsed_stmt(_CASE, _number("1"), _COLON, _SEMICOLON)
        assert isinstance(result, Case)
        assert result.identifier == ""

    def test_default_governing_null_statement(self) -> None:
        """``default: ;`` → ``Default(statement=Null())``."""
        result = _parsed_stmt(_DEFAULT, _COLON, _SEMICOLON)
        assert result == Default(statement=Null())

    def test_default_governing_return_statement(self) -> None:
        """``default: return 0;`` → ``Default(statement=Return(Constant(ConstInt(0))))``."""
        result = _parsed_stmt(_DEFAULT, _COLON, _RETURN, _number("0"), _SEMICOLON)
        assert result == Default(statement=Return(exp=Constant(ConstInt(0))))

    def test_default_identifier_is_empty_before_semantic_analysis(self) -> None:
        """``Default.identifier`` is the empty string until semantic analysis assigns one."""
        result = _parsed_stmt(_DEFAULT, _COLON, _SEMICOLON)
        assert isinstance(result, Default)
        assert result.identifier == ""

    def test_fall_through_via_nested_case(self) -> None:
        """``case 1: case 2: return 0;`` — the inner ``Case`` is the governed statement of the outer.

        This is the canonical fall-through representation: ``Case(1, Case(2, Return(0)))``.
        """
        result = _parsed_stmt(
            _CASE, _number("1"), _COLON, _CASE, _number("2"), _COLON, _RETURN, _number("0"), _SEMICOLON
        )
        assert result == Case(
            value=Constant(ConstInt(1)),
            statement=Case(value=Constant(ConstInt(2)), statement=Return(exp=Constant(ConstInt(0)))),
        )

    def test_switch_with_case_in_compound_body(self) -> None:
        """``switch (x) { case 1: return 1; }`` — single case inside a compound body."""
        result = _parsed_stmt(
            _SWITCH,
            _LPAREN,
            _ident("x"),
            _RPAREN,
            _LBRACE,
            _CASE,
            _number("1"),
            _COLON,
            _RETURN,
            _number("1"),
            _SEMICOLON,
            _RBRACE,
        )
        assert result == Switch(
            control=Var("x"),
            body=Compound(
                body=(S(statement=Case(value=Constant(ConstInt(1)), statement=Return(exp=Constant(ConstInt(1))))),)
            ),
        )

    def test_switch_with_case_and_default_in_compound_body(self) -> None:
        """``switch (x) { case 1: return 1; default: return 0; }`` — case and default as siblings."""
        result = _parsed_stmt(
            _SWITCH,
            _LPAREN,
            _ident("x"),
            _RPAREN,
            _LBRACE,
            _CASE,
            _number("1"),
            _COLON,
            _RETURN,
            _number("1"),
            _SEMICOLON,
            _DEFAULT,
            _COLON,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        )
        assert result == Switch(
            control=Var("x"),
            body=Compound(
                body=(
                    S(statement=Case(value=Constant(ConstInt(1)), statement=Return(exp=Constant(ConstInt(1))))),
                    S(statement=Default(statement=Return(exp=Constant(ConstInt(0))))),
                )
            ),
        )

    def test_switch_missing_lparen_raises_parse_error(self) -> None:
        """``switch 1) ;`` — missing ``(`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _SWITCH,
            _number("1"),
            _RPAREN,
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected LPAREN"):
            parse(tokens)

    def test_switch_missing_rparen_raises_parse_error(self) -> None:
        """``switch (1 ;`` — missing ``)`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _SWITCH,
            _LPAREN,
            _number("1"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected RPAREN"):
            parse(tokens)

    def test_case_missing_colon_raises_parse_error(self) -> None:
        """``case 1 return 0;`` — missing ``:`` after the case value raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _CASE,
            _number("1"),
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected COLON"):
            parse(tokens)

    def test_default_missing_colon_raises_parse_error(self) -> None:
        """``default return 0;`` — missing ``:`` after ``default`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _DEFAULT,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected COLON"):
            parse(tokens)
