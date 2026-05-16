"""Parser tests for declarations and program-level constructs."""

import pytest

from compiler.parser import parse
from compiler.syntax.ast import (
    Constant,
    ConstInt,
    D,
    Function,
    FunctionCall,
    FunDecl,
    FunType,
    IntType,
    LongType,
    Parameter,
    Program,
    Return,
    S,
    StorageClass,
    Unary,
    VarDecl,
    VariableDeclaration,
)
from compiler.syntax.ast.expressions import Binary, Var
from compiler.syntax.ast.operators import Add, PostIncrement
from compiler.types.tokens import TokenType
from compiler.utils.exceptions import ParseError
from test.parser.helpers import (
    _ASSIGNMENT,
    _COMMA,
    _EXTERN,
    _INCREMENT,
    _INT,
    _LBRACE,
    _LPAREN,
    _PLUS,
    _RBRACE,
    _RETURN,
    _RPAREN,
    _SEMICOLON,
    _STATIC,
    _VOID,
    _ident,
    _number,
    _parsed_body,
    _parsed_exp,
    _tok,
)

# ---------------------------------------------------------------------------
# Valid programs
# ---------------------------------------------------------------------------


class TestValidPrograms:
    """Parser accepts well-formed programs and produces the correct AST."""

    def test_simple_function_void_params(self) -> None:
        """``int main(void) { return 0; }`` produces the expected AST."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        result = parse(tokens)
        expected = Program(
            declarations=(
                FunDecl(
                    function_declaration=Function(
                        name="main",
                        params=(),
                        body=(S(statement=Return(exp=Constant(value=ConstInt(0)))),),
                        fun_type=FunType(params=(), ret=IntType()),
                    ),
                ),
            )
        )
        assert result == expected

    def test_simple_function_empty_params(self) -> None:
        """``int main() { return 0; }`` — empty parens produce no parameters."""
        tokens = [_INT, _ident("main"), _LPAREN, _RPAREN, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.params == ()

    def test_nonzero_return_value(self) -> None:
        """``return 42;`` produces a ``Constant`` with the correct integer value."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("42"), _SEMICOLON, _RBRACE]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.body == (S(statement=Return(exp=Constant(value=ConstInt(42)))),)

    def test_single_int_parameter(self) -> None:
        """``int foo(int x) { return x; }`` — tests that a single typed parameter
        is parsed into a ``Parameter`` node."""
        tokens = [
            _INT,
            _ident("foo"),
            _LPAREN,
            _INT,
            _ident("x"),
            _RPAREN,
            _LBRACE,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        ]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.params == (Parameter(param_type=IntType(), name="x"),)

    def test_multiple_functions(self) -> None:
        """Two consecutive function definitions both appear in ``Program.declarations``."""

        def _simple_fn(name: str) -> list:
            return [_INT, _ident(name), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]

        tokens = _simple_fn("foo") + _simple_fn("bar")
        result = parse(tokens)
        assert len(result.declarations) == 2
        decl0 = result.declarations[0]
        decl1 = result.declarations[1]
        assert isinstance(decl0, FunDecl)
        assert isinstance(decl1, FunDecl)
        assert decl0.function_declaration.name == "foo"
        assert decl1.function_declaration.name == "bar"

    def test_multiple_statements_in_body(self) -> None:
        """Multiple ``return`` statements are all captured in the function body."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _RETURN,
            _number("1"),
            _SEMICOLON,
            _RETURN,
            _number("2"),
            _SEMICOLON,
            _RBRACE,
        ]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.body == (
            S(statement=Return(exp=Constant(value=ConstInt(1)))),
            S(statement=Return(exp=Constant(value=ConstInt(2)))),
        )


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestParseErrors:
    """Parser raises ``ParseError`` for malformed token streams."""

    def test_empty_token_stream(self) -> None:
        """An empty token stream raises ``ParseError``."""
        with pytest.raises(ParseError):
            parse([])

    def test_missing_function_name(self) -> None:
        """``int (void) { ... }`` — LPAREN where IDENTIFIER expected."""
        tokens = [_INT, _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        with pytest.raises(ParseError, match="expected IDENTIFIER"):
            parse(tokens)

    def test_missing_opening_paren(self) -> None:
        """``int main void) { ... }`` — parsed as variable declaration; fails at SEMICOLON."""
        tokens = [_INT, _ident("main"), _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        with pytest.raises(ParseError, match="expected SEMICOLON"):
            parse(tokens)

    def test_missing_closing_paren(self) -> None:
        """``int main(void { ... }`` — LBRACE where RPAREN expected."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        with pytest.raises(ParseError, match="expected RPAREN"):
            parse(tokens)

    def test_missing_opening_brace(self) -> None:
        """``int main(void) return 0; }`` — RETURN where LBRACE expected."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        with pytest.raises(ParseError, match="expected LBRACE"):
            parse(tokens)

    def test_empty_function_body(self) -> None:
        """``int main(void) {}`` — empty body is valid; produces an empty block-item tuple."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RBRACE]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.body == ()

    def test_missing_semicolon(self) -> None:
        """``return 0 }`` — missing semicolon after expression."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _RBRACE]
        with pytest.raises(ParseError, match="expected SEMICOLON"):
            parse(tokens)

    def test_non_int_return_type(self) -> None:
        """``void main(void) { ... }`` — VOID is not a valid return type in the grammar."""
        _VOID_tok = _tok(TokenType.VOID, "void")
        tokens = [
            _VOID_tok,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected a type"):
            parse(tokens)

    def test_non_number_expression(self) -> None:
        """``return );`` — RPAREN is not a valid expression start."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _RPAREN, _SEMICOLON, _RBRACE]
        with pytest.raises(ParseError, match="Malformed expression."):
            parse(tokens)

    def test_unexpected_eof_in_body(self) -> None:
        """Token stream ending mid-body raises ``ParseError`` for unexpected EOF."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("0")]
        with pytest.raises(ParseError):
            parse(tokens)

    def test_parse_error_carries_location(self) -> None:
        """``ParseError`` records the filename, line, and column of the offending token."""
        bad_token = _tok(TokenType.LPAREN, "(", line=5, col=12, filename="src/foo.c")
        tokens = [_INT, bad_token]
        with pytest.raises(ParseError) as exc_info:
            parse(tokens)
        err = exc_info.value
        assert err.filename == "src/foo.c"
        assert err.line == 5
        assert err.col == 12

    def test_parse_error_str_contains_location_header(self) -> None:
        """``str(ParseError)`` begins with the clang-style location header."""
        bad_token = _tok(TokenType.LPAREN, "(", line=3, col=7, filename="main.c")
        tokens = [_INT, bad_token]
        with pytest.raises(ParseError) as exc_info:
            parse(tokens)
        text = str(exc_info.value)
        assert text.startswith("ParseError at main.c:3:7:")

    def test_parse_error_str_header_only_when_no_source_lines(self) -> None:
        """When the source file cannot be read, ``str()`` is just the header line."""
        bad_token = _tok(TokenType.LPAREN, "(", line=1, col=1, filename="nonexistent.c")
        tokens = [_INT, bad_token]
        with pytest.raises(ParseError) as exc_info:
            parse(tokens)
        text = str(exc_info.value)
        assert "\n" not in text


# ---------------------------------------------------------------------------
# Multiple parameters
# ---------------------------------------------------------------------------


class TestMultipleParameters:
    """Parser correctly builds multi-parameter ``Function`` nodes."""

    def test_two_parameters(self) -> None:
        """``int foo(int a, int b) { return 0; }`` → two ``Parameter`` nodes."""
        tokens = [
            _INT,
            _ident("foo"),
            _LPAREN,
            _INT,
            _ident("a"),
            _COMMA,
            _INT,
            _ident("b"),
            _RPAREN,
            _LBRACE,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        ]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.params == (
            Parameter(param_type=IntType(), name="a"),
            Parameter(param_type=IntType(), name="b"),
        )

    def test_three_parameters(self) -> None:
        """``int foo(int a, int b, int c) { return 0; }`` → three ``Parameter`` nodes."""
        tokens = [
            _INT,
            _ident("foo"),
            _LPAREN,
            _INT,
            _ident("a"),
            _COMMA,
            _INT,
            _ident("b"),
            _COMMA,
            _INT,
            _ident("c"),
            _RPAREN,
            _LBRACE,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        ]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert len(decl.function_declaration.params) == 3
        assert decl.function_declaration.params[2] == Parameter(param_type=IntType(), name="c")

    def test_missing_type_in_param_list_raises_parse_error(self) -> None:
        """``int foo(int a, b)`` — missing type before second parameter raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("foo"),
            _LPAREN,
            _INT,
            _ident("a"),
            _COMMA,
            _ident("b"),
            _RPAREN,
            _LBRACE,
            _RBRACE,
        ]
        with pytest.raises(ParseError):
            parse(tokens)


# ---------------------------------------------------------------------------
# Function forward declarations
# ---------------------------------------------------------------------------


class TestFunctionForwardDeclarations:
    """Parser correctly builds forward declaration ``Function`` nodes (``body=None``)."""

    def test_forward_declaration_void_params(self) -> None:
        """``int foo(void);`` → ``FunDecl`` wrapping ``Function(name="foo", params=(), body=None)``."""
        tokens = [_INT, _ident("foo"), _LPAREN, _VOID, _RPAREN, _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration == Function(
            name="foo", params=(), body=None, fun_type=FunType(params=(), ret=IntType())
        )

    def test_forward_declaration_with_params(self) -> None:
        """``int add(int a, int b);`` → forward declaration with two params and ``body=None``."""
        tokens = [_INT, _ident("add"), _LPAREN, _INT, _ident("a"), _COMMA, _INT, _ident("b"), _RPAREN, _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        fn = decl.function_declaration
        assert fn.name == "add"
        assert fn.body is None
        assert fn.params == (Parameter(param_type=IntType(), name="a"), Parameter(param_type=IntType(), name="b"))

    def test_forward_declaration_followed_by_definition(self) -> None:
        """``int foo(void); int foo(void) { return 0; }`` → two entries in ``Program.declarations``."""
        proto = [_INT, _ident("foo"), _LPAREN, _VOID, _RPAREN, _SEMICOLON]
        defn = [_INT, _ident("foo"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        result = parse(proto + defn)
        assert len(result.declarations) == 2
        decl0 = result.declarations[0]
        decl1 = result.declarations[1]
        assert isinstance(decl0, FunDecl)
        assert isinstance(decl1, FunDecl)
        assert decl0.function_declaration.body is None
        assert decl1.function_declaration.body is not None

    def test_function_declaration_as_block_item(self) -> None:
        """A forward declaration inside a function body is parsed as a ``FunDecl`` block item."""
        proto = [_INT, _ident("bar"), _LPAREN, _VOID, _RPAREN, _SEMICOLON]
        body = _parsed_body(*proto, _RETURN, _number("0"), _SEMICOLON)
        item = body[0]
        assert isinstance(item, D)
        assert isinstance(item.declaration, FunDecl)
        assert item.declaration.function_declaration.name == "bar"
        assert item.declaration.function_declaration.body is None


# ---------------------------------------------------------------------------
# Function call expressions
# ---------------------------------------------------------------------------


class TestFunctionCallExpressions:
    """Parser correctly builds ``FunctionCall`` nodes for function call primaries."""

    def test_call_no_args(self) -> None:
        """``return foo();`` → ``FunctionCall(name="foo", args=[])``."""
        result = _parsed_exp(_ident("foo"), _LPAREN, _RPAREN)
        assert result == FunctionCall(name="foo", args=[])

    def test_call_one_arg(self) -> None:
        """``return foo(1);`` → ``FunctionCall(name="foo", args=[Constant(ConstInt(1))])``."""
        result = _parsed_exp(_ident("foo"), _LPAREN, _number("1"), _RPAREN)
        assert result == FunctionCall(name="foo", args=[Constant(ConstInt(1))])

    def test_call_two_args(self) -> None:
        """``return foo(1, 2);`` → two-element args list."""
        result = _parsed_exp(_ident("foo"), _LPAREN, _number("1"), _COMMA, _number("2"), _RPAREN)
        assert result == FunctionCall(name="foo", args=[Constant(ConstInt(1)), Constant(ConstInt(2))])

    def test_call_three_args(self) -> None:
        """``return foo(1, 2, 3);`` → three-element args list."""
        result = _parsed_exp(_ident("foo"), _LPAREN, _number("1"), _COMMA, _number("2"), _COMMA, _number("3"), _RPAREN)
        assert result == FunctionCall(
            name="foo", args=[Constant(ConstInt(1)), Constant(ConstInt(2)), Constant(ConstInt(3))]
        )

    def test_call_with_expression_arg(self) -> None:
        """``return foo(a + 1);`` — binary expression as the sole argument."""
        result = _parsed_exp(_ident("foo"), _LPAREN, _ident("a"), _PLUS, _number("1"), _RPAREN)
        assert result == FunctionCall(name="foo", args=[Binary(Add(), Var("a"), Constant(ConstInt(1)))])

    def test_call_result_used_in_binary(self) -> None:
        """``return foo() + 1;`` — function call as left operand of addition."""
        result = _parsed_exp(_ident("foo"), _LPAREN, _RPAREN, _PLUS, _number("1"))
        assert result == Binary(Add(), FunctionCall(name="foo", args=[]), Constant(ConstInt(1)))

    def test_call_with_postfix_increment(self) -> None:
        """``return foo()++;`` — postfix ``++`` applied to the call result."""
        result = _parsed_exp(_ident("foo"), _LPAREN, _RPAREN, _INCREMENT)
        assert result == Unary(PostIncrement(), FunctionCall(name="foo", args=[]))

    def test_call_missing_closing_paren_raises_parse_error(self) -> None:
        """``return foo(1;`` — missing ``)`` raises ``ParseError``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _RETURN,
            _ident("foo"),
            _LPAREN,
            _number("1"),
            _SEMICOLON,
            _RBRACE,
        ]
        with pytest.raises(ParseError, match="expected RPAREN"):
            parse(tokens)

    def test_identifier_followed_by_non_paren_is_var(self) -> None:
        """``return a;`` — an identifier not followed by ``(`` is a ``Var``, not a call."""
        result = _parsed_exp(_ident("a"))
        assert result == Var("a")


# ---------------------------------------------------------------------------
# Storage-class specifiers
# ---------------------------------------------------------------------------


class TestStorageClassSpecifiers:
    """Parser correctly attaches storage-class specifiers to function and variable declarations."""

    def test_static_variable_declaration(self) -> None:
        """``static int x;`` → ``VarDecl`` whose ``VariableDeclaration`` carries ``StorageClass.Static``."""
        tokens = [_STATIC, _INT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration == VariableDeclaration(
            name="x", init=None, var_type=IntType(), storage_class=StorageClass.Static
        )

    def test_extern_variable_declaration(self) -> None:
        """``extern int x;`` → ``VarDecl`` carrying ``StorageClass.Extern``."""
        tokens = [_EXTERN, _INT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.storage_class == StorageClass.Extern

    def test_variable_declaration_no_storage_class(self) -> None:
        """``int x;`` → ``VariableDeclaration`` with ``storage_class=None``."""
        tokens = [_INT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.storage_class is None

    def test_static_variable_with_initialiser(self) -> None:
        """``static int x = 5;`` → storage class and initialiser are both recorded."""
        tokens = [_STATIC, _INT, _ident("x"), _ASSIGNMENT, _number("5"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        vd = decl.variable_declaration
        assert vd.storage_class == StorageClass.Static
        assert vd.init == Constant(ConstInt(5))

    def test_static_function_forward_declaration(self) -> None:
        """``static int foo(void);`` → ``FunDecl`` whose ``Function`` carries ``StorageClass.Static``."""
        tokens = [_STATIC, _INT, _ident("foo"), _LPAREN, _VOID, _RPAREN, _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.storage_class == StorageClass.Static
        assert decl.function_declaration.body is None

    def test_extern_function_forward_declaration(self) -> None:
        """``extern int foo(void);`` → ``FunDecl`` carrying ``StorageClass.Extern``."""
        tokens = [_EXTERN, _INT, _ident("foo"), _LPAREN, _VOID, _RPAREN, _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.storage_class == StorageClass.Extern

    def test_static_function_definition(self) -> None:
        """``static int foo(void) { return 0; }`` → definition with ``StorageClass.Static``."""
        tokens = [
            _STATIC,
            _INT,
            _ident("foo"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        ]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.storage_class == StorageClass.Static
        assert decl.function_declaration.body is not None

    def test_specifiers_in_int_first_order(self) -> None:
        """``int static x;`` — specifiers in either order are accepted."""
        tokens = [_INT, _STATIC, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.storage_class == StorageClass.Static

    def test_static_local_variable(self) -> None:
        """``static int x;`` as a block item carries ``StorageClass.Static``."""
        body = _parsed_body(_STATIC, _INT, _ident("x"), _SEMICOLON)
        item = body[0]
        assert isinstance(item, D)
        assert isinstance(item.declaration, VarDecl)
        assert item.declaration.variable_declaration.storage_class == StorageClass.Static

    def test_conflicting_storage_class_raises_parse_error(self) -> None:
        """``static extern int x;`` — two storage-class specifiers raise ``ParseError``."""
        tokens = [_STATIC, _EXTERN, _INT, _ident("x"), _SEMICOLON]
        with pytest.raises(ParseError, match="conflicting storage class"):
            parse(tokens)

    def test_missing_type_specifier_raises_parse_error(self) -> None:
        """``static x;`` — absent type specifier raises ``ParseError``."""
        tokens = [_STATIC, _ident("x"), _SEMICOLON]
        with pytest.raises(ParseError, match="expected a type"):
            parse(tokens)


# ---------------------------------------------------------------------------
# Long type specifiers
# ---------------------------------------------------------------------------


class TestLongTypeSpecifiers:
    """Parser correctly handles ``long``, ``long int``, and ``int long`` specifiers."""

    def test_long_variable_declaration(self) -> None:
        """``long x;`` → ``VarDecl`` with ``var_type=LongType()``."""
        from test.parser.helpers import _LONG

        tokens = [_LONG, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration == VariableDeclaration(name="x", init=None, var_type=LongType())

    def test_long_int_variable_declaration(self) -> None:
        """``long int x;`` → ``VarDecl`` with ``var_type=LongType()`` — C standard combination."""
        from test.parser.helpers import _LONG

        tokens = [_LONG, _INT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == LongType()

    def test_int_long_variable_declaration(self) -> None:
        """``int long x;`` → ``VarDecl`` with ``var_type=LongType()`` — specifiers in either order."""
        from test.parser.helpers import _LONG

        tokens = [_INT, _LONG, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == LongType()

    def test_long_variable_with_storage_class(self) -> None:
        """``static long x;`` — storage class and ``long`` type are both recorded."""
        from test.parser.helpers import _LONG

        tokens = [_STATIC, _LONG, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        vd = decl.variable_declaration
        assert vd.var_type == LongType()
        assert vd.storage_class == StorageClass.Static

    def test_long_function_return_type(self) -> None:
        """``long foo(void);`` → forward declaration with ``LongType`` return."""
        from test.parser.helpers import _LONG

        tokens = [_LONG, _ident("foo"), _LPAREN, _VOID, _RPAREN, _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        fn = decl.function_declaration
        assert isinstance(fn.fun_type, FunType)
        assert fn.fun_type.ret == LongType()

    def test_long_parameter(self) -> None:
        """``int foo(long x) {}`` — parameter has ``param_type=LongType()``."""
        from test.parser.helpers import _LONG

        tokens = [_INT, _ident("foo"), _LPAREN, _LONG, _ident("x"), _RPAREN, _LBRACE, _RBRACE]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.params == (Parameter(param_type=LongType(), name="x"),)

    def test_long_int_parameter(self) -> None:
        """``int foo(long int x) {}`` — ``long int`` parameter resolves to ``LongType``."""
        from test.parser.helpers import _LONG

        tokens = [_INT, _ident("foo"), _LPAREN, _LONG, _INT, _ident("x"), _RPAREN, _LBRACE, _RBRACE]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.params[0].param_type == LongType()

    def test_duplicate_long_raises_parse_error(self) -> None:
        """``long long x;`` — repeated ``long`` specifier raises ``ParseError``."""
        from test.parser.helpers import _LONG

        tokens = [_LONG, _LONG, _ident("x"), _SEMICOLON]
        with pytest.raises(ParseError, match="duplicate type specifier"):
            parse(tokens)

    def test_duplicate_int_raises_parse_error(self) -> None:
        """``int int x;`` — repeated ``int`` specifier raises ``ParseError``."""
        tokens = [_INT, _INT, _ident("x"), _SEMICOLON]
        with pytest.raises(ParseError, match="duplicate type specifier"):
            parse(tokens)
