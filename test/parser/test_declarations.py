"""Parser tests for declarations and program-level constructs."""

import pytest

from compiler.parser import parse
from compiler.syntax.ast import (
    Array,
    Char,
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
    Pointer,
    Program,
    Return,
    S,
    SChar,
    SingleInit,
    StorageClass,
    String,
    UChar,
    Unary,
    VarDecl,
    VariableDeclaration,
)
from compiler.syntax.ast.constants import ConstFloat, ConstLongDouble, ConstLongLong, ConstULongLong
from compiler.syntax.ast.expressions import Binary, Var  # noqa: F401
from compiler.syntax.ast.operators import Add, PostIncrement
from compiler.syntax.ast.types import (
    BoolType,
    FloatType,
    Int128Type,
    LongDoubleType,
    LongLongType,
    ShortType,
    UInt128Type,
    ULongLongType,
    UShortType,
)
from compiler.types.tokens import TokenType
from compiler.utils.exceptions import ParseError
from test.parser.helpers import (
    _ASSIGNMENT,
    _ASTERISK,
    _AUTO,
    _CHAR,
    _COMMA,
    _CONST,
    _EXTERN,
    _INCREMENT,
    _INT,
    _LBRACE,
    _LBRACKET,
    _LONG,
    _LPAREN,
    _PLUS,
    _RBRACE,
    _RBRACKET,
    _REGISTER,
    _RETURN,
    _RPAREN,
    _SEMICOLON,
    _SIGNED,
    _STATIC,
    _UNSIGNED,
    _VOID,
    _ident,
    _number,
    _parsed_body,
    _parsed_exp,
    _string_tok,
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
                        source_file="test.c",
                        source_line=1,
                    ),
                ),
            )
        )
        assert result == expected

    def test_simple_function_empty_params(self) -> None:
        """``int main() { return 0; }`` - empty parens produce no parameters."""
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
        """``int foo(int x) { return x; }`` - tests that a single typed parameter
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
        """``int (void) { ... }`` - LPAREN where IDENTIFIER expected."""
        tokens = [_INT, _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        with pytest.raises(ParseError, match="expected IDENTIFIER"):
            parse(tokens)

    def test_missing_opening_paren(self) -> None:
        """``int main void) { ... }`` - parsed as variable declaration; fails at SEMICOLON."""
        tokens = [_INT, _ident("main"), _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        with pytest.raises(ParseError, match="expected SEMICOLON"):
            parse(tokens)

    def test_missing_closing_paren(self) -> None:
        """``int main(void { ... }`` - LBRACE where RPAREN expected."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        with pytest.raises(ParseError, match="expected RPAREN"):
            parse(tokens)

    def test_missing_opening_brace(self) -> None:
        """``int main(void) return 0; }`` - RETURN where LBRACE expected."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        with pytest.raises(ParseError, match="expected LBRACE"):
            parse(tokens)

    def test_empty_function_body(self) -> None:
        """``int main(void) {}`` - empty body is valid; produces an empty block-item tuple."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RBRACE]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.body == ()

    def test_missing_semicolon(self) -> None:
        """``return 0 }`` - missing semicolon after expression."""
        tokens = [_INT, _ident("main"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _RBRACE]
        with pytest.raises(ParseError, match="expected SEMICOLON"):
            parse(tokens)

    def test_non_number_expression(self) -> None:
        """``return );`` - RPAREN is not a valid expression start."""
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
        """``int foo(int a, int b) { return 0; }`` -> two ``Parameter`` nodes."""
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
        """``int foo(int a, int b, int c) { return 0; }`` -> three ``Parameter`` nodes."""
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
        """``int foo(int a, b)`` - missing type before second parameter raises ``ParseError``."""
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
        """``int foo(void);`` -> ``FunDecl`` wrapping ``Function(name="foo", params=(), body=None)``."""
        tokens = [_INT, _ident("foo"), _LPAREN, _VOID, _RPAREN, _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration == Function(
            name="foo",
            params=(),
            body=None,
            fun_type=FunType(params=(), ret=IntType()),
            source_file="test.c",
            source_line=1,
        )

    def test_forward_declaration_with_params(self) -> None:
        """``int add(int a, int b);`` -> forward declaration with two params and ``body=None``."""
        tokens = [_INT, _ident("add"), _LPAREN, _INT, _ident("a"), _COMMA, _INT, _ident("b"), _RPAREN, _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        fn = decl.function_declaration
        assert fn.name == "add"
        assert fn.body is None
        assert fn.params == (Parameter(param_type=IntType(), name="a"), Parameter(param_type=IntType(), name="b"))

    def test_forward_declaration_followed_by_definition(self) -> None:
        """``int foo(void); int foo(void) { return 0; }`` -> two entries in ``Program.declarations``."""
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
        """``return foo();`` -> ``FunctionCall(callee=Var("foo"), args=[])``."""
        result = _parsed_exp(_ident("foo"), _LPAREN, _RPAREN)
        assert result == FunctionCall(callee=Var("foo"), args=[])

    def test_call_one_arg(self) -> None:
        """``return foo(1);`` -> ``FunctionCall(callee=Var("foo"), args=[Constant(ConstInt(1))])``."""
        result = _parsed_exp(_ident("foo"), _LPAREN, _number("1"), _RPAREN)
        assert result == FunctionCall(callee=Var("foo"), args=[Constant(ConstInt(1))])

    def test_call_two_args(self) -> None:
        """``return foo(1, 2);`` -> two-element args list."""
        result = _parsed_exp(_ident("foo"), _LPAREN, _number("1"), _COMMA, _number("2"), _RPAREN)
        assert result == FunctionCall(callee=Var("foo"), args=[Constant(ConstInt(1)), Constant(ConstInt(2))])

    def test_call_three_args(self) -> None:
        """``return foo(1, 2, 3);`` -> three-element args list."""
        result = _parsed_exp(_ident("foo"), _LPAREN, _number("1"), _COMMA, _number("2"), _COMMA, _number("3"), _RPAREN)
        assert result == FunctionCall(
            callee=Var("foo"), args=[Constant(ConstInt(1)), Constant(ConstInt(2)), Constant(ConstInt(3))]
        )

    def test_call_with_expression_arg(self) -> None:
        """``return foo(a + 1);`` - binary expression as the sole argument."""
        result = _parsed_exp(_ident("foo"), _LPAREN, _ident("a"), _PLUS, _number("1"), _RPAREN)
        assert result == FunctionCall(callee=Var("foo"), args=[Binary(Add(), Var("a"), Constant(ConstInt(1)))])

    def test_call_result_used_in_binary(self) -> None:
        """``return foo() + 1;`` - function call as left operand of addition."""
        result = _parsed_exp(_ident("foo"), _LPAREN, _RPAREN, _PLUS, _number("1"))
        assert result == Binary(Add(), FunctionCall(callee=Var("foo"), args=[]), Constant(ConstInt(1)))

    def test_call_with_postfix_increment(self) -> None:
        """``return foo()++;`` - postfix ``++`` applied to the call result."""
        result = _parsed_exp(_ident("foo"), _LPAREN, _RPAREN, _INCREMENT)
        assert result == Unary(PostIncrement(), FunctionCall(callee=Var("foo"), args=[]))

    def test_call_missing_closing_paren_raises_parse_error(self) -> None:
        """``return foo(1;`` - missing ``)`` raises ``ParseError``."""
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
        """``return a;`` - an identifier not followed by ``(`` is a ``Var``, not a call."""
        result = _parsed_exp(_ident("a"))
        assert result == Var("a")


# ---------------------------------------------------------------------------
# Storage-class specifiers
# ---------------------------------------------------------------------------


class TestStorageClassSpecifiers:
    """Parser correctly attaches storage-class specifiers to function and variable declarations."""

    def test_static_variable_declaration(self) -> None:
        """``static int x;`` -> ``VarDecl`` whose ``VariableDeclaration`` carries ``StorageClass.Static``."""
        tokens = [_STATIC, _INT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration == VariableDeclaration(
            name="x", init=None, var_type=IntType(), storage_class=StorageClass.Static
        )

    def test_extern_variable_declaration(self) -> None:
        """``extern int x;`` -> ``VarDecl`` carrying ``StorageClass.Extern``."""
        tokens = [_EXTERN, _INT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.storage_class == StorageClass.Extern

    def test_variable_declaration_no_storage_class(self) -> None:
        """``int x;`` -> ``VariableDeclaration`` with ``storage_class=None``."""
        tokens = [_INT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.storage_class is None

    def test_static_variable_with_initialiser(self) -> None:
        """``static int x = 5;`` -> storage class and initialiser are both recorded."""
        tokens = [_STATIC, _INT, _ident("x"), _ASSIGNMENT, _number("5"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        vd = decl.variable_declaration
        assert vd.storage_class == StorageClass.Static
        assert vd.init == SingleInit(exp=Constant(ConstInt(5)))

    def test_static_function_forward_declaration(self) -> None:
        """``static int foo(void);`` -> ``FunDecl`` whose ``Function`` carries ``StorageClass.Static``."""
        tokens = [_STATIC, _INT, _ident("foo"), _LPAREN, _VOID, _RPAREN, _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.storage_class == StorageClass.Static
        assert decl.function_declaration.body is None

    def test_extern_function_forward_declaration(self) -> None:
        """``extern int foo(void);`` -> ``FunDecl`` carrying ``StorageClass.Extern``."""
        tokens = [_EXTERN, _INT, _ident("foo"), _LPAREN, _VOID, _RPAREN, _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.storage_class == StorageClass.Extern

    def test_static_function_definition(self) -> None:
        """``static int foo(void) { return 0; }`` -> definition with ``StorageClass.Static``."""
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
        """``int static x;`` - specifiers in either order are accepted."""
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
        """``static extern int x;`` - two storage-class specifiers raise ``ParseError``."""
        tokens = [_STATIC, _EXTERN, _INT, _ident("x"), _SEMICOLON]
        with pytest.raises(ParseError, match="conflicting storage class"):
            parse(tokens)

    def test_missing_type_specifier_raises_parse_error(self) -> None:
        """``static x;`` - absent type specifier raises ``ParseError``."""
        tokens = [_STATIC, _ident("x"), _SEMICOLON]
        with pytest.raises(ParseError, match="expected a type"):
            parse(tokens)

    def test_auto_variable_declaration_is_no_op(self) -> None:
        """``auto int x;`` -> ``VarDecl`` with ``storage_class=None`` (auto is ignored)."""
        tokens = [_AUTO, _INT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.storage_class is None

    def test_register_local_variable_is_no_op(self) -> None:
        """``register int x;`` inside a block -> ``VarDecl`` with ``storage_class=None``."""
        tokens = [
            _INT,
            _ident("main"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _LBRACE,
            _REGISTER,
            _INT,
            _ident("y"),
            _SEMICOLON,
            _RBRACE,
        ]
        result = parse(tokens)
        fn = result.declarations[0]
        assert isinstance(fn, FunDecl)
        body = fn.function_declaration.body
        assert body is not None
        item = body[0]
        assert isinstance(item, D)
        assert isinstance(item.declaration, VarDecl)
        assert item.declaration.variable_declaration.storage_class is None

    def test_auto_combined_with_type_qualifier(self) -> None:
        """``auto const int x;`` -> parsed without error, ``storage_class=None``."""
        tokens = [_AUTO, _CONST, _INT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.storage_class is None


# ---------------------------------------------------------------------------
# Long type specifiers
# ---------------------------------------------------------------------------


class TestLongTypeSpecifiers:
    """Parser correctly handles ``long``, ``long int``, and ``int long`` specifiers."""

    def test_long_variable_declaration(self) -> None:
        """``long x;`` -> ``VarDecl`` with ``var_type=LongType()``."""
        tokens = [_LONG, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration == VariableDeclaration(name="x", init=None, var_type=LongType())

    def test_long_int_variable_declaration(self) -> None:
        """``long int x;`` -> ``VarDecl`` with ``var_type=LongType()`` - C standard combination."""
        tokens = [_LONG, _INT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == LongType()

    def test_int_long_variable_declaration(self) -> None:
        """``int long x;`` -> ``VarDecl`` with ``var_type=LongType()`` - specifiers in either order."""
        tokens = [_INT, _LONG, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == LongType()

    def test_long_variable_with_storage_class(self) -> None:
        """``static long x;`` - storage class and ``long`` type are both recorded."""
        tokens = [_STATIC, _LONG, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        vd = decl.variable_declaration
        assert vd.var_type == LongType()
        assert vd.storage_class == StorageClass.Static

    def test_long_function_return_type(self) -> None:
        """``long foo(void);`` -> forward declaration with ``LongType`` return."""
        tokens = [_LONG, _ident("foo"), _LPAREN, _VOID, _RPAREN, _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        fn = decl.function_declaration
        assert isinstance(fn.fun_type, FunType)
        assert fn.fun_type.ret == LongType()

    def test_long_parameter(self) -> None:
        """``int foo(long x) {}`` - parameter has ``param_type=LongType()``."""
        tokens = [_INT, _ident("foo"), _LPAREN, _LONG, _ident("x"), _RPAREN, _LBRACE, _RBRACE]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.params == (Parameter(param_type=LongType(), name="x"),)

    def test_long_int_parameter(self) -> None:
        """``int foo(long int x) {}`` - ``long int`` parameter resolves to ``LongType``."""
        tokens = [_INT, _ident("foo"), _LPAREN, _LONG, _INT, _ident("x"), _RPAREN, _LBRACE, _RBRACE]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.params[0].param_type == LongType()

    def test_duplicate_int_raises_parse_error(self) -> None:
        """``int int x;`` - repeated ``int`` specifier raises ``ParseError``."""
        tokens = [_INT, _INT, _ident("x"), _SEMICOLON]
        with pytest.raises(ParseError, match="duplicate type specifier"):
            parse(tokens)


# ---------------------------------------------------------------------------
# Pointer declarators
# ---------------------------------------------------------------------------


class TestPointerVariableDeclarations:
    """Parser correctly handles pointer declarators in variable declarations."""

    def test_simple_pointer_declaration(self) -> None:
        """``int *p;`` -> ``VarDecl`` with ``var_type=Pointer(IntType())``."""
        tokens = [_INT, _ASTERISK, _ident("p"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration == VariableDeclaration(name="p", init=None, var_type=Pointer(IntType()))

    def test_parenthesised_pointer_declaration(self) -> None:
        """``int *(p);`` -> same as ``int *p;``."""
        tokens = [_INT, _ASTERISK, _LPAREN, _ident("p"), _RPAREN, _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == Pointer(IntType())
        assert decl.variable_declaration.name == "p"

    def test_double_pointer_declaration(self) -> None:
        """``int *(*(p));`` -> ``var_type=Pointer(Pointer(IntType()))``."""
        tokens = [
            _INT,
            _ASTERISK,
            _LPAREN,
            _ASTERISK,
            _LPAREN,
            _ident("p"),
            _RPAREN,
            _RPAREN,
            _SEMICOLON,
        ]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == Pointer(Pointer(IntType()))

    def test_pointer_declaration_with_initialiser(self) -> None:
        """``int *p = 0;`` - pointer variable with an integer initialiser."""
        tokens = [_INT, _ASTERISK, _ident("p"), _ASSIGNMENT, _number("0"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        vd = decl.variable_declaration
        assert vd.var_type == Pointer(IntType())
        assert vd.init == SingleInit(exp=Constant(ConstInt(0)))

    def test_pointer_with_storage_class(self) -> None:
        """``static int *p;`` - pointer with storage-class specifier."""
        tokens = [_STATIC, _INT, _ASTERISK, _ident("p"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        vd = decl.variable_declaration
        assert vd.var_type == Pointer(IntType())
        assert vd.storage_class == StorageClass.Static


class TestPointerFunctionDeclarations:
    """Parser correctly handles pointer declarators in function declarations."""

    def test_function_returning_pointer(self) -> None:
        """``int *foo(void);`` -> forward declaration returning ``Pointer(IntType())``."""
        tokens = [_INT, _ASTERISK, _ident("foo"), _LPAREN, _VOID, _RPAREN, _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        fn = decl.function_declaration
        assert fn.name == "foo"
        assert fn.body is None
        assert fn.fun_type == FunType(params=(), ret=Pointer(IntType()))

    def test_function_returning_pointer_with_body(self) -> None:
        """``int *foo(void) { return 0; }`` - definition returning pointer type."""
        tokens = [
            _INT,
            _ASTERISK,
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
        fn = decl.function_declaration
        assert fn.fun_type == FunType(params=(), ret=Pointer(IntType()))
        assert fn.body is not None

    def test_pointer_parameter(self) -> None:
        """``int foo(int *p)`` - parameter with pointer type."""
        tokens = [
            _INT,
            _ident("foo"),
            _LPAREN,
            _INT,
            _ASTERISK,
            _ident("p"),
            _RPAREN,
            _LBRACE,
            _RBRACE,
        ]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.params == (Parameter(param_type=Pointer(IntType()), name="p"),)

    def test_multiple_pointer_parameters(self) -> None:
        """``int foo(int *a, int *b)`` - two pointer parameters."""
        tokens = [
            _INT,
            _ident("foo"),
            _LPAREN,
            _INT,
            _ASTERISK,
            _ident("a"),
            _COMMA,
            _INT,
            _ASTERISK,
            _ident("b"),
            _RPAREN,
            _LBRACE,
            _RBRACE,
        ]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.params == (
            Parameter(param_type=Pointer(IntType()), name="a"),
            Parameter(param_type=Pointer(IntType()), name="b"),
        )

    def test_mixed_parameters(self) -> None:
        """``int foo(int x, int *p)`` - one plain and one pointer parameter."""
        tokens = [
            _INT,
            _ident("foo"),
            _LPAREN,
            _INT,
            _ident("x"),
            _COMMA,
            _INT,
            _ASTERISK,
            _ident("p"),
            _RPAREN,
            _LBRACE,
            _RBRACE,
        ]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.params == (
            Parameter(param_type=IntType(), name="x"),
            Parameter(param_type=Pointer(IntType()), name="p"),
        )


class TestFunctionPointerDeclarations:
    """Parser correctly builds function-pointer variable declarations."""

    def test_named_function_pointer_unnamed_params(self) -> None:
        """``int (*ptr)(int, int)`` -> VarDecl with Pointer(FunType) type."""
        tokens = [
            _INT,
            _LPAREN,
            _ASTERISK,
            _ident("ptr"),
            _RPAREN,
            _LPAREN,
            _INT,
            _COMMA,
            _INT,
            _RPAREN,
            _SEMICOLON,
        ]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        vd = decl.variable_declaration
        assert vd.name == "ptr"
        assert vd.var_type == Pointer(FunType(params=(IntType(), IntType()), ret=IntType()))

    def test_named_function_pointer_named_params(self) -> None:
        """``int (*ptr)(int a, int b)`` - param names are accepted but not required."""
        tokens = [
            _INT,
            _LPAREN,
            _ASTERISK,
            _ident("ptr"),
            _RPAREN,
            _LPAREN,
            _INT,
            _ident("a"),
            _COMMA,
            _INT,
            _ident("b"),
            _RPAREN,
            _SEMICOLON,
        ]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == Pointer(FunType(params=(IntType(), IntType()), ret=IntType()))

    def test_function_pointer_no_params(self) -> None:
        """``int (*ptr)(void)`` -> Pointer(FunType(params=(), ret=IntType()))."""
        tokens = [_INT, _LPAREN, _ASTERISK, _ident("ptr"), _RPAREN, _LPAREN, _VOID, _RPAREN, _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == Pointer(FunType(params=(), ret=IntType()))

    def test_function_pointer_with_pointer_return(self) -> None:
        """``int *(*ptr)(void)`` -> Pointer(FunType((), Pointer(IntType())))."""
        tokens = [
            _INT,
            _ASTERISK,
            _LPAREN,
            _ASTERISK,
            _ident("ptr"),
            _RPAREN,
            _LPAREN,
            _VOID,
            _RPAREN,
            _SEMICOLON,
        ]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == Pointer(FunType(params=(), ret=Pointer(IntType())))

    def test_function_pointer_with_initialiser(self) -> None:
        """``int (*ptr)(int, int) = add;`` - initialiser is parsed as a Var."""
        tokens = [
            _INT,
            _LPAREN,
            _ASTERISK,
            _ident("ptr"),
            _RPAREN,
            _LPAREN,
            _INT,
            _COMMA,
            _INT,
            _RPAREN,
            _ASSIGNMENT,
            _ident("add"),
            _SEMICOLON,
        ]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        vd = decl.variable_declaration
        assert vd.var_type == Pointer(FunType(params=(IntType(), IntType()), ret=IntType()))
        assert vd.init == SingleInit(exp=Var("add"))

    def test_function_pointer_param_with_pointer_param(self) -> None:
        """``int (*ptr)(int *)`` - function pointer whose param is itself a pointer."""
        tokens = [
            _INT,
            _LPAREN,
            _ASTERISK,
            _ident("ptr"),
            _RPAREN,
            _LPAREN,
            _INT,
            _ASTERISK,
            _RPAREN,
            _SEMICOLON,
        ]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == Pointer(FunType(params=(Pointer(IntType()),), ret=IntType()))

    def test_function_ptr_not_mistaken_for_function_decl(self) -> None:
        """``int (*ptr)(void)`` must produce VarDecl, not FunDecl."""
        tokens = [_INT, _LPAREN, _ASTERISK, _ident("ptr"), _RPAREN, _LPAREN, _VOID, _RPAREN, _SEMICOLON]
        result = parse(tokens)
        assert isinstance(result.declarations[0], VarDecl)


class TestPointerDeclaratorErrors:
    """Parser rejects invalid pointer-related declarator constructs."""

    def test_function_returning_function_raises_parse_error(self) -> None:
        """``int (foo(void))(void);`` - parenthesised function declarator followed by a
        second param-list would declare a function returning a function, which C forbids.
        """
        tokens = [
            _INT,
            _LPAREN,
            _ident("foo"),
            _LPAREN,
            _VOID,
            _RPAREN,
            _RPAREN,
            _LPAREN,
            _VOID,
            _RPAREN,
            _SEMICOLON,
        ]
        with pytest.raises(ParseError, match="function returning a function"):
            parse(tokens)


# ---------------------------------------------------------------------------
# char, signed char, and unsigned char type declarations
# ---------------------------------------------------------------------------


class TestCharTypeDeclarations:
    """Parser builds ``VariableDeclaration`` nodes with the correct char type variant."""

    def test_char_variable_declaration(self) -> None:
        """``char x;`` -> ``VariableDeclaration("x", None, Char())``."""
        body = _parsed_body(_CHAR, _ident("x"), _SEMICOLON)
        item = body[0]
        assert isinstance(item, D)
        assert isinstance(item.declaration, VarDecl)
        vd = item.declaration.variable_declaration
        assert vd.var_type == Char()
        assert vd.name == "x"
        assert vd.init is None

    def test_signed_char_variable_declaration(self) -> None:
        """``signed char x;`` -> ``VariableDeclaration("x", None, SChar())``."""
        body = _parsed_body(_SIGNED, _CHAR, _ident("x"), _SEMICOLON)
        item = body[0]
        assert isinstance(item, D)
        assert isinstance(item.declaration, VarDecl)
        assert item.declaration.variable_declaration.var_type == SChar()

    def test_char_signed_variable_declaration_order_independent(self) -> None:
        """``char signed x;`` produces the same ``SChar`` type as ``signed char x;``."""
        body = _parsed_body(_CHAR, _SIGNED, _ident("x"), _SEMICOLON)
        item = body[0]
        assert isinstance(item, D)
        assert isinstance(item.declaration, VarDecl)
        assert item.declaration.variable_declaration.var_type == SChar()

    def test_unsigned_char_variable_declaration(self) -> None:
        """``unsigned char x;`` -> ``VariableDeclaration("x", None, UChar())``."""
        body = _parsed_body(_UNSIGNED, _CHAR, _ident("x"), _SEMICOLON)
        item = body[0]
        assert isinstance(item, D)
        assert isinstance(item.declaration, VarDecl)
        assert item.declaration.variable_declaration.var_type == UChar()

    def test_char_array_declaration(self) -> None:
        """``char arr[5];`` -> array type ``Array(Char(), 5)``."""
        body = _parsed_body(_CHAR, _ident("arr"), _LBRACKET, _number("5"), _RBRACKET, _SEMICOLON)
        item = body[0]
        assert isinstance(item, D)
        assert isinstance(item.declaration, VarDecl)
        assert item.declaration.variable_declaration.var_type == Array(Char(), 5)

    def test_char_array_with_string_literal_initializer(self) -> None:
        """``char arr[6] = "hello";`` -> ``SingleInit(String(b"hello"))``."""
        body = _parsed_body(
            _CHAR,
            _ident("arr"),
            _LBRACKET,
            _number("6"),
            _RBRACKET,
            _ASSIGNMENT,
            _string_tok('"hello"'),
            _SEMICOLON,
        )
        item = body[0]
        assert isinstance(item, D)
        assert isinstance(item.declaration, VarDecl)
        vd = item.declaration.variable_declaration
        assert vd.var_type == Array(Char(), 6)
        assert vd.init == SingleInit(String(b"hello"))

    def test_char_pointer_with_string_literal_initializer(self) -> None:
        """``char *s = "hi";`` -> ``SingleInit(String(b"hi"))`` with pointer type."""
        body = _parsed_body(
            _CHAR,
            _tok(TokenType.ASTERISK, "*"),
            _ident("s"),
            _ASSIGNMENT,
            _string_tok('"hi"'),
            _SEMICOLON,
        )
        item = body[0]
        assert isinstance(item, D)
        assert isinstance(item.declaration, VarDecl)
        vd = item.declaration.variable_declaration
        assert vd.var_type == Pointer(Char())
        assert vd.init == SingleInit(String(b"hi"))


# ---------------------------------------------------------------------------
# 16-bit integer types (short / unsigned short)
# ---------------------------------------------------------------------------

_SHORT = _tok(TokenType.SHORT, "short")
_INT128 = _tok(TokenType.INT128, "__int128")


class TestShortTypes:
    """Parser recognises ``short`` and ``unsigned short`` type specifiers."""

    def test_short_variable_declaration(self) -> None:
        """``short x;`` at file scope produces a ``ShortType`` variable."""
        tokens = [_SHORT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == ShortType()

    def test_short_int_variable_declaration(self) -> None:
        """``short int x;`` is equivalent to ``short x;``."""
        tokens = [_SHORT, _INT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == ShortType()

    def test_signed_short_variable_declaration(self) -> None:
        """``signed short x;`` produces a ``ShortType`` variable."""
        tokens = [_SIGNED, _SHORT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == ShortType()

    def test_unsigned_short_variable_declaration(self) -> None:
        """``unsigned short x;`` produces a ``UShortType`` variable."""
        tokens = [_UNSIGNED, _SHORT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == UShortType()

    def test_unsigned_short_int_variable_declaration(self) -> None:
        """``unsigned short int x;`` produces a ``UShortType`` variable."""
        tokens = [_UNSIGNED, _SHORT, _INT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == UShortType()

    def test_short_return_type(self) -> None:
        """``short foo(void) { return 0; }`` parses with a ``ShortType`` return."""
        tokens = [_SHORT, _ident("foo"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.fun_type.ret == ShortType()

    def test_short_parameter(self) -> None:
        """``int foo(short x) { ... }`` has a ``ShortType`` parameter."""
        tokens = [
            _INT,
            _ident("foo"),
            _LPAREN,
            _SHORT,
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
        assert decl.function_declaration.params[0].param_type == ShortType()

    def test_short_long_combination_is_rejected(self) -> None:
        """``short long x;`` is a parse error."""
        tokens = [_SHORT, _tok(TokenType.LONG, "long"), _ident("x"), _SEMICOLON]
        with pytest.raises(ParseError):
            parse(tokens)


# ---------------------------------------------------------------------------
# 128-bit integer types (__int128 / unsigned __int128)
# ---------------------------------------------------------------------------


class TestInt128Types:
    """Parser recognises ``__int128`` and ``unsigned __int128`` type specifiers."""

    def test_int128_variable_declaration(self) -> None:
        """``__int128 x;`` at file scope produces an ``Int128Type`` variable."""
        tokens = [_INT128, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == Int128Type()

    def test_unsigned_int128_variable_declaration(self) -> None:
        """``unsigned __int128 x;`` produces a ``UInt128Type`` variable."""
        tokens = [_UNSIGNED, _INT128, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == UInt128Type()

    def test_signed_int128_variable_declaration(self) -> None:
        """``signed __int128 x;`` is equivalent to ``__int128 x;``."""
        tokens = [_SIGNED, _INT128, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == Int128Type()

    def test_int128_return_type(self) -> None:
        """``__int128 foo(void) { ... }`` parses with ``Int128Type`` return."""
        tokens = [
            _INT128,
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
        assert decl.function_declaration.fun_type.ret == Int128Type()

    def test_int128_combined_with_int_is_rejected(self) -> None:
        """``__int128 int x;`` is a parse error."""
        tokens = [_INT128, _INT, _ident("x"), _SEMICOLON]
        with pytest.raises(ParseError):
            parse(tokens)

    def test_int128_combined_with_long_is_rejected(self) -> None:
        """``__int128 long x;`` is a parse error."""
        tokens = [_INT128, _tok(TokenType.LONG, "long"), _ident("x"), _SEMICOLON]
        with pytest.raises(ParseError):
            parse(tokens)


# ---------------------------------------------------------------------------
# 64-bit long long types (long long / unsigned long long)
# ---------------------------------------------------------------------------


class TestLongLongTypes:
    """Parser recognises ``long long`` and ``unsigned long long`` type specifiers."""

    def test_long_long_variable_declaration(self) -> None:
        """``long long x;`` at file scope produces a ``LongLongType`` variable."""
        tokens = [_LONG, _LONG, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == LongLongType()

    def test_long_long_int_variable_declaration(self) -> None:
        """``long long int x;`` is equivalent to ``long long x;``."""
        tokens = [_LONG, _LONG, _INT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == LongLongType()

    def test_signed_long_long_variable_declaration(self) -> None:
        """``signed long long x;`` produces a ``LongLongType`` variable."""
        tokens = [_SIGNED, _LONG, _LONG, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == LongLongType()

    def test_unsigned_long_long_variable_declaration(self) -> None:
        """``unsigned long long x;`` produces a ``ULongLongType`` variable."""
        tokens = [_UNSIGNED, _LONG, _LONG, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == ULongLongType()

    def test_unsigned_long_long_int_variable_declaration(self) -> None:
        """``unsigned long long int x;`` produces a ``ULongLongType`` variable."""
        tokens = [_UNSIGNED, _LONG, _LONG, _INT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == ULongLongType()

    def test_long_long_return_type(self) -> None:
        """``long long foo(void) { ... }`` parses with ``LongLongType`` return."""
        tokens = [
            _LONG,
            _LONG,
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
        assert decl.function_declaration.fun_type.ret == LongLongType()

    def test_long_long_param_type(self) -> None:
        """``int foo(long long x) { ... }`` parses with ``LongLongType`` parameter."""
        tokens = [
            _INT,
            _ident("foo"),
            _LPAREN,
            _LONG,
            _LONG,
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
        assert decl.function_declaration.params[0].param_type == LongLongType()

    def test_three_longs_is_rejected(self) -> None:
        """``long long long x;`` is a parse error."""
        tokens = [_LONG, _LONG, _LONG, _ident("x"), _SEMICOLON]
        with pytest.raises(ParseError):
            parse(tokens)

    def test_ll_literal_produces_const_long_long(self) -> None:
        """``42LL`` parses as ``ConstLongLong(42)`` with ``LongLongType``."""
        ll_tok = _tok(TokenType.LONGLONG, "42LL")
        exp = _parsed_exp(ll_tok)
        assert isinstance(exp, Constant)
        assert exp.value == ConstLongLong(42)
        assert exp.exp_type == LongLongType()

    def test_ull_literal_produces_const_ulong_long(self) -> None:
        """``42ULL`` parses as ``ConstULongLong(42)`` with ``ULongLongType``."""
        ull_tok = _tok(TokenType.ULONGLONG, "42ULL")
        exp = _parsed_exp(ull_tok)
        assert isinstance(exp, Constant)
        assert exp.value == ConstULongLong(42)
        assert exp.exp_type == ULongLongType()


# ---------------------------------------------------------------------------
# float type specifier
# ---------------------------------------------------------------------------

_FLOAT = _tok(TokenType.FLOAT, "float")


class TestFloatType:
    """Parser recognises ``float`` type specifier and float-suffix literals."""

    def test_float_variable_declaration(self) -> None:
        """``float x;`` at file scope produces a ``FloatType`` variable."""
        tokens = [_FLOAT, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == FloatType()

    def test_float_return_type(self) -> None:
        """``float foo(void) { return 0; }`` parses with a ``FloatType`` return."""
        tokens = [_FLOAT, _ident("foo"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.fun_type.ret == FloatType()

    def test_float_parameter(self) -> None:
        """``int foo(float x) { ... }`` has a ``FloatType`` parameter."""
        tokens = [
            _INT,
            _ident("foo"),
            _LPAREN,
            _FLOAT,
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
        assert decl.function_declaration.params[0].param_type == FloatType()

    def test_float_literal_produces_const_float(self) -> None:
        """``1.5f`` parses as ``ConstFloat(1.5)`` with ``FloatType``."""
        fp_tok = _tok(TokenType.FPFLOAT, "1.5f")
        exp = _parsed_exp(fp_tok)
        assert isinstance(exp, Constant)
        assert exp.value == ConstFloat(1.5)
        assert exp.exp_type == FloatType()

    def test_float_exponent_literal_produces_const_float(self) -> None:
        """``1e5F`` parses as ``ConstFloat(1e5)`` with ``FloatType``."""
        fp_tok = _tok(TokenType.FPFLOAT, "1e5F")
        exp = _parsed_exp(fp_tok)
        assert isinstance(exp, Constant)
        assert exp.value == ConstFloat(1e5)
        assert exp.exp_type == FloatType()


# ---------------------------------------------------------------------------
# long double type specifier
# ---------------------------------------------------------------------------

_DOUBLE_TOK = _tok(TokenType.DOUBLE, "double")


class TestLongDoubleType:
    """Parser recognises ``long double`` type specifier and L-suffix literals."""

    def test_long_double_variable_declaration(self) -> None:
        """``long double x;`` at file scope produces a ``LongDoubleType`` variable."""
        tokens = [_LONG, _DOUBLE_TOK, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == LongDoubleType()

    def test_long_double_return_type(self) -> None:
        """``long double foo(void) { return 0; }`` parses with ``LongDoubleType`` return."""
        tokens = [
            _LONG,
            _DOUBLE_TOK,
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
        assert decl.function_declaration.fun_type.ret == LongDoubleType()

    def test_long_double_parameter(self) -> None:
        """``int foo(long double x) { ... }`` has a ``LongDoubleType`` parameter."""
        tokens = [
            _INT,
            _ident("foo"),
            _LPAREN,
            _LONG,
            _DOUBLE_TOK,
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
        assert decl.function_declaration.params[0].param_type == LongDoubleType()

    def test_long_double_literal_produces_const_long_double(self) -> None:
        """``1.5L`` parses as ``ConstLongDouble(1.5)`` with ``LongDoubleType``."""
        fp_tok = _tok(TokenType.FPLONGDOUBLE, "1.5L")
        exp = _parsed_exp(fp_tok)
        assert isinstance(exp, Constant)
        assert exp.value == ConstLongDouble(1.5)
        assert exp.exp_type == LongDoubleType()

    def test_long_double_exponent_literal_produces_const_long_double(self) -> None:
        """``1e5l`` parses as ``ConstLongDouble(1e5)`` with ``LongDoubleType``."""
        fp_tok = _tok(TokenType.FPLONGDOUBLE, "1e5l")
        exp = _parsed_exp(fp_tok)
        assert isinstance(exp, Constant)
        assert exp.value == ConstLongDouble(1e5)
        assert exp.exp_type == LongDoubleType()

    def test_double_combined_with_other_specifiers_raises(self) -> None:
        """``long double long`` is a parse error."""
        with pytest.raises(Exception):
            parse([_LONG, _DOUBLE_TOK, _LONG, _ident("x"), _SEMICOLON])


class TestVariadicDeclarations:
    """Parser recognises variadic parameter lists using the ``...`` ellipsis."""

    def test_variadic_forward_declaration_sets_variadic_flag(self) -> None:
        """``int foo(int x, ...)`` forward declaration has ``fun_type.variadic == True``."""
        _ELLIPSIS = _tok(TokenType.ELLIPSIS, "...")
        tokens = [_INT, _ident("foo"), _LPAREN, _INT, _ident("x"), _COMMA, _ELLIPSIS, _RPAREN, _SEMICOLON]
        prog = parse(tokens)
        decl = prog.declarations[0]
        assert isinstance(decl, FunDecl)
        fd = decl.function_declaration
        assert fd.fun_type.variadic is True

    def test_variadic_forward_declaration_preserves_named_params(self) -> None:
        """Named parameters before ``...`` are still captured."""
        _ELLIPSIS = _tok(TokenType.ELLIPSIS, "...")
        tokens = [_INT, _ident("foo"), _LPAREN, _INT, _ident("x"), _COMMA, _ELLIPSIS, _RPAREN, _SEMICOLON]
        prog = parse(tokens)
        decl = prog.declarations[0]
        assert isinstance(decl, FunDecl)
        fd = decl.function_declaration
        assert len(fd.params) == 1
        assert fd.params[0].name == "x"

    def test_non_variadic_declaration_has_variadic_false(self) -> None:
        """A normal function declaration has ``fun_type.variadic == False``."""
        tokens = [_INT, _ident("foo"), _LPAREN, _INT, _ident("x"), _RPAREN, _SEMICOLON]
        prog = parse(tokens)
        decl = prog.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.fun_type.variadic is False

    def test_variadic_definition_sets_variadic_flag(self) -> None:
        """A variadic function definition also has ``fun_type.variadic == True``."""
        _ELLIPSIS = _tok(TokenType.ELLIPSIS, "...")
        tokens = [
            _INT,
            _ident("foo"),
            _LPAREN,
            _INT,
            _ident("x"),
            _COMMA,
            _ELLIPSIS,
            _RPAREN,
            _LBRACE,
            _RETURN,
            _number("0"),
            _SEMICOLON,
            _RBRACE,
        ]
        prog = parse(tokens)
        decl = prog.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.fun_type.variadic is True

    def test_ellipsis_only_allowed_after_at_least_one_named_param(self) -> None:
        """``int foo(...)`` with no preceding named param raises ``ParseError``."""
        _ELLIPSIS = _tok(TokenType.ELLIPSIS, "...")
        tokens = [_INT, _ident("foo"), _LPAREN, _ELLIPSIS, _RPAREN, _SEMICOLON]
        with pytest.raises(ParseError):
            parse(tokens)


# ---------------------------------------------------------------------------
# _Bool type (BoolType)
# ---------------------------------------------------------------------------

_BOOL = _tok(TokenType.BOOL, "_Bool")


class TestBoolType:
    """Parser recognises the ``_Bool`` type specifier."""

    def test_bool_variable_declaration(self) -> None:
        """``_Bool x;`` at file scope produces a ``BoolType`` variable."""
        tokens = [_BOOL, _ident("x"), _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, VarDecl)
        assert decl.variable_declaration.var_type == BoolType()

    def test_bool_return_type(self) -> None:
        """``_Bool foo(void) { return 0; }`` parses with a ``BoolType`` return."""
        tokens = [_BOOL, _ident("foo"), _LPAREN, _VOID, _RPAREN, _LBRACE, _RETURN, _number("0"), _SEMICOLON, _RBRACE]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.fun_type.ret == BoolType()

    def test_bool_parameter_type(self) -> None:
        """``int foo(_Bool x);`` parses with a ``BoolType`` parameter."""
        tokens = [_INT, _ident("foo"), _LPAREN, _BOOL, _ident("x"), _RPAREN, _SEMICOLON]
        result = parse(tokens)
        decl = result.declarations[0]
        assert isinstance(decl, FunDecl)
        assert decl.function_declaration.fun_type.params == (BoolType(),)

    def test_bool_cannot_combine_with_int(self) -> None:
        """``_Bool int x;`` is a type-specifier conflict and raises ``ParseError``."""
        tokens = [_BOOL, _INT, _ident("x"), _SEMICOLON]
        with pytest.raises(ParseError):
            parse(tokens)

    def test_bool_cannot_combine_with_unsigned(self) -> None:
        """``unsigned _Bool x;`` is invalid and raises ``ParseError``."""
        tokens = [_UNSIGNED, _BOOL, _ident("x"), _SEMICOLON]
        with pytest.raises(ParseError):
            parse(tokens)
