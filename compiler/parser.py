"""Parsing stage: converts a token stream into an abstract syntax tree.

The parser uses recursive descent for all constructs except binary expressions,
which are handled by precedence climbing.  The grammar is:

    program             ::= function
    function            ::= "int" identifier "(" "void" ")" "{" statement "}"
    statement           ::= "return" exp ";"
    exp                 ::= factor | exp binop exp
    factor              ::= int | unop factor | "(" exp ")"
    unop                ::= "-" | "~" | "!"
    binop               ::= "+" | "-" | "*" | "/" | "%" | "<<" | ">>" | "&" | "^" | "|"
                          | "&&" | "||" | "==" | "!=" | "<" | "<=" | ">" | ">="
    identifier          ::= ? An identifier token ?
    int                 ::= ? A constant token ?

Each grammar rule (except ``exp``) maps directly to a private ``_parse_*``
method on ``Parser``.  ``_parse_exp`` implements precedence climbing so that
operator associativity and precedence are handled without ambiguity.
The public ``parse()`` function is the only entry point intended for use by
other pipeline stages.
"""

from pathlib import Path

from compiler.syntax.ast import (
    Add,
    And,
    Binary,
    BinaryOperator,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Complement,
    Constant,
    Divide,
    Equal,
    Exp,
    Function,
    GreaterOrEqual,
    GreaterThan,
    IntType,
    LeftShift,
    LessOrEqual,
    LessThan,
    Multiply,
    Negate,
    Not,
    NotEqual,
    Or,
    Parameter,
    Program,
    Remainder,
    Return,
    RightShift,
    Statement,
    Subtract,
    Type,
    Unary,
    UnaryOperator,
)
from compiler.types.tokens import Token, TokenType
from compiler.utils.exceptions import ParseError

# Precedence table for binary operators used by the precedence-climbing parser.
# Higher values bind more tightly.  Operators absent from this table are not
# binary operators and will terminate the climbing loop.
_BINARY_PRECEDENCE: dict[TokenType, int] = {
    TokenType.OR: 5,
    TokenType.AND: 10,
    TokenType.PIPE: 15,
    TokenType.CARET: 20,
    TokenType.AMPERSAND: 25,
    TokenType.EQ: 30,
    TokenType.NEQ: 30,
    TokenType.LT: 35,
    TokenType.LEQ: 35,
    TokenType.GT: 35,
    TokenType.GEQ: 35,
    TokenType.PLUS: 45,
    TokenType.MINUS: 45,
    TokenType.LSHIFT: 40,
    TokenType.RSHIFT: 40,
    TokenType.TIMES: 50,
    TokenType.DIVIDE: 50,
    TokenType.PERCENT: 50,
}


class Parser:
    """Recursive descent parser for the supported subset of C.

    Consumes a token stream produced by the lexer and constructs an immutable
    AST rooted at a ``Program`` node. Raises ``ParseError`` on any token that
    does not conform to the grammar.

    Attributes:
        _tokens: The complete token stream to parse.
        _pos: Index of the next token to be consumed.
        _last: The most recently consumed token, used to anchor EOF errors.
    """

    def __init__(self, tokens: list[Token]) -> None:
        """Initialise the parser with a token stream.

        Args:
            tokens: Ordered list of tokens produced by the lexer.
        """
        self._tokens = tokens
        self._pos = 0
        self._last: Token | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _peek(self) -> Token | None:
        """Return the next token without consuming it, or ``None`` at EOF.

        Returns:
            The token at the current position, or ``None`` if the stream is
            exhausted.
        """
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self) -> Token:
        """Consume and return the next token.

        Returns:
            The token at the current position.

        Raises:
            ParseError: If the stream is already exhausted.
        """
        token = self._peek()
        if token is None:
            raise self._eof_error("unexpected end of input")
        self._last = token
        self._pos += 1
        return token

    def _expect(self, token_type: TokenType) -> Token:
        """Consume the next token, raising ``ParseError`` if its type does not match.

        Args:
            token_type: The required token type.

        Returns:
            The consumed token.

        Raises:
            ParseError: If the next token has a different type, or if the stream
                is exhausted.
        """
        token = self._peek()
        if token is None:
            raise self._eof_error(f"expected {token_type.name}, got end of input")
        if token.type != token_type:
            raise self._token_error(
                f"expected {token_type.name}, got {token.type.name} '{token.value}'",
                token,
            )
        return self._consume()

    def _at_end(self) -> bool:
        """Return ``True`` if all tokens have been consumed.

        Returns:
            ``True`` when the stream is exhausted, ``False`` otherwise.
        """
        return self._pos >= len(self._tokens)

    def _source_lines(self, filename: str) -> list[str]:
        """Read and return the lines of a source file, or ``[]`` on failure.

        Args:
            filename: Path to the original C source file.

        Returns:
            The file's lines as a list of strings, or an empty list if the
            file cannot be read (e.g. in tests where the path is synthetic).
        """
        try:
            return Path(filename).read_text().splitlines()
        except OSError:
            return []

    def _token_error(self, message: str, token: Token) -> ParseError:
        """Build a ``ParseError`` anchored to a specific token's position.

        Args:
            message: Human-readable description of what went wrong.
            token: The offending token supplying position and source text.

        Returns:
            A ``ParseError`` with position and source context from ``token``.
        """
        lines = self._source_lines(token.filename)
        return ParseError(message, token.filename, token.line, token.col, token.value, lines)

    def _eof_error(self, message: str) -> ParseError:
        """Build a ``ParseError`` anchored to the last seen token position.

        Used when EOF is encountered unexpectedly and no current token exists
        to provide location information.  ``token_value`` is set to ``''`` so
        the underline renders as a single ``^`` with no trailing tildes.

        Args:
            message: Human-readable description of the unexpected EOF.

        Returns:
            A ``ParseError`` with position taken from the last consumed token,
            or zeroed fields if no token was ever consumed.
        """
        if self._last is not None:
            lines = self._source_lines(self._last.filename)
            return ParseError(message, self._last.filename, self._last.line, self._last.col, "", lines)
        return ParseError(message, "<unknown>", 0, 0, "", [])

    # ------------------------------------------------------------------
    # Grammar rules
    # ------------------------------------------------------------------

    def parse_program(self) -> Program:
        """Parse the full token stream as a translation unit.

        Expects one or more function definitions followed by end of input.

        Returns:
            A ``Program`` node containing all parsed function definitions.

        Raises:
            ParseError: If no function definitions are found, or if any
                function definition is malformed.
        """
        functions: list[Function] = []
        while not self._at_end():
            functions.append(self._parse_function())
        if not functions:
            raise self._eof_error("expected a function definition, got empty input")
        return Program(functions=tuple(functions))

    def _parse_function(self) -> Function:
        """Parse a single function definition.

        Grammar::

            function_definition = Function(type return_type, identifier name,
                                           parameter* params, statement+ body)

        Returns:
            A ``Function`` node.

        Raises:
            ParseError: If any part of the function definition is malformed.
        """
        return_type = self._parse_type()
        name_token = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.LPAREN)
        params = self._parse_params()
        self._expect(TokenType.RPAREN)
        self._expect(TokenType.LBRACE)
        body = self._parse_body()
        self._expect(TokenType.RBRACE)
        return Function(
            return_type=return_type,
            name=name_token.value,
            params=params,
            body=body,
        )

    def _parse_params(self) -> tuple[Parameter, ...]:
        """Parse the parameter list between the parentheses of a function definition.

        Handles three forms:

        * ``()`` — empty parameter list.
        * ``(void)`` — explicit "no parameters" marker; treated as empty.
        * ``(type identifier)`` — a single typed parameter.

        Multiple parameters are not currently supported (no ``COMMA`` token
        exists in the grammar). If more than one parameter appears, the
        ``_expect(RPAREN)`` call in ``_parse_function`` will raise a
        ``ParseError``.

        Returns:
            A tuple of zero or one ``Parameter`` nodes.

        Raises:
            ParseError: If the parameter type is unrecognised.
        """
        token = self._peek()
        if token is None or token.type == TokenType.RPAREN:
            return ()
        if token.type == TokenType.VOID:
            self._consume()
            return ()
        param = self._parse_param()
        return (param,)

    def _parse_param(self) -> Parameter:
        """Parse a single ``type identifier`` parameter declaration.

        Returns:
            A ``Parameter`` node.

        Raises:
            ParseError: If the type or identifier is missing or malformed.
        """
        param_type = self._parse_type()
        name_token = self._expect(TokenType.IDENTIFIER)
        return Parameter(param_type=param_type, name=name_token.value)

    def _parse_body(self) -> tuple[Statement, ...]:
        """Parse the statement sequence forming a function body.

        Consumes statements until a ``RBRACE`` is reached. At least one
        statement is required by the grammar.

        Returns:
            A non-empty tuple of ``Statement`` nodes.

        Raises:
            ParseError: If the body contains no statements, or if any
                statement is malformed.
        """
        statements: list[Statement] = []
        while True:
            token = self._peek()
            if token is None:
                raise self._eof_error("expected '}', got end of input")
            if token.type == TokenType.RBRACE:
                break
            statements.append(self._parse_statement())
        if not statements:
            token = self._peek()
            assert token is not None
            raise self._token_error("function body must contain at least one statement", token)
        return tuple(statements)

    def _parse_statement(self) -> Statement:
        """Parse a single statement.

        Currently only ``return`` statements are supported.

        Returns:
            A ``Statement`` node.

        Raises:
            ParseError: If the token does not begin a recognised statement.
        """
        token = self._peek()
        if token is None:
            raise self._eof_error("expected a statement, got end of input")
        if token.type == TokenType.RETURN:
            return self._parse_return()
        raise self._token_error(f"expected a statement, got {token.type.name} '{token.value}'", token)

    def _parse_return(self) -> Return:
        """Parse a ``return <exp>;`` statement.

        Returns:
            A ``Return`` node wrapping the parsed expression.

        Raises:
            ParseError: If the expression or trailing semicolon is missing.
        """
        self._expect(TokenType.RETURN)
        exp = self._parse_exp()
        self._expect(TokenType.SEMICOLON)
        return Return(exp=exp)

    def _parse_unaryop(self) -> UnaryOperator:
        """Parse a unary operator token and return the correct AST node for it

        Returns:
            A ``UnaryOperator`` node.
        """
        token = self._consume()

        match token.type:
            case TokenType.BITWISE_NOT:
                return Complement()
            case TokenType.MINUS:
                return Negate()
            case TokenType.NOT:
                return Not()
            case _:
                raise self._token_error("Syntax Error: Expected a unary operator token.", token)

    def _parse_binop(self) -> BinaryOperator:
        """Consume the next token and return the corresponding binary operator node.

        Returns:
            A ``BinaryOperator`` node.

        Raises:
            ParseError: If the next token is not a recognised binary operator.
        """
        token = self._consume()
        match token.type:
            case TokenType.PLUS:
                return Add()
            case TokenType.MINUS:
                return Subtract()
            case TokenType.TIMES:
                return Multiply()
            case TokenType.DIVIDE:
                return Divide()
            case TokenType.PERCENT:
                return Remainder()
            case TokenType.LSHIFT:
                return LeftShift()
            case TokenType.RSHIFT:
                return RightShift()
            case TokenType.AMPERSAND:
                return BitwiseAnd()
            case TokenType.CARET:
                return BitwiseXor()
            case TokenType.PIPE:
                return BitwiseOr()
            case TokenType.AND:
                return And()
            case TokenType.OR:
                return Or()
            case TokenType.EQ:
                return Equal()
            case TokenType.NEQ:
                return NotEqual()
            case TokenType.LT:
                return LessThan()
            case TokenType.LEQ:
                return LessOrEqual()
            case TokenType.GT:
                return GreaterThan()
            case TokenType.GEQ:
                return GreaterOrEqual()
            case _:
                raise self._token_error("expected a binary operator", token)

    def _parse_exp(self, min_prec: int = 0) -> Exp:
        """Parse an expression using precedence climbing.

        Parses a left-hand ``factor``, then repeatedly folds in any following
        binary operator whose precedence is at least ``min_prec``, recursing
        with ``precedence + 1`` to enforce left-associativity.

        Args:
            min_prec: Minimum operator precedence to consume at this level.
                      Callers should omit this argument (defaults to ``0``).

        Returns:
            An ``Exp`` node.
        """
        left: Exp = self._parse_factor()
        prec = _BINARY_PRECEDENCE.get(self._peek().type) if self._peek() is not None else None
        while prec is not None and prec >= min_prec:
            op = self._parse_binop()
            right = self._parse_exp(prec + 1)
            left = Binary(op, left, right)
            prec = _BINARY_PRECEDENCE.get(self._peek().type) if self._peek() is not None else None
        return left

    def _parse_factor(self) -> Exp:
        """Parse a factor: a constant, a unary expression, or a parenthesised expression.

        Grammar::

            factor ::= int | unop factor | "(" exp ")"

        Returns:
            An ``Exp`` node.

        Raises:
            ParseError: If the token does not begin a recognised factor.
        """
        token = self._peek()

        if token is None:
            raise self._eof_error("Syntax error: expected expression, got end of input")

        match token.type:
            case TokenType.NUMBER:
                return self._parse_constant()
            case TokenType.BITWISE_NOT | TokenType.MINUS | TokenType.NOT:
                operator = self._parse_unaryop()
                inner_exp = self._parse_factor()
                return Unary(operator, inner_exp)
            case TokenType.LPAREN:
                self._consume()
                inner_exp = self._parse_exp()
                self._expect(TokenType.RPAREN)
                return inner_exp
            case _:
                raise self._token_error("Syntax error: Malformed expression.", token)

    def _parse_constant(self) -> Constant:
        """Parse an integer constant expression.

        Returns:
            A ``Constant`` node with the parsed integer value.

        Raises:
            ParseError: If the next token is not a ``NUMBER``.
        """
        token = self._expect(TokenType.NUMBER)
        return Constant(value=int(token.value))

    def _parse_type(self) -> Type:
        """Parse a C type annotation.

        Currently only ``int`` is supported as a type.

        Returns:
            A ``Type`` node.

        Raises:
            ParseError: If the next token is not a recognised type keyword.
        """
        token = self._peek()
        if token is None:
            raise self._eof_error("expected a type, got end of input")
        if token.type == TokenType.INT:
            self._consume()
            return IntType()
        raise self._token_error(f"expected a type, got {token.type.name} '{token.value}'", token)


def parse(token_stream: list[Token]) -> Program:
    """Convert a token stream into an abstract syntax tree.

    Constructs a ``Parser`` and drives it to completion, returning the
    ``Program`` node at the root of the AST.

    Args:
        token_stream: Ordered list of tokens produced by the lexer.

    Returns:
        The ``Program`` node representing the full translation unit.

    Raises:
        ParseError: If the token stream does not conform to the grammar.
    """
    return Parser(token_stream).parse_program()
