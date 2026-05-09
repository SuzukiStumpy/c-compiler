"""Parsing stage: converts a token stream into an abstract syntax tree.

The parser uses recursive descent for all constructs except binary expressions,
which are handled by precedence climbing.  The grammar is:

    program             ::= function
    function            ::= "int" identifier "(" "void" ")" "{" block_item* "}"
    block_item          ::= statement | declaration
    declaration         ::= "int" identifier [ "=" exp ] ";"
    statement           ::= "return" exp ";"
                          | exp ";"
                          | "if" "(" exp ")" statement ["else" statement]
                          | ";"
    exp                 ::= factor | exp binop exp | exp "=" exp | exp assignop exp | exp "?" exp ":" exp
    factor              ::= prefix_unop factor | primary postfix_unop?
    primary             ::= int | identifier | "(" exp ")"
    prefix_unop         ::= "-" | "~" | "!" | "++" | "--"
    postfix_unop        ::= "++" | "--"
    binop               ::= "+" | "-" | "*" | "/" | "%" | "<<" | ">>" | "&" | "^" | "|"
                          | "&&" | "||" | "==" | "!=" | "<" | "<=" | ">" | ">="
    assignop            ::= "+=" | "-=" | "*=" | "/=" | "%="
                          | "&=" | "|=" | "^=" | "<<=" | ">>="
    identifier          ::= ? An identifier token ?
    int                 ::= ? A constant token ?

Each grammar rule (except ``exp``) maps directly to a private ``_parse_*``
method on ``Parser``.  ``_parse_exp`` implements precedence climbing so that
operator associativity and precedence are handled without ambiguity; assignment
is right-associative at the lowest precedence level.
The public ``parse()`` function is the only entry point intended for use by
other pipeline stages.
"""

from pathlib import Path

from compiler.syntax.ast import (
    Add,
    And,
    Assignment,
    Binary,
    BinaryOperator,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BlockItem,
    Complement,
    Constant,
    D,
    Declaration,
    Divide,
    Equal,
    Exp,
    Expression,
    Function,
    GreaterOrEqual,
    GreaterThan,
    If,
    IntType,
    LeftShift,
    LessOrEqual,
    LessThan,
    Multiply,
    Negate,
    Not,
    NotEqual,
    Null,
    Or,
    Parameter,
    PostDecrement,
    PostIncrement,
    PreDecrement,
    PreIncrement,
    Program,
    Remainder,
    Return,
    RightShift,
    S,
    Statement,
    Subtract,
    Type,
    Unary,
    UnaryOperator,
    Var,
)
from compiler.syntax.ast.expressions import Conditional
from compiler.types.tokens import Token, TokenType
from compiler.utils.exceptions import ParseError

# Precedence table for binary operators used by the precedence-climbing parser.
# Higher values bind more tightly.  Operators absent from this table are not
# binary operators and will terminate the climbing loop.
_BINARY_PRECEDENCE: dict[TokenType, int] = {
    TokenType.ASSIGNMENT: 1,
    TokenType.PLUS_ASSIGN: 1,
    TokenType.MINUS_ASSIGN: 1,
    TokenType.MUL_ASSIGN: 1,
    TokenType.DIV_ASSIGN: 1,
    TokenType.MOD_ASSIGN: 1,
    TokenType.BITWISE_AND_ASSIGN: 1,
    TokenType.BITWISE_OR_ASSIGN: 1,
    TokenType.BITWISE_XOR_ASSIGN: 1,
    TokenType.LSHIFT_ASSIGN: 1,
    TokenType.RSHIFT_ASSIGN: 1,
    TokenType.QUERY: 3,
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

    def _consume_compound_assignment(self) -> BinaryOperator:
        """Consumes a compound assignment operator and returns the actual
        binary operator which it encodes.

        Returns:
            The correct binary operator to which the compound assignment refers

        Raises:
            ParserError: If the stream is already exhausted, or if an incorrect token type
                has been passed in.
        """
        token = self._consume()
        match token.type:
            case TokenType.PLUS_ASSIGN:
                return Add()
            case TokenType.MINUS_ASSIGN:
                return Subtract()
            case TokenType.MUL_ASSIGN:
                return Multiply()
            case TokenType.DIV_ASSIGN:
                return Divide()
            case TokenType.MOD_ASSIGN:
                return Remainder()
            case TokenType.BITWISE_AND_ASSIGN:
                return BitwiseAnd()
            case TokenType.BITWISE_OR_ASSIGN:
                return BitwiseOr()
            case TokenType.BITWISE_XOR_ASSIGN:
                return BitwiseXor()
            case TokenType.LSHIFT_ASSIGN:
                return LeftShift()
            case TokenType.RSHIFT_ASSIGN:
                return RightShift()
            case _:
                raise self._token_error("expected a compound assignment operator", token)

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

    def _match(self, token_type: TokenType) -> bool:
        """Return ``True`` if the next token's type matches ``token_type``.

        Does not consume the token. Returns ``False`` at end of stream.

        Args:
            token_type: The token type to test against.

        Returns:
            ``True`` when the next token has the given type, ``False`` otherwise.
        """
        token = self._peek()
        return token is not None and token.type == token_type

    def _match_compound_assignment(self) -> bool:
        """Return ``True`` if the next token's type is one of the compound assignment
        tokens.  Does not consume the token.  Returns ``False`` at end of stream.

        Args:
            token_type: The token type to test against.

        Returns:
            ``True`` when the next token has the given type, ``False`` otherwise.
        """
        token = self._peek()

        if token is None:
            return False

        match token.type:
            case (
                TokenType.PLUS_ASSIGN
                | TokenType.MINUS_ASSIGN
                | TokenType.MUL_ASSIGN
                | TokenType.DIV_ASSIGN
                | TokenType.MOD_ASSIGN
                | TokenType.BITWISE_AND_ASSIGN
                | TokenType.BITWISE_OR_ASSIGN
                | TokenType.BITWISE_XOR_ASSIGN
                | TokenType.LSHIFT_ASSIGN
                | TokenType.RSHIFT_ASSIGN
            ):
                return True
            case _:
                return False

    def _get_precedence(self) -> int | None:
        """Return the binary-operator precedence of the next token, or ``None``.

        Looks up the next token's type in ``_BINARY_PRECEDENCE``. Returns
        ``None`` if the stream is exhausted or the token is not a binary
        operator.

        Returns:
            An integer precedence level, or ``None``.
        """
        token = self._peek()
        if token is None:
            return None
        return _BINARY_PRECEDENCE.get(token.type)

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
                                           parameter* params, block_item* body)

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

    def _parse_body(self) -> tuple[BlockItem, ...]:
        """Parse the block-item sequence forming a function body.

        Consumes block items (statements and declarations interleaved) until a
        ``RBRACE`` is reached.

        Returns:
            A tuple of ``BlockItem`` nodes (possibly empty).

        Raises:
            ParseError: If the stream ends before ``}`` is found, or if any
                block item is malformed.
        """
        items: list[BlockItem] = []
        while True:
            token = self._peek()
            if token is None:
                raise self._eof_error("expected '}', got end of input")
            if token.type == TokenType.RBRACE:
                break
            items.append(self._parse_block_item())
        return tuple(items)

    def _parse_block_item(self) -> BlockItem:
        """Parse a single block item: either a declaration or a statement.

        A declaration begins with a type keyword (currently only ``int``);
        everything else is treated as a statement.

        Returns:
            A ``D`` node wrapping a ``Declaration``, or an ``S`` node wrapping
            a ``Statement``.

        Raises:
            ParseError: If the item is malformed.
        """
        token = self._peek()
        if token is not None and token.type == TokenType.INT:
            return D(declaration=self._parse_declaration())
        return S(statement=self._parse_statement())

    def _parse_declaration(self) -> Declaration:
        """Parse a variable declaration, with an optional initialiser.

        Grammar::

            declaration ::= "int" identifier [ "=" exp ] ";"

        Returns:
            A ``Declaration`` node.

        Raises:
            ParseError: If the identifier or semicolon is missing, or if the
                initialiser expression is malformed.
        """
        self._expect(TokenType.INT)
        name_token = self._expect(TokenType.IDENTIFIER)
        init: Exp | None = None
        if self._match(TokenType.ASSIGNMENT):
            self._consume()
            init = self._parse_exp()
        self._expect(TokenType.SEMICOLON)
        return Declaration(name=name_token.value, init=init)

    def _parse_statement(self) -> Statement:
        """Parse a single statement.

        Grammar::

            statement ::= "return" exp ";"
                        | exp ";"
                        | "if" "(" exp ")" statement ["else" statement]
                        | ";"

        Returns:
            A ``Statement`` node.

        Raises:
            ParseError: If the token does not begin a recognised statement.
        """
        token = self._peek()
        if token is None:
            raise self._eof_error("expected a statement, got end of input")

        match token.type:
            case TokenType.RETURN:
                return self._parse_return()
            case TokenType.SEMICOLON:
                self._consume()
                return Null()
            case TokenType.IF:
                self._consume()
                self._expect(TokenType.LPAREN)
                exp = self._parse_exp()
                self._expect(TokenType.RPAREN)
                then = self._parse_statement()
                else_ = None

                if self._match(TokenType.ELSE):
                    self._consume()
                    else_ = self._parse_statement()
                return If(exp, then, else_)
            case _:
                exp = self._parse_exp()
                self._expect(TokenType.SEMICOLON)
                return Expression(exp=exp)

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
            case TokenType.INCREMENT:
                return PreIncrement()
            case TokenType.DECREMENT:
                return PreDecrement()
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
        binary operator whose precedence is at least ``min_prec``. Binary
        operators recurse with ``precedence + 1`` for left-associativity;
        assignment (``=``) and ternary (``?:``) recurse with the same
        precedence level for right-associativity and produce ``Assignment``
        and ``Conditional`` nodes respectively, rather than ``Binary`` nodes.
        The true-branch of a ternary is parsed at precedence zero so that any
        operator (including assignment) is valid inside it.

        Args:
            min_prec: Minimum operator precedence to consume at this level.
                      Callers should omit this argument (defaults to ``0``).

        Returns:
            An ``Exp`` node.
        """
        left: Exp = self._parse_factor()
        prec = self._get_precedence()
        while prec is not None and prec >= min_prec:
            if self._match(TokenType.ASSIGNMENT):
                self._consume()
                right = self._parse_exp(prec)  # right-associative
                left = Assignment(left, right)
            elif self._match_compound_assignment():
                op = self._consume_compound_assignment()
                right = self._parse_exp(prec)
                left = Assignment(left, Binary(op, left, right))
            elif self._match(TokenType.QUERY):
                middle = self._parse_conditional_middle()
                right = self._parse_exp(prec)
                left = Conditional(left, middle, right)
            else:
                op = self._parse_binop()
                right = self._parse_exp(prec + 1)  # left-associative
                left = Binary(op, left, right)
            prec = self._get_precedence()
        return left

    def _parse_conditional_middle(self) -> Exp:
        """Parse the true-branch expression of a ternary ``?:`` expression.

        Consumes the ``?`` token (already confirmed by the caller via ``_match``)
        and the closing ``:`` separator, returning the expression between them.
        The middle is parsed at precedence zero so that any operator, including
        assignment, is valid as the true-branch.

        Returns:
            An ``Exp`` node representing the true-branch expression.

        Raises:
            ParseError: If the ``:`` separator is absent after the true-branch
                expression.
        """
        self._consume()  # consume the already-matched '?'
        exp = self._parse_exp()
        self._expect(TokenType.COLON)
        return exp

    def _parse_factor(self) -> Exp:
        """Parse a factor: a primary expression optionally wrapped in prefix/postfix operators.

        Prefix unary operators (``-``, ``~``, ``!``, ``++``, ``--``) are parsed
        recursively and returned immediately. Primary expressions (constants,
        identifiers, parenthesised expressions) are then checked for a trailing
        postfix ``++`` or ``--`` before returning.

        Grammar::

            factor   ::= prefix_unop factor | primary postfix_unop?
            primary  ::= int | identifier | "(" exp ")"
            prefix_unop  ::= "-" | "~" | "!" | "++" | "--"
            postfix_unop ::= "++" | "--"

        Returns:
            An ``Exp`` node.

        Raises:
            ParseError: If the token does not begin a recognised factor.
        """
        token = self._peek()

        if token is None:
            raise self._eof_error("Syntax error: expected expression, got end of input")

        match token.type:
            case TokenType.BITWISE_NOT | TokenType.MINUS | TokenType.NOT | TokenType.INCREMENT | TokenType.DECREMENT:
                return Unary(self._parse_unaryop(), self._parse_factor())
            case TokenType.NUMBER:
                primary: Exp = self._parse_constant()
            case TokenType.IDENTIFIER:
                self._consume()
                primary = Var(name=token.value)
            case TokenType.LPAREN:
                self._consume()
                primary = self._parse_exp()
                self._expect(TokenType.RPAREN)
            case _:
                raise self._token_error("Syntax error: Malformed expression.", token)

        while True:
            if self._match(TokenType.INCREMENT):
                self._consume()
                primary = Unary(PostIncrement(), primary)
            elif self._match(TokenType.DECREMENT):
                self._consume()
                primary = Unary(PostDecrement(), primary)
            else:
                return primary

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
