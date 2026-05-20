"""Parsing stage: converts a token stream into an abstract syntax tree.

The parser uses recursive descent for all constructs except binary expressions,
which are handled by precedence climbing.  The grammar is:

    program              ::= {declaration}
    declaration          ::= variable-declaration | function-declaration
    function-declaration ::= {specifier}+ identifier "(" param-list ")" (block | ";")
    variable-declaration ::= {specifier}+ identifier ["=" exp] ";"
    param-list           ::= "void" | {type-specifier}+ identifier {"," {type-specifier}+ identifier}
    type-specifier       ::= "int" | "long" | "unsigned" | "signed" | "double"
    specifier            ::= type-specifier | "static" | "extern"
    block                ::= "{" {block_item} "}"
    block_item           ::= statement | declaration
    for_init             ::= variable-declaration | [exp] ";"
    statement            ::= "return" exp ";"
                           | exp ";"
                           | "if" "(" exp ")" statement ["else" statement]
                           | "goto" identifier ";"
                           | identifier ":" statement
                           | block
                           | "break" ";"
                           | "continue" ";"
                           | "while" "(" exp ")" statement
                           | "do" statement "while" "(" exp ")" ";"
                           | "for" "(" for_init [exp] ";" [exp] ")" statement
                           | "switch" "(" exp ")" statement
                           | "case" int ":" statement
                           | "default" ":" statement
                           | ";"
    exp                  ::= factor | exp binop exp | exp "=" exp | exp assignop exp | exp "?" exp ":" exp
    factor               ::= prefix_unop factor
                           | primary postfix_unop?
    primary              ::= const
                           | identifier
                           | "(" {type-specifier}+ ")" factor
                           | "(" exp ")"
                           | identifier "(" [argument-list] ")"
    argument-list        ::= exp {"," exp}
    prefix_unop          ::= "-" | "~" | "!" | "++" | "--"
    postfix_unop         ::= "++" | "--"
    binop                ::= "+" | "-" | "*" | "/" | "%" | "<<" | ">>" | "&" | "^" | "|"
                           | "&&" | "||" | "==" | "!=" | "<" | "<=" | ">" | ">="
    assignop             ::= "+=" | "-=" | "*=" | "/=" | "%="
                           | "&=" | "|=" | "^=" | "<<=" | ">>="
    const                ::= int | long | uint | ulong | double
    identifier           ::= ? An identifier token ?
    int                  ::= ? An int token ?
    long                 ::= ? An int or long token ?
    uint                 ::= ? An unsigned int token ?
    ulong                ::= ? An unsigned int or unsigned long token ?
    double               ::= ? A floating-point constant token ?

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
    Block,
    BlockItem,
    Break,
    Case,
    Cast,
    Complement,
    Compound,
    Conditional,
    Constant,
    ConstDouble,
    ConstInt,
    ConstLong,
    ConstUInt,
    ConstULong,
    Continue,
    D,
    Declaration,
    Default,
    Divide,
    DoubleType,
    DoWhile,
    Equal,
    Exp,
    Expression,
    For,
    ForInit,
    Function,
    FunctionCall,
    FunDecl,
    FunType,
    Goto,
    GreaterOrEqual,
    GreaterThan,
    If,
    InitDecl,
    InitExp,
    IntType,
    Label,
    LeftShift,
    LessOrEqual,
    LessThan,
    LongType,
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
    StorageClass,
    Subtract,
    Switch,
    Type,
    UIntType,
    ULongType,
    Unary,
    UnaryOperator,
    Var,
    VarDecl,
    VariableDeclaration,
    While,
)
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

# All token types that may appear as part of a C type specifier.
_TYPE_SPECIFIER_TOKENS: frozenset[TokenType] = frozenset(
    {
        TokenType.INT,
        TokenType.LONG,
        TokenType.SIGNED,
        TokenType.UNSIGNED,
        TokenType.DOUBLE,
    }
)

# Type specifiers plus storage-class keywords; the full set consumed by _parse_specifiers.
_DECLARATION_SPECIFIER_TOKENS: frozenset[TokenType] = _TYPE_SPECIFIER_TOKENS | frozenset(
    {
        TokenType.STATIC,
        TokenType.EXTERN,
    }
)


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

    def _peek(self, offset: int = 0) -> Token | None:
        """Return a token without consuming it, or ``None`` at EOF.

        Args:
            offset: How many positions ahead of the current position to look.
                    Defaults to ``0`` (the next token to be consumed).

        Returns:
            The token at ``current + offset``, or ``None`` if that position is
            beyond the end of the stream.
        """
        pos = self._pos + offset
        if pos < len(self._tokens):
            return self._tokens[pos]
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
    # Type-specifier helpers
    # ------------------------------------------------------------------

    def _consume_type_specifiers(self) -> tuple[frozenset[TokenType], Token | None]:
        """Consume a run of type-specifier tokens and return the set seen.

        Reads tokens from the stream as long as they belong to
        ``_TYPE_SPECIFIER_TOKENS``.  Raises immediately on a duplicate specifier.

        Returns:
            A ``(seen, last_tok)`` pair.  ``seen`` is the frozen set of token
            types consumed; ``last_tok`` is the final token consumed, used to
            anchor subsequent error messages.  ``last_tok`` is ``None`` only
            when ``seen`` is empty.

        Raises:
            ParseError: If the same specifier token appears more than once.
        """
        seen: set[TokenType] = set()
        last_tok: Token | None = None
        while (peek := self._peek()) is not None and peek.type in _TYPE_SPECIFIER_TOKENS:  # type: ignore[missing-attribute]
            tok = self._consume()
            if tok.type in seen:
                raise self._token_error(f"duplicate type specifier '{tok.value}'", tok)
            seen.add(tok.type)
            last_tok = tok
        return frozenset(seen), last_tok

    def _resolve_type(self, seen: frozenset[TokenType], last_tok: Token | None) -> Type:
        """Map a set of consumed type-specifier token types to a ``Type`` node.

        Args:
            seen: Frozen set of ``TokenType`` values collected by
                ``_consume_type_specifiers``.
            last_tok: The final token consumed from the stream, used as the
                error anchor when a conflict is detected.  Non-``None``
                whenever ``seen`` is non-empty.

        Returns:
            The ``Type`` node corresponding to the combination of specifiers.

        Raises:
            ParseError: If ``seen`` is empty, if ``double`` is combined with
                any other specifier, or if ``signed`` and ``unsigned`` appear
                together.
        """
        if not seen:
            tok = self._peek()
            if tok is None:
                raise self._eof_error("expected a type specifier, got end of input")
            raise self._token_error(f"expected a type, got {tok.type.name} '{tok.value}'", tok)
        if TokenType.DOUBLE in seen:
            if len(seen) > 1:
                assert last_tok is not None
                raise self._token_error("'double' cannot be combined with other type specifiers", last_tok)
            return DoubleType()
        assert last_tok is not None
        if TokenType.SIGNED in seen and TokenType.UNSIGNED in seen:
            raise self._token_error("invalid type specifier: can't have both 'signed' and 'unsigned'", last_tok)
        if TokenType.LONG in seen and TokenType.UNSIGNED in seen:
            return ULongType()
        if TokenType.UNSIGNED in seen:
            return UIntType()
        if TokenType.LONG in seen:
            return LongType()
        return IntType()

    # ------------------------------------------------------------------
    # Grammar rules
    # ------------------------------------------------------------------

    def parse_program(self) -> Program:
        """Parse the full token stream as a translation unit.

        Expects one or more top-level declarations (functions or variables)
        followed by end of input.

        Returns:
            A ``Program`` node containing all parsed top-level declarations.

        Raises:
            ParseError: If no declarations are found, or if any declaration is
                malformed.
        """
        declarations: list[Declaration] = []
        while not self._at_end():
            declarations.append(self._parse_declaration())
        if not declarations:
            raise self._eof_error("expected a declaration, got empty input")
        return Program(declarations=tuple(declarations))

    def _parse_specifiers(self) -> tuple[Type, StorageClass | None]:
        """Parse one or more declaration specifiers.

        Grammar::

            type-specifier ::= "int" | "long" | "signed" | "unsigned" | "double"
            specifier      ::= type-specifier | "static" | "extern"

        Consumes a run of specifier tokens in any order.  At least one type
        specifier must appear; ``unsigned`` or ``signed`` alone are treated as
        shorthand for ``unsigned int`` / ``signed int``; at most one
        storage-class specifier (``static`` or ``extern``) may appear.

        Returns:
            A ``(type, storage_class)`` pair where ``storage_class`` is
            ``None`` when no storage-class specifier was present.

        Raises:
            ParseError: If no type specifier is present, if a type specifier
                appears more than once, if both ``signed`` and ``unsigned``
                appear, if ``double`` is combined with another type specifier,
                or if two storage-class specifiers conflict.
        """
        type_seen: set[TokenType] = set()
        last_type_tok: Token | None = None
        storage: StorageClass | None = None

        while (peek := self._peek()) is not None and peek.type in _DECLARATION_SPECIFIER_TOKENS:  # type: ignore[missing-attribute]
            tok = self._consume()
            if tok.type in {TokenType.STATIC, TokenType.EXTERN}:
                if storage is not None:
                    raise self._token_error("conflicting storage class specifiers", tok)
                storage = StorageClass.Static if tok.type == TokenType.STATIC else StorageClass.Extern
            else:
                if tok.type in type_seen:
                    raise self._token_error(f"duplicate type specifier '{tok.value}'", tok)
                type_seen.add(tok.type)
                last_type_tok = tok

        return self._resolve_type(frozenset(type_seen), last_type_tok), storage

    def _parse_params(self) -> tuple[Parameter, ...]:
        """Parse the parameter list between the parentheses of a function declaration.

        Grammar::

            param-list ::= "void" | "int" identifier {"," "int" identifier}

        Handles three forms:

        * ``()`` — empty parameter list.
        * ``(void)`` — explicit "no parameters" marker; treated as empty.
        * ``("int" identifier {"," "int" identifier})`` — one or more typed parameters.

        Returns:
            A tuple of zero or more ``Parameter`` nodes.

        Raises:
            ParseError: If any parameter type or identifier is missing or malformed.
        """
        token = self._peek()
        if token is None or token.type == TokenType.RPAREN:
            return ()
        if token.type == TokenType.VOID:
            self._consume()
            return ()
        params: list[Parameter] = [self._parse_param()]
        while self._match(TokenType.COMMA):
            self._consume()
            params.append(self._parse_param())
        return tuple(params)

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

    def _parse_block(self) -> Block:
        """Parse a complete ``{ block_item* }`` block, including surrounding braces.

        Returns:
            A ``Block`` (i.e., ``tuple[BlockItem, ...]``) of zero or more items.

        Raises:
            ParseError: If the opening ``{`` is absent, the stream ends before
                ``}`` is found, or any block item is malformed.
        """
        self._expect(TokenType.LBRACE)

        items: list[BlockItem] = []
        while True:
            token = self._peek()
            if token is None:
                raise self._eof_error("expected '}', got end of input")
            if token.type == TokenType.RBRACE:
                break
            items.append(self._parse_block_item())

        # Eat the final RBRACE before we return
        self._consume()
        return tuple(items)

    def _parse_block_item(self) -> BlockItem:
        """Parse a single block item: either a declaration or a statement.

        A declaration begins with a declaration-specifier keyword; everything
        else is treated as a statement.

        Returns:
            A ``D`` node wrapping a ``VarDecl`` or ``FunDecl``, or an ``S``
            node wrapping a ``Statement``.

        Raises:
            ParseError: If the item is malformed.
        """
        token = self._peek()
        if token is not None and token.type in _DECLARATION_SPECIFIER_TOKENS:
            return D(declaration=self._parse_declaration())
        return S(statement=self._parse_statement())

    def _parse_declaration(self) -> Declaration:
        """Parse a variable or function declaration with optional storage class.

        Grammar::

            declaration          ::= variable-declaration | function-declaration
            function-declaration ::= {specifier}+ identifier "(" param-list ")" (block | ";")
            variable-declaration ::= {specifier}+ identifier ["=" exp] ";"
            specifier            ::= "int" | "static" | "extern"

        Parses the leading specifier list, then dispatches on the token after
        the identifier: ``(`` indicates a function declaration (returning a
        ``FunDecl``); anything else is a variable declaration (``VarDecl``).

        Returns:
            A ``VarDecl`` or ``FunDecl`` node.

        Raises:
            ParseError: If no type specifier is present, the identifier or
                terminator is missing, or the initialiser expression is malformed.
        """
        type_, storage = self._parse_specifiers()
        name_token = self._expect(TokenType.IDENTIFIER)
        if self._match(TokenType.LPAREN):
            self._consume()
            params = self._parse_params()
            self._expect(TokenType.RPAREN)
            if self._match(TokenType.SEMICOLON):
                self._consume()
                body: Block | None = None
            else:
                body = self._parse_block()
            return FunDecl(
                function_declaration=Function(
                    name=name_token.value,
                    params=params,
                    body=body,
                    fun_type=FunType(params=tuple(p.param_type for p in params), ret=type_),
                    storage_class=storage,
                )
            )
        init: Exp | None = None
        if self._match(TokenType.ASSIGNMENT):
            self._consume()
            init = self._parse_exp()
        self._expect(TokenType.SEMICOLON)
        return VarDecl(
            variable_declaration=VariableDeclaration(
                name=name_token.value, init=init, var_type=type_, storage_class=storage
            )
        )

    def _parse_optional_exp(self, terminator: TokenType) -> Exp | None:
        """Parse an optional expression and consume its terminating token.

        If the very next token already matches ``terminator``, that token is
        consumed immediately and ``None`` is returned.  Otherwise an expression
        is parsed, ``terminator`` is consumed, and the expression is returned.

        Args:
            terminator: The token type that ends this clause (e.g.
                ``TokenType.SEMICOLON`` or ``TokenType.RPAREN``).

        Returns:
            The parsed ``Exp`` node, or ``None`` when no expression was present.

        Raises:
            ParseError: If an expression is present but malformed, or if
                ``terminator`` is absent after the expression.
        """
        if self._match(terminator):
            self._consume()
            return None

        exp = self._parse_exp()
        self._expect(terminator)
        return exp

    def _parse_variable_declaration(self) -> VarDecl:
        """Parse a variable declaration with an optional initialiser.

        Grammar::

            variable-declaration ::= "int" identifier ["=" exp] ";"

        Storage-class specifiers (``static``, ``extern``) are parsed by
        ``_parse_specifiers`` and recorded on the resulting ``VariableDeclaration``
        so that the semantic analysis stage can reject them with a ``SemanticError``.
        A ``(`` after the identifier is rejected immediately because function
        declarations are not permitted in for-loop initialisers.

        Returns:
            A ``VarDecl`` node.

        Raises:
            ParseError: If a function declaration is attempted, if the identifier
                or semicolon is missing, or if the initialiser expression is malformed.
        """
        type_, storage = self._parse_specifiers()
        name_token = self._expect(TokenType.IDENTIFIER)
        if self._match(TokenType.LPAREN):
            raise self._token_error(
                "function declarations are not permitted in for-loop initialisers",
                name_token,
            )
        init: Exp | None = None
        if self._match(TokenType.ASSIGNMENT):
            self._consume()
            init = self._parse_exp()
        self._expect(TokenType.SEMICOLON)
        return VarDecl(
            variable_declaration=VariableDeclaration(
                name=name_token.value, init=init, var_type=type_, storage_class=storage
            )
        )

    def _parse_for_init(self) -> ForInit:
        """Parse the initialiser clause of a ``for`` loop.

        Grammar::

            for_init ::= variable-declaration | [exp] ";"

        Dispatches on the next token: any specifier keyword (``int``,
        ``static``, ``extern``) begins a variable declaration (which consumes
        its own trailing semicolon); storage-class specifiers on the
        ``VariableDeclaration`` are later rejected by semantic analysis.
        Function declarations are rejected immediately.  Anything else delegates
        to ``_parse_optional_exp`` for an expression-or-empty clause followed
        by ``;``.

        Returns:
            An ``InitDecl`` node wrapping the parsed ``VarDecl``, or an
            ``InitExp`` node whose ``exp`` field is ``None`` when the clause
            is absent.

        Raises:
            ParseError: If a function declaration is attempted, if the optional
                expression is malformed, or if the trailing ``;`` is missing.
        """
        if self._peek() is not None and self._peek().type in _DECLARATION_SPECIFIER_TOKENS:  # type:ignore[missing-attribute]
            return InitDecl(self._parse_variable_declaration())
        return InitExp(self._parse_optional_exp(TokenType.SEMICOLON))

    def _parse_statement(self) -> Statement:
        """Parse a single statement.

        Grammar::

            statement ::= "return" exp ";"
                        | exp ";"
                        | "if" "(" exp ")" statement ["else" statement]
                        | "goto" identifier ";"
                        | identifier ":" statement
                        | block
                        | "break" ";"
                        | "continue" ";"
                        | "while" "(" exp ")" statement
                        | "do" statement "while" "(" exp ")" ";"
                        | "for" "(" for_init [exp] ";" [exp] ")" statement
                        | "switch" "(" exp ")" statement
                        | "case" int ":" statement
                        | "default" ":" statement
                        | ";"

        A labeled statement is detected via one token of lookahead before the
        main dispatch, so identifiers not followed by ``:`` fall through to the
        expression-statement path in the catch-all branch.

        Returns:
            A ``Statement`` node.

        Raises:
            ParseError: If the token does not begin a recognised statement.
        """
        token = self._peek()
        if token is None:
            raise self._eof_error("expected a statement, got end of input")

        next_tok = self._peek(1)
        if token.type == TokenType.IDENTIFIER and next_tok is not None and next_tok.type == TokenType.COLON:
            name_token = self._consume()
            self._consume()  # consume ':'
            return Label(name=name_token.value, statement=self._parse_statement())

        match token.type:
            case TokenType.LBRACE:
                return Compound(body=self._parse_block())
            case TokenType.BREAK:
                self._consume()
                self._expect(TokenType.SEMICOLON)
                return Break()
            case TokenType.CONTINUE:
                self._consume()
                self._expect(TokenType.SEMICOLON)
                return Continue()
            case TokenType.WHILE:
                self._consume()
                self._expect(TokenType.LPAREN)
                test = self._parse_exp()
                self._expect(TokenType.RPAREN)
                body = self._parse_statement()
                return While(test, body)
            case TokenType.DO:
                self._consume()
                body = self._parse_statement()
                self._expect(TokenType.WHILE)
                self._expect(TokenType.LPAREN)
                test = self._parse_exp()
                self._expect(TokenType.RPAREN)
                self._expect(TokenType.SEMICOLON)
                return DoWhile(body, test)
            case TokenType.FOR:
                self._consume()
                self._expect(TokenType.LPAREN)
                init = self._parse_for_init()
                cond = self._parse_optional_exp(TokenType.SEMICOLON)
                post = self._parse_optional_exp(TokenType.RPAREN)
                body = self._parse_statement()
                return For(init, cond, post, body)
            case TokenType.SWITCH:
                self._consume()
                self._expect(TokenType.LPAREN)
                control = self._parse_exp()
                self._expect(TokenType.RPAREN)
                return Switch(control=control, body=self._parse_statement())
            case TokenType.CASE:
                self._consume()
                value = self._parse_constant()
                self._expect(TokenType.COLON)
                return Case(value=value, statement=self._parse_statement())
            case TokenType.DEFAULT:
                self._consume()
                self._expect(TokenType.COLON)
                return Default(statement=self._parse_statement())
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
            case TokenType.GOTO:
                self._consume()
                label = self._expect(TokenType.IDENTIFIER)
                self._expect(TokenType.SEMICOLON)
                return Goto(label.value)
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
        identifiers, function calls, parenthesised expressions) are then
        checked for a trailing postfix ``++`` or ``--`` before returning.

        Grammar::

            factor        ::= prefix_unop factor | primary postfix_unop?
            primary       ::= int | identifier | "(" exp ")"
                            | identifier "(" [argument-list] ")"
            argument-list ::= exp {"," exp}
            prefix_unop   ::= "-" | "~" | "!" | "++" | "--"
            postfix_unop  ::= "++" | "--"

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
            case TokenType.NUMBER | TokenType.LONGINT | TokenType.UINT | TokenType.ULONG | TokenType.FPDOUBLE:
                primary: Exp = self._parse_constant()
            case TokenType.IDENTIFIER:
                self._consume()
                if self._match(TokenType.LPAREN):
                    primary = self._parse_function_call(token.value)
                else:
                    primary = Var(name=token.value)
            case TokenType.LPAREN:
                next_tok = self._peek(1)
                if next_tok is not None and next_tok.type in _TYPE_SPECIFIER_TOKENS:
                    self._consume()  # consume "("
                    target_type = self._parse_type()
                    self._expect(TokenType.RPAREN)
                    primary = Cast(target_type=target_type, exp=self._parse_factor())
                else:
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

    def _parse_function_call(self, name: str) -> FunctionCall:
        """Parse a function call expression after the identifier has been consumed.

        Grammar::

            identifier "(" [argument-list] ")"
            argument-list ::= exp {"," exp}

        Consumes the opening ``(``, parses zero or more comma-separated
        argument expressions, and consumes the closing ``)``.

        Args:
            name: The function name, already consumed by the caller.

        Returns:
            A ``FunctionCall`` node.

        Raises:
            ParseError: If any argument expression is malformed or the closing
                ``)`` is absent.
        """
        self._consume()  # consume "("
        args: list[Exp] = []
        if not self._match(TokenType.RPAREN):
            args.append(self._parse_exp())
            while self._match(TokenType.COMMA):
                self._consume()
                args.append(self._parse_exp())
        self._expect(TokenType.RPAREN)
        return FunctionCall(name=name, args=args)

    def _parse_constant(self) -> Constant:
        """Parse an integer, long or double constant expression.

        Follows C11 §6.4.4.1:

        - An unsuffixed decimal constant has the type of the first of ``int``,
          ``long`` in which its value can be represented.
        - A constant with an ``l`` or ``L`` suffix is always ``long``.
        - A constant with a ``u`` or ``U`` suffix has the type of the first of
          ``unsigned int``, ``unsigned long`` in which its value can be
          represented.
        - A constant with both ``u``/``U`` and ``l``/``L`` suffixes is always
          ``unsigned long``.

        Returns:
            A ``Constant`` node wrapping a ``ConstInt``, ``ConstLong``,
            ``ConstUInt``, ``ConstULong``, or ``ConstDouble``.

        Raises:
            ParseError: If the next token is not a numeric constant, or if the
                value exceeds the range of the largest supported type.
        """
        token = self._peek()
        if token is None:
            raise self._eof_error("expected a constant, got end of input")
        if token.type == TokenType.FPDOUBLE:
            self._consume()
            return Constant(value=ConstDouble(token.value), exp_type=DoubleType())
        if token.type == TokenType.NUMBER:
            self._consume()
            val = int(token.value)
            if val <= 2_147_483_647:
                return Constant(value=ConstInt(val), exp_type=IntType())
            if val <= 9_223_372_036_854_775_807:
                return Constant(value=ConstLong(val), exp_type=LongType())
            raise self._token_error(
                f"integer constant {val} is too large for type 'long' (max 9223372036854775807)", token
            )
        if token.type == TokenType.LONGINT:
            self._consume()
            val = int(token.value.rstrip("lL"))
            if val > 9_223_372_036_854_775_807:
                raise self._token_error(
                    f"integer constant {val} is too large for type 'long' (max 9223372036854775807)", token
                )
            return Constant(value=ConstLong(val), exp_type=LongType())
        if token.type == TokenType.UINT:
            self._consume()
            val = int(token.value.rstrip("uU"))
            if val <= 4_294_967_295:
                return Constant(value=ConstUInt(val), exp_type=UIntType())
            if val <= 18_446_744_073_709_551_615:
                return Constant(value=ConstULong(val), exp_type=ULongType())
            raise self._token_error(
                f"integer constant {val} is too large for type 'unsigned long' (max 18446744073709551615)", token
            )
        if token.type == TokenType.ULONG:
            self._consume()
            val = int(token.value.rstrip("uUlL"))
            if val > 18_446_744_073_709_551_615:
                raise self._token_error(
                    f"integer constant {val} is too large for type 'unsigned long' (max 18446744073709551615)", token
                )
            return Constant(value=ConstULong(val), exp_type=ULongType())
        raise self._token_error(f"expected a constant, got {token.type.name} '{token.value}'", token)

    def _parse_type(self) -> Type:
        """Parse a C type annotation from one or more type-specifier tokens.

        Grammar::

            type-specifier ::= "int" | "long" | "signed" | "unsigned" | "double"

        Consumes a run of specifier tokens in any order.  ``unsigned`` or
        ``signed`` alone are treated as shorthand for ``unsigned int`` /
        ``signed int`` respectively.

        Returns:
            A ``Type`` node.

        Raises:
            ParseError: If no recognised type-specifier token is present, if a
                specifier appears more than once, or if ``double`` is combined
                with another type specifier.
        """
        seen, last_tok = self._consume_type_specifiers()
        return self._resolve_type(seen, last_tok)


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
