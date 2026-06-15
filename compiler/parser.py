"""Parsing stage: converts a token stream into an abstract syntax tree.

The parser uses recursive descent for all constructs except binary expressions,
which are handled by precedence climbing.  The grammar is:

    program              ::= {declaration}
    declaration          ::= variable-declaration | function-declaration | struct-declaration
    struct-declaration   ::= "struct" identifier ["{" member-declaration}+ "}"]
    member-declaration   ::= {type-specifier}+ declarator ";"
    function-declaration ::= {specifier}+ declarator "(" param-list ")" (block | ";")
    variable-declaration ::= {specifier}+ declarator ["=" initializer] ";"
    declarator           ::= "*" declarator | direct-declarator
    direct-declarator    ::= simple-declarator [declarator-suffix]
    declarator-suffix    ::= param-list | {"[" const "]"}+
    simple-declarator    ::= identifier | "(" declarator ")"
    initializer          ::= "{" initializer {"," initializer} [","] "}" | exp
    param-list           ::= "(" "void" ")" | "(" param {"," param} ")"
    param                ::= {type-specifier}+ declarator
                           | {type-specifier}+ abstract-declarator
                           | {type-specifier}+
    type-specifier       ::= "int" | "long" | "unsigned" | "signed" | "double" | "char" | "void" | "struct" identifier
    specifier            ::= type-specifier | "static" | "extern"
    block                ::= "{" {block_item} "}"
    block_item           ::= statement | declaration
    for_init             ::= variable-declaration | [exp] ";"
    statement            ::= "return" [exp] ";"
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
    exp                  ::= cast-exp | exp binop exp | exp "=" exp | exp assignop exp | exp "?" exp ":" exp
    cast-exp             ::= "(" type-name ")" cast-exp
                           | factor
    factor               ::= prefix_unop cast-exp
                           | "sizeof" unary-exp
                           | "sizeof" "(" type-name ")"
                           | primary postfix_unop?
    primary              ::= const
                           | identifier
                           | "(" {type-specifier}+ [abstract-declarator] ")" factor
                           | "(" exp ")"
                           | identifier "(" [argument-list] ")"
    type-name            ::= {type-specifier} + [abstract-declarator]
    postfix-exp          ::= primary {postfix-op}
    postfix-op           ::= "[" exp "]"
                           | "." identifier
                           | "->" identifier
    argument-list        ::= exp {"," exp}
    abstract-declarator  ::= "*" [abstract-declarator]
                           | direct-abstract-declarator
    direct-abstract-declarator ::= "(" abstract-declarator ")" {"[" const "]"}
                           | {"[" const "]"}+
    prefix_unop          ::= "-" | "~" | "!" | "++" | "--" | "*" | "&"
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

from collections.abc import Callable
from pathlib import Path

from compiler.syntax.ast import (
    Add,
    AddrOf,
    And,
    Array,
    Arrow,
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
    Char,
    Complement,
    Compound,
    CompoundAssignment,
    CompoundInit,
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
    Dereference,
    Divide,
    Dot,
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
    Initializer,
    IntType,
    Label,
    LeftShift,
    LessOrEqual,
    LessThan,
    LongType,
    MemberDeclaration,
    Multiply,
    Negate,
    Not,
    NotEqual,
    Null,
    Or,
    Parameter,
    Pointer,
    PostDecrement,
    PostIncrement,
    PreDecrement,
    PreIncrement,
    Program,
    Remainder,
    Return,
    RightShift,
    S,
    SChar,
    SingleInit,
    SizeOf,
    SizeOfT,
    Statement,
    StorageClass,
    String,
    StructDecl,
    StructDeclaration,
    Structure,
    Subscript,
    Subtract,
    Switch,
    Type,
    UChar,
    UInt128Type,
    UIntType,
    ULongType,
    Unary,
    UnaryOperator,
    Union,
    UnionDecl,
    UnionDeclaration,
    UShortType,
    Var,
    VarDecl,
    VariableDeclaration,
    VoidType,
    While,
)
from compiler.syntax.ast.types import Int128Type, ShortType
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
    TokenType.ASTERISK: 50,
    TokenType.DIVIDE: 50,
    TokenType.PERCENT: 50,
}

# All token types that may appear as part of a C type specifier.
# TokenType.STRUCT is included because "struct identifier" is a two-token type
# specifier; callers that look ahead for a type (cast, sizeof) rely on this set.
_TYPE_SPECIFIER_TOKENS: frozenset[TokenType] = frozenset(
    {
        TokenType.INT,
        TokenType.LONG,
        TokenType.SIGNED,
        TokenType.UNSIGNED,
        TokenType.DOUBLE,
        TokenType.CHAR,
        TokenType.SHORT,
        TokenType.INT128,
        TokenType.VOID,
        TokenType.STRUCT,
        TokenType.UNION,
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

    def _consume_type_specifiers(self) -> tuple[frozenset[TokenType], Token | None, str | None, str | None]:
        """Consume a run of type-specifier tokens and return the set seen.

        Reads tokens from the stream as long as they belong to
        ``_TYPE_SPECIFIER_TOKENS``.  Raises immediately on a duplicate specifier.
        When a ``struct`` or ``union`` keyword is encountered the following
        identifier token is consumed immediately as the tag; neither can be
        combined with any other type specifier.

        Returns:
            A ``(seen, last_tok, struct_tag, union_tag)`` 4-tuple.  ``seen`` is
            the frozen set of non-aggregate token types consumed; ``last_tok`` is
            the final token consumed, used to anchor error messages; ``struct_tag``
            is the struct tag name when a ``struct identifier`` pair was consumed
            (otherwise ``None``); ``union_tag`` is the union tag name when a
            ``union identifier`` pair was consumed (otherwise ``None``).

        Raises:
            ParseError: If the same specifier token appears more than once, if
                ``struct`` is combined with other type specifiers, or if the
                identifier following ``struct`` is absent.
        """
        seen: set[TokenType] = set()
        last_tok: Token | None = None
        struct_tag: str | None = None
        union_tag: str | None = None
        while (peek := self._peek()) is not None and peek.type in _TYPE_SPECIFIER_TOKENS:  # type: ignore[missing-attribute]
            if peek.type == TokenType.STRUCT:
                if seen:
                    raise self._token_error("'struct' cannot be combined with other type specifiers", peek)
                self._consume()  # consume 'struct'
                tag_token = self._expect(TokenType.IDENTIFIER)
                struct_tag = tag_token.value
                last_tok = tag_token
                break  # struct tag exhausts the type-specifier sequence
            if peek.type == TokenType.UNION:
                if seen:
                    raise self._token_error("'union' cannot be combined with other type specifiers", peek)
                self._consume()  # consume 'union'
                tag_token = self._expect(TokenType.IDENTIFIER)
                union_tag = tag_token.value
                last_tok = tag_token
                break  # union tag exhausts the type-specifier sequence
            tok = self._consume()
            if tok.type in seen:
                raise self._token_error(f"duplicate type specifier '{tok.value}'", tok)
            seen.add(tok.type)
            last_tok = tok
        return frozenset(seen), last_tok, struct_tag, union_tag

    def _resolve_type(  # noqa: C901 - function too complex
        self,
        seen: frozenset[TokenType],
        last_tok: Token | None,
        struct_tag: str | None = None,
        union_tag: str | None = None,
    ) -> Type:
        """Map a set of consumed type-specifier token types to a ``Type`` node.

        Args:
            seen: Frozen set of ``TokenType`` values collected by
                ``_consume_type_specifiers``.
            last_tok: The final token consumed from the stream, used as the
                error anchor when a conflict is detected.  Non-``None``
                whenever ``seen`` is non-empty or ``struct_tag`` / ``union_tag``
                is set.
            struct_tag: When a ``struct identifier`` pair was consumed this
                is the tag name; in that case ``seen`` must be empty.
            union_tag: When a ``union identifier`` pair was consumed this
                is the tag name; in that case ``seen`` must be empty.

        Returns:
            The ``Type`` node corresponding to the combination of specifiers.

        Raises:
            ParseError: If ``seen`` is empty and both ``struct_tag`` and
                ``union_tag`` are ``None``, if ``double`` is combined with any
                other specifier, if ``signed`` and ``unsigned`` appear together,
                or if ``char`` is combined with specifiers other than ``signed``
                / ``unsigned``.
        """
        if struct_tag is not None:
            return Structure(tag=struct_tag)
        if union_tag is not None:
            return Union(tag=union_tag)
        if not seen:
            tok = self._peek()
            if tok is None:
                raise self._eof_error("expected a type specifier, got end of input")
            raise self._token_error(f"expected a type, got {tok.type.name} '{tok.value}'", tok)
        if TokenType.VOID in seen:
            assert last_tok is not None
            if len(seen) > 1:
                raise self._token_error("'void' cannot be combined with other type specifiers", last_tok)
            return VoidType()
        if TokenType.DOUBLE in seen:
            if len(seen) > 1:
                assert last_tok is not None
                raise self._token_error("'double' cannot be combined with other type specifiers", last_tok)
            return DoubleType()
        assert last_tok is not None
        if TokenType.SIGNED in seen and TokenType.UNSIGNED in seen:
            raise self._token_error("invalid type specifier: can't have both 'signed' and 'unsigned'", last_tok)
        if TokenType.CHAR in seen:
            invalid = seen - {TokenType.CHAR, TokenType.SIGNED, TokenType.UNSIGNED}
            if invalid:
                raise self._token_error("'char' may only be combined with 'signed' or 'unsigned'", last_tok)
            if TokenType.UNSIGNED in seen:
                return UChar()
            if TokenType.SIGNED in seen:
                return SChar()
            return Char()
        if TokenType.SHORT in seen:
            invalid = seen - {TokenType.SHORT, TokenType.INT, TokenType.SIGNED, TokenType.UNSIGNED}
            if invalid:
                raise self._token_error("'short' may only be combined with 'int', 'signed', or 'unsigned'", last_tok)
            if TokenType.UNSIGNED in seen:
                return UShortType()
            return ShortType()
        if TokenType.INT128 in seen:
            invalid = seen - {TokenType.INT128, TokenType.SIGNED, TokenType.UNSIGNED}
            if invalid:
                raise self._token_error("'__int128' may only be combined with 'signed' or 'unsigned'", last_tok)
            if TokenType.UNSIGNED in seen:
                return UInt128Type()
            return Int128Type()
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
                              | "struct" identifier
            specifier      ::= type-specifier | "static" | "extern"

        Consumes a run of specifier tokens in any order.  At least one type
        specifier must appear; ``unsigned`` or ``signed`` alone are treated as
        shorthand for ``unsigned int`` / ``signed int``; at most one
        storage-class specifier (``static`` or ``extern``) may appear.
        When ``struct`` is encountered the following identifier is consumed
        immediately as the tag; ``struct`` cannot be combined with other type
        specifiers (though a storage-class specifier preceding it is fine,
        e.g. ``static struct Foo``).

        Returns:
            A ``(type, storage_class)`` pair where ``storage_class`` is
            ``None`` when no storage-class specifier was present.

        Raises:
            ParseError: If no type specifier is present, if a type specifier
                appears more than once, if both ``signed`` and ``unsigned``
                appear, if ``double`` is combined with another type specifier,
                if ``struct`` is combined with other type specifiers, or if two
                storage-class specifiers conflict.
        """
        type_seen: set[TokenType] = set()
        last_type_tok: Token | None = None
        storage: StorageClass | None = None
        struct_tag: str | None = None
        union_tag: str | None = None

        while (peek := self._peek()) is not None and peek.type in _DECLARATION_SPECIFIER_TOKENS:  # type: ignore[missing-attribute]
            if peek.type == TokenType.STRUCT:
                if type_seen:
                    raise self._token_error("'struct' cannot be combined with other type specifiers", peek)
                self._consume()  # consume 'struct'
                tag_token = self._expect(TokenType.IDENTIFIER)
                struct_tag = tag_token.value
                last_type_tok = tag_token
                break  # struct tag exhausts the type-specifier sequence
            if peek.type == TokenType.UNION:
                if type_seen:
                    raise self._token_error("'union' cannot be combined with other type specifiers", peek)
                self._consume()  # consume 'union'
                tag_token = self._expect(TokenType.IDENTIFIER)
                union_tag = tag_token.value
                last_type_tok = tag_token
                break  # union tag exhausts the type-specifier sequence
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

        return self._resolve_type(frozenset(type_seen), last_type_tok, struct_tag, union_tag), storage

    def _parse_params(self) -> tuple[Parameter, ...]:
        """Parse the parameter list between the parentheses of a function declaration.

        Grammar::

            param-list ::= "void" | "int" identifier {"," "int" identifier}

        Handles three forms:

        * ``()`` - empty parameter list.
        * ``(void)`` - explicit "no parameters" marker; treated as empty.
        * ``("int" identifier {"," "int" identifier})`` - one or more typed parameters.

        Returns:
            A tuple of zero or more ``Parameter`` nodes.

        Raises:
            ParseError: If any parameter type or identifier is missing or malformed.
        """
        token = self._peek()
        if token is None or token.type == TokenType.RPAREN:
            return ()
        if token.type == TokenType.VOID and self._peek(1) is not None and self._peek(1).type == TokenType.RPAREN:  # type: ignore[union-attr]
            self._consume()
            return ()
        params: list[Parameter] = [self._parse_param()]
        while self._match(TokenType.COMMA):
            self._consume()
            params.append(self._parse_param())
        return tuple(params)

    def _parse_param(self) -> Parameter:
        """Parse a single typed parameter declaration.

        Grammar::

            param ::= {type-specifier}+ declarator
                    | {type-specifier}+ abstract-declarator
                    | {type-specifier}+

        The identifier is optional: unnamed parameters (e.g. ``int`` or ``int *``
        in a function-pointer type) are represented with an empty-string name.

        Returns:
            A ``Parameter`` node.

        Raises:
            ParseError: If the type is missing or the declarator is malformed.
        """
        base_type = self._parse_type()
        name, transform, _params = self._parse_param_declarator()
        return Parameter(param_type=transform(base_type), name=name or "")

    def _parse_param_simple_declarator(self) -> tuple[str | None, Callable[[Type], Type], tuple[Parameter, ...] | None]:
        """Parse a simple-declarator inside a parameter list (identifier is optional).

        Grammar::

            simple-declarator ::= identifier | "(" declarator ")"

        Mirrors ``_parse_simple_declarator`` but returns ``None`` for the name when
        no identifier is present, supporting abstract/unnamed parameters.

        Returns:
            A ``(name, type_builder, params)`` triple where ``name`` may be ``None``
            for an unnamed parameter.

        Raises:
            ParseError: If a parenthesised declarator is opened but not closed.
        """
        if self._match(TokenType.LPAREN):
            self._consume()
            name, transform, params = self._parse_param_declarator()
            self._expect(TokenType.RPAREN)
            return name, transform, params
        if self._match(TokenType.IDENTIFIER):
            name_token = self._consume()
            return name_token.value, lambda t: t, None
        return None, lambda t: t, None

    def _parse_param_direct_declarator(
        self,
    ) -> tuple[str | None, Callable[[Type], Type], tuple[Parameter, ...] | None]:
        """Parse a direct-declarator inside a parameter list (identifier is optional).

        Grammar::

            direct-declarator ::= simple-declarator [declarator-suffix]
            declarator-suffix  ::= param-list | {"[" const "]"}+

        Mirrors ``_parse_direct_declarator`` with optional-name semantics.

        Returns:
            A ``(name, type_builder, params)`` triple.

        Raises:
            ParseError: If a function declarator is nested inside another function
                declarator, or the structure is otherwise malformed.
        """
        name, simple_transform, simple_params = self._parse_param_simple_declarator()

        if self._match(TokenType.LBRACKET):
            sizes: list[int] = []
            while self._match(TokenType.LBRACKET):
                self._consume()  # consume "["
                sizes.append(self._parse_array_dimension())
                self._expect(TokenType.RBRACKET)

            def _arr(t: Type, _st: Callable[[Type], Type] = simple_transform, _sizes: list[int] = sizes) -> Type:
                for size in reversed(_sizes):
                    t = Array(t, size)
                return _st(t)

            return name, _arr, None

        if not self._match(TokenType.LPAREN):
            return name, simple_transform, simple_params

        if simple_params is not None:
            tok = self._peek()
            display = name or "<unnamed>"
            msg = f"'{display}' declared as function returning a function"
            raise self._token_error(msg, tok) if tok is not None else self._eof_error(msg)

        if isinstance(simple_transform(FunType(params=(), ret=IntType())), Array):
            tok = self._peek()
            display = name or "<unnamed>"
            msg = f"'{display}' declared as array of functions"
            raise self._token_error(msg, tok) if tok is not None else self._eof_error(msg)

        self._consume()  # consume "("
        params = self._parse_params()
        self._expect(TokenType.RPAREN)
        param_types = tuple(p.param_type for p in params)

        def _fun(t: Type, _st: Callable[[Type], Type] = simple_transform, _pt: tuple[Type, ...] = param_types) -> Type:
            return _st(FunType(params=_pt, ret=t))

        return name, _fun, params

    def _parse_param_declarator(self) -> tuple[str | None, Callable[[Type], Type], tuple[Parameter, ...] | None]:
        """Parse a declarator inside a parameter list (identifier is optional).

        Grammar::

            declarator        ::= "*" declarator | direct-declarator
            direct-declarator ::= simple-declarator [param-list]
            simple-declarator ::= identifier | "(" declarator ")"

        Mirrors ``_parse_declarator`` with optional-name semantics, supporting
        unnamed parameters such as ``int``, ``int *``, or ``int (*)(int)``.

        Returns:
            A ``(name, type_builder, params)`` triple where ``name`` is ``None``
            when no identifier was present.

        Raises:
            ParseError: If the declarator is malformed.
        """
        if self._match(TokenType.ASTERISK):
            self._consume()
            inner_name, inner_transform, params = self._parse_param_declarator()

            def _ptr(t: Type, _inner: Callable[[Type], Type] = inner_transform) -> Type:
                return _inner(Pointer(t))

            return inner_name, _ptr, params
        return self._parse_param_direct_declarator()

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
        """Parse a variable, function, or struct declaration.

        Grammar::

            declaration          ::= variable-declaration | function-declaration
                                   | struct-declaration
            struct-declaration   ::= "struct" identifier ["{" member-declaration}+ "}"] ";"
            function-declaration ::= {specifier}+ declarator (block | ";")
            variable-declaration ::= {specifier}+ declarator ["=" initializer] ";"
            declarator           ::= "*" declarator | direct-declarator
            direct-declarator    ::= simple-declarator [declarator-suffix]
            declarator-suffix    ::= param-list | {"[" const "]"}+
            simple-declarator    ::= identifier | "(" declarator ")"
            specifier            ::= type-specifier | "static" | "extern"

        A struct declaration is distinguished from a variable/function declaration
        that uses a struct type by one token of extra lookahead: ``struct Tag ;``
        and ``struct Tag {`` are struct declarations; ``struct Tag *``, ``struct Tag
        identifier``, and similar are variable/function declarations.

        Returns:
            A ``VarDecl``, ``FunDecl``, or ``StructDecl`` node.

        Raises:
            ParseError: If no type specifier is present, the declarator is malformed,
                or the initialiser expression or function body is malformed.
        """
        # Two-token lookahead to distinguish "struct/union Tag { ... };" / "struct/union Tag ;"
        # (a StructDecl/UnionDecl) from "struct/union Tag varname;" (a VarDecl).
        if self._match(TokenType.STRUCT):
            next1 = self._peek(1)
            next2 = self._peek(2)
            if (
                next1 is not None
                and next1.type == TokenType.IDENTIFIER
                and next2 is not None
                and next2.type in {TokenType.LBRACE, TokenType.SEMICOLON}
            ):
                return self._parse_struct_declaration()
        if self._match(TokenType.UNION):
            next1 = self._peek(1)
            next2 = self._peek(2)
            if (
                next1 is not None
                and next1.type == TokenType.IDENTIFIER
                and next2 is not None
                and next2.type in {TokenType.LBRACE, TokenType.SEMICOLON}
            ):
                return self._parse_union_declaration()

        first_token = self._peek()
        base_type, storage = self._parse_specifiers()
        name, transform, params = self._parse_declarator()
        full_type = transform(base_type)

        # A FunType result means the declared entity is itself a function.
        # A Pointer(FunType(...)) - or any other non-FunType - means a variable
        # (possibly of function-pointer type), even if params was non-None.
        if isinstance(full_type, FunType):
            if self._match(TokenType.SEMICOLON):
                self._consume()
                body: Block | None = None
            else:
                body = self._parse_block()
            assert params is not None
            return FunDecl(
                function_declaration=Function(
                    name=name,
                    params=params,
                    body=body,
                    fun_type=full_type,
                    storage_class=storage,
                    source_file=first_token.filename if first_token is not None else "",
                    source_line=first_token.line if first_token is not None else 0,
                )
            )
        init: Initializer | None = None
        if self._match(TokenType.ASSIGNMENT):
            self._consume()
            init = self._parse_initializer()
        self._expect(TokenType.SEMICOLON)
        return VarDecl(
            variable_declaration=VariableDeclaration(name=name, init=init, var_type=full_type, storage_class=storage)
        )

    def _parse_struct_declaration(self) -> StructDecl:
        """Parse a struct type declaration or definition.

        Grammar::

            struct-declaration ::= "struct" identifier ["{" member-declaration}+ "}"] ";"

        The member list is optional (a bare ``struct Tag;`` is a forward
        declaration), but when braces are present at least one member must appear
        inside them.

        Returns:
            A ``StructDecl`` node wrapping the ``StructDeclaration``.

        Raises:
            ParseError: If the tag identifier is missing, if the brace-enclosed
                member list is empty, if any member declaration is malformed, or
                if the trailing semicolon is absent.
        """
        self._expect(TokenType.STRUCT)
        tag_token = self._expect(TokenType.IDENTIFIER)
        members: tuple[MemberDeclaration, ...] = ()
        if self._match(TokenType.LBRACE):
            self._consume()
            if self._match(TokenType.RBRACE):
                tok = self._peek()
                msg = "struct definition must contain at least one member"
                raise self._token_error(msg, tok) if tok is not None else self._eof_error(msg)
            members_list: list[MemberDeclaration] = []
            while not self._match(TokenType.RBRACE):
                if self._peek() is None:
                    raise self._eof_error("expected '}', got end of input")
                members_list.append(self._parse_member_declaration())
            self._consume()  # consume '}'
            members = tuple(members_list)
        self._expect(TokenType.SEMICOLON)
        return StructDecl(struct_declaration=StructDeclaration(tag=tag_token.value, members=members))

    def _parse_union_declaration(self) -> UnionDecl:
        """Parse a union type declaration or definition.

        Grammar::

            union-declaration ::= "union" identifier ["{" member-declaration}+ "}"] ";"

        The member list is optional (a bare ``union Tag;`` is a forward
        declaration), but when braces are present at least one member must appear
        inside them.

        Returns:
            A ``UnionDecl`` node wrapping the ``UnionDeclaration``.

        Raises:
            ParseError: If the tag identifier is missing, if the brace-enclosed
                member list is empty, if any member declaration is malformed, or
                if the trailing semicolon is absent.
        """
        self._expect(TokenType.UNION)
        tag_token = self._expect(TokenType.IDENTIFIER)
        members: tuple[MemberDeclaration, ...] = ()
        if self._match(TokenType.LBRACE):
            self._consume()
            if self._match(TokenType.RBRACE):
                tok = self._peek()
                msg = "union definition must contain at least one member"
                raise self._token_error(msg, tok) if tok is not None else self._eof_error(msg)
            members_list: list[MemberDeclaration] = []
            while not self._match(TokenType.RBRACE):
                if self._peek() is None:
                    raise self._eof_error("expected '}', got end of input")
                members_list.append(self._parse_member_declaration())
            self._consume()  # consume '}'
            members = tuple(members_list)
        self._expect(TokenType.SEMICOLON)
        return UnionDecl(union_declaration=UnionDeclaration(tag=tag_token.value, members=members))

    def _parse_member_declaration(self) -> MemberDeclaration:
        """Parse a single struct member declaration.

        Grammar::

            member-declaration ::= {type-specifier}+ declarator ";"

        Unlike a variable declaration, no storage-class specifier is permitted
        and no initialiser may follow the declarator.  Function-type declarators
        (e.g. ``int (*fn)(void)``) are also rejected.

        Returns:
            A ``MemberDeclaration`` node.

        Raises:
            ParseError: If the type specifier is absent, if a function-type
                declarator is present, or if the trailing semicolon is missing.
        """
        base_type = self._parse_type()
        name, transform, _params = self._parse_declarator()
        full_type = transform(base_type)
        if isinstance(full_type, FunType):
            err_tok = self._last
            msg = "function type declarators are not permitted in struct members"
            raise self._token_error(msg, err_tok) if err_tok is not None else self._eof_error(msg)
        self._expect(TokenType.SEMICOLON)
        return MemberDeclaration(member_name=name, member_type=full_type)

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

    def _parse_initializer(self) -> Initializer:
        """Parse a variable initialiser: either a brace-enclosed compound or a scalar expression.

        Grammar::

            initializer ::= "{" initializer {"," initializer} [","] "}" | exp

        Returns:
            A ``CompoundInit`` node for brace-enclosed initialisers, or a
            ``SingleInit`` node wrapping the parsed expression.

        Raises:
            ParseError: If the initialiser is malformed.
        """
        if self._match(TokenType.LBRACE):
            self._consume()
            if self._match(TokenType.RBRACE):
                tok = self._peek()
                msg = "empty initializer list is not permitted in C11"
                raise self._token_error(msg, tok) if tok is not None else self._eof_error(msg)
            initializers: list[Initializer] = []
            initializers.append(self._parse_initializer())
            while self._match(TokenType.COMMA):
                self._consume()
                if self._match(TokenType.RBRACE):  # trailing comma
                    break
                initializers.append(self._parse_initializer())
            self._expect(TokenType.RBRACE)
            return CompoundInit(initializers=initializers)
        return SingleInit(exp=self._parse_exp())

    def _parse_variable_declaration(self) -> VarDecl:
        """Parse a variable declaration with an optional initialiser.

        Grammar::

            variable-declaration ::= {specifier}+ declarator ["=" initializer] ";"

        Storage-class specifiers are parsed and recorded on the resulting
        ``VariableDeclaration``.  A function declarator is rejected immediately
        because function declarations are not permitted in for-loop initialisers.

        Returns:
            A ``VarDecl`` node.

        Raises:
            ParseError: If a function declarator is detected, if the identifier or
                semicolon is missing, or if the initialiser expression is malformed.
        """
        base_type, storage = self._parse_specifiers()
        name, transform, _params = self._parse_declarator()
        full_type = transform(base_type)
        if isinstance(full_type, FunType):
            err_tok = self._last
            msg = "function declarations are not permitted in for-loop initialisers"
            raise self._token_error(msg, err_tok) if err_tok is not None else self._eof_error(msg)
        init: Initializer | None = None
        if self._match(TokenType.ASSIGNMENT):
            self._consume()
            init = self._parse_initializer()
        self._expect(TokenType.SEMICOLON)
        return VarDecl(
            variable_declaration=VariableDeclaration(name=name, init=init, var_type=full_type, storage_class=storage)
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
        """Parse a ``return [exp];`` statement.

        The return expression is optional; a bare ``return;`` is valid for
        void functions.

        Returns:
            A ``Return`` node whose ``exp`` field is ``None`` when no expression
            was present.

        Raises:
            ParseError: If the trailing semicolon is missing.
        """
        self._expect(TokenType.RETURN)
        exp = self._parse_optional_exp(TokenType.SEMICOLON)
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
            case TokenType.ASTERISK:
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
        left: Exp = self._parse_cast_exp()
        prec = self._get_precedence()
        while prec is not None and prec >= min_prec:
            if self._match(TokenType.ASSIGNMENT):
                self._consume()
                right = self._parse_exp(prec)  # right-associative
                left = Assignment(left, right)
            elif self._match_compound_assignment():
                op = self._consume_compound_assignment()
                right = self._parse_exp(prec)
                left = CompoundAssignment(left, op, right)
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

    def _parse_cast_exp(self) -> Exp:
        """Parse a cast expression or fall through to a factor.

        Grammar::

            cast-exp ::= "(" type-name ")" cast-exp | factor

        When the next token is ``(`` followed by a type-specifier, the sequence
        is consumed as a C-style cast; otherwise control falls through to
        ``_parse_factor``.

        Returns:
            A ``Cast`` node when a cast was present, otherwise whatever
            ``_parse_factor`` returns.

        Raises:
            ParseError: If the cast type or the operand expression is malformed.
        """
        if self._match(TokenType.LPAREN):
            next_tok = self._peek(1)
            if next_tok is not None and next_tok.type in _TYPE_SPECIFIER_TOKENS:
                self._consume()  # consume "("
                base_cast_type = self._parse_type()
                if self._match(TokenType.ASTERISK) or self._match(TokenType.LPAREN) or self._match(TokenType.LBRACKET):
                    target_type: Type = self._parse_abstract_declarator()(base_cast_type)
                else:
                    target_type = base_cast_type
                self._expect(TokenType.RPAREN)
                return Cast(target_type=target_type, exp=self._parse_cast_exp())
        return self._parse_factor()

    def _parse_factor(self) -> Exp:
        """Parse a factor: a primary expression optionally wrapped in prefix/postfix operators.

        Prefix unary operators (``-``, ``~``, ``!``, ``++``, ``--``, ``*``, ``&``)
        apply to a ``cast-exp`` so that ``(int)-x`` parses correctly.  ``sizeof``
        may be followed by a type name in parentheses (``SizeOfT``) or by an
        arbitrary expression (``SizeOf``). Primary expressions (constants,
        identifiers, function calls, parenthesised expressions) are then checked for
        trailing postfix operators before returning: ``++``, ``--``, ``[`` subscript,
        ``.`` member access, and ``->`` pointer-member access.

        Grammar::

            factor        ::= prefix_unop cast-exp
                            | "sizeof" "(" type-name ")"
                            | "sizeof" cast-exp
                            | primary {postfix-op}
            primary       ::= const
                            | identifier
                            | "(" exp ")"
                            | identifier "(" [argument-list] ")"
            argument-list ::= exp {"," exp}
            prefix_unop   ::= "-" | "~" | "!" | "++" | "--" | "*" | "&"
            postfix-op    ::= "++" | "--" | "[" exp "]" | "." identifier | "->" identifier

        Returns:
            An ``Exp`` node.

        Raises:
            ParseError: If the token does not begin a recognised factor.
        """
        token = self._peek()

        if token is None:
            raise self._eof_error("Syntax error: expected expression, got end of input")

        match token.type:
            case TokenType.ASTERISK:
                self._consume()
                return Dereference(self._parse_cast_exp())
            case TokenType.AMPERSAND:
                self._consume()
                return AddrOf(self._parse_cast_exp())
            case TokenType.BITWISE_NOT | TokenType.MINUS | TokenType.NOT | TokenType.INCREMENT | TokenType.DECREMENT:
                return Unary(self._parse_unaryop(), self._parse_cast_exp())
            case TokenType.SIZEOF:
                self._consume()
                if self._match(TokenType.LPAREN):
                    next_tok = self._peek(1)
                    if next_tok is not None and next_tok.type in _TYPE_SPECIFIER_TOKENS:
                        self._consume()  # consume "("
                        base_type = self._parse_type()
                        if (
                            self._match(TokenType.ASTERISK)
                            or self._match(TokenType.LPAREN)
                            or self._match(TokenType.LBRACKET)
                        ):
                            sizeof_type: Type = self._parse_abstract_declarator()(base_type)
                        else:
                            sizeof_type = base_type
                        self._expect(TokenType.RPAREN)
                        return SizeOfT(t=sizeof_type)
                return SizeOf(exp=self._parse_cast_exp())
            case (
                TokenType.NUMBER
                | TokenType.LONGINT
                | TokenType.UINT
                | TokenType.ULONG
                | TokenType.FPDOUBLE
                | TokenType.CONSTCHAR
            ):
                primary: Exp = self._parse_constant()
            case TokenType.STRING:
                primary = self._parse_string_literal()
            case TokenType.IDENTIFIER:
                self._consume()
                if self._match(TokenType.LPAREN):
                    primary = self._parse_function_call(token.value)
                else:
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
            elif self._match(TokenType.LBRACKET):
                self._consume()
                index = self._parse_exp()
                self._expect(TokenType.RBRACKET)
                primary = Subscript(exp1=primary, exp2=index)
            elif self._match(TokenType.PERIOD):
                self._consume()
                member_token = self._expect(TokenType.IDENTIFIER)
                primary = Dot(structure=primary, member=member_token.value)
            elif self._match(TokenType.ARROW):
                self._consume()
                member_token = self._expect(TokenType.IDENTIFIER)
                primary = Arrow(pointer=primary, member=member_token.value)
            else:
                return primary

    def _parse_function_call(self, name: str) -> FunctionCall:
        """Parse a function call expression after the identifier has been consumed.

        Grammar::

            identifier "(" [argument-list] ")"
            argument-list ::= exp {"," exp}

        Consumes the opening ``(``, parses zero or more comma-separated
        argument expressions, and consumes the closing ``)``.

        The callee is represented as a ``Var`` wrapping the identifier so that
        direct calls and indirect calls through function-pointer variables are
        both expressed as ``FunctionCall(callee=<Exp>, args=...)``.

        Args:
            name: The function name, already consumed by the caller.

        Returns:
            A ``FunctionCall`` node whose ``callee`` is ``Var(name)``.

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
        return FunctionCall(callee=Var(name=name), args=args)

    # Maps single-character escape letter to its ASCII value.
    _ESCAPE_VALUES: dict[str, int] = {
        "'": 39,
        '"': 34,
        "?": 63,
        "\\": 92,
        "a": 7,
        "b": 8,
        "f": 12,
        "n": 10,
        "r": 13,
        "t": 9,
        "v": 11,
    }

    def _parse_constant(self) -> Constant:
        """Parse an integer, long, double, or character constant expression.

        Follows C11 sec. 6.4.4.1 for numeric constants and sec. 6.4.4.4 for character
        constants.

        - An unsuffixed decimal constant has the type of the first of ``int``,
          ``long`` in which its value can be represented.
        - A constant with an ``l`` or ``L`` suffix is always ``long``.
        - A constant with a ``u`` or ``U`` suffix has the type of the first of
          ``unsigned int``, ``unsigned long`` in which its value can be
          represented.
        - A constant with both ``u``/``U`` and ``l``/``L`` suffixes is always
          ``unsigned long``.
        - A character constant (e.g. ``'a'``, ``'\\n'``) is converted to the
          ``int`` value of its ASCII code.

        Returns:
            A ``Constant`` node wrapping a ``ConstInt``, ``ConstLong``,
            ``ConstUInt``, ``ConstULong``, or ``ConstDouble``.

        Raises:
            ParseError: If the next token is not a numeric or character
                constant, or if the value exceeds the range of the largest
                supported type.
        """
        token = self._peek()
        if token is None:
            raise self._eof_error("expected a constant, got end of input")
        if token.type == TokenType.FPDOUBLE:
            self._consume()
            return Constant(value=ConstDouble(float(token.value)), exp_type=DoubleType())
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
        if token.type == TokenType.CONSTCHAR:
            self._consume()
            inner = token.value[1:-1]  # strip surrounding single quotes
            if inner.startswith("\\"):
                char_val = self._ESCAPE_VALUES[inner[1]]
            else:
                char_val = ord(inner)
            return Constant(value=ConstInt(char_val), exp_type=IntType())
        raise self._token_error(f"expected a constant, got {token.type.name} '{token.value}'", token)

    @staticmethod
    def _unescape_string(raw: str) -> bytes:
        """Convert the raw source text of a string literal to a byte sequence.

        Args:
            raw: The token value including surrounding double-quote characters,
                e.g. ``'"hello\\\\n"'``.

        Returns:
            A ``bytes`` object containing the unescaped content, without the
            NUL terminator.
        """
        _ESC: dict[str, int] = {
            "'": 39,
            '"': 34,
            "?": 63,
            "\\": 92,
            "a": 7,
            "b": 8,
            "f": 12,
            "n": 10,
            "r": 13,
            "t": 9,
            "v": 11,
        }
        content = raw[1:-1]  # strip surrounding double quotes
        result: list[int] = []
        i = 0
        while i < len(content):
            ch = content[i]
            if ch == "\\":
                result.append(_ESC[content[i + 1]])
                i += 2
            else:
                result.append(ord(ch))
                i += 1
        return bytes(result)

    def _parse_string_literal(self) -> String:
        """Parse one or more adjacent string-literal tokens into a single ``String`` node.

        C allows adjacent string literals to be concatenated at compile time,
        e.g. ``"hello " "world"`` is equivalent to ``"hello world"``.

        Returns:
            A ``String`` node whose ``string`` field holds the concatenated,
            unescaped byte content of all adjacent tokens (NUL not included).

        Raises:
            ParseError: If the current token is not a ``STRING`` token.
        """
        token = self._peek()
        if token is None or token.type != TokenType.STRING:
            msg = "expected a string literal"
            raise self._eof_error(msg) if token is None else self._token_error(msg, token)
        buf = bytearray()
        while (tok := self._peek()) is not None and tok.type == TokenType.STRING:
            self._consume()
            buf.extend(self._unescape_string(tok.value))
        return String(string=bytes(buf))

    def _parse_type(self) -> Type:
        """Parse a C type annotation from one or more type-specifier tokens.

        Grammar::

            type-specifier ::= "int" | "long" | "signed" | "unsigned" | "double"
                              | "char" | "signed" "char" | "unsigned" "char"
                              | "struct" identifier

        Consumes a run of specifier tokens in any order.  ``unsigned`` or
        ``signed`` alone are treated as shorthand for ``unsigned int`` /
        ``signed int`` respectively.  ``char`` may only be combined with
        ``signed`` or ``unsigned``; pairing it with ``int``, ``long``, or
        ``double`` is a parse error.  ``struct identifier`` is consumed as a
        two-token unit and may not be combined with any other specifier.

        Returns:
            A ``Type`` node.

        Raises:
            ParseError: If no recognised type-specifier token is present, if a
                specifier appears more than once, or if ``double`` is combined
                with another type specifier.
        """
        seen, last_tok, struct_tag, union_tag = self._consume_type_specifiers()
        return self._resolve_type(seen, last_tok, struct_tag, union_tag)

    def _parse_abstract_declarator(self) -> Callable[[Type], Type]:
        """Parse an abstract declarator (no identifier) for use in cast expressions.

        Grammar::

            abstract-declarator          ::= "*" [abstract-declarator]
                                           | direct-abstract-declarator
            direct-abstract-declarator   ::= "(" abstract-declarator ")" {"[" const "]"}
                                           | {"[" const "]"}+

        Returns:
            A callable that maps the base type to the fully-derived type.
            For example, ``*`` applied to ``IntType()`` yields ``Pointer(IntType())``,
            and ``(*)[3]`` applied to ``IntType()`` yields ``Pointer(Array(IntType(), 3))``.

        Raises:
            ParseError: If the abstract declarator is malformed.
        """
        if self._match(TokenType.ASTERISK):
            self._consume()
            if self._match(TokenType.ASTERISK) or self._match(TokenType.LPAREN) or self._match(TokenType.LBRACKET):
                inner = self._parse_abstract_declarator()

                def _ptr_inner(t: Type, _i: Callable[[Type], Type] = inner) -> Type:
                    return _i(Pointer(t))

                return _ptr_inner
            return lambda t: Pointer(t)

        if self._match(TokenType.LPAREN):
            self._consume()
            inner = self._parse_abstract_declarator()
            self._expect(TokenType.RPAREN)
            if not self._match(TokenType.LBRACKET):
                return inner
            suffix_sizes: list[int] = []
            while self._match(TokenType.LBRACKET):
                self._consume()
                suffix_sizes.append(self._parse_array_dimension())
                self._expect(TokenType.RBRACKET)

            def _parens_arr(t: Type, _i: Callable[[Type], Type] = inner, _sizes: list[int] = suffix_sizes) -> Type:
                for size in reversed(_sizes):
                    t = Array(t, size)
                return _i(t)

            return _parens_arr

        bare_sizes: list[int] = []
        while self._match(TokenType.LBRACKET):
            self._consume()
            bare_sizes.append(self._parse_array_dimension())
            self._expect(TokenType.RBRACKET)

        if not bare_sizes:
            tok = self._peek()
            msg = "expected abstract declarator ('*', '(', or '[')"
            raise self._token_error(msg, tok) if tok is not None else self._eof_error(msg)

        def _bare_arr(t: Type, _sizes: list[int] = bare_sizes) -> Type:
            for size in reversed(_sizes):
                t = Array(t, size)
            return t

        return _bare_arr

    def _parse_simple_declarator(self) -> tuple[str, Callable[[Type], Type], tuple[Parameter, ...] | None]:
        """Parse a simple-declarator: an identifier or a parenthesised declarator.

        Grammar::

            simple-declarator ::= identifier | "(" declarator ")"

        Returns:
            A ``(name, type_builder, params)`` triple.  For a plain identifier the
            type_builder is the identity and params is ``None``.  For a parenthesised
            declarator all three values are propagated from the inner declarator so
            that ``_parse_direct_declarator`` can detect and reject a function
            declarator that is itself wrapped in another function declarator.

        Raises:
            ParseError: If neither an identifier nor a valid parenthesised declarator
                is found.
        """
        if self._match(TokenType.LPAREN):
            self._consume()
            name, transform, params = self._parse_declarator()
            self._expect(TokenType.RPAREN)
            return name, transform, params
        name_token = self._expect(TokenType.IDENTIFIER)
        return name_token.value, lambda t: t, None

    def _parse_array_dimension(self) -> int:
        """Parse a single array dimension from a ``[`` ``]`` declarator suffix.

        Expects the opening ``[`` to have been consumed already.  Reads an
        integer constant (plain, ``l``/``L``, ``u``/``U``, ``ul``/``UL``
        suffix, or a character literal) and leaves the stream positioned at
        the closing ``]``, which
        the caller must consume via ``_expect``.

        Returns:
            The dimension size as a Python ``int``.

        Raises:
            ParseError: If the dimension is missing or is not an integer constant.
        """
        tok = self._peek()
        if tok is None:
            raise self._eof_error("Expected array size")
        match tok.type:
            case TokenType.NUMBER:
                val = int(tok.value)
            case TokenType.LONGINT:
                val = int(tok.value.rstrip("lL"))
            case TokenType.UINT:
                val = int(tok.value.rstrip("uU"))
            case TokenType.ULONG:
                val = int(tok.value.rstrip("uUlL"))
            case TokenType.CONSTCHAR:
                inner = tok.value[1:-1]
                val = self._ESCAPE_VALUES[inner[1]] if inner.startswith("\\") else ord(inner)
            case _:
                raise self._token_error("Array size must be an integer constant", tok)
        self._consume()
        return val

    def _parse_direct_declarator(self) -> tuple[str, Callable[[Type], Type], tuple[Parameter, ...] | None]:
        """Parse a direct-declarator.

        Grammar::

            direct-declarator ::= simple-declarator [declarator-suffix]
            declarator-suffix  ::= param-list | {"[" const "]"}+

        A function declarator nested inside a parenthesised simple-declarator that
        is itself followed by a param-list would produce a function-returning-function
        type, which C does not permit.  Such a construction is rejected here.

        Returns:
            A ``(name, type_builder, params)`` triple. ``params`` is ``None`` when no
            parameter list follows; otherwise a (possibly empty) tuple of ``Parameter``
            nodes making this a function declarator.

        Raises:
            ParseError: If the direct-declarator is malformed, or if a function
                declarator is nested inside another function declarator.
        """
        name, simple_transform, simple_params = self._parse_simple_declarator()

        if self._match(TokenType.LBRACKET):
            sizes: list[int] = []
            while self._match(TokenType.LBRACKET):
                self._consume()  # consume "["
                sizes.append(self._parse_array_dimension())
                self._expect(TokenType.RBRACKET)

            def _arr(t: Type, _st: Callable[[Type], Type] = simple_transform, _sizes: list[int] = sizes) -> Type:
                for size in reversed(_sizes):
                    t = Array(t, size)
                return _st(t)

            return name, _arr, simple_params

        if not self._match(TokenType.LPAREN):
            return name, simple_transform, simple_params

        # A param-list here would make this a function declarator.  If the
        # simple-declarator already produced a function declarator (simple_params
        # is not None), that would mean a function returning a function - invalid.
        if simple_params is not None:
            tok = self._peek()
            msg = f"'{name}' declared as function returning a function"
            raise self._token_error(msg, tok) if tok is not None else self._eof_error(msg)

        # If the simple-declarator's transform wraps the result in an Array, applying
        # a function suffix would yield an array-of-functions type, which C forbids.
        if isinstance(simple_transform(FunType(params=(), ret=IntType())), Array):
            tok = self._peek()
            msg = f"'{name}' declared as array of functions"
            raise self._token_error(msg, tok) if tok is not None else self._eof_error(msg)

        self._consume()  # consume "("
        params = self._parse_params()
        self._expect(TokenType.RPAREN)
        param_types = tuple(p.param_type for p in params)

        def _fun(t: Type, _st: Callable[[Type], Type] = simple_transform, _pt: tuple[Type, ...] = param_types) -> Type:
            return _st(FunType(params=_pt, ret=t))

        return name, _fun, params

    def _parse_declarator(self) -> tuple[str, Callable[[Type], Type], tuple[Parameter, ...] | None]:
        """Parse a declarator, returning the name, a type-builder, and optional parameters.

        Grammar::

            declarator        ::= "*" declarator | direct-declarator
            direct-declarator ::= simple-declarator [param-list]
            simple-declarator ::= identifier | "(" declarator ")"

        The type-builder takes the base (specifier) type and returns the full derived type.
        Examples with base type ``IntType()``:

        * ``x``          -> ``IntType()``
        * ``*x``         -> ``Pointer(IntType())``
        * ``*(*(x))``    -> ``Pointer(Pointer(IntType()))``
        * ``*foo(void)`` -> ``FunType(params=(), ret=Pointer(IntType()))``

        Returns:
            A ``(name, type_builder, params)`` triple where ``name`` is the declared
            identifier, ``type_builder`` maps the base type to the derived type, and
            ``params`` is a tuple of ``Parameter`` nodes when a parameter list was
            present (function declarator) or ``None`` otherwise.

        Raises:
            ParseError: If the declarator is malformed.
        """
        if self._match(TokenType.ASTERISK):
            self._consume()
            inner_name, inner_transform, params = self._parse_declarator()

            def _ptr(t: Type, _inner: Callable[[Type], Type] = inner_transform) -> Type:
                return _inner(Pointer(t))

            return inner_name, _ptr, params
        return self._parse_direct_declarator()


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
