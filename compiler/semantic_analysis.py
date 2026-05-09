"""Semantic analysis stage: variable resolution over the AST.

Walks the ``compiler.syntax.ast`` tree produced by the parser and performs
semantic checks that cannot be expressed by the grammar alone.  At present the
sole responsibility is *variable resolution*: every variable reference is
verified to have a corresponding declaration in scope, and each declared name is
assigned a unique internal identifier so that later pipeline stages (TAC
generation, code generation) need not reason about shadowing or scoping rules.

The main entry point for external callers is ``resolve``, which constructs a
``SemanticAnalyser``, runs it over the program, and returns a semantically
validated, uniquely-renamed ``ast.Program``.  The ``SemanticAnalyser`` class
is also exposed directly for testing.

Raises:
    SemanticError: If the program contains a semantic violation such as
        a duplicate declaration or a reference to an undeclared variable.
"""

from itertools import count

import compiler.syntax.ast as ast
from compiler.utils.exceptions import SemanticError


class SemanticAnalyser:
    """Resolves variable references across a single AST program.

    Each instance owns a fresh counter and should be used for exactly one
    ``resolve_program`` call.  Constructing a new instance per program
    guarantees that unique suffixes start from zero regardless of how many
    programs are analysed within the same process, which keeps test output
    deterministic.

    Attributes:
        _counter: Monotonically increasing integer source for unique suffixes.
        _variable_map: Mapping from source-level variable names to their unique
            internal names for the function currently being analysed.
    """

    def __init__(self) -> None:
        """Initialise the analyser with a fresh counter and an empty symbol table."""
        self._counter: count[int] = count()
        self._variable_map: dict[str, str] = {}

    def _make_unique(self, name: str) -> str:
        """Generate a unique internal name for a source-level variable.

        Produces identifiers of the form ``.Vx.0``, ``.Vx.1``, etc.  The
        leading ``.V`` prefix ensures the name cannot collide with any valid C
        identifier (C identifiers may not begin with a dot).

        Args:
            name: The source-level variable name to uniquify.

        Returns:
            A string of the form ``".V<name>.<n>"`` where ``<n>`` is the next
            value drawn from ``_counter``.
        """
        return f".V{name}.{next(self._counter)}"

    def resolve_declaration(self, declaration: ast.Declaration) -> ast.Declaration:
        """Resolve a variable declaration against the current symbol table.

        Verifies that ``declaration.name`` has not already been declared in the
        current scope, assigns it a fresh unique internal name, records the
        mapping in ``_variable_map``, and returns a new ``Declaration`` node
        whose ``name`` field holds the unique internal name.

        The binding is added to ``_variable_map`` *before* the initialiser is
        resolved.  This matches the C standard (C11 §6.2.1p7), under which the
        scope of a declared identifier begins immediately after its declarator —
        before the ``=`` initialiser — so constructs such as
        ``int a = 0 && a;`` are valid (the right-hand side of ``&&`` is never
        evaluated when the left is zero).  A self-referential initialiser such
        as ``int x = x;`` is accepted at this stage; it is undefined behaviour
        at runtime, but not a constraint violation.

        Args:
            declaration: The ``Declaration`` AST node to resolve.  Its ``name``
                field is the source-level identifier; its ``init`` field is
                either ``None`` (no initialiser) or an expression tree.

        Returns:
            A new ``Declaration`` node with ``name`` replaced by the unique
            internal identifier and ``init`` (if present) rewritten so that all
            variable references within it use their unique internal names.

        Raises:
            SemanticError: If ``declaration.name`` is already present in
                ``_variable_map``, indicating a duplicate declaration in the
                same scope.
        """
        if declaration.name in self._variable_map:
            raise SemanticError(f"Duplicate variable declaration: {declaration.name}!")

        unique_name = self._make_unique(declaration.name)
        self._variable_map[declaration.name] = unique_name
        init = self.resolve_exp(declaration.init) if declaration.init is not None else None

        return ast.Declaration(unique_name, init)

    def resolve_exp(self, e: ast.Exp) -> ast.Exp:
        """Resolve all variable references within an expression.

        Recursively walks the expression tree, substituting every ``ast.Var``
        node with a new ``ast.Var`` whose name is the unique internal identifier
        recorded in ``_variable_map``.  Assignment expressions are additionally
        validated to ensure their left-hand side is an ``ast.Var`` (i.e. a
        simple lvalue).

        Args:
            e: The expression node to resolve.

        Returns:
            A new expression node structurally identical to ``e`` but with all
            ``ast.Var`` references rewritten to use unique internal names.

        Raises:
            SemanticError: If a variable reference cannot be found in
                ``_variable_map`` (undeclared variable), or if the left-hand
                side of an assignment is not a plain ``ast.Var`` (invalid
                lvalue).
        """
        match e:
            case ast.Assignment(lvalue=ast.Var() as lvalue_var, rvalue=rvalue):
                return ast.Assignment(self.resolve_exp(lvalue_var), self.resolve_exp(rvalue))
            case ast.Assignment():
                raise SemanticError("Invalid lvalue in assignment expression.")
            case ast.Var(name=name):
                if name in self._variable_map:
                    return ast.Var(self._variable_map[name])
                else:
                    raise SemanticError(f"Undeclared variable {name}!")
            case ast.Constant():
                return e
            case ast.Unary(operator=op, exp=exp):
                return ast.Unary(op, self.resolve_exp(exp))
            case ast.Binary(operator=op, left=left, right=right):
                return ast.Binary(op, self.resolve_exp(left), self.resolve_exp(right))
            case _:
                raise SemanticError("Unhandled expression type in semantic analyser")

    def _resolve_statement(self, statement: ast.Statement) -> ast.Statement:
        """Resolve all variable references within a statement.

        Recursively resolves any expressions contained in the statement,
        returning a new statement node with all ``ast.Var`` references rewritten
        to their unique internal names.

        Args:
            statement: The statement node to resolve.

        Returns:
            A new statement node with all variable references resolved.

        Raises:
            SemanticError: If any expression within the statement contains an
                undeclared variable or an invalid lvalue.
        """
        match statement:
            case ast.Return(exp=exp):
                return ast.Return(self.resolve_exp(exp))
            case ast.Expression(exp=exp):
                return ast.Expression(self.resolve_exp(exp))
            case ast.Null():
                return statement
            case _:
                raise SemanticError("Unhandled statement type in semantic analyser")

    def _resolve_block_item(self, item: ast.BlockItem) -> ast.BlockItem:
        """Resolve a single block item (statement or declaration).

        Dispatches to ``resolve_declaration`` for ``ast.D`` items and to
        ``_resolve_statement`` for ``ast.S`` items.

        Args:
            item: The block item to resolve.

        Returns:
            A new block item with all variable references resolved.

        Raises:
            SemanticError: If the item contains a semantic violation.
        """
        match item:
            case ast.D(declaration=declaration):
                return ast.D(self.resolve_declaration(declaration))
            case ast.S(statement=statement):
                return ast.S(self._resolve_statement(statement))
            case _:
                raise SemanticError("Unhandled block item type in semantic analyser")

    def _resolve_function(self, function: ast.Function) -> ast.Function:
        """Resolve all variable references within a function definition.

        Resets ``_variable_map`` to an empty scope for this function, then
        walks its body resolving every block item in order.

        Args:
            function: The ``Function`` AST node to resolve.

        Returns:
            A new ``Function`` node with all variable references resolved and
            all declared names replaced by unique internal identifiers.

        Raises:
            SemanticError: If the function body contains any semantic violation.
        """
        self._variable_map = {}
        return ast.Function(
            return_type=function.return_type,
            name=function.name,
            params=function.params,
            body=tuple(self._resolve_block_item(item) for item in function.body),
        )

    def resolve_program(self, program: ast.Program) -> ast.Program:
        """Perform semantic analysis on a parsed program.

        Resolves every function in the program in order, returning a new
        ``ast.Program`` in which all variable names have been replaced by
        unique internal identifiers and all semantic constraints have been
        verified.

        Args:
            program: The ``Program`` AST node produced by the parser.

        Returns:
            A semantically validated ``Program`` node with uniquely renamed
            variable references throughout.

        Raises:
            SemanticError: If the program contains any semantic violation such
                as a duplicate declaration or a reference to an undeclared
                variable.
        """
        return ast.Program(functions=tuple(self._resolve_function(fn) for fn in program.functions))


def resolve(program: ast.Program) -> ast.Program:
    """Convenience entry point for the semantic analysis stage.

    Constructs a fresh ``SemanticAnalyser`` and immediately runs it over
    ``program``.  Callers that need access to the analyser instance (e.g. for
    testing individual helpers) should instantiate ``SemanticAnalyser``
    directly.

    Args:
        program: The ``Program`` AST node produced by the parser.

    Returns:
        A semantically validated ``Program`` node with uniquely renamed
        variable references throughout.

    Raises:
        SemanticError: If the program contains any semantic violation.
    """
    return SemanticAnalyser().resolve_program(program)
