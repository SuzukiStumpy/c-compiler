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
from typing import NamedTuple

import compiler.syntax.ast as ast
from compiler.utils.exceptions import SemanticError


class VariableBinding(NamedTuple):
    """A single entry in the variable map, recording a resolved name and its scope origin.

    Attributes:
        unique_name: The unique internal identifier assigned to the source-level variable
            (e.g. ``".Vx.0"``).
        from_current_block: ``True`` if this binding was declared in the block currently
            being analysed; ``False`` if it was inherited from an enclosing scope.  Used
            by ``resolve_declaration`` to distinguish shadowing (permitted) from a
            same-block duplicate (an error).
    """

    unique_name: str
    from_current_block: bool


class SemanticAnalyser:
    """Resolves variable references across a single AST program.

    Each instance owns a fresh counter and should be used for exactly one
    ``_resolve_program`` call.  Constructing a new instance per program
    guarantees that unique suffixes start from zero regardless of how many
    programs are analysed within the same process, which keeps test output
    deterministic.

    Attributes:
        _counter: Monotonically increasing integer source for unique variable
            name suffixes.
        _loop_counter: Monotonically increasing integer source for unique loop
            label suffixes, kept separate from ``_counter`` so that loop
            identifiers are predictable in tests regardless of how many
            variables have been declared.
        _variable_map: Mapping from source-level variable names to
            ``VariableBinding`` values.  Each binding records both the unique
            internal name and whether it was declared in the current block.
            Outer-scope bindings are retained in the map with
            ``from_current_block=False`` so that variable lookups can still
            resolve them; when a compound block exits the map is restored to
            its saved snapshot.
        _label_set: Set of all label names declared in the function currently
            being analysed.  Populated by a first pass before resolution begins
            so that forward ``goto`` targets can be validated.
        _loop_stack: Stack of loop identifiers for the loops currently
            enclosing the statement being resolved.  The top of the stack
            is the identifier of the immediately enclosing loop and is
            attached to any ``Break`` or ``Continue`` encountered during
            resolution.  An empty stack indicates that the current statement
            is not within any loop; a ``Break`` or ``Continue`` in that
            context is a semantic error.
    """

    def __init__(self) -> None:
        """Initialise the analyser with fresh counters and an empty symbol table."""
        self._counter: count[int] = count()
        self._loop_counter: count[int] = count()
        self._variable_map: dict[str, VariableBinding] = {}
        self._label_set: set[str] = set()
        self._loop_stack: list[str] = []

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

    def _make_loop_label(self, kind: str) -> str:
        """Generate a unique identifier for a loop construct.

        Produces identifiers of the form ``.L.while.0``, ``.L.for.1``,
        ``.L.do.2``, etc.  The ``.L.`` prefix distinguishes loop labels from
        the ``.V`` prefix used for renamed variables.  The ``kind`` segment
        names the construct for readability in TAC and assembly output.

        Args:
            kind: A short string identifying the loop type, e.g. ``"while"``,
                ``"for"``, or ``"do"``.

        Returns:
            A string of the form ``".L.<kind>.<n>"`` where ``<n>`` is the next
            value drawn from ``_loop_counter``.
        """
        return f".L.{kind}.{next(self._loop_counter)}"

    def _collect_labels_in_statement(self, statement: ast.Statement) -> None:
        """Recursively collect label names from a statement into ``_label_set``.

        Raises ``SemanticError`` immediately upon finding a duplicate label name.
        Recurses into ``If``, ``Label``, ``Compound``, and all loop body nodes
        so that labels nested at any depth are captured.

        Args:
            statement: The statement node to walk.

        Raises:
            SemanticError: If a label name already present in ``_label_set`` is
                encountered, indicating a duplicate within the current function.
        """
        match statement:
            case ast.Label(name=name, statement=inner):
                if name in self._label_set:
                    raise SemanticError(f"Duplicate label '{name}' in function.")
                self._label_set.add(name)
                self._collect_labels_in_statement(inner)
            case ast.If(then=then_stmt, else_=else_stmt):
                self._collect_labels_in_statement(then_stmt)
                if else_stmt is not None:
                    self._collect_labels_in_statement(else_stmt)
            case ast.Compound(body=body):
                for item in body:
                    match item:
                        case ast.S(statement=inner_stmt):
                            self._collect_labels_in_statement(inner_stmt)
            case ast.While(body=loop_body) | ast.DoWhile(body=loop_body):
                self._collect_labels_in_statement(loop_body)
            case ast.For(body=loop_body):
                self._collect_labels_in_statement(loop_body)
            case _:
                pass

    def _collect_labels(self, body: tuple[ast.BlockItem, ...]) -> None:
        """First-pass walk over a function body to populate ``_label_set``.

        Only statement block items are walked; declaration items contain no
        labels.  Delegates to ``_collect_labels_in_statement`` for each
        statement found.

        Args:
            body: The sequence of block items forming the function body.

        Raises:
            SemanticError: If any duplicate label name is found within ``body``.
        """
        for item in body:
            match item:
                case ast.S(statement=statement):
                    self._collect_labels_in_statement(statement)

    def resolve_declaration(self, declaration: ast.Declaration) -> ast.Declaration:
        """Resolve a variable declaration against the current symbol table.

        Verifies that ``declaration.name`` has not already been declared in the
        *current* block (detected via the ``from_current_block`` flag on the
        existing ``VariableBinding``), assigns it a fresh unique internal name,
        records the mapping in ``_variable_map``, and returns a new
        ``Declaration`` node whose ``name`` field holds the unique internal
        name.  Declarations with the same source name in an *outer* block are
        silently shadowed: the new binding overwrites the map entry for the
        duration of the enclosing compound block, and the outer binding is
        restored when that block exits.

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
            SemanticError: If an existing ``VariableBinding`` for
                ``declaration.name`` has ``from_current_block=True``, indicating
                a duplicate declaration in the same block.
        """
        binding = self._variable_map.get(declaration.name)
        if binding is not None and binding.from_current_block:
            raise SemanticError(f"Duplicate variable declaration: {declaration.name}!")

        unique_name = self._make_unique(declaration.name)
        self._variable_map[declaration.name] = VariableBinding(unique_name, from_current_block=True)
        init = self.resolve_exp(declaration.init) if declaration.init is not None else None

        return ast.Declaration(unique_name, init)

    def _enter_scope(self) -> dict[str, VariableBinding]:
        """Open a new nested scope and return a snapshot of the current bindings.

        Saves a copy of ``_variable_map``, then marks every existing binding as
        ``from_current_block=False`` so that declarations in the new scope may
        shadow outer names without being treated as same-block duplicates.

        Returns:
            A copy of ``_variable_map`` as it stood before entering the scope.
            Assign this value back to ``self._variable_map`` once the nested
            scope has been fully processed to restore the enclosing bindings.
        """
        saved = dict(self._variable_map)
        self._variable_map = {k: v._replace(from_current_block=False) for k, v in self._variable_map.items()}
        return saved

    def _resolve_block(self, body: ast.Block) -> ast.Block:
        """Resolve a sequence of block items in a new nested scope.

        Opens a nested scope via ``_enter_scope``, processes ``body``, then
        restores the enclosing bindings.  Any variable declared inside ``body``
        is invisible after this call returns; outer-scope bindings remain
        accessible throughout.

        Args:
            body: The ordered sequence of block items to resolve.

        Returns:
            A new tuple of resolved block items.

        Raises:
            SemanticError: If any block item within ``body`` contains a
                semantic violation.
        """
        saved_map = self._enter_scope()
        result = tuple(self._resolve_block_item(item) for item in body)
        self._variable_map = saved_map
        return result

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
                    return ast.Var(self._variable_map[name].unique_name)
                else:
                    raise SemanticError(f"Undeclared variable {name}!")
            case ast.Constant():
                return e
            case ast.Unary(operator=op, exp=exp):
                return ast.Unary(op, self.resolve_exp(exp))
            case ast.Binary(operator=op, left=left, right=right):
                return ast.Binary(op, self.resolve_exp(left), self.resolve_exp(right))
            case ast.Conditional(condition=cond, exp_true=exp_true, exp_false=exp_false):
                return ast.Conditional(self.resolve_exp(cond), self.resolve_exp(exp_true), self.resolve_exp(exp_false))
            case _:
                raise SemanticError("Unhandled expression type in semantic analyser")

    def _resolve_statement(self, statement: ast.Statement) -> ast.Statement:
        """Resolve all variable references within a statement and assign loop identifiers.

        Recursively resolves any expressions contained in the statement,
        returning a new statement node with all ``ast.Var`` references rewritten
        to their unique internal names.  Additionally, each loop construct
        (``While``, ``DoWhile``, ``For``) is assigned a fresh unique identifier
        via ``_make_loop_label``, and every ``Break`` or ``Continue`` encountered
        within the loop body is tagged with that same identifier.

        Args:
            statement: The statement node to resolve.

        Returns:
            A new statement node with all variable references resolved and all
            loop-control statements tagged with their enclosing loop's identifier.

        Raises:
            SemanticError: If any expression within the statement contains an
                undeclared variable or an invalid lvalue, or if a ``Break`` or
                ``Continue`` appears outside any enclosing loop.
        """
        match statement:
            case ast.Compound(body=block):
                return ast.Compound(self._resolve_block(block))
            case ast.Return(exp=exp):
                return ast.Return(self.resolve_exp(exp))
            case ast.Expression(exp=exp):
                return ast.Expression(self.resolve_exp(exp))
            case ast.Null():
                return statement
            case ast.Break():
                if not self._loop_stack:
                    raise SemanticError("'break' statement not within a loop.")
                return ast.Break(self._loop_stack[-1])
            case ast.Continue():
                if not self._loop_stack:
                    raise SemanticError("'continue' statement not within a loop.")
                return ast.Continue(self._loop_stack[-1])
            case ast.If(condition=cond, then=then_stmt, else_=None):
                return ast.If(self.resolve_exp(cond), self._resolve_statement(then_stmt), None)
            case ast.If(condition=cond, then=then_stmt, else_=else_stmt):
                return ast.If(
                    self.resolve_exp(cond), self._resolve_statement(then_stmt), self._resolve_statement(else_stmt)
                )
            case ast.While(condition=condition, body=body):
                label = self._make_loop_label("while")
                self._loop_stack.append(label)
                result = ast.While(self.resolve_exp(condition), self._resolve_statement(body), label)
                self._loop_stack.pop()
                return result
            case ast.DoWhile(body=body, condition=condition):
                label = self._make_loop_label("do")
                self._loop_stack.append(label)
                result = ast.DoWhile(self._resolve_statement(body), self.resolve_exp(condition), label)
                self._loop_stack.pop()
                return result
            case ast.For(init=init, condition=condition, post=post, body=body):
                saved_variable_map = self._enter_scope()
                label = self._make_loop_label("for")
                self._loop_stack.append(label)
                init = self._resolve_for_init(init)
                condition = self._resolve_optional_exp(condition)
                post = self._resolve_optional_exp(post)
                body = self._resolve_statement(body)
                self._loop_stack.pop()
                self._variable_map = saved_variable_map
                return ast.For(init, condition, post, body, label)
            case ast.Goto(target=target):
                if target not in self._label_set:
                    raise SemanticError(f"goto targets undefined label '{target}'.")
                return statement
            case ast.Label(name=name, statement=inner):
                return ast.Label(name, self._resolve_statement(inner))
            case _:
                raise SemanticError("Unhandled statement type in semantic analyser")

    def _resolve_for_init(self, init: ast.ForInit) -> ast.ForInit:
        """Resolve the initialiser clause of a ``for`` loop.

        Dispatches to ``_resolve_optional_exp`` for an expression initialiser
        (``InitExp``) and to ``resolve_declaration`` for a declaration
        initialiser (``InitDecl``).

        Args:
            init: The ``ForInit`` node from the parser — either an ``InitExp``
                wrapping an optional expression, or an ``InitDecl`` wrapping a
                variable declaration.

        Returns:
            A new ``ForInit`` node with all variable references resolved.

        Raises:
            SemanticError: If a declaration within ``init`` duplicates a name
                already declared in the same scope, or if an expression within
                ``init`` contains an undeclared variable or an invalid lvalue.
        """
        match init:
            case ast.InitExp(exp=exp):
                return ast.InitExp(self._resolve_optional_exp(exp))
            case ast.InitDecl(declaration=declaration):
                return ast.InitDecl(self.resolve_declaration(declaration))
            case _:
                raise SemanticError("Unhandled for-init type in semantic analyser")

    def _resolve_optional_exp(self, exp: ast.Exp | None) -> ast.Exp | None:
        """Resolve an optional expression, returning ``None`` when absent.

        A convenience wrapper around ``resolve_exp`` for contexts where an
        expression is syntactically optional, such as the condition and post
        expressions of a ``for`` loop.

        Args:
            exp: The expression to resolve, or ``None`` if absent.

        Returns:
            The resolved expression, or ``None`` if ``exp`` is ``None``.

        Raises:
            SemanticError: If ``exp`` is not ``None`` and contains an
                undeclared variable or an invalid lvalue.
        """
        if exp is not None:
            return self.resolve_exp(exp)
        return None

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
        self._label_set = set()
        self._loop_stack = []
        self._collect_labels(function.body)
        return ast.Function(
            return_type=function.return_type,
            name=function.name,
            params=function.params,
            body=tuple(self._resolve_block_item(item) for item in function.body),
        )

    def _resolve_program(self, program: ast.Program) -> ast.Program:
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
    return SemanticAnalyser()._resolve_program(program)
