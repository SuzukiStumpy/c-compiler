"""Semantic analysis stage: variable and function resolution over the AST.

Walks the ``compiler.syntax.ast`` tree produced by the parser and performs
semantic checks that cannot be expressed by the grammar alone.  Responsibilities
are *variable resolution*, *function declaration validation*, and *type checking*:

* Every variable reference is verified to have a corresponding declaration in
  scope, and each declared name is assigned a unique internal identifier so that
  later pipeline stages (TAC generation, code generation) need not reason about
  shadowing or scoping rules.  File-scope variables keep their original names;
  block-scope variables (including ``static`` locals) are renamed to unique
  internal identifiers.
* Every function name is tracked in a program-wide function map.  Multiple
  *declarations* of the same function are permitted; a function *definition*
  (one with a body) may appear at most once per program.  Top-level functions
  are processed in source order; a function must be declared or defined before
  any call site that precedes it.
* File-scope variable declarations are added to the identifier table unaltered.
  ``static`` file-scope variables receive ``Linkage.INTERNAL``; all others
  receive ``Linkage.EXTERNAL``.
* Block-scope ``extern`` declarations are checked to ensure they carry no
  initialiser and are not mixed with a non-``extern`` declaration of the same
  name in the same scope.  Block-scope ``static`` declarations are renamed
  exactly like regular local variables but carry ``Linkage.INTERNAL``.
* ``FunctionCall`` expressions are validated to ensure the callee has been
  declared or defined at an earlier point in the source, and that the callee
  name is not a local variable.
* ``Var`` expressions are checked to ensure the name was declared as a variable
  rather than as a function.
* Linkage is recorded for each binding: ``Linkage.NONE`` for local variables;
  ``Linkage.INTERNAL`` for ``static`` variables and for ``extern`` declarations
  that inherit a prior internal linkage (C11 §6.2.2p4); ``Linkage.EXTERNAL``
  for functions and file-scope variables that have not established internal
  linkage.

The main entry point for external callers is ``resolve``, which constructs a
``SemanticAnalyser``, runs it over the program, and returns a semantically
validated, uniquely-renamed ``ast.Program``.  The ``SemanticAnalyser`` class
is also exposed directly for testing.

Raises:
    SemanticError: If the program contains a semantic violation such as
        a duplicate variable declaration, a reference to an undeclared variable
        or function, a redefinition of a function, an invalid lvalue, a function
        name used as a variable, a variable called as a function, mixed linkage
        on a block-scope variable, or a block-scope ``extern`` declaration with
        an initialiser.
"""

from itertools import count
from typing import NamedTuple

import compiler.syntax.ast as ast
from compiler.syntax.ast.declarations import StorageClass
from compiler.syntax.ast.types import Linkage
from compiler.utils.exceptions import SemanticError


class VariableBinding(NamedTuple):
    """A single entry in the variable map, recording a resolved name and its scope origin.

    Attributes:
        unique_name: The unique internal identifier assigned to the source-level variable
            (e.g. ``".Vx.0"``).  File-scope variables use their original source name.
        from_current_block: ``True`` if this binding was declared in the block currently
            being analysed; ``False`` if it was inherited from an enclosing scope or the
            file scope.  Used by ``resolve_variable_declaration`` to distinguish shadowing
            (permitted) from a same-block duplicate (an error).
        linkage: The linkage classification for this variable.  Local variables carry
            ``Linkage.NONE``; ``static`` variables carry ``Linkage.INTERNAL``; file-scope
            and ``extern``-declared variables carry ``Linkage.EXTERNAL``.
    """

    unique_name: str
    from_current_block: bool
    linkage: Linkage = Linkage.NONE


class FunctionBinding(NamedTuple):
    """A single entry in the function map, tracking whether a definition has been seen.

    Functions are globally scoped and their names are never renamed.  Unlike
    variables, multiple declarations of the same function are permitted; only a
    second *definition* (a ``Function`` node with a non-``None`` body) is an
    error.  All declarations of the same function must agree on parameter count
    and types; a mismatch is a conflict error.

    Attributes:
        has_definition: ``True`` once a definition (body present) has been
            registered for this function name; ``False`` if only forward
            declarations have been seen so far.
        param_types: The ordered tuple of parameter types established by the
            first declaration seen.  Every subsequent declaration must match
            this tuple exactly.
        linkage: The linkage classification for this function.  Functions
            declared without ``static`` carry ``Linkage.EXTERNAL``; those
            declared with ``static`` carry ``Linkage.INTERNAL``.
    """

    has_definition: bool
    param_types: tuple[ast.Type, ...]
    linkage: Linkage = Linkage.EXTERNAL


class SemanticAnalyser:
    """Resolves variable references and validates function declarations across a single AST program.

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
            ``VariableBinding`` values.  Each binding records the unique
            internal name, whether it was declared in the current block, and
            the variable's linkage.  File-scope variables accumulate in this
            map (with ``from_current_block=False``) as top-level ``VarDecl``
            nodes are processed; when a ``FunDecl`` definition is resolved the
            map is snapshotted before entering the body and restored afterward
            so that local variables declared inside the function are not visible
            to subsequent top-level declarations.  Outer-scope bindings are
            retained in the map with ``from_current_block=False`` so that
            variable lookups can still resolve them; when a compound block exits
            the map is restored to its saved snapshot.
        _function_map: Program-wide mapping from function names to
            ``FunctionBinding`` values.  Each binding records whether a
            definition has been seen and the function's linkage
            (``Linkage.INTERNAL`` for ``static`` functions; ``Linkage.EXTERNAL``
            otherwise).  Populated by a pre-pass in ``_resolve_program`` and
            extended when ``FunDecl`` block items are processed.  Never reset
            between functions.
        _label_set: Set of all label names declared in the function currently
            being analysed.  Populated by a first pass before resolution begins
            so that forward ``goto`` targets can be validated.
        _loop_stack: Stack of identifiers for the loops currently enclosing the
            statement being resolved.  Used exclusively by ``Continue``:
            ``continue`` is only valid inside a loop, never a bare switch.
        _break_stack: Stack of identifiers for the loops *and* switches
            currently enclosing the statement being resolved.  The top entry is
            the target for any ``Break`` encountered.  An empty stack indicates
            that ``break`` would be invalid at the current position.
        _switch_stack: Stack of identifiers for the switches currently
            enclosing the statement being resolved.  The top entry is the
            enclosing switch identifier, which is attached to every ``Case``
            and ``Default`` encountered during resolution.  An empty stack
            indicates that ``case``/``default`` appear outside any switch,
            which is a semantic error.
        _case_values_stack: Parallel stack to ``_switch_stack``.  Each entry
            is the set of ``ast.Constant`` nodes already seen by ``Case`` labels
            within the corresponding switch.  Storing the node rather than its
            raw value keeps the check type-independent as the expression language
            grows: equality and hashing are delegated to the frozen dataclass.
            Used to detect duplicate case values within a single switch statement.
        _default_seen_stack: Parallel stack to ``_switch_stack``.  Each entry
            is ``True`` once a ``Default`` label has been encountered within the
            corresponding switch, and ``False`` until then.  Used to enforce the
            C rule that at most one ``default`` clause may appear per switch.
        _current_block_funs: Set of function names declared as block items in
            the innermost enclosing block scope.  Saved and cleared on every
            ``_enter_scope`` call and restored when that scope exits, mirroring
            the ``from_current_block`` flag on ``VariableBinding``.  Used to
            detect within-scope conflicts between a function declaration and a
            variable declaration sharing the same name.
        _file_scope_linkage: Persistent mapping from file-scope names to their
            established linkage.  Written by both file-scope variable declarations
            and block-scope ``extern`` declarations; never reset between functions.
            Enables cross-boundary linkage-conflict detection regardless of
            declaration order.
        _file_scope_definitions: Set of file-scope variable names for which a
            definition (a declaration carrying an initialiser) has already been
            seen.  A second definition of the same name is a constraint violation
            in C.  Tentative definitions (declarations without initialisers) do
            not add entries here and are always permitted to repeat.
    """

    def __init__(self) -> None:
        """Initialise the analyser with fresh counters and empty symbol tables."""
        self._counter: count[int] = count()
        self._loop_counter: count[int] = count()
        self._variable_map: dict[str, VariableBinding] = {}
        self._function_map: dict[str, FunctionBinding] = {}
        self._file_scope_linkage: dict[str, Linkage] = {}
        self._file_scope_definitions: set[str] = set()
        self._current_block_funs: set[str] = set()
        self._label_set: set[str] = set()
        self._loop_stack: list[str] = []
        self._break_stack: list[str] = []
        self._switch_stack: list[str] = []
        self._case_values_stack: list[set[ast.Constant]] = []
        self._default_seen_stack: list[bool] = []

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
            case ast.Switch(body=switch_body):
                self._collect_labels_in_statement(switch_body)
            case ast.Case(statement=inner_stmt) | ast.Default(statement=inner_stmt):
                self._collect_labels_in_statement(inner_stmt)
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

    def resolve_variable_declaration(self, var_decl: ast.VariableDeclaration) -> ast.VariableDeclaration:
        """Resolve a block-scope variable declaration against the current symbol table.

        ``extern`` declarations keep their original source name and are recorded
        with ``Linkage.EXTERNAL``; an initialiser is forbidden.  All other
        declarations (plain or ``static``) are renamed to a unique internal
        identifier.  ``static`` declarations are recorded with
        ``Linkage.INTERNAL``; plain declarations with ``Linkage.NONE``.

        If the same name has already been declared in the *current* block (i.e.
        the existing binding has ``from_current_block=True``), a
        ``SemanticError`` is raised.  When the existing binding has linkage and
        the new declaration does not (or vice versa), the error message
        identifies the linkage conflict specifically; otherwise it reports a
        plain duplicate.

        This method is *not* used for file-scope variable declarations; those
        are handled by ``_resolve_file_scope_variable_declaration``.

        Args:
            var_decl: The ``VariableDeclaration`` AST node to resolve.

        Returns:
            ``var_decl`` unchanged for ``extern`` declarations.  A new
            ``VariableDeclaration`` with a unique internal name (and resolved
            ``init``) for all other declarations.

        Raises:
            SemanticError: If ``var_decl`` is an ``extern`` declaration with an
                initialiser; if the same name was declared with conflicting
                linkage in the same block; if the same name was already declared
                in this block; or if a non-``extern`` name conflicts with a
                same-scope function declaration.
        """
        binding = self._variable_map.get(var_decl.name)
        if var_decl.storage_class == StorageClass.Extern:
            if var_decl.init is not None:
                raise SemanticError(f"Block-scope 'extern' variable '{var_decl.name}' may not have an initialiser.")
            if var_decl.name in self._function_map:
                raise SemanticError(
                    f"'{var_decl.name}' redeclared as a variable but was previously declared as a function."
                )
            if binding is not None and binding.from_current_block:
                if binding.unique_name == var_decl.name:
                    return var_decl  # repeated extern declaration (EXTERNAL or INTERNAL) — permitted
                # Prior declaration was a renamed local or static local — block-scope extern conflicts.
                raise SemanticError(f"'{var_decl.name}' previously declared without linkage in this scope.")
            prior_linkage = self._file_scope_linkage.get(var_decl.name)
            # Inherit whichever linkage was already established; default to EXTERNAL when unseen.
            effective_linkage = prior_linkage if prior_linkage is not None else Linkage.EXTERNAL
            self._file_scope_linkage[var_decl.name] = effective_linkage
            self._variable_map[var_decl.name] = VariableBinding(
                var_decl.name, from_current_block=True, linkage=effective_linkage
            )
            return var_decl
        else:
            if binding is not None and binding.from_current_block:
                if binding.linkage == Linkage.EXTERNAL:
                    raise SemanticError(f"'{var_decl.name}' previously declared with linkage in this scope.")
                raise SemanticError(f"Duplicate variable declaration: {var_decl.name}!")
            if var_decl.name in self._current_block_funs:
                raise SemanticError(
                    f"'{var_decl.name}' redeclared as a variable but was previously declared as a function "
                    "in this scope."
                )
            unique_name = self._make_unique(var_decl.name)
            linkage = Linkage.INTERNAL if var_decl.storage_class == StorageClass.Static else Linkage.NONE
            self._variable_map[var_decl.name] = VariableBinding(unique_name, from_current_block=True, linkage=linkage)
            if var_decl.init is not None and var_decl.storage_class == StorageClass.Static:
                if not isinstance(var_decl.init, ast.Constant):
                    raise SemanticError(
                        f"Initialiser of 'static' variable '{var_decl.name}' must be a constant expression."
                    )
            init = self.resolve_exp(var_decl.init) if var_decl.init is not None else None
            return ast.VariableDeclaration(unique_name, init, var_decl.storage_class)

    def _resolve_file_scope_variable_declaration(self, var_decl: ast.VariableDeclaration) -> ast.VariableDeclaration:
        """Record a file-scope variable declaration in the identifier table.

        File-scope variables are *not* renamed — they keep their original source
        names throughout the compilation pipeline.  The binding is added to
        ``_variable_map`` with ``from_current_block=False`` so that subsequent
        function bodies can reference it.

        Linkage is determined by storage class: ``static`` yields
        ``Linkage.INTERNAL``; everything else (``extern`` or no specifier) yields
        ``Linkage.EXTERNAL``.

        Args:
            var_decl: The ``VariableDeclaration`` AST node from a top-level
                ``VarDecl`` block.

        Returns:
            ``var_decl`` unchanged — file-scope variables are not renamed.

        Raises:
            SemanticError: If ``var_decl.name`` was previously declared as a
                function (conflicting entity kind on the same external name); or
                if the initialiser is not a constant expression (file-scope
                variables have static storage duration and must be initialised
                at compile time).
        """
        if var_decl.name in self._function_map:
            raise SemanticError(
                f"'{var_decl.name}' redeclared as a variable but was previously declared as a function."
            )
        if var_decl.init is not None and not isinstance(var_decl.init, ast.Constant):
            raise SemanticError(f"Initialiser of file-scope variable '{var_decl.name}' must be a constant expression.")
        if var_decl.init is not None:
            if var_decl.name in self._file_scope_definitions:
                raise SemanticError(f"Duplicate definition of file-scope variable '{var_decl.name}'.")
            self._file_scope_definitions.add(var_decl.name)
        prior_linkage = self._file_scope_linkage.get(var_decl.name)
        if var_decl.storage_class == StorageClass.Static:
            new_linkage = Linkage.INTERNAL
        elif var_decl.storage_class == StorageClass.Extern and prior_linkage is not None:
            new_linkage = prior_linkage  # extern inherits whatever linkage was already established
        else:
            new_linkage = Linkage.EXTERNAL
        if prior_linkage is not None and prior_linkage != new_linkage:
            raise SemanticError(
                f"'{var_decl.name}' previously declared with "
                f"{'external' if prior_linkage == Linkage.EXTERNAL else 'internal'} linkage; "
                f"cannot redeclare with {'internal' if new_linkage == Linkage.INTERNAL else 'external'} linkage."
            )
        self._file_scope_linkage[var_decl.name] = new_linkage
        self._variable_map[var_decl.name] = VariableBinding(
            var_decl.name, from_current_block=False, linkage=new_linkage
        )
        return var_decl

    def _validate_params_unique(self, fun: ast.Function) -> None:
        """Verify that no two parameters in a function declaration share the same name.

        Parameter names in a declaration need not be identical to those in the
        matching definition, but within any single declaration or definition they
        must be distinct.  This check applies equally to forward declarations
        (``body is None``) and full definitions.

        Args:
            fun: The ``Function`` node whose parameter list is to be validated.

        Raises:
            SemanticError: If two or more parameters in ``fun.params`` carry the
                same source-level name.
        """
        seen: set[str] = set()
        for param in fun.params:
            if param.name in seen:
                raise SemanticError(f"Duplicate parameter name '{param.name}' in declaration of '{fun.name}'.")
            seen.add(param.name)

    def _register_fun_declaration(self, fun: ast.Function) -> None:
        """Register a function declaration or definition in the program-wide function map.

        Records the function name in ``_function_map``.  A function may be
        declared any number of times; a definition (``fun.body is not None``)
        may appear at most once — a second definition raises ``SemanticError``.

        Args:
            fun: The ``Function`` node whose name is to be registered.

        Raises:
            SemanticError: If ``fun`` is a definition and a definition for the
                same name has already been registered; if the parameter types of
                this declaration conflict with a prior declaration of the same
                name; if ``fun.name`` was already declared as a variable in the
                current block scope; or if ``fun.name`` was previously declared
                as an ``extern`` variable at any scope level (conflicting entity
                kind on the same external name).
        """
        var_binding = self._variable_map.get(fun.name)
        if var_binding is not None and (var_binding.from_current_block or var_binding.linkage == Linkage.EXTERNAL):
            raise SemanticError(f"'{fun.name}' redeclared as a function but was previously declared as a variable.")
        new_param_types = tuple(p.param_type for p in fun.params)
        existing = self._function_map.get(fun.name)
        if fun.storage_class == StorageClass.Static:
            new_linkage = Linkage.INTERNAL
        elif existing is not None:
            new_linkage = existing.linkage  # inherit whatever linkage was already established
        else:
            new_linkage = Linkage.EXTERNAL
        if existing is not None:
            if existing.param_types != new_param_types:
                raise SemanticError(
                    f"Conflicting declarations of '{fun.name}': parameter types do not match prior declaration."
                )
            if existing.linkage != new_linkage:
                raise SemanticError(
                    f"'{fun.name}' previously declared with "
                    f"{'external' if existing.linkage == Linkage.EXTERNAL else 'internal'} linkage; "
                    f"cannot redeclare with {'internal' if new_linkage == Linkage.INTERNAL else 'external'} linkage."
                )
            if fun.body is not None and existing.has_definition:
                raise SemanticError(f"Redefinition of function '{fun.name}'.")
            if fun.body is not None:
                self._function_map[fun.name] = FunctionBinding(
                    has_definition=True, param_types=new_param_types, linkage=new_linkage
                )
        else:
            self._function_map[fun.name] = FunctionBinding(
                has_definition=fun.body is not None, param_types=new_param_types, linkage=new_linkage
            )
        self._current_block_funs.add(fun.name)

    def _enter_scope(self) -> tuple[dict[str, VariableBinding], set[str]]:
        """Open a new nested scope and return snapshots of the current block state.

        Saves a copy of ``_variable_map`` and ``_current_block_funs``, then
        marks every existing variable binding as ``from_current_block=False``
        and clears ``_current_block_funs`` so that the new scope starts with no
        locally-declared names of either kind.

        Returns:
            A ``(saved_variable_map, saved_block_funs)`` pair.  Assign both
            values back to the corresponding attributes once the nested scope
            has been fully processed to restore the enclosing bindings.
        """
        saved_vars = dict(self._variable_map)
        saved_funs = self._current_block_funs
        self._variable_map = {k: v._replace(from_current_block=False) for k, v in self._variable_map.items()}
        self._current_block_funs = set()
        return saved_vars, saved_funs

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
        saved_vars, saved_funs = self._enter_scope()
        result = tuple(self._resolve_block_item(item) for item in body)
        self._variable_map = saved_vars
        self._current_block_funs = saved_funs
        return result

    def resolve_exp(self, e: ast.Exp) -> ast.Exp:
        """Resolve all variable and function references within an expression.

        Recursively walks the expression tree, substituting every ``ast.Var``
        node with a new ``ast.Var`` whose name is the unique internal identifier
        recorded in ``_variable_map``.  Assignment expressions are additionally
        validated to ensure their left-hand side is an ``ast.Var`` (i.e. a
        simple lvalue).  ``FunctionCall`` expressions are validated to ensure
        the callee name appears in ``_function_map``.

        Args:
            e: The expression node to resolve.

        Returns:
            A new expression node structurally identical to ``e`` but with all
            ``ast.Var`` references rewritten to use unique internal names.

        Raises:
            SemanticError: If a ``Var`` node names an identifier present in
                ``_current_block_funs`` or in ``_function_map`` but not
                ``_variable_map`` (function name used as a variable); if a
                variable reference cannot be found in ``_variable_map``
                (undeclared variable); if the left-hand side of an assignment
                is not a plain ``ast.Var`` (invalid lvalue); if a
                ``FunctionCall`` names an identifier present in
                ``_variable_map`` but not shadowed by a block-scope function
                declaration in ``_current_block_funs`` (variable called as a
                function); if a ``FunctionCall`` names a function not present
                in ``_function_map`` (undeclared function); or if the number
                of arguments does not match the function's declared arity.
        """
        match e:
            case ast.Assignment(lvalue=ast.Var() as lvalue_var, rvalue=rvalue):
                return ast.Assignment(self.resolve_exp(lvalue_var), self.resolve_exp(rvalue))
            case ast.Assignment():
                raise SemanticError("Invalid lvalue in assignment expression.")
            case ast.Var(name=name):
                if name in self._current_block_funs:
                    raise SemanticError(f"Function name '{name}' used as a variable.")
                if name in self._variable_map:
                    return ast.Var(self._variable_map[name].unique_name)
                if name in self._function_map:
                    raise SemanticError(f"Function name '{name}' used as a variable.")
                raise SemanticError(f"Undeclared variable {name}!")
            case ast.Constant():
                return e
            case ast.Unary(operator=op, exp=exp):
                return ast.Unary(op, self.resolve_exp(exp))
            case ast.Binary(operator=op, left=left, right=right):
                return ast.Binary(op, self.resolve_exp(left), self.resolve_exp(right))
            case ast.Conditional(condition=cond, exp_true=exp_true, exp_false=exp_false):
                return ast.Conditional(self.resolve_exp(cond), self.resolve_exp(exp_true), self.resolve_exp(exp_false))
            case ast.FunctionCall(name=name, args=args):
                if name in self._variable_map and name not in self._current_block_funs:
                    raise SemanticError(f"Variable '{name}' called as a function.")
                if name not in self._function_map:
                    raise SemanticError(f"Call to undeclared function '{name}'.")
                expected = len(self._function_map[name].param_types)
                if len(args) != expected:
                    raise SemanticError(
                        f"Function '{name}' called with {len(args)} argument(s) but declared with {expected}."
                    )
                return ast.FunctionCall(name, [self.resolve_exp(arg) for arg in args])
            case _:
                raise SemanticError("Unhandled expression type in semantic analyser")

    def _resolve_statement(self, statement: ast.Statement) -> ast.Statement:  # noqa: C901 - ignore complexity warning
        """Resolve all variable references within a statement and assign loop identifiers.

        Recursively resolves any expressions contained in the statement,
        returning a new statement node with all ``ast.Var`` references rewritten
        to their unique internal names.  Additionally:

        * Each loop construct (``While``, ``DoWhile``, ``For``) is assigned a
          fresh unique identifier via ``_make_loop_label`` and pushes that
          identifier onto both ``_loop_stack`` and ``_break_stack``.
        * Each ``Switch`` is assigned a fresh identifier and pushes it onto
          ``_break_stack`` and ``_switch_stack``.
        * ``Break`` is tagged with the top of ``_break_stack`` (the nearest
          enclosing loop or switch).
        * ``Continue`` is tagged with the top of ``_loop_stack`` (the nearest
          enclosing loop; a bare switch does not satisfy ``continue``).
        * ``Case`` and ``Default`` are tagged with the top of ``_switch_stack``
          (the identifier of their enclosing switch).

        Args:
            statement: The statement node to resolve.

        Returns:
            A new statement node with all variable references resolved and all
            control-flow statements tagged with their enclosing construct's identifier.

        Raises:
            SemanticError: If any expression contains an undeclared variable or
                an invalid lvalue; if ``Break`` appears outside a loop or switch;
                if ``Continue`` appears outside a loop; or if ``Case``/``Default``
                appear outside a switch.
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
                if not self._break_stack:
                    raise SemanticError("'break' statement not within a loop or switch.")
                return ast.Break(self._break_stack[-1])
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
                self._break_stack.append(label)
                result = ast.While(self.resolve_exp(condition), self._resolve_statement(body), label)
                self._loop_stack.pop()
                self._break_stack.pop()
                return result
            case ast.DoWhile(body=body, condition=condition):
                label = self._make_loop_label("do")
                self._loop_stack.append(label)
                self._break_stack.append(label)
                result = ast.DoWhile(self._resolve_statement(body), self.resolve_exp(condition), label)
                self._loop_stack.pop()
                self._break_stack.pop()
                return result
            case ast.For(init=init, condition=condition, post=post, body=body):
                saved_vars, saved_funs = self._enter_scope()
                label = self._make_loop_label("for")
                self._loop_stack.append(label)
                self._break_stack.append(label)
                init = self._resolve_for_init(init)
                condition = self._resolve_optional_exp(condition)
                post = self._resolve_optional_exp(post)
                body = self._resolve_statement(body)
                self._loop_stack.pop()
                self._break_stack.pop()
                self._variable_map = saved_vars
                self._current_block_funs = saved_funs
                return ast.For(init, condition, post, body, label)
            case ast.Switch(control=control, body=body):
                label = self._make_loop_label("switch")
                self._break_stack.append(label)
                self._switch_stack.append(label)
                self._case_values_stack.append(set())
                self._default_seen_stack.append(False)
                result = ast.Switch(self.resolve_exp(control), self._resolve_statement(body), label)
                self._break_stack.pop()
                self._switch_stack.pop()
                self._case_values_stack.pop()
                self._default_seen_stack.pop()
                return result
            case ast.Case(value=value, statement=inner):
                if not self._switch_stack:
                    raise SemanticError("'case' label not within a switch statement.")
                if value in self._case_values_stack[-1]:
                    raise SemanticError(f"Duplicate case value {value.value} in switch statement.")
                self._case_values_stack[-1].add(value)
                return ast.Case(value, self._resolve_statement(inner), self._switch_stack[-1])
            case ast.Default(statement=inner):
                if not self._switch_stack:
                    raise SemanticError("'default' label not within a switch statement.")
                if self._default_seen_stack[-1]:
                    raise SemanticError("Multiple 'default' labels in one switch statement.")
                self._default_seen_stack[-1] = True
                return ast.Default(self._resolve_statement(inner), self._switch_stack[-1])
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
        (``InitExp``) and to ``resolve_variable_declaration`` for a declaration
        initialiser (``InitDecl``).

        Args:
            init: The ``ForInit`` node from the parser — either an ``InitExp``
                wrapping an optional expression, or an ``InitDecl`` wrapping a
                variable declaration.

        Returns:
            A new ``ForInit`` node with all variable references resolved.

        Raises:
            SemanticError: If a declaration within ``init`` carries a storage
                class specifier; if a declaration within ``init`` duplicates a
                name already declared in the same scope; or if an expression
                within ``init`` contains an undeclared variable or an invalid
                lvalue.
        """
        match init:
            case ast.InitExp(exp=exp):
                return ast.InitExp(self._resolve_optional_exp(exp))
            case ast.InitDecl(declaration=ast.VarDecl(variable_declaration=var_decl)):
                if var_decl.storage_class is not None:
                    raise SemanticError(
                        f"Variable '{var_decl.name}' in 'for' loop initialiser may not have a storage class specifier."
                    )
                return ast.InitDecl(ast.VarDecl(self.resolve_variable_declaration(var_decl)))
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
        """Resolve a single block item (statement, variable declaration, or function declaration).

        Dispatches to ``resolve_variable_declaration`` for ``ast.D`` items wrapping
        a ``VarDecl``, to ``_register_fun_declaration`` for ``ast.D`` items wrapping a
        ``FunDecl`` (forward declarations only — definitions raise immediately), and to
        ``_resolve_statement`` for ``ast.S`` items.

        Args:
            item: The block item to resolve.

        Returns:
            A new block item with all variable references resolved.

        Raises:
            SemanticError: If the item contains a semantic violation, including a
                ``FunDecl`` whose ``body`` is not ``None`` (nested definition), or
                a ``FunDecl`` with a ``static`` storage class specifier (``static``
                is not permitted on block-scope function declarations).
        """
        match item:
            case ast.D(declaration=ast.VarDecl(variable_declaration=var_decl)):
                return ast.D(ast.VarDecl(self.resolve_variable_declaration(var_decl)))
            case ast.D(declaration=ast.FunDecl(function_declaration=fun)):
                if fun.body is not None:
                    raise SemanticError(f"Nested function definition of '{fun.name}' is not permitted.")
                if fun.storage_class == StorageClass.Static:
                    raise SemanticError(
                        f"Block-scope function declaration '{fun.name}' may not have 'static' storage class."
                    )
                self._validate_params_unique(fun)
                self._register_fun_declaration(fun)
                return item
            case ast.S(statement=statement):
                return ast.S(self._resolve_statement(statement))
            case _:
                raise SemanticError("Unhandled block item type in semantic analyser")

    def _resolve_function(self, function: ast.Function) -> ast.Function:
        """Resolve all variable references within a function definition.

        Validates that parameter names are unique (via ``_validate_params_unique``)
        for both declarations and definitions.  If ``function.body`` is ``None``
        this is a forward declaration and is returned immediately after the
        parameter check (the name has already been registered by the pre-pass in
        ``_resolve_program``).  Otherwise:

        1. Resets per-function state; marks all entries in ``_variable_map`` as
           ``from_current_block=False`` so that file-scope variables accumulated
           in ``_resolve_program`` are visible inside the function body.
        2. Adds each parameter to ``_variable_map`` under a unique internal name.
        3. Collects all labels in a first pass.
        4. Resolves every block item in the body.

        Args:
            function: The ``Function`` AST node to resolve.

        Returns:
            A new ``Function`` node with all variable references resolved,
            parameter names replaced by unique internal identifiers, and
            all declared variable names replaced by unique internal identifiers.
            Returns ``function`` unchanged if ``body`` is ``None``.

        Raises:
            SemanticError: If the function body contains any semantic violation,
                or if two parameters share the same name.
        """
        self._validate_params_unique(function)
        if function.body is None:
            return function

        self._variable_map = {k: v._replace(from_current_block=False) for k, v in self._variable_map.items()}
        self._current_block_funs = set()
        self._label_set = set()
        self._loop_stack = []
        self._break_stack = []
        self._switch_stack = []
        self._case_values_stack = []
        self._default_seen_stack = []

        resolved_params: list[ast.Parameter] = []
        for param in function.params:
            resolved_var = self.resolve_variable_declaration(ast.VariableDeclaration(param.name, None))
            resolved_params.append(ast.Parameter(param_type=param.param_type, name=resolved_var.name))

        self._collect_labels(function.body)
        return ast.Function(
            name=function.name,
            params=tuple(resolved_params),
            body=tuple(self._resolve_block_item(item) for item in function.body),
        )

    def _resolve_program(self, program: ast.Program) -> ast.Program:
        """Perform semantic analysis on a parsed program.

        Processes top-level declarations in source order.  For each ``FunDecl``,
        ``_register_fun_declaration`` is called first (making the name visible to
        the bodies that follow), then ``_resolve_function`` resolves the body if
        present.  For each ``VarDecl``, ``_resolve_file_scope_variable_declaration``
        validates linkage consistency and records the variable in
        ``_variable_map`` and ``_file_scope_linkage`` so that subsequent function
        bodies can reference it.

        A function must therefore be declared or defined before any subsequent
        function body may call it; mutual recursion requires a forward declaration.
        A file-scope variable must be declared before any function body that
        references it.

        Args:
            program: The ``Program`` AST node produced by the parser.

        Returns:
            A semantically validated ``Program`` node with uniquely renamed
            variable references throughout.

        Raises:
            SemanticError: If the program contains any semantic violation such
                as a duplicate variable declaration, a reference to an undeclared
                variable or function, a redefinition of a function, a call to a
                function whose declaration has not yet appeared in source order,
                conflicting linkage on a file-scope variable, or a duplicate
                file-scope definition.
        """
        resolved: list[ast.Declaration] = []
        for decl in program.declarations:
            match decl:
                case ast.FunDecl(function_declaration=fn):
                    file_scope_state = dict(self._variable_map)
                    self._register_fun_declaration(fn)
                    resolved.append(ast.FunDecl(self._resolve_function(fn)))
                    self._variable_map = file_scope_state
                    self._current_block_funs = set()
                case ast.VarDecl(variable_declaration=var_decl):
                    resolved.append(ast.VarDecl(self._resolve_file_scope_variable_declaration(var_decl)))
                case _:
                    raise SemanticError("Unhandled top-level declaration type in semantic analyser")
        return ast.Program(declarations=tuple(resolved))


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
