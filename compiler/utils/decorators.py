import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def run_or_raise(
    exception: type[Exception] | Callable[[R], Exception],
    expected: set[R] | None = None,
    message: str | Callable[[R], str] | None = None,
    ok: Callable[[R], bool] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that raises an exception when the wrapped function returns an unexpected value.

    Args:
        exception: Either an exception type or a factory callable ``(result) -> Exception``.
            When a factory is provided, it receives the return value and must return a
            ready-to-raise exception instance — ``message`` is ignored in that case.
            When an exception type is provided, ``message`` controls how it is constructed.
        expected: The set of values considered successful. Defaults to ``{0}`` when
            not provided. Ignored when ``ok`` is provided.
        ok: Optional predicate ``(result) -> bool``. When provided, takes precedence
            over ``expected`` and the default ``result == 0`` check.
        message: Optional exception message (ignored when ``exception`` is a factory).
            A plain string is used as-is; a callable receives the actual return value
            and must return a string. When omitted the exception is raised with no message.

    Returns:
        A decorator that wraps a callable, checks its return value, and raises
        ``exception`` if the value is not in ``expected``.

    Raises:
        exception: When the wrapped function returns a value not in ``expected``.

    Example:
        >>> @run_or_raise(RuntimeError, message=lambda code: f"exited with code {code}")
        ... def compile() -> int:
        ...     return 1
        >>> compile()
        Traceback (most recent call last):
            ...
        RuntimeError: exited with code 1
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            result = func(*args, **kwargs)
            is_ok = ok(result) if ok is not None else (result in expected if expected is not None else result == 0)
            if not is_ok:
                if not isinstance(exception, type):
                    raise exception(result)
                elif callable(message):
                    raise exception(message(result))
                elif message is not None:
                    raise exception(message)
                else:
                    raise exception()
            return result

        return wrapper

    return decorator
