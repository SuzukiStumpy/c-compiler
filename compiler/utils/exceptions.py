"""Custom exceptions we raise as part of the compiler"""

from subprocess import CalledProcessError


class SubprocessError(CalledProcessError):
    """Enhances the CalledProcessError so we also capture the stdout/stderr from the failed process"""

    def __str__(self) -> str:
        base = super().__str__()
        output = self.output or self.stderr
        return f"{base}\n{output.strip()}" if output else base
