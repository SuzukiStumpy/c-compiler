import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def preprocess_c_file(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], Path]:
    """Preprocesses a fixture .c file with GCC and returns the path to the .i output.

    Results are cached per session so each fixture file is only preprocessed once.
    The path passed to GCC is absolute so linemarkers contain absolute paths.
    """
    tmp = tmp_path_factory.mktemp("preprocessed")
    cache: dict[Path, Path] = {}

    def _run(relative_path: str) -> Path:
        src = (FIXTURES_DIR / relative_path).resolve()
        if src not in cache:
            out = tmp / src.with_suffix(".i").name
            result = subprocess.run(
                ["gcc", "-E", str(src), "-o", str(out)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"GCC preprocessing failed:\n{result.stderr}"
            cache[src] = out
        return cache[src]

    return _run
