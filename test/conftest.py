import subprocess
from collections.abc import Callable, Generator
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"

_ARTEFACT_SUFFIXES = frozenset({".i", ".s"})


@pytest.fixture(scope="session", autouse=True)
def _cleanup_compiler_artefacts() -> Generator[None, None, None]:
    """Remove compiler-generated artefacts from fixture directories after the test session.

    The compiler places ``.i`` (preprocessed), ``.s`` (assembly), and executable
    files (no extension, same stem as the ``.c`` source) next to the source file.
    This fixture deletes those files so the working tree stays clean between runs.

    Yields:
        Nothing; cleanup runs in the teardown phase after the session ends.
    """
    yield
    for directory in (FIXTURES_DIR, FIXTURES_DIR / "valid", FIXTURES_DIR / "invalid"):
        if not directory.is_dir():
            continue
        c_stems = {p.stem for p in directory.glob("*.c")}
        for path in directory.iterdir():
            if path.suffix in _ARTEFACT_SUFFIXES or (path.suffix == "" and path.stem in c_stems):
                path.unlink(missing_ok=True)


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
