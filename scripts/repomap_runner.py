import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_REPOMAPPER_SRC = REPO_ROOT / "third_party" / "repomapper"
REPOMAPPER_CACHE = REPO_ROOT / ".tool-cache" / "repomapper"


def main() -> int:
    repomapper_src = VENDORED_REPOMAPPER_SRC
    if not (repomapper_src / "repomap.py").exists():
        print(f"Missing vendored RepoMapper source at {repomapper_src}.", file=sys.stderr)
        return 1

    repomapper_main = repomapper_src / "repomap.py"

    REPOMAPPER_CACHE.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    try:
        os.chdir(REPOMAPPER_CACHE)
        sys.path.insert(0, str(repomapper_src))
        import repomap  # type: ignore
    finally:
        os.chdir(original_cwd)

    sys.argv[0] = str(repomapper_main)
    repomap.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
