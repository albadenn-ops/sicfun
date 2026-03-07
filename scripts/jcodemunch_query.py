import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_JCODEMUNCH_SRC = REPO_ROOT / "third_party" / "jcodemunch_mcp" / "src"

if not VENDORED_JCODEMUNCH_SRC.exists():
    raise SystemExit(f"Missing vendored jCodeMunch source at {VENDORED_JCODEMUNCH_SRC}")

sys.path.insert(0, str(VENDORED_JCODEMUNCH_SRC))

from jcodemunch_mcp.tools.get_file_tree import get_file_tree
from jcodemunch_mcp.tools.get_file_outline import get_file_outline
from jcodemunch_mcp.tools.list_repos import list_repos
from jcodemunch_mcp.tools.get_repo_outline import get_repo_outline
from jcodemunch_mcp.tools.get_symbol import get_symbol
from jcodemunch_mcp.tools.index_folder import index_folder
from jcodemunch_mcp.tools.search_symbols import search_symbols
from jcodemunch_mcp.tools.search_text import search_text

DEFAULT_STORAGE = REPO_ROOT / ".jcodemunch-index"
DEFAULT_INDEX_IGNORES = [
    ".venv-tools/",
    ".github-tools/",
    ".tool-cache/",
    "data/",
    "dist/",
    ".idea/",
    "third_party/",
]


def default_repo(path_text: str | None = None) -> str:
    target = Path(path_text).expanduser().resolve() if path_text else REPO_ROOT
    return f"local/{target.name}"


def print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2))


def add_shared_repo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo",
        default=default_repo(),
        help=f"Indexed repo id (default: {default_repo()})",
    )
    parser.add_argument(
        "--storage-path",
        default=str(DEFAULT_STORAGE),
        help=f"Index storage directory (default: {DEFAULT_STORAGE})",
    )


def handle_index_folder(args: argparse.Namespace) -> int:
    ignore_patterns = [*DEFAULT_INDEX_IGNORES]
    if args.extra_ignore_pattern:
        ignore_patterns.extend(args.extra_ignore_pattern)

    result = index_folder(
        path=args.path,
        use_ai_summaries=args.ai_summaries,
        storage_path=args.storage_path,
        extra_ignore_patterns=ignore_patterns,
        follow_symlinks=args.follow_symlinks,
        incremental=not args.full,
    )
    print_json(result)
    return 0 if result.get("success", True) else 1


def handle_repo_outline(args: argparse.Namespace) -> int:
    result = get_repo_outline(
        repo=args.repo,
        storage_path=args.storage_path,
    )
    print_json(result)
    return 0 if "error" not in result else 1


def handle_file_outline(args: argparse.Namespace) -> int:
    result = get_file_outline(
        repo=args.repo,
        file_path=args.file_path,
        storage_path=args.storage_path,
    )
    print_json(result)
    return 0 if "error" not in result else 1


def handle_file_tree(args: argparse.Namespace) -> int:
    result = get_file_tree(
        repo=args.repo,
        path_prefix=args.path_prefix,
        include_summaries=args.include_summaries,
        storage_path=args.storage_path,
    )
    print_json(result)
    return 0 if "error" not in result else 1


def handle_list_repos(args: argparse.Namespace) -> int:
    result = list_repos(
        storage_path=args.storage_path,
    )
    print_json(result)
    return 0 if "error" not in result else 1


def handle_search_symbols(args: argparse.Namespace) -> int:
    result = search_symbols(
        repo=args.repo,
        query=args.query,
        kind=args.kind,
        file_pattern=args.file_pattern,
        language=args.language,
        max_results=args.max_results,
        storage_path=args.storage_path,
    )
    print_json(result)
    return 0 if "error" not in result else 1


def handle_search_text(args: argparse.Namespace) -> int:
    result = search_text(
        repo=args.repo,
        query=args.query,
        file_pattern=args.file_pattern,
        max_results=args.max_results,
        storage_path=args.storage_path,
    )
    print_json(result)
    return 0 if "error" not in result else 1


def handle_get_symbol(args: argparse.Namespace) -> int:
    result = get_symbol(
        repo=args.repo,
        symbol_id=args.symbol_id,
        verify=args.verify,
        context_lines=args.context_lines,
        storage_path=args.storage_path,
    )
    print_json(result)
    return 0 if "error" not in result else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Direct wrapper around selected jCodeMunch tools for this repo."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "index-folder",
        help="Index a local folder into repo-local jCodeMunch storage.",
    )
    index_parser.add_argument(
        "path",
        nargs="?",
        default=str(REPO_ROOT),
        help=f"Folder to index (default: {REPO_ROOT})",
    )
    index_parser.add_argument(
        "--storage-path",
        default=str(DEFAULT_STORAGE),
        help=f"Index storage directory (default: {DEFAULT_STORAGE})",
    )
    index_parser.add_argument(
        "--ai-summaries",
        action="store_true",
        help="Enable AI-generated summaries if provider credentials are configured.",
    )
    index_parser.add_argument(
        "--extra-ignore-pattern",
        action="append",
        help="Additional gitignore-style path pattern to exclude. May be repeated.",
    )
    index_parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow symlinks during indexing.",
    )
    index_parser.add_argument(
        "--full",
        action="store_true",
        help="Disable incremental mode and rebuild from scratch.",
    )
    index_parser.set_defaults(func=handle_index_folder)

    outline_parser = subparsers.add_parser(
        "repo-outline",
        help="Show a high-level outline for an indexed repo.",
    )
    add_shared_repo_args(outline_parser)
    outline_parser.set_defaults(func=handle_repo_outline)

    file_outline_parser = subparsers.add_parser(
        "file-outline",
        help="Show the indexed symbol outline for one file.",
    )
    add_shared_repo_args(file_outline_parser)
    file_outline_parser.add_argument("file_path", help="Repo-relative file path.")
    file_outline_parser.set_defaults(func=handle_file_outline)

    file_tree_parser = subparsers.add_parser(
        "file-tree",
        help="Show the indexed file tree for the repo or a prefix.",
    )
    add_shared_repo_args(file_tree_parser)
    file_tree_parser.add_argument(
        "--path-prefix",
        default="",
        help="Optional repo-relative path prefix filter.",
    )
    file_tree_parser.add_argument(
        "--include-summaries",
        action="store_true",
        help="Include stored file summaries in file nodes.",
    )
    file_tree_parser.set_defaults(func=handle_file_tree)

    list_repos_parser = subparsers.add_parser(
        "list-repos",
        help="List repositories indexed in the local storage path.",
    )
    list_repos_parser.add_argument(
        "--storage-path",
        default=str(DEFAULT_STORAGE),
        help=f"Index storage directory (default: {DEFAULT_STORAGE})",
    )
    list_repos_parser.set_defaults(func=handle_list_repos)

    search_parser = subparsers.add_parser(
        "search-symbols",
        help="Search indexed symbols.",
    )
    add_shared_repo_args(search_parser)
    search_parser.add_argument("query", help="Symbol search text.")
    search_parser.add_argument("--kind", help="Optional symbol kind filter.")
    search_parser.add_argument("--file-pattern", help="Optional file glob filter.")
    search_parser.add_argument("--language", help="Optional language filter.")
    search_parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum results to return.",
    )
    search_parser.set_defaults(func=handle_search_symbols)

    search_text_parser = subparsers.add_parser(
        "search-text",
        help="Search indexed raw file text.",
    )
    add_shared_repo_args(search_text_parser)
    search_text_parser.add_argument("query", help="Text search query.")
    search_text_parser.add_argument("--file-pattern", help="Optional file glob filter.")
    search_text_parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Maximum matching lines to return.",
    )
    search_text_parser.set_defaults(func=handle_search_text)

    symbol_parser = subparsers.add_parser(
        "get-symbol",
        help="Retrieve full source for a single symbol id.",
    )
    add_shared_repo_args(symbol_parser)
    symbol_parser.add_argument("symbol_id", help="Stable symbol id to fetch.")
    symbol_parser.add_argument(
        "--context-lines",
        type=int,
        default=0,
        help="Include surrounding source lines.",
    )
    symbol_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the retrieved source hash against stored metadata.",
    )
    symbol_parser.set_defaults(func=handle_get_symbol)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
