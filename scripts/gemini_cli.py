import argparse
import os
import sys
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    print(
        "Error: missing dependency. Install it with: pip install google-generativeai",
        file=sys.stderr,
    )
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gemini-1.5-flash"
DEFAULT_SHARED_SYSTEM_FILE = REPO_ROOT / "AI_ENTRYPOINT.md"
DEFAULT_SYSTEM_FILE = REPO_ROOT / "GEMINI.md"
ENV_FILE_CANDIDATES = (
    REPO_ROOT / ".env",
    REPO_ROOT / ".env.local",
    Path(__file__).resolve().with_name(".env"),
)
EXIT_WORDS = {"salir", "exit", "quit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemini CLI helper for repo-aware one-shot prompts and chat."
    )
    parser.add_argument(
        "--prompt",
        help="Send one prompt and exit instead of starting the interactive chat loop.",
    )
    parser.add_argument(
        "--context-file",
        action="append",
        default=[],
        help="Inject a file into the prompt context. Repeat for multiple files.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--system-file",
        default=str(DEFAULT_SYSTEM_FILE) if DEFAULT_SYSTEM_FILE.exists() else None,
        help="Optional provider overlay file. Defaults to GEMINI.md when it exists and is prepended by AI_ENTRYPOINT.md when present.",
    )
    parser.add_argument(
        "--no-system-file",
        action="store_true",
        help="Disable loading the default system prompt file.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output.",
    )
    return parser.parse_args()


def load_env_value(path: Path, name: str) -> str | None:
    if not path.exists():
        return None

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        return value.strip().strip("'\"")

    return None


def get_api_key() -> str | None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return api_key

    for candidate in ENV_FILE_CANDIDATES:
        api_key = load_env_value(candidate, "GEMINI_API_KEY")
        if api_key:
            return api_key

    return None


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def repo_relative_label(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def render_context_block(path: Path) -> str:
    content = read_text_file(path)
    language = path.suffix.lstrip(".") or "text"
    label = repo_relative_label(path)
    return f"File: {label}\n```{language}\n{content}\n```"


def build_prompt(task: str, context_files: list[str]) -> str:
    sections = []
    if context_files:
        sections.append("Repository context:")
        for raw_path in context_files:
            path = resolve_path(raw_path)
            if not path.exists():
                raise FileNotFoundError(f"Context file not found: {raw_path}")
            sections.append(render_context_block(path))

    sections.append("Task:")
    sections.append(task)
    return "\n\n".join(sections)


def load_system_instruction(args: argparse.Namespace) -> str | None:
    if args.no_system_file:
        return None

    sections: list[str] = []

    if DEFAULT_SHARED_SYSTEM_FILE.exists():
        sections.append(
            "Shared repository contract "
            f"({repo_relative_label(DEFAULT_SHARED_SYSTEM_FILE)}):\n\n"
            f"{read_text_file(DEFAULT_SHARED_SYSTEM_FILE)}"
        )

    if args.system_file:
        path = resolve_path(args.system_file)
        if not path.exists():
            raise FileNotFoundError(f"System file not found: {path}")
        if path != DEFAULT_SHARED_SYSTEM_FILE:
            sections.append(
                f"Provider role overlay ({repo_relative_label(path)}):\n\n"
                f"{read_text_file(path)}"
            )

    if not sections:
        return None

    return "\n\n".join(sections)


def create_model(args: argparse.Namespace):
    api_key = get_api_key()
    if not api_key:
        print("Error: GEMINI_API_KEY is not configured.", file=sys.stderr)
        print("Set it in the environment or add it to .env / .env.local.", file=sys.stderr)
        print("PowerShell: $env:GEMINI_API_KEY='your_key'", file=sys.stderr)
        sys.exit(1)

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        args.model,
        system_instruction=load_system_instruction(args),
    )


def write_stream(response) -> None:
    for chunk in response:
        text = getattr(chunk, "text", "")
        if text:
            print(text, end="", flush=True)
    print()


def send_one_shot(model, prompt: str, stream: bool) -> None:
    response = model.generate_content(prompt, stream=stream)
    if stream:
        write_stream(response)
        return

    print(response.text)


def run_interactive(model) -> int:
    chat = model.start_chat(history=[])

    print("=" * 60)
    print("SICFUN Gemini CLI")
    print("  - Type a prompt normally.")
    print("  - Use '/read <path>' or '/leer <path>' to inject a file.")
    print("  - Type 'salir', 'exit', or 'quit' to stop.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.lower() in EXIT_WORDS:
                print("Bye.")
                return 0

            if user_input.startswith("/read ") or user_input.startswith("/leer "):
                filepath = user_input.split(" ", 1)[1].strip()
                prompt = build_prompt(
                    "Confirm you read the file and summarize what it does.",
                    [filepath],
                )
            else:
                prompt = user_input

            print("\nGemini: ", end="", flush=True)
            write_stream(chat.send_message(prompt, stream=True))
        except FileNotFoundError as exc:
            print(f"\nError: {exc}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nBye.")
            return 0
        except Exception as exc:
            print(f"\nError: {exc}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    model = create_model(args)

    if args.prompt:
        try:
            prompt = build_prompt(args.prompt, args.context_file)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        send_one_shot(model, prompt, stream=not args.no_stream)
        return 0

    return run_interactive(model)


if __name__ == "__main__":
    raise SystemExit(main())
