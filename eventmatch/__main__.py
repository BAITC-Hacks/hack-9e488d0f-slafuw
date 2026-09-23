import argparse
import json
from pathlib import Path
import sys

from .catalog import load_catalog
from .engine import recommend


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="EventMatch #79-lite — подбор с основаниями")
    commands = parser.add_subparsers(dest="command", required=True)
    rec = commands.add_parser("recommend", help="Подбор по JSON-файлу запроса")
    rec.add_argument("--request", type=Path, required=True)
    server = commands.add_parser("serve", help="Локальный HTTP API и демоинтерфейс")
    server.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        catalog = load_catalog()
        if args.command == "serve":
            from .server import serve
            serve(catalog, args.port)
        else:
            request = json.loads(args.request.read_text(encoding="utf-8-sig"))
            print(json.dumps(recommend(catalog, request), ensure_ascii=False, indent=2))
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": "invalid_input", "message": str(exc)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
