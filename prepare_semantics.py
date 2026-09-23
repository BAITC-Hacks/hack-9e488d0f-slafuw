"""Run once with NVIDIA_API_KEY; serving pins the resulting artifact at startup."""

import argparse
import json
from pathlib import Path

from eventmatch.catalog import load_catalog
from eventmatch.config import load_local_env
from eventmatch.providers import NvidiaClient
from eventmatch.semantic import prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/nvidia-artifact.json"))
    args = parser.parse_args()
    if not args.output.parent.is_dir():
        parser.error("Родительская папка output должна существовать")
    load_local_env()
    artifact = prepare(load_catalog(), NvidiaClient())
    # A failed API call never overwrites a previously prepared artifact.
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(artifact, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"artifact_sha256": artifact["artifact_sha256"],
                      "api_calls": len(artifact["payload"]["api_calls"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
