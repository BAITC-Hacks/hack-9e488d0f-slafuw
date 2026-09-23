"""Measure this reference core locally; not an HTTP/concurrent production SLA."""

import json
import math
import platform
import statistics
from time import perf_counter

from eventmatch.catalog import ROOT, load_catalog
from eventmatch.engine import recommend


def main():
    start = perf_counter()
    catalog = load_catalog()
    load_ms = (perf_counter() - start) * 1000
    requests = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted((ROOT / "examples").glob("*.json"))]
    for request in requests:
        recommend(catalog, request)
    timings = []
    for _ in range(50):
        for request in requests:
            start = perf_counter()
            recommend(catalog, request)
            timings.append((perf_counter() - start) * 1000)
    timings.sort()
    result = {
        "scope": "warm in-process core; seven demo queries, fifty repetitions; excludes HTTP/browser",
        "python": platform.python_version(), "platform": platform.system(),
        "dataset_sha256": catalog.sha256, "requests": len(timings),
        "catalog_load_ms": round(load_ms, 3),
        "p50_ms": round(statistics.median(timings), 3),
        "p95_ms": round(timings[math.ceil(0.95 * len(timings)) - 1], 3),
        "max_ms": round(max(timings), 3),
    }
    target = ROOT / "data" / "reports" / "benchmark.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
