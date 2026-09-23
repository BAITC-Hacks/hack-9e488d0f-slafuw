"""Convert organizer CSV to typed JSONL, without fabricating profiles or calendars."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

from eventmatch.catalog import DEFAULT_DATASET, ROOT
from eventmatch.models import Profile


def boolean(raw):
    if raw not in ("True", "False"):
        raise ValueError(f"Некорректный boolean CSV: {raw!r}")
    return raw == "True"


def convert(path):
    records = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            for key in ("categories", "event_formats", "languages", "busy_dates"):
                row[key] = row[key].split("|") if row[key] else []
            for key in ("synthetic", "city_imputed", "price_imputed"):
                row[key] = boolean(row[key])
            row["price_from_kzt"] = int(row["price_from_kzt"])
            row["max_hours"] = float(row["max_hours"]) if row["max_hours"] else None
            Profile.parse(row)
            records.append(row)
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("Пустой датасет или повторяющиеся id")
    return sorted(records, key=lambda row: row["id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    records = convert(args.source)
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records)
    DEFAULT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_DATASET.write_bytes(content.encode("utf-8"))
    manifest = {
        "source_filename": args.source.name,
        "source_format": "organizer CSV, pipe-separated array fields",
        "source_sha256": sha256(args.source.read_bytes()).hexdigest(),
        "canonical_file": DEFAULT_DATASET.relative_to(ROOT).as_posix(),
        "canonical_sha256": sha256(content.encode("utf-8")).hexdigest(),
        "profile_count": len(records),
        "team_added_profiles": 0,
        "transformations": ["CSV arrays split on |", "strict boolean parsing",
                            "numeric price and max_hours; empty max_hours -> null",
                            "sort by id; UTF-8 JSONL with LF; descriptions preserved"],
    }
    (ROOT / "data" / "manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
