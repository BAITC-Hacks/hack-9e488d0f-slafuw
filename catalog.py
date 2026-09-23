"""Load the canonical JSONL once; hash the exact bytes for reproducibility."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from .models import Profile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "raw" / "hackathon-dataset-anonymized.jsonl"


@dataclass(frozen=True)
class Catalog:
    profiles: tuple[Profile, ...]
    sha256: str


def load_catalog(path=DEFAULT_DATASET):
    content = Path(path).read_bytes()
    profiles = []
    for number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            profiles.append(Profile.parse(json.loads(line)))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Датасет, строка {number}: {exc}") from exc
    if not profiles:
        raise ValueError("Датасет пуст")
    if len({p.id for p in profiles}) != len(profiles):
        raise ValueError("Датасет: повторяющиеся id")
    return Catalog(tuple(sorted(profiles, key=lambda p: p.id)), sha256(content).hexdigest())
