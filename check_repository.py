"""Check local Markdown links and reproducible data/response artifacts."""

import json
import re

from eventmatch.catalog import ROOT, load_catalog
from eventmatch.engine import recommend
from .audit import audit


def main():
    errors = []
    link_count = 0
    for source in ROOT.rglob("*.md"):
        content = source.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
            path = target.split("#", 1)[0]
            if not path or "://" in path or path.startswith("mailto:"):
                continue
            link_count += 1
            if not (source.parent / path).exists():
                errors.append(f"{source.relative_to(ROOT)}: broken link {target}")
    catalog = load_catalog()
    saved_audit = json.loads((ROOT / "data" / "reports" / "audit.json").read_text(encoding="utf-8"))
    if audit(catalog) != saved_audit:
        errors.append("Audit report is stale")
    snapshots = 0
    for source in sorted((ROOT / "examples").glob("*.json")):
        actual = recommend(catalog, json.loads(source.read_text(encoding="utf-8")))
        target = ROOT / "examples" / "responses" / source.name
        if not target.exists() or json.loads(target.read_text(encoding="utf-8")) != actual:
            errors.append(f"Stale or missing response: {target.relative_to(ROOT)}")
        snapshots += 1
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"OK: {link_count} local document links; audit; {snapshots} exact response snapshots")


if __name__ == "__main__":
    main()
