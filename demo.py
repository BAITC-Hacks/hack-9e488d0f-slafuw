"""Run live requests, assert business outcomes and optionally write exact snapshots."""

import argparse
import json
from time import perf_counter

from eventmatch.catalog import ROOT, load_catalog
from eventmatch.engine import recommend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Обновить снимки в examples/responses/")
    args = parser.parse_args()
    catalog = load_catalog()
    results = {}
    for source in sorted((ROOT / "examples").glob("*.json")):
        request = json.loads(source.read_text(encoding="utf-8"))
        started = perf_counter()
        result = recommend(catalog, request)
        elapsed_ms = (perf_counter() - started) * 1000
        assert result == recommend(catalog, request), "Недетерминированный ответ"
        results[source.stem] = result
        if args.write:
            target = ROOT / "examples" / "responses" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        print(f"\n{source.stem}: {result['status']} ({elapsed_ms:.2f} ms)")
        print(result["summary"])
        for card in result["cards"]:
            print(f"  {card['id']} / {card['name']}: {card['explanation']}")
        for suggestion in result["suggestions"]:
            print("  Подсказка: " + suggestion["message"])
    assert results["01-dense"]["diagnostics"]["eligible_count"] > 3
    assert [c["id"] for c in results["01-dense"]["cards"]] != [
        c["id"] for c in results["02-other-date"]["cards"]]
    assert 0 < len(results["03-rare"]["cards"]) < 3
    assert results["04-busy"]["status"] == "no_match"
    assert results["04-busy"]["diagnostics"]["blocked_only_by_date"] > 0
    assert results["05-no-category"]["status"] == "no_category_in_city"
    assert results["06-venue"]["status"] == "matched"
    assert results["07-budget"]["status"] == "no_match"
    print("\nВсе демоусловия проверены на текущем датасете.")


if __name__ == "__main__":
    main()
