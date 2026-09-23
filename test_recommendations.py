from dataclasses import replace
from datetime import timedelta
import json
import os
import subprocess
import sys
import unittest

from eventmatch.catalog import Catalog, ROOT, load_catalog
from eventmatch.engine import recommend
from eventmatch.models import END, START, Request


class RecommendationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.by_id = {p.id: p for p in cls.catalog.profiles}
        cls.base = json.loads((ROOT / "examples" / "01-dense.json").read_text(encoding="utf-8"))

    def query(self, **changes):
        return {**self.base, **changes}

    def test_dataset_provenance(self):
        manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(self.catalog.sha256, manifest["canonical_sha256"])
        self.assertEqual(len(self.catalog.profiles), 66)
        self.assertEqual(sum(p.synthetic for p in self.catalog.profiles), 13)
        self.assertEqual(manifest["team_added_profiles"], 0)

    def test_dense_ranking_has_real_selection(self):
        result = recommend(self.catalog, self.base)
        self.assertEqual(result["diagnostics"]["eligible_count"], 7)
        self.assertEqual([c["id"] for c in result["cards"]], ["HK-88430", "HK-44733", "HK-75012"])

    def test_date_pair_has_causal_busy_evidence(self):
        first = recommend(self.catalog, self.base)
        second = recommend(self.catalog, self.query(event_date="2026-10-13"))
        self.assertNotEqual([c["id"] for c in first["cards"]], [c["id"] for c in second["cards"]])
        self.assertIn({"id": "HK-27222", "reasons": ["busy"]}, first["diagnostics"]["rejected"])
        self.assertIn("Сон Гоку", first["summary"])
        self.assertIn("HK-27222", [c["id"] for c in second["cards"]])

    def test_all_100_dates_obey_constraints_and_count_accounting(self):
        # Independent direct-field oracle instead of reusing the engine's failures().
        for offset in range((END - START).days + 1):
            query = self.query(event_date=str(START + timedelta(days=offset)))
            result = recommend(self.catalog, query)
            expected = [p for p in self.catalog.profiles
                        if p.city == query["city"] and query["category"] in p.categories
                        and query["event_date"] not in p.busy_dates
                        and p.price_from_kzt <= query["budget_kzt"]
                        and query["event_format"] in p.event_formats
                        and query["language"] in p.languages
                        and (p.max_hours is None or p.max_hours >= query["duration_hours"])]
            with self.subTest(date=query["event_date"]):
                self.assertEqual(result["diagnostics"]["eligible_count"], len(expected))
                self.assertEqual(len(result["cards"]), min(3, len(expected)))
                self.assertTrue({c["id"] for c in result["cards"]} <= {p.id for p in expected})
                d = result["diagnostics"]
                self.assertEqual(d["eligible_count"] + sum(d["primary_exclusions"].values()), d["cohort_count"])

    def test_rare_null_hours_and_provenance_are_visible(self):
        result = recommend(self.catalog, self.query(category="Флорист", event_format="свадьба",
                                                    budget_kzt=300000, duration_hours=8))
        self.assertEqual(len(result["cards"]), 2)
        self.assertIn("всего 2 профиля", result["summary"])
        self.assertEqual({c["provenance"]["synthetic"] for c in result["cards"]}, {True, False})
        self.assertTrue(all("неприменимо" in c["explanation"] for c in result["cards"]))

    def test_absence_is_not_constraints_failure(self):
        result = recommend(self.catalog, self.query(city="Астана", category="Декоратор"))
        self.assertEqual(result["status"], "no_category_in_city")
        self.assertEqual(result["diagnostics"]["cohort_count"], 0)
        self.assertEqual(result["cards"], [])
        self.assertIn("нет категории", result["summary"])

    def test_no_match_explains_actual_availability(self):
        result = recommend(self.catalog, self.query(event_date="2026-12-12", language="английский"))
        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["cards"], [])
        self.assertEqual(result["diagnostics"]["eligible_ignoring_date"], 3)
        self.assertEqual(result["diagnostics"]["blocked_only_by_date"], 3)

    def test_budget_boundary(self):
        result = recommend(self.catalog, self.query(budget_kzt=500000))
        self.assertEqual([c["id"] for c in result["cards"]], ["HK-88430"])
        self.assertEqual(recommend(self.catalog, self.query(budget_kzt=499999))["status"], "no_match")

    def test_duration_boundary_and_language(self):
        result = recommend(self.catalog, self.query(duration_hours=6.5, language="английский"))
        self.assertEqual([c["id"] for c in result["cards"]], ["HK-35215"])
        self.assertGreater(result["diagnostics"]["all_exclusions"]["duration"], 0)
        self.assertGreater(result["diagnostics"]["all_exclusions"]["language"], 0)

    def test_venue_and_multi_category(self):
        query = self.query(event_date="2026-11-14", event_format="свадьба", category="Банкетный зал",
                           budget_kzt=4000000, duration_hours=8)
        result = recommend(self.catalog, query)
        self.assertEqual({c["id"] for c in result["cards"]}, {"HK-64395", "HK-90011"})
        self.assertNotIn("HK-58236", {c["id"] for c in result["cards"]})
        self.assertTrue(any(c["provenance"]["city_imputed"] for c in result["cards"]))

    def test_description_does_not_override_structured_format(self):
        kiki = self.by_id["HK-35215"]
        catalog = Catalog((kiki,), "test-fixture")
        result = recommend(catalog, self.query(event_format="конференция"))
        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["diagnostics"]["all_exclusions"]["format"], 1)

    def test_suggestions_actually_work_changing_one_field(self):
        for query in [self.query(budget_kzt=100000),
                      self.query(event_date="2026-12-12", language="английский")]:
            result = recommend(self.catalog, query)
            self.assertTrue(result["suggestions"])
            for suggestion in result["suggestions"]:
                changed = {**query, suggestion["field"]: suggestion["value"]}
                actual = recommend(self.catalog, changed)
                self.assertEqual(actual["status"], "matched")
                self.assertEqual(actual["diagnostics"]["eligible_count"], suggestion["eligible_count"])

    def test_source_spans_and_distinct_explanations_on_all_demos(self):
        for path in (ROOT / "examples").glob("*.json"):
            request = json.loads(path.read_text(encoding="utf-8"))
            result = recommend(self.catalog, request)
            snapshot = ROOT / "examples" / "responses" / path.name
            self.assertEqual(result, json.loads(snapshot.read_text(encoding="utf-8")))
            explanations = []
            for card in result["cards"]:
                source = self.by_id[card["id"]].description
                quote = card["evidence"][-1]
                self.assertEqual(source[quote["start"]:quote["end"]], quote["quote"])
                self.assertIn(quote["quote"], card["explanation"])
                stripped = card["explanation"]
                for p in self.catalog.profiles:
                    stripped = stripped.replace(p.anon_name, "[имя]")
                explanations.append(stripped)
            self.assertEqual(len(explanations), len(set(explanations)))

    def test_repeated_query_and_permuted_catalog(self):
        expected = recommend(self.catalog, self.base)
        reversed_catalog = Catalog(tuple(reversed(self.catalog.profiles)), self.catalog.sha256)
        self.assertEqual(expected, recommend(reversed_catalog, self.base))
        for _ in range(10):
            self.assertEqual(expected, recommend(self.catalog, self.base))

    def test_restart_and_hash_seed_determinism(self):
        command = [sys.executable, "-X", "utf8", "-m", "eventmatch", "recommend", "--request",
                   "examples/01-dense.json"]
        a = subprocess.check_output(command, cwd=ROOT, env={**os.environ, "PYTHONHASHSEED": "1"})
        b = subprocess.check_output(command, cwd=ROOT, env={**os.environ, "PYTHONHASHSEED": "99"})
        self.assertEqual(a, b)

    def test_exact_tie_uses_id(self):
        p = self.by_id["HK-88430"]
        catalog = Catalog((replace(p, id="TEST-B"), replace(p, id="TEST-A")), "fixture-only")
        self.assertEqual([c["id"] for c in recommend(catalog, self.base)["cards"]], ["TEST-A", "TEST-B"])

    def test_invalid_request_is_not_empty_result(self):
        for changes in [dict(event_date="2027-01-01"), dict(event_date="2026-09-22"),
                        dict(event_date="2026-02-30"), dict(event_date="20261012"),
                        dict(budget_kzt=0), dict(budget_kzt=True), dict(budget_kzt="500000"),
                        dict(duration_hours=float("nan")), dict(duration_hours=-1),
                        dict(language="французский"), dict(event_format="вечеринка"),
                        dict(city=""), dict(extra="ignored?")]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                recommend(self.catalog, self.query(**changes))
        with self.assertRaises(ValueError):
            Request.parse([])

    def test_optional_inputs_and_case_normalization(self):
        omitted = {key: value for key, value in self.base.items()
                   if key not in ("language", "duration_hours")}
        a = recommend(self.catalog, omitted)
        b = recommend(self.catalog, {**omitted, "city": " алматы ", "category": "ведущий"})
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
