"""Agent tools. Only backend code selects cards, verifies sources and renders text."""

from copy import deepcopy
from dataclasses import replace
from itertools import product
import json
import os
from time import monotonic

from .catalog import ROOT
from .engine import evidence_terms, failures, money, normalize, recommend, useful_evidence
from .models import END, FORMATS, LANGUAGES, START, Request, iso_date
from .semantic import BaselineRanker, SemanticRanker, digest, fragments
from .storage import Store

SERVICE_VERSION = "eventmatch-agent-v2"


def facts_for(profile, request):
    values = [
        ("format", "event_formats", request.event_format, f"Берёт формат «{request.event_format}»"),
        ("price", "price_from_kzt", profile.price_from_kzt,
         f"стартовая цена {money(profile.price_from_kzt)} укладывается в бюджет {money(request.budget_kzt)}"),
        ("date", "busy_dates", request.event_date,
         f"на {request.event_date} в известном календаре нет указанной занятости"),
        ("languages", "languages", list(profile.languages),
         f"требуемый язык «{request.language}» указан; также: " + ", ".join(profile.languages)
         if request.language else "языки работы: " + ", ".join(profile.languages)),
        ("hours", "max_hours", profile.max_hours,
         "ограничение по часам присутствия неприменимо" if profile.max_hours is None
         else f"до {profile.max_hours:g} ч при запросе {request.duration_hours:g} ч"
         if request.duration_hours is not None else f"до {profile.max_hours:g} ч"),
    ]
    return [{"fact_id": f"{profile.id}:{key}", "profile_id": profile.id, "field": field,
             "value": value, "text": text} for key, field, value, text in values]


class EventMatchService:
    def __init__(self, catalog, store=None, ranker=None):
        self.catalog = catalog
        self.by_id = {p.id: p for p in catalog.profiles}
        self.store = store or Store()
        self.ranker = ranker or BaselineRanker()
        self.scope_sha256 = digest([{
            **p.__dict__, "busy_dates": sorted(p.busy_dates)
        } for p in sorted(catalog.profiles, key=lambda p: p.id)])

    @classmethod
    def from_env(cls, catalog):
        mode = os.environ.get("EVENTMATCH_RANKER", "baseline")
        if mode == "baseline":
            ranker = BaselineRanker()
        elif mode == "nvidia":
            path = os.environ.get("EVENTMATCH_NVIDIA_ARTIFACT", str(ROOT / "data/nvidia-artifact.json"))
            ranker = SemanticRanker(catalog, path)
        else:
            raise ValueError("EVENTMATCH_RANKER: допустимы baseline или nvidia")
        store = Store(os.environ.get("EVENTMATCH_DB", str(ROOT / "data/eventmatch.sqlite3")))
        return cls(catalog, store, ranker)

    def metadata(self):
        return {
            "cities": sorted({p.city for p in self.catalog.profiles}),
            "categories": sorted({c for p in self.catalog.profiles for c in p.categories}),
            "event_formats": list(FORMATS), "languages": list(LANGUAGES),
            "calendar_start": str(START), "calendar_end": str(END),
            "catalog_version": self.catalog.sha256, "scope_sha256": self.scope_sha256,
            "algorithm_version": SERVICE_VERSION, "ranking": self.ranker.metadata(),
        }

    def execute(self, tool, arguments, trace_id, caller="backend"):
        started = monotonic()
        methods = {"get_catalog_metadata": self.metadata, "recommend": self.recommend,
                   "compare_dates": self.compare_dates, "finalize_result": self.finalize_result}
        if tool not in methods or not isinstance(arguments, dict):
            raise ValueError("Неизвестный инструмент или аргументы")
        entry = {"tool": tool, "arguments": arguments, "caller": caller,
                 "versions": self.metadata()}
        try:
            result = methods[tool](**arguments)
            entry["status"] = "ok"
            results = result.get("results", [result])
            if tool in ("recommend", "compare_dates"):
                entry["arguments"] = {**arguments, "request": results[0]["normalized_request"]}
            entry["results"] = [{"result_id": r.get("result_id"),
                                 "normalized_request": r.get("normalized_request"),
                                 "diagnostics": r.get("diagnostics"),
                                 "selected_ids": [c["id"] for c in r.get("cards", [])],
                                 "selected_sources": [c.get("selected_sources")
                                                      for c in r.get("cards", [])]}
                                for r in results]
            return result
        except ValueError:
            entry["status"] = "invalid_arguments"
            raise
        except (TypeError, KeyError):
            entry["status"] = "invalid_arguments"
            raise ValueError("Некорректные аргументы инструмента " + tool) from None
        finally:
            entry["duration_ms"] = round((monotonic() - started) * 1000, 3)
            self.store.log(trace_id, entry)

    def recommend(self, request):
        result = recommend(self.catalog, request, self.ranker)
        normalized = Request.parse(result["request"])
        for card in result["cards"]:
            profile = self.by_id[card["id"]]
            card["facts"] = facts_for(profile, normalized)
            parts = [part for part in fragments(profile)
                     if useful_evidence(part["quote"], normalized.event_format)]
            parts.sort(key=lambda part: (-self.ranker.evidence_score(part, normalized), part["start"]))
            card["available_evidence"] = parts
            card["ranking"] = {"score_int": self.ranker.score(profile, normalized),
                               "mode": self.ranker.mode, "price_tiebreak_kzt": profile.price_from_kzt,
                               "id_tiebreak": profile.id}
            # No per-profile synthetic origin field exists in this dataset schema.
            card["provenance"]["labels"][0] = ("Синтетический профиль" if profile.synthetic
                                               else "Исходный анонимизированный профиль")
            card["tags"] = [f["text"] for f in card["facts"]]
        result["normalized_request"] = result["request"]
        result["metadata"] = {**self.metadata(), "explanation_mode": "template explanation fallback",
                              "openai": {"mode": "not_called", "model": None}}
        result["meta"] = result["metadata"]
        result["result_id"] = digest({"request": result["request"], "versions": self.metadata()})
        if result["status"] == "matched":
            n = len(result["cards"])
            heading = (f"Подобрали {n} вариантов: {normalized.category}, "
                       f"{normalized.city}, {normalized.event_date}.")
            if result["diagnostics"]["eligible_count"] > 3:
                heading += (f" Показаны 3 из {result['diagnostics']['eligible_count']} подрядчиков, "
                            "соответствующих условиям.")
            result["summary"] = heading + " " + result["summary"]
        self._assemble(result, self.default_plan(result))
        self.store.put("results", result["result_id"], result)
        return result

    def default_plan(self, result):
        request = Request.parse(result["normalized_request"])
        cards = result["cards"]
        candidate_sets = []
        for card in cards:
            parts = card["available_evidence"]
            if parts:
                best_score = max(self.ranker.evidence_score(part, request) for part in parts)
                tied = [part for part in parts
                        if self.ranker.evidence_score(part, request) == best_score]
                tied.sort(key=lambda part: (-len(evidence_terms(part["quote"])), part["start"],
                                            part["evidence_id"]))
                candidate_sets.append(tied[:8])
            else:
                candidate_sets.append([None])

        def diversity_key(assignment):
            words = [evidence_terms(part["quote"]) if part else set() for part in assignment]
            overlap = sum(len(left & right) / max(1, len(left | right))
                          for index, left in enumerate(words) for right in words[index + 1:])
            unique = 0
            for index, current in enumerate(words):
                other_words = set()
                for other_index, other in enumerate(words):
                    if other_index != index:
                        other_words.update(other)
                unique += len(current - other_words) / max(1, len(current))
            return (overlap, -unique, sum(len(part["quote"]) for part in assignment if part),
                    tuple(part["evidence_id"] if part else "" for part in assignment))

        evidence_assignment = min(product(*candidate_sets), key=diversity_key) if cards else ()
        secondary_fields = []
        if request.duration_hours is not None:
            secondary_fields.append("hours")
        if request.language is not None:
            secondary_fields.append("languages")
        secondary_fields.extend(("format", "date"))
        fact_maps = [{fact["fact_id"].rsplit(":", 1)[-1]: fact for fact in card["facts"]}
                     for card in cards]

        def distinct_values(field):
            return len({json.dumps(facts[field]["value"], ensure_ascii=False, sort_keys=True)
                        for facts in fact_maps})

        second_field = max(
            secondary_fields,
            key=lambda field: (distinct_values(field), -secondary_fields.index(field)),
        )
        return {"cards": [{
            "profile_id": card["id"],
            "fact_ids": [fact_maps[index]["price"]["fact_id"],
                         fact_maps[index][second_field]["fact_id"]],
            "evidence_id": evidence_assignment[index]["evidence_id"]
            if evidence_assignment[index] else None,
        } for index, card in enumerate(cards)]}

    def _assemble(self, result, plan):
        if not isinstance(plan, dict) or set(plan) != {"cards"} or not isinstance(plan["cards"], list):
            raise ValueError("Неверный план объяснений")
        if len(plan["cards"]) != len(result["cards"]):
            raise ValueError("План должен содержать ровно выбранные карточки")
        if any(not isinstance(p, dict) or set(p) != {"profile_id", "fact_ids", "evidence_id"}
               for p in plan["cards"]):
            raise ValueError("Неверные поля плана")
        ids = [p["profile_id"] for p in plan["cards"]]
        if any(not isinstance(pid, str) for pid in ids):
            raise ValueError("profile_id должен быть строкой")
        if len(set(ids)) != len(ids) or set(ids) != {c["id"] for c in result["cards"]}:
            raise ValueError("Нельзя добавлять, удалять или дублировать подрядчиков")
        by_id = {p["profile_id"]: p for p in plan["cards"]}
        event_format = Request.parse(result["normalized_request"]).event_format
        for card in result["cards"]:  # Snapshot order, never plan order.
            chosen = by_id[card["id"]]
            facts = {f["fact_id"]: f for f in card["facts"]}
            fact_ids = chosen["fact_ids"]
            if (not isinstance(fact_ids, list) or not 1 <= len(fact_ids) <= 2
                    or any(not isinstance(fid, str) for fid in fact_ids)
                    or len(set(fact_ids)) != len(fact_ids) or any(fid not in facts for fid in fact_ids)):
                raise ValueError("Неизвестный факт или слишком длинное объяснение")
            parts = {p["evidence_id"]: p for p in card["available_evidence"]}
            evidence_id = chosen["evidence_id"]
            if evidence_id is not None and (not isinstance(evidence_id, str) or evidence_id not in parts):
                raise ValueError("Цитата принадлежит другому профилю или неизвестна")
            if (evidence_id is not None
                    and not useful_evidence(parts[evidence_id]["quote"], event_format)):
                raise ValueError("Цитата не содержит полезного основания или противоречит формату")
            text = "; ".join(facts[fid]["text"] for fid in fact_ids)
            text = text[0].upper() + text[1:] + "."
            part = parts.get(evidence_id)
            if part:
                profile = self.by_id[card["id"]]
                if profile.description[part["start"]:part["end"]] != part["quote"]:
                    raise ValueError("Источник цитаты изменён")
                text += f" В описании профиля: «{part['quote']}»."
            card["explanation"] = text
            card["evidence"] = [e for e in card["evidence"] if e["field"] != "description"]
            if part:
                card["evidence"].append(part)
            card["selected_sources"] = deepcopy(chosen)
        signatures = {(c["price_from_kzt"], tuple(self.by_id[c["id"]].languages),
                       self.by_id[c["id"]].max_hours,
                       normalize(self.by_id[c["id"]].description).replace(normalize(c["name"]), ""))
                      for c in result["cards"]}
        result["explanation_note"] = ("Данных недостаточно, чтобы содержательно различить эти профили."
                                      if len(result["cards"]) > 1 and len(signatures) == 1 else None)

    def finalize_result(self, result_id, explanation_plan):
        result = self.store.get("results", result_id)
        if result["metadata"]["scope_sha256"] != self.scope_sha256 or result["metadata"]["ranking"] != self.ranker.metadata():
            raise ValueError("Результат относится к другой версии сервиса")
        request = Request.parse(result["normalized_request"])
        cards = result["cards"]
        if len(cards) > 3 or len({c["id"] for c in cards}) != len(cards):
            raise ValueError("Нарушен состав результата")
        for card in cards:
            p = self.by_id.get(card["id"])
            if p is None or p.city != request.city or request.category not in p.categories or failures(p, request):
                raise ValueError("Нарушены ограничения подбора")
        self._assemble(result, explanation_plan)
        result["metadata"]["explanation_mode"] = "validated fact and quote selection"
        result["meta"] = result["metadata"]
        return result

    def compare_dates(self, request, date_a, date_b):
        request = Request.parse(request)
        iso_date(date_a, "date_a")
        iso_date(date_b, "date_b")
        a = self.recommend(replace(request, event_date=date_a))
        b = self.recommend(replace(request, event_date=date_b))
        request_a = Request.parse(a["normalized_request"])
        request_b = Request.parse(b["normalized_request"])
        ids_a, ids_b = ([c["id"] for c in r["cards"]] for r in (a, b))
        changes = []
        for p in sorted(self.catalog.profiles, key=lambda p: p.id):
            if p.city != request_a.city or request_a.category not in p.categories:
                continue
            fail_a, fail_b = failures(p, request_a), failures(p, request_b)
            if bool(fail_a) != bool(fail_b):
                reason = "availability_changed"
                message = (f"{p.anon_name}: на {date_a} {'занят' if 'busy' in fail_a else 'нет указанной занятости'}, "
                           f"на {date_b} {'занят' if 'busy' in fail_b else 'нет указанной занятости'}.")
            elif not fail_a and not fail_b and ((p.id in ids_a) != (p.id in ids_b)):
                reason = "top3_displacement"
                message = f"{p.anon_name}: проходит обе даты; состав top-3 изменился из-за доступности других кандидатов."
            elif any(f != "busy" for f in fail_a):
                reason = "other_constraints"
                message = f"{p.anon_name}: на обеих датах не проходит другие условия."
            else:
                reason = "unchanged"
                message = f"{p.anon_name}: положение в выдаче не изменилось."
            changes.append({"profile_id": p.id, "reason": reason, "message": message,
                            "failures_a": fail_a, "failures_b": fail_b,
                            "eligible_a": not fail_a, "eligible_b": not fail_b,
                            "in_top3_a": p.id in ids_a, "in_top3_b": p.id in ids_b})
        return {"status": "compared", "results": [a, b],
                "appeared_ids": [pid for pid in ids_b if pid not in ids_a],
                "disappeared_ids": [pid for pid in ids_a if pid not in ids_b],
                "changes": changes, "metadata": self.metadata()}
