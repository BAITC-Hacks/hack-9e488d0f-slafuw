"""Hard constraints -> explainable lexical baseline -> grounded evidence -> top three.

This version deliberately makes no LLM or embedding claims. See docs/ARCHITECTURE.md.
"""

from collections import Counter
from dataclasses import replace
from datetime import date, timedelta
from hashlib import sha256
import json
import re

from .models import END, START, Request

POLICY_VERSION = "rules-lexical-v2"
REASONS = {
    "busy": "заняты на дату",
    "budget": "стартовая цена выше бюджета",
    "format": "не берут выбранный формат",
    "language": "не указан требуемый язык",
    "duration": "максимальная длительность меньше требуемой",
}
# Transparent, deliberately modest baseline: number of distinct relevant word roots.
# No rewards for awards, popularity, description length or spending the whole budget.
ROOTS = {
    "свадьба": ("свад", "молодож", "невест", "бракосочет"),
    "той": ("той", "казах", "традиц", "националь"),
    "корпоратив": ("корпоратив", "бизнес", "делов", "бренд"),
    "конференция": ("конференц", "форум", "презентац", "делов"),
    "юбилей": ("юбиле",),
    "день рождения": ("рождени",),
}
GENERIC_ROOTS = (
    "луч", "топ", "профессион", "качеств", "высок", "незабыва", "неповтор",
    "идеал", "уникал", "индивидуал", "опыт", "отлич", "прекрас", "люб",
    "впечатлен", "мастер", "команд", "услуг", "подход", "событ", "мероприят",
    "праздник", "вечер", "эмоци", "выбор", "ваш", "наш", "клиент", "уров",
    "стиль", "работ", "проведен", "созда",
    "алмат", "астан", "казахстан", "ведущ", "фотограф", "флорист", "декорат",
    "ансамбл", "артист", "подрядчик", "отел", "банкет", "ресторан", "центр", "зал",
    "свад", "молодож", "невест", "бракосочет", "той", "казах", "традиц",
    "националь", "корпоратив", "бизнес", "делов", "бренд", "конференц", "форум",
    "презентац", "юбиле", "рождени",
)
PROMOTIONAL_CLAIM = re.compile(
    r"\b(?:лучш\w*|топ(?:[- ]?\d+)?|профессиональн\w*|качественн\w*|"
    r"высок\w+\s+уров\w*|незабываем\w*|неповторим\w*|идеальн\w*|"
    r"уникальн\w*|индивидуальн\w+\s+подход)\b"
)
REFERENCE_LIST = re.compile(r"\b(?:среди (?:наших )?клиентов|клиенты и партнеры|сотрудничали)\b")
EXCLUSION_ACTION_ROOTS = (
    "бер", "работ", "провод", "подход", "обслуж", "выступ", "организ", "приним",
    "дел", "занима", "предостав", "участв",
)


def normalize(value):
    return value.casefold().replace("ё", "е")


def tokens(value):
    return set(re.findall(r"[а-яa-z]{4,}", normalize(value)))


def hits(value, event_format):
    words = re.findall(r"[а-яa-z]+", normalize(value))
    return tuple(root for root in ROOTS[event_format]
                 if any(word.startswith(root) for word in words))


def evidence_hits(value, event_format):
    # Brand mentions may affect ranking, but alone do not explain corporate fit.
    return tuple(root for root in hits(value, event_format)
                 if not (event_format == "корпоратив" and root == "бренд"))


def category_hits(value, category):
    words = re.findall(r"[а-яa-z]+", normalize(value))
    roots = tokens(category)
    return sum(any(word.startswith(root) for word in words) for root in roots)


def evidence_terms(value):
    terms = {word for word in tokens(value)
             if not any(word.startswith(root) for root in GENERIC_ROOTS)}
    terms.update(re.findall(r"\b\d+(?:[.,]\d+)?\b", value))
    return terms


def useful_evidence(value, event_format):
    """Reject generic praise and sentences that appear to deny the requested format."""
    normalized = normalize(value)
    if REFERENCE_LIST.search(normalized):
        return False
    words = re.findall(r"[а-яa-z]+", normalized)
    roots = ROOTS[event_format]
    format_positions = [index for index, word in enumerate(words)
                        if any(word.startswith(root) for root in roots)]
    action_positions = [index for index, word in enumerate(words)
                        if any(word.startswith(root) for root in EXCLUSION_ACTION_ROOTS)]
    for index, word in enumerate(words):
        if word != "не":
            continue
        if index + 1 < len(words):
            next_word = words[index + 1]
            if next_word.startswith(("только", "просто", "единствен", "огранич", "исключ",
                                     "отказыва", "запрещ")):
                continue
        nearby_formats = [position for position in format_positions if abs(position - index) <= 4]
        nearby_actions = [position for position in action_positions if abs(position - index) <= 4]
        is_adjacent = min((abs(position - index) for position in nearby_formats), default=99) <= 1
        if nearby_formats and (nearby_actions or is_adjacent):
            return False
        if (index + 1 < len(words) and words[index + 1] == "для"
                and any(abs(position - index) <= 4 for position in format_positions)):
            return False
    terms = evidence_terms(value)
    if PROMOTIONAL_CLAIM.search(normalized) and len(terms) < 2:
        return False
    return bool(terms)


def failures(profile, request):
    failed = []
    if request.event_date in profile.busy_dates:
        failed.append("busy")
    if profile.price_from_kzt > request.budget_kzt:
        failed.append("budget")
    if request.event_format not in profile.event_formats:
        failed.append("format")
    if request.language and request.language not in profile.languages:
        failed.append("language")
    if (request.duration_hours is not None and profile.max_hours is not None
            and request.duration_hours > profile.max_hours):
        failed.append("duration")
    return failed


def ranking_key(profile, request):
    return (-len(hits(profile.description, request.event_format)), profile.price_from_kzt, profile.id)


def source_excerpt(profile, request, peers):
    """Select an exact contiguous source span; rarity helps explanations differ.

    Peers are the whole city/category cohort, independent of current availability.
    Relevance first, then uncommon vocabulary, then earliest source position.
    This is lexical extraction, not proof that a marketing claim is verified.
    """
    document_frequency = Counter(word for peer in peers for word in tokens(peer.description))
    candidates = []
    description = profile.description
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]|$)", description):
        raw = match.group()
        stripped = raw.strip().rstrip(".!?").rstrip()
        if len(stripped) < 20:
            continue
        start = match.start() + len(raw) - len(raw.lstrip())
        if len(stripped) > 220:
            cut = stripped.rfind(" ", 0, 220)
            stripped = stripped[:cut if cut > 0 else 220]
        if not useful_evidence(stripped, request.event_format):
            continue
        words = evidence_terms(stripped)
        rarity = sum(1 for word in words if document_frequency[word] == 1)
        # Quantize a ratio; don't reward verbosity or depend on set iteration order.
        distinctiveness = 1000 * rarity // max(1, len(words))
        key = (-len(evidence_hits(stripped, request.event_format)),
               -category_hits(stripped, request.category), -len(words), -distinctiveness, start)
        candidates.append((key, start, start + len(stripped)))
    if not candidates:
        return None
    _, start, end = min(candidates)
    quote = description[start:end]
    return {"field": "description", "start": start, "end": end, "quote": quote}


def money(value):
    return f"{value:,}".replace(",", " ") + " ₸"


def make_card(profile, request, peers):
    evidence = source_excerpt(profile, request, peers)
    facts = [
        f"На {request.event_date} нет занятости в календаре",
        f"формат «{request.event_format}» указан",
        f"цена от {money(profile.price_from_kzt)} при бюджете {money(request.budget_kzt)}",
    ]
    if request.language:
        facts.append(f"язык — {request.language}")
    if request.duration_hours is not None:
        if profile.max_hours is None:
            facts.append("ограничение по часам присутствия неприменимо")
        else:
            facts.append(f"до {profile.max_hours:g} ч при запросе {request.duration_hours:g} ч")
    text = "; ".join(facts) + "."
    if evidence:
        text += f" В описании профиля: «{evidence['quote']}»."
    labels = ["Синтетический профиль организаторов" if profile.synthetic
              else "Исходный анонимизированный профиль"]
    if profile.price_imputed:
        labels.append("Цена проставлена при подготовке датасета")
    if profile.city_imputed:
        labels.append("Город проставлен при подготовке датасета")
    return {
        "id": profile.id, "name": profile.anon_name, "category": request.category,
        "categories": list(profile.categories), "city": profile.city,
        "price_from_kzt": profile.price_from_kzt,
        "explanation": text,
        "evidence": [
            {"field": "busy_dates", "operator": "not_contains", "value": request.event_date},
            {"field": "event_formats", "operator": "contains", "value": request.event_format},
            {"field": "price_from_kzt", "value": profile.price_from_kzt,
             "budget_kzt": request.budget_kzt},
            {"field": "languages", "value": list(profile.languages), "requested": request.language},
            {"field": "max_hours", "value": profile.max_hours, "requested": request.duration_hours},
        ] + ([evidence] if evidence else []),
        "ranking": {"lexical_hits": list(hits(profile.description, request.event_format)),
                    "price_tiebreak_kzt": profile.price_from_kzt, "id_tiebreak": profile.id},
        "provenance": {"synthetic": profile.synthetic, "city_imputed": profile.city_imputed,
                       "price_imputed": profile.price_imputed, "labels": labels},
    }


def suggestions(profiles, request):
    """Only verifiable single-condition changes; never add out-of-scope cards."""
    result = []
    budget_only = [p for p in profiles if failures(p, request) == ["budget"]]
    if budget_only:
        minimum = min(p.price_from_kzt for p in budget_only)
        changed = replace(request, budget_kzt=minimum)
        count = sum(not failures(p, changed) for p in profiles)
        result.append({"field": "budget_kzt", "value": minimum, "eligible_count": count,
                       "message": f"При бюджете {money(minimum)} пройдут {count}; остальные условия те же"})
    day = date.fromisoformat(request.event_date)
    nearby = sorted((day + timedelta(days=delta) for delta in range(-7, 8) if delta),
                    key=lambda item: (abs((item - day).days), item))
    for candidate in nearby:
        if not START <= candidate <= END:
            continue
        count = sum(not failures(p, replace(request, event_date=candidate.isoformat()))
                    for p in profiles)
        if count:
            result.append({"field": "event_date", "value": candidate.isoformat(),
                           "eligible_count": count,
                           "message": f"На {candidate} пройдут {count}; остальные условия те же"})
            break
    return result


def recommend(catalog, request, ranker=None):
    if not isinstance(request, Request):
        request = Request.parse(request)
    cities = {normalize(p.city): p.city for p in catalog.profiles}
    categories = {normalize(c): c for p in catalog.profiles for c in p.categories}
    if normalize(request.city) not in cities:
        raise ValueError("city: неизвестный город каталога")
    if normalize(request.category) not in categories:
        raise ValueError("category: неизвестная категория каталога")
    request = replace(request, city=cities[normalize(request.city)],
                      category=categories[normalize(request.category)])
    cohort = [p for p in catalog.profiles
              if normalize(p.city) == normalize(request.city)
              and any(normalize(c) == normalize(request.category) for c in p.categories)]
    # Return canonical spelling even if the user typed a different case.
    if cohort:
        request = replace(request, city=cohort[0].city,
                          category=next(c for c in cohort[0].categories
                                        if normalize(c) == normalize(request.category)))
    primary = Counter()
    all_reasons = Counter()
    rejected = []
    eligible = []
    date_only_names = []
    for profile in cohort:
        reasons = failures(profile, request)
        if reasons:
            primary[reasons[0]] += 1
            all_reasons.update(reasons)
            rejected.append({"id": profile.id, "reasons": reasons})
            if reasons == ["busy"]:
                date_only_names.append(profile.anon_name)
        else:
            eligible.append(profile)
    date_only = len(date_only_names)
    date_only_names.sort()
    eligible.sort(key=lambda p: (-ranker.score(p, request), p.price_from_kzt, p.id)
                  if ranker else ranking_key(p, request))
    if not cohort:
        status = "no_category_in_city"
        summary = f"В каталоге города «{request.city}» нет категории «{request.category}»"
    elif not eligible:
        status = "no_match"
        summary = f"Кандидатов в городе и категории: {len(cohort)}; ни один не проходит все условия"
    else:
        status = "matched"
        summary = f"Подобрали {min(3, len(eligible))}; всем условиям соответствуют {len(eligible)} из {len(cohort)}"
        if len(eligible) < 3 and len(cohort) < 3:
            summary += f"; в этом городе и категории всего {len(cohort)} профиля"
    if primary:
        summary += ". Причины исключения (каждый профиль учтён один раз): " + "; ".join(
            f"{REASONS[key]} — {primary[key]}" for key in REASONS if primary[key])
    if cohort:
        names = f" ({', '.join(date_only_names[:3])}{' и другие' if date_only > 3 else ''})" if date_only else ""
        summary += (f". Только из-за занятости на {request.event_date} исключены {date_only}{names}"
                    f"; без проверки даты прошли бы {len(eligible) + date_only}")
    fingerprint_input = {"request": request.to_dict(), "dataset": catalog.sha256,
                         "policy": ranker.metadata() if ranker else POLICY_VERSION}
    fingerprint = sha256(json.dumps(fingerprint_input, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
    return {
        "status": status, "summary": summary + ".", "request": request.to_dict(),
        "cards": [make_card(p, request, cohort) for p in eligible[:3]],
        "diagnostics": {
            "cohort_count": len(cohort), "eligible_count": len(eligible),
            "shown_count": min(3, len(eligible)),
            "primary_exclusions": {key: primary[key] for key in REASONS},
            "all_exclusions": {key: all_reasons[key] for key in REASONS},
            "blocked_only_by_date": date_only,
            "eligible_ignoring_date": len(eligible) + date_only,
            "rejected": sorted(rejected, key=lambda item: item["id"]),
        },
        "suggestions": suggestions(cohort, request) if cohort and not eligible else [],
        "price_note": "Цена «от» — нижняя граница за мероприятие, итоговая смета неизвестна.",
        "meta": {"dataset_sha256": catalog.sha256, "policy_version": POLICY_VERSION,
                 "request_fingerprint": fingerprint, "explanation_mode": "extractive-lexical"},
    }
