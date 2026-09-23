"""Input and dataset validation. Structured fields are authoritative."""

from dataclasses import asdict, dataclass
from datetime import date
import math

START = date(2026, 9, 23)
END = date(2026, 12, 31)
FORMATS = ("свадьба", "той", "корпоратив", "конференция", "юбилей", "день рождения")
LANGUAGES = ("русский", "казахский", "английский")


def text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: требуется непустая строка")
    return value.strip()


def iso_date(value, field):
    if not isinstance(value, str):
        raise ValueError(f"{field}: требуется дата YYYY-MM-DD")
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field}: требуется дата YYYY-MM-DD") from exc
    if result.isoformat() != value:
        raise ValueError(f"{field}: требуется дата YYYY-MM-DD")
    if not START <= result <= END:
        raise ValueError(f"{field}: календарь известен только с {START} по {END}")
    return result


def positive(value, field, integer=False):
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or (type(value) is float and not math.isfinite(value)) or value <= 0:
        kind = "целое положительное число" if integer else "положительное конечное число"
        raise ValueError(f"{field}: требуется {kind}")
    return value


@dataclass(frozen=True)
class Request:
    city: str
    event_date: str
    event_format: str
    category: str
    budget_kzt: int
    duration_hours: float | None = None
    language: str | None = None

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw, dict):
            raise ValueError("Запрос должен быть JSON-объектом")
        required = {"city", "event_date", "event_format", "category", "budget_kzt"}
        allowed = required | {"duration_hours", "language"}
        if missing := required - raw.keys():
            raise ValueError("Не заполнены поля: " + ", ".join(sorted(missing)))
        if extra := raw.keys() - allowed:
            raise ValueError("Неизвестные поля: " + ", ".join(sorted(extra)))
        city = text(raw["city"], "city")
        category = text(raw["category"], "category")
        event_format = text(raw["event_format"], "event_format").casefold()
        if event_format not in FORMATS:
            raise ValueError("event_format: допустимы " + ", ".join(FORMATS))
        language = raw.get("language")
        if language is not None:
            language = text(language, "language").casefold()
            if language not in LANGUAGES:
                raise ValueError("language: допустимы " + ", ".join(LANGUAGES))
        duration = raw.get("duration_hours")
        if duration is not None:
            duration = positive(duration, "duration_hours")
        return cls(city, iso_date(raw["event_date"], "event_date").isoformat(), event_format,
                   category, positive(raw["budget_kzt"], "budget_kzt", integer=True),
                   duration, language)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Profile:
    id: str
    anon_name: str
    categories: tuple[str, ...]
    city: str
    price_from_kzt: int
    event_formats: tuple[str, ...]
    languages: tuple[str, ...]
    max_hours: float | None
    busy_dates: frozenset[str]
    description: str
    synthetic: bool
    city_imputed: bool
    price_imputed: bool

    @classmethod
    def parse(cls, raw):
        expected = set(cls.__dataclass_fields__)
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("Профиль должен содержать ровно поля схемы датасета")
        values = dict(raw)
        for key in ("id", "anon_name", "city", "description"):
            # Preserve description byte-for-byte at the string level for source offsets.
            text(values[key], key)
        for key in ("categories", "event_formats", "languages", "busy_dates"):
            items = values[key]
            if not isinstance(items, list) or (not items and key != "busy_dates"):
                raise ValueError(f"{key}: требуется список")
            for item in items:
                if text(item, key) != item:
                    raise ValueError(f"{key}: лишние пробелы в значении")
            if len(items) != len(set(items)):
                raise ValueError(f"{key}: повторяющиеся значения")
            values[key] = tuple(items)
        if set(values["event_formats"]) - set(FORMATS):
            raise ValueError("Неизвестный формат в профиле")
        if set(values["languages"]) - set(LANGUAGES):
            raise ValueError("Неизвестный язык в профиле")
        for item in values["busy_dates"]:
            iso_date(item, "busy_dates")
        values["busy_dates"] = frozenset(values["busy_dates"])
        positive(values["price_from_kzt"], "price_from_kzt", integer=True)
        if values["max_hours"] is not None:
            positive(values["max_hours"], "max_hours")
        for key in ("synthetic", "city_imputed", "price_imputed"):
            if type(values[key]) is not bool:
                raise ValueError(f"{key}: требуется boolean")
        return cls(**values)
