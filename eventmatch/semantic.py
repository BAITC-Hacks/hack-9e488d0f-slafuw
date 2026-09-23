"""Frozen NVIDIA scores: preparation is online; serving never calls embeddings APIs."""

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from .engine import hits
from .models import FORMATS
from .providers import NVIDIA_ENDPOINT, NVIDIA_MODEL

VERSION = "nvidia-cosine-max-v1"
FRAGMENT_VERSION = "whole-sentence-v1"


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fragments(profile):
    """Whole sentences, with exact offsets. Never remove a negation or shorten a quote."""
    result = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", profile.description):
        raw = match.group()
        quote = raw.strip()
        if not quote:
            continue
        start = match.start() + len(raw) - len(raw.lstrip())
        end = start + len(quote)
        result.append({"evidence_id": f"{profile.id}:description:{start}:{end}:{digest(quote)[:12]}",
                       "profile_id": profile.id, "field": "description", "start": start,
                       "end": end, "quote": quote})
    return result


def intent_key(category, event_format):
    return f"{category}|{event_format}"


def catalog_inputs(catalog):
    pieces = [part for p in sorted(catalog.profiles, key=lambda p: p.id) for part in fragments(p)]
    intents = {intent_key(c, f): f"Нужен подрядчик категории «{c}» для мероприятия «{f}»."
               for c in sorted({c for p in catalog.profiles for c in p.categories}) for f in FORMATS}
    return pieces, intents


def cosine_int(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("Несовместимые размерности embeddings")
    denominator = math.sqrt(math.fsum(x*x for x in a) * math.fsum(x*x for x in b))
    if not denominator:
        raise ValueError("Нулевой embedding")
    return round(max(-1, min(1, math.fsum(x*y for x, y in zip(a, b)) / denominator)) * 1000000)


def prepare(catalog, client):
    pieces, intents = catalog_inputs(catalog)
    passage_vectors = client.embed([part["quote"] for part in pieces], "passage")
    query_vectors = client.embed(list(intents.values()), "query")
    if len(passage_vectors) != len(pieces) or len(query_vectors) != len(intents):
        raise ValueError("Неполный набор embeddings")
    evidence_scores = {}
    scores = {}
    for (key, _), query in zip(intents.items(), query_vectors):
        relevant = {p.id for p in catalog.profiles if key.split("|", 1)[0] in p.categories}
        per_piece = {part["evidence_id"]: cosine_int(vector, query)
                     for part, vector in zip(pieces, passage_vectors)
                     if part["profile_id"] in relevant}
        evidence_scores[key] = per_piece
        scores[key] = {pid: max(per_piece[part["evidence_id"]] for part in pieces
                                if part["profile_id"] == pid) for pid in sorted(relevant)}
    payload = {
        "version": VERSION, "fragment_version": FRAGMENT_VERSION, "provider": "NVIDIA",
        "model": client.model, "endpoint": NVIDIA_ENDPOINT,
        "model_revision": "provider-managed; exact weights revision not exposed",
        "request_parameters": {"encoding_format": "float", "truncate": "NONE",
                               "description_input_type": "passage", "intent_input_type": "query"},
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset_sha256": catalog.sha256,
        "fragments": pieces, "intents": intents, "input_sha256": digest([pieces, intents]),
        "passage_vectors": passage_vectors, "query_vectors": query_vectors,
        "scores": scores, "evidence_scores": evidence_scores, "api_calls": client.calls,
    }
    return {"artifact_sha256": digest(payload), "payload": payload}


class BaselineRanker:
    version = "rules-lexical-v1"
    mode = "deterministic baseline"

    def score(self, profile, request):
        return len(hits(profile.description, request.event_format))

    def evidence_score(self, part, request):
        return len(hits(part["quote"], request.event_format))

    def metadata(self):
        return {"mode": self.mode, "version": self.version, "nvidia": None}


class SemanticRanker:
    version = VERSION
    mode = "NVIDIA semantic artifact"

    def __init__(self, catalog, path):
        try:
            self._load(catalog, path)
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError("Повреждена структура NVIDIA-артефакта") from exc

    def _load(self, catalog, path):
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = artifact["payload"]
        self.sha256 = digest(payload)
        pieces, intents = catalog_inputs(catalog)
        if (self.sha256 != artifact["artifact_sha256"]
                or payload["dataset_sha256"] != catalog.sha256
                or payload["version"] != VERSION or payload["fragment_version"] != FRAGMENT_VERSION
                or payload["provider"] != "NVIDIA" or payload["model"] != NVIDIA_MODEL
                or payload["endpoint"] != NVIDIA_ENDPOINT or not payload["api_calls"]
                or payload["fragments"] != pieces or payload["intents"] != intents
                or payload["input_sha256"] != digest([pieces, intents])):
            raise ValueError("NVIDIA-артефакт не соответствует каталогу или версии алгоритма")
        # Validate coverage and quantized values at startup, even for currently ineligible profiles.
        for key in intents:
            expected = {p.id for p in catalog.profiles if key.split("|", 1)[0] in p.categories}
            if set(payload["scores"][key]) != expected:
                raise ValueError("Неполные семантические оценки")
            for pid in expected:
                values = [payload["evidence_scores"][key][p["evidence_id"]]
                          for p in pieces if p["profile_id"] == pid]
                score = payload["scores"][key][pid]
                if (any(type(v) is not int or abs(v) > 1000000 for v in values)
                        or type(score) is not int or score != max(values)):
                    raise ValueError("Некорректные семантические оценки")
        self.payload = payload

    def score(self, profile, request):
        return self.payload["scores"][intent_key(request.category, request.event_format)][profile.id]

    def evidence_score(self, part, request):
        return self.payload["evidence_scores"][intent_key(request.category, request.event_format)][part["evidence_id"]]

    def metadata(self):
        return {"mode": self.mode, "version": self.version, "nvidia": {
            "usage": "prepared artifact; no online embedding call",
            "model": self.payload["model"], "model_revision": self.payload["model_revision"],
            "endpoint": self.payload["endpoint"], "artifact_sha256": self.sha256,
            "version": self.version, "created_at": self.payload["created_at"],
        }}
