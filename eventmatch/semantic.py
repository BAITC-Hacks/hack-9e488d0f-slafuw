"""Offline NVIDIA embeddings -> versioned integer scores. No request-time model calls."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from .models import FORMATS
from .providers import NVIDIA_ENDPOINT

VERSION = 'nvidia-chunks-cosine-v1'


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def fragments(profile):
    # Full sentences retain negation/context. Split only long sentences at word boundaries.
    out = []
    for match in re.finditer(r'[^.!?\n]+(?:[.!?]+|$)', profile.description):
        start, end = match.span()
        while start < end and profile.description[start].isspace():
            start += 1
        while end > start and profile.description[end - 1].isspace():
            end -= 1
        while start < end:
            stop = min(end, start + 400)
            if stop < end:
                split = profile.description.rfind(' ', start, stop)
                if split > start:
                    stop = split
            quote = profile.description[start:stop]
            out.append({'profile_id': profile.id, 'evidence_id':
                        f'{profile.id}:description:{start}:{stop}',
                        'field': 'description', 'start': start, 'end': stop, 'quote': quote})
            start = stop
            while start < end and profile.description[start].isspace():
                start += 1
    return out


def intent_key(category, event_format):
    return category + '|' + event_format


def canonical_intents(catalog):
    return {intent_key(c, f): f'Подрядчик: {c}. Формат мероприятия: {f}.'
            for c in sorted({c for p in catalog.profiles for c in p.categories}) for f in FORMATS}


def cosine_int(a, b):
    if len(a) != len(b):
        raise ValueError('Embedding dimensions differ')
    denominator = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b))
    return round(max(-1, min(1, sum(x*y for x, y in zip(a, b)) / denominator)) * 1_000_000)


def prepare(catalog, client):
    chunks = [c for p in catalog.profiles for c in fragments(p)]
    intents = canonical_intents(catalog)
    passage_vectors = client.embed([c['quote'] for c in chunks], 'passage')
    query_vectors = client.embed(list(intents.values()), 'query')
    scores = {}
    for (key, _), vector in zip(intents.items(), query_vectors):
        scores[key] = {c['evidence_id']: cosine_int(vector, embedding)
                       for c, embedding in zip(chunks, passage_vectors)}
    artifact = {'version': VERSION, 'provider': 'nvidia', 'model': client.model,
                'endpoint': NVIDIA_ENDPOINT, 'dataset_sha256': catalog.sha256,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'api_calls': client.calls, 'input_types': ['passage', 'query'],
                'dimensions': len(passage_vectors[0]), 'chunks': chunks, 'intents': intents,
                'scores': scores, 'embeddings_sha256': digest([passage_vectors, query_vectors])}
    artifact['artifact_sha256'] = digest(artifact)
    return artifact


class SemanticRanker:
    def __init__(self, catalog, path, expected_model):
        artifact = json.loads(Path(path).read_text(encoding='utf-8'))
        supplied_hash = artifact.pop('artifact_sha256')
        if supplied_hash != digest(artifact):
            raise ValueError('NVIDIA artifact checksum mismatch')
        if (artifact['version'] != VERSION or artifact['provider'] != 'nvidia'
                or artifact['dataset_sha256'] != catalog.sha256
                or artifact['model'] != expected_model or artifact['endpoint'] != NVIDIA_ENDPOINT
                or type(artifact['api_calls']) is not int or artifact['api_calls'] < 1):
            raise ValueError('NVIDIA artifact provenance/version mismatch')
        expected_chunks = [c for p in catalog.profiles for c in fragments(p)]
        if artifact['chunks'] != expected_chunks or artifact['intents'] != canonical_intents(catalog):
            raise ValueError('NVIDIA artifact sources mismatch')
        chunk_ids = {c['evidence_id'] for c in expected_chunks}
        if set(artifact['scores']) != set(artifact['intents']):
            raise ValueError('NVIDIA artifact intents incomplete')
        for values in artifact['scores'].values():
            if set(values) != chunk_ids or any(type(v) is not int or abs(v) > 1_000_000
                                               for v in values.values()):
                raise ValueError('NVIDIA artifact scores invalid')
        self.scores = artifact['scores']
        self.by_profile = {p.id: fragments(p) for p in catalog.profiles}
        self.metadata = {k: artifact[k] for k in ('version', 'model', 'endpoint', 'created_at',
                                                  'api_calls', 'embeddings_sha256')}
        self.metadata.update(artifact_sha256=supplied_hash, usage='prepared_artifact', online_calls=0)

    def evidence(self, profile, request):
        values = self.scores[intent_key(request.category, request.event_format)]
        return sorted(self.by_profile[profile.id], key=lambda c: (-values[c['evidence_id']], c['start']))

    def score(self, profile, request):
        values = self.scores[intent_key(request.category, request.event_format)]
        return max(values[c['evidence_id']] for c in self.by_profile[profile.id])

    def key(self, profile, request):
        return (-self.score(profile, request), profile.price_from_kzt, profile.id)
