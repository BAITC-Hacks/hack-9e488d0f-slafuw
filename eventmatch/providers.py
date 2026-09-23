"""Documented HTTPS providers, bounded total wait, sanitized errors, no retries."""
import json
import math
from queue import Queue, Empty
from threading import BoundedSemaphore, Thread
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPRedirectHandler

OPENAI_ENDPOINT = 'https://api.openai.com/v1/chat/completions'
NVIDIA_ENDPOINT = 'https://integrate.api.nvidia.com/v1/embeddings'
_SLOTS = BoundedSemaphore(8)


class ProviderError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post_json(endpoint, key, payload, deadline):
    if not key:
        raise ProviderError('API key is not configured')
    if deadline <= monotonic() or not _SLOTS.acquire(blocking=False):
        raise ProviderError('Provider deadline or concurrency limit')
    output = Queue(maxsize=1)

    def run():
        try:
            request = Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode(),
                              headers={'Authorization': 'Bearer ' + key,
                                       'Content-Type': 'application/json'})
            with build_opener(NoRedirect()).open(
                    request, timeout=max(.1, deadline - monotonic())) as response:
                body = response.read(8 * 1024 * 1024 + 1)
                if len(body) > 8 * 1024 * 1024:
                    raise ProviderError('Provider response exceeds limit')
                output.put((True, json.loads(body)))
        except HTTPError as exc:
            output.put((False, ProviderError(f'Provider HTTP {exc.code}')))
        except (OSError, URLError, ValueError, ProviderError):
            output.put((False, ProviderError('Provider unavailable or invalid response')))
        finally:
            _SLOTS.release()
    Thread(target=run, daemon=True).start()
    try:
        ok, value = output.get(timeout=max(.001, deadline - monotonic()))
    except Empty:
        raise ProviderError('Provider deadline exceeded') from None
    if not ok:
        raise value
    return value


class OpenAI:
    def __init__(self, settings):
        self.key, self.model = settings.openai_key, settings.openai_model

    def complete(self, messages, tools, deadline, tool_choice='required'):
        payload = {'model': self.model, 'messages': messages, 'tools': tools,
                   'tool_choice': tool_choice, 'parallel_tool_calls': False,
                   'max_completion_tokens': 1800}
        result = post_json(OPENAI_ENDPOINT, self.key, payload, deadline)
        try:
            message = result['choices'][0]['message']
            if not isinstance(message, dict) or not message.get('tool_calls'):
                raise ValueError()
            return message
        except (KeyError, IndexError, TypeError, ValueError):
            raise ProviderError('OpenAI returned no valid tool call') from None


class NVIDIA:
    def __init__(self, settings):
        self.key, self.model = settings.nvidia_key, settings.nvidia_model
        self.calls = 0

    def embed(self, texts, input_type):
        if input_type not in ('query', 'passage'):
            raise ValueError('Invalid input_type')
        vectors = []
        for start in range(0, len(texts), 16):
            batch = texts[start:start + 16]
            result = post_json(NVIDIA_ENDPOINT, self.key,
                               {'model': self.model, 'input': batch, 'input_type': input_type,
                                'encoding_format': 'float', 'truncate': 'NONE'}, monotonic() + 45)
            self.calls += 1
            try:
                rows = sorted(result['data'], key=lambda r: r['index'])
                if [r['index'] for r in rows] != list(range(len(batch))):
                    raise ValueError()
                for row in rows:
                    v = row['embedding']
                    if (not isinstance(v, list) or not v
                            or any(type(x) not in (int, float) or not math.isfinite(x) for x in v)
                            or not any(v)):
                        raise ValueError()
                    vectors.append(v)
            except (KeyError, TypeError, ValueError):
                raise ProviderError('NVIDIA returned invalid embeddings') from None
        if vectors and len({len(v) for v in vectors}) != 1:
            raise ProviderError('NVIDIA embedding dimensions differ')
        return vectors
