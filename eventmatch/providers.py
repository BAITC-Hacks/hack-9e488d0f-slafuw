"""Server-side provider clients. No credentials or provider error bodies in responses."""

import json
import math
import os
from queue import Empty, Queue
from threading import Thread
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

NVIDIA_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/embeddings"
OPENAI_ENDPOINT = "https://api.openai.com/v1/responses"


class ProviderError(RuntimeError):
    pass


def post_json(endpoint, key, body, timeout):
    if not key:
        raise ProviderError("API не настроен на сервере")
    if timeout <= 0:
        raise ProviderError("Истёк общий deadline")
    request = Request(endpoint, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                      headers={"Authorization": f"Bearer {key}",
                               "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        raise ProviderError(f"API вернул HTTP {exc.code}") from None
    except (URLError, OSError, ValueError) as exc:
        raise ProviderError("API недоступен или вернул некорректный ответ") from None


class OpenAIClient:
    def __init__(self):
        self.key = os.environ.get("OPENAI_API_KEY", "")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14")

    @property
    def configured(self):
        return bool(self.key)

    def call(self, instructions, inputs, tools, deadline, tool_choice="required"):
        body = {
            "model": self.model, "instructions": instructions, "input": inputs,
            "tools": tools, "tool_choice": tool_choice, "parallel_tool_calls": False,
            "store": False, "max_output_tokens": 2400,
        }
        if self.model == "gpt-5.5" or self.model.startswith("gpt-5.5-"):
            body["reasoning"] = {"effort": "none"}
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise ProviderError("Истёк общий deadline")
        # Socket timeouts alone don't bound slow streaming bodies. The caller's total
        # budget is enforced independently; a late result cannot mutate order state.
        queue = Queue(maxsize=1)
        def perform():
            try:
                queue.put((True, post_json(OPENAI_ENDPOINT, self.key, body, remaining)))
            except ProviderError as exc:
                queue.put((False, exc))
        Thread(target=perform, daemon=True).start()
        try:
            ok, result = queue.get(timeout=remaining)
        except Empty:
            raise ProviderError("Истёк общий deadline") from None
        if not ok:
            raise result
        if not isinstance(result, dict) or result.get("status") != "completed":
            raise ProviderError("OpenAI не завершил вызов инструмента")
        if not isinstance(result.get("output"), list):
            raise ProviderError("Некорректный ответ OpenAI")
        return result


class NvidiaClient:
    def __init__(self):
        self.key = os.environ.get("NVIDIA_API_KEY", "")
        self.model = os.environ.get("NVIDIA_EMBEDDING_MODEL", NVIDIA_MODEL)
        # This request contract and Russian support were checked for this model only.
        if self.model != NVIDIA_MODEL:
            raise ValueError("Для выбранной NVIDIA-модели требуется отдельная проверка контракта")
        self.calls = []

    def embed(self, texts, input_type):
        vectors = []
        for offset in range(0, len(texts), 16):
            batch = texts[offset:offset + 16]
            started = monotonic()
            response = post_json(NVIDIA_ENDPOINT, self.key, {
                "model": self.model, "input": batch, "input_type": input_type,
                "encoding_format": "float", "truncate": "NONE",
            }, 60)
            try:
                data = sorted(response["data"], key=lambda item: item["index"])
                if [item["index"] for item in data] != list(range(len(batch))):
                    raise ValueError("indices")
                batch_vectors = [item["embedding"] for item in data]
                for vector in batch_vectors:
                    if (not isinstance(vector, list) or not vector
                            or any(type(x) not in (int, float) or not math.isfinite(x)
                                   for x in vector) or not any(vector)):
                        raise ValueError("vector")
                vectors.extend(batch_vectors)
            except (KeyError, TypeError, ValueError):
                raise ProviderError("Некорректные embeddings NVIDIA") from None
            self.calls.append({"input_type": input_type, "count": len(batch),
                               "duration_ms": round((monotonic() - started) * 1000, 2)})
        if len({len(vector) for vector in vectors}) > 1:
            raise ProviderError("NVIDIA вернул разные размерности embeddings")
        return vectors
