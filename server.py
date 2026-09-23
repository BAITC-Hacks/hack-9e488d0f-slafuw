"""Local demo adapter; domain logic is independent of HTTP and UI."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from uuid import uuid4

from .agent import EventMatchAgent
from .catalog import ROOT
from .service import EventMatchService


def handler_for(catalog, service=None, agent=None):
    service = service or EventMatchService(catalog)
    agent = agent or EventMatchAgent(service)
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload, content_type="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else json.dumps(
                payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.respond(200, (ROOT / "web" / "index.html").read_bytes(),
                             "text/html; charset=utf-8")
            elif self.path == "/health":
                self.respond(200, {"status": "ok", "profiles": len(catalog.profiles),
                                   "dataset_sha256": catalog.sha256,
                                   "ranking": service.ranker.metadata()})
            elif self.path == "/metadata":
                metadata = service.execute("get_catalog_metadata", {}, uuid4().hex)
                metadata["openai_configured"] = agent.client.configured
                metadata["explicit_default_year"] = agent.default_year
                self.respond(200, metadata)
            elif self.path.startswith("/traces/"):
                trace = service.store.trace(self.path.removeprefix("/traces/"))
                self.respond(200 if trace else 404, {"actions": trace})
            else:
                self.respond(404, {"error": "not_found", "message": "Маршрут не найден"})

        def do_POST(self):
            if self.path not in ("/recommend", "/chat", "/compare-dates", "/finalize-result"):
                self.respond(404, {"error": "not_found", "message": "Маршрут не найден"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32768:
                    raise ValueError("Требуется JSON-запрос размером от 1 до 32768 байт")
                raw = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("Требуется JSON-объект")
                if self.path == "/recommend":
                    result = agent.structured(raw, self.headers.get("X-EventMatch-Session"))
                elif self.path == "/chat":
                    if set(raw) - {"message", "session_id"} or "message" not in raw:
                        raise ValueError("Нужны message и опциональный session_id")
                    result = agent.chat(**raw)
                elif self.path == "/compare-dates":
                    if set(raw) != {"request", "date_a", "date_b"}:
                        raise ValueError("Нужны request, date_a и date_b")
                    result = agent.compare_structured(**raw, session_id=self.headers.get("X-EventMatch-Session"))
                else:
                    result = service.execute("finalize_result", raw, uuid4().hex, "HTTP")
            except (ValueError, UnicodeError, TypeError) as exc:
                self.respond(422, {"error": "invalid_request", "message": str(exc)})
                return
            except Exception:
                self.respond(503, {"status": "technical_error", "message":
                                   "Технически невозможно проверить каталог и выполнить подбор"})
                return
            self.respond(503 if result.get("status") == "technical_error" else 200, result)

    return Handler


def serve(catalog, port):
    service = EventMatchService.from_env(catalog)
    agent = EventMatchAgent.from_env(service)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(catalog, service, agent))
    print(f"EventMatch: http://127.0.0.1:{port} (остановка: Ctrl+C)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.store.close()
