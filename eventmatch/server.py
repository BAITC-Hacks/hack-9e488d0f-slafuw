"""Local HTTP adapter. Bind loopback by default; put auth/TLS proxy before public use."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket

from .agent import Agent
from .catalog import ROOT
from .config import Settings
from .service import Service


def handler_for(catalog, settings=None):
    service = Service(catalog, settings or Settings.from_env())
    agent = Agent(service)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, format, *args):
            # Do not log URLs, request bodies, provider messages or session tokens.
            pass

        def respond(self, status, payload, content_type='application/json; charset=utf-8'):
            body = payload if isinstance(payload, bytes) else json.dumps(
                payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == '/':
                self.respond(200, (ROOT / 'web/index.html').read_bytes(), 'text/html; charset=utf-8')
            elif self.path == '/health':
                self.respond(200, {'status': 'ok', 'profiles': len(catalog.profiles),
                                   'dataset_sha256': catalog.sha256,
                                   'ranking_mode': service.settings.ranker})
            elif self.path == '/metadata':
                self.respond(200, service.get_catalog_metadata())
            else:
                self.respond(404, {'error': 'not_found', 'message': 'Маршрут не найден'})

        def do_POST(self):
            if self.path not in ('/recommend', '/compare-dates', '/finalize-result', '/chat'):
                self.respond(404, {'error': 'not_found', 'message': 'Маршрут не найден'})
                return
            try:
                if not self.headers.get('Content-Type', '').startswith('application/json'):
                    raise ValueError('Требуется Content-Type: application/json')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 32768:
                    raise ValueError('Требуется JSON размером от 1 до 32768 байт')
                raw = json.loads(self.rfile.read(length).decode('utf-8'))
                if not isinstance(raw, dict):
                    raise ValueError('Требуется JSON-объект')
                if self.path == '/recommend':
                    result = service.recommend(raw)
                elif self.path == '/compare-dates':
                    if set(raw) != {'request', 'date_a', 'date_b'} or not isinstance(raw['request'], dict):
                        raise ValueError('Требуются request, date_a, date_b')
                    result = service.compare_dates(raw['request'], raw['date_a'], raw['date_b'])
                elif self.path == '/finalize-result':
                    if set(raw) != {'result_id', 'explanation_plan'}:
                        raise ValueError('Требуются result_id, explanation_plan')
                    result = service.finalize_result(raw['result_id'], raw['explanation_plan'])
                else:
                    result = agent.chat(raw)
            except (ValueError, UnicodeError) as exc:
                self.respond(422, {'error': 'invalid_request', 'message': str(exc)})
                return
            except (OSError, socket.timeout):
                self.respond(503, {'error': 'service_unavailable', 'message': 'Подбор временно недоступен'})
                return
            except Exception:
                self.respond(503, {'error': 'service_unavailable', 'message': 'Не удалось проверить подбор'})
                return
            self.respond(503 if result.get('status') == 'agent_unavailable' else 200, result)

    Handler.service = service
    return Handler


def serve(catalog, port, host='127.0.0.1'):
    handler = handler_for(catalog)
    server = ThreadingHTTPServer((host, port), handler)
    print(f'EventMatch: http://{host}:{port} (Ctrl+C)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        handler.service.store.close()
