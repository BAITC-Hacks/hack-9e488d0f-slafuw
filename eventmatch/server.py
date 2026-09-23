"""Local demo adapter; domain logic is independent of HTTP and UI."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json

from .catalog import ROOT
from .engine import POLICY_VERSION, recommend
from .models import END, FORMATS, LANGUAGES, START


def handler_for(catalog):
    class Handler(BaseHTTPRequestHandler):
        def respond(
            self,
            status,
            payload,
            content_type="application/json; charset=utf-8",
        ):
            body = (
                payload
                if isinstance(payload, bytes)
                else json.dumps(payload, ensure_ascii=False).encode("utf-8")
            )

            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def read_json(self):
            length = int(self.headers.get("Content-Length", "0"))

            if not 0 < length <= 16384:
                raise ValueError(
                    "Требуется JSON-запрос размером от 1 до 16384 байт"
                )

            return json.loads(
                self.rfile.read(length).decode("utf-8")
            )

        def do_GET(self):
            if self.path == "/":
                self.respond(
                    200,
                    (ROOT / "web" / "index.html").read_bytes(),
                    "text/html; charset=utf-8",
                )

            elif self.path == "/health":
                self.respond(
                    200,
                    {
                        "status": "ok",
                        "profiles": len(catalog.profiles),
                        "dataset_sha256": catalog.sha256,
                        "policy": POLICY_VERSION,
                    },
                )

            elif self.path == "/metadata":
                self.respond(
                    200,
                    {
                        "cities": sorted(
                            {p.city for p in catalog.profiles}
                        ),
                        "categories": sorted(
                            {
                                c
                                for p in catalog.profiles
                                for c in p.categories
                            }
                        ),
                        "event_formats": FORMATS,
                        "languages": LANGUAGES,
                        "calendar_start": str(START),
                        "calendar_end": str(END),
                    },
                )

            else:
                self.respond(
                    404,
                    {
                        "error": "not_found",
                        "message": "Маршрут не найден",
                    },
                )

        def do_POST(self):

            # -------------------------
            # EVENTMATCH RECOMMENDATIONS
            # -------------------------

            if self.path == "/recommend":
                try:
                    raw = self.read_json()
                    result = recommend(catalog, raw)

                except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
                    self.respond(
                        422,
                        {
                            "error": "invalid_request",
                            "message": str(exc),
                        },
                    )
                    return

                self.respond(200, result)
                return

            # -------------------------
            # GOOGLE CALENDAR
            # -------------------------

            if self.path == "/calendar/event":
                try:
                    raw = self.read_json()

                    title = str(raw.get("title", "")).strip()
                    start_time = str(raw.get("start_time", "")).strip()
                    end_time = str(raw.get("end_time", "")).strip()

                    description = str(
                        raw.get("description", "")
                    ).strip()

                    location = str(
                        raw.get("location", "")
                    ).strip()

                    if not title:
                        raise ValueError(
                            "Не указано название события"
                        )

                    if not start_time:
                        raise ValueError(
                            "Не указано время начала"
                        )

                    if not end_time:
                        raise ValueError(
                            "Не указано время окончания"
                        )

                    from .integrations.google_calendar import create_event

                    event = create_event(
                        title=title,
                        start_time=start_time,
                        end_time=end_time,
                        description=description,
                        location=location,
                    )

                    self.respond(
                        201,
                        {
                            "status": "created",
                            "event": event,
                        },
                    )

                except (
                    ValueError,
                    UnicodeError,
                    json.JSONDecodeError,
                ) as exc:
                    self.respond(
                        422,
                        {
                            "error": "invalid_request",
                            "message": str(exc),
                        },
                    )

                except Exception as exc:
                    print(
                        f"Google Calendar error: {exc}",
                        flush=True,
                    )

                    self.respond(
                        502,
                        {
                            "error": "calendar_error",
                            "message": (
                                "Не удалось создать событие "
                                "в Google Calendar"
                            ),
                        },
                    )

                return

            # -------------------------
            # UNKNOWN ROUTE
            # -------------------------

            self.respond(
                404,
                {
                    "error": "not_found",
                    "message": "Маршрут не найден",
                },
            )

    return Handler


def serve(catalog, port):
    server = ThreadingHTTPServer(
        ("127.0.0.1", port),
        handler_for(catalog),
    )

    print(
        f"EventMatch: http://127.0.0.1:{port} "
        "(остановка: Ctrl+C)",
        flush=True,
    )

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        pass

    finally:
        server.server_close()