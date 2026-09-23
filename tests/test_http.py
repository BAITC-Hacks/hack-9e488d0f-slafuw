from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from types import SimpleNamespace

from eventmatch.agent import EventMatchAgent
from eventmatch.catalog import ROOT, load_catalog
from eventmatch.server import handler_for
from eventmatch.service import EventMatchService


class HTTPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = EventMatchService(load_catalog())

        agent = EventMatchAgent(
            cls.service,
            SimpleNamespace(
                configured=False,
                model="not-called",
            ),
        )

        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            handler_for(
                cls.service.catalog,
                cls.service,
                agent,
            ),
        )

        cls.thread = threading.Thread(
            target=cls.server.serve_forever,
            daemon=True,
        )

        cls.thread.start()

        cls.base = (
            f"http://127.0.0.1:"
            f"{cls.server.server_port}"
        )

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.service.store.close()

    def post(self, data, path="/recommend"):
        request = Request(
            self.base + path,
            data=data,
            headers={
                "Content-Type": "application/json"
            },
        )

        return urlopen(
            request,
            timeout=5,
        )

    def test_static_ui_and_metadata(self):
        with urlopen(
            self.base,
            timeout=5,
        ) as response:
            html = response.read().decode()

        for text in (
            "eventmatch",
            "Параметры события",
            "Бюджет на подрядчика",
            "Дополнительные условия",
            "Почему подходит",
            "Цена «от»",
            'aria-live="polite"',
            "Категории нет в этом городе",
            "Подходящих вариантов нет",
            "no_category_in_city",
            "result.status==='matched'",
            (
                "const article="
                "document.createElement('article');"
                "article.className='contractor';"
            ),
        ):
            with self.subTest(text=text):
                self.assertIn(text, html)

        with urlopen(
            self.base + "/metadata",
            timeout=5,
        ) as response:
            metadata = json.load(response)

        self.assertEqual(
            metadata["calendar_end"],
            "2026-12-31",
        )

    def test_real_request(self):
        with self.post(
            (
                ROOT
                / "examples"
                / "01-dense.json"
            ).read_bytes()
        ) as response:
            result = json.load(response)

        self.assertEqual(
            result["status"],
            "matched",
        )

        self.assertEqual(
            len(result["cards"]),
            3,
        )

    def test_valid_empty_result_uses_200(self):
        with self.post(
            (
                ROOT
                / "examples"
                / "04-busy.json"
            ).read_bytes()
        ) as response:
            self.assertEqual(
                response.status,
                200,
            )

            self.assertEqual(
                json.load(response)["status"],
                "no_match",
            )

    def test_malformed_request_uses_422_with_explanation(self):
        payloads = (
            b"[]",
            b"{",
            b'{"event_date":"2027-01-01"}',
        )

        for payload in payloads:
            with (
                self.subTest(payload=payload),
                self.assertRaises(HTTPError) as caught,
            ):
                self.post(payload)

            self.assertEqual(
                caught.exception.code,
                422,
            )

            self.assertTrue(
                json.load(
                    caught.exception
                )["message"]
            )

    def test_chat_provider_failure_is_503_not_empty_match(self):
        body = json.dumps(
            {
                "message": "Нужен ведущий"
            }
        ).encode()

        with self.assertRaises(
            HTTPError
        ) as caught:
            self.post(
                body,
                "/chat",
            )

        self.assertEqual(
            caught.exception.code,
            503,
        )

        result = json.load(
            caught.exception
        )

        self.assertEqual(
            result["status"],
            "technical_error",
        )

        self.assertNotIn(
            "cards",
            result,
        )

        self.assertEqual(
            result["metadata"]["openai"]["mode"],
            "not_called",
        )

    def test_comparison_and_persisted_trace(self):
        request_data = json.loads(
            (
                ROOT
                / "examples"
                / "01-dense.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        body = {
            "request": request_data,
            "date_a": "2026-10-12",
            "date_b": "2026-10-13",
        }

        with self.post(
            json.dumps(body).encode(),
            "/compare-dates",
        ) as response:
            result = json.load(response)

        self.assertEqual(
            result["status"],
            "compared",
        )

        self.assertEqual(
            len(result["results"]),
            2,
        )

        self.assertIn(
            "HK-27222",
            result["appeared_ids"],
        )

        trace_url = (
            self.base
            + "/traces/"
            + result["trace_id"]
        )

        with urlopen(
            trace_url,
            timeout=5,
        ) as response:
            trace = json.load(response)

        self.assertEqual(
            trace["actions"],
            result["actions"],
        )

    def test_google_calendar_event(self):
        fake_event = {
            "id": "test-event-id",
            "title": "EventMatch Test",
            "link": (
                "https://calendar.google.com/test"
            ),
        }

        with patch(
            (
                "eventmatch.integrations."
                "google_calendar.create_event"
            ),
            return_value=fake_event,
        ) as create_event_mock:

            body = {
                "title": "EventMatch Test",
                "start_time": (
                    "2026-09-27T18:00:00+05:00"
                ),
                "end_time": (
                    "2026-09-27T19:00:00+05:00"
                ),
                "description": (
                    "Автоматический тест"
                ),
                "location": "Казахстан",
            }

            with self.post(
                json.dumps(
                    body,
                    ensure_ascii=False,
                ).encode("utf-8"),
                "/calendar/event",
            ) as response:
                result = json.load(response)

                self.assertEqual(
                    response.status,
                    201,
                )

            self.assertEqual(
                result["status"],
                "created",
            )

            self.assertEqual(
                result["event"]["id"],
                "test-event-id",
            )

            self.assertEqual(
                result["event"]["title"],
                "EventMatch Test",
            )

            create_event_mock.assert_called_once_with(
                title="EventMatch Test",
                start_time=(
                    "2026-09-27T18:00:00+05:00"
                ),
                end_time=(
                    "2026-09-27T19:00:00+05:00"
                ),
                description="Автоматический тест",
                location="Казахстан",
            )

    def test_google_calendar_missing_fields_uses_422(self):
        body = {
            "title": "EventMatch Test"
        }

        with self.assertRaises(
            HTTPError
        ) as caught:
            self.post(
                json.dumps(body).encode(),
                "/calendar/event",
            )

        self.assertEqual(
            caught.exception.code,
            422,
        )

        result = json.load(
            caught.exception
        )

        self.assertEqual(
            result["error"],
            "invalid_request",
        )

    def test_unknown_result_finalization_is_422(self):
        body = {
            "result_id": "unknown",
            "explanation_plan": {
                "cards": []
            },
        }

        with self.assertRaises(
            HTTPError
        ) as caught:
            self.post(
                json.dumps(body).encode(),
                "/finalize-result",
            )

        self.assertEqual(
            caught.exception.code,
            422,
        )


if __name__ == "__main__":
    unittest.main()