from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from eventmatch.catalog import ROOT, load_catalog
from eventmatch.server import handler_for


class HTTPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(load_catalog()))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def post(self, data):
        request = Request(self.base + "/recommend", data=data,
                          headers={"Content-Type": "application/json"})
        return urlopen(request, timeout=5)

    def test_static_ui_and_metadata(self):
        with urlopen(self.base, timeout=5) as response:
            html = response.read().decode()
        for text in ("eventmatch", "Параметры события", "Бюджет на подрядчика",
                     "Дополнительные условия", "Почему подходит", "Цена «от»",
                     "aria-live=\"polite\"", "Категории нет в этом городе",
                     "Подходящих вариантов нет", "no_category_in_city",
                     "result.status==='matched'",
                     "const article=document.createElement('article');article.className='contractor';"):
            with self.subTest(text=text):
                self.assertIn(text, html)
        with urlopen(self.base + "/metadata", timeout=5) as response:
            self.assertEqual(json.load(response)["calendar_end"], "2026-12-31")

    def test_real_request(self):
        with self.post((ROOT / "examples" / "01-dense.json").read_bytes()) as response:
            result = json.load(response)
        self.assertEqual(result["status"], "matched")
        self.assertEqual(len(result["cards"]), 3)

    def test_valid_empty_result_uses_200(self):
        with self.post((ROOT / "examples" / "04-busy.json").read_bytes()) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.load(response)["status"], "no_match")

    def test_malformed_request_uses_422_with_explanation(self):
        for payload in (b"[]", b"{", b'{"event_date":"2027-01-01"}'):
            with self.subTest(payload=payload), self.assertRaises(HTTPError) as caught:
                self.post(payload)
            self.assertEqual(caught.exception.code, 422)
            self.assertTrue(json.load(caught.exception)["message"])


if __name__ == "__main__":
    unittest.main()
