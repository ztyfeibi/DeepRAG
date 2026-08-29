import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from retriever import HyperGraphHTTPRetriever, HyperGraphHTTPRetrieverError


class _Handler(BaseHTTPRequestHandler):
    status = 200
    body = {"context": "context evidence"}
    calls = 0

    def do_POST(self):
        type(self).calls += 1
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        body = json.dumps(type(self).body).encode("utf-8")
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class HyperGraphHTTPRetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.endpoint = "http://127.0.0.1:{}".format(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def setUp(self):
        _Handler.calls = 0

    def test_context_is_one_evidence_block(self):
        _Handler.status, _Handler.body = 200, {"context": "context evidence"}
        _, docs = HyperGraphHTTPRetriever(self.endpoint).retrieve(["question"])
        self.assertEqual(docs.shape, (1, 1))
        self.assertEqual(docs[0][0], "context evidence")

    def test_four_xx_is_not_retried(self):
        _Handler.status, _Handler.body = 400, {"error": "bad request"}
        with self.assertRaises(HyperGraphHTTPRetrieverError):
            HyperGraphHTTPRetriever(self.endpoint, max_retries=2).retrieve(["question"])
        self.assertEqual(_Handler.calls, 1)

    def test_five_xx_is_retried(self):
        _Handler.status, _Handler.body = 503, {"error": "busy"}
        with self.assertRaises(HyperGraphHTTPRetrieverError):
            HyperGraphHTTPRetriever(self.endpoint, max_retries=2).retrieve(["question"])
        self.assertEqual(_Handler.calls, 3)

    def test_empty_context_errors_without_fallback(self):
        _Handler.status, _Handler.body = 200, {"context": ""}
        with self.assertRaises(HyperGraphHTTPRetrieverError):
            HyperGraphHTTPRetriever(self.endpoint).retrieve(["question"])

    def test_connection_failure_errors(self):
        with self.assertRaises(HyperGraphHTTPRetrieverError):
            HyperGraphHTTPRetriever("http://127.0.0.1:1", timeout=0.01,
                                     max_retries=0).retrieve(["question"])


if __name__ == "__main__":
    unittest.main()
