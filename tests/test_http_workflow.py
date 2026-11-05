"""End-to-end protocol/transport/resolution/scoring contract on a local test server."""

import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deid.metrics import score_all
from deid.run_model import run_model, write_json

spec = importlib.util.spec_from_file_location("http_test_timing", ROOT / "scripts/summarize_timing.py")
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


class HttpWorkflowTest(unittest.TestCase):
    def test_real_local_http_two_pass_workflow_and_measured_timing(self):
        payloads = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payloads.append(payload)
                literal = "Ann" if len(payloads) == 1 else "Bob"
                body = {"choices": [{"finish_reason": "stop", "message": {
                    "content": json.dumps({"identifiers": [{"exact_text": literal,
                                                            "identifier_type": "01_NAME"}]}),
                    "reasoning_content": "not extraction content",
                }}], "usage": {"completion_tokens": 12}, "latency_seconds": 999999}
                encoded = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                notes, gold = root / "notes", root / "gold"
                notes.mkdir()
                (notes / "TEST.txt").write_text("Ann met Bob.", encoding="utf-8")
                write_json(gold / "TEST.json", {
                    "text": "Ann met Bob.",
                    "denotations": [{"id": "A", "span": {"begin": 0, "end": 3}},
                                    {"id": "B", "span": {"begin": 8, "end": 11}}],
                    "attributes": [{"subj": key, "pred": "identifier_type", "obj": "01_NAME"}
                                   for key in ("A", "B")],
                })
                output = root / "predictions"
                model = run_model("harness", notes, output, ROOT / "protocol", use_stub=False,
                                  api_base=f"http://127.0.0.1:{server.server_port}")
                self.assertEqual(len(payloads), 2)
                for payload in payloads:
                    self.assertEqual(payload["temperature"], 0)
                    self.assertEqual(payload["seed"], 42)
                    self.assertEqual(payload["reasoning_effort"], "low")
                    self.assertEqual(payload["response_format"]["type"], "json_schema")
                    self.assertNotIn("max_tokens", payload)
                    self.assertNotIn("max_completion_tokens", payload)
                self.assertIn("[PHI] met Bob.", payloads[1]["messages"][-1]["content"])
                results = score_all(output, gold)
                headline = [r for r in results["metrics"] if r["pass"] == "cumulative_pass2"
                            and r["cohort"] == "all_expected" and r["phi_type"] == "ALL"][0]
                self.assertEqual((headline["char_tp"], headline["char_fp"], headline["char_fn"]), (6, 0, 0))
                self.assertEqual(headline["zero_residual_notes"], 1)
                summary = timing.summarize([output])[0]
                self.assertEqual(summary["n_notes"], 1)
                self.assertGreater(summary["mean_median_seconds_per_note"], 0)
                self.assertLess(summary["mean_median_seconds_per_note"], 999999)
                for path in model.rglob("*.raw.json"):
                    content = path.read_text()
                    self.assertNotIn("Ann", content)
                    self.assertNotIn("Bob", content)
                    self.assertNotIn("reasoning_content", content)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
