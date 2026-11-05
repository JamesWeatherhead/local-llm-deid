"""Synthetic regression checks for the documented extraction/mapping contract.

These checks establish reference-software behavior, not historical execution
provenance. No clinical inputs, checkpoints, or model server are used.
"""

from __future__ import annotations

import json
import http.client
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deid import pipeline  # noqa: E402
from deid.inference import LlamaServerClient, validate_request_timeout  # noqa: E402
from deid.run_model import _run_segments, process_document, run_model  # noqa: E402


def envelope(identifiers=None, finish_reason="stop", **extra):
    body = {"identifiers": [] if identifiers is None else identifiers}
    body.update(extra)
    return {
        "choices": [{"finish_reason": finish_reason,
                     "message": {"content": json.dumps(body)}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def name(literal):
    return {"exact_text": literal, "identifier_type": "01_NAME"}


class CoordinateContractTest(unittest.TestCase):
    def test_segments_count_unicode_codepoints_and_keep_exact_overlap(self):
        text = "a🧬é" * 3000
        segments = pipeline.fixed_segments(text)
        self.assertEqual([item.start for item in segments], [0, 3100, 6200])
        self.assertEqual([len(item.text) for item in segments], [3500, 3500, 2800])
        self.assertEqual(segments[0].text[-400:], segments[1].text[:400])
        self.assertEqual(segments[-1].end, len(text))

    def test_empty_source_retains_one_empty_segment(self):
        segments = pipeline.fixed_segments("")
        self.assertEqual(len(segments), 1)
        self.assertEqual((segments[0].start, segments[0].end, segments[0].text), (0, 0, ""))

    def test_segment_configuration_rejects_noninteger_or_invalid_windows(self):
        for size, overlap in [(True, 0), (4.0, 1), (4, False), (4, 1.0),
                              (0, 0), (-1, 0), (4, -1), (4, 4)]:
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(pipeline.PipelineError):
                    pipeline.fixed_segments("synthetic", size, overlap)
        with self.assertRaises(pipeline.PipelineError):
            pipeline.fixed_segments(None)

    def test_mapping_rejects_bounds_instead_of_truncating_slice(self):
        for start, end in [(-2, 3), (0, 5), (0, 0), (2, 1), (3, 3),
                           (True, 2), (0, 2.0)]:
            with self.subTest(start=start, end=end):
                self.assertIsNone(pipeline.map_interval([0, 1, 2], start, end))
        self.assertEqual(pipeline.map_interval([8, 9, 10], 0, 3), (8, 11))

    def test_mapping_rejects_invalid_source_coordinate_types(self):
        for mapping in [[0, True, 2], [0, -1, 2], [0, "1", 2], [0, 1.0, 2]]:
            with self.subTest(mapping=mapping):
                with self.assertRaises(pipeline.PipelineError):
                    pipeline.map_interval(mapping, 0, 3)

    def test_mapping_drops_marker_and_noncontiguous_matches(self):
        self.assertIsNone(pipeline.map_interval([0, None, 2], 0, 3))
        self.assertIsNone(pipeline.map_interval([0, 2, 3], 0, 3))

    def test_resolver_rejects_mismatched_maps_instead_of_redacting_prefix(self):
        for mapping in [[0, 1, 2], list(range(6))]:
            with self.subTest(mapping=mapping):
                with self.assertRaisesRegex(pipeline.PipelineError, "length"):
                    pipeline.resolve_pairs("ABCDE", mapping, [pipeline.Pair("ABCDE", "01_NAME")])

    def test_resolver_requires_nondecreasing_unique_source_positions(self):
        for mapping in [[0, 1, 1], [2, None, 1], [0, False, 2], [0, -1, 2]]:
            with self.subTest(mapping=mapping):
                with self.assertRaises(pipeline.PipelineError):
                    pipeline.resolve_pairs("ABC", mapping, [])

    def test_resolver_rejects_malformed_pairs(self):
        for pair in [pipeline.Pair("", "01_NAME"), pipeline.Pair("A", "unknown"),
                     pipeline.Pair(1, "01_NAME"), {"exact_text": "A"}]:
            with self.subTest(pair=pair):
                with self.assertRaisesRegex(pipeline.PipelineError, "pair"):
                    pipeline.resolve_pairs("ABC", [0, 1, 2], [pair])

    def test_exact_propagation_is_case_sensitive_and_has_no_word_boundary(self):
        source = "Ann Annual ann"
        predictions = pipeline.resolve_pairs(
            source, pipeline.identity_map(source), [pipeline.Pair("Ann", "01_NAME")])
        self.assertEqual([(item.start, item.end) for item in predictions], [(0, 3), (4, 7)])


class ParserAndGateContractTest(unittest.TestCase):
    def test_grounding_uses_supplied_segment_not_other_document_text(self):
        validated = pipeline.validate_response({"identifiers": [name("Alice")]}, "Bob")
        self.assertTrue(validated["envelope_usable"])
        self.assertTrue(validated["strict_schema_valid"])
        self.assertEqual(validated["pairs"], [])
        self.assertEqual(validated["counts"]["ungrounded_items"], 1)

    def test_duplicate_ungrounded_records_violate_schema_uniqueness(self):
        item = name("not in supplied segment")
        validated = pipeline.validate_response({"identifiers": [item, dict(item)]}, "Ann")
        self.assertTrue(validated["envelope_usable"])
        self.assertFalse(validated["strict_schema_valid"])
        self.assertEqual(validated["pairs"], [])
        self.assertEqual(validated["counts"]["ungrounded_items"], 2)
        self.assertEqual(validated["counts"]["rejected_items"], 2)
        # This counter remains the number of duplicate accepted/grounded pairs.
        self.assertEqual(validated["counts"]["duplicate_pairs"], 0)

    def test_invalid_records_do_not_hide_later_valid_records(self):
        validated = pipeline.validate_response(
            {"identifiers": [None, {"exact_text": "Alice"}, name("Alice"), name("Alice")]},
            "Alice")
        self.assertTrue(validated["envelope_usable"])
        self.assertFalse(validated["strict_schema_valid"])
        self.assertEqual(validated["pairs"], [pipeline.Pair("Alice", "01_NAME")])
        self.assertEqual(validated["counts"]["duplicate_pairs"], 1)
        self.assertEqual(validated["counts"]["rejected_items"], 2)

    def test_empty_identifier_list_is_valid_and_usable(self):
        validated = pipeline.validate_response({"identifiers": []}, "Alice")
        self.assertTrue(validated["envelope_usable"])
        self.assertTrue(validated["strict_schema_valid"])
        self.assertEqual(validated["pairs"], [])

    def test_parser_does_not_salvage_fences_prefixes_or_duplicate_keys(self):
        for content in ['```json\n{"identifiers":[]}\n```', 'Answer: {"identifiers":[]}',
                        '{"identifiers":[],"identifiers":[]}', '{"identifiers":NaN}']:
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    pipeline.strict_json_loads(content)
        validated = pipeline.validate_response(pipeline.strict_json_loads("[]"), "Alice")
        self.assertFalse(validated["envelope_usable"])

    def test_error_finish_reason_discards_detections_without_changing_format_flags(self):
        class Client:
            def complete(self, _text):
                return envelope([name("Alice")], finish_reason="error")

        pairs, records = _run_segments(Client(), pipeline.fixed_segments("Alice"))
        self.assertEqual(pairs, [])
        self.assertTrue(records[0]["envelope_usable"])
        self.assertTrue(records[0]["strict_schema_valid"])
        self.assertEqual(records[0]["finish_reason"], "error")

    def test_empty_body_and_valid_empty_json_are_distinct(self):
        class Client:
            def complete(self, _text):
                response = envelope()
                response["choices"][0]["message"]["content"] = ""
                return response

        pairs, records = _run_segments(Client(), pipeline.fixed_segments("Alice"))
        self.assertEqual(pairs, [])
        self.assertFalse(records[0]["envelope_usable"])
        self.assertFalse(records[0]["strict_schema_valid"])

    def test_late_success_does_not_bypass_failed_segment_in_pass2(self):
        responses = [envelope([name("Alice")]), envelope(),
                     envelope(finish_reason="length"), envelope([name("Bob")])]

        class Client:
            def complete(self, _text):
                return responses.pop(0)

        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp)
            process_document(Client(), "TEST", "Alice " + "." * 24 + " Bob", model_dir,
                             segment_size=20, overlap=4)
            pass1 = json.loads((model_dir / "resolved/TEST/resolved_predictions.json").read_text())
            cumulative = json.loads(
                (model_dir / "pass_2/resolved/TEST/resolved_predictions.json").read_text())
            self.assertEqual(cumulative, pass1)
            self.assertEqual(len(cumulative["predictions"]), 1)
        self.assertEqual(responses, [])

    def test_strict_schema_failure_alone_does_not_block_usable_pass2(self):
        responses = [envelope(), envelope([name("Alice")], extra_key=True)]

        class Client:
            def complete(self, _text):
                return responses.pop(0)

        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp)
            process_document(Client(), "TEST", "Alice", model_dir)
            validation = json.loads(
                (model_dir / "pass_2/validation/TEST/P2-S0001.validation.json").read_text())
            cumulative = json.loads(
                (model_dir / "pass_2/resolved/TEST/resolved_predictions.json").read_text())
            self.assertTrue(validation["envelope_usable"])
            self.assertFalse(validation["strict_schema_valid"])
            self.assertEqual(len(cumulative["predictions"]), 1)


class ReferenceClientContractTest(unittest.TestCase):
    @staticmethod
    def framing_client(responses):
        """Inject response bodies/framing failures without a model or network."""
        sequence = iter(responses)

        class Response:
            def __init__(self, value):
                self.value = value

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                if isinstance(self.value, Exception):
                    raise self.value
                return json.dumps(self.value).encode("utf-8")

        class Opener:
            def open(self, *_args, **_kwargs):
                return Response(next(sequence))

        client = LlamaServerClient("synthetic", "", "{{CLINICAL_TEXT}}", {})
        client._opener = Opener()
        return client

    def test_http_framing_failures_are_recorded_with_latency(self):
        for error in [http.client.IncompleteRead(b"partial", 100),
                      http.client.BadStatusLine("invalid synthetic status")]:
            with self.subTest(error=type(error).__name__):
                client = self.framing_client([error])
                with patch("deid.inference.time.monotonic", side_effect=[10, 10.25]):
                    response = client.complete("synthetic")
                self.assertEqual(response["choices"][0]["finish_reason"], "error")
                self.assertEqual(response["choices"][0]["message"]["content"], "")
                self.assertEqual(response["transport_error"], type(error).__name__)
                self.assertEqual(response["latency_seconds"], 0.25)

    def test_truncated_request_preserves_valid_other_pass1_detections(self):
        client = self.framing_client([
            envelope([name("Ann")]), http.client.IncompleteRead(b"partial", 100),
            envelope(), envelope(),
        ])
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp)
            process_document(client, "TEST", "Ann " + "." * 25, model_dir,
                             segment_size=20, overlap=4)
            pass1 = json.loads((model_dir / "resolved/TEST/resolved_predictions.json").read_text())
            cumulative = json.loads((model_dir / "pass_2/resolved/TEST/resolved_predictions.json").read_text())
            self.assertEqual([(p["start"], p["end"]) for p in pass1["predictions"]], [(0, 3)])
            self.assertEqual(cumulative, pass1)
            raw = json.loads((model_dir / "raw_responses/TEST/P1-S0002.raw.json").read_text())
            self.assertEqual(raw["transport_error"], "IncompleteRead")
            self.assertEqual(len(list(model_dir.rglob("*.raw.json"))), 4)

    def test_truncated_pass2_request_keeps_pass1_and_rejects_other_new_detections(self):
        client = self.framing_client([
            envelope([name("Ann")]), envelope(),
            http.client.IncompleteRead(b"partial", 100), envelope([name("Bob")]),
        ])
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp)
            process_document(client, "TEST", "Ann " + "." * 22 + " Bob", model_dir,
                             segment_size=20, overlap=4)
            pass1 = json.loads((model_dir / "resolved/TEST/resolved_predictions.json").read_text())
            cumulative = json.loads((model_dir / "pass_2/resolved/TEST/resolved_predictions.json").read_text())
            self.assertEqual(cumulative, pass1)
            self.assertEqual(len(cumulative["predictions"]), 1)
            good = json.loads((model_dir / "pass_2/validation/TEST/P2-S0002.validation.json").read_text())
            failed = json.loads((model_dir / "pass_2/validation/TEST/P2-S0001.validation.json").read_text())
            self.assertTrue(good["envelope_usable"])
            self.assertEqual(failed["finish_reason"], "error")

    def test_full_note_plan_exists_before_first_request_and_contains_no_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            notes = root / "notes"
            notes.mkdir()
            for document_id, text in [("TEST1", "synthetic Alice"), ("TEST2", "synthetic Bob")]:
                (notes / f"{document_id}.txt").write_text(text, encoding="utf-8")

            class InterruptedClient:
                def complete(inner_self, _text):
                    manifest_path = root / "predictions/test-model/inference_manifest.json"
                    manifest = json.loads(manifest_path.read_text())
                    self.assertEqual(manifest["notes"], [
                        {"document_id": "TEST1", "source_sha256": pipeline.sha256_text("synthetic Alice")},
                        {"document_id": "TEST2", "source_sha256": pipeline.sha256_text("synthetic Bob")},
                    ])
                    self.assertNotIn("synthetic Alice", manifest_path.read_text())
                    self.assertNotIn("synthetic Bob", manifest_path.read_text())
                    raise RuntimeError("synthetic interruption")

            with patch("deid.run_model.make_client", return_value=InterruptedClient()):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    run_model("test-model", notes, root / "predictions", REPO_ROOT / "protocol",
                              use_stub=True)

    def test_invalid_note_input_cannot_delete_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            notes = root / "notes"
            notes.mkdir()
            (notes / "bad name.txt").write_text("synthetic")
            model_dir = root / "predictions/test-model"
            model_dir.mkdir(parents=True)
            marker = model_dir / "existing.txt"
            marker.write_text("keep me")
            with self.assertRaisesRegex(ValueError, "unexpected note id"):
                run_model("test-model", notes, root / "predictions", REPO_ROOT / "protocol",
                          use_stub=True, overwrite_model_output=True)
            self.assertEqual(marker.read_text(), "keep me")

    def test_optional_timeout_must_be_finite_positive_and_not_boolean(self):
        for value in [None, 1, 0.1]:
            self.assertEqual(validate_request_timeout(value), value)
        for value in [0, -1, True, False, "1", float("nan"), float("inf"),
                      float("-inf"), 10 ** 1000]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_request_timeout(value)

    def test_invalid_timeout_cannot_delete_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir = root / "predictions/test-model"
            model_dir.mkdir(parents=True)
            marker = model_dir / "existing.txt"
            marker.write_text("keep me")
            with self.assertRaisesRegex(ValueError, "request_timeout"):
                run_model("test-model", REPO_ROOT / "fixtures/synthetic/notes", root / "predictions",
                          REPO_ROOT / "protocol", use_stub=True, request_timeout=-1,
                          overwrite_model_output=True)
            self.assertEqual(marker.read_text(), "keep me")

    def test_measured_latency_overrides_untrusted_server_duration(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return b'{"choices":[],"latency_seconds":-999}'

        class Opener:
            def open(self, *_args, **_kwargs):
                return Response()

        client = LlamaServerClient("test", "", "{{CLINICAL_TEXT}}", {})
        client._opener = Opener()
        with patch("deid.inference.time.monotonic", side_effect=[10, 10.25]):
            response = client.complete("synthetic")
        self.assertEqual(response["latency_seconds"], 0.25)

    def test_http_redirect_never_contacts_an_unselected_endpoint(self):
        visited = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                visited.append(self.path)
                self.send_response(302)
                self.send_header("Location", "/unselected")
                self.end_headers()

            def do_GET(self):
                visited.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"choices":[]}')

            def log_message(self, *_args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = LlamaServerClient("test", "", "{{CLINICAL_TEXT}}", {},
                                       api_base=f"http://127.0.0.1:{server.server_port}",
                                       request_timeout=1)
            response = client.complete("synthetic")
            self.assertEqual(response["choices"][0]["finish_reason"], "error")
            self.assertEqual(response["transport_error"], "HTTPError")
            self.assertEqual(visited, ["/v1/chat/completions"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
