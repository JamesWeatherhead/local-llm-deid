import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("summarize_timing", ROOT / "scripts/summarize_timing.py")
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)

from deid.pipeline import fixed_segments, sha256_text
from deid.provenance import segment_manifest
from deid.run_model import write_json


class TimingTests(unittest.TestCase):
    def create_run(self, root, values, backend="llama-server"):
        model = root / "model"
        sources = {"A": "Ann", "B": "Bob"}
        notes = [{"document_id": k, "source_sha256": sha256_text(v)} for k, v in sources.items()]
        comparator = backend == "presidio"
        write_json(model / ("comparator_manifest.json" if comparator else "inference_manifest.json"),
                   {"schema_version": "comparator-manifest-v1" if comparator else "inference-manifest-v1",
                    "model_id": "model", "backend": backend, "notes": notes})
        for doc, source in sources.items():
            if comparator:
                write_json(model / "pipeline_timing" / f"{doc}.json",
                           {"document_id": doc, "source_sha256": sha256_text(source),
                            "completed": True, "pipeline_seconds": sum(values[doc])})
            else:
                for p in (1, 2):
                    base = model if p == 1 else model / "pass_2"
                    write_json(base / "expected_segments" / doc / "manifest.json",
                               segment_manifest(doc, source, source, p, fixed_segments(source), 3500, 400))
                    write_json(base / "raw_responses" / doc / f"P{p}-S0001.raw.json",
                               {"latency_seconds": values[doc][p - 1]})
        return model

    def test_mean_of_matched_execution_note_medians(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            self.create_run(a, {"A": [1, 2], "B": [4, 5]})  # median6
            self.create_run(b, {"A": [2, 3], "B": [5, 6]})  # median8
            result = timing.summarize([a, b])[0]
            self.assertEqual(result["mean_median_seconds_per_note"], 7)
            self.assertEqual(json.loads(result["execution_medians_seconds"]), [6, 8])
            self.assertEqual(result["n_notes"], 2)

    def test_missing_request_or_latency_never_becomes_zero(self):
        for mode in ("file", "field", "negative", "nan"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                model = self.create_run(root, {"A": [1, 2], "B": [4, 5]})
                path = model / "pass_2/raw_responses/A/P2-S0001.raw.json"
                if mode == "file":
                    path.unlink()
                else:
                    path.write_text(json.dumps({} if mode == "field" else {"latency_seconds": -1 if mode == "negative" else float("nan")}))
                with self.assertRaises(ValueError):
                    timing.summarize([root])

    def test_source_identity_and_duplicate_execution_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            self.create_run(a, {"A": [1, 2], "B": [4, 5]}, "presidio")
            model = self.create_run(b, {"A": [1, 2], "B": [4, 5]}, "presidio")
            with self.assertRaises(ValueError):
                timing.summarize([a, a])
            path = model / "comparator_manifest.json"
            obj = json.loads(path.read_text())
            obj["notes"][0]["source_sha256"] = "0" * 64
            write_json(path, obj)
            with self.assertRaises(ValueError):
                timing.summarize([a, b])

    def test_comparator_separate_method_and_incomplete_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = self.create_run(root, {"A": [1, 2], "B": [4, 5]}, "presidio")
            self.assertEqual(timing.summarize([root])[0]["timing_method"], "full_comparator_pipeline")
            (model / "pipeline_timing/A.json").unlink()
            with self.assertRaises(ValueError):
                timing.summarize([root])


if __name__ == "__main__":
    unittest.main()
