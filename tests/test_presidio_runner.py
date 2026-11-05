import json
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deid.presidio_runner import process_document, run_comparator


def hit(start, end):
    return SimpleNamespace(start=start, end=end)


class FakeAnalyzer:
    def __init__(self, callback):
        self.callback = callback
        self.calls = []

    def analyze(self, **kwargs):
        self.calls.append(kwargs)
        return self.callback(kwargs["text"], len(self.calls))


class PresidioRunnerTests(unittest.TestCase):
    def test_offset_only_two_pass_mapping_and_timing(self):
        source = "Ann met Ann and Bob."
        analyzer = FakeAnalyzer(lambda text, n: [hit(0, 3)] if n == 1 else [hit(text.index("Bob"), text.index("Bob") + 3)])
        ticks = iter([10.0, 10.25])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = process_document(analyzer, "TEST001", source, root, clock=lambda: next(ticks))
            p1 = json.loads((root / "resolved/TEST001/resolved_predictions.json").read_text())["predictions"]
            p2 = json.loads((root / "pass_2/resolved/TEST001/resolved_predictions.json").read_text())["predictions"]
            # Only the returned occurrence is removed, not the second 'Ann'.
            self.assertEqual([(x["start"], x["end"]) for x in p1], [(0, 3)])
            self.assertEqual([(x["start"], x["end"]) for x in p2], [(0, 3), (16, 19)])
            self.assertEqual(analyzer.calls[1]["text"], "[PHI] met Ann and Bob.")
            self.assertTrue(all(set(call) == {"text", "language"} for call in analyzer.calls))
            self.assertEqual(record["pipeline_seconds"], 0.25)
            for path in root.rglob("*.json"):
                self.assertNotIn("Ann", path.read_text())
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_overlapping_segments_merge_duplicates_without_propagation(self):
        analyzer = FakeAnalyzer(lambda text, n: [hit(text.index("XYZ"), text.index("XYZ") + 3)] if "XYZ" in text else [])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process_document(analyzer, "T", "aaaaXYZbbbb", root, segment_size=8, overlap=4)
            obj = json.loads((root / "resolved/T/resolved_predictions.json").read_text())
            self.assertEqual([(p["start"], p["end"]) for p in obj["predictions"]], [(4, 7)])

    def test_marker_matches_are_rejected(self):
        analyzer = FakeAnalyzer(lambda text, n: [hit(0, 3)] if n == 1 else [hit(0, 5)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process_document(analyzer, "T", "Ann Bob", root)
            obj = json.loads((root / "pass_2/resolved/T/resolved_predictions.json").read_text())
            self.assertEqual([(p["start"], p["end"]) for p in obj["predictions"]], [(0, 3)])

    def test_invalid_offsets_fail_without_completed_output(self):
        for result in [hit(-1, 3), hit(0, 999), hit(True, 3), hit(2, 2)]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with self.assertRaises(ValueError):
                    process_document(FakeAnalyzer(lambda *_: [result]), "T", "Ann", root)
                self.assertFalse((root / "pipeline_timing/T.json").exists())

    def test_failed_second_pass_does_not_manufacture_completion(self):
        def callback(text, n):
            if n == 2:
                raise RuntimeError("analyzer failed")
            return []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(RuntimeError):
                process_document(FakeAnalyzer(callback), "T", "Ann", root)
            self.assertFalse((root / "pass_2/resolved/T/resolved_predictions.json").exists())
            self.assertFalse((root / "pipeline_timing/T.json").exists())

    def test_manifest_fresh_output_and_safe_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notes = root / "notes"
            notes.mkdir()
            (notes / "T.txt").write_text("Ann", encoding="utf-8")
            model = run_comparator(notes, root / "out", analyzer=FakeAnalyzer(lambda *_: []))
            manifest = json.loads((model / "comparator_manifest.json").read_text())
            self.assertEqual(manifest["backend"], "presidio")
            self.assertEqual(manifest["notes"][0]["document_id"], "T")
            self.assertTrue(manifest["injected_test_analyzer"])
            self.assertFalse((model / "raw_responses").exists())
            with self.assertRaises(FileExistsError):
                run_comparator(notes, root / "out", analyzer=FakeAnalyzer(lambda *_: []))
            with self.assertRaises(ValueError):
                run_comparator(notes, root / "out", "../escape", analyzer=FakeAnalyzer(lambda *_: []))


if __name__ == "__main__":
    unittest.main()
