"""Synthetic tests for reference identity, cumulative masks, and scoring denominators."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deid import metrics
from deid.pipeline import PREDICTIONS_SCHEMA_VERSION
from deid.presidio_runner import run_comparator


def gold_document(text="Ada May 01/02", spans=((0, 3, "01_NAME"), (8, 13, "03_DATE"))):
    return {"sourceid": "NOTE1", "text": text,
            "denotations": [{"id": f"T{i}", "obj": "PHI_IDENTIFIER", "span": {"begin": a, "end": b}}
                            for i, (a, b, _) in enumerate(spans)],
            "attributes": [{"subj": f"T{i}", "pred": "identifier_type", "obj": code}
                           for i, (_, _, code) in enumerate(spans)]}


def write_prediction(model, pass_number, predictions, text="Ada May 01/02"):
    base = model if pass_number == 1 else model / "pass_2"
    path = base / "resolved" / "NOTE1" / "resolved_predictions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": PREDICTIONS_SCHEMA_VERSION,
        "document_id": "NOTE1", "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "offset_unit": "Unicode codepoint; half-open [start,end)", "contains_literal_text": False,
        "predictions": [{"start": a, "end": b, "identifier_type": code} for a, b, code in predictions]}))


class ReferenceIntegrityTest(unittest.TestCase):
    def test_raw_literal_boundaries_are_distinct_from_union_coverage(self):
        gold = {"length": 5, "all": metrics.span_mask(0, 5),
                "type": {"01_NAME": metrics.span_mask(0, 5)},
                "spans": [(0, 5, "01_NAME")]}
        separate = metrics.score_document(gold, [(0, 3, "01_NAME"), (2, 5, "01_NAME")])
        merged = metrics.score_document(gold, [(0, 5, "01_NAME")])
        self.assertEqual(separate["tp"], merged["tp"])
        self.assertEqual(separate["all_spans"]["full"], 1)
        self.assertEqual(separate["all_spans"]["strict"], 0)
        self.assertEqual(merged["all_spans"]["strict"], 1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gold_dir = self.root / "gold"
        self.gold_dir.mkdir()
        self.gold_path = self.gold_dir / "NOTE1.json"
        self.write_gold(gold_document())
        self.model = self.root / "predictions" / "model"
        self.model.mkdir(parents=True)

    def write_gold(self, doc):
        self.gold_path.write_text(json.dumps(doc))

    def headline(self, results, pass_label):
        return next(row for row in results["metrics"] if row["pass"] == pass_label
                    and row["cohort"] == "all_expected" and row["phi_type"] == "ALL")

    def test_missing_pass2_preserves_first_pass_detections(self):
        write_prediction(self.model, 1, [(0, 3, "01_NAME")])
        results = metrics.score_all(self.model.parent, self.gold_dir)
        first = self.headline(results, "pass1")
        second = self.headline(results, "cumulative_pass2")
        self.assertEqual((first["char_tp"], second["char_tp"]), (3, 3))
        self.assertEqual(second["gt_phi_chars"], 8)
        self.assertEqual(second["n_notes"], 1)

    def test_missing_both_passes_is_full_miss_not_excluded(self):
        results = metrics.score_all(self.model.parent, self.gold_dir)
        row = self.headline(results, "cumulative_pass2")
        self.assertEqual((row["char_tp"], row["char_fn"], row["char_f1"]), (0, 8, 0))
        self.assertIsNone(row["char_precision"])
        self.assertEqual(row["n_notes"], 1)

    def test_present_pass2_cannot_remove_first_pass_coverage(self):
        write_prediction(self.model, 1, [(0, 3, "01_NAME")])
        write_prediction(self.model, 2, [(8, 13, "03_DATE")])
        with self.assertRaisesRegex(metrics.PredictionArtifactError, "removes valid Pass 1 coverage"):
            metrics.score_all(self.model.parent, self.gold_dir)

    def test_merged_pass2_can_preserve_coverage_with_different_boundaries(self):
        write_prediction(self.model, 1, [(0, 3, "01_NAME")])
        write_prediction(self.model, 2, [(0, 13, "01_NAME")])
        row = self.headline(metrics.score_all(self.model.parent, self.gold_dir), "cumulative_pass2")
        self.assertEqual((row["char_tp"], row["char_fp"], row["char_fn"]), (8, 5, 0))
        self.assertEqual(row["over_redaction_rate"], 5 / 13)
        self.assertEqual(row["nonreference_removal_rate"], 1)
        self.assertEqual(row["zero_residual_notes"], 1)
        self.assertEqual(row["full_span_recall"], 1)
        self.assertEqual(row["strict_span_recall"], 0)

    def test_invalid_gold_never_silently_discards_annotations(self):
        cases = []
        for start, end in [(False, 3), (0.5, 3), (-1, 3), (3, 3), (0, 14)]:
            doc = gold_document()
            doc["denotations"][0]["span"] = {"begin": start, "end": end}
            cases.append(doc)
        doc = gold_document(); del doc["denotations"][0]["span"]; cases.append(doc)
        doc = gold_document(); doc["attributes"].pop(); cases.append(doc)
        doc = gold_document(); doc["attributes"][0]["obj"] = "UNKNOWN"; cases.append(doc)
        doc = gold_document(); doc["sourceid"] = "OTHER"; cases.append(doc)
        doc = gold_document(); doc["denotations"][1]["id"] = "T0"; cases.append(doc)
        doc = gold_document(); doc["attributes"].append(copy.deepcopy(doc["attributes"][0])); cases.append(doc)
        doc = gold_document(); del doc["denotations"][0]["span"]
        doc["denotations"][0]["spans"] = [{"begin": 0, "end": 1}, {"begin": 2, "end": 3}]; cases.append(doc)
        for doc in cases:
            with self.subTest(doc=doc):
                self.write_gold(doc)
                with self.assertRaises(ValueError):
                    metrics.load_gold(self.gold_path)

    def test_reference_fingerprint_changes_even_with_equal_denominators(self):
        first = metrics.load_gold(self.gold_path)
        doc = gold_document(spans=((1, 4, "01_NAME"), (8, 13, "03_DATE")))
        self.write_gold(doc)
        second = metrics.load_gold(self.gold_path)
        self.assertEqual(metrics.popcount(first["all"]), metrics.popcount(second["all"]))
        self.assertNotEqual(first["reference_sha256"], second["reference_sha256"])
        self.assertNotEqual(metrics.reference_set_sha256({"NOTE1": first}), metrics.reference_set_sha256({"NOTE1": second}))
        doc["denotations"].reverse(); doc["attributes"].reverse(); self.write_gold(doc)
        self.assertEqual(second["reference_sha256"], metrics.load_gold(self.gold_path)["reference_sha256"])

    def test_exact_precision_counts_matched_predictions_not_duplicated_gold_hits(self):
        self.write_gold(gold_document(spans=((0, 3, "01_NAME"), (0, 3, "02_GEOGRAPHIC_SUBDIVISION"))))
        write_prediction(self.model, 1, [(0, 3, "01_NAME")])
        row = self.headline(metrics.score_all(self.model.parent, self.gold_dir), "pass1")
        self.assertEqual((row["strict_span_recall"], row["strict_span_precision"]), (1, 1))
        self.assertEqual((row["gt_span_count"], row["pred_span_count"]), (2, 1))

    def test_relaxed_matching_is_nonexclusive_and_character_scoring_uses_union(self):
        self.write_gold(gold_document(spans=((0, 3, "01_NAME"), (1, 4, "01_NAME"))))
        write_prediction(self.model, 1, [(0, 4, "03_DATE"), (0, 3, "03_DATE")])
        row = self.headline(metrics.score_all(self.model.parent, self.gold_dir), "pass1")
        self.assertEqual((row["char_tp"], row["gt_phi_chars"], row["pred_phi_chars"]), (4, 4, 4))
        self.assertEqual((row["relaxed_span_recall"], row["relaxed_span_precision"]), (1, 1))
        name_row = next(r for r in metrics.score_all(self.model.parent, self.gold_dir)["metrics"]
                        if r["pass"] == "pass1" and r["cohort"] == "all_expected" and r["phi_type"] == "01_NAME")
        self.assertEqual(name_row["char_recall"], 1)  # grouped by gold, not predicted type
        self.assertEqual(name_row["char_recall_typematched"], 0)

    def test_untyped_predictions_require_comparator_manifest(self):
        write_prediction(self.model, 1, [(0, 3, "UNSPECIFIED")])
        with self.assertRaisesRegex(metrics.PredictionArtifactError, "require a Presidio"):
            metrics.score_all(self.model.parent, self.gold_dir)
        (self.model / "comparator_manifest.json").write_text(json.dumps({
            "schema_version": "comparator-manifest-v1", "backend": "presidio"}))
        results = metrics.score_all(self.model.parent, self.gold_dir)
        self.assertEqual(self.headline(results, "pass1")["char_tp"], 3)
        row = next(r for r in results["metrics"] if r["pass"] == "pass1"
                   and r["cohort"] == "all_expected" and r["phi_type"] == "01_NAME")
        self.assertIsNone(row["char_precision_typematched"])

    def test_type_matched_precision_includes_predictions_on_notes_without_that_gold_type(self):
        first = metrics.score_document(metrics.load_gold(self.gold_path), [(0, 3, "01_NAME")])
        doc = gold_document(text="ordinary", spans=((0, 2, "03_DATE"),))
        doc["sourceid"] = "NOTE2"
        second_path = self.gold_dir / "NOTE2.json"
        second_path.write_text(json.dumps(doc))
        second = metrics.score_document(metrics.load_gold(second_path), [(0, 4, "01_NAME")])
        ops = metrics.doc_operations(self.model, "NOTE1", 1)
        aggregate = metrics.aggregate([("NOTE1", first, ops), ("NOTE2", second, ops)])["all_expected"]
        rows = metrics.emit_rows({}, "pass1", "all_expected", aggregate, ["01_NAME", "03_DATE"])
        name = next(row for row in rows if row["phi_type"] == "01_NAME")
        self.assertEqual(name["char_precision_typematched"], 3 / 7)
        self.assertEqual(name["char_recall"], 1)

    def test_written_tables_and_manifest_bind_actual_reference_set(self):
        results = metrics.score_all(self.model.parent, self.gold_dir)
        metrics.write_outputs(self.root / "out", self.gold_dir, results)
        manifest = json.loads((self.root / "out" / "run_manifest.json").read_text())
        self.assertEqual(manifest["reference_denominators"][0]["n_notes"], 1)
        self.assertEqual(manifest["reference_sets"], [results["per_doc"][0]["reference_set_sha256"]])
        self.assertIn("reference_sha256", results["per_doc"][0])
        self.assertNotIn("code_version", manifest)
        self.assertEqual(manifest["scorer_sha256"], metrics.file_sha256(Path(metrics.__file__)))
        self.assertIn("repository_commit", manifest)

    def test_presidio_runner_scores_untyped_source_masks_and_separate_completion(self):
        source = "Ann met Bob."
        self.write_gold(gold_document(text=source, spans=((0, 3, "01_NAME"), (8, 11, "01_NAME"))))
        notes = self.root / "comparator_notes"
        notes.mkdir()
        (notes / "NOTE1.txt").write_text(source)

        class FakeAnalyzer:
            def __init__(self):
                self.calls = 0

            def analyze(self, *, text, language):
                self.calls += 1
                name = "Ann" if self.calls == 1 else "Bob"
                start = text.index(name)
                return [SimpleNamespace(start=start, end=start + 3)]

        out = self.root / "comparator_out"
        model = run_comparator(notes, out, analyzer=FakeAnalyzer())
        results = metrics.score_all(out, self.gold_dir)
        first = self.headline(results, "pass1")
        final = self.headline(results, "cumulative_pass2")
        self.assertEqual((first["char_tp"], final["char_tp"], final["char_fn"]), (3, 6, 0))
        self.assertEqual(final["zero_residual_notes"], 1)
        complete = next(r for r in results["metrics"] if r["pass"] == "cumulative_pass2"
                        and r["cohort"] == "operational_complete" and r["phi_type"] == "ALL")
        self.assertEqual(complete["n_notes"], 1)
        for field in ("finish_stop_rate", "usable_requests", "json_valid_requests", "prompt_tokens", "completion_tokens"):
            self.assertIsNone(final[field], field)
        gold = metrics.load_gold(self.gold_path)
        self.assertEqual(results["per_doc"][1]["source_sha256"], gold["source_sha256"])
        self.assertEqual(results["per_doc"][1]["reference_sha256"], gold["reference_sha256"])
        # Loss of timing cannot erase saved accuracy or manufacture LLM failures.
        (model / "pipeline_timing" / "NOTE1.json").unlink()
        missing = metrics.score_all(out, self.gold_dir)
        self.assertEqual(self.headline(missing, "cumulative_pass2")["char_tp"], 6)
        empty = next(r for r in missing["metrics"] if r["pass"] == "cumulative_pass2"
                     and r["cohort"] == "operational_complete" and r["phi_type"] == "ALL")
        self.assertEqual(empty["n_notes"], 0)
        self.assertIsNone(empty["usable_requests"])


if __name__ == "__main__":
    unittest.main()
