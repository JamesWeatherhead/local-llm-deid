"""Regression tests for multi-execution bootstrap preparation."""

from __future__ import annotations

import csv
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import bootstrap_ci  # noqa: E402


FIELDS = ["model_id", "doc", "pass_", "char_tp", "gt_phi_chars", "zero_residual",
          "char_fp", "n_docs", "source_sha256", "reference_sha256", "reference_set_sha256"]


def write_run(path: Path, rows) -> None:
    docs = sorted({row["doc"] for row in rows})
    references = {doc: hashlib.sha256(("reference-" + doc).encode()).hexdigest() for doc in docs}
    fingerprint = hashlib.sha256(json.dumps([(doc, references[doc]) for doc in docs], separators=(",", ":")).encode()).hexdigest()
    rows = [{"char_fp": 2, "n_docs": len(docs), "source_sha256": "b" * 64,
             "reference_sha256": references[row["doc"]], "reference_set_sha256": fingerprint, **row}
            for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class MultiRunBootstrapTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.run1 = root / "run1.csv"
        self.run2 = root / "run2.csv"
        base = {"model_id": "model-a", "pass_": "cumulative_pass2", "gt_phi_chars": 10}
        write_run(self.run1, [
            {**base, "doc": "A", "char_tp": 5, "zero_residual": 0},
            {**base, "doc": "B", "char_tp": 10, "zero_residual": 1},
        ])
        write_run(self.run2, [
            {**base, "doc": "A", "char_tp": 7, "zero_residual": 0},
            {**base, "doc": "B", "char_tp": 8, "zero_residual": 0},
        ])

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_counts_and_completeness_are_averaged_by_note(self) -> None:
        groups = bootstrap_ci.load_notes([self.run1, self.run2], "cumulative_pass2")
        notes = groups[("model-a", "cumulative_pass2")]
        self.assertEqual(notes, [bootstrap_ci.NoteCounts(6.0, 10.0, 0.0, 2.0), bootstrap_ci.NoteCounts(9.0, 10.0, 0.5, 2.0)])
        result = bootstrap_ci.bootstrap_intervals(notes, resamples=100, seed=20260801, alpha=0.05)
        self.assertEqual(result["char_recall"], 0.75)
        self.assertEqual(result["complete_note_rate"], 0.25)
        self.assertEqual(result["complete_note_count"], 0.5)
        self.assertFalse(any("complete_ci" in key for key in result))

    def test_execution_key_mismatch_is_rejected(self) -> None:
        with self.run2.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))[:1]
        write_run(self.run2, rows)
        with self.assertRaisesRegex(ValueError, "execution key mismatch|incomplete or mixed reference cohort"):
            bootstrap_ci.load_notes([self.run1, self.run2], "cumulative_pass2")

    def test_duplicate_execution_file_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate per-doc file"):
            bootstrap_ci.load_notes([self.run1, self.run1], "cumulative_pass2")

    def test_invalid_counts_and_completeness_are_rejected(self) -> None:
        invalid = Path(self._tmp.name) / "invalid.csv"
        base = {"model_id": "model-a", "pass_": "cumulative_pass2", "doc": "A",
                "gt_phi_chars": 10}
        cases = [
            {**base, "char_tp": 11, "zero_residual": 0},
            {**base, "char_tp": 5, "zero_residual": 0.5},
            {**base, "char_tp": "nan", "zero_residual": 0},
            {**base, "char_tp": 9.5, "zero_residual": 0},
            {**base, "char_tp": 10, "zero_residual": 0},
        ]
        for candidate in cases:
            with self.subTest(candidate=candidate):
                write_run(invalid, [candidate])
                with self.assertRaises(ValueError):
                    bootstrap_ci.load_notes([invalid], "cumulative_pass2")


    def test_precision_and_f1_bootstrap_use_false_positive_counts(self):
        notes = [bootstrap_ci.NoteCounts(6, 10, 0, 2), bootstrap_ci.NoteCounts(9, 10, 0.5, 2)]
        result = bootstrap_ci.bootstrap_intervals(notes, 100, 20260721, 0.05, ("recall", "precision", "f1"))
        self.assertEqual(result["char_precision"], 15 / 19)
        self.assertEqual(result["char_f1"], 30 / 39)
        self.assertIn("precision_ci_low", result)
        self.assertNotIn("complete_ci_low", result)
        self.assertEqual(result, bootstrap_ci.bootstrap_intervals(notes, 100, 20260721, 0.05, ("recall", "precision", "f1")))

    def test_missing_false_positive_counts_do_not_invent_precision(self):
        notes = [bootstrap_ci.NoteCounts(0, 10, 0)]
        with self.assertRaisesRegex(ValueError, "require per-note char_fp"):
            bootstrap_ci.bootstrap_intervals(notes, 100, 1, 0.05, ("precision",))

    def test_undefined_recall_is_blank_and_zero_f1_is_zero(self):
        result = bootstrap_ci.bootstrap_intervals([bootstrap_ci.NoteCounts(0, 0, 1, 0)], 10, 1, .05)
        self.assertIsNone(result["char_recall"])
        self.assertIsNone(result["ci_low"])
        result = bootstrap_ci.bootstrap_intervals([bootstrap_ci.NoteCounts(0, 10, 0, 0)], 10, 1, .05, ("f1",))
        self.assertEqual(result["char_f1"], 0)

    def test_subset_cannot_keep_the_full_cohort_fingerprint(self):
        with self.run1.open() as stream:
            rows = list(csv.DictReader(stream))[:1]
        write_run(self.run1, rows)
        with self.assertRaisesRegex(ValueError, "incomplete or mixed reference cohort"):
            bootstrap_ci.load_notes([self.run1], "cumulative_pass2")

    def test_same_character_totals_do_not_hide_reference_changes_across_runs(self):
        with self.run2.open() as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["source_sha256"] = "c" * 64
        write_run(self.run2, rows)
        with self.assertRaisesRegex(ValueError, "reference identity differs"):
            bootstrap_ci.load_notes([self.run1, self.run2], "cumulative_pass2")

    def test_cli_outputs_character_intervals_but_no_completeness_interval(self):
        destination = Path(self._tmp.name) / "bootstrap.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bootstrap_ci.main([
                "--per-doc", str(self.run1),
                "--character-metrics", "recall", "precision", "f1",
                "--seed", "20260721", "--resamples", "25", "--out", str(destination),
            ]), 0)
        with destination.open() as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(records[0]["seed"], "20260721")
        self.assertEqual(records[0]["n_notes"], "2")
        self.assertIn("precision_ci_low", records[0])
        self.assertIn("f1_ci_high", records[0])
        self.assertEqual(float(records[0]["complete_note_count"]), 1)
        self.assertFalse(any("complete_ci" in field for field in records[0]))

    def test_cli_rejects_multirun_precision_f1_instead_of_changing_estimator(self):
        # All runs recall every gold character. Means of per-run P/F1 differ
        # sharply from ratios of the averaged counts when FP counts vary.
        root = Path(self._tmp.name)
        paths = []
        for number, fp in enumerate((0, 0, 90), start=1):
            path = root / f"precision-run{number}.csv"
            write_run(path, [{"model_id": "model-a", "pass_": "cumulative_pass2",
                              "doc": "A", "char_tp": 10, "gt_phi_chars": 10,
                              "char_fp": fp, "zero_residual": 1}])
            paths.append(path)
        mean_precision = (1 + 1 + .1) / 3
        mean_f1 = (1 + 1 + 20 / 110) / 3
        groups = bootstrap_ci.load_notes(paths, "cumulative_pass2")
        notes = groups[("model-a", "cumulative_pass2")]
        pooled = bootstrap_ci.bootstrap_intervals(notes, 10, 42, .05,
                                                   ("recall", "precision", "f1"))
        self.assertAlmostEqual(mean_precision, .7)
        self.assertAlmostEqual(mean_f1, 8 / 11)
        self.assertEqual(pooled["char_precision"], .25)
        self.assertEqual(pooled["char_f1"], .4)
        arguments = [item for path in paths for item in ("--per-doc", str(path))]
        error = io.StringIO()
        with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as caught:
            bootstrap_ci.main(arguments + ["--character-metrics", "recall", "precision", "f1"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("precision/F1 comparator intervals require one execution", error.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bootstrap_ci.main(arguments + ["--resamples", "10"]), 0)


if __name__ == "__main__":
    unittest.main()
