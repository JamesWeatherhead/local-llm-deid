"""Hand-checkable independent-annotation agreement on invented text only."""

from __future__ import annotations

import contextlib
import io
import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deid.agreement import _entity_matches, compute_agreement, main  # noqa: E402


def write_annotation(folder, document, text, spans):
    denotations = []
    attributes = []
    for index, span in enumerate(spans, start=1):
        start, end = span[:2]
        label = span[2] if len(span) > 2 else "01_NAME"
        annotation_id = f"T{index}"
        denotations.append({"id": annotation_id, "obj": "PHI_IDENTIFIER",
                            "span": {"begin": start, "end": end}})
        attributes.append({"subj": annotation_id, "pred": "identifier_type", "obj": label})
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{document}.json").write_text(json.dumps({
        "sourceid": document, "text": text, "denotations": denotations,
        "attributes": attributes,
    }), encoding="utf-8")


class IndependentAgreementTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.first, self.second = self.root / "first", self.root / "second"

    def paired(self, text, spans1, spans2, document="TEST"):
        write_annotation(self.first, document, text, spans1)
        write_annotation(self.second, document, text, spans2)

    def test_identical_boundaries_ignore_identifier_type(self):
        self.paired("A B C", [(0, 1, "01_NAME")], [(0, 1, "03_DATE")])
        result = compute_agreement(self.first, self.second, expected_notes=1)
        self.assertEqual(result["n_notes"], 1)
        self.assertEqual(result["character"], {
            "units": 5, "annotator_1_positive": 1, "annotator_2_positive": 1,
            "both_positive": 1, "f1_dice": 1.0, "cohen_kappa": 1.0})
        self.assertEqual(result["whitespace_token"]["units"], 3)
        self.assertEqual(result["entity"]["exact_matches"], 1)
        self.assertEqual(result["entity"]["exact_f1"], 1.0)
        self.assertEqual(result["entity"]["relaxed_f1"], 1.0)

    def test_partial_overlap_has_distinct_character_token_and_entity_scores(self):
        self.paired("AB CD", [(0, 2)], [(1, 4)])
        result = compute_agreement(self.first, self.second)
        self.assertEqual(result["character"]["f1_dice"], 2 / 5)
        self.assertAlmostEqual(result["character"]["cohen_kappa"], -2 / 13)
        self.assertEqual(result["whitespace_token"]["f1_dice"], 2 / 3)
        self.assertEqual(result["whitespace_token"]["cohen_kappa"], 0)
        self.assertEqual(result["entity"]["exact_f1"], 0)
        self.assertEqual(result["entity"]["relaxed_f1"], 1)

    def test_whitespace_tokenization_uses_any_overlap_and_unicode_codepoints(self):
        self.paired("🧬é\tX\n", [(0, 1)], [(1, 2)])
        result = compute_agreement(self.first, self.second)
        self.assertEqual(result["character"]["units"], 5)
        self.assertEqual(result["character"]["both_positive"], 0)
        self.assertEqual(result["whitespace_token"]["units"], 2)
        self.assertEqual(result["whitespace_token"]["both_positive"], 1)
        self.assertEqual(result["whitespace_token"]["f1_dice"], 1)
        self.assertEqual(result["entity"]["relaxed_matches"], 0)

    def test_overlapping_annotations_union_character_masks_but_match_entities_once(self):
        self.paired("abcde", [(0, 3), (2, 5)], [(0, 5)])
        result = compute_agreement(self.first, self.second)
        self.assertEqual(result["character"]["annotator_1_positive"], 5)
        self.assertEqual(result["character"]["f1_dice"], 1)
        self.assertEqual(result["entity"]["annotator_1_spans"], 2)
        self.assertEqual(result["entity"]["relaxed_matches"], 1)
        self.assertEqual(result["entity"]["relaxed_f1"], 2 / 3)

    def test_greatest_shared_text_greedy_does_not_claim_maximum_cardinality(self):
        # A maximum-cardinality algorithm could make two pairs. The specified
        # greedy rule first takes the six-character shared interval, leaving
        # only one one-to-one pair. This is distinct from model span recall.
        first = [(0, 3, "01_NAME"), (2, 9, "01_NAME")]
        second = [(0, 8, "01_NAME"), (8, 9, "01_NAME")]
        self.assertEqual(_entity_matches(first, second), (0, 1))

    def test_exact_pairs_are_reserved_before_partial_overlap(self):
        first = [(0, 4, "01_NAME"), (0, 8, "03_DATE")]
        second = [(0, 4, "03_DATE"), (4, 8, "01_NAME")]
        self.assertEqual(_entity_matches(first, second), (1, 2))

    def test_documents_are_scored_separately_then_counts_are_pooled(self):
        self.paired("A B", [(0, 1)], [], "TEST1")
        self.paired("A B", [], [(0, 1)], "TEST2")
        result = compute_agreement(self.first, self.second)
        self.assertEqual(result["n_notes"], 2)
        self.assertEqual(result["character"]["units"], 6)
        self.assertEqual(result["character"]["both_positive"], 0)
        self.assertEqual(result["character"]["f1_dice"], 0)
        self.assertEqual(result["entity"]["relaxed_matches"], 0)

    def test_undefined_denominators_are_null_instead_of_perfect_agreement(self):
        self.paired(" \n", [], [])
        result = compute_agreement(self.first, self.second)
        self.assertIsNone(result["character"]["f1_dice"])
        self.assertIsNone(result["character"]["cohen_kappa"])
        self.assertEqual(result["whitespace_token"]["units"], 0)
        self.assertIsNone(result["whitespace_token"]["f1_dice"])
        self.assertIsNone(result["entity"]["exact_f1"])
        self.assertIsNone(result["entity"]["relaxed_f1"])
        self.paired("A", [(0, 1)], [(0, 1)])
        result = compute_agreement(self.first, self.second)
        self.assertEqual(result["character"]["f1_dice"], 1)
        self.assertIsNone(result["character"]["cohen_kappa"])

    def test_mismatched_document_sets_or_source_text_fail_closed(self):
        self.paired("ABC", [(0, 1)], [(0, 1)])
        write_annotation(self.second, "EXTRA", "ABC", [])
        with self.assertRaisesRegex(ValueError, "identical nonempty"):
            compute_agreement(self.first, self.second)
        (self.second / "EXTRA.json").unlink()
        write_annotation(self.second, "TEST", "ABD", [(0, 1)])
        with self.assertRaisesRegex(ValueError, "different source text"):
            compute_agreement(self.first, self.second)

    def test_count_guards_and_invalid_annotation_bounds(self):
        self.paired("ABC", [(0, 1)], [(0, 1)])
        for expected in [True, 1.0, 0, -1, 2]:
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError):
                    compute_agreement(self.first, self.second, expected)
        write_annotation(self.second, "TEST", "ABC", [(0, 5)])
        with self.assertRaisesRegex(ValueError, "bounds"):
            compute_agreement(self.first, self.second)

    def test_cli_writes_only_private_aggregate_output_outside_inputs(self):
        self.paired("Invented SecretPhrase", [(9, 21)], [(9, 21)])
        output = self.root / "results/agreement.json"
        main(["--annotator-1", str(self.first), "--annotator-2", str(self.second),
              "--out", str(output), "--expected-notes", "1"])
        text = output.read_text()
        self.assertNotIn("SecretPhrase", text)
        self.assertNotIn("Invented", text)
        self.assertNotIn("TEST", text)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(output.parent.stat().st_mode), 0o700)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--annotator-1", str(self.first), "--annotator-2", str(self.second),
                      "--out", str(self.first / "overwrite.json")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
