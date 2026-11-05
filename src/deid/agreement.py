"""Type-agnostic agreement between two independent annotation sets.

This companion implementation follows the supplementary methods. It does not
reconstruct study results without the corresponding independent annotations.
Only aggregate counts and scores are written; source prose is never exported.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from .metrics import load_gold, popcount, span_mask
from .pipeline import sha256_text
from .run_model import write_json


def _binary_summary(total: int, positive1: int, positive2: int,
                    both: int) -> dict[str, Any]:
    """Pooled binary agreement; undefined F1/kappa are explicitly null."""
    positive_sum = positive1 + positive2
    # This algebraic form avoids subtracting near-equal floating-point chance
    # agreement values. It is exactly (observed - chance)/(1 - chance).
    kappa_denominator = total * positive_sum - 2 * positive1 * positive2
    return {
        "units": total,
        "annotator_1_positive": positive1,
        "annotator_2_positive": positive2,
        "both_positive": both,
        "f1_dice": 2 * both / positive_sum if positive_sum else None,
        "cohen_kappa": (2 * (total * both - positive1 * positive2) / kappa_denominator
                        if kappa_denominator else None),
    }


def _entity_matches(spans1: Sequence[tuple[int, int, str]],
                    spans2: Sequence[tuple[int, int, str]]) -> tuple[int, int]:
    """Exact-first, then greatest-shared-text, one-to-one type-agnostic pairs.

    Equal-overlap ties are broken by source boundaries, then annotation order,
    to make this reference implementation deterministic. Duplicate boundaries
    carrying different labels remain distinct annotation items.
    """
    left = sorted((start, end) for start, end, _label in spans1)
    right = sorted((start, end) for start, end, _label in spans2)
    available1, available2 = set(range(len(left))), set(range(len(right)))
    exact = 0
    for i, interval in enumerate(left):
        for j in sorted(available2):
            if interval == right[j]:
                available1.remove(i)
                available2.remove(j)
                exact += 1
                break

    candidates = []
    for i in available1:
        for j in available2:
            overlap = min(left[i][1], right[j][1]) - max(left[i][0], right[j][0])
            if overlap > 0:
                candidates.append((-overlap, left[i], right[j], i, j))
    relaxed = exact
    for _negative_overlap, _left, _right, i, j in sorted(candidates):
        if i in available1 and j in available2:
            available1.remove(i)
            available2.remove(j)
            relaxed += 1
    return exact, relaxed


def compute_agreement(annotator1_dir: Path, annotator2_dir: Path,
                      expected_notes: Optional[int] = None) -> dict[str, Any]:
    """Pool agreement over matched, validated PubAnnotation JSON note sets.

    Both directories must contain the same nonempty set of ``<document>.json``
    files with identical source text per document. ``expected_notes`` can guard
    an intended cohort size; it never supplies a scoring denominator. Character
    positions are Unicode codepoints; nonempty whitespace-delimited tokens are
    positive if any character overlaps an annotation. Human entity matching is
    one-to-one and differs from the scorer's nonexclusive model-span matches.
    """
    if expected_notes is not None and (type(expected_notes) is not int or expected_notes <= 0):
        raise ValueError("expected_notes must be a positive integer or None")
    annotator1_dir, annotator2_dir = Path(annotator1_dir), Path(annotator2_dir)
    files1 = {path.stem: path for path in annotator1_dir.glob("*.json")}
    files2 = {path.stem: path for path in annotator2_dir.glob("*.json")}
    if not files1 or set(files1) != set(files2):
        raise ValueError("annotation directories must contain identical nonempty JSON document sets")
    if expected_notes is not None and len(files1) != expected_notes:
        raise ValueError(f"expected {expected_notes} notes, found {len(files1)}")

    character_counts = [0, 0, 0, 0]
    token_counts = [0, 0, 0, 0]
    entity1 = entity2 = exact = relaxed = 0
    for document_id in sorted(files1):
        gold1, gold2 = load_gold(files1[document_id]), load_gold(files2[document_id])
        if (gold1["length"] != gold2["length"]
                or gold1["source_sha256"] != gold2["source_sha256"]):
            raise ValueError(f"annotators have different source text for {document_id}")
        source = json.loads(files1[document_id].read_text(encoding="utf-8"))["text"]
        if sha256_text(source) != gold1["source_sha256"]:
            raise ValueError(f"annotation source changed while reading {document_id}")
        mask1, mask2 = gold1["all"], gold2["all"]
        per_character = [gold1["length"], popcount(mask1), popcount(mask2),
                         popcount(mask1 & mask2)]
        character_counts = [a + b for a, b in zip(character_counts, per_character)]

        token1 = token2 = token_both = token_total = 0
        for token in re.finditer(r"\S+", source):
            token_total += 1
            token_mask = span_mask(token.start(), token.end())
            positive1, positive2 = bool(mask1 & token_mask), bool(mask2 & token_mask)
            token1 += positive1
            token2 += positive2
            token_both += positive1 and positive2
        token_counts = [a + b for a, b in
                        zip(token_counts, [token_total, token1, token2, token_both])]
        entity1 += len(gold1["spans"])
        entity2 += len(gold2["spans"])
        note_exact, note_relaxed = _entity_matches(gold1["spans"], gold2["spans"])
        exact += note_exact
        relaxed += note_relaxed

    entity_sum = entity1 + entity2
    return {
        "schema_version": "annotation-agreement-v1",
        "implementation": "companion-reference",
        "n_notes": len(files1),
        "character": _binary_summary(*character_counts),
        "whitespace_token": _binary_summary(*token_counts),
        "entity": {
            "annotator_1_spans": entity1,
            "annotator_2_spans": entity2,
            "exact_matches": exact,
            "relaxed_matches": relaxed,
            "exact_f1": 2 * exact / entity_sum if entity_sum else None,
            "relaxed_f1": 2 * relaxed / entity_sum if entity_sum else None,
            "matching": "type-agnostic; exact-first; greatest-overlap; one-to-one",
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-1", type=Path, required=True)
    parser.add_argument("--annotator-2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-notes", type=int)
    args = parser.parse_args(argv)
    output = args.out.resolve()
    if any(folder.resolve() in output.parents for folder in (args.annotator_1, args.annotator_2)):
        parser.error("write --out outside the input annotation directories")
    result = compute_agreement(args.annotator_1, args.annotator_2, args.expected_notes)
    write_json(args.out, result)


if __name__ == "__main__":
    main()
