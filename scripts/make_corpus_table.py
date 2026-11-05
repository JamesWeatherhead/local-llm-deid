#!/usr/bin/env python3
"""Rebuild the manuscript's corpus table (Table 1) from a directory of gold files.

The manuscript's corpus table summarizes the reference standard: how many notes it
holds, how long they are, how many PHI spans and characters they carry, and how
those spans distribute across the HIPAA Safe Harbor categories. This script
recomputes those counts from gold PubAnnotation files using the scorer's own gold
loader (``deid.metrics.load_gold``), so the definitions match the scored metrics
exactly.

The validated gold loader reads source text to check bounds and identity, but
the table emits only offsets-derived counts. The synthetic default describes
the fixture, not the clinical study. The real corpus is not shipped here.

Median and interquartile range use linearly interpolated sample percentiles.

Standard library only.

Usage::

    python3 scripts/make_corpus_table.py
    python3 scripts/make_corpus_table.py --gold-dir fixtures/synthetic/gold --csv out/corpus.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from deid.metrics import load_gold, popcount, reference_set_sha256  # noqa: E402


def linear_percentile(values, q):
    """Percentile of ``values`` at ``q`` in [0, 100], NumPy linear convention."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("percentile of an empty sample")
    if n == 1:
        return float(ordered[0])
    rank = (q / 100.0) * (n - 1)
    low = int(rank)
    high = min(low + 1, n - 1)
    frac = rank - low
    return ordered[low] + frac * (ordered[high] - ordered[low])


def summarize(gold_dir):
    """Aggregate corpus- and category-level counts from every gold file."""
    note_lengths = []
    total_phi_chars = 0
    total_spans = 0
    per_type_spans = {}
    per_type_chars = {}

    files = sorted(gold_dir.glob("*.json"))
    references = {}
    for path in files:
        gold = load_gold(path)
        references[path.stem] = gold
        note_lengths.append(gold["length"])
        total_phi_chars += popcount(gold["all"])
        total_spans += len(gold["spans"])
        for code, mask in gold["type"].items():
            per_type_chars[code] = per_type_chars.get(code, 0) + popcount(mask)
        for _start, _end, code in gold["spans"]:
            per_type_spans[code] = per_type_spans.get(code, 0) + 1

    return {
        "n_notes": len(files),
        "reference_set_sha256": reference_set_sha256(references),
        "note_lengths": note_lengths,
        "total_chars": sum(note_lengths),
        "total_phi_chars": total_phi_chars,
        "total_spans": total_spans,
        "per_type_spans": per_type_spans,
        "per_type_chars": per_type_chars,
    }


def panel_a(summary):
    """Corpus-level summary rows (label, value)."""
    lengths = summary["note_lengths"]
    median = linear_percentile(lengths, 50.0)
    q1 = linear_percentile(lengths, 25.0)
    q3 = linear_percentile(lengths, 75.0)
    density = ("{:.1f}%".format(summary["total_phi_chars"] / summary["total_chars"] * 100.0)
               if summary["total_chars"] else "undefined")
    return [
        ("Notes", str(summary["n_notes"])),
        ("Total characters", str(summary["total_chars"])),
        ("Median characters per note (Q1, Q3)",
         "{:.0f} ({:.0f}, {:.0f})".format(median, q1, q3)),
        ("PHI spans", str(summary["total_spans"])),
        ("PHI characters", str(summary["total_phi_chars"])),
        ("PHI character density", density),
        ("Safe Harbor categories present", str(len(summary["per_type_spans"]))),
    ]


def panel_b(summary):
    """Per-category rows sorted by span count descending (code, spans, share, chars)."""
    total = summary["total_spans"]
    codes = sorted(summary["per_type_spans"],
                   key=lambda code: (-summary["per_type_spans"][code], code))
    rows = []
    for code in codes:
        spans = summary["per_type_spans"][code]
        chars = summary["per_type_chars"].get(code, 0)
        share = (spans / total * 100.0) if total else 0.0
        rows.append((code, spans, "{:.1f}%".format(share), chars))
    return rows


def print_markdown(summary, gold_dir) -> None:
    print("Corpus and reference standard (supplied gold set)")
    print("gold: {}".format(gold_dir))
    print("reference_set_sha256: {}".format(summary["reference_set_sha256"]))
    print()
    print("| Measure | Value |")
    print("| --- | --- |")
    for label, value in panel_a(summary):
        print("| {} | {} |".format(label, value))
    print()
    print("| Safe Harbor category | Spans | Share of spans | PHI characters |")
    print("| --- | --: | --: | --: |")
    for code, spans, share, chars in panel_b(summary):
        print("| {} | {} | {} | {} |".format(code, spans, share, chars))
    print("| All categories | {} | {} | {} |".format(
        summary["total_spans"], "100.0%" if summary["total_spans"] else "undefined",
        summary["total_phi_chars"]))


def write_csv(path, summary) -> None:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"refusing symlinked output file: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "spans", "share_pct", "phi_chars", "n_notes", "reference_set_sha256"])
        for code, spans, share, chars in panel_b(summary):
            writer.writerow([code, spans, share.rstrip("%"), chars, summary["n_notes"], summary["reference_set_sha256"]])
        writer.writerow(["ALL", summary["total_spans"], "100.0" if summary["total_spans"] else "",
                         summary["total_phi_chars"], summary["n_notes"], summary["reference_set_sha256"]])
    path.chmod(0o600)


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--gold-dir", type=Path,
        default=REPO_ROOT / "fixtures" / "synthetic" / "gold",
        help="directory of gold PubAnnotation JSON files (default: the synthetic fixture)",
    )
    parser.add_argument("--csv", default=None,
                        help="also write the per-category table as CSV to this path")
    args = parser.parse_args()

    if not args.gold_dir.is_dir():
        parser.error("gold directory not found: {}".format(args.gold_dir))
    summary = summarize(args.gold_dir)
    if summary["n_notes"] == 0:
        parser.error("no gold JSON files in {}".format(args.gold_dir))

    print_markdown(summary, args.gold_dir)
    if args.csv:
        write_csv(args.csv, summary)
        print("\nwrote {}".format(args.csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
