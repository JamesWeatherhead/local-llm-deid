#!/usr/bin/env python3
"""Reproduce the manuscript's 95% confidence interval for character recall.

The results table reports a 95% confidence interval for "PHI removed" (character
recall) for every model. That interval is a note-level bootstrap: draw many
samples of the notes with replacement, pool the character counts and recompute
recall in each sample, then take the 2.5th and 97.5th percentiles. This script
reproduces the procedure from the per-note counts the scorer already writes
(``per_doc_long.csv``). It runs no new inference and reads only character counts,
never identifier text, so it stays PHI-free like the rest of ``out/``.

Defaults follow the manuscript: 10,000 resamples and seed 20260801 for the
language models (the Presidio comparator used seed 20260721). Each model is
resampled with its own ``random.Random(seed)``, so the result does not depend on
row order and is reproducible from run to run. The interval is a Monte Carlo
estimate, so its endpoints depend on the random-number stream; this standard
library version reproduces the method and, on the study corpus, the manuscript
intervals up to resampling noise. On the synthetic fixture (three notes) it is a
runnable demonstration, not the manuscript's numbers.

Percentiles use the same linear interpolation convention as NumPy's default, so
the endpoints match the tooling the manuscript used.

Standard library only.

Usage::

    python3 scripts/bootstrap_ci.py
    python3 scripts/bootstrap_ci.py --per-doc out/per_doc_long.csv --pass cumulative_pass2
    python3 scripts/bootstrap_ci.py --resamples 10000 --seed 20260801 --out out/recall_ci.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import OrderedDict
from pathlib import Path


def linear_percentile(values, q):
    """Percentile of ``values`` at ``q`` in [0, 100], NumPy linear convention."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("percentile of an empty sample")
    if n == 1:
        return ordered[0]
    rank = (q / 100.0) * (n - 1)
    low = int(rank)
    high = min(low + 1, n - 1)
    frac = rank - low
    return ordered[low] + frac * (ordered[high] - ordered[low])


def load_notes(per_doc_path, pass_filter):
    """Map ``(model_id, pass_)`` to a list of ``(char_tp, gt_phi_chars)`` per note."""
    groups = OrderedDict()
    with open(per_doc_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pass_ = row["pass_"]
            if pass_filter != "all" and pass_ != pass_filter:
                continue
            key = (row["model_id"], pass_)
            groups.setdefault(key, []).append(
                (int(row["char_tp"]), int(row["gt_phi_chars"]))
            )
    return groups


def bootstrap_recall_ci(notes, resamples, seed, alpha):
    """Return the point recall and the ``(low, high)`` percentile CI for one model."""
    tp_total = sum(tp for tp, _ in notes)
    gt_total = sum(gt for _, gt in notes)
    point = tp_total / gt_total if gt_total else None
    rng = random.Random(seed)
    n = len(notes)
    recalls = []
    for _ in range(resamples):
        sample = rng.choices(notes, k=n)
        tp_sum = sum(tp for tp, _ in sample)
        gt_sum = sum(gt for _, gt in sample)
        if gt_sum:
            recalls.append(tp_sum / gt_sum)
    low = linear_percentile(recalls, 100.0 * (alpha / 2.0))
    high = linear_percentile(recalls, 100.0 * (1.0 - alpha / 2.0))
    return point, low, high


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--per-doc", default="out/per_doc_long.csv",
        help="per_doc_long.csv written by deid.metrics (default: %(default)s)",
    )
    parser.add_argument(
        "--pass", dest="pass_filter", default="cumulative_pass2",
        choices=["pass1", "cumulative_pass2", "all"],
        help="which pass to summarize (default: %(default)s)",
    )
    parser.add_argument(
        "--resamples", type=int, default=10000,
        help="bootstrap resamples per model (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=20260801,
        help="random seed; the manuscript used 20260801 for models and "
             "20260721 for Presidio (default: %(default)s)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="two-sided alpha; 0.05 gives a 95%% CI (default: %(default)s)",
    )
    parser.add_argument(
        "--out", default=None,
        help="optional path to write the CI table as CSV",
    )
    args = parser.parse_args()

    per_doc_path = Path(args.per_doc)
    if not per_doc_path.exists():
        parser.error(f"per-doc file not found: {per_doc_path}")

    groups = load_notes(per_doc_path, args.pass_filter)
    if not groups:
        parser.error(f"no rows for pass '{args.pass_filter}' in {per_doc_path}")

    rows = []
    for (model_id, pass_), notes in groups.items():
        point, low, high = bootstrap_recall_ci(
            notes, args.resamples, args.seed, args.alpha
        )
        rows.append({
            "model_id": model_id,
            "pass": pass_,
            "n_notes": len(notes),
            "char_recall": point,
            "ci_low": low,
            "ci_high": high,
            "resamples": args.resamples,
            "seed": args.seed,
        })

    header = ["model_id", "pass", "n_notes", "char_recall", "ci_low", "ci_high",
              "resamples", "seed"]
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"wrote {len(rows)} rows to {args.out}")

    width = max(len(row["model_id"]) for row in rows)
    conf = int(round((1.0 - args.alpha) * 100))
    print(f"{'model':<{width}}  {'pass':<16}  notes  recall   {conf}% CI")
    for row in rows:
        recall = "n/a" if row["char_recall"] is None else f"{row['char_recall']:.3f}"
        print(f"{row['model_id']:<{width}}  {row['pass']:<16}  "
              f"{row['n_notes']:>5}  {recall:>6}   "
              f"({row['ci_low']:.3f}, {row['ci_high']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
