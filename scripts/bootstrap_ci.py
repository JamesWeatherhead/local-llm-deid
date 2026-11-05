#!/usr/bin/env python3
"""Bootstrap character-metric intervals from matched per-note scorer counts.

Counts are averaged by note across repeated executions before paired note-level
resampling. Descriptive mean complete-note counts/rates are reported without a
completeness interval. Character recall is the default; add precision and F1 for
a comparator with --character-metrics recall precision f1. Their calculation
requires per-note false-positive character counts, not rounded aggregate rates.
Precision/F1 intervals require one execution, matching the single-run comparator;
multi-execution intervals support recall only. Ratios from averaged counts need
not equal means of the individual-execution precision/F1 values.

Defaults are 10,000 resamples and the manuscript's model seed 20260801; select
--seed 20260721 for Presidio. The standard-library random-number stream and
linearly interpolated percentiles are explicit companion implementation choices,
not evidence of the historical software or exact published interval endpoints.
The bundled synthetic fixture illustrates the calculation only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple, Optional


class NoteCounts(NamedTuple):
    tp: float
    gt: float
    zero: float
    fp: Optional[float] = None


def linear_percentile(values, q):
    """Linearly interpolated sample percentile for q in [0, 100]."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = (q / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (rank - low) * (ordered[high] - ordered[low])


def _count(row, field, key, path):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid {field} in {path}: {key}") from exc
    if not math.isfinite(value) or not value.is_integer() or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer in {path}: {key}")
    return value


def _load_run(per_doc_path, pass_filter, require_fp=False):
    """Load one execution and verify the complete identified reference cohort."""
    rows, identities = {}, {}
    with open(per_doc_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pass_ = row.get("pass_")
            if pass_filter != "all" and pass_ != pass_filter:
                continue
            key = (row.get("model_id"), pass_, row.get("doc"))
            if not all(isinstance(value, str) and value for value in key):
                raise ValueError(f"missing model/pass/document identity in {per_doc_path}")
            if key in rows:
                raise ValueError(f"duplicate per-note row in {per_doc_path}: {key}")
            tp = _count(row, "char_tp", key, per_doc_path)
            gt = _count(row, "gt_phi_chars", key, per_doc_path)
            zero = _count(row, "zero_residual", key, per_doc_path)
            if tp > gt or zero not in (0, 1) or zero != float(tp == gt):
                raise ValueError(f"inconsistent character counts or zero_residual in {per_doc_path}: {key}")
            fp = (_count(row, "char_fp", key, per_doc_path)
                  if require_fp or row.get("char_fp") not in (None, "") else None)
            n_docs = _count(row, "n_docs", key, per_doc_path)
            hashes = tuple(row.get(field, "") for field in
                           ("source_sha256", "reference_sha256", "reference_set_sha256"))
            if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
                raise ValueError(f"per-note reference hashes are required in {per_doc_path}: {key}; rescore the locked reference")
            rows[key] = NoteCounts(tp, gt, zero, fp)
            identities[key] = (n_docs, *hashes)
    groups = {}
    for (model, pass_, doc), identity in identities.items():
        groups.setdefault((model, pass_), []).append((doc, identity))
    signatures = set()
    for group, members in groups.items():
        n_docs = {identity[0] for _, identity in members}
        declared_hashes = {identity[3] for _, identity in members}
        if n_docs != {len(members)} or len(declared_hashes) != 1:
            raise ValueError(f"incomplete or mixed reference cohort in {per_doc_path}: {group}")
        payload = [(doc, identity[2]) for doc, identity in sorted(members)]
        fingerprint = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()
        if declared_hashes != {fingerprint}:
            raise ValueError(f"reference-set fingerprint does not match document rows in {per_doc_path}: {group}")
        signatures.add(tuple((doc, identity) for doc, identity in sorted(members)))
    if len(signatures) > 1:
        raise ValueError(f"models or passes use different reference cohorts in {per_doc_path}")
    return rows, identities


def load_notes(per_doc_paths, pass_filter, require_fp=False):
    """Average matched raw counts and binary completeness indicators by note."""
    if isinstance(per_doc_paths, (str, Path)):
        per_doc_paths = [per_doc_paths]
    paths = [Path(path) for path in per_doc_paths]
    if not paths:
        raise ValueError("at least one per-doc file is required")
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("duplicate per-doc file supplied as more than one execution")
    loaded = [_load_run(path, pass_filter, require_fp) for path in paths]
    runs = [records for records, _ in loaded]
    expected = set(runs[0])
    for path, run in zip(paths[1:], runs[1:]):
        if set(run) != expected:
            raise ValueError(f"execution key mismatch in {path}: missing={len(expected - set(run))}, extra={len(set(run) - expected)}")
    groups = OrderedDict()
    for key in sorted(expected):
        values = [run[key] for run in runs]
        if any(identities[key] != loaded[0][1][key] for _, identities in loaded[1:]):
            raise ValueError(f"reference identity differs across executions: {key}")
        if len({value.gt for value in values}) != 1:
            raise ValueError(f"gold-character count differs across executions: {key}")
        fp_values = [value.fp for value in values]
        if any(value is None for value in fp_values) and not all(value is None for value in fp_values):
            raise ValueError(f"false-positive counts exist in only some executions: {key}")
        fp = None if fp_values[0] is None else sum(fp_values) / len(values)
        averaged = NoteCounts(*(sum(value[index] for value in values) / len(values)
                                for index in range(3)), fp)
        groups.setdefault(key[:2], []).append(averaged)
    return groups


def _metrics(notes, requested):
    tp, gt = sum(note.tp for note in notes), sum(note.gt for note in notes)
    fp = sum(note.fp for note in notes) if all(note.fp is not None for note in notes) else None
    values = {}
    for metric in requested:
        if metric == "recall":
            denominator, numerator = gt, tp
        elif metric == "precision":
            denominator, numerator = tp + fp, tp
        else:
            denominator, numerator = gt + tp + fp, 2 * tp
        values[metric] = numerator / denominator if denominator else None
    return values


def bootstrap_intervals(notes, resamples, seed, alpha, character_metrics=("recall",)):
    """Return character percentile intervals and descriptive mean completeness."""
    if not notes or resamples <= 0 or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("nonempty notes, positive resamples and alpha between 0 and 1 are required")
    requested = tuple(dict.fromkeys(character_metrics))
    if not requested or any(metric not in {"recall", "precision", "f1"} for metric in requested):
        raise ValueError("character metrics must be recall, precision or f1")
    if any(metric != "recall" for metric in requested) and any(note.fp is None for note in notes):
        raise ValueError("precision and F1 require per-note char_fp counts")
    points = _metrics(notes, requested)
    samples = {metric: [] for metric in requested}
    rng = random.Random(seed)
    for _ in range(resamples):
        values = _metrics(rng.choices(notes, k=len(notes)), requested)
        for metric, value in values.items():
            if value is not None:
                samples[metric].append(value)
    result = {"complete_note_count": sum(note.zero for note in notes),
              "complete_note_rate": sum(note.zero for note in notes) / len(notes)}
    for metric in requested:
        low = linear_percentile(samples[metric], 100 * alpha / 2)
        high = linear_percentile(samples[metric], 100 * (1 - alpha / 2))
        result["char_" + metric] = points[metric]
        if metric == "recall":
            result.update(ci_low=low, ci_high=high)
        else:
            result[metric + "_ci_low"] = low
            result[metric + "_ci_high"] = high
    return result


def main(argv=None) -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-doc", action="append", help="scorer per_doc_long.csv; repeat for matched executions")
    parser.add_argument("--pass", dest="pass_filter", default="cumulative_pass2",
                        choices=["pass1", "cumulative_pass2", "all"])
    parser.add_argument("--character-metrics", nargs="+", choices=["recall", "precision", "f1"], default=["recall"])
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260801,
                        help="model seed20260801 by default; use20260721 for Presidio")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", help="optional CSV destination")
    args = parser.parse_args(argv)
    if args.resamples <= 0 or not math.isfinite(args.alpha) or not 0 < args.alpha < 1:
        parser.error("positive --resamples and finite --alpha between 0 and 1 are required")
    paths = [Path(path) for path in (args.per_doc or ["out/per_doc_long.csv"])]
    if len(paths) > 1 and any(metric != "recall" for metric in args.character_metrics):
        parser.error(
            "multi-execution character intervals support recall only; precision/F1 "
            "comparator intervals require one execution"
        )
    try:
        groups = load_notes(paths, args.pass_filter,
                            require_fp=any(metric != "recall" for metric in args.character_metrics))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not groups:
        parser.error(f"no rows for pass '{args.pass_filter}'")
    rows = []
    for (model, pass_), notes in groups.items():
        result = bootstrap_intervals(notes, args.resamples, args.seed, args.alpha, args.character_metrics)
        rows.append({"model_id": model, "pass": pass_, "n_notes": len(notes), "n_runs": len(paths),
                     **result, "resamples": args.resamples, "seed": args.seed})
    header = list(rows[0])
    if args.out:
        output = Path(args.out)
        if output.is_symlink():
            parser.error(f"refusing symlinked output file: {output}")
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        output.chmod(0o600)
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
