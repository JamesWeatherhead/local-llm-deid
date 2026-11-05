#!/usr/bin/env python3
"""Summarize timing with the manuscript's distinct LLM/comparator denominators.

For each execution: sum all segment request latencies across both passes for
each LLM note, then take the median over notes. Report the mean of those medians
across matched executions. Presidio uses separately measured full-pipeline note
times. Missing note plans, requests or timing are errors, never zero durations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deid.provenance import expected_scopes
from deid.run_model import DOC_ID_PATTERN, write_private_text


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def duration(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("timing must be a recorded finite, nonnegative duration")
    return value


def _note_plan(manifest):
    notes = manifest.get("notes")
    if not isinstance(notes, list) or not notes:
        raise ValueError("complete planned-note manifest is required; timing coverage is otherwise unknown")
    result = {}
    for note in notes:
        if not isinstance(note, dict):
            raise ValueError("invalid planned note")
        doc_id, source_hash = note.get("document_id"), note.get("source_sha256")
        if (not isinstance(doc_id, str) or not DOC_ID_PATTERN.fullmatch(doc_id)
                or doc_id in result or not isinstance(source_hash, str)
                or len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash)):
            raise ValueError("invalid or duplicate planned note identity")
        result[doc_id] = source_hash
    return result


def execution_timing(model_dir):
    """Return actual matched cohort, timing method, and this execution's median."""
    comparator = model_dir / "comparator_manifest.json"
    manifest_path = comparator if comparator.exists() else model_dir / "inference_manifest.json"
    manifest = read_json(manifest_path)
    notes = _note_plan(manifest)
    backend = manifest.get("backend")
    if manifest.get("model_id") != model_dir.name:
        raise ValueError("manifest model ID differs from output folder")
    values = []
    if backend == "presidio":
        if manifest.get("schema_version") != "comparator-manifest-v1":
            raise ValueError("invalid comparator manifest")
        observed = {p.stem for p in (model_dir / "pipeline_timing").glob("*.json")}
        if observed != set(notes):
            raise ValueError("comparator note timing set differs from full planned cohort")
        for doc_id, source_hash in sorted(notes.items()):
            record = read_json(model_dir / "pipeline_timing" / f"{doc_id}.json")
            if (record.get("document_id") != doc_id or record.get("source_sha256") != source_hash
                    or record.get("completed") is not True):
                raise ValueError("comparator timing is incomplete or belongs to a different note")
            values.append(duration(record.get("pipeline_seconds")))
        method = "full_comparator_pipeline"
    elif backend in ("llama-server", "offline-stub"):
        if manifest.get("schema_version") != "inference-manifest-v1":
            raise ValueError("invalid inference manifest")
        for pass_number in (1, 2):
            base = model_dir if pass_number == 1 else model_dir / "pass_2"
            if {p.name for p in (base / "expected_segments").iterdir() if p.is_dir()} != set(notes):
                raise ValueError("segment plan set differs from full planned cohort")
            if {p.name for p in (base / "raw_responses").iterdir() if p.is_dir()} != set(notes):
                raise ValueError("response note set differs from full planned cohort")
        for doc_id, source_hash in sorted(notes.items()):
            total = 0.0
            for pass_number in (1, 2):
                base = model_dir if pass_number == 1 else model_dir / "pass_2"
                expected = expected_scopes(base, doc_id, pass_number, source_hash)
                if expected is None:
                    raise ValueError("expected segment coverage is unknown")
                files = {p.name.removesuffix(".raw.json"): p
                         for p in (base / "raw_responses" / doc_id).glob("*.raw.json")}
                if set(files) != expected:
                    raise ValueError("request timing coverage differs from expected segments")
                for record_path in files.values():
                    total += duration(read_json(record_path).get("latency_seconds"))
            values.append(total)
        method = "sum_segment_request_latencies" if backend == "llama-server" else "synthetic_stub_request_latencies"
    else:
        raise ValueError("unknown timing backend")
    return notes, method, statistics.median(values)


def summarize(pred_dirs):
    if not pred_dirs:
        raise ValueError("at least one execution is required")
    roots = [Path(p).resolve() for p in pred_dirs]
    if len(set(roots)) != len(roots):
        raise ValueError("the same execution directory was supplied more than once")
    models_by_run = [{p.name: p for p in root.iterdir() if p.is_dir() and
                      ((p / "inference_manifest.json").exists() or
                       (p / "comparator_manifest.json").exists())} for root in roots]
    if not models_by_run[0] or any(set(run) != set(models_by_run[0]) for run in models_by_run):
        raise ValueError("executions must contain the same nonempty model set")
    rows = []
    shared_cohort = None
    for model_id in sorted(models_by_run[0]):
        results = [execution_timing(run[model_id]) for run in models_by_run]
        notes, method, _ = results[0]
        if any(cohort != notes or other_method != method for cohort, other_method, _ in results):
            raise ValueError("timing executions differ in note identity, source text or timing method")
        if shared_cohort is None:
            shared_cohort = notes
        elif shared_cohort != notes:
            raise ValueError("models do not share the same note cohort")
        medians = [median for _, _, median in results]
        cohort_hash = hashlib.sha256(json.dumps(sorted(notes.items()), separators=(",", ":")).encode()).hexdigest()
        rows.append({"model_id": model_id, "timing_method": method,
                     "n_notes": len(notes), "n_executions": len(results),
                     "mean_median_seconds_per_note": statistics.mean(medians),
                     "execution_medians_seconds": json.dumps(medians),
                     "source_set_sha256": cohort_hash})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", action="append", type=Path, required=True,
                        help="output root for one execution; repeat for matched executions")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = summarize(args.pred_dir)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    write_private_text(args.out, buffer.getvalue())
    print(f"wrote timing summary: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
