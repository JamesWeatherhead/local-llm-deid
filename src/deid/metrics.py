"""Re-derive every de-identification metric from gold spans + resolved predictions.

The scorer reads only two things per note:

* the gold PubAnnotation JSON (reference ``text`` plus typed character spans), and
* the model's ``resolved_predictions.json`` (offsets + types, no identifier text).

From those it recomputes character-, span-, note-, and reliability-level metrics
under formulas defined here, and writes one long-format table plus supporting
manifests. All outputs are PHI-free: offsets, counts, and rates only.

Coordinate system: Unicode codepoints, half-open ``[start, end)``. The gold
``text`` is the reference; Python ``str`` indexing is codepoint-based.

Scoring is deterministic -- no randomness, no network. Point it at a directory
of per-model outputs and a directory of gold files::

    python -m deid.metrics --pred-dir OUTPUT_ROOT --gold-dir GOLD --out-dir out/
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import statistics
from pathlib import Path
from typing import Any, Optional

from . import __version__


# --- bitmask helpers ---------------------------------------------------------
# Each note's gold PHI characters and predicted-redaction characters are held as
# a Python big-integer bitmask over codepoint positions. Set operations then give
# exact character confusion counts.

_bit_count = getattr(int, "bit_count", None)


def popcount(mask: int) -> int:
    """Number of set bits (redacted/covered codepoints) in a mask."""
    return _bit_count(mask) if _bit_count else bin(mask).count("1")


def span_mask(start: int, end: int) -> int:
    """Bitmask with bits ``[start, end)`` set (empty if ``end <= start``)."""
    if end <= start:
        return 0
    return ((1 << (end - start)) - 1) << start


def code_prefix(identifier_type: Optional[str]) -> str:
    """Reduce an identifier type to its two-digit code (``01_NAME`` -> ``01``)."""
    match = re.match(r"\s*(\d{2})", identifier_type or "")
    return match.group(1) if match else (identifier_type or "")


def rate(numerator: float, denominator: float) -> Optional[float]:
    """Safe division: ``None`` when the denominator is zero."""
    return (numerator / denominator) if denominator else None


def f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    """Harmonic mean of precision and recall (``None`` if undefined)."""
    if precision is not None and recall is not None and (precision + recall) > 0:
        return 2 * precision * recall / (precision + recall)
    return None


def mean(values: list[float]) -> Optional[float]:
    """Mean of a list, or ``None`` when empty."""
    return statistics.mean(values) if values else None


# i2b2 relaxed-entity tolerance: a predicted span matches a gold span if both
# boundaries are within this many codepoints (tolerates trailing space/punct).
REL_TOL = 2


def relaxed_hit(start: int, end: int, spans: set[tuple[int, int]]) -> bool:
    """True iff some span in ``spans`` is within REL_TOL of both boundaries."""
    for a, b in spans:
        if abs(a - start) <= REL_TOL and abs(b - end) <= REL_TOL:
            return True
    return False


# --- gold + prediction loading ----------------------------------------------

def load_gold(gold_path: Path) -> dict[str, Any]:
    """Load one gold PubAnnotation file into masks and a span list.

    Reads the reference ``text``, the typed character spans (``denotations``),
    and the type code for each span (``attributes`` where ``pred`` is
    ``identifier_type``, joining the attribute's ``subj`` onto the denotation's
    ``id``).
    """
    doc = json.loads(gold_path.read_text(encoding="utf-8"))
    text = doc.get("text", "")
    length = len(text)

    type_of = {
        attr["subj"]: attr.get("obj")
        for attr in doc.get("attributes", [])
        if attr.get("pred") == "identifier_type"
    }

    spans: list[tuple[int, int, str]] = []
    all_mask = 0
    type_mask: dict[str, int] = {}
    for denotation in doc.get("denotations", []):
        span = denotation.get("span") or (denotation.get("spans") or [{}])[0]
        start, end = span.get("begin"), span.get("end")
        if start is None or end is None:
            continue
        code = type_of.get(denotation.get("id"), "UNKNOWN")
        mask = span_mask(start, end)
        all_mask |= mask
        type_mask[code] = type_mask.get(code, 0) | mask
        spans.append((start, end, code))

    return {"length": length, "all": all_mask, "type": type_mask, "spans": spans}


def load_predictions(model_dir: Path, doc_id: str, pass_number: int) -> Optional[list[tuple[int, int, str]]]:
    """Load ``(start, end, type)`` predictions for one note and pass.

    Returns ``None`` when the file is absent -- the caller treats a missing
    output as an empty prediction set (a full miss), so non-production is
    penalised rather than silently skipped.
    """
    if pass_number == 1:
        path = model_dir / "resolved" / doc_id / "resolved_predictions.json"
    else:
        path = model_dir / "pass_2" / "resolved" / doc_id / "resolved_predictions.json"
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None

    predictions: list[tuple[int, int, str]] = []
    for pred in doc.get("predictions", []):
        start, end, identifier_type = pred.get("start"), pred.get("end"), pred.get("identifier_type", "")
        if isinstance(start, int) and isinstance(end, int):
            predictions.append((start, end, identifier_type))
    return predictions


def doc_operations(base_dir: Path, doc_id: str, pass_number: int) -> dict[str, Any]:
    """Reliability and cost for one note/pass.

    Segment count, finish reason, and tokens come from ``raw_responses`` (always
    present); usable/valid envelope counts come from ``validation`` when that
    subtree exists. ``base_dir`` is the pass-specific root (the model dir for
    Pass 1, its ``pass_2`` child for Pass 2).
    """
    segments = stop = prompt_tokens = completion_tokens = 0
    raw_dir = base_dir / "raw_responses" / doc_id
    for raw_path in sorted(raw_dir.glob("*.raw.json")):
        if not raw_path.name.startswith(f"P{pass_number}-"):
            continue
        try:
            envelope = json.loads(raw_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        segments += 1
        choice = (envelope.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "stop":
            stop += 1
        usage = envelope.get("usage") or {}
        prompt_tokens += usage.get("prompt_tokens", 0) or 0
        completion_tokens += usage.get("completion_tokens", 0) or 0

    usable = valid = validated_seen = 0
    val_dir = base_dir / "validation" / doc_id
    for val_path in val_dir.glob("*.validation.json"):
        try:
            record = json.loads(val_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if not str(record.get("scope_id", "")).startswith(f"P{pass_number}"):
            continue
        validated_seen += 1
        if record.get("envelope_usable"):
            usable += 1
        if record.get("strict_schema_valid"):
            valid += 1

    have_validation = validated_seen > 0
    complete = segments > 0 and stop == segments and (usable == segments if have_validation else True)
    return {
        "segments": segments, "stop": stop, "usable": usable, "valid": valid,
        "have_validation": have_validation, "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens, "complete": complete,
    }


# --- per-document scoring ----------------------------------------------------

def score_document(gold: dict[str, Any], predictions: list[tuple[int, int, str]]) -> dict[str, Any]:
    """Score one prediction set against one gold record.

    Character recall is type-agnostic: a gold PHI character counts as covered if
    it falls under *any* redaction. Type-matched counts (for per-category
    precision) require the redaction's type prefix to match the gold type.
    """
    length = gold["length"]

    pred_all = 0
    pred_by_type: dict[str, int] = {}
    for start, end, identifier_type in predictions:
        mask = span_mask(start, min(end, length))
        pred_all |= mask
        prefix = code_prefix(identifier_type)
        pred_by_type[prefix] = pred_by_type.get(prefix, 0) | mask

    gold_all = gold["all"]
    gt = popcount(gold_all)
    pred_chars = popcount(pred_all)
    tp = popcount(gold_all & pred_all)
    record = {
        "doc_chars": length, "gt": gt, "pred_chars": pred_chars, "tp": tp,
        "fp": pred_chars - tp, "fn": gt - tp,
        "tn": length - popcount(gold_all | pred_all), "residual": gt - tp,
    }

    # Per-type character coverage: type-agnostic (any redaction) and type-matched.
    per_type: dict[str, dict[str, int]] = {}
    for code, gold_mask in gold["type"].items():
        matched = pred_by_type.get(code_prefix(code), 0)
        per_type[code] = {
            "gt": popcount(gold_mask),
            "tp": popcount(gold_mask & pred_all),            # covered by any redaction
            "tpM": popcount(gold_mask & matched),            # covered by a same-type redaction
            "pcM": popcount(matched),                        # chars predicted AS this type
            "fn": popcount(gold_mask) - popcount(gold_mask & pred_all),
            "spans": 0, "any": 0, "full": 0, "strict": 0, "relaxed": 0,
        }

    # Span-level detection: any / full / strict (exact) / relaxed (+-2).
    predicted_spans = {(start, end) for start, end, _ in predictions}
    all_spans = {"spans": 0, "any": 0, "full": 0, "strict": 0, "relaxed": 0}
    for start, end, code in gold["spans"]:
        covered = popcount(span_mask(start, end) & pred_all)
        width = end - start
        hits = {
            "spans": 1,
            "any": 1 if covered > 0 else 0,
            "full": 1 if covered == width else 0,
            "strict": 1 if (start, end) in predicted_spans else 0,
            "relaxed": 1 if relaxed_hit(start, end, predicted_spans) else 0,
        }
        for key, value in hits.items():
            all_spans[key] += value
            per_type[code][key] += value

    # Relaxed-precision numerator: predicted spans within +-2 of some gold span.
    gold_spans = {(start, end) for start, end, _ in gold["spans"]}
    record["relaxed_pred_tp"] = sum(1 for span in predicted_spans if relaxed_hit(span[0], span[1], gold_spans))
    record["per_type"] = per_type
    record["all_spans"] = all_spans
    record["pred_span_count"] = len(predicted_spans)
    return record


# --- aggregation across documents -------------------------------------------

COHORTS = ("all_expected", "operational_complete")


def aggregate(doc_records: list[tuple[str, dict[str, Any], dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Aggregate per-document records into each cohort's micro sums + macro lists.

    ``doc_records`` is a list of ``(doc_id, score_record, ops_record)``.
    ``all_expected`` keeps every note (a missing output is a full miss);
    ``operational_complete`` keeps only notes whose every segment finished
    cleanly and (where validated) was usable.
    """
    out: dict[str, dict[str, Any]] = {}
    for cohort in COHORTS:
        selected = [
            (doc_id, score, ops) for (doc_id, score, ops) in doc_records
            if cohort == "all_expected" or ops.get("complete")
        ]
        agg: dict[str, Any] = {
            "n_notes": len(selected),
            "gt": 0, "pred_chars": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "doc_chars": 0,
            "spans": 0, "any": 0, "full": 0, "strict": 0, "relaxed": 0,
            "relaxed_pred": 0, "pred_span_count": 0,
            "zero_resid": 0, "leak": 0, "resid_list": [],
            "prompt_tokens": 0, "completion_tokens": 0,
            "expected_requests": 0, "val_requests": 0, "usable": 0, "valid": 0, "stop": 0,
            "recall_list": [], "precision_list": [], "f1_list": [],
            "strict_recall_list": [], "relaxed_recall_list": [],
            "per_type": {},
        }

        for _doc_id, score, ops in selected:
            for key in ("gt", "pred_chars", "tp", "fp", "fn", "tn", "doc_chars", "pred_span_count"):
                agg[key] += score[key]
            for key in ("spans", "any", "full", "strict", "relaxed"):
                agg[key] += score["all_spans"][key]
            agg["relaxed_pred"] += score["relaxed_pred_tp"]

            agg["resid_list"].append(score["residual"])
            agg["zero_resid"] += 1 if score["residual"] == 0 else 0
            agg["leak"] += 1 if score["residual"] > 0 else 0

            agg["expected_requests"] += ops["segments"]
            agg["stop"] += ops["stop"]
            agg["prompt_tokens"] += ops["prompt_tokens"]
            agg["completion_tokens"] += ops["completion_tokens"]
            if ops["have_validation"]:
                agg["val_requests"] += ops["segments"]
                agg["usable"] += ops["usable"]
                agg["valid"] += ops["valid"]

            # Per-note (macro) rates; skip a note whose denominator is zero.
            doc_recall = rate(score["tp"], score["gt"])
            doc_precision = rate(score["tp"], score["tp"] + score["fp"])
            doc_f1 = rate(2 * score["tp"], 2 * score["tp"] + score["fp"] + score["fn"])
            if doc_recall is not None:
                agg["recall_list"].append(doc_recall)
            if doc_precision is not None:
                agg["precision_list"].append(doc_precision)
            if doc_f1 is not None:
                agg["f1_list"].append(doc_f1)
            note_spans = score["all_spans"]["spans"]
            if note_spans:
                agg["strict_recall_list"].append(score["all_spans"]["strict"] / note_spans)
                agg["relaxed_recall_list"].append(score["all_spans"]["relaxed"] / note_spans)

            for code, values in score["per_type"].items():
                bucket = agg["per_type"].setdefault(code, {
                    "gt": 0, "tp": 0, "tpM": 0, "pcM": 0, "fn": 0,
                    "spans": 0, "any": 0, "full": 0, "strict": 0, "relaxed": 0,
                })
                for key, value in values.items():
                    bucket[key] += value

        out[cohort] = agg
    return out


# --- row emission ------------------------------------------------------------

COLUMNS = [
    "model_id", "hf_link", "hf_gguf_repo", "developer", "family",
    "params_total_B", "params_active_B", "moe", "is_medical", "is_reasoning",
    "quant", "n_docs", "doc_provenance",
    "pass", "surface", "cohort", "phi_type",
    "n_notes", "doc_chars_total", "gt_phi_chars", "pred_phi_chars",
    "gt_span_count", "pred_span_count",
    # character level -- type-agnostic (headline)
    "char_tp", "char_fp", "char_fn", "char_tn",
    "char_recall", "char_precision", "char_f1", "char_specificity",
    "over_redaction_rate", "residual_phi_chars", "residual_phi_char_rate",
    # character level -- type-matched (per-category; equals the agnostic twin on ALL)
    "char_recall_typematched", "char_precision_typematched", "char_f1_typematched",
    # span / entity level -- any, full, strict (exact), relaxed (+-2, i2b2)
    "any_span_tp", "any_span_recall", "full_span_tp", "full_span_recall",
    "strict_span_tp", "strict_span_recall", "strict_span_precision", "strict_span_f1",
    "relaxed_span_tp", "relaxed_span_recall", "relaxed_span_precision", "relaxed_span_f1",
    # note level (ALL row only)
    "zero_residual_notes", "zero_residual_note_rate", "notes_with_leak", "notes_with_leak_rate",
    "mean_residual_per_note", "median_residual_per_note", "max_residual_per_note",
    # reliability & cost (ALL row only)
    "expected_requests", "usable_requests", "usable_rate",
    "json_valid_requests", "json_valid_rate", "finish_stop_rate",
    "prompt_tokens", "completion_tokens",
    # macro (per-note, unweighted) alongside the micro headline -- ALL row only
    "char_recall_macro", "char_precision_macro", "char_f1_macro",
    "strict_span_recall_macro", "relaxed_span_recall_macro",
]


def emit_rows(meta: dict[str, Any], pass_label: str, cohort: str, agg: dict[str, Any],
              gold_types_present: list[str]) -> list[dict[str, Any]]:
    """Emit the ``ALL`` row (every column) and the partial per-type rows."""
    rows: list[dict[str, Any]] = []
    base = {**meta, "pass": pass_label, "surface": "final", "cohort": cohort}
    n = agg["n_notes"]

    gt, tp, fp, fn, tn, pred_chars = agg["gt"], agg["tp"], agg["fp"], agg["fn"], agg["tn"], agg["pred_chars"]
    precision = rate(tp, tp + fp)
    recall = rate(tp, gt)
    strict_recall = rate(agg["strict"], agg["spans"])
    strict_precision = rate(agg["strict"], agg["pred_span_count"])
    relaxed_recall = rate(agg["relaxed"], agg["spans"])
    relaxed_precision = rate(agg["relaxed_pred"], agg["pred_span_count"])

    rows.append({
        **base, "phi_type": "ALL",
        "n_notes": n, "doc_chars_total": agg["doc_chars"], "gt_phi_chars": gt,
        "pred_phi_chars": pred_chars, "gt_span_count": agg["spans"],
        "pred_span_count": agg["pred_span_count"],
        "char_tp": tp, "char_fp": fp, "char_fn": fn, "char_tn": tn,
        "char_recall": recall, "char_precision": precision, "char_f1": f1(precision, recall),
        "char_specificity": rate(tn, tn + fp),
        "over_redaction_rate": rate(fp, tp + fp),
        "residual_phi_chars": fn, "residual_phi_char_rate": rate(fn, gt),
        # On the ALL row the type-matched trio mirrors the type-agnostic trio;
        # the genuine per-category signal lives in the per-type rows below.
        "char_recall_typematched": recall, "char_precision_typematched": precision,
        "char_f1_typematched": f1(precision, recall),
        "any_span_tp": agg["any"], "any_span_recall": rate(agg["any"], agg["spans"]),
        "full_span_tp": agg["full"], "full_span_recall": rate(agg["full"], agg["spans"]),
        "strict_span_tp": agg["strict"], "strict_span_recall": strict_recall,
        "strict_span_precision": strict_precision, "strict_span_f1": f1(strict_precision, strict_recall),
        "relaxed_span_tp": agg["relaxed"], "relaxed_span_recall": relaxed_recall,
        "relaxed_span_precision": relaxed_precision, "relaxed_span_f1": f1(relaxed_precision, relaxed_recall),
        "zero_residual_notes": agg["zero_resid"], "zero_residual_note_rate": rate(agg["zero_resid"], n),
        "notes_with_leak": agg["leak"], "notes_with_leak_rate": rate(agg["leak"], n),
        "mean_residual_per_note": mean(agg["resid_list"]),
        "median_residual_per_note": statistics.median(agg["resid_list"]) if agg["resid_list"] else None,
        "max_residual_per_note": max(agg["resid_list"]) if agg["resid_list"] else None,
        "expected_requests": agg["expected_requests"], "usable_requests": agg["usable"],
        "usable_rate": rate(agg["usable"], agg["val_requests"]),
        "json_valid_requests": agg["valid"], "json_valid_rate": rate(agg["valid"], agg["val_requests"]),
        "finish_stop_rate": rate(agg["stop"], agg["expected_requests"]),
        "prompt_tokens": agg["prompt_tokens"], "completion_tokens": agg["completion_tokens"],
        "char_recall_macro": mean(agg["recall_list"]), "char_precision_macro": mean(agg["precision_list"]),
        "char_f1_macro": mean(agg["f1_list"]),
        "strict_span_recall_macro": mean(agg["strict_recall_list"]),
        "relaxed_span_recall_macro": mean(agg["relaxed_recall_list"]),
    })

    # Per-type rows carry recall/coverage, type-matched precision/F1, and span
    # recall; type-agnostic precision, note-, reliability-, and macro-columns are
    # left blank because a false positive is not attributable to a gold type.
    for code in gold_types_present:
        values = agg["per_type"].get(code)
        if not values or values["gt"] == 0:
            continue
        matched_precision = rate(values["tpM"], values["pcM"])
        matched_recall = rate(values["tpM"], values["gt"])
        rows.append({
            **base, "phi_type": code,
            "n_notes": n, "gt_phi_chars": values["gt"], "gt_span_count": values["spans"],
            "char_tp": values["tp"], "char_fn": values["fn"],
            "char_recall": rate(values["tp"], values["gt"]),
            "char_recall_typematched": matched_recall,
            "char_precision_typematched": matched_precision,
            "char_f1_typematched": f1(matched_precision, matched_recall),
            "residual_phi_chars": values["fn"], "residual_phi_char_rate": rate(values["fn"], values["gt"]),
            "any_span_tp": values["any"], "any_span_recall": rate(values["any"], values["spans"]),
            "full_span_tp": values["full"], "full_span_recall": rate(values["full"], values["spans"]),
            "strict_span_tp": values["strict"], "strict_span_recall": rate(values["strict"], values["spans"]),
            "relaxed_span_tp": values["relaxed"], "relaxed_span_recall": rate(values["relaxed"], values["spans"]),
        })
    return rows


# --- model discovery + metadata ---------------------------------------------

def discover_models(pred_dir: Path) -> list[str]:
    """Every immediate subdirectory of ``pred_dir`` that holds a ``resolved/`` tree."""
    return sorted(
        entry.name for entry in pred_dir.iterdir()
        if entry.is_dir() and (entry / "resolved").is_dir()
    )


def list_documents(model_dir: Path) -> list[str]:
    """Document ids this model produced Pass-1 output for."""
    resolved = model_dir / "resolved"
    if not resolved.is_dir():
        return []
    return sorted(
        entry.name for entry in resolved.iterdir()
        if entry.is_dir() and (entry / "resolved_predictions.json").exists()
    )


def load_config(model_dir: Path) -> dict[str, Any]:
    """Read optional ``model_config.json`` metadata; blanks if absent."""
    config: dict[str, Any] = {}
    path = model_dir / "model_config.json"
    if path.exists():
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            config = {}
    repo = config.get("hf_gguf_repo", "") or ""
    return {
        "developer": config.get("developer", ""), "family": config.get("family", ""),
        "params_total_B": config.get("params_total_B"), "params_active_B": config.get("params_active_B"),
        "moe": config.get("moe"), "is_medical": config.get("is_medical"),
        "is_reasoning": config.get("reasoning"), "quant": config.get("quant_target", ""),
        "hf_gguf_repo": repo, "hf_link": f"https://huggingface.co/{repo}" if repo else "",
    }


# --- driver ------------------------------------------------------------------

def score_all(pred_dir: Path, gold_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Score every model under ``pred_dir`` against gold in ``gold_dir``."""
    gold_cache: dict[str, dict[str, Any]] = {}

    def gold_for(doc_id: str) -> dict[str, Any]:
        if doc_id not in gold_cache:
            gold_cache[doc_id] = load_gold(gold_dir / f"{doc_id}.json")
        return gold_cache[doc_id]

    metrics_rows: list[dict[str, Any]] = []
    per_doc_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for model_id in discover_models(pred_dir):
        model_dir = pred_dir / model_id
        config = load_config(model_dir)
        docs = list_documents(model_dir)
        meta = {
            "model_id": model_id, "n_docs": len(docs), "doc_provenance": "",
            "hf_link": config["hf_link"], "hf_gguf_repo": config["hf_gguf_repo"],
            "developer": config["developer"], "family": config["family"],
            "params_total_B": config["params_total_B"], "params_active_B": config["params_active_B"],
            "moe": config["moe"], "is_medical": config["is_medical"],
            "is_reasoning": config["is_reasoning"], "quant": config["quant"],
        }

        gold_types: set[str] = set()
        agg_by_pass: dict[str, dict[str, Any]] = {}
        for pass_number, pass_label in ((1, "pass1"), (2, "cumulative_pass2")):
            doc_records = []
            base_dir = model_dir if pass_number == 1 else model_dir / "pass_2"
            for doc_id in docs:
                gold = gold_for(doc_id)
                gold_types |= set(gold["type"].keys())
                predictions = load_predictions(model_dir, doc_id, pass_number)
                if predictions is None:
                    predictions = []
                score = score_document(gold, predictions)
                ops = doc_operations(base_dir, doc_id, pass_number)
                doc_records.append((doc_id, score, ops))
                per_doc_rows.append({
                    "model_id": model_id, "n_docs": len(docs), "doc": doc_id, "pass_": pass_label,
                    "doc_chars": score["doc_chars"], "gt_phi_chars": score["gt"],
                    "pred_phi_chars": score["pred_chars"], "char_tp": score["tp"],
                    "char_fp": score["fp"], "char_fn": score["fn"],
                    "residual_phi_chars": score["residual"], "zero_residual": int(score["residual"] == 0),
                    "gt_spans": score["all_spans"]["spans"], "any_span_hits": score["all_spans"]["any"],
                    "n_segments": ops["segments"], "usable_segments": ops["usable"],
                    "finish_stop": ops["stop"], "operational_complete": int(ops["complete"]),
                })
            agg_by_pass[pass_label] = aggregate(doc_records)

        gold_types_sorted = sorted(gold_types)
        for pass_label, cohorts in agg_by_pass.items():
            for cohort, agg in cohorts.items():
                metrics_rows.extend(emit_rows(meta, pass_label, cohort, agg, gold_types_sorted))
        manifest_rows.append({**meta, "gold_types_present": ";".join(gold_types_sorted),
                              "model_dir": model_id})

        headline = agg_by_pass["cumulative_pass2"]["all_expected"]
        print(f"{model_id:32} n={len(docs):<4} "
              f"recall(cum,p2,all)={rate(headline['tp'], headline['gt']) or 0:.3f}", flush=True)

    return {"metrics": metrics_rows, "per_doc": per_doc_rows, "manifest": manifest_rows}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(out_dir: Path, gold_dir: Path, results: dict[str, list[dict[str, Any]]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "metrics_long.csv", COLUMNS, results["metrics"])
    per_doc_columns = [
        "model_id", "n_docs", "doc", "pass_", "doc_chars", "gt_phi_chars", "pred_phi_chars",
        "char_tp", "char_fp", "char_fn", "residual_phi_chars", "zero_residual", "gt_spans",
        "any_span_hits", "n_segments", "usable_segments", "finish_stop", "operational_complete",
    ]
    write_csv(out_dir / "per_doc_long.csv", per_doc_columns, results["per_doc"])
    manifest_columns = [
        "model_id", "n_docs", "doc_provenance", "hf_link", "hf_gguf_repo", "developer", "family",
        "params_total_B", "params_active_B", "moe", "is_medical", "is_reasoning", "quant",
        "gold_types_present", "model_dir",
    ]
    write_csv(out_dir / "model_manifest.csv", manifest_columns, results["manifest"])

    (out_dir / "run_manifest.json").write_text(json.dumps({
        "code_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gold_source": str(gold_dir),
        "offset_unit": "Unicode codepoint; half-open [start,end)",
        "reference_text": "gold PubAnnotation `text`",
        "models": [row["model_id"] for row in results["manifest"]],
        "n_models": len(results["manifest"]),
        "cohorts": [
            "all_expected (all docs; a missing output is a full miss)",
            "operational_complete (docs whose every segment finished stop and, if validated, was usable)",
        ],
        "passes": ["pass1", "cumulative_pass2 (Pass 1 union Pass 2)"],
        "surfaces": ["final (deterministic redaction from resolved_predictions.json)"],
        "recall_definition": "type-agnostic character coverage: gold PHI chars under any redaction / gold PHI chars",
        "span_recall": [
            "any_span (>=1 char covered)", "full_span (all chars covered)",
            "strict (exact boundary match)", "relaxed (|dstart|<=2 AND |dend|<=2 codepoints, i2b2-style)",
        ],
        "entity_relaxed_tolerance": REL_TOL,
        "averaging": "micro by default (sum counts across docs, then rate); *_macro = mean of per-note rates",
        "n_cols_metrics_long": len(COLUMNS),
        "n_rows_metrics_long": len(results["metrics"]),
        "n_rows_per_doc": len(results["per_doc"]),
    }, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score de-identification predictions against gold.")
    parser.add_argument("--pred-dir", type=Path, required=True,
                        help="root holding one subdirectory of outputs per model")
    parser.add_argument("--gold-dir", type=Path, required=True,
                        help="directory of gold PubAnnotation <doc>.json files")
    parser.add_argument("--out-dir", type=Path, default=Path("out"),
                        help="directory to write the metric tables into")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    results = score_all(args.pred_dir, args.gold_dir)
    write_outputs(args.out_dir, args.gold_dir, results)
    print("-" * 60)
    print(f"wrote {len(results['metrics'])} rows -> {args.out_dir / 'metrics_long.csv'}")
    print(f"wrote {len(results['per_doc'])} rows -> {args.out_dir / 'per_doc_long.csv'}")
    print(f"wrote {len(results['manifest'])} models -> {args.out_dir / 'model_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
