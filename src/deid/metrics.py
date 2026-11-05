"""Score reference annotations against source-offset predictions.

Character, span, and note-level accuracy use the reference PubAnnotation JSON
and resolved_predictions.json. Operational metrics additionally read response
telemetry, validation records, and the expected-segment manifests. The scorer
does not render or read a final redacted note.

Offsets use Unicode codepoints and half-open [start, end) intervals. Outputs do
not contain clinical text, but document IDs, hashes, and provenance can still be
sensitive. Keep real-data artifacts in controlled storage.

    python -m deid.metrics --pred-dir OUTPUT_ROOT --gold-dir GOLD --out-dir out/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any, Optional

from .pipeline import CANONICAL_TYPES, PREDICTIONS_SCHEMA_VERSION
from .provenance import expected_scopes, file_sha256, git_revision


class PredictionArtifactError(ValueError):
    """Raised when a resolved-prediction artifact fails integrity validation."""


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
    if precision == 0 or recall == 0:
        return 0.0
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
    if not isinstance(doc, dict):
        raise ValueError(f"gold artifact must be an object: {gold_path}")
    text = doc.get("text")
    if not isinstance(text, str):
        raise ValueError(f"gold artifact must contain string text: {gold_path}")
    length = len(text)
    if "sourceid" in doc and doc["sourceid"] != gold_path.stem:
        raise ValueError(f"gold sourceid does not match document filename: {gold_path}")
    denotations, attributes = doc.get("denotations"), doc.get("attributes", [])
    if not isinstance(denotations, list) or not isinstance(attributes, list):
        raise ValueError(f"gold denotations and attributes must be lists: {gold_path}")
    type_of: dict[str, str] = {}
    for attr in attributes:
        if not isinstance(attr, dict):
            raise ValueError(f"gold attribute must be an object: {gold_path}")
        if attr.get("pred") != "identifier_type":
            continue
        subject, code = attr.get("subj"), attr.get("obj")
        if not isinstance(subject, str) or not isinstance(code, str) or code not in CANONICAL_TYPES:
            raise ValueError(f"gold identifier_type is missing or noncanonical: {gold_path}")
        if subject in type_of:
            raise ValueError(f"duplicate gold identifier_type attribute: {gold_path}")
        type_of[subject] = code

    spans: list[tuple[int, int, str]] = []
    all_mask = 0
    type_mask: dict[str, int] = {}
    ids: set[str] = set()
    seen: set[tuple[int, int, str]] = set()
    for denotation in denotations:
        if not isinstance(denotation, dict):
            raise ValueError(f"gold denotation must be an object: {gold_path}")
        annotation_id = denotation.get("id")
        if not isinstance(annotation_id, str) or not annotation_id or annotation_id in ids:
            raise ValueError(f"gold denotation id must be unique and nonempty: {gold_path}")
        ids.add(annotation_id)
        if "span" in denotation and "spans" in denotation:
            raise ValueError(f"ambiguous gold span representation: {gold_path}")
        span = denotation.get("span")
        if "spans" in denotation:
            parts = denotation["spans"]
            if not isinstance(parts, list) or len(parts) != 1:
                raise ValueError(f"gold requires one contiguous interval per denotation: {gold_path}")
            span = parts[0]
        if not isinstance(span, dict):
            raise ValueError(f"gold span must contain begin/end: {gold_path}")
        start, end = span.get("begin"), span.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= length:
            raise ValueError(f"gold span has invalid Unicode codepoint bounds: {gold_path}")
        code = type_of.get(annotation_id)
        if code is None:
            raise ValueError(f"gold denotation lacks an identifier_type: {gold_path}")
        if (start, end, code) in seen:
            raise ValueError(f"duplicate gold span: {gold_path}")
        seen.add((start, end, code))
        mask = span_mask(start, end)
        all_mask |= mask
        type_mask[code] = type_mask.get(code, 0) | mask
        spans.append((start, end, code))
    if set(type_of) - ids:
        raise ValueError(f"gold identifier_type refers to an absent denotation: {gold_path}")
    source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    reference_sha256 = hashlib.sha256(json.dumps(
        [source_sha256, length, sorted(spans)], ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return {
        "length": length,
        "source_sha256": source_sha256,
        "reference_sha256": reference_sha256,
        "all": all_mask,
        "type": type_mask,
        "spans": spans,
    }


def reference_set_sha256(gold_by_document: dict[str, dict[str, Any]]) -> str:
    """Bind document identities, text, and typed offsets independently of JSON formatting."""
    payload = [(doc_id, gold["reference_sha256"])
               for doc_id, gold in sorted(gold_by_document.items())]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_predictions(model_dir: Path, doc_id: str, pass_number: int,
                     expected_source_sha256: str,
                     source_length: int) -> Optional[list[tuple[int, int, str]]]:
    """Load ``(start, end, type)`` predictions for one note and pass.

    Returns ``None`` when the file is absent -- the caller treats a missing
    Pass 1 output as an empty set; a missing cumulative Pass 2 artifact retains
    the valid Pass 1 predictions. A present but malformed or
    mismatched artifact raises :class:`PredictionArtifactError`; corrupt output
    must never be silently converted into a scientific result.
    """
    if pass_number == 1:
        path = model_dir / "resolved" / doc_id / "resolved_predictions.json"
    else:
        path = model_dir / "pass_2" / "resolved" / doc_id / "resolved_predictions.json"
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PredictionArtifactError(f"invalid prediction JSON: {path}") from exc

    required_keys = {
        "schema_version", "document_id", "source_sha256", "offset_unit",
        "contains_literal_text", "predictions",
    }
    if not isinstance(doc, dict) or set(doc) != required_keys:
        raise PredictionArtifactError(f"invalid prediction artifact schema: {path}")
    if doc.get("schema_version") != PREDICTIONS_SCHEMA_VERSION:
        raise PredictionArtifactError(f"prediction schema version mismatch: {path}")
    if doc.get("document_id") != doc_id:
        raise PredictionArtifactError(f"prediction document id mismatch for {doc_id}: {path}")
    if doc.get("source_sha256") != expected_source_sha256:
        raise PredictionArtifactError(f"prediction source hash mismatch for {doc_id}: {path}")
    if doc.get("offset_unit") != "Unicode codepoint; half-open [start,end)":
        raise PredictionArtifactError(f"prediction offset unit mismatch for {doc_id}: {path}")
    if doc.get("contains_literal_text") is not False:
        raise PredictionArtifactError(f"prediction artifact may contain literal text: {path}")
    if not isinstance(doc.get("predictions"), list):
        raise PredictionArtifactError(f"predictions must be a list: {path}")

    predictions: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()
    for index, pred in enumerate(doc["predictions"]):
        if not isinstance(pred, dict) or set(pred) != {"start", "end", "identifier_type"}:
            raise PredictionArtifactError(
                f"invalid prediction item {index} for {doc_id}: {path}"
            )
        start, end = pred["start"], pred["end"]
        identifier_type = pred["identifier_type"]
        if type(start) is not int or type(end) is not int:
            raise PredictionArtifactError(
                f"prediction offsets must be integers for {doc_id}: {path}"
            )
        if not (0 <= start < end <= source_length):
            raise PredictionArtifactError(
                f"prediction offsets out of bounds for {doc_id}: {path}"
            )
        if identifier_type == "UNSPECIFIED":
            manifest_path = model_dir / "comparator_manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise PredictionArtifactError("untyped predictions require a Presidio comparator manifest") from exc
            if (not isinstance(manifest, dict)
                    or manifest.get("schema_version") != "comparator-manifest-v1"
                    or manifest.get("backend") != "presidio"):
                raise PredictionArtifactError("untyped predictions require a Presidio comparator manifest")
        elif not isinstance(identifier_type, str) or identifier_type not in CANONICAL_TYPES:
            raise PredictionArtifactError(
                f"invalid prediction identifier type for {doc_id}: {path}"
            )
        item = (start, end, identifier_type)
        if item in seen:
            raise PredictionArtifactError(f"duplicate prediction for {doc_id}: {path}")
        seen.add(item)
        predictions.append(item)
    return predictions


def _comparator_operations(model_dir: Path, doc_id: str, pass_number: int,
                           source_sha256: Optional[str]) -> dict[str, Any]:
    """Comparator completion has pipeline records, never LLM finish/format flags."""
    manifest = json.loads((model_dir / "comparator_manifest.json").read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("backend") != "presidio"
            or manifest.get("schema_version") != "comparator-manifest-v1"):
        raise ValueError("invalid Presidio comparator manifest")
    plans = [expected_scopes(model_dir if number == 1 else model_dir / "pass_2",
                             doc_id, number, source_sha256) for number in (1, 2)]
    expected = plans[pass_number - 1]
    timing_path = model_dir / "pipeline_timing" / f"{doc_id}.json"
    verified = complete = False
    if timing_path.exists():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        if (not isinstance(timing, dict) or timing.get("document_id") != doc_id
                or (source_sha256 is not None and timing.get("source_sha256") != source_sha256)):
            raise ValueError(f"comparator timing document/source mismatch: {doc_id}")
        seconds = timing.get("pipeline_seconds")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"invalid recorded comparator pipeline time: {doc_id}")
        notes = manifest.get("notes", [])
        planned = [note for note in notes if isinstance(note, dict)
                   and note.get("document_id") == doc_id] if isinstance(notes, list) else []
        if len(planned) != 1 or planned[0].get("source_sha256") != timing.get("source_sha256"):
            raise ValueError(f"comparator timing does not match a uniquely planned source: {doc_id}")
        verified = timing.get("completed") is True and all(plan is not None for plan in plans) and all(
            type(timing.get(f"expected_segments_pass_{number}")) is int
            and timing[f"expected_segments_pass_{number}"] == len(plans[number - 1])
            for number in (1, 2)
        )
        complete = verified
    return {"segments": 0, "stop": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "expected_segments": len(expected) if expected is not None else None,
            "validation_records": 0, "usable": 0, "valid": 0, "have_validation": False,
            "coverage_verified": verified, "complete": complete, "llm_applicable": False}


def doc_operations(base_dir: Path, doc_id: str, pass_number: int,
                   source_sha256: Optional[str] = None) -> dict[str, Any]:
    """Read observed records and check them against the pre-request segment plan.

    Format rates remain conditional on available validation records. A missing
    plan (including legacy output) cannot establish operational completeness.
    ``segments`` is the observed parseable-response count, not the expected count.
    """
    model_dir = base_dir if pass_number == 1 else base_dir.parent
    if (model_dir / "comparator_manifest.json").exists():
        return _comparator_operations(model_dir, doc_id, pass_number, source_sha256)
    expected = expected_scopes(base_dir, doc_id, pass_number, source_sha256)
    raw_scopes: set[str] = set()
    validation_scopes: set[str] = set()
    artifacts_valid = True
    finishes: dict[str, Any] = {}
    segments = stop = prompt_tokens = completion_tokens = 0
    raw_dir = base_dir / "raw_responses" / doc_id
    for raw_path in sorted(raw_dir.glob("*.raw.json")):
        scope = raw_path.name[:-len(".raw.json")]
        if not scope.startswith(f"P{pass_number}-"):
            artifacts_valid = False
            continue
        try:
            envelope = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            artifacts_valid = False
            continue
        if not isinstance(envelope, dict):
            artifacts_valid = False
            continue
        raw_scopes.add(scope)
        segments += 1
        choices = envelope.get("choices")
        choice = choices[0] if (isinstance(choices, list) and choices
                                and isinstance(choices[0], dict)) else {}
        finishes[scope] = choice.get("finish_reason")
        if finishes[scope] == "stop":
            stop += 1
        usage_value = envelope.get("usage")
        usage = usage_value if isinstance(usage_value, dict) else {}
        prompt_value = usage.get("prompt_tokens")
        completion_value = usage.get("completion_tokens")
        prompt_tokens += prompt_value if type(prompt_value) is int else 0
        completion_tokens += completion_value if type(completion_value) is int else 0

    usable = valid = validated_seen = 0
    val_dir = base_dir / "validation" / doc_id
    for val_path in val_dir.glob("*.validation.json"):
        scope = val_path.name[:-len(".validation.json")]
        try:
            record = json.loads(val_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            artifacts_valid = False
            continue
        if (not isinstance(record, dict)
                or not scope.startswith(f"P{pass_number}-")
                or record.get("scope_id") != scope
                or record.get("document_id") != doc_id
                or type(record.get("envelope_usable")) is not bool
                or type(record.get("strict_schema_valid")) is not bool):
            artifacts_valid = False
            continue
        validation_scopes.add(scope)
        validated_seen += 1
        usable += int(record["envelope_usable"])
        valid += int(record["strict_schema_valid"])
        if record.get("finish_reason") != finishes.get(scope):
            artifacts_valid = False

    if expected is not None and ((raw_scopes | validation_scopes) - expected):
        raise ValueError(f"unexpected segment records for {doc_id}, pass {pass_number}")

    coverage_verified = (expected is not None and artifacts_valid
                         and raw_scopes == expected and validation_scopes == expected)
    complete = coverage_verified and stop == len(expected) and usable == len(expected)
    return {
        "segments": segments, "stop": stop, "usable": usable, "valid": valid,
        "expected_segments": len(expected) if expected is not None else None,
        "coverage_verified": coverage_verified,
        "have_validation": validated_seen > 0, "validation_records": validated_seen,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "complete": complete,
    }

# --- per-document scoring ----------------------------------------------------

def score_document(gold: dict[str, Any], predictions: list[tuple[int, int, str]]) -> dict[str, Any]:
    """Score one prediction set against one gold record.

    Character recall is type-agnostic: a gold PHI character counts as covered if
    it falls under *any* redaction. Type-matched counts (for per-category
    precision) require the redaction's type prefix to match the gold type.

    Boundary metrics use the unique start/end intervals in the supplied resolved
    predictions. LLM artifacts preserve accepted-literal boundaries before the
    overlap union used for redaction; comparator artifacts may already contain
    merged intervals. Character and any/full-span coverage use the union mask.
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
        "typed_predictions": not any(code == "UNSPECIFIED" for _, _, code in predictions),
    }

    # Per-type character coverage: type-agnostic (any redaction) and type-matched.
    per_type: dict[str, dict[str, int]] = {}
    typed_codes = set(gold["type"]) | {code for _, _, code in predictions if code in CANONICAL_TYPES}
    for code in typed_codes:
        gold_mask = gold["type"].get(code, 0)
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
    record["strict_pred_tp"] = len(predicted_spans & gold_spans)
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
    ``all_expected`` keeps every note (missing Pass 1 is a full miss; missing
    cumulative Pass 2 retains Pass 1);
    ``operational_complete`` keeps only notes whose every segment finished
    with verified request coverage, a stop finish reason, and a usable response.
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
            "strict_pred": 0, "relaxed_pred": 0, "pred_span_count": 0,
            "zero_resid": 0, "leak": 0, "resid_list": [],
            "prompt_tokens": 0, "completion_tokens": 0,
            "expected_requests": 0, "expected_known": True, "val_requests": 0, "usable": 0, "valid": 0, "stop": 0,
            "recall_list": [], "precision_list": [], "f1_list": [],
            "strict_recall_list": [], "relaxed_recall_list": [],
            "per_type": {},
            "typed_predictions": all(score["typed_predictions"] for _, score, _ in doc_records),
            "llm_applicable": all(ops.get("llm_applicable", True) for _, _, ops in doc_records),
        }

        for _doc_id, score, ops in selected:
            agg["typed_predictions"] = agg["typed_predictions"] and score["typed_predictions"]
            agg["llm_applicable"] = agg["llm_applicable"] and ops.get("llm_applicable", True)
            for key in ("gt", "pred_chars", "tp", "fp", "fn", "tn", "doc_chars", "pred_span_count"):
                agg[key] += score[key]
            for key in ("spans", "any", "full", "strict", "relaxed"):
                agg[key] += score["all_spans"][key]
            agg["relaxed_pred"] += score["relaxed_pred_tp"]
            agg["strict_pred"] += score["strict_pred_tp"]

            agg["resid_list"].append(score["residual"])
            agg["zero_resid"] += 1 if score["residual"] == 0 else 0
            agg["leak"] += 1 if score["residual"] > 0 else 0

            if ops.get("expected_segments") is None:
                agg["expected_known"] = False
            else:
                agg["expected_requests"] += ops["expected_segments"]
            agg["stop"] += ops["stop"]
            agg["prompt_tokens"] += ops["prompt_tokens"]
            agg["completion_tokens"] += ops["completion_tokens"]
            if ops["have_validation"]:
                # Usability and strict-validity rates are conditional on the
                # validation records that actually exist, not on raw requests.
                agg["val_requests"] += ops["validation_records"]
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

        if not agg.pop("expected_known"):
            agg["expected_requests"] = None
        out[cohort] = agg
    return out


# --- row emission ------------------------------------------------------------

COLUMNS = [
    "model_id", "hf_link", "hf_gguf_repo", "developer", "family",
    "params_total_B", "params_effective_B", "params_active_B", "moe", "is_medical", "is_reasoning",
    "quant", "n_docs", "doc_provenance", "reference_set_sha256",
    "pass", "surface", "cohort", "phi_type",
    "n_notes", "doc_chars_total", "gt_phi_chars", "pred_phi_chars",
    "gt_span_count", "pred_span_count",
    # character level -- type-agnostic (headline)
    "char_tp", "char_fp", "char_fn", "char_tn",
    "char_recall", "char_precision", "char_f1", "char_specificity",
    "over_redaction_rate", "nonreference_removal_rate", "residual_phi_chars", "residual_phi_char_rate",
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
    strict_precision = rate(agg["strict_pred"], agg["pred_span_count"])
    relaxed_recall = rate(agg["relaxed"], agg["spans"])
    relaxed_precision = rate(agg["relaxed_pred"], agg["pred_span_count"])

    rows.append({
        **base, "phi_type": "ALL",
        "n_notes": n, "doc_chars_total": agg["doc_chars"], "gt_phi_chars": gt,
        "pred_phi_chars": pred_chars, "gt_span_count": agg["spans"],
        "pred_span_count": agg["pred_span_count"],
        "char_tp": tp, "char_fp": fp, "char_fn": fn, "char_tn": tn,
        "char_recall": recall, "char_precision": precision, "char_f1": rate(2 * tp, 2 * tp + fp + fn),
        "char_specificity": rate(tn, tn + fp),
        "over_redaction_rate": rate(fp, tp + fp),
        "nonreference_removal_rate": rate(fp, fp + tn),
        "residual_phi_chars": fn, "residual_phi_char_rate": rate(fn, gt),
        # On the ALL row the type-matched trio mirrors the type-agnostic trio;
        # the genuine per-category signal lives in the per-type rows below.
        "char_recall_typematched": recall, "char_precision_typematched": precision,
        "char_f1_typematched": rate(2 * tp, 2 * tp + fp + fn),
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
        "expected_requests": agg["expected_requests"],
        "usable_requests": agg["usable"] if agg["llm_applicable"] else None,
        "usable_rate": rate(agg["usable"], agg["val_requests"]) if agg["llm_applicable"] else None,
        "json_valid_requests": agg["valid"] if agg["llm_applicable"] else None,
        "json_valid_rate": rate(agg["valid"], agg["val_requests"]) if agg["llm_applicable"] else None,
        "finish_stop_rate": rate(agg["stop"], agg["expected_requests"]) if agg["llm_applicable"] else None,
        "prompt_tokens": agg["prompt_tokens"] if agg["llm_applicable"] else None,
        "completion_tokens": agg["completion_tokens"] if agg["llm_applicable"] else None,
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
        matched_precision = rate(values["tpM"], values["pcM"]) if agg["typed_predictions"] else None
        matched_recall = rate(values["tpM"], values["gt"]) if agg["typed_predictions"] else None
        rows.append({
            **base, "phi_type": code,
            "n_notes": n, "gt_phi_chars": values["gt"], "gt_span_count": values["spans"],
            "char_tp": values["tp"], "char_fn": values["fn"],
            "char_recall": rate(values["tp"], values["gt"]),
            "char_recall_typematched": matched_recall,
            "char_precision_typematched": matched_precision,
            "char_f1_typematched": (rate(2 * values["tpM"], values["pcM"] + values["gt"])
                                    if agg["typed_predictions"] else None),
            "residual_phi_chars": values["fn"], "residual_phi_char_rate": rate(values["fn"], values["gt"]),
            "any_span_tp": values["any"], "any_span_recall": rate(values["any"], values["spans"]),
            "full_span_tp": values["full"], "full_span_recall": rate(values["full"], values["spans"]),
            "strict_span_tp": values["strict"], "strict_span_recall": rate(values["strict"], values["spans"]),
            "relaxed_span_tp": values["relaxed"], "relaxed_span_recall": rate(values["relaxed"], values["spans"]),
        })
    return rows


# --- model discovery + metadata ---------------------------------------------

def discover_models(pred_dir: Path) -> list[str]:
    """Every immediate subdirectory of ``pred_dir`` (one attempted model run)."""
    return sorted(
        entry.name for entry in pred_dir.iterdir()
        if entry.is_dir()
    )


def list_documents(model_dir: Path, pass_number: int = 1) -> list[str]:
    """Document ids for which this model has a resolved artifact for one pass."""
    resolved = (model_dir / "resolved" if pass_number == 1
                else model_dir / "pass_2" / "resolved")
    if not resolved.is_dir():
        return []
    return sorted(
        entry.name for entry in resolved.iterdir()
        if entry.is_dir() and (entry / "resolved_predictions.json").exists()
    )


def output_documents(model_dir: Path) -> set[str]:
    """All document ids represented anywhere in a model's output subtrees."""
    doc_ids: set[str] = set()
    for relative in (
        Path("resolved"), Path("raw_responses"), Path("validation"), Path("expected_segments"),
        Path("pass_2/resolved"), Path("pass_2/raw_responses"), Path("pass_2/validation"),
        Path("pass_2/expected_segments"),
    ):
        subtree = model_dir / relative
        if subtree.is_dir():
            doc_ids.update(entry.name for entry in subtree.iterdir() if entry.is_dir())
    return doc_ids


def expected_documents(gold_dir: Path) -> list[str]:
    """Document ids defined by the locked gold set, which drives scoring scope."""
    if not gold_dir.is_dir():
        raise FileNotFoundError(f"gold directory not found: {gold_dir}")
    files = sorted(gold_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no gold JSON files found in {gold_dir}")
    doc_ids = [path.stem for path in files]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError(f"duplicate gold document ids in {gold_dir}")
    return doc_ids


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
        "params_total_B": config.get("params_total_B"),
        "params_effective_B": config.get("params_effective_B"),
        "params_active_B": config.get("params_active_B"),
        "moe": config.get("moe"), "is_medical": config.get("is_medical"),
        "is_reasoning": config.get("reasoning"), "quant": config.get("quant_target", ""),
        "hf_gguf_repo": repo, "hf_link": f"https://huggingface.co/{repo}" if repo else "",
    }


# --- driver ------------------------------------------------------------------

def score_all(pred_dir: Path, gold_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Score every model under ``pred_dir`` against gold in ``gold_dir``."""
    docs = expected_documents(gold_dir)
    expected_doc_set = set(docs)
    gold_cache = {doc_id: load_gold(gold_dir / f"{doc_id}.json") for doc_id in docs}
    reference_fingerprint = reference_set_sha256(gold_cache)

    metrics_rows: list[dict[str, Any]] = []
    per_doc_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for model_id in discover_models(pred_dir):
        model_dir = pred_dir / model_id
        config = load_config(model_dir)
        extra_docs = output_documents(model_dir) - expected_doc_set
        if extra_docs:
            raise ValueError(
                f"prediction outputs without matching gold for model {model_id}: "
                + ", ".join(sorted(extra_docs))
            )
        meta = {
            "model_id": model_id, "n_docs": len(docs), "doc_provenance": "",
            "reference_set_sha256": reference_fingerprint,
            "hf_link": config["hf_link"], "hf_gguf_repo": config["hf_gguf_repo"],
            "developer": config["developer"], "family": config["family"],
            "params_total_B": config["params_total_B"],
            "params_effective_B": config["params_effective_B"],
            "params_active_B": config["params_active_B"],
            "moe": config["moe"], "is_medical": config["is_medical"],
            "is_reasoning": config["is_reasoning"], "quant": config["quant"],
        }

        gold_types: set[str] = set()
        agg_by_pass: dict[str, dict[str, Any]] = {}
        first_pass: dict[str, list[tuple[int, int, str]]] = {}
        for pass_number, pass_label in ((1, "pass1"), (2, "cumulative_pass2")):
            doc_records = []
            base_dir = model_dir if pass_number == 1 else model_dir / "pass_2"
            for doc_id in docs:
                gold = gold_cache[doc_id]
                gold_types |= set(gold["type"].keys())
                predictions = load_predictions(
                    model_dir, doc_id, pass_number,
                    expected_source_sha256=gold["source_sha256"],
                    source_length=gold["length"],
                )
                if pass_number == 1:
                    predictions = predictions or []
                    first_pass[doc_id] = predictions
                elif predictions is None:
                    predictions = first_pass[doc_id]
                else:
                    prior_mask = current_mask = 0
                    for start, end, _code in first_pass[doc_id]:
                        prior_mask |= span_mask(start, end)
                    for start, end, _code in predictions:
                        current_mask |= span_mask(start, end)
                    if prior_mask & ~current_mask:
                        raise PredictionArtifactError(
                            f"cumulative Pass 2 removes valid Pass 1 coverage: {model_id}/{doc_id}"
                        )
                score = score_document(gold, predictions)
                ops = doc_operations(base_dir, doc_id, pass_number, gold["source_sha256"])
                if not ops.get("llm_applicable", True):
                    score["typed_predictions"] = False
                doc_records.append((doc_id, score, ops))
                per_doc_rows.append({
                    "model_id": model_id, "n_docs": len(docs), "doc": doc_id, "pass_": pass_label,
                    "source_sha256": gold["source_sha256"],
                    "reference_sha256": gold["reference_sha256"],
                    "reference_set_sha256": reference_fingerprint,
                    "doc_chars": score["doc_chars"], "gt_phi_chars": score["gt"],
                    "pred_phi_chars": score["pred_chars"], "char_tp": score["tp"],
                    "char_fp": score["fp"], "char_fn": score["fn"],
                    "residual_phi_chars": score["residual"], "zero_residual": int(score["residual"] == 0),
                    "gt_spans": score["all_spans"]["spans"], "any_span_hits": score["all_spans"]["any"],
                    "n_segments": ops["segments"] if ops.get("llm_applicable", True) else None,
                    "usable_segments": ops["usable"] if ops.get("llm_applicable", True) else None,
                    "finish_stop": ops["stop"] if ops.get("llm_applicable", True) else None,
                    "operational_complete": int(ops["complete"]),
                    "expected_segments": ops["expected_segments"],
                    "segment_coverage_verified": int(ops["coverage_verified"]),
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
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
        path.chmod(0o600)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_outputs(out_dir: Path, gold_dir: Path, results: dict[str, list[dict[str, Any]]]) -> None:
    if out_dir.is_symlink():
        raise ValueError(f"refusing symlinked output directory: {out_dir}")
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    out_dir.chmod(0o700)
    write_csv(out_dir / "metrics_long.csv", COLUMNS, results["metrics"])
    per_doc_columns = [
        "model_id", "n_docs", "doc", "pass_", "source_sha256", "reference_sha256",
        "reference_set_sha256", "doc_chars", "gt_phi_chars", "pred_phi_chars",
        "char_tp", "char_fp", "char_fn", "residual_phi_chars", "zero_residual", "gt_spans",
        "any_span_hits", "n_segments", "usable_segments", "finish_stop", "operational_complete",
        "expected_segments", "segment_coverage_verified",
    ]
    write_csv(out_dir / "per_doc_long.csv", per_doc_columns, results["per_doc"])
    manifest_columns = [
        "model_id", "n_docs", "doc_provenance", "reference_set_sha256", "hf_link", "hf_gguf_repo", "developer", "family",
        "params_total_B", "params_effective_B", "params_active_B", "moe", "is_medical", "is_reasoning", "quant",
        "gold_types_present", "model_dir",
    ]
    write_csv(out_dir / "model_manifest.csv", manifest_columns, results["manifest"])

    manifest_path = out_dir / "run_manifest.json"
    manifest_text = json.dumps({
        "repository_commit": git_revision(Path(__file__).resolve().parents[2]),
        "scorer_sha256": file_sha256(Path(__file__)),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gold_source": gold_dir.name,
        "reference_sets": sorted({row["reference_set_sha256"] for row in results["manifest"]}),
        "reference_denominators": [
            {field: row[field] for field in ("model_id", "n_notes", "doc_chars_total", "gt_phi_chars", "gt_span_count")}
            for row in results["metrics"] if row["pass"] == "pass1"
            and row["cohort"] == "all_expected" and row["phi_type"] == "ALL"
        ],
        "offset_unit": "Unicode codepoint; half-open [start,end)",
        "reference_text": "gold PubAnnotation `text`",
        "models": [row["model_id"] for row in results["manifest"]],
        "n_models": len(results["manifest"]),
        "cohorts": [
            "all_expected (all reference docs; missing Pass 1 has no detections; missing Pass 2 retains Pass 1)",
            "operational_complete (LLM: verified expected segments, every response finished stop and was usable; comparator: completed source-matched pipeline timing and both plans)",
        ],
        "request_coverage": "expected-segment manifests; legacy coverage is unknown without them",
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
    }, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.", suffix=".tmp", dir=out_dir, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(manifest_text)
        os.replace(temporary_path, manifest_path)
        manifest_path.chmod(0o600)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


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
