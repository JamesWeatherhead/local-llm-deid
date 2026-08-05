"""Run one model over a set of notes: two sequential passes, then redact.

For each note the runner performs the locked protocol:

1. **Pass 1** -- segment the note, send each segment to the model, validate and
   collect the ``(exact_text, type)`` pairs, then ground them to source offsets
   and write ``resolved/<doc>/resolved_predictions.json``.
2. **Pass 2** -- rebuild the note with Pass-1 regions replaced by ``[PHI]``,
   run the model again over that redacted representation, map any new findings
   back to source coordinates, and union them with Pass 1. New Pass-2 findings
   are only applied for a note whose every segment finished cleanly.

The output tree mirrors what the scorer reads::

    <out>/<model_id>/resolved/<doc>/resolved_predictions.json          # Pass 1
    <out>/<model_id>/pass_2/resolved/<doc>/resolved_predictions.json   # cumulative
    <out>/<model_id>/raw_responses/<doc>/P1-*.raw.json                 # reliability/cost
    <out>/<model_id>/validation/<doc>/P1-*.validation.json
    <out>/<model_id>/pass_2/raw_responses/<doc>/P2-*.raw.json
    <out>/<model_id>/pass_2/validation/<doc>/P2-*.validation.json

Run with ``--offline-stub`` to exercise the pipeline offline with the
deterministic test stub (no model, no GPU); every reported result comes from a
real local model via the llama-server client.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

from .inference import LlamaServerClient, StubExtractor
from .pipeline import (
    fixed_segments,
    identity_map,
    merge_pairs,
    merge_predictions,
    redacted_representation_with_map,
    resolve_pairs,
    resolved_predictions_document,
    SEGMENT_OVERLAP,
    SEGMENT_SIZE,
    strict_json_loads,
    union_regions,
    validate_response,
)

DOC_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as pretty, deterministic JSON, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def load_protocol(protocol_dir: Path) -> tuple[str, str, dict[str, Any]]:
    """Load the frozen system prompt, user template, and JSON schema."""
    system_prompt = (protocol_dir / "system_prompt.txt").read_text(encoding="utf-8")
    user_template = (protocol_dir / "user_prompt_template.txt").read_text(encoding="utf-8")
    schema = json.loads((protocol_dir / "model_output.schema.json").read_text(encoding="utf-8"))
    if user_template.count("{{CLINICAL_TEXT}}") != 1:
        raise ValueError("user_prompt_template.txt must contain exactly one {{CLINICAL_TEXT}} placeholder")
    return system_prompt, user_template, schema


def _run_segments(client, segments) -> tuple[list, list[dict[str, Any]]]:
    """Send each segment to the model; return accepted pairs and per-segment records."""
    pairs = []
    records: list[dict[str, Any]] = []
    for segment in segments:
        envelope = client.complete(segment.text)
        choice = (envelope.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason") or "error"
        content = (choice.get("message") or {}).get("content") or ""
        usage = envelope.get("usage") or {}

        usable = strict = False
        if finish_reason == "stop" and content:
            try:
                validated = validate_response(strict_json_loads(content), segment.text)
                usable = validated["envelope_usable"]
                strict = validated["strict_schema_valid"]
                pairs.extend(validated["pairs"])
            except ValueError:
                pass  # malformed JSON -> unusable segment

        records.append({
            "segment": segment,
            "envelope": envelope,
            "finish_reason": finish_reason,
            "envelope_usable": usable,
            "strict_schema_valid": strict,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
        })
    return pairs, records


def _strip_envelope_content(envelope: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the model envelope with the generated text removed.

    The scorer reads only ``finish_reason`` and ``usage`` from raw responses,
    never the message content, so blanking that content (which on real data
    quotes back the PHI the model found) keeps every metric identical while
    making the persisted ``raw_responses`` free of PHI.
    """
    stripped = dict(envelope)
    choices = []
    for choice in envelope.get("choices") or []:
        choice = dict(choice)
        message = dict(choice.get("message") or {})
        if "content" in message:
            message["content"] = ""
        message.pop("reasoning_content", None)
        choice["message"] = message
        choices.append(choice)
    if choices:
        stripped["choices"] = choices
    return stripped


def _persist_segments(base_dir: Path, doc_id: str, pass_number: int, records: list[dict[str, Any]],
                      strip_content: bool = False) -> None:
    """Write the raw-response and validation artifacts the scorer reads."""
    for record in records:
        scope_id = f"P{pass_number}-{record['segment'].segment_id}"
        envelope = _strip_envelope_content(record["envelope"]) if strip_content else record["envelope"]
        write_json(base_dir / "raw_responses" / doc_id / f"{scope_id}.raw.json", envelope)
        write_json(base_dir / "validation" / doc_id / f"{scope_id}.validation.json", {
            "document_id": doc_id,
            "scope_id": scope_id,
            "envelope_usable": record["envelope_usable"],
            "strict_schema_valid": record["strict_schema_valid"],
            "finish_reason": record["finish_reason"],
            "contains_literal_text": False,
        })


def process_document(client, doc_id: str, source: str, model_dir: Path,
                     segment_size: int = SEGMENT_SIZE, overlap: int = SEGMENT_OVERLAP,
                     strip_content: bool = False) -> None:
    """Run both passes for one note and write all artifacts.

    ``segment_size`` and ``overlap`` control the segmentation windows. Their
    defaults (3,500 and 400 codepoints) are the values used for the study; change
    them here or with the ``--segment-size`` / ``--overlap`` command-line flags.
    ``strip_content`` blanks the model text in the persisted raw responses.
    """
    # --- Pass 1: over the original note ---
    segments = fixed_segments(source, segment_size, overlap)
    pairs, records = _run_segments(client, segments)
    _persist_segments(model_dir, doc_id, 1, records, strip_content)

    pass1 = resolve_pairs(source, identity_map(source), merge_pairs(pairs))
    write_json(model_dir / "resolved" / doc_id / "resolved_predictions.json",
               resolved_predictions_document(doc_id, source, pass1))

    # --- Pass 2: over the note with Pass-1 regions blanked to [PHI] ---
    pass2_dir = model_dir / "pass_2"
    representation, coordinate_map = redacted_representation_with_map(source, union_regions(pass1))
    segments2 = fixed_segments(representation, segment_size, overlap)
    pairs2, records2 = _run_segments(client, segments2)
    _persist_segments(pass2_dir, doc_id, 2, records2, strip_content)

    # Apply new Pass-2 findings only when every Pass-2 segment finished cleanly.
    operational_complete = bool(records2) and all(
        r["finish_reason"] == "stop" and r["envelope_usable"] for r in records2)
    new_predictions = resolve_pairs(representation, coordinate_map, merge_pairs(pairs2))
    applied = new_predictions if operational_complete else []
    cumulative = merge_predictions(pass1, applied)
    write_json(pass2_dir / "resolved" / doc_id / "resolved_predictions.json",
               resolved_predictions_document(doc_id, source, cumulative))


def make_client(model_id: str, protocol_dir: Path, use_stub: bool,
                api_base: str, name_gazetteer: Optional[list[str]],
                request_timeout: Optional[float] = None):
    """Build the requested back-end: the llama-server client, or the test stub."""
    if use_stub:
        return StubExtractor(model_id=model_id, name_gazetteer=name_gazetteer)
    system_prompt, user_template, schema = load_protocol(protocol_dir)
    return LlamaServerClient(model_id, system_prompt, user_template, schema,
                             api_base=api_base, request_timeout=request_timeout)


def run_model(model_id: str, notes_dir: Path, out_dir: Path, protocol_dir: Path,
              use_stub: bool = False, api_base: str = "http://127.0.0.1:8081",
              name_gazetteer: Optional[list[str]] = None,
              segment_size: int = SEGMENT_SIZE, overlap: int = SEGMENT_OVERLAP,
              model_config: Optional[Path] = None, strip_content: bool = False,
              request_timeout: Optional[float] = None) -> Path:
    """Run ``model_id`` over every ``*.txt`` note in ``notes_dir``."""
    client = make_client(model_id, protocol_dir, use_stub, api_base, name_gazetteer, request_timeout)
    model_dir = out_dir / model_id
    notes = sorted(notes_dir.glob("*.txt"))
    if not notes:
        raise FileNotFoundError(f"no notes found in {notes_dir}")

    # Optional metadata: copy the model_config.json the scorer carries into results.
    if model_config is not None:
        if not model_config.exists():
            raise FileNotFoundError(f"model config not found: {model_config}")
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model_config.json").write_text(
            model_config.read_text(encoding="utf-8"), encoding="utf-8")

    for note_path in notes:
        doc_id = note_path.stem
        if not DOC_ID_PATTERN.match(doc_id):
            raise ValueError(f"unexpected note id: {doc_id!r}")
        source = note_path.read_text(encoding="utf-8")
        process_document(client, doc_id, source, model_dir, segment_size, overlap, strip_content)
        print(f"{model_id}: {doc_id} done ({len(source)} codepoints)", flush=True)

    return model_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one model over a set of notes (two passes).")
    parser.add_argument("--model-id", required=True, help="run label / model identifier")
    parser.add_argument("--notes-dir", type=Path, required=True, help="directory of <doc>.txt notes")
    parser.add_argument("--out-dir", type=Path, required=True, help="output root (per-model subdir created)")
    parser.add_argument("--protocol-dir", type=Path, default=Path("protocol"),
                        help="directory holding the frozen prompt + schema")
    parser.add_argument("--offline-stub", action="store_true",
                        help="use the offline deterministic test stub instead of llama-server")
    parser.add_argument("--api-base", default="http://127.0.0.1:8081",
                        help="llama-server base URL (ignored with --offline-stub)")
    parser.add_argument("--name-file", type=Path, default=None,
                        help="optional newline-separated name gazetteer for --offline-stub")
    parser.add_argument("--segment-size", type=int, default=SEGMENT_SIZE,
                        help=f"segmentation window width in codepoints (study default: {SEGMENT_SIZE})")
    parser.add_argument("--overlap", type=int, default=SEGMENT_OVERLAP,
                        help=f"overlap between windows in codepoints (study default: {SEGMENT_OVERLAP})")
    parser.add_argument("--model-config", type=Path, default=None,
                        help="model_config.json to copy into the output dir (metadata for the scorer)")
    parser.add_argument("--strip-raw-content", action="store_true",
                        help="blank the model text in raw_responses so out/ holds no PHI (metrics unchanged)")
    parser.add_argument("--request-timeout", type=float, default=None,
                        help="per-request timeout in seconds (default: none; set it to fail a hung server)")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    gazetteer = None
    if args.name_file and args.name_file.exists():
        gazetteer = [line.strip() for line in args.name_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    model_dir = run_model(
        args.model_id, args.notes_dir, args.out_dir, args.protocol_dir,
        use_stub=args.offline_stub, api_base=args.api_base, name_gazetteer=gazetteer,
        segment_size=args.segment_size, overlap=args.overlap,
        model_config=args.model_config, strip_content=args.strip_raw_content,
        request_timeout=args.request_timeout,
    )
    print(f"wrote artifacts under {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
