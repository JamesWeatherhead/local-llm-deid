"""Reference implementation of the manuscript's segmented Presidio comparator.

This is a new companion implementation, not the original execution script.
It writes type-agnostic source-offset masks and full-pipeline timing, without
source prose or detected strings. It does not apply the LLM literal-propagation
rule, or invent LLM finish/JSON-validation records for Presidio.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .pipeline import (
    Prediction, SEGMENT_OVERLAP, SEGMENT_SIZE, fixed_segments, identity_map,
    map_interval, redacted_representation_with_map, resolved_predictions_document,
    sha256_text, union_regions,
)
from .provenance import file_sha256, git_revision, segment_manifest
from .run_model import DOC_ID_PATTERN, MODEL_ID_PATTERN, write_json

UNTYPED = "UNSPECIFIED"
STUDY_PACKAGES = {"presidio-analyzer": "2.2.364", "spacy": "3.8.14",
                  "en-core-web-lg": "3.8.0"}


def build_analyzer():
    """Load the pinned local spaCy model and the default English recognizers.

    Dependencies and the language model must already be installed. This command
    performs no automatic package/model download and adds no recognizers,
    thresholds, entities filter, allowlist or retry policy.
    """
    for package, expected in STUDY_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"install the optional comparator dependency: {package}=={expected}") from exc
        if actual != expected:
            raise RuntimeError(f"{package} must be {expected}; installed: {actual}")
    if importlib.util.find_spec("en_core_web_lg") is None:
        raise RuntimeError("install en_core_web_lg 3.8.0 locally before running the comparator")
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    })
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
    # A changed registry is not silently represented as the manuscript's setup.
    recognizers = analyzer.registry.recognizers
    if len(recognizers) != 17:
        raise RuntimeError(f"expected 17 default recognizers, found {len(recognizers)}")
    return analyzer


def _merged_mask(predictions: list[Prediction]) -> list[Prediction]:
    """Union duplicate/overlapping intervals; categories are not inferred."""
    return [Prediction(start, end, UNTYPED) for start, end in union_regions(predictions)]


def _detect(analyzer, representation: str, coordinate_map: list[Optional[int]],
            segment_size: int, overlap: int) -> list[Prediction]:
    predictions: list[Prediction] = []
    for segment in fixed_segments(representation, segment_size, overlap):
        # Deliberately no entities, score_threshold, allow_list or custom args.
        results = analyzer.analyze(text=segment.text, language="en")
        if not isinstance(results, (list, tuple)):
            raise ValueError("Presidio must return a list of offset detections")
        for result in results:
            start, end = getattr(result, "start", None), getattr(result, "end", None)
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(segment.text):
                raise ValueError("Presidio returned invalid segment offsets")
            mapped = map_interval(coordinate_map, segment.start + start, segment.start + end)
            if mapped is not None:
                predictions.append(Prediction(mapped[0], mapped[1], UNTYPED))
    return _merged_mask(predictions)


def process_document(analyzer, doc_id: str, source: str, model_dir: Path,
                     segment_size: int = SEGMENT_SIZE, overlap: int = SEGMENT_OVERLAP,
                     clock=None) -> dict[str, Any]:
    """Analyze/redact/resegment/analyze one note; save cumulative source masks.

    Analyzer exceptions abort this note rather than manufacturing a completed
    comparator result. Timing covers both passes, segmentation, coordinate
    mapping and redaction; disk writes and model loading are excluded.
    """
    if not DOC_ID_PATTERN.fullmatch(doc_id):
        raise ValueError("unsafe document ID")
    timer = clock or time.perf_counter
    started = timer()
    segments1 = fixed_segments(source, segment_size, overlap)
    plan1 = segment_manifest(doc_id, source, source, 1, segments1, segment_size, overlap)
    pass1 = _detect(analyzer, source, identity_map(source), segment_size, overlap)
    representation, mapping = redacted_representation_with_map(source, union_regions(pass1))
    segments2 = fixed_segments(representation, segment_size, overlap)
    plan2 = segment_manifest(doc_id, source, representation, 2, segments2, segment_size, overlap)
    new = _detect(analyzer, representation, mapping, segment_size, overlap)
    cumulative = _merged_mask(pass1 + new)
    elapsed = timer() - started
    if elapsed < 0:
        raise ValueError("nonmonotonic timer")
    base2 = model_dir / "pass_2"
    write_json(model_dir / "expected_segments" / doc_id / "manifest.json", plan1)
    write_json(base2 / "expected_segments" / doc_id / "manifest.json", plan2)
    write_json(model_dir / "resolved" / doc_id / "resolved_predictions.json",
               resolved_predictions_document(doc_id, source, pass1))
    write_json(base2 / "resolved" / doc_id / "resolved_predictions.json",
               resolved_predictions_document(doc_id, source, cumulative))
    timing = {"document_id": doc_id, "source_sha256": sha256_text(source),
              "pipeline_seconds": elapsed, "completed": True,
              "expected_segments_pass_1": len(segments1),
              "expected_segments_pass_2": len(segments2)}
    write_json(model_dir / "pipeline_timing" / f"{doc_id}.json", timing)
    return timing


def run_comparator(notes_dir: Path, out_dir: Path, model_id: str = "presidio", *,
                   analyzer=None, segment_size: int = SEGMENT_SIZE,
                   overlap: int = SEGMENT_OVERLAP,
                   overwrite_model_output: bool = False) -> Path:
    """Create a fresh comparator output tree, compatible with deid-score."""
    if not MODEL_ID_PATTERN.fullmatch(model_id):
        raise ValueError("model ID must be one safe path component")
    fixed_segments("", segment_size, overlap)
    notes = sorted(notes_dir.glob("*.txt"))
    if not notes:
        raise FileNotFoundError(f"no notes found in {notes_dir}")
    if any(not DOC_ID_PATTERN.fullmatch(note.stem) for note in notes):
        raise ValueError("unsafe document ID")
    # Read inputs and load dependencies before a caller-authorized overwrite.
    sources = [(note.stem, note.read_text(encoding="utf-8")) for note in notes]
    injected = analyzer is not None
    engine = analyzer if injected else build_analyzer()
    if out_dir.is_symlink():
        raise ValueError("refusing symlinked output root")
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    out_dir.chmod(0o700)
    model_dir = out_dir / model_id
    if model_dir.is_symlink():
        raise ValueError("refusing symlinked model output")
    if model_dir.exists():
        if not overwrite_model_output:
            raise FileExistsError(f"model output already exists: {model_dir}; choose a fresh directory")
        if not model_dir.is_dir():
            raise ValueError("refusing to overwrite a non-directory")
        shutil.rmtree(model_dir)
    model_dir.mkdir(mode=0o700)
    package = Path(__file__).resolve().parent
    manifest = {
        "schema_version": "comparator-manifest-v1", "backend": "presidio",
        "model_id": model_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "companion_implementation": True, "original_execution_code": False,
        "python_version": platform.python_version(),
        "repository_commit": git_revision(package.parents[1]),
        "code_sha256": {path.name: file_sha256(path) for path in sorted(package.glob("*.py"))},
        "packages": STUDY_PACKAGES if not injected else None,
        "injected_test_analyzer": injected,
        "recognizers": sorted(r.name for r in engine.registry.recognizers) if not injected else None,
        "segmentation": {"size": segment_size, "overlap": overlap},
        "prediction_type": UNTYPED,
        "prediction_type_meaning": "type-agnostic merged comparator intervals; no Safe Harbor type inferred",
        "literal_propagation": False, "passes": 2,
        "timing_method": "full two-pass pipeline; excludes dependency/model loading and artifact writes",
        "notes": [{"document_id": doc_id, "source_sha256": sha256_text(source)}
                  for doc_id, source in sources],
    }
    write_json(model_dir / "comparator_manifest.json", manifest)
    write_json(model_dir / "model_config.json", {"model_id": model_id, "family": "Presidio",
                                               "backend": "presidio"})
    for doc_id, source in sources:
        process_document(engine, doc_id, source, model_dir, segment_size, overlap)
    return model_dir


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="presidio")
    parser.add_argument("--segment-size", type=int, default=SEGMENT_SIZE)
    parser.add_argument("--overlap", type=int, default=SEGMENT_OVERLAP)
    parser.add_argument("--overwrite-model-output", action="store_true")
    args = parser.parse_args(argv)
    output = run_comparator(args.notes_dir, args.out_dir, args.model_id,
                            segment_size=args.segment_size, overlap=args.overlap,
                            overwrite_model_output=args.overwrite_model_output)
    print(f"wrote comparator artifacts under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
