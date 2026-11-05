# Manuscript to repository map

This map relates the submitted manuscript, *Clinical Text De-identification with
Locally Deployed Open-Weight Large Language Models: A Real-World Evaluation of
Discharge Notes*, to the public reference implementation. The code is derived
from the study scripts; it is not the original execution archive. Study inputs
and retained outputs are restricted. See [Reproducibility](REPRODUCIBILITY.md)
for the distinction between the study record and the companion implementation.

## Methods

| Manuscript method | Implementation |
| --- | --- |
| Windows of at most 3,500 characters, overlapping by 400 | `pipeline.fixed_segments`, `SEGMENT_SIZE`, `SEGMENT_OVERLAP`. Offsets are Unicode codepoints. |
| Study-specific 18-category exact-text extraction | `protocol/system_prompt.txt`, `user_prompt_template.txt`, and `model_output.schema.json`; loaded by `run_model.load_protocol`. |
| Temperature 0, seed 42, reasoning effort low, JSON schema | `inference.build_payload`. These are request settings, not cross-runtime determinism guarantees. |
| Verbatim grounding and document-wide exact-string propagation | `pipeline.validate_response`, `merge_pairs`, and `resolve_pairs`; all exact case-sensitive substring occurrences are matched. |
| Two passes and original-coordinate mapping | `run_model.process_document`, `redacted_representation_with_map`, and `map_interval`. Matches touching inserted markers are discarded. |
| Pass 2 additions require all expected segments to stop and be usable | The in-memory gate in `run_model.process_document`; strict-schema validity is separate. |
| Cumulative redaction retains every Pass 1 finding | `merge_predictions`; optional text rendering uses `neutral_redaction` or `typed_redaction`. The public runner saves offsets, not final text files. |
| Every test note contributes to the accuracy denominator | `metrics.expected_documents` uses the reference directory; missing Pass 1 adds no detections and missing cumulative Pass 2 retains Pass 1. Notes are not omitted. |
| Source and prediction identity | `metrics.load_predictions` checks document IDs, source hashes, types, bounds, and artifact schema. |
| Matching reference sets for comparisons | Source, per-note annotation, and full reference-set fingerprints; analysis helpers reject mismatched identities and denominators. |
| Independent human-annotation agreement | `deid.agreement`; pooled character and whitespace-token metrics, exact-first greatest-overlap one-to-one entity pairs. Requires the two independent annotation sets, not the consensus alone. |
| Untuned segmented two-pass Presidio baseline | `deid.presidio_runner`; pinned optional dependencies and 17 default recognizers. Produces untyped cumulative masks without LLM literal propagation. |

The protocol files contain the extraction instructions and response constraints
described in the manuscript and reproduced in Supplementary Materials 1–3. The
example is embedded in the system prompt; no separate few-shot file is read.

## Outcomes

The formulas below are implemented by `metrics.score_document` and `emit_rows`.
The [data dictionary](../DATA_DICTIONARY.md) defines every output column.

| Manuscript outcome | Repository field or rule |
| --- | --- |
| Character recall | `char_recall = char_tp / gt_phi_chars`; type-agnostic coverage. |
| Character precision | `char_precision = char_tp / pred_phi_chars`. |
| Character F1 | `char_f1`, harmonic mean of character precision and recall. |
| Over-redaction | `over_redaction_rate = char_fp / pred_phi_chars`, not `FP/(FP+TN)`. |
| Nonreference-character removal | `nonreference_removal_rate = char_fp / (char_fp + char_tn)`; the additional Presidio measure with a different denominator. |
| Residual reference-identifier characters | `residual_phi_chars = char_fn`; `residual_phi_char_rate = char_fn / gt_phi_chars`. Not a re-identification risk estimate. |
| Any-overlap and full-span recall | `any_span_recall`, `full_span_recall`; at least one versus all reference-span characters covered. |
| Exact and boundary-tolerant matching | `strict_span_*`, `relaxed_span_*`; relaxed endpoints each differ by at most two codepoints (`REL_TOL = 2`). Matches are non-exclusive, not one-to-one assignment. |
| Reported boundary-tolerant span F1 | `relaxed_span_f1`. |
| Note-level completeness | `zero_residual_notes`, `zero_residual_note_rate`; no annotated identifier character remains. Report execution means descriptively, without completeness confidence intervals. Does not establish anonymity. |
| Added recall from Pass 2 | Cumulative `char_recall` minus Pass 1 `char_recall`, an absolute difference. |
| Processing time | `scripts/summarize_timing.py`; sum segment latencies across both passes per note, take the median over notes, then mean those medians over matched executions. Separately labeled Presidio pipeline time uses its own records. |
| Character-recall confidence intervals | `scripts/bootstrap_ci.py`: matched per-note character counts, across-execution averaging, then 10,000 note resamples with seed `20260801`. The current manuscript does not report completeness confidence intervals. |
| Presidio character precision/recall/F1 intervals | The same helper, one execution, `--character-metrics recall precision f1 --seed 20260721`; precision/F1 require raw false-positive counts. |

The manuscript's accuracy summaries use the `all_expected` cohort and
micro-averaging within each execution, followed by averaging across three locked
executions. The public runner performs one execution at a time. The `*_macro`
columns, `operational_complete` cohort, `finish_stop_rate`, `usable_rate`, and
`json_valid_rate` are additional diagnostics, not reported study outcomes.
Conditional Pass 2 format-validity rates use available validation records as
their denominator and do not establish request-level reliability. They were
removed from the current manuscript's outcome tables.

Presidio mask predictions use `UNSPECIFIED`, accepted only with comparator
provenance. Its gold-category recall remains type-agnostic, while type-matched
category metrics are blank. Human agreement uses one-to-one overlap pairing;
the model boundary metrics above use non-exclusive endpoint matching.

Current `main` saves expected-segment manifests before requests and verifies
record coverage before labeling a pass operationally complete. Legacy outputs
without plans have unknown expected counts and unverified coverage. Conditional
format rates remain conditional. This diagnostic change does not alter the
`all_expected` character/span accuracy formulas or the saved clinical results.

## Tables

| Manuscript table | Public script and limits |
| --- | --- |
| Table 1: corpus and reference standard | `scripts/make_corpus_table.py` summarizes the supplied reference directory. The clinical reference has 3,537 spans; the default synthetic one does not. The restricted source-corpus selection funnel is not rebuilt from these examples. |
| Table 2: checkpoints | `scripts/make_checkpoints_table.py`, using `model_configs/`; see [Models](MODELS.md). Missing original checkpoint/runtime provenance is not inferred. |
| Table 3: model results | `scripts/make_results_table.py` selects cumulative two-pass, all-expected, `ALL` rows and averages matched executions. Complete-note means are descriptive, without confidence limits. Character-recall intervals and timings require their separate inputs and procedures. |

## Figures

| Manuscript figure | Repository material |
| --- | --- |
| Figure 1: study workflow | `docs/images/methods-pipeline.png` is Panel B. Calibration-set selection (Panel A) is a study-design step, not performed by the runner. |
| Figure 2: character recall and complete notes | `char_recall`, `zero_residual_note_rate`, and mean `zero_residual_notes`; plotted CSV records the actual note denominator. |
| Figure 3: recall versus over-redaction | `char_recall`, `over_redaction_rate`. |
| Figure 4: coverage and boundaries | `any_span_recall`, `full_span_recall`, `relaxed_span_recall`, `strict_span_recall`. |
| Figure 5: second-pass gain | Pass 1 and cumulative Pass 2 `char_recall`. |
| Figure 6: category recall | Per-category `char_recall`, with the pooled `ALL` result; the overall value is not the average of the displayed categories. |
| Supplementary Figure S1 | `docs/images/segment-workflow.png`, the fixed synthetic workflow schematic. |
| Supplementary Figure S2C | `deid.agreement` calculates the described cohort-level metrics from two independent annotation directories. It does not contain the clinical inputs or recreate the submitted chart from synthetic data. |

`scripts/make_figures.py` draws its own SVG charts for the analyses behind Figures
2–6 and writes the plotted values as CSV. The committed example charts use one
model on three synthetic notes. Neither their values nor their appearance
reproduce the submitted clinical figures. Commands and examples are in
[Analysis scripts](ANALYSIS.md).

## Implementation limits

The agreement, Presidio, and timing helpers are new companion implementations
of the described methods, not recovered historical scripts or new clinical
results. Their synthetic tests establish behavior on those examples only.

The source-corpus selection, calibration decisions, independent annotation,
consensus/QC history, original study outputs, and exact historical environment
cannot be recovered from this repository. Supplementary Figures S2A/S2B require
the corresponding original annotation coordinates; a single-note example cannot
substitute for the 130-note agreement analysis. This repository does not include
those clinical inputs or reproduce the reported clinical numbers.
