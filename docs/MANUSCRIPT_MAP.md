# Manuscript to repository map

This companion repository implements the computational methods of the manuscript
*Clinical Text De-identification with Locally Deployed Open-Weight Large Language
Models: A Real-World Evaluation of Discharge Notes*. This file maps what the
manuscript describes to where it lives in the code, so a reader can check any
definition, number, table, or figure against the implementation.

The manuscript is the fixed reference; nothing here changes it. Where the
repository and the manuscript differ, the difference is stated below.

## Terminology

| Manuscript | Repository |
| --- | --- |
| character (unit of offset) | Unicode codepoint, half-open `[start, end)`; the two coincide for this corpus |
| PHI removed | `char_recall` (`char_tp / gt_phi_chars`) |
| redactions within PHI | `char_precision` (`char_tp / pred_phi_chars`) |
| removed characters outside PHI | `over_redaction_rate` (`char_fp / pred_phi_chars`) |
| within 2 characters / boundary-tolerant | `relaxed_span_*` (both boundaries within 2 codepoints; `REL_TOL = 2`) |
| exact boundaries | `strict_span_*` |
| usable structured response | `usable_rate` |
| strict JSON valid | `json_valid_rate` |
| all annotated PHI removed (per note) | `zero_residual_note_rate` |
| means across three locked stability executions | the runner scores one execution at a time; `bootstrap_ci.py` accepts all three matched per-note outputs and averages them by note before resampling |

## Methods

| Manuscript | Repository |
| --- | --- |
| 3,500-character segments, 400-character overlap | `SEGMENT_SIZE = 3500`, `SEGMENT_OVERLAP = 400` in `src/deid/pipeline.py` (`fixed_segments`) |
| extract `{exact_text, identifier_type}` over the 18 Safe Harbor categories | `pipeline.validate_response`; the schema is `protocol/model_output.schema.json` |
| verbatim grounding, every occurrence, no normalization | `pipeline.resolve_pairs` (kept only if `exact_text` occurs verbatim; then located at all occurrences) |
| local inference: temperature 0, seed 42, constrained JSON | `TEMPERATURE = 0`, `SEED = 42`, `response_format` json_schema in `src/deid/inference.py` (`build_payload`) |
| two passes; every Pass-1 redaction retained; Pass-2 union | `run_model.process_document`; cumulative = union(Pass 1, applied Pass 2); a Pass-2 match crossing a `[PHI]` marker is dropped; Pass-2 additions require every segment to finish with `stop` and be usable |
| Apple M4 Max, 48 GB, Metal; means across three locked executions | recorded as provenance in the README; the runner scores one execution at a time and the bootstrap helper reconciles all three |
| every gold note remains in the denominator | `metrics.expected_documents`; missing output is scored as an empty prediction set, while output for an unknown document is rejected |
| scored artifacts belong to the expected source note | `metrics.load_predictions`; schema version, document ID, source SHA-256, offset unit, type vocabulary, duplicates, and bounds are validated before scoring |

Both the manuscript and repository specify `reasoning_effort = "low"`
(`REASONING_EFFORT` and `build_payload` in `inference.py`).

## Metric definitions

Every column is defined in [../DATA_DICTIONARY.md](../DATA_DICTIONARY.md); the
formulas live in `src/deid/metrics.py` (`score_document`, `emit_rows`).

| Manuscript measure | Column in `metrics_long.csv` | Formula |
| --- | --- | --- |
| PHI removed (character recall) | `char_recall` | `char_tp / gt_phi_chars` |
| residual PHI (character) | `residual_phi_char_rate` | `char_fn / gt_phi_chars` (1 minus recall) |
| redactions within PHI (precision) | `char_precision` | `char_tp / pred_phi_chars` |
| character F1 | `char_f1` | harmonic mean of the two |
| removed characters outside PHI | `over_redaction_rate` | `char_fp / pred_phi_chars` |
| any / full / exact / boundary-tolerant span | `any_span_recall` / `full_span_recall` / `strict_span_recall` / `relaxed_span_recall` | see `score_document`; boundary-tolerant is within 2 codepoints on both ends |
| reported Span F1 | `relaxed_span_f1` | harmonic mean of relaxed precision and recall |
| notes with all annotated PHI removed | `zero_residual_notes`, `zero_residual_note_rate` | notes with `residual == 0` |
| usable structured response | `usable_rate` | usable / validated second-pass requests |
| strict JSON valid | `json_valid_rate` | valid / validated requests |
| seconds per note | not aggregated by the scorer | per-request `latency_seconds` is recorded in `raw_responses/*.raw.json` |
| 95% CIs for recall and complete-note rate | reproduced by `scripts/bootstrap_ci.py` (not scorer columns) | verify matched keys and invariant gold counts across the three execution files; average stored values by note; 10,000 note-level resamples with replacement, seed 20260801; 2.5th and 97.5th percentiles (linear) |

Micro-averaging (sum counts across notes, then divide) is the manuscript default
and the default here. The `*_macro` columns (mean of per-note rates), the
`operational_complete` cohort, and `finish_stop_rate` are diagnostics the
manuscript does not report. The scorer emits point estimates for one execution.
`scripts/bootstrap_ci.py` reproduces the manuscript's recall and completeness
interval preparation from three repeated `--per-doc` inputs by averaging matched
per-note values before the note-level bootstrap (10,000 resamples, seed 20260801).

## Tables

| Manuscript | Repository |
| --- | --- |
| Table 1 `tab:corpus` (corpus and reference standard; 3,537 spans) | rebuilt by `scripts/make_corpus_table.py` from a gold directory (the synthetic fixture by default); the per-category spans also map to the per-type rows of `metrics_long.csv` (`phi_type` other than `ALL`) |
| Table 2 `tab:model-checkpoints` (checkpoints: params, quant, size, repository) | rebuilt by `scripts/make_checkpoints_table.py` from `model_configs/build_configs.py`; also the Models table in `README.md` |
| Table 3 `tab:results` (per-model results after both passes) | scorer-derived fields come from three matched `metrics_long.csv` files with `pass=cumulative_pass2`, `cohort=all_expected`, `phi_type=ALL`; `scripts/make_results_table.py` key-matches and averages the executions before ranking; published CIs require the three restricted `per_doc_long.csv` files, and the timing column requires retained timing inputs |

Table 3 column by column: PHI removed is `char_recall`, redactions within PHI is
`char_precision`, Character F1 is `char_f1`, Added by Pass 2 is `char_recall` at
`cumulative_pass2` minus at `pass1`, Span F1 is `relaxed_span_f1`, notes with all
PHI removed is `zero_residual_notes`, and usable structured response is
`usable_rate`. Seconds per note is not produced by this scorer; the 95% CI for
PHI removed is reproduced separately by `scripts/bootstrap_ci.py` (note-level
bootstrap, seed 20260801).

## Figures

`scripts/make_figures.py` reads one or more matched `metrics_long.csv` files, averages numeric outcomes across supplied executions, and writes one
self-contained SVG plus a CSV of the plotted values per figure
(`out/figures/figure2..6.svg` and `.csv`). These SVGs are the repository's own
renders of the analyses the manuscript presents as its Figures 2 to 6: the same
metric definitions and the same view, in a simple built-in chart style. They are
**not** the manuscript's published figures. The example copies under
`docs/images/figures/` were generated from the Gemma 3 1B run on the synthetic
fixture (a single model), whereas the manuscript's figures span all seventeen
models on the real corpus and look different. The table below maps each manuscript
figure to the columns its repository render draws. Figure 1 and Supplementary
Figure S1 are fixed schematics shipped as PNGs under `docs/images/`.

| Manuscript figure | Repository |
| --- | --- |
| Figure 1 `fig:methods-pipeline` (study workflow) | `docs/images/methods-pipeline.png` shows Panel B of this figure (the evaluation workflow) in the README; Panel A (calibration-set protocol selection) is a study-design step and is omitted. Implemented by `pipeline.py`, `run_model.py`, and `protocol/` |
| Figure 2 `fig:primary-outcomes` (PHI removal and complete-note removal) | `char_recall` (Panel A) and `zero_residual_note_rate` (Panel B) |
| Figure 3 `fig:safety-utility` (removal vs non-PHI removal) | `char_recall` against `over_redaction_rate` |
| Figure 4 `fig:boundary-agreement` (coverage and boundaries) | `any_span_recall`, `full_span_recall`, `relaxed_span_recall`, `strict_span_recall` |
| Figure 5 `fig:two-pass` (second-pass gain) | `char_recall` at `pass1` vs `cumulative_pass2` |
| Figure 6 `fig:categories` (per-category removal) | per-type `char_recall` rows (`03_DATE`, `01_NAME`, `05_TELEPHONE_NUMBER`, `02_GEOGRAPHIC_SUBDIVISION`, plus the `ALL` row for Overall) |
| Supplementary Figure S1 `sfig:s1` (segment-level workflow) | `docs/images/segment-workflow.png`, shown in the README under "How the pipeline works": overlapping segmentation (Panel A), local-model extraction (Panel B), and reassembly with the second pass (Panel C). Implemented by `pipeline.fixed_segments`, `pipeline.resolve_pairs` / `neutral_redaction`, and `run_model.process_document` |

Where the manuscript reports a Microsoft Presidio baseline alongside the models,
that baseline is intentionally not reproduced here: it is a separate third-party
system, not part of this pipeline, and `scripts/make_figures.py` plots only the
local-model results the scorer emits.

## Manuscript analyses outside this repository

A few analyses in the manuscript and its supplement are reported for context but
are not reproduced by this code. Each depends on inputs that are not distributed
here, or on a separate third-party system:

| Manuscript analysis | Status in this repository |
| --- | --- |
| Inter-annotator agreement: Cohen's kappa, character Dice, whitespace-token F1, and exact and overlap entity F1 (Supplementary Figures S2A to S2C) | Not reproduced. It is computed from the two reviewers' independent, pre-reconciliation annotations, which are PHI-bearing and are not distributed. |
| Microsoft Presidio baseline | Intentionally not reproduced (see the Figures section above): a separate third-party system, not part of this pipeline. |
| Median seconds per note | Not aggregated by the scorer (see Metric definitions above); per-request `latency_seconds` is recorded in `raw_responses/*.raw.json`. |
| Means across three locked stability executions | The runner scores one temperature-0 execution at a time; the bootstrap helper accepts and reconciles three matched execution files, but the restricted real outputs are not distributed. |

The repository's scope is the computational core behind the manuscript's primary
results: segmentation, constrained-JSON extraction, verbatim grounding, two-pass
redaction, and character- and span-level scoring.

## Protocol and supplement

`protocol/system_prompt.txt`, `protocol/user_prompt_template.txt`, and
`protocol/model_output.schema.json` are byte-for-byte identical to the
manuscript's supplementary protocol, and are the complete set of inputs the
pipeline loads (`run_model.load_protocol`). The system prompt embeds its own
worked synthetic example; the runner reads no separate few-shot file.

## Model roster reconciliation

`model_configs/` and the README Models table carry the manuscript checkpoint
table (parameters, quantization, on-disk size, and resolving GGUF repository)
for all 17 models, in results-table order. The generated table now matches the
final manuscript, including:

* Gemma-4 E4B as `8.0/4.5` total/effective parameters;
* Gemma-4 E2B as `5.1/2.3` total/effective parameters and repository
  `ggml-org/gemma-4-E2B-it-GGUF`;
* Gemma-4 26B-A4B as `25.2/3.8` total/active parameters; and
* the resolving GGUF repositories for Ministral-3 and both Granite-4.1 models.

For Gemma-4 E2B, the configuration also records the manuscript's exact BF16
filename and SHA-256 digest. Exact checkpoint-byte digests for the other 16
models and the exact `llama.cpp` source revision/build manifest are not present
in the public snapshot; the README identifies these as remaining provenance
limitations rather than implying byte-for-byte reproduction.
