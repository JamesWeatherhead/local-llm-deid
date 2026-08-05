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
| mean of three independent runs | this runner scores a single temperature-0 run |

## Methods

| Manuscript | Repository |
| --- | --- |
| 3,500-character segments, 400-character overlap | `SEGMENT_SIZE = 3500`, `SEGMENT_OVERLAP = 400` in `src/deid/pipeline.py` (`fixed_segments`) |
| extract `{exact_text, identifier_type}` over the 18 Safe Harbor categories | `pipeline.validate_response`; the schema is `protocol/model_output.schema.json` |
| verbatim grounding, every occurrence, no normalization | `pipeline.resolve_pairs` (kept only if `exact_text` occurs verbatim; then located at all occurrences) |
| local inference: temperature 0, seed 42, constrained JSON | `TEMPERATURE = 0`, `SEED = 42`, `response_format` json_schema in `src/deid/inference.py` (`build_payload`) |
| two passes; every Pass-1 redaction retained; Pass-2 union | `run_model.process_document`; cumulative = union(Pass 1, applied Pass 2); a Pass-2 match crossing a `[PHI]` marker is dropped |
| Apple M4 Max, 48 GB, Metal; mean of three runs | recorded as provenance in the README; this runner scores a single run |

The repository additionally always sends `reasoning_effort = "low"`
(`inference.py`); the manuscript does not call this parameter out separately.

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
| 95% CI for PHI removed (recall) | reproduced by `scripts/bootstrap_ci.py` (not a scorer column) | note-level bootstrap over `per_doc_long.csv`: 10,000 resamples with replacement, seed 20260801, 2.5th and 97.5th percentiles (linear) |

Micro-averaging (sum counts across notes, then divide) is the manuscript default
and the default here. The `*_macro` columns (mean of per-note rates), the
`operational_complete` cohort, and `finish_stop_rate` are diagnostics the
manuscript does not report. The results table's 95% confidence interval for PHI
removed is a note-level bootstrap: the scorer emits point estimates only, and
`scripts/bootstrap_ci.py` reproduces the interval from `per_doc_long.csv`
(10,000 resamples, seed 20260801).

## Tables

| Manuscript | Repository |
| --- | --- |
| Table 1 `tab:corpus` (corpus and reference standard; 3,537 spans) | rebuilt by `scripts/make_corpus_table.py` from a gold directory (the synthetic fixture by default); the per-category spans also map to the per-type rows of `metrics_long.csv` (`phi_type` other than `ALL`) |
| Table 2 `tab:model-checkpoints` (checkpoints: params, quant, size, repository) | rebuilt by `scripts/make_checkpoints_table.py` from `model_configs/build_configs.py`; also the Models table in `README.md` |
| Table 3 `tab:results` (per-model results after both passes) | `metrics_long.csv` rows with `pass=cumulative_pass2`, `cohort=all_expected`, `phi_type=ALL`; ranked by `scripts/make_results_table.py` |

Table 3 column by column: PHI removed is `char_recall`, redactions within PHI is
`char_precision`, Character F1 is `char_f1`, Added by Pass 2 is `char_recall` at
`cumulative_pass2` minus at `pass1`, Span F1 is `relaxed_span_f1`, notes with all
PHI removed is `zero_residual_notes`, and usable structured response is
`usable_rate`. Seconds per note is not produced by this scorer; the 95% CI for
PHI removed is reproduced separately by `scripts/bootstrap_ci.py` (note-level
bootstrap, seed 20260801).

## Figures

`scripts/make_figures.py` reads `out/metrics_long.csv` and writes one
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
| Figure 1 `fig:methods-pipeline` (study workflow) | `docs/images/methods-pipeline.png` shows Panel B of this figure (the evaluation workflow) in the README; Panel A (development-set protocol selection) is a study-design step and is omitted. Implemented by `pipeline.py`, `run_model.py`, and `protocol/` |
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
| Reasoning-recurrence ("looping") analysis (Supplementary Figure S3 and Box S1) | Not reproduced. It is derived from raw model reasoning traces, which remain inside the IRB environment; the runner discards `reasoning_content` rather than retaining it. |
| Microsoft Presidio baseline | Intentionally not reproduced (see the Figures section above): a separate third-party system, not part of this pipeline. |
| Median seconds per note | Not aggregated by the scorer (see Metric definitions above); per-request `latency_seconds` is recorded in `raw_responses/*.raw.json`. |
| Mean of three independent runs | This runner scores a single temperature-0 run (see Terminology and Methods above). |

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
table (parameters, quantization, on-disk size, repository) for all 17 models, in
the results-table (recall) order. Two cells were corrected in the repository to
match the manuscript exactly:

| Model | Field | Was | Now (matches manuscript) |
| --- | --- | --- | --- |
| Gemma-4 12B | parameters | 11.95 | 12.0 |
| Gemma-4 26B-A4B | quantization | Q5_K_M | UD-Q5_K_M |

Three repository cells intentionally differ from the manuscript checkpoint table,
because the manuscript cites a source that is not a usable GGUF download while the
repository lists the resolving GGUF repository that provides the evaluated quant:

| Model | Manuscript cell | Repository (kept) | Reason |
| --- | --- | --- | --- |
| Ministral-3 14B | `mistralai/Ministral-3-14B-Instruct-2512` | `unsloth/Ministral-3-14B-Instruct-2512-GGUF` | the manuscript id hosts safetensors only; the evaluated Q6_K checkpoint is a GGUF from the unsloth mirror |
| Granite-4.1 30B | `ibm-granite/granite-4.1-30b-instruct-GGUF` | `ibm-granite/granite-4.1-30b-GGUF` | the repository id resolves and hosts the Q5_K_M quant matching the reported 20.49 GB |
| Granite-4.1 8B | `ibm-granite/granite-4.1-8b-instruct-GGUF` | `ibm-granite/granite-4.1-8b-GGUF` | the repository id resolves and hosts the Q6_K quant matching the reported 7.22 GB |

These are the only differences between the repository's model metadata and the
manuscript checkpoint table.
