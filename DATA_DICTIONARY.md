# Data dictionary

The scorer (`deid.metrics`) writes four files into its output directory. All are
PHI-free: they contain offsets, counts, and rates only, never identifier text.

- `metrics_long.csv` — the main results table (one row per model × pass × cohort × PHI type)
- `per_doc_long.csv` — per-note character counts and reliability (one row per model × note × pass)
- `model_manifest.csv` — the metadata carried for each model, plus the gold types it was scored against
- `run_manifest.json` — a description of the scoring run (definitions, coordinate system, counts)

Coordinate system throughout: Unicode codepoints, half-open `[start, end)`, with
the gold `text` as the reference. An empty cell means the value is undefined (a
zero denominator) or not applicable to that row.

## Correspondence to the manuscript

The manuscript and this scorer use the same definitions; some names differ, and
the scorer also emits diagnostic columns the manuscript does not report. A full
crosswalk (tables, figures, and methods) is in
[docs/MANUSCRIPT_MAP.md](docs/MANUSCRIPT_MAP.md). In brief:

- The manuscript's "character" is a Unicode codepoint here; the two coincide for
  this corpus.
- `char_recall` is the manuscript's headline "PHI removed"; `char_precision` is
  "redactions within PHI"; `over_redaction_rate` is the share of removed
  characters outside PHI shown on the horizontal axis of the safety-utility figure.
- `any_span_recall`, `full_span_recall`, `relaxed_span_recall`, and
  `strict_span_recall` are the manuscript's any-overlap, full-coverage,
  "within 2 characters" (boundary-tolerant), and exact-boundary criteria; the
  reported "Span F1" is `relaxed_span_f1`.
- `usable_rate` is the manuscript's "usable structured response"; `json_valid_rate`
  is its strict-JSON-valid measure.
- `zero_residual_note_rate` is the manuscript's complete-removal-per-note measure.
- The manuscript reports the `all_expected` cohort with micro-averaging and the
  mean of three temperature-0 runs. The `operational_complete` cohort, every
  `*_macro` column, and `finish_stop_rate` are diagnostics it does not report, and
  this scorer scores a single run.
- The results table's 95% confidence interval for PHI removed is a note-level
  bootstrap over `per_doc_long.csv`, reproduced by `scripts/bootstrap_ci.py`
  (10,000 resamples, seed 20260801); the scorer itself emits point estimates only.

Per-column definitions follow.

## `metrics_long.csv` (69 columns)

Each model emits one `ALL` row per (pass, cohort) carrying every column, plus one
partial row per gold PHI type present. On the per-type rows only recall,
coverage, type-matched precision, and span recall are populated; note-level,
reliability, macro, and type-agnostic precision columns are blank because a false
positive cannot be attributed to a single gold type.

### Model identity and metadata

Sourced from an optional `model_config.json` beside the model's outputs; blank if
that file is absent (as in the synthetic fixture run).

| Column | Meaning |
| --- | --- |
| `model_id` | Model run label (the directory name under the predictions root). |
| `hf_link` | Hugging Face URL for the weights (derived from `hf_gguf_repo`). |
| `hf_gguf_repo` | Hugging Face repository id of the quantized GGUF weights. |
| `developer` | Organization that trained the base model. |
| `family` | Model family. |
| `params_total_B` | Total parameters, in billions. |
| `params_active_B` | Active parameters per token, in billions (mixture-of-experts only). |
| `moe` | Whether the model is mixture-of-experts. |
| `is_medical` | Whether the model is domain-adapted for medicine. |
| `is_reasoning` | Whether the model is a dedicated reasoning model. |
| `quant` | Quantization of the evaluated GGUF (e.g. `Q5_K_M`). |
| `n_docs` | Number of notes this model produced Pass-1 output for. |
| `doc_provenance` | Reserved provenance label (blank in this release). |

### Row keys

| Column | Meaning |
| --- | --- |
| `pass` | `pass1` or `cumulative_pass2` (Pass 1 ∪ the applied Pass 2). |
| `surface` | Redaction surface scored; always `final`. |
| `cohort` | `all_expected` (every note; a missing output is a full miss) or `operational_complete` (only notes whose every segment finished cleanly and, where validated, was usable). |
| `phi_type` | `ALL` for the note-wide row, or a two-digit Safe Harbor code (`01_NAME`, `02_GEOGRAPHIC_SUBDIVISION`, …) for a per-category row. |

### Totals

| Column | Meaning |
| --- | --- |
| `n_notes` | Notes included in this (pass, cohort). |
| `doc_chars_total` | Total characters across the included notes. |
| `gt_phi_chars` | Gold PHI characters. |
| `pred_phi_chars` | Predicted (redacted) characters. |
| `gt_span_count` | Gold PHI spans. |
| `pred_span_count` | Predicted spans. |

### Character level — type-agnostic (headline)

A gold PHI character counts as covered if it falls under *any* redaction.

| Column | Meaning |
| --- | --- |
| `char_tp` | Gold PHI chars covered by any redaction. |
| `char_fp` | Redacted chars that are not gold PHI (over-redaction). |
| `char_fn` | Gold PHI chars left unredacted (equals `residual_phi_chars`). |
| `char_tn` | Non-PHI chars correctly left intact. |
| `char_recall` | `char_tp / gt_phi_chars` — the headline PHI-removal metric. |
| `char_precision` | `char_tp / pred_phi_chars`. |
| `char_f1` | Harmonic mean of `char_recall` and `char_precision`. |
| `char_specificity` | `char_tn / (char_tn + char_fp)`. |
| `over_redaction_rate` | `char_fp / pred_phi_chars` (i.e. `1 − char_precision`). |
| `residual_phi_chars` | Gold PHI chars still present after redaction (equals `char_fn`). |
| `residual_phi_char_rate` | `residual_phi_chars / gt_phi_chars` (i.e. `1 − char_recall`). |

### Character level — type-matched (per-category)

A gold character counts only when covered by a redaction of the *same* type. On
the `ALL` row this trio mirrors the type-agnostic trio; the real per-category
signal lives on the per-type rows.

| Column | Meaning |
| --- | --- |
| `char_recall_typematched` | Gold chars covered by a same-type redaction / gold chars. |
| `char_precision_typematched` | Same-type-covered gold chars / chars predicted as this type. |
| `char_f1_typematched` | Harmonic mean of the two. |

### Span / entity level

Four detection criteria per gold span: any overlap, full coverage, strict
(exact boundary) match, and relaxed (both boundaries within ±2 codepoints,
i2b2-style).

| Column | Meaning |
| --- | --- |
| `any_span_tp` | Gold spans with ≥1 character covered. |
| `any_span_recall` | `any_span_tp / gt_span_count`. |
| `full_span_tp` | Gold spans fully covered. |
| `full_span_recall` | `full_span_tp / gt_span_count`. |
| `strict_span_tp` | Predicted spans matching a gold span exactly on both boundaries. |
| `strict_span_recall` | `strict_span_tp / gt_span_count`. |
| `strict_span_precision` | `strict_span_tp / pred_span_count`. |
| `strict_span_f1` | Harmonic mean of strict precision and recall. |
| `relaxed_span_tp` | Gold spans matched within ±2 codepoints on both boundaries. |
| `relaxed_span_recall` | `relaxed_span_tp / gt_span_count`. |
| `relaxed_span_precision` | Predicted spans within ±2 of some gold span / `pred_span_count`. |
| `relaxed_span_f1` | Harmonic mean of relaxed precision and recall. |

### Note level (`ALL` row only)

| Column | Meaning |
| --- | --- |
| `zero_residual_notes` | Notes with no residual PHI after redaction (complete removal). |
| `zero_residual_note_rate` | `zero_residual_notes / n_notes`. |
| `notes_with_leak` | Notes with ≥1 residual PHI character. |
| `notes_with_leak_rate` | `notes_with_leak / n_notes`. |
| `mean_residual_per_note` | Mean residual PHI chars per note. |
| `median_residual_per_note` | Median residual PHI chars per note. |
| `max_residual_per_note` | Worst-case residual PHI chars in a single note. |

### Reliability and cost (`ALL` row only)

| Column | Meaning |
| --- | --- |
| `expected_requests` | Total segment requests issued (sum of segments over notes). |
| `usable_requests` | Requests whose response envelope was usable (validated subset). |
| `usable_rate` | `usable_requests /` validated requests. |
| `json_valid_requests` | Requests whose content passed strict schema validation. |
| `json_valid_rate` | `json_valid_requests /` validated requests. |
| `finish_stop_rate` | Requests finishing with `finish_reason == "stop"` / `expected_requests`. |
| `prompt_tokens` | Total prompt tokens across requests. |
| `completion_tokens` | Total completion tokens across requests. |

### Macro averages (`ALL` row only)

Micro-averaging (sum counts, then take the rate) is the default everywhere above.
These columns give the macro alternative: the mean of the per-note rates.

| Column | Meaning |
| --- | --- |
| `char_recall_macro` | Mean of per-note character recall. |
| `char_precision_macro` | Mean of per-note character precision. |
| `char_f1_macro` | Mean of per-note character F1. |
| `strict_span_recall_macro` | Mean of per-note strict span recall. |
| `relaxed_span_recall_macro` | Mean of per-note relaxed span recall. |

## `per_doc_long.csv`

One row per model × note × pass.

| Column | Meaning |
| --- | --- |
| `model_id`, `n_docs` | As above. |
| `doc` | Note id. |
| `pass_` | `pass1` or `cumulative_pass2`. |
| `doc_chars` | Note length in characters. |
| `gt_phi_chars`, `pred_phi_chars` | Gold and predicted character counts for the note. |
| `char_tp`, `char_fp`, `char_fn` | Per-note character confusion counts. |
| `residual_phi_chars` | Gold PHI chars left unredacted in this note. |
| `zero_residual` | `1` if the note has no residual PHI, else `0`. |
| `gt_spans`, `any_span_hits` | Gold spans and how many had ≥1 char covered. |
| `n_segments`, `usable_segments`, `finish_stop` | Per-note reliability counts. |
| `operational_complete` | `1` if every segment finished cleanly (drives the cohort split). |

## `model_manifest.csv`

The metadata block for each model (the identity columns above) plus
`gold_types_present` (the Safe Harbor codes the model was scored against, joined
by `;`) and `model_dir` (its output subdirectory).

## `run_manifest.json`

A machine-readable description of the run: the gold source, the coordinate unit,
the recall and span-recall definitions, the relaxed tolerance, the averaging
convention, the list of models scored, and the row/column counts of the tables.
