# Reproducibility and run provenance

## Study software and public code

This companion implements the methods described in the manuscript. It is derived
from the study software, not a byte-for-byte archive of the original execution
software. Its agreement, Presidio, and timing helpers implement the described
methods; they are not recovered original scripts. The synthetic examples do
not recompute the reported clinical results.

The source is identified by its Git commit and file hashes. Run records retain
the settings and provenance needed to identify each new execution.

## What the public materials support

The synthetic fixtures support execution and inspection of the reference
implementation. The scoring and analysis scripts can process compatible
reference annotations and prediction artifacts. Their example outputs are not
the manuscript's published values or figure images.

Verification of the reported results would require controlled access to the
original restricted inputs and retained study outputs, with compatible artifact
formats. Running this public implementation on another corpus is a new
evaluation, not a replication of the numerical results. The Presidio,
inter-rater agreement, and timing helpers require their corresponding inputs;
their inclusion does not recover the study records or verify the reported
aggregates. See [the manuscript map](MANUSCRIPT_MAP.md).

## Recorded study environment and missing details

The manuscript reports `llama-server` Homebrew build **10050** on an Apple M4 Max
with 48 GB unified memory and the Metal backend. Requests used temperature `0`,
seed `42`, `reasoning_effort: low`, and the locked JSON schema. Every model
processed the same 100 notes in two sequential passes, with three separate
executions under those settings. Executions were stability checks, not
independent samples. The reported character-metric confidence intervals resample
notes, not model executions. Note-level completeness is now reported
descriptively as mean complete-note counts, without confidence intervals. A
rounded mean does not identify which notes were complete in each execution or
establish that the executions were identical.

The exact Gemma-4 E2B filename, size, and SHA-256 were retained. Exact checkpoint
filenames, digests, or source revisions for the other 16 checkpoints, and the
exact `llama.cpp` source revision and build flags, were not preserved in the
available study record. The recorded Homebrew build number does not fill those
gaps. Public metadata therefore cannot reconstruct the original environment
byte for byte. See [Models](MODELS.md) for the checkpoint record.

## Provenance for new executions

Each model output directory now contains `inference_manifest.json`. It records:

- the backend, time, Python/platform information, available checkout commit,
  and hashes of the Python source files;
- the actual segment size and overlap, protocol-file hashes, request settings,
  timeout, and whether raw response content is retained;
- hashes and sizes of supplied checkpoint files, plus supplied server version
  and command information.

The shell launcher supplies the checkpoint it launches and records the server's
version output and the exact argument list it uses. Hashes are calculated by
reading the local files, including all parts of standard numbered GGUF shards.
An unknown or unavailable value is left empty or `null`, not inferred from a
model label. The offline stub records its name-list hash; prompt and decoding
fields are `null` because it does not use the model protocol.

Direct runner calls do not independently interrogate or authenticate a model
endpoint's loaded weights. Their optional checkpoint and server details are
caller-supplied provenance. A matching filename or hash of a local file does not
prove that an already-running endpoint loaded it. Code hashes also matter when a
checkout has uncommitted changes or no Git metadata. `run_manifest.json` records
the scorer's file hash separately.

The explicit server command may contain local paths or credentials. Do not pass
secrets as provenance. Manifests use private permissions, but their metadata still
belongs inside the same controls as the rest of the run.

## Expected segments versus surviving records

Before issuing requests for each note and pass, the runner saves
`expected_segments/<doc-id>/manifest.json` under that pass's directory. The plan
contains the source hash, representation length, segment settings, and each
expected segment's ID and offsets. It contains no note or segment text.

The scorer checks this plan against the observed raw-response and validation
records. `segment_coverage_verified` requires a valid plan, matching record IDs,
and consistent records. `operational_complete` additionally requires every
expected segment to finish with `stop` and yield a usable response. Missing
records cannot turn an incomplete pass into an apparently complete one merely
by reducing the observed denominator. Malformed plans and unexpected segment
IDs are rejected.

Older outputs without plans remain usable for annotation-based accuracy
calculations. Their expected request count and `finish_stop_rate` are unknown
(blank), their coverage is unverified, and they are not included in the
`operational_complete` diagnostic cohort. Do not silently manufacture plans from
surviving records: that would restore the original uncertainty rather than
resolve it.

Usable-response and strict-schema-valid rates retain their conditional
denominator: available validation records. A rate of 1.0 can coexist with missing
requests or incomplete identifier removal. The current diagnostic changes do
not change the saved predictions or the `all_expected` accuracy formulas.

The scorer validates reference span coordinates, annotation types, and source
identity before scoring. It writes source, per-note annotation, and full
reference-set fingerprints into its analysis outputs. Across-execution and
across-model helpers require compatible fingerprints and denominators; old
tables without this identity information must be rescored from compatible
artifacts before those helpers compare them. These safeguards validate supplied
inputs, not the completeness of clinical annotation or historical run provenance.
