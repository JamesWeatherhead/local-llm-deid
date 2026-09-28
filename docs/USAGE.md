# Usage

Run the commands below from the repository root. The Python modules use the
standard library. The shell launcher needs bash, curl, and a compatible
`llama-server`; the model downloader is a separate tool.

## Offline example without make

```bash
PYTHONPATH=src python3 -m deid.run_model \
    --model-id offline-stub \
    --notes-dir fixtures/synthetic/notes \
    --out-dir out/offline/predictions \
    --offline-stub --name-file fixtures/synthetic/name_gazetteer.txt

PYTHONPATH=src python3 -m deid.metrics \
    --pred-dir out/offline/predictions \
    --gold-dir fixtures/synthetic/gold --out-dir out/offline

python3 scripts/make_results_table.py --metrics out/offline/metrics_long.csv
```

The stub is a pattern matcher for testing, not a model evaluated in the paper.
It deliberately omits geographic identifiers. Its expected counts are documented
in [the fixture README](../fixtures/synthetic/README.md). The separate
[long fixture](../fixtures/long_synthetic/README.md) exercises overlapping windows.

Choose a new output directory for a new execution. To replace a previous run of
the same model, add `--overwrite-model-output`; this removes only that model's
subtree. `make selftest` uses this flag for its `offline-stub` output.

## Optional package installation

The `PYTHONPATH=src` examples run from a checkout without installing the package.
Alternatively, use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install .
```

This installs `deid-run` and `deid-score` and makes `deid` importable without the
`PYTHONPATH` prefix. Use `python3 -m pip install -e .` for an editable install.
The prompts and examples remain repository files; supply `--protocol-dir` when
running outside the checkout. The permission controls target POSIX systems.

## Other models

Download a compatible GGUF and give the launcher its path and a run label.
Using a label that matches a file in `model_configs/` attaches the corresponding
metadata automatically. For example:

```bash
hf download unsloth/gemma-3-4b-it-GGUF \
    gemma-3-4b-it-Q8_0.gguf --local-dir models/gemma-3-4b-it

scripts/run_local_end_to_end.sh \
    --gguf models/gemma-3-4b-it/gemma-3-4b-it-Q8_0.gguf \
    --model-id gemma-3-4b-it --out-dir out/gemma-4b-example
```

The [study checkpoint table](MODELS.md) records the evaluated configurations.
Check the repository for available files and license terms. Some checkpoints are
split across several GGUF files: download every part and pass the first part to
the launcher. An arbitrary single shard is not a complete checkpoint. Runtime
memory includes the weights, context cache, and other buffers; on-disk size alone
is not a memory requirement.

The launcher can download one file itself:

```bash
scripts/run_local_end_to_end.sh \
    --hf-repo ggml-org/gemma-3-1b-it-GGUF \
    --hf-file gemma-3-1b-it-Q8_0.gguf \
    --model-id gemma-3-1b-it --out-dir out/download-example
```

Install the downloader with `python3 -m pip install -U huggingface_hub`.
The current command is `hf`; see the [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
Automatic download fetches only the named file, not the other parts of a split
checkpoint. `--model-config PATH` overrides the metadata selected by model ID.

## Other hardware

The README's installation example uses Homebrew on Apple Silicon. The launcher
requests GPU offload with `--n-gpu-layers 999`; the backend is determined by the
installed `llama.cpp` build, not by that number. Use `--n-gpu-layers 0` for CPU
execution. A CUDA build is needed for NVIDIA GPU offload. Other installations may
require different server settings; consult [llama.cpp](https://github.com/ggml-org/llama.cpp).

Models must support the requested chat template, context length, and structured
output. Changing the GGUF path does not establish equivalent runtime behavior
across checkpoints or reproduce the study environment.

## Evaluate your own notes

Use a directory of UTF-8 `<doc-id>.txt` files and a matching directory of reference
JSON files. IDs may contain letters, numbers, underscores, and hyphens. Keep the
reference `text` identical to the source note and verify the offset convention;
see [Reference annotations](ANNOTATIONS.md). Do not place clinical data in a
public repository.

```bash
scripts/run_local_end_to_end.sh \
    --gguf /path/to/model.gguf --model-id my-model \
    --notes-dir /approved/notes --gold-dir /approved/gold \
    --out-dir /approved/evaluation
```

The launcher always scores against reference annotations. To run extraction
without references, start a server yourself and use the Python runner:

```bash
llama-server -m /path/to/model.gguf --host 127.0.0.1 --port 8081 \
    --ctx-size 8192 --jinja

# In a second terminal:
PYTHONPATH=src python3 -m deid.run_model \
    --model-id my-model --notes-dir /approved/notes \
    --out-dir /approved/predictions --protocol-dir protocol \
    --api-base http://127.0.0.1:8081 --checkpoint-file /path/to/model.gguf
```

For an already-running server, checkpoint paths and optional `--server-version`
and `--server-command` values are caller-supplied provenance, not independent
verification of what the endpoint loaded. `--server-command` consumes all
remaining arguments and must come last. Do not include secrets in metadata.
The shell launcher records its actual launch command and version output.

Score one or more completed model directories together:

```bash
PYTHONPATH=src python3 -m deid.metrics \
    --pred-dir /approved/predictions --gold-dir /approved/gold \
    --out-dir /approved/scores
```

Every note in the reference directory remains in the accuracy denominator.
Missing predictions count as empty predictions; unexpected documents or invalid
prediction artifacts raise an error. Keep different corpora and independent
executions in separate output roots.

The runner saves prediction offsets, not redacted note files. The rendering
functions in `src/deid/pipeline.py` are available for inspection workflows; their
text may still contain identifiers and needs separate review and storage controls.

## Segmentation and request settings

`--segment-size` and `--overlap` on the Python runner override the study defaults
of 3,500 and 400 codepoints. Changing these values is a new evaluation setting,
not a reproduction of the locked study protocol. The defaults need not be edited
in source. `--request-timeout SECONDS` limits a request's duration; the default
has no timeout. A timeout is recorded as an error, without an automatic retry.

The request payload retains temperature `0`, seed `42`, `reasoning_effort: low`,
and the supplied JSON schema. These are request settings, not a guarantee of
identical outputs or reasoning behavior across models and server versions.

## Analysis and tests

```bash
make selftest
make figures
make tables
make ci
make test
```

`make all` runs that chain using the synthetic example. The scripts also accept
explicit input paths; see [Analysis scripts](ANALYSIS.md). Synthetic execution
and passing tests do not reproduce clinical performance or establish suitability
for releasing clinical text.
