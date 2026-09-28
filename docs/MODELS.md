# Study checkpoints

The study evaluated these 17 checkpoints on the same 100 notes, in two passes and
three separate locked executions. The table retains the manuscript's descriptive
cumulative-recall ordering; it is not a deployment recommendation. Checkpoint
metadata are stored in [model_configs](../model_configs) and can be printed with:

```bash
python3 scripts/make_checkpoints_table.py
```

Parameters are in billions; size is the recorded on-disk size in GB, not a measured
peak memory requirement. E2B and E4B list total/effective parameters. The
26B-A4B mixture-of-experts checkpoint lists total/active parameters.

| Model | Parameters (B) | Quantization | Size (GB) | GGUF repository |
| --- | ---: | --- | ---: | --- |
| Gemma-4 31B | 30.7 | Q5_K_M | 21.66 | [unsloth/gemma-4-31B-it-GGUF](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) |
| Gemma-4 12B | 12.0 | Q8_0 | 12.67 | [unsloth/gemma-4-12B-it-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-GGUF) |
| Mistral-Small 24B | 24.0 | Q5_K_M | 16.76 | [unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF](https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF) |
| Gemma-4 E4B | 8.0/4.5 | Q8_0 | 8.19 | [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) |
| Gemma-4 26B-A4B | 25.2/3.8 | UD-Q5_K_M | 21.15 | [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) |
| Ministral-3 14B | 14.0 | Q6_K | 11.09 | [unsloth/Ministral-3-14B-Instruct-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-14B-Instruct-2512-GGUF) |
| Gemma-4 E2B | 5.1/2.3 | BF16 | 9.31 | [ggml-org/gemma-4-E2B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF) |
| Gemma-3 27B | 27.0 | Q5_K_M | 19.27 | [unsloth/gemma-3-27b-it-GGUF](https://huggingface.co/unsloth/gemma-3-27b-it-GGUF) |
| Phi-4 14B | 14.0 | Q6_K | 12.03 | [unsloth/phi-4-GGUF](https://huggingface.co/unsloth/phi-4-GGUF) |
| MedGemma 27B | 27.0 | Q5_K_M | 19.27 | [unsloth/medgemma-27b-text-it-GGUF](https://huggingface.co/unsloth/medgemma-27b-text-it-GGUF) |
| Granite-4.1 30B | 30.0 | Q5_K_M | 20.49 | [ibm-granite/granite-4.1-30b-GGUF](https://huggingface.co/ibm-granite/granite-4.1-30b-GGUF) |
| Granite-4.1 8B | 8.0 | Q6_K | 7.22 | [ibm-granite/granite-4.1-8b-GGUF](https://huggingface.co/ibm-granite/granite-4.1-8b-GGUF) |
| Gemma-3 4B | 4.0 | Q8_0 | 4.13 | [unsloth/gemma-3-4b-it-GGUF](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF) |
| Llama-3.1 8B | 8.0 | Q6_K | 6.60 | [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) |
| OLMo-3 7B | 7.0 | Q8_0 | 7.76 | [lmstudio-community/Olmo-3-7B-Instruct-GGUF](https://huggingface.co/lmstudio-community/Olmo-3-7B-Instruct-GGUF) |
| Gemma-3 1B | 1.0 | Q8_0 | 1.07 | [ggml-org/gemma-3-1b-it-GGUF](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF) |
| MedGemma-1.5 4B | 4.0 | Q8_0 | 4.13 | [unsloth/medgemma-1.5-4b-it-GGUF](https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF) |

The configuration files also retain developer, family, architecture, and medical
specialization fields. MedGemma is the medical-domain-adapted family in this
roster. Model weights are governed by their respective licenses, not this
repository's MIT license.

The Gemma-4 E2B record retains the exact file:

```text
filename: gemma-4-E2B-it-BF16.gguf
size_bytes: 9311305152
sha256: 246c5fb19e64b941b531e86639f070bc67b00f20e9ec8556a6c02653321bc65d
```

Exact checkpoint filenames, digests, or source revisions for the other 16 models
are not available in the preserved study record. A repository link or
quantization label does not identify checkpoint bytes uniquely. The new run
manifest records supplied local checkpoint hashes for future executions; it does
not fill gaps in the original study metadata. See [Reproducibility](REPRODUCIBILITY.md).
