#!/usr/bin/env bash
#
# run_local_end_to_end.sh -- chain the whole de-identification pipeline on one
# Apple Silicon Mac: launch a local model with llama.cpp, run both passes over a
# set of notes, score against gold, print the ranked table, and render the
# result figures. The llama-server process is started and stopped for you.
#
# This script targets Apple Silicon MacBooks (the llama.cpp Metal backend). On
# other hardware the same steps apply but the llama-server flags differ; see
# "Other hardware" under the Quickstart in the README.
#
# By default it runs on the bundled synthetic, PHI-free fixture, so the whole
# chain works with no real data and no weights beyond the model you pick. Point
# --notes-dir / --gold-dir at your own corpus to reproduce the study, and add
# --strip-raw-content there to keep the out/ tree free of PHI. Model metadata is
# resolved automatically from model_configs/<model-id>.json when present.
#
# Usage:
#   scripts/run_local_end_to_end.sh --gguf models/gemma-3-1b-it/gemma-3-1b-it-Q8_0.gguf
#   scripts/run_local_end_to_end.sh --hf-repo ggml-org/gemma-3-1b-it-GGUF --hf-file gemma-3-1b-it-Q8_0.gguf
#   scripts/run_local_end_to_end.sh --gguf /path/model.gguf --notes-dir /data/notes --gold-dir /data/gold
#
set -euo pipefail

# --- defaults ---------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GGUF=""
HF_REPO=""
HF_FILE=""
MODEL_ID=""
NOTES_DIR="$REPO_ROOT/fixtures/synthetic/notes"
GOLD_DIR="$REPO_ROOT/fixtures/synthetic/gold"
OUT_DIR="$REPO_ROOT/out"
HOST="127.0.0.1"
PORT="8081"
CTX_SIZE="8192"
NGL="999"           # offload all layers to the Metal GPU on Apple Silicon
LOAD_TIMEOUT="600"  # seconds to wait for the model to finish loading
MODEL_CONFIG=""     # optional model_config.json (auto-resolved from --model-id)
STRIP_RAW=""        # set by --strip-raw-content to keep out/ PHI-free

# Print the leading comment block as help: skip the shebang, strip the "# "
# prefix, and stop at the first line of real code (so section dividers below
# are not swept in).
usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit "${1:-0}"; }

# --- parse args -------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --gguf)          GGUF="$2"; shift 2 ;;
    --hf-repo)       HF_REPO="$2"; shift 2 ;;
    --hf-file)       HF_FILE="$2"; shift 2 ;;
    --model-id)      MODEL_ID="$2"; shift 2 ;;
    --notes-dir)     NOTES_DIR="$2"; shift 2 ;;
    --gold-dir)      GOLD_DIR="$2"; shift 2 ;;
    --out-dir)       OUT_DIR="$2"; shift 2 ;;
    --host)          HOST="$2"; shift 2 ;;
    --port)          PORT="$2"; shift 2 ;;
    --ctx-size)      CTX_SIZE="$2"; shift 2 ;;
    --n-gpu-layers)  NGL="$2"; shift 2 ;;
    --model-config)  MODEL_CONFIG="$2"; shift 2 ;;
    --strip-raw-content) STRIP_RAW="1"; shift ;;
    -h|--help)       usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done

# --- 0. required tools ------------------------------------------------------
command -v llama-server >/dev/null || { echo "llama-server not found. Install llama.cpp:  brew install llama.cpp" >&2; exit 1; }
command -v python3      >/dev/null || { echo "python3 not found." >&2; exit 1; }
command -v curl         >/dev/null || { echo "curl not found." >&2; exit 1; }

# --- 1. resolve the model file (download from Hugging Face if requested) -----
if [ -z "$GGUF" ]; then
  if [ -n "$HF_REPO" ] && [ -n "$HF_FILE" ]; then
    command -v huggingface-cli >/dev/null || { echo "huggingface-cli not found. Install it:  pip install -U 'huggingface_hub[cli]'" >&2; exit 1; }
    dest="$REPO_ROOT/models/$(basename "$HF_REPO")"
    echo ">> downloading $HF_FILE from $HF_REPO into $dest"
    huggingface-cli download "$HF_REPO" "$HF_FILE" --local-dir "$dest"
    GGUF="$dest/$HF_FILE"
  else
    echo "provide --gguf PATH, or --hf-repo REPO --hf-file FILE" >&2; usage 1
  fi
fi
[ -f "$GGUF" ] || { echo "model file not found: $GGUF" >&2; exit 1; }
[ -n "$MODEL_ID" ] || MODEL_ID="$(basename "$GGUF" .gguf)"

# Metadata: if the caller did not pass --model-config, use a shipped one whose
# name matches the model id (populates the params/link columns in the table).
if [ -z "$MODEL_CONFIG" ] && [ -f "$REPO_ROOT/model_configs/$MODEL_ID.json" ]; then
  MODEL_CONFIG="$REPO_ROOT/model_configs/$MODEL_ID.json"
fi
[ -n "$MODEL_CONFIG" ] && echo ">> model metadata: $MODEL_CONFIG"

mkdir -p "$OUT_DIR"
SERVER_LOG="$OUT_DIR/llama-server.log"

# --- 2. launch llama-server (stopped automatically on exit) -----------------
# Refuse to start if something already answers on this port: a stale server
# would be scored in place of the model we mean to launch.
if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo "error: http://$HOST:$PORT already answers (a stale llama-server?)." >&2
  echo "       stop it, or pass --port N to use a free port." >&2
  exit 1
fi
echo ">> starting llama-server for '$MODEL_ID'"
echo "   model: $GGUF"
echo "   http://$HOST:$PORT   ctx=$CTX_SIZE  n-gpu-layers=$NGL   (log: $SERVER_LOG)"
llama-server -m "$GGUF" --host "$HOST" --port "$PORT" \
    --ctx-size "$CTX_SIZE" --n-gpu-layers "$NGL" --jinja \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() { echo ">> stopping llama-server (pid $SERVER_PID)"; kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# --- 3. wait for the model to load ------------------------------------------
printf ">> waiting for the model to load "
for i in $(seq 1 "$LOAD_TIMEOUT"); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo; echo "llama-server exited early; last log lines:" >&2; tail -n 25 "$SERVER_LOG" >&2; exit 1
  fi
  if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then echo " ready."; break; fi
  if [ "$i" -eq "$LOAD_TIMEOUT" ]; then
    echo; echo "timed out after ${LOAD_TIMEOUT}s; last log lines:" >&2; tail -n 25 "$SERVER_LOG" >&2; exit 1
  fi
  sleep 1; printf "."
done

# --- 4. inference: two passes over the notes --------------------------------
echo ">> [1/4] running the model over the notes (Pass 1, then Pass 2)"
RUN_ARGS=(--model-id "$MODEL_ID" --notes-dir "$NOTES_DIR"
    --out-dir "$OUT_DIR/predictions" --protocol-dir "$REPO_ROOT/protocol"
    --api-base "http://$HOST:$PORT")
[ -n "$MODEL_CONFIG" ] && RUN_ARGS+=(--model-config "$MODEL_CONFIG")
[ -n "$STRIP_RAW" ] && RUN_ARGS+=(--strip-raw-content)
PYTHONPATH="$REPO_ROOT/src" python3 -m deid.run_model "${RUN_ARGS[@]}"

# --- 5. scoring -------------------------------------------------------------
echo ">> [2/4] scoring predictions against gold"
PYTHONPATH="$REPO_ROOT/src" python3 -m deid.metrics \
    --pred-dir "$OUT_DIR/predictions" --gold-dir "$GOLD_DIR" --out-dir "$OUT_DIR"

# --- 6. ranked headline table -----------------------------------------------
echo ">> [3/4] building the ranked headline table"
python3 "$REPO_ROOT/scripts/make_results_table.py" --metrics "$OUT_DIR/metrics_long.csv"

# --- 7. result figures ------------------------------------------------------
echo ">> [4/4] rendering result figures from the metrics"
python3 "$REPO_ROOT/scripts/make_figures.py" --metrics "$OUT_DIR/metrics_long.csv" --out-dir "$OUT_DIR/figures"

echo
echo ">> done. Artifacts under: $OUT_DIR"
echo "   predictions: $OUT_DIR/predictions/$MODEL_ID"
echo "   metrics:     $OUT_DIR/metrics_long.csv"
echo "   figures:     $OUT_DIR/figures/figure2..6.svg (and .csv)"
