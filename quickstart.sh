#!/usr/bin/env bash
# Serve Qwen3.8-27B — or one of its finetunes — on a free Kaggle TPU v5e-8.
#
#   ./quickstart.sh                # Qwen/Qwen3.8-27B (base)
#   ./quickstart.sh serenity       # ReadyArt/Serenity-27B
#   ./quickstart.sh uncensored     # orcarouter/Qwen3.8-27B-Uncensored (gated, needs HF_TOKEN)
#
# Extra arguments go straight to `launch.py serve`:
#   ./quickstart.sh serenity --max-model-len 131072 --max-num-seqs 16 --mtp 0
#
# Environment:
#   HF_TOKEN            Hugging Face token; required for a gated repo
#   WEIGHTS_DATASET     Kaggle dataset mirroring the weights. Empty means the kernel
#                       downloads them from Hugging Face (~10 min slower per launch).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

preset="${1:-base}"
[ $# -gt 0 ] && shift

case "$preset" in
  base)
    hf_model="Qwen/Qwen3.8-27B"
    served_name="qwen3.8-27b"
    weights_default="rahim3/qwen3-8-27b-bf16" ;;
  serenity)
    hf_model="ReadyArt/Serenity-27B"
    served_name="serenity-27b"
    weights_default="darkessid/serenity-27b-bf16" ;;
  uncensored)
    hf_model="orcarouter/Qwen3.8-27B-Uncensored"
    served_name="qwen3.8-27b-uncensored"
    weights_default="" ;;
  -h|--help)
    sed -n '2,14p' "$0" | sed 's/^# \?//'; exit 0 ;;
  *)
    echo "Unknown preset '$preset' — expected base, serenity or uncensored." >&2
    echo "Run './quickstart.sh --help' for usage." >&2
    exit 1 ;;
esac

weights="${WEIGHTS_DATASET-$weights_default}"

# launch.py checks the Kaggle CLI and its credentials itself, so we only need Python.
command -v python3 >/dev/null || { echo "python3 is not on PATH." >&2; exit 1; }

# --- gated repos need a token ------------------------------------------------
token_arg=()
if [ "$preset" = "uncensored" ]; then
  if [ -z "${HF_TOKEN:-}" ] && [ -r "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
  fi
  if [ -z "${HF_TOKEN:-}" ]; then
    cat >&2 <<'EOF'
orcarouter/Qwen3.8-27B-Uncensored is gated. Request access on its model page, then
export a read-only token before launching:

    export HF_TOKEN=hf_...

The token is embedded in the source of the (private) Kaggle kernel, so use a
read-only one.
EOF
    exit 1
  fi
  token_arg=(--hf-token "$HF_TOKEN")
fi

# --- launch ------------------------------------------------------------------
echo "Model    : $hf_model  (served as '$served_name')"
echo "Weights  : ${weights:-downloaded from Hugging Face inside the kernel}"
echo

exec python3 launch.py serve \
  --hf-model "$hf_model" \
  --served-model-name "$served_name" \
  --weights-dataset "$weights" \
  "${token_arg[@]}" "$@"
