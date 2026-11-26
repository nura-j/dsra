#!/usr/bin/env bash
# Cross-model DSRA causal-tracing runs (Chapter 6 cross-family generality check).
#
# Usage:
#   bash run_dsra_models.sh                    # the four CAP-set models
#   bash run_dsra_models.sh ch6                # the five Chapter 6 (COMPASS) models
#   bash run_dsra_models.sh gpt2-large         # one model
#   N_FACTS=25 bash run_dsra_models.sh ch6     # quick pilot
#   DTYPE=bfloat16 bash run_dsra_models.sh     # memory fallback (preferred over float16)
#   WINDOW=6 bash run_dsra_models.sh <model>   # override the severed-window width
#                                              # (applies to EVERY model in the run,
#                                              #  replacing each per-model value)
#
# See RUNBOOK_cross_model.md for environment requirements (in particular the
# transformers >= 4.43 requirement for Llama-3.2) and gated-model access.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
export TOKENIZERS_PARALLELISM=false

N_FACTS=${N_FACTS:-800}
NOISE=${NOISE:-s3}

python - <<'PY'
import torch, transformers
assert torch.cuda.is_available(), "CUDA GPU required"
print(f"torch {torch.__version__}, transformers {transformers.__version__}, "
      f"GPU: {torch.cuda.get_device_name(0)}")
PY

transformers_at_least () {
  python - "$1" <<'PY'
import sys
from packaging import version
import transformers
sys.exit(0 if version.parse(transformers.__version__) >= version.parse(sys.argv[1]) else 1)
PY
}

run_model () {
  local model="$1" template="$2" dtype="${DTYPE:-$3}" win="${WINDOW:-${4:-10}}"
  local tag="${model//\//_}"
  local facts="data/facts_${tag}_n${N_FACTS}.json"
  # A window under 2 restores no layers, so the severed mlp/attn maps come out
  # flat at low_score; the run still completes and writes a full set of npz.
  # Catch it here too, before the model is loaded. Note that a WINDOW in the
  # environment replaces the per-model value for every model in the run, so one
  # stale WINDOW=0 silently degrades the whole group.
  if ! [[ "${win}" =~ ^[0-9]+$ ]] || [ "${win}" -lt 2 ]; then
    echo "Refusing to trace ${model} with window='${win}': the severed-window" >&2
    echo "traces need at least 2 layers. Check for a stale WINDOW in the" >&2
    echo "environment (unset WINDOW) or pass a WINDOW of 2 or more." >&2
    exit 1
  fi
  echo "=== ${model} (dtype=${dtype}, window=${win}, template=${template}) ==="
  if [ ! -f "${facts}" ]; then
    python -m dataset_construction.facts.build_facts \
      --model_name "${model}" --template "${template}" \
      --torch_dtype "${dtype}" --screen --max_facts "${N_FACTS}" \
      --out "${facts}" 2>&1 | tee "logs/facts_${tag}.log"
  else
    echo "Fact file ${facts} exists; reusing it."
  fi
  python -m tracing.trace \
    --model_name "${model}" --fact_file "${facts}" \
    --noise_level "${NOISE}" --torch_dtype "${dtype}" \
    --window "${win}" \
    2>&1 | tee "logs/trace_${tag}.log"
}

# Model groups: no arguments = the four CAP-set models; `ch6` = the sampled
# Chapter 6 (COMPASS) models: Pythia-1B, Llama-3.2-1B, Qwen2.5-1.5B. The
# remaining COMPASS models (Pythia-14M, Pythia-410M) keep their case entries
# and can be run by naming them explicitly. Explicit HF ids can be mixed
# with group names.
CAP_SET=(gpt2-large Qwen/Qwen2.5-3B google/gemma-2b meta-llama/Llama-3.2-3B)
CH6_SET=(EleutherAI/pythia-1b meta-llama/Llama-3.2-1B Qwen/Qwen2.5-1.5B)
if [ "$#" -gt 0 ]; then
  MODELS=()
  for arg in "$@"; do
    case "${arg}" in
      ch6) MODELS+=("${CH6_SET[@]}") ;;
      cap) MODELS+=("${CAP_SET[@]}") ;;
      *)   MODELS+=("${arg}") ;;
    esac
  done
else
  MODELS=("${CAP_SET[@]}")
fi

# Severed-window width per model, stated explicitly in every case entry.
# The rule preserves GPT-J's window-to-depth ratio of 10/28 = 0.36:
#
#   w = 10                          for L >= 28
#   w = max(2, round(10 * L / 28))  for L <  28
#
# The threshold is the depth at which the fixed GPT-J window of 10 stops being
# a minority of the network. An earlier version of this rule cut over at 18
# layers, which left Gemma-2B (L=18) at w/L = 0.56 — a wider relative window
# than any other run and the opposite of what the rule exists to prevent. See
# the runbook, Section 5.
for model in "${MODELS[@]}"; do
  case "${model}" in
    gpt2-large)                 # 36 layers
      run_model "${model}" '{} is called a "' float32 10
      ;;
    Qwen/Qwen2.5-3B)            # 36 layers
      run_model "${model}" '{} is commonly known as "' float32 10
      ;;
    google/gemma-2b)            # 18 layers
      run_model "${model}" '{} is called a "' float32 6
      ;;
    meta-llama/Llama-3.2-3B)    # 28 layers
      if transformers_at_least 4.43.0; then
        run_model "${model}" '{} is called a "' float32 10
      else
        echo "SKIPPING ${model}: Llama-3.2 configs (rope_type=llama3) need"
        echo "transformers >= 4.43; this environment has an older version."
        echo "See RUNBOOK_cross_model.md for the one-line environment clone."
      fi
      ;;
    # --- Chapter 6 (COMPASS) model set.
    EleutherAI/pythia-14m)      # 6 layers
      run_model "${model}" '{} is called a "' float32 2
      ;;
    EleutherAI/pythia-410m)     # 24 layers
      run_model "${model}" '{} is called a "' float32 9
      ;;
    EleutherAI/pythia-1b)       # 16 layers
      run_model "${model}" '{} is called a "' float32 6
      ;;
    meta-llama/Llama-3.2-1B)    # 16 layers
      if transformers_at_least 4.43.0; then
        run_model "${model}" '{} is called a "' float32 6
      else
        echo "SKIPPING ${model}: Llama-3.2 configs (rope_type=llama3) need"
        echo "transformers >= 4.43; this environment has an older version."
      fi
      ;;
    Qwen/Qwen2.5-1.5B)          # 28 layers
      run_model "${model}" '{} is commonly known as "' float32 10
      ;;
    *)
      echo "Unknown model '${model}'; add a case with its IDM template and dtype." >&2
      exit 1
      ;;
  esac
done

echo "Done. Trace tensors: results/n${NOISE}_r0_<model>/causal_trace/cases/knowledge_<id>[_mlp|_attn].npz"
