#!/usr/bin/env bash
# GPT-2 CE activation-steering Token Presence via runner.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHODS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONTEXT="${CONTEXT:-Once upon a time}"
K="${K:-30}"
TOKENS=("ActionCode" " mere")

MODEL_ID="gpt2"
PROPOSAL_TYPE="CE_ACTIVATION"
EVENT_TYPE="token"
EVENT_CONFIG_JSON='{"surrogate":"sum"}'
OUTPUT_PATH="${OUTPUT_PATH:-${METHODS_DIR}/../runs/ce_activation_token}"

STEPS=300
EVAL_STEPS=500
BATCH_SIZE=128
SEED=12
LOG_EVERY=1

CE_ELITE_RATIO=0.3
CE_SMOOTHING=0.5
CE_SIGMA_INIT=0.1
CE_STEERING_INIT_SCALE=0.0
CE_GENERATION_TEMP=1.0
CE_EVAL_USE_MEAN_ONLY=true
CE_EARLY_STOP_ON_LOW_ESS=true
RARE_EVENT_ESS_MIN_RATE=0.1
POP_ESS_TARGET=0.005
ESS_TARGETS=(0.1)

if [ -n "${GPU_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

CMD=(
  "$PYTHON_BIN" "${METHODS_DIR}/runner.py"
  "$CONTEXT" "$K"
  --tokens "${TOKENS[@]}"
  --event_type "$EVENT_TYPE"
  --event_config_json "$EVENT_CONFIG_JSON"
  --ess_target "${ESS_TARGETS[@]}"
  --proposal_type "$PROPOSAL_TYPE"
  --model_id "$MODEL_ID"
  --output_path "$OUTPUT_PATH"
  --steps "$STEPS"
  --eval_steps "$EVAL_STEPS"
  --batch_size "$BATCH_SIZE"
  --seed "$SEED"
  --log_every "$LOG_EVERY"
  --no-use_lora
  --adaptive_reg_enabled false
  --ce_elite_ratio "$CE_ELITE_RATIO"
  --ce_smoothing "$CE_SMOOTHING"
  --ce_sigma_init "$CE_SIGMA_INIT"
  --ce_steering_init_scale "$CE_STEERING_INIT_SCALE"
  --ce_generation_temperature "$CE_GENERATION_TEMP"
  --ce_eval_use_mean_only "$CE_EVAL_USE_MEAN_ONLY"
  --ce_early_stop_on_low_ess "$CE_EARLY_STOP_ON_LOW_ESS"
  --rare_event_ess_min_rate "$RARE_EVENT_ESS_MIN_RATE"
  --pop_ess_target "$POP_ESS_TARGET"
)

echo "Launching $PROPOSAL_TYPE token runner"
echo "  model_id=$MODEL_ID  output_path=$OUTPUT_PATH"
echo "  context='$CONTEXT'  k=$K  tokens=${TOKENS[*]}"
"${CMD[@]}"
