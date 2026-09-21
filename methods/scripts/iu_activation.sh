#!/usr/bin/env bash
# GPT-2 IU activation-steering Token Presence via runner.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHODS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONTEXT="${CONTEXT:-Once upon a time}"
K="${K:-30}"
TOKENS=("ActionCode" " mere")

ESS_TARGETS=(0.1)
INIT_LAMBDAS=(10)
LAMBDA_FLOOR=0.1
RARE_EVENT_ESS_MIN_RATE=0.1
POP_ESS_TARGET=0.005

MODEL_ID="gpt2"
PROPOSAL_TYPE="IU_ACTIVATION"
EVENT_TYPE="token"
EVENT_CONFIG_JSON='{"surrogate":"sum"}'
OUTPUT_PATH="${OUTPUT_PATH:-${METHODS_DIR}/../runs/iu_activation_token}"

STEPS=300
EVAL_STEPS=500
BATCH_SIZE=128
LR=1e-2
DUAL_LR=0.01
DUAL_OPTIMIZER="sgd"
SEED=12
LOG_EVERY=1
USE_LORA=false
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
USE_CHAT_TEMPLATE=false
IU_ACT_STEERING_INIT_SCALE=0.01

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
  --init_lambda "${INIT_LAMBDAS[@]}"
  --lambda_floor "$LAMBDA_FLOOR"
  --rare_event_ess_min_rate "$RARE_EVENT_ESS_MIN_RATE"
  --pop_ess_target "$POP_ESS_TARGET"
  --proposal_type "$PROPOSAL_TYPE"
  --model_id "$MODEL_ID"
  --output_path "$OUTPUT_PATH"
  --steps "$STEPS"
  --eval_steps "$EVAL_STEPS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --dual_lr "$DUAL_LR"
  --dual_optimizer "$DUAL_OPTIMIZER"
  --seed "$SEED"
  --log_every "$LOG_EVERY"
  --lora_rank "$LORA_RANK"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --iu_act_steering_init_scale "$IU_ACT_STEERING_INIT_SCALE"
  --no-use_lora
)

echo "Launching $PROPOSAL_TYPE token runner"
echo "  model_id=$MODEL_ID  output_path=$OUTPUT_PATH"
echo "  context='$CONTEXT'  k=$K  tokens=${TOKENS[*]}"
"${CMD[@]}"
