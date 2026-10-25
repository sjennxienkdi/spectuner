#!/usr/bin/env bash
# SpecTuner GPU 探针运行器（项目 04 W1；~12-15 分钟，PRO 6000 实测节奏标定）
#
# 依次起探针臂（默认 target-only / K=3 / K=1），各跑指定并发的小样本，
# 服务端默认开 VLLM_TIMING_OUT 分项计时（需先打 bench/patch_timing.py 补丁；
# TIMING=off 可关——开销评估用）。
# 产出：$OUT_DIR/{arm}/serve_c*.json + $OUT_DIR/timing_{arm}.jsonl + server log
#
# 用法（云机）：
#   python3 <repo>/bench/patch_timing.py <venv>/.../vllm/v1/worker/gpu/model_runner.py
#   bash probe_run.sh                                    # 默认 FP8 三臂
#   QUANT=int4 SPEC_SET="none dspark" bash probe_run.sh
#   CONCURRENCY=16 TIMING=off OUT_DIR=... bash probe_run.sh   # 开销评估
#
# 完成后本地：
#   spectuner fit --timing $OUT_DIR/timing_mtp3.jsonl
#   spectuner recommend --matrix $OUT_DIR   # measured 模式（推荐先行）
set -euo pipefail

QUANT=${QUANT:-fp8}
SPEC_SET=${SPEC_SET:-"none mtp3 mtp1"}   # 探针臂序列：K=0 基线 + K=3 + K=1
CONCURRENCY=${CONCURRENCY:-"1 16 64"}
TIMING=${TIMING:-on}
PORT=${PORT:-8010}
MODELS_DIR=${MODELS_DIR:-/root/autodl-tmp/models}
INPUT_LEN=${INPUT_LEN:-2048}
OUTPUT_LEN=${OUTPUT_LEN:-256}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
OUT_DIR=${OUT_DIR:-probe_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)          # 绝对化，后续路径全部由它派生
# 探针小样本（10 分钟预算的关键；c1 档接受长度统计需 ≥8 请求）
declare -A NP=( [1]=8 [16]=24 [64]=32 )

case "$QUANT" in
  fp8)  MODEL="$MODELS_DIR/qwen3.8-27b-fp8"; REASONING=on ;;
  int4) MODEL="$MODELS_DIR/qwen3.8-27b-int4"; REASONING=off ;;
  *) echo "unknown QUANT=$QUANT"; exit 1 ;;
esac

echo "== probe start: model=$MODEL arms=[$SPEC_SET] conc=[$CONCURRENCY] timing=$TIMING out=$OUT_DIR"

for SPEC in $SPEC_SET; do
  SPEC_ARGS=()
  ARM=$SPEC
  case "$SPEC" in
    none)  ;;
    mtp[0-9]) SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC#mtp}}") ;;
    dspark)
      ARM=dspark
      SPEC_ARGS=(--speculative-config "{\"method\":\"dspark\",\"model\":\"$MODELS_DIR/drafter-dspark\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\"}") ;;
    *) echo "unknown SPEC=$SPEC"; exit 1 ;;
  esac
  REASON_ARGS=()
  [ "$REASONING" = on ] && REASON_ARGS=(--reasoning-parser qwen3)

  echo "== arm $ARM: starting server (timing=$TIMING)"
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
  SERVE_CMD=(vllm serve "$MODEL" --tensor-parallel-size 1
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 128 --language-model-only
    --port "$PORT" --async-scheduling "${REASON_ARGS[@]}" "${SPEC_ARGS[@]}")
  if [ "$TIMING" = on ]; then
    VLLM_TIMING_OUT="$OUT_DIR/timing_${ARM}.jsonl" \
      nohup "${SERVE_CMD[@]}" > "$OUT_DIR/server_${ARM}.log" 2>&1 &
  else
    nohup "${SERVE_CMD[@]}" > "$OUT_DIR/server_${ARM}.log" 2>&1 &
  fi
  SERVER_PID=$!
  trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

  for i in $(seq 1 120); do
    curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "server died:"; tail -30 "$OUT_DIR/server_${ARM}.log"; exit 1; }
    sleep 5
  done
  curl -s "http://127.0.0.1:$PORT/health" >/dev/null || { echo "not healthy after 600s"; exit 1; }
  echo "== arm $ARM: server ready"

  mkdir -p "$OUT_DIR/${ARM}"
  for C in $CONCURRENCY; do
    echo "== arm $ARM: bench c${C} (np=${NP[$C]})"
    vllm bench serve \
      --backend vllm --model "$MODEL" --port "$PORT" \
      --dataset-name random \
      --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
      --max-concurrency "$C" --num-prompts "${NP[$C]}" \
      2>&1 | tee "$OUT_DIR/${ARM}/serve_c${C}.json"
  done
  kill $SERVER_PID 2>/dev/null || true
  sleep 5
done

echo "== probe done -> $OUT_DIR"
echo "next: spectuner fit --timing $OUT_DIR/timing_mtp3.jsonl ; spectuner recommend --matrix $OUT_DIR"
