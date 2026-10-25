#!/usr/bin/env bash
# go.sh 的免门槛版：迁移已确认稳定，直接执行 验证→解包→补丁→探针→开销评估
set -uo pipefail

VENV=/root/autodl-tmp/venv
export PATH="$VENV/bin:/root/miniconda3/bin:$PATH"
LOG=/root/go/orchestrator.log
mkdir -p /root/go
log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }

# ---------- 验证 ----------
PY=$VENV/bin/python
$PY - <<'EOF' || { log "FATAL: fp8 incomplete"; exit 1; }
import json, os
os.chdir("/root/autodl-tmp/models/qwen3.8-27b-fp8")
idx = json.load(open("model.safetensors.index.json"))
need = set(idx["weight_map"].values()); have = set(os.listdir("."))
miss = [f for f in sorted(need) if f not in have]
assert not miss and os.path.exists("mtp.safetensors")
print("fp8 OK")
EOF
$PY -c "import torch,vllm; assert torch.cuda.is_available() and vllm.__version__=='0.32.0'; print('venv OK')" \
  || { log "FATAL: venv broken"; exit 1; }
MR=$VENV/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py
[ -f "$MR" ] || { log "FATAL: no model_runner at $MR"; exit 1; }
log "verify OK"
touch /root/go/stage_verify.done

# ---------- 部署 ----------
mkdir -p /root/spectuner
tar -xzf /root/st_bundle.tar.gz -C /root/spectuner || { log "FATAL: extract"; exit 1; }
log "bundle extracted"
touch /root/go/stage_deploy.done

# ---------- 补丁 ----------
$PY /root/spectuner/bench/patch_timing.py "$MR" >> "$LOG" 2>&1 \
  || { log "FATAL: patch_timing failed"; exit 1; }
grep -q "_stt_begin_step" "$MR" || { log "FATAL: patch marker absent"; exit 1; }
log "patch_timing applied"
touch /root/go/stage_patch.done

# ---------- 探针三臂 ----------
cd /root/spectuner/probe
OUT=/root/autodl-tmp/probe_st1
log "probe starting -> $OUT"
QUANT=fp8 OUT_DIR=$OUT bash probe_run.sh >> "$LOG" 2>&1
RC=$?
log "probe finished rc=$RC"
touch /root/go/stage_probe.done
[ $RC -eq 0 ] || exit 1

# ---------- 开销评估 ----------
log "overhead eval: timing-off c16"
CONCURRENCY=16 TIMING=off QUANT=fp8 SPEC_SET=mtp3 \
  OUT_DIR=/root/autodl-tmp/probe_st1_off PORT=8011 \
  bash probe_run.sh >> "$LOG" 2>&1
log "overhead rc=$?"
touch /root/go/stage_overhead.done
log "ALL DONE"
