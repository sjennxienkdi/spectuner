#!/usr/bin/env bash
# SpecTuner 窗口编排器：等数据盘迁移稳定 → 验证 → 部署 → 打补丁 → 发探针
# 全程 nohup 自治，进度用 /root/go/stage_*.done 标记，日志 /root/go/orchestrator.log
set -uo pipefail

VENV=/root/autodl-tmp/venv
export PATH="$VENV/bin:/root/miniconda3/bin:$PATH"
LOG=/root/go/orchestrator.log
MD=/root/go
mkdir -p "$MD"
log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }

# ---------- 1) 等迁移稳定：du 连续 3 次不增长 且 总量 > 45G ----------
# （45G = FP8 29G + venv ~15G + 杂项；INT4/NVFP4 迁没迁不阻塞 FP8 探针）
log "waiting for data-disk migration to settle"
LAST=-1; STABLE=0; ELAPSED=0
while [ $ELAPSED -lt 5400 ]; do
  CUR=$(du -sm /root/autodl-tmp 2>/dev/null | cut -f1)
  CUR=${CUR:-0}
  if [ "$CUR" = "$LAST" ] && [ "$CUR" -gt 45000 ]; then
    STABLE=$((STABLE + 1))
  else
    STABLE=0
  fi
  [ $STABLE -ge 3 ] && break
  LAST=$CUR
  sleep 60
  ELAPSED=$((ELAPSED + 60))
done
if [ $STABLE -lt 3 ]; then
  log "FATAL: migration never settled in 90min (last=${LAST}MB)"
  exit 1
fi
log "migration settled at ${LAST}MB"
touch "$MD/stage_settle.done"

# ---------- 2) 验证 FP8 + venv ----------
PY=$VENV/bin/python
$PY - <<'EOF' || { log "FATAL: fp8 incomplete"; exit 1; }
import json, os, sys
os.chdir("/root/autodl-tmp/models/qwen3.8-27b-fp8")
idx = json.load(open("model.safetensors.index.json"))
need = set(idx["weight_map"].values())
have = set(os.listdir("."))
miss = [f for f in sorted(need) if f not in have]
print("fp8 shards need", len(need), "missing", len(miss), miss[:3])
assert not miss and os.path.exists("mtp.safetensors")
EOF
log "fp8 checkpoint OK"

$PY -c "import torch,vllm; assert torch.cuda.is_available(); assert vllm.__version__=='0.32.0'; print('venv OK', torch.__version__, vllm.__version__)" \
  || { log "FATAL: venv broken"; exit 1; }
MR=$VENV/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py
[ -f "$MR" ] || { log "FATAL: no model_runner at $MR"; exit 1; }
log "venv OK, model_runner present ($(wc -l < "$MR") lines)"
touch "$MD/stage_verify.done"

# ---------- 3) 部署脚本包 ----------
mkdir -p /root/spectuner
tar -xzf /root/st_bundle.tar.gz -C /root/spectuner || { log "FATAL: extract"; exit 1; }
log "bundle extracted to /root/spectuner"
touch "$MD/stage_deploy.done"

# ---------- 4) 打计时补丁 ----------
$PY /root/spectuner/bench/patch_timing.py "$MR" >> "$LOG" 2>&1 \
  || { log "FATAL: patch_timing failed"; exit 1; }
grep -q "_stt_begin_step" "$MR" || { log "FATAL: patch marker absent"; exit 1; }
log "patch_timing applied"
touch "$MD/stage_patch.done"

# ---------- 5) 发探针（三臂，FP8）----------
cd /root/spectuner/probe
OUT=/root/autodl-tmp/probe_st1
log "probe starting -> $OUT"
QUANT=fp8 OUT_DIR=$OUT bash probe_run.sh >> "$LOG" 2>&1
RC=$?
log "probe finished rc=$RC"
touch "$MD/stage_probe.done"
[ $RC -eq 0 ] || exit 1

# ---------- 6) 开销评估：同臂 K=3 计时关，只跑 c16 ----------
log "overhead eval: timing-off c16 short run"
cd /root/spectuner/probe
CONCURRENCY=16 TIMING=off QUANT=fp8 SPEC_SET=mtp3 \
  OUT_DIR=/root/autodl-tmp/probe_st1_off PORT=8011 \
  bash probe_run.sh >> "$LOG" 2>&1
RC2=$?
log "overhead run rc=$RC2"
touch "$MD/stage_overhead.done"
log "ALL DONE"
