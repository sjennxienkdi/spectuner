#!/usr/bin/env python3
"""W3 窗口验收分析（2026-10-25 探针数据 → 项目书 §四 验收项）。

产出：
1. 成本模型 fit（三臂 timing 合并）+ step_gap 自检 MAPE
2. TPOT 预测 vs 实测（9 格）——稳态口径，验收 MAPE ≤ 20%
3. measured / model 两路推荐 vs confirm 真值（{1:3, 16:3, 64:0}）
4. 计时开销（开/关 c16）
5. 15 分钟端到端计时（orchestrator.log 时间戳）
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from spectuner.acceptance import AcceptanceModel
from spectuner.costmodel import CostModel, load_timing
from spectuner.fixtures import load_matrix
from spectuner.parse_bench import parse_serve_file
from spectuner.recommend import predict_tps, recommend_from_measured, recommend_from_model

ROOT = Path(__file__).resolve().parents[2] / "results" / "probe_st_20261025"
PROBE = ROOT / "probe_st1"
PROBE_OFF = ROOT / "probe_st1_off"


def main() -> None:
    print("== 1) cost model fit (3 arms merged) ==")
    recs = []
    for arm in ("none", "mtp1", "mtp3"):
        recs.extend(load_timing(PROBE / f"timing_{arm}.jsonl"))
    cm = CostModel.fit(recs)
    print(f"records={len(recs)} step_gap_mape={cm.mape_against(recs):.3f}")
    print(f"verify: 4tok={cm.verify_ms(4):.1f}ms 64={cm.verify_ms(64):.1f} "
          f"128={cm.verify_ms(128):.1f} 256(extrap)={cm.verify_ms(256):.1f}")
    print(f"other(1)={cm.other_ms(1):.2f} other(16)={cm.other_ms(16):.2f} "
          f"other(64)={cm.other_ms(64):.2f}")

    print("\n== 2) TPOT prediction vs measured (acceptance: MAPE<=20%) ==")
    # 探针小样本的有效 batch = min(并发上限, 请求数)：c64 档 np=32<64，
    # 稳态批达不到 64——预测必须用有效 batch，否则 c64 格系统性高估 step
    NP = {1: 8, 16: 24, 64: 32}
    e1 = parse_serve_file(PROBE / "mtp1" / "serve_c1.json")
    e3 = parse_serve_file(PROBE / "mtp3" / "serve_c1.json")
    am = AcceptanceModel(pos_rates=e3.pos_rates,
                         anchors={1: e1.acc_len, 3: e3.acc_len})
    print(f"acceptance: e(1)={am.e(1):.2f} e(2)={am.e(2):.2f} e(3)={am.e(3):.2f} "
          f"(pos_rates={[round(p,3) for p in am.pos_rates]})")
    errs = []
    for arm, k in (("none", 0), ("mtp1", 1), ("mtp3", 3)):
        for c in (1, 16, 64):
            m = parse_serve_file(PROBE / arm / f"serve_c{c}.json")
            b = min(c, NP[c])
            pred = cm.step_ms(b, k) / am.e(k) if k > 0 else cm.step_ms(b, k)
            ape = abs(pred - m.tpot_med) / m.tpot_med
            errs.append(ape)
            print(f"  {arm:5s} c{c:2d}(b={b:2d}): pred={pred:6.1f}ms "
                  f"meas={m.tpot_med:6.1f}ms ape={ape:.1%}")
    mape = sum(errs) / len(errs)
    print(f"  TPOT MAPE = {mape:.1%}  -> {'PASS' if mape <= 0.20 else 'FAIL'} (<=20%)")

    print("\n== 3) recommendations vs confirm ground truth ==")
    matrix = load_matrix(PROBE)
    rec_m = recommend_from_measured(matrix, family="mtp", quant_prefix=("P",))
    picks_m = {b.conc: b.k for b in rec_m.bands}
    expect = {1: 3, 16: 3, 64: 0}
    print(f"measured picks: {picks_m} -> {'PASS' if picks_m == expect else 'DIFF'}")
    for b in sorted(rec_m.bands, key=lambda x: x.conc):
        print(f"  c{b.conc}: {'K='+str(b.k) if b.k else 'disable'} "
              f"tps={b.expected_tps:.0f} base={b.baseline_tps:.0f} ({b.gain_vs_baseline:+.0%})")
    # model 模式按真实部署假设 batch=c（饱和）；c64 的 verify(256) 只有
    # full-cg 曲线外推（>128tok 无 full-cg 样本），该档预测置信度低
    rec_p = recommend_from_model(cm, am, [1, 16, 64], kmax=3)
    picks_p = {b.conc: b.k for b in rec_p.bands}
    print(f"model    picks: {picks_p} -> {'PASS' if picks_p == expect else 'DIFF'}")
    for b in rec_p.bands:
        print(f"  c{b.conc}: K={b.k} pred_tps={b.expected_tps:.0f} "
              f"base={b.baseline_tps:.0f} ({b.gain_vs_baseline:+.0%})")

    print("\n== 4) timing overhead (c16, timing on vs off) ==")
    on = parse_serve_file(PROBE / "mtp3" / "serve_c16.json")
    off = parse_serve_file(PROBE_OFF / "mtp3" / "serve_c16.json")
    print(f"on={on.out_tps:.1f} tok/s off={off.out_tps:.1f} tok/s "
          f"delta={(on.out_tps/off.out_tps-1):+.1%} (single-run noise floor ~5-7%)")

    print("\n== 5) end-to-end wall clock ==")
    log_local = ROOT / "orchestrator.log"
    if not log_local.exists():
        print("  (orchestrator.log not fetched — fetch separately)")
        return
    log = log_local.read_text(encoding="utf-8")
    ts = {}
    for line in log.splitlines():
        m = re.match(r"\[(\d\d:\d\d:\d\d)\] (.*)", line)
        if m:
            ts[m.group(2)] = m.group(1)
    for key in ("probe starting", "probe finished", "ALL DONE"):
        hit = [v for k, v in ts.items() if k.startswith(key)]
        if hit:
            print(f"  {key}: {hit[0]}")
    def hms(t: str) -> int:
        h, m, s = map(int, t.split(":"))
        return h * 3600 + m * 60 + s
    if "probe starting" in ts and "probe finished" in ts:
        d1 = hms(ts["probe finished"]) - hms(ts["probe starting"])
        print(f"  three-arm probe: {d1//60}m{d1%60}s (budget 15min)")
    if "probe starting" in ts and "ALL DONE" in ts:
        d2 = hms(ts["ALL DONE"]) - hms(ts["probe starting"])
        print(f"  incl. overhead eval: {d2//60}m{d2%60}s")


if __name__ == "__main__":
    main()
