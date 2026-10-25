"""依据报告生成（Markdown）。所有数字可溯源：measured 模式引用 confirm/probe
原始目录，model 模式引用 timing.jsonl 与成本表；方法学偏差明示（docs/08 §15 精神）。"""
from __future__ import annotations

from .recommend import Recommendation


def render(rec: Recommendation, source: str = "") -> str:
    lines: list[str] = []
    a = lines.append
    a("# SpecTuner 配置推荐报告")
    a("")
    a(f"- 模式：{'实测矩阵（measured）' if rec.mode == 'measured' else '成本模型（model）'}")
    if rec.model:
        a(f"- 模型：`{rec.model}`")
    if source:
        a(f"- 数据源：`{source}`")
    a("")

    a("## 每并发档建议")
    a("")
    a("| 并发 | 建议 | 预期 out_tps | vs 基线 | TPOT(ms) | TTFT 中位(ms) |")
    a("|---|---|---|---|---|---|")
    for b in sorted(rec.bands, key=lambda b: b.conc):
        act = "关投机" if b.action == "disable" else f"K={b.k}"
        a(f"| c{b.conc} | {act} | {b.expected_tps:.1f} | "
          f"{b.gain_vs_baseline:+.1%} | {b.tpot_ms:.1f} | "
          f"{b.ttft_med:.0f} |" if b.ttft_med else
          f"| c{b.conc} | {act} | {b.expected_tps:.1f} | "
          f"{b.gain_vs_baseline:+.1%} | {b.tpot_ms:.1f} | — |")
    a("")

    a("## vLLM 启动配置")
    a("")
    import json
    a("```json")
    a(json.dumps(rec.spec_config(), ensure_ascii=False, indent=2))
    a("```")
    thr = rec.disable_threshold()
    if thr is not None:
        a("")
        a(f"**运维提示**：并发 ≥ {thr} 时建议关闭投机解码（该档投机最优配置"
          f"对基线无 ≥5% 净增益；且投机在高并发同时伤害吞吐与 TTFT——确认矩阵 "
          f"A6/B2 c64 实测 TTFT 13-14s vs 基线 1.7-5.3s）。")
    a("")

    if rec.reference_arms:
        a("## 参照臂（不进 K 阶梯，供人工对照）")
        a("")
        a("| 臂 | " + " | ".join(f"c{c}" for c in
              sorted(next(iter(rec.reference_arms.values())))) + " |")
        arms = rec.reference_arms
        cs = sorted(next(iter(arms.values())))
        a("|---|" + "---|" * len(cs))
        for arm, by_c in sorted(arms.items()):
            a(f"| {arm} | " + " | ".join(f"{by_c.get(c, float('nan')):.1f}"
                                         for c in cs) + " |")
        a("")

    if rec.notes:
        a("## 说明与限制")
        a("")
        for n in rec.notes:
            a(f"- {n}")
        a("- 调度表的 batch-size 与并发档的映射假设 decode 批大小 ≈ 并发"
          "（饱和时成立；低负载排队时批更小，K 建议向更高档靠拢）。")
        a("- measured 模式的数字即实测中位数（n≥3 轮）；model 模式为预测值，"
          "MAPE 验收目标 ≤ 20%（项目书 §四），未验证前注明置信度。")
        a("")
    return "\n".join(lines)
