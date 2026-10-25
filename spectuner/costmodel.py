"""分项成本模型（docs/08 §8.2）：T_step(batch, K) = T_draft + T_verify + T_sampler + T_other。

数据源：bench/patch_timing.py 落盘的 timing.jsonl，每步一行：
  {step, num_reqs, num_tokens, draft_sum, draft_k_max, full_cg,
   gpu_ms:{fwd,sampler,draft}, host_ms:{...}, step_gap_ms}

模型形态（H1/H2 机制结论决定：接受率跨并发稳定、高并发回退是成本主导）：
- T_verify(tokens)：按 num_tokens 分组取中位（decode、full graph）；线性外推
- T_draft(batch, K)：按 (num_reqs, K) 分组取中位；稀疏时线性回归
  T ≈ a + b·batch + c·K（MTP 草稿是 K 次小 forward，一阶近似可加）
- T_sampler(tokens)：同 T_verify
- T_other(batch)：K=0 探针的 step_gap − verify − sampler（调度/提交/间隙，
  端到端 cadence 与 GPU 分项之差，含 async overlap 时不重复计——用于预测
  cadence 而非分项求和）
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path


def load_timing(path: Path | str) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def is_decode(rec: dict) -> bool:
    """decode 步判别：full graph 且 tokens 与请求数同量级（prefill 的
    num_tokens 远大于 num_reqs）。"""
    if not rec.get("full_cg"):
        return False
    n, t = rec.get("num_reqs", 0), rec.get("num_tokens", 0)
    return n > 0 and t <= n * 16  # K≤7 时 tokens ≤ 8n；阈值放宽到 16 防边界


def infer_k(rec: dict) -> int | None:
    """从计时记录推断该步的名义 K：tokens/reqs − 1（uniform K 时精确）。"""
    n, t = rec.get("num_reqs", 0), rec.get("num_tokens", 0)
    if n and t % n == 0:
        return t // n - 1
    return rec.get("draft_k_max")


def _group_median(pairs: list[tuple[int, float]]) -> dict[int, float]:
    g: dict[int, list[float]] = {}
    for k, v in pairs:
        g.setdefault(k, []).append(v)
    return {k: statistics.median(v) for k, v in g.items()}


def _lin(xs: list[tuple[int, int]], ys: list[float]) -> tuple[float, float, float]:
    """最小二乘 y ≈ a + b·x0 + c·x1（x0=batch, x1=K）。"""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n < 3:
        return (statistics.median(ys), 0.0, 0.0)
    s0 = s1 = s2 = s3 = s4 = s5 = s6 = 0.0
    s_y = statistics.fmean(ys)
    for (x0, x1), y in zip(xs, ys):
        s0 += 1; s1 += x0; s2 += x1; s3 += x0 * x0; s4 += x0 * x1; s5 += x1 * x1
        s6 += x0 * y
    s7 = sum(x1 * y for (_, x1), y in zip(xs, ys))
    sy = sum(ys)
    # 3×3 正规方程 [n,Σx0,Σx1; ·,Σx0²,Σx0x1; ·,·,Σx1²] a = [Σy,Σx0y,Σx1y]
    m = [[s0, s1, s2, sy], [s1, s3, s4, s6], [s2, s4, s5, s7]]
    # 高斯消元（奇异时回退常数）
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(m[r][i]))
        if abs(m[p][i]) < 1e-12:
            return (s_y, 0.0, 0.0)
        m[i], m[p] = m[p], m[i]
        for r in range(3):
            if r != i:
                f = m[r][i] / m[i][i]
                m[r] = [a - f * b for a, b in zip(m[r], m[i])]
    return (m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2])


@dataclass
class CostModel:
    verify_by_tokens: dict[int, float] = field(default_factory=dict)
    sampler_by_tokens: dict[int, float] = field(default_factory=dict)
    draft_by_bk: dict[tuple[int, int], float] = field(default_factory=dict)
    other_by_batch: dict[int, float] = field(default_factory=dict)
    draft_lin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ---- 查询 ----
    def verify_ms(self, tokens: int) -> float:
        return _lookup_lin(self.verify_by_tokens, tokens)

    def sampler_ms(self, tokens: int) -> float:
        return _lookup_lin(self.sampler_by_tokens, tokens)

    def draft_ms(self, batch: int, k: int) -> float:
        if (batch, k) in self.draft_by_bk:
            return self.draft_by_bk[(batch, k)]
        a, b, c = self.draft_lin
        return max(a + b * batch + c * k, 0.0)

    def other_ms(self, batch: int) -> float:
        return _lookup_lin(self.other_by_batch, batch)

    def step_ms(self, batch: int, k: int) -> float:
        """K=0 时 cadence ≈ other + verify(batch) + sampler(batch)。"""
        if k <= 0:
            return self.other_ms(batch) + self.verify_ms(batch) + self.sampler_ms(batch)
        return (
            self.draft_ms(batch, k)
            + self.verify_ms(batch * (k + 1))
            + self.sampler_ms(batch * (k + 1))
            + self.other_ms(batch)
        )

    # ---- 拟合 ----
    @classmethod
    def fit(cls, records: list[dict]) -> "CostModel":
        dec = [r for r in records if is_decode(r) and r.get("gpu_ms")]
        m = cls()
        m.verify_by_tokens = _group_median([
            (r["num_tokens"], r["gpu_ms"].get("fwd", 0.0))
            for r in dec if "fwd" in r["gpu_ms"]
        ])
        m.sampler_by_tokens = _group_median([
            (r["num_tokens"], r["gpu_ms"].get("sampler", 0.0))
            for r in dec if "sampler" in r["gpu_ms"]
        ])
        draft_pts = [
            ((r["num_reqs"], infer_k(r) or 0), r["gpu_ms"].get("draft", 0.0))
            for r in dec if "draft" in r["gpu_ms"] and infer_k(r) is not None
        ]
        m.draft_by_bk = _group_median(draft_pts)
        if draft_pts:
            m.draft_lin = _lin([p for p, _ in draft_pts],
                               [v for _, v in draft_pts])
        # T_other：用端到端 cadence（step_gap）扣掉 GPU 分项。优先 K=0 记录
        # （纯 baseline cadence）；没有则用各 K 记录统一扣。
        gaps = [
            (r["num_reqs"], r.get("step_gap_ms", 0.0)
             - r["gpu_ms"].get("fwd", 0.0) - r["gpu_ms"].get("sampler", 0.0)
             - r["gpu_ms"].get("draft", 0.0))
            for r in dec if r.get("step_gap_ms")
            and infer_k(r) in (0, None)
        ]
        if not gaps:
            gaps = [
                (r["num_reqs"], r.get("step_gap_ms", 0.0)
                 - r["gpu_ms"].get("fwd", 0.0) - r["gpu_ms"].get("sampler", 0.0)
                 - r["gpu_ms"].get("draft", 0.0))
                for r in dec if r.get("step_gap_ms")
            ]
        m.other_by_batch = {b: max(v, 0.0)
                            for b, v in _group_median(gaps).items()}
        return m

    def mape_against(self, records: list[dict]) -> float | None:
        """对 decode 记录的端到端 step_gap 预测 MAPE（模型自检）。"""
        errs, n = 0.0, 0
        for r in records:
            if not (is_decode(r) and r.get("step_gap_ms")):
                continue
            k = infer_k(r) or 0
            pred = self.step_ms(r["num_reqs"], k)
            if pred <= 0:
                continue
            errs += abs(pred - r["step_gap_ms"]) / r["step_gap_ms"]
            n += 1
        return errs / n if n else None


def _lookup_lin(table: dict[int, float], x: int) -> float:
    """表内取值；表外线性外推（两端点斜率）。"""
    if not table:
        return 0.0
    xs = sorted(table)
    if x in table:
        return table[x]
    if x < xs[0]:
        lo, hi = xs[0], xs[min(1, len(xs) - 1)]
    else:
        lo, hi = xs[-2] if len(xs) > 1 else xs[-1], xs[-1]
    if hi == lo:
        return table[lo]
    slope = (table[hi] - table[lo]) / (hi - lo)
    return max(table[lo] + slope * (x - lo), 0.0)
