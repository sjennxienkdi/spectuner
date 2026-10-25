"""解析 `vllm bench serve` 的 stdout 文本表（0.32.0 无 --output-json，口径见 docs/BASELINE.md §5）。

字段语义（对 results/confirm_* 实测核对）：
- Acceptance length = 每轮 emitted token 数 = 接受草稿数 + 1（bonus）
- Drafts = 验证轮数（名字有误导性）
- Accepted tokens = 接受的草稿 token 总数（不含 bonus）
- Acceptance rate (%) = 分位接受率的算术平均
- Position N = 位置 N 的接受率（%）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

FLOAT_RE = re.compile(r"^([^:]+?):\s+(-?[\d.]+(?:e-?\d+)?)\s*$", re.M)
POS_RE = re.compile(r"^\s+Position (\d+):\s+([\d.]+)", re.M)

WANT = {
    "Output token throughput (tok/s)": "out_tps",
    "Request throughput (req/s)": "req_tps",
    "Benchmark duration (s)": "dur_s",
    "Mean TTFT (ms)": "ttft_mean",
    "Median TTFT (ms)": "ttft_med",
    "P99 TTFT (ms)": "ttft_p99",
    "Mean TPOT (ms)": "tpot_mean",
    "Median TPOT (ms)": "tpot_med",
    "P99 TPOT (ms)": "tpot_p99",
    "Mean ITL (ms)": "itl_mean",
    "P99 ITL (ms)": "itl_p99",
    "Acceptance rate (%)": "acc_rate",
    "Acceptance length": "acc_len",
    "Drafts": "drafts",
    "Accepted tokens": "acc_tokens",
}


@dataclass
class BenchResult:
    """一次 bench serve 运行的解析结果。pos_rates 为小数概率（已除 100）。"""
    out_tps: float = 0.0
    tpot_med: float = 0.0
    ttft_med: float = 0.0
    ttft_p99: float = 0.0
    acc_len: float | None = None
    acc_rate: float | None = None
    drafts: float | None = None
    acc_tokens: float | None = None
    pos_rates: list[float] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def k_measured(self) -> int | None:
        """该次运行的名义 K = 分位接受率个数 - 1。"""
        return len(self.pos_rates) - 1 if self.pos_rates else None


def parse_serve_file(path: Path) -> BenchResult | None:
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    kv: dict[str, float] = {}
    for m in FLOAT_RE.finditer(txt):
        key = m.group(1).strip()
        if key in WANT:
            kv[WANT[key]] = float(m.group(2))
    if "out_tps" not in kv:
        return None
    r = BenchResult(raw=kv)
    for f in ("out_tps", "tpot_med", "ttft_med", "ttft_p99", "acc_len",
              "acc_rate", "drafts", "acc_tokens"):
        if f in kv:
            setattr(r, f, kv[f])
    r.pos_rates = [float(m.group(2)) / 100.0 for m in POS_RE.finditer(txt)]
    return r
