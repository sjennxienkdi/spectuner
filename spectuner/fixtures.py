"""确认性矩阵夹具：装载 results/confirm_r{N}_{arm}/ 为 臂×并发 指标表。

用途（项目书 §四 验收）：SpecTuner 的推荐结果在目标硬件上的"手调最优"
真值就是这份数据——离线回归的夹具。
"""
from __future__ import annotations

import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from .parse_bench import parse_serve_file

DIR_RE = re.compile(r"^confirm_r(\d+)_(.+)$")
# 探针目录形态（probe/probe_run.sh 直接以臂名建目录，单轮）
PROBE_ARM_RE = re.compile(r"^(none|mtp\d+|dspark)$")

# 臂名 → (家族, 名义K)。家族决定它进不进 K 阶梯（mtp 系进；
# L0/L1/dspark 是调度表/别的 drafter，只作参照臂）。
ARM_INFO: dict[str, tuple[str, int | None]] = {
    "A0_none": ("none", 0),
    "A1_mtp3": ("mtp", 3),
    "A1_mtp2aoff": ("mtp", 2),      # async scheduling off，仍是固定 K=2
    "A2_L0": ("L0", None),
    "A3_L1v3": ("L1", None),
    "A6_dspark": ("dspark", 7),
    "B1_int4_none": ("none", 0),
    "B2_int4_dspark": ("dspark", 7),
}


def arm_info(arm: str) -> tuple[str, int | None]:
    if arm in ARM_INFO:
        return ARM_INFO[arm]
    if arm.startswith("P_"):  # 探针臂：P_none / P_mtp3 / P_dspark
        rest = arm[2:]
        if rest == "none":
            return ("none", 0)
        if rest.startswith("mtp") and rest[3:].isdigit():
            return ("mtp", int(rest[3:]))
        if rest == "dspark":
            return ("dspark", 7)
    return ("unknown", None)


@dataclass
class ArmCell:
    """一个 臂×并发 档的跨轮聚合（median + 范围）。"""
    arm: str
    conc: int
    out_tps_med: float
    out_tps_minmax: tuple[float, float]
    tpot_med: float
    ttft_med: float
    ttft_p99: float
    acc_len_med: float | None
    rounds: int
    pos_rates: list[float]

    @property
    def family(self) -> str:
        return arm_info(self.arm)[0]

    @property
    def k(self) -> int | None:
        return arm_info(self.arm)[1]


def load_matrix(results_root: Path | str) -> dict[tuple[str, int], ArmCell]:
    """扫描 confirm_* 目录 → {(arm, c): ArmCell}。"""
    root = Path(results_root)
    cells: dict[tuple[str, int], dict[int, dict[str, list[float]]]] = {}
    pos_rates_by: dict[tuple[str, int], dict[int, list[list[float]]]] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = DIR_RE.match(d.name)
        if m:
            rnd, arm = int(m.group(1)), m.group(2)
        elif PROBE_ARM_RE.match(d.name):
            rnd, arm = 1, f"P_{d.name}"
        else:
            continue
        for f in d.glob("serve_c*.json"):
            c = int(re.search(r"serve_c(\d+)\.json", f.name).group(1))
            r = parse_serve_file(f)
            if r is None:
                continue
            cells.setdefault((arm, c), {}).setdefault(rnd, {
                "out_tps": r.out_tps, "tpot_med": r.tpot_med,
                "ttft_med": r.ttft_med, "ttft_p99": r.ttft_p99,
                "acc_len": r.acc_len,
            })
            pos_rates_by.setdefault((arm, c), {}).setdefault(rnd, r.pos_rates)

    out: dict[tuple[str, int], ArmCell] = {}
    for (arm, c), by_round in cells.items():
        cols = {k: [v[k] for v in by_round.values()]
                for k in next(iter(by_round.values()))}
        acc = cols.get("acc_len") or [None]
        acc_vals = [a for a in acc if a is not None]
        # 分位接受率跨轮取均值（同臂同轮内一致）
        pr: list[float] = []
        for rnd_vals in pos_rates_by[(arm, c)].values():
            if len(rnd_vals) > len(pr):
                pr = rnd_vals
        n = len(by_round)
        out[(arm, c)] = ArmCell(
            arm=arm, conc=c,
            out_tps_med=statistics.median(cols["out_tps"]),
            out_tps_minmax=(min(cols["out_tps"]), max(cols["out_tps"])),
            tpot_med=statistics.median(cols["tpot_med"]),
            ttft_med=statistics.median(cols["ttft_med"]),
            ttft_p99=statistics.median(cols["ttft_p99"]),
            acc_len_med=statistics.median(acc_vals) if acc_vals else None,
            rounds=n,
            pos_rates=pr,
        )
    return out


def find_results_root() -> Path | None:
    """定位仓库 results/（含 confirm_agg.csv），供 selftest 直接使用。"""
    env = os.environ.get("SPECTUNER_RESULTS_ROOT")
    if env and Path(env).joinpath("confirm_agg.csv").exists():
        return Path(env)
    p = Path(__file__).resolve()
    for parent in p.parents:
        cand = parent / "results"
        if (cand / "confirm_agg.csv").exists():
            return cand
    return None
