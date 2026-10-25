"""推荐引擎：每并发档最优 K + dynamic 调度表 + 关投机阈值。

两种模式：
1. measured（MVP 主路径）：直接用实测 臂×并发 吞吐表（confirm 矩阵形态），
   每档取 K 阶梯最优；最优投机档达不到 (1−margin)·基线 则判 disable。
2. model（探针升级路径）：CostModel + AcceptanceModel 预测未测 (c, K)，
   同一套判据。探针只跑 K=0/1/3 三臂即可覆盖 K∈{0..3} 全表。

判据（项目书 §四 / docs/08 §11 预注册精神的工程化转译）：
- 档内最优：max 预测/实测 out_tps（K=0 基线在阶梯内直接参与竞争）
- 关投机：argmax 胜出的 K>0 档，吞吐增益还须 ≥ min_gain（默认 5%）才保留
  ——低于噪声地板的"赢"不值得开投机（docs/08 §11：差异 <5% 记不确定；
  且投机在高并发有 TTFT 排队风险，保守取向）
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .acceptance import AcceptanceModel
from .costmodel import CostModel


@dataclass
class BandRec:
    conc: int
    k: int                          # 0 = 关投机
    expected_tps: float
    baseline_tps: float
    tpot_ms: float | None = None
    ttft_med: float | None = None   # measured 模式可带实测
    action: str = "K"               # "K" | "disable"
    gain_vs_baseline: float = 0.0


@dataclass
class Recommendation:
    mode: str                                  # measured | model
    quant: str = ""
    model: str = ""
    bands: list[BandRec] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reference_arms: dict[str, dict[int, float]] = field(default_factory=dict)

    @property
    def kmax(self) -> int:
        return max((b.k for b in self.bands), default=0)

    def l0_table(self) -> list[list[int]]:
        """vLLM 0.32 num_speculative_tokens_per_batch_size 形态的调度表。
        batch-size ≈ 并发档（饱和 decode 时成立，报告须注明该映射假设）。"""
        bands = sorted(self.bands, key=lambda b: b.conc)
        table: list[list[int]] = []
        for i, b in enumerate(bands):
            k = max(b.k, 1)  # 表内不能表达 K=0；disable 档由阈值建议承担
            lo = 1 if i == 0 else bands[i - 1].conc + 1
            hi = b.conc
            if table and table[-1][2] == k and table[-1][1] == lo - 1:
                table[-1][1] = hi
            else:
                table.append([lo, hi, k])
        return table

    def spec_config(self, base_k: int | None = None) -> dict:
        """输出 --speculative-config JSON（mtp 家族）。"""
        cfg: dict = {"method": "mtp", "num_speculative_tokens": base_k or self.kmax}
        if len({b.k for b in self.bands}) > 1:
            cfg["num_speculative_tokens_per_batch_size"] = self.l0_table()
        return cfg

    def disable_threshold(self) -> int | None:
        """首个判 disable 的并发档；全开返回 None。"""
        for b in sorted(self.bands, key=lambda b: b.conc):
            if b.action == "disable":
                return b.conc
        return None


# ---------------- measured 模式 ----------------

def recommend_from_measured(
    matrix: dict,                      # fixtures.load_matrix 的 {(arm,c): ArmCell}
    family: str = "mtp",
    quant_prefix: tuple[str, ...] = ("A",),
    min_gain: float = 0.05,
    model: str = "",
) -> Recommendation:
    rec = Recommendation(mode="measured", model=model)
    ladder: dict[int, dict[int, tuple[float, BandRec | None]]] = {}
    baseline: dict[int, tuple[float, BandRec | None]] = {}
    for (arm, c), cell in matrix.items():
        if not arm.startswith(quant_prefix):
            continue
        fam, k = cell.family, cell.k
        if fam == family and k is not None:
            ladder.setdefault(c, {})[k] = (cell.out_tps_med, cell)
        elif fam == "none":
            baseline[c] = (cell.out_tps_med, cell)
        else:
            rec.reference_arms.setdefault(arm, {})[c] = cell.out_tps_med
    for c in sorted(ladder):
        if c not in baseline:
            rec.notes.append(f"c={c} 无 K=0 基线档，跳过")
            continue
        base_tps, base_cell = baseline[c]
        best_k, best_tps, best_cell = 0, base_tps, base_cell
        for k, (tps, cell) in sorted(ladder[c].items()):
            if tps > best_tps:
                best_k, best_tps, best_cell = k, tps, cell
        if best_k > 0 and best_tps < (1 + min_gain) * base_tps:
            best_k, best_tps, best_cell = 0, base_tps, base_cell
        b = BandRec(
            conc=c, k=best_k, expected_tps=best_tps, baseline_tps=base_tps,
            tpot_ms=best_cell.tpot_med, ttft_med=best_cell.ttft_med,
            gain_vs_baseline=best_tps / base_tps - 1,
            action="disable" if best_k == 0 else "K",
        )
        rec.bands.append(b)
    return rec


# ---------------- model 模式 ----------------

def predict_tps(c: int, k: int, cm: CostModel, am: AcceptanceModel) -> float:
    """out_tps = c · e(K) / T_step(batch≈c, K)（tok/s，T 为 ms）。"""
    t = cm.step_ms(c, k)
    return 1000.0 * c * am.e(k) / t if t > 0 else 0.0


def recommend_from_model(
    cm: CostModel,
    am: AcceptanceModel,
    concs: list[int],
    kmax: int = 3,
    min_gain: float = 0.05,
    model: str = "",
) -> Recommendation:
    rec = Recommendation(mode="model", model=model)
    for c in sorted(concs):
        base = predict_tps(c, 0, cm, am)
        best_k, best_tps = 0, base
        for k in range(1, kmax + 1):
            tps = predict_tps(c, k, cm, am)
            if tps > best_tps:
                best_k, best_tps = k, tps
        if best_k > 0 and best_tps < (1 + min_gain) * base:
            best_k, best_tps = 0, base
        step_ms = cm.step_ms(c, best_k)
        rec.bands.append(BandRec(
            conc=c, k=best_k, expected_tps=best_tps, baseline_tps=base,
            tpot_ms=step_ms / am.e(best_k) if best_k > 0 else step_ms,
            gain_vs_baseline=best_tps / base - 1 if base else 0.0,
            action="disable" if best_k == 0 else "K",
        ))
    return rec
