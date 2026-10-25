"""E[emitted | K] 估计：分位接受率 → 各 K 的每轮期望产出 token 数。

口径：e(K) 含 bonus（每轮至少 1 个 target token），与 bench 的
"Acceptance length" 同义（对 results/confirm_r1_A1_mtp3/serve_c1.json
核对：Accepted/Drafts + 1 = 2560/1543 + 1 = 2.66 ✓）。

方法与已知偏差（docs/08 §2 H1 / 项目书 §三）：
- bench 的 Position N 是边际接受率；若按条件概率链乘会系统性低估联合接受
  （mtp3 c1 实测：链乘 e(3)=2.23 vs 实测 2.66）。
- 因此拟合用"锚定校准"：以实测 (K_meas, e_meas) 为锚，对链乘值做比例缩放；
  两个实测锚（K=1、K=3 探针各一）时可做分段校准。
"""
from __future__ import annotations

from dataclasses import dataclass


def chain_emitted(K: int, pos_rates: list[float]) -> float:
    """链乘估计：e = 1 + Σ_{i<K} Π_{j≤i} p_j（pos_rates 不足时按已有位置截断）。"""
    e, surv = 1.0, 1.0
    for i in range(min(K, len(pos_rates))):
        surv *= pos_rates[i]
        e += surv
    return e


@dataclass
class AcceptanceModel:
    pos_rates: list[float]
    anchors: dict[int, float]          # K → 实测 e(K)（bench acc_len）

    def raw(self, K: int) -> float:
        return chain_emitted(K, self.pos_rates)

    def e(self, K: int) -> float:
        """校准后的 e(K)：有实测锚用锚；两锚之间按链乘偏差比例缩放。"""
        if K in self.anchors:
            return self.anchors[K]
        ks = sorted(self.anchors)
        if not ks:
            return self.raw(K)
        if K < ks[0]:
            base_k = ks[0]
        elif K > ks[-1]:
            base_k = ks[-1]
        else:  # 落在两锚之间：取不超过 K 的最近锚
            base_k = max(k for k in ks if k < K)
        ratio = (self.anchors[base_k] - 1.0) / max(self.raw(base_k) - 1.0, 1e-9)
        return 1.0 + (self.raw(K) - 1.0) * ratio

    @classmethod
    def from_bench(cls, k_meas: int, e_meas: float, pos_rates: list[float]) -> "AcceptanceModel":
        return cls(pos_rates=pos_rates, anchors={k_meas: e_meas})
