"""推荐引擎：合成矩阵验证档内择优、disable 判据、L0 表生成。"""
import pytest

from spectuner.fixtures import ArmCell
from spectuner.recommend import recommend_from_measured


def cell(arm, c, tps):
    return ArmCell(arm=arm, conc=c, out_tps_med=tps, out_tps_minmax=(tps, tps),
                   tpot_med=20.0, ttft_med=300.0, ttft_p99=400.0,
                   acc_len_med=None, rounds=3, pos_rates=[])


def make_matrix():
    """数值取自 confirm 矩阵量级：c1/c16 投机赢、c64 投机输到 95% 线下。"""
    tps = {
        ("A0_none", 1): 44, ("A1_mtp1", 1): 60, ("A1_mtp2", 1): 70, ("A1_mtp3", 1): 73,
        ("A0_none", 16): 446, ("A1_mtp1", 16): 500, ("A1_mtp2", 16): 530, ("A1_mtp3", 16): 544,
        ("A0_none", 64): 901, ("A1_mtp1", 64): 750, ("A1_mtp2", 64): 809, ("A1_mtp3", 64): 782,
        ("A2_L0", 1): 83, ("A2_L0", 16): 537, ("A2_L0", 64): 796,
        ("A6_dspark", 1): 81, ("A6_dspark", 16): 433, ("A6_dspark", 64): 530,
    }
    return {(a, c): cell(a, c, v) for (a, c), v in tps.items()}


def test_picks_and_disable():
    rec = recommend_from_measured(make_matrix())
    got = {b.conc: (b.k, b.action) for b in rec.bands}
    assert got == {1: (3, "K"), 16: (3, "K"), 64: (0, "disable")}
    assert rec.disable_threshold() == 64
    # 参照臂不进阶梯
    assert "A2_L0" in rec.reference_arms and "A6_dspark" in rec.reference_arms


def test_min_gain_guard():
    """c64 基线 901 绝对占优 → 任何 margin 都 disable。另构造"微弱赢"档：
    spec 455 vs 基线 446（+2%<5% 噪声守卫）→ 默认 disable，min_gain=0.01 保。"""
    m = make_matrix()
    rec = recommend_from_measured(m, min_gain=0.05)
    assert rec.bands[2].action == "disable"
    rec2 = recommend_from_measured(m, min_gain=0.5)  # 放宽守卫也不改绝对劣势
    assert rec2.bands[2].action == "disable"

    m2 = dict(make_matrix())
    m2[("A1_mtp3", 16)] = cell("A1_mtp3", 16, 455)  # 446 的 1.02 倍
    keep = recommend_from_measured(m2, min_gain=0.05)
    assert {b.conc: b.k for b in keep.bands} == {1: 3, 16: 0, 64: 0}  # c16 被守卫拦下
    loose = recommend_from_measured(m2, min_gain=0.01)
    assert {b.conc: b.k for b in loose.bands} == {1: 3, 16: 3, 64: 0}


def test_l0_table_merges_and_floor():
    rec = recommend_from_measured(make_matrix())
    # c1/c16 均 K=3 → 合并；c64 disable → 表内下限 K=1 + 阈值建议兜底
    assert rec.l0_table() == [[1, 16, 3], [17, 64, 1]]
    cfg = rec.spec_config()
    assert cfg["method"] == "mtp"
    assert cfg["num_speculative_tokens"] == 3
    assert cfg["num_speculative_tokens_per_batch_size"] == [[1, 16, 3], [17, 64, 1]]


def test_single_k_no_table():
    m = {(a, c): cell for (a, c), cell in make_matrix().items()
         if a in ("A0_none", "A1_mtp3")}
    rec = recommend_from_measured(m)
    assert rec.bands[0].k == 3 and rec.bands[2].k == 0
    assert "num_speculative_tokens_per_batch_size" in rec.spec_config()


def test_no_baseline_notes():
    m = {(a, c): cell for (a, c), cell in make_matrix().items() if a != "A0_none"}
    rec = recommend_from_measured(m)
    assert not rec.bands
    assert any("无 K=0 基线" in n for n in rec.notes)
