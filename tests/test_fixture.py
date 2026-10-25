"""confirm 矩阵真值回归（验收夹具，项目书 §四 的"手调最优"）。

无 results/ 时自动 skip（云端/独立 repo 场景）。
"""
import pytest

from spectuner.acceptance import chain_emitted
from spectuner.fixtures import find_results_root, load_matrix
from spectuner.recommend import recommend_from_measured

pytestmark = pytest.mark.skipif(find_results_root() is None,
                                reason="confirm results not found")


@pytest.fixture(scope="module")
def matrix():
    return load_matrix(find_results_root())


def test_known_medians(matrix):
    """抽查 confirm_agg.md 的冻结数字（2026-10-24 确认性矩阵）。"""
    assert matrix[("A0_none", 64)].out_tps_med == pytest.approx(901.1, abs=1.0)
    assert matrix[("A1_mtp3", 1)].out_tps_med == pytest.approx(72.8, abs=1.0)
    assert matrix[("B2_int4_dspark", 1)].out_tps_med == pytest.approx(95.4, abs=1.0)
    assert matrix[("A0_none", 64)].rounds == 3


def test_acceptance_fields(matrix):
    c = matrix[("A1_mtp3", 1)]
    assert c.acc_len_med is not None and 2.3 < c.acc_len_med < 2.8
    assert len(c.pos_rates) == 3
    # 口径一致性：分位链乘 ≤ 实测（联合接受偏差，acceptance.py 的立论）
    assert chain_emitted(3, c.pos_rates) < c.acc_len_med


def test_recommend_fp8_group(matrix):
    """FP8 组推荐：c1/c16 → K=3，c64 → 关投机（L1 判负后的工程结论）。"""
    rec = recommend_from_measured(matrix, family="mtp", quant_prefix=("A",))
    got = {b.conc: b.k for b in rec.bands}
    assert got == {1: 3, 16: 3, 64: 0}
    assert rec.disable_threshold() == 64
    # A2_L0（82.9 的"全能王"）作为参照臂保留，不进 K 阶梯
    assert rec.reference_arms["A2_L0"][1] == pytest.approx(82.9, abs=1.0)
