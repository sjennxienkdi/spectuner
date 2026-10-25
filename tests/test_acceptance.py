import pytest

from spectuner.acceptance import AcceptanceModel, chain_emitted

# results/confirm_r1_A1_mtp3/serve_c1.json 实测：positions 69.73/54.18/42.00%
RATES = [0.6973, 0.5418, 0.4200]
E3_MEAS = 2.66


def test_chain_basic():
    assert chain_emitted(0, RATES) == 1.0
    assert chain_emitted(1, RATES) == pytest.approx(1.6973, abs=1e-4)
    assert chain_emitted(3, RATES) == pytest.approx(
        1 + 0.6973 + 0.6973 * 0.5418 + 0.6973 * 0.5418 * 0.42, abs=1e-4)


def test_chain_truncated_rates():
    """位置率不足 K 时按已有位置截断（不外推）。"""
    assert chain_emitted(5, RATES[:2]) == pytest.approx(
        1 + 0.6973 + 0.6973 * 0.5418, abs=1e-4)


def test_chain_underestimates_joint():
    """链乘（按边际率）系统性低估联合接受——校准的存在理由。"""
    assert chain_emitted(3, RATES) < E3_MEAS


def test_calibration_anchor_exact():
    am = AcceptanceModel.from_bench(3, E3_MEAS, RATES)
    assert am.e(3) == E3_MEAS


def test_calibration_interpolates():
    am = AcceptanceModel.from_bench(3, E3_MEAS, RATES)
    e1, e2 = am.e(1), am.e(2)
    assert 1.0 < e1 < e2 < E3_MEAS
    # 校准比例 >1（链乘低估），e(1) 应高于原始链乘
    assert e1 > chain_emitted(1, RATES)


def test_calibration_two_anchors():
    am = AcceptanceModel(pos_rates=RATES, anchors={1: 1.90, 3: E3_MEAS})
    assert am.e(1) == 1.90
    assert am.e(3) == E3_MEAS
    assert 1.90 < am.e(2) < E3_MEAS


def test_no_rates_returns_bonus_only():
    am = AcceptanceModel(pos_rates=[], anchors={})
    assert am.e(3) == 1.0
