"""合成计时数据 → 拟合恢复真值（本地无 GPU 时成本模型的全部可验证性）。"""
import pytest

from spectuner.costmodel import CostModel, _lookup_lin, infer_k, is_decode

BATCHES = [1, 4, 8, 16, 32, 64]
KS = [0, 1, 3]


def t_verify(t):  return 2.0 + 0.10 * t
def t_sampler(t): return 0.5 + 0.01 * t
def t_draft(b, k): return 0.6 + 0.4 * k * b  # MTP 一阶近似：K 次小 forward
def t_other(b):   return 1.5 + 0.01 * b


def make_records():
    recs = []
    step = 0
    for b in BATCHES:
        for k in KS:
            for _ in range(3):  # 每组 3 步取中位
                step += 1
                tokens = b * (k + 1)
                draft = t_draft(b, k) if k > 0 else 0.0
                gap = draft + t_verify(tokens) + t_sampler(tokens) + t_other(b)
                recs.append({
                    "step": step, "num_reqs": b, "num_tokens": tokens,
                    "full_cg": True, "gpu_ms": {
                        "fwd": t_verify(tokens), "sampler": t_sampler(tokens),
                        "draft": draft,
                    },
                    "host_ms": {}, "step_gap_ms": gap,
                })
    return recs


def test_infer_k_and_decode_filter():
    r = make_records()[0]
    assert is_decode(r)
    assert infer_k({"num_reqs": 16, "num_tokens": 64}) == 3
    assert infer_k({"num_reqs": 16, "num_tokens": 16}) == 0
    assert not is_decode({"num_reqs": 2, "num_tokens": 2048, "full_cg": True})


def test_fit_recovers_step_ms():
    cm = CostModel.fit(make_records())
    for b in BATCHES:
        for k in KS:
            truth = (t_draft(b, k) if k else 0) + t_verify(b * (k + 1)) \
                + t_sampler(b * (k + 1)) + t_other(b)
            assert cm.step_ms(b, k) == pytest.approx(truth, rel=0.02), (b, k)


def test_fit_predicts_unmeasured_combo():
    """探针只测 K∈{0,1,3}，K=2 靠线性内插——预测偏差须可接受。"""
    cm = CostModel.fit(make_records())
    b, k = 16, 2
    truth = t_draft(b, k) + t_verify(b * 3) + t_sampler(b * 3) + t_other(b)
    assert cm.step_ms(b, k) == pytest.approx(truth, rel=0.15)


def test_mape_selfcheck():
    recs = make_records()
    cm = CostModel.fit(recs)
    assert cm.mape_against(recs) < 0.05


def test_lookup_lin_extrapolation():
    table = {8: 3.0, 32: 7.0}
    assert _lookup_lin(table, 8) == 3.0
    assert _lookup_lin(table, 32) == 7.0
    # 斜率 4/24=1/6 per token
    assert _lookup_lin(table, 64) == pytest.approx(7.0 + 32 / 6)
    assert _lookup_lin(table, 1) == pytest.approx(3.0 - 7 / 6)
    assert _lookup_lin(table, 16) == pytest.approx(3.0 + 8 / 6)
    assert _lookup_lin({}, 5) == 0.0
