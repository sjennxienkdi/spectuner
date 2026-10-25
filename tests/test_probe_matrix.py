"""探针目录形态（probe/probe_run.sh 产出）也能直接喂 recommend。"""
import pytest

from spectuner.fixtures import load_matrix
from spectuner.recommend import recommend_from_measured

SERVE_TXT = """Benchmark duration (s):              60.0
Output token throughput (tok/s):         {tps}
Median TPOT (ms):                          12.0
Median TTFT (ms):                         300.0
P99 TTFT (ms):                            400.0
Acceptance rate (%):                     55.30
Acceptance length:                       2.66
Drafts:                                  1543
Accepted tokens:                         2560
"""


@pytest.fixture
def probe_dir(tmp_path):
    tps = {("none", 1): 44, ("none", 16): 446, ("none", 64): 901,
           ("mtp3", 1): 73, ("mtp3", 16): 544, ("mtp3", 64): 782,
           ("mtp1", 1): 60, ("mtp1", 16): 500, ("mtp1", 64): 750}
    for (arm, c), v in tps.items():
        d = tmp_path / arm
        d.mkdir(exist_ok=True)
        (d / f"serve_c{c}.json").write_text(SERVE_TXT.format(tps=v),
                                            encoding="utf-8")
    return tmp_path


def test_probe_dirs_load_and_recommend(probe_dir):
    m = load_matrix(probe_dir)
    got = {b.conc: b.k for b in recommend_from_measured(
        m, family="mtp", quant_prefix=("P",)).bands}
    assert got == {1: 3, 16: 3, 64: 0}
