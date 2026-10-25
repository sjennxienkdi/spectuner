"""SpecTuner CLI。

子命令：
  probe      打印 GPU 探针的执行计划（下次窗口照跑 probe/probe_run.sh）
  recommend  从实测矩阵出推荐（--matrix results/，confirm_* 目录形态）
  fit        从 timing.jsonl 拟合成本模型（探针数据到位后）
  report     渲染 Markdown 报告
  selftest   离线回归：confirm 矩阵夹具 + 合成数据（不依赖 pytest）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .acceptance import AcceptanceModel
from .costmodel import CostModel, load_timing
from .fixtures import find_results_root, load_matrix
from .recommend import recommend_from_measured, recommend_from_model
from .report import render


def _cmd_recommend(args) -> int:
    matrix = load_matrix(args.matrix)
    if not matrix:
        print(f"no confirm_* results under {args.matrix}", file=sys.stderr)
        return 1
    rec = recommend_from_measured(
        matrix, family=args.family,
        quant_prefix=tuple(args.quant_prefix),
        min_gain=args.min_gain, model=args.model,
    )
    print(json.dumps(rec.spec_config(), ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(render(rec, source=str(args.matrix)),
                                  encoding="utf-8")
        print(f"report -> {args.out}", file=sys.stderr)
    return 0


def _cmd_fit(args) -> int:
    recs: list[dict] = []
    for p in args.timing:
        recs.extend(load_timing(p))
    cm = CostModel.fit(recs)
    mape = cm.mape_against(recs)
    out = {
        "records": len(recs),
        "verify_by_tokens": cm.verify_by_tokens,
        "sampler_by_tokens": cm.sampler_by_tokens,
        "draft_by_bk": {f"{b}x{k}": v for (b, k), v in cm.draft_by_bk.items()},
        "other_by_batch": cm.other_by_batch,
        "draft_lin": {"a": cm.draft_lin[0], "b_batch": cm.draft_lin[1],
                      "c_k": cm.draft_lin[2]},
        "step_gap_mape": mape,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


def _cmd_probe(args) -> int:
    here = Path(__file__).resolve().parent.parent / "probe" / "probe_run.sh"
    print(f"""GPU 探针计划（~10-15 分钟，PRO 6000 + Qwen3.8-27B-FP8 实测节奏）：
1. 对 site-packages 的 model_runner.py 打计时补丁（幂等，备份自动留）：
   python3 bench/patch_timing.py <venv>/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py
2. 依次起 target-only / K=3 /（可选）K=1 三个短探针臂（各 c1/c16/c64 小样本），
   环境变量 VLLM_TIMING_OUT 指向结果目录：
   QUANT={args.quant} SPEC=none CONCURRENCY="1 16 64" NUM_PROMPTS_OVERRIDE="8 24 48" \\
     OUT_DIR=probe_none VLLM_TIMING_OUT=<dir>/timing_none.jsonl bash {here}
   QUANT={args.quant} SPEC=mtp3 ... OUT_DIR=probe_k3 ...
3. 回收数据后本地：
   spectuner fit --timing <dir>/timing_k3.jsonl
   spectuner recommend --matrix <dir>   # measured 模式先行
""")
    return 0


def _cmd_selftest(args) -> int:
    ok = True
    root = find_results_root()
    if root is None:
        print("results root not found (set SPECTUNER_RESULTS_ROOT)", file=sys.stderr)
        return 1
    matrix = load_matrix(root)
    rec = recommend_from_measured(matrix, family="mtp", quant_prefix=("A",))
    got = {b.conc: b.k for b in rec.bands}
    expect = {1: 3, 16: 3, 64: 0}
    for c, k in expect.items():
        mark = "PASS" if got.get(c) == k else "FAIL"
        ok &= got.get(c) == k
        print(f"[{mark}] c{c}: recommend K={got.get(c)} (expect {k})")
    thr = rec.disable_threshold()
    mark = "PASS" if thr == 64 else "FAIL"
    ok &= thr == 64
    print(f"[{mark}] disable_threshold={thr} (expect 64)")
    print("selftest", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="spectuner")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="打印 GPU 探针执行计划")
    p.add_argument("--quant", default="fp8")
    p.set_defaults(fn=_cmd_probe)

    p = sub.add_parser("recommend", help="从实测矩阵出推荐")
    p.add_argument("--matrix", required=True)
    p.add_argument("--family", default="mtp")
    p.add_argument("--quant-prefix", nargs="+", default=["A", "P"],
                   help="臂名前缀过滤（A=confirm 矩阵，P=探针目录；默认都收）")
    p.add_argument("--min-gain", type=float, default=0.05,
                   help="投机档保留所需的最小吞吐增益（默认 0.05=噪声守卫）")
    p.add_argument("--model", default="")
    p.add_argument("--out", default=None, help="同时写 Markdown 报告到此路径")
    p.set_defaults(fn=_cmd_recommend)

    p = sub.add_parser("fit", help="从 timing.jsonl 拟合成本模型（可多文件合并）")
    p.add_argument("--timing", required=True, nargs="+")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=_cmd_fit)

    p = sub.add_parser("selftest", help="离线回归（confirm 夹具）")
    p.set_defaults(fn=_cmd_selftest)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
