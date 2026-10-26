# SpecTuner

**English** | [简体中文](README.zh-CN.md)

**SpecTuner: probe-and-recommend configuration tuning for vLLM speculative decoding.**

Give it one GPU + one model: within ~15 minutes it runs short probes, fits
draft/verify cost curves, and outputs the recommended vLLM `--speculative-config`
for that environment (fixed K / dynamic schedule / disable threshold) with an
evidence report. **Spec configs should not be hand-tuned or copied from other
people's hardware numbers.**

## Measured example (RTX PRO 6000 × Qwen3.8-27B-FP8, 2026-10-25, 8-min probe)

| Concurrency | Recommendation | Expected tput | vs baseline | TPOT |
|---|---|---|---|---|
| c1 | K=3 | 69.4 tok/s | +59.7% | 12.7ms |
| c16 | K=3 | 492.1 tok/s | +31.7% | 18.2ms |
| c64 | **disable** | 763.2 tok/s | — | 35.0ms |

Output includes a paste-ready vLLM startup JSON (dynamic schedule) and ops
warnings (on this stack, speculation at c64 = throughput + TTFT queueing
double-hit: measured TTFT 13–14s vs baseline 3.2–5.2s, confirm-matrix medians).
Full sample: [examples/report_measured.md](examples/report_measured.md).

## Usage

```bash
# Offline recommendation from confirm-matrix-shaped measured data (no GPU)
spectuner recommend --matrix <results-root> --out report.md

# Cost-model fitting: probe timing.jsonl -> T_draft/T_verify/T_other curves
spectuner fit --timing timing_mtp3.jsonl

# GPU probe (12-15 min)
spectuner probe --quant fp8

# Offline regression (no pytest needed)
spectuner selftest
```

Inside the repo without installing: `PYTHONPATH=. python -m spectuner.cli <cmd>`.
Tests: `python -m pytest tests` (21 tests: acceptance model / cost-model
recovery / recommendation criteria / confirm-matrix ground-truth regression;
CI is the source of truth).

## Architecture

```
probe/probe_run.sh        GPU probe: K=0/3/1 arms x c1/c16/c64 small samples (12-15min)
  |   server-side bench/patch_timing.py per-stage timing (CUDA events, env-gated)
  |- serve_c*.json        vllm bench serve: tput/TPOT/TTFT/accept length/quantile rates
  '- timing_*.jsonl       per-step fwd/sampler/draft GPU stages + host cadence
spectuner/
  parse_bench.py          bench text-table parsing (0.32 has no --output-json)
  fixtures.py             confirm-matrix loader = offline acceptance fixture
  acceptance.py           E[emitted|K]: quantile chain product + measured-anchor calibration
  costmodel.py            T_step(batch,K) stage model; grouped medians + linear gap-fill
  recommend.py            per-band argmax + >=5% noise guard + disable threshold + L0 table
  report.py               Markdown evidence report
  cli.py                  probe / fit / recommend / selftest
```

## Benchmark dataset (benchmarks/)

The repo ships the tool's calibration and acceptance data — a **systematic
single-RTX-PRO-6000-96GB performance matrix for Qwen3.8-27B on vLLM 0.32.0**;
we are not aware of a public equivalent for this hardware class:

- `benchmarks/confirm/`: 24 arms × 3 repeats confirmatory matrix (FP8/INT4 ×
  MTP/DSpark/schedules × c1/c16/c64; metrics are output throughput / TPOT /
  TTFT — never total throughput)
- `benchmarks/w2/`: real-content workload replication (w2c_agg.md; includes a
  disclosure that the first-pass w3_* dirs ran with random-load config —
  manifest-verifiable)
- `benchmarks/timing/`: per-stage GPU timing for draft/verify/sampler
  (cost-model fuel)
- `benchmarks/manifests/` + `BASELINE.md`: full environment records per run
- Per-position confidence data (9MB) not in this repo — available on request

Metric note: **all throughputs in this dataset are output-token throughputs**
(we fell into the total-throughput trap ourselves — see the B1 blog post).

## Companion tool

Recommendations only care about *fast*, not *correct* — pair with
[SpecSafe](https://github.com/sjennxienkdi/specsafe) (margin-aware differential
tester; it caught a real vLLM state-corruption bug,
[#63593](https://github.com/vllm-project/vllm/issues/63593)).

## Decision criteria (engineering translation of the pre-registered protocol)

- Per-band optimum = argmax out_tps (the K=0 baseline competes directly)
- A speculation arm stays enabled only if its throughput gain ≥ 5% (noise guard)
- The c64-disable call on FP8 rests on confirm-matrix A6/B2 TTFT 13–14s vs
  baseline 3.2–5.2s (throughput + queueing double-hit)
- The batch-size≈concurrency mapping assumption is stated explicitly in reports

## Verified / not verified

- ✅ 21 offline tests (CI-verified); on the real confirm matrix the
  recommendation = c1/c16 K=3, c64 disable, matching the hand-tuned conclusion
- ✅ **2026-10-25 first GPU run all green**: end-to-end 8m04s (15-min budget);
  TPOT prediction MAPE 8.8% (acceptance ≤20%); probe recommendation =
  confirm ground truth {1:3, 16:3, 64:0}; timing on/off overhead undetectable
- ⚠️ model mode deviates at c64 (picks K=1): probe np=32 doesn't reach true
  batch=64, no full-CG samples above 128 tokens, and queueing/prefill is
  outside the step-model scope — trust measured mode at c64
- ✅ **2026-10-26 cross-hardware acceptance (W4)**: RTX 4090 (modded 48G, sm89)
  and RTX 5090D (32G, sm120 consumer) probes ran end-to-end — with
  **materially divergent recommendations**, evidence that configs cannot be
  copied across GPUs (table below)
- ⬜ More hardware welcome — probe-data PRs to benchmarks/ are accepted

## Three-hardware recommendation comparison (same model, Qwen3.8-27B, vLLM 0.32.0)

| GPU | c1 | c16 | c64 | Data |
|---|---|---|---|---|
| PRO 6000 96GB (FP8) | K=3 +60% | K=3 +32% | disable | benchmarks/confirm |
| 4090 modded 48G (FP8, cu129 build*) | K=3 +62% | **K=1 +12%** | disable | benchmarks/4090 |
| 4090 modded 48G (INT4, cu129 build*) | **K=1 only +8%** | disable | disable | benchmarks/4090 |
| 5090D 32G (INT4) | MTP K=3 +23.1% (DSpark anchor only +12.9%) | **disable (DSpark collapse −95%)** | disable | benchmarks/5090d |

**Load stratification (important)**: every "disable" above comes from the
random 2K/256 workload. Under a real-content short-input workload (90
sharegpt-sourced prompts, `benchmarks/w2/w2c_agg.md`) the conclusion reverses:
MTP3 c64 = 1.686× baseline (TPOT 21.9 vs 36.5ms, TTFT median only +62ms).
Speculation gain depends heavily on workload composition — figure out which
one yours resembles first.

Three cross-hardware facts: the INT4 recipe that wins single-stream on PRO 6000
is nearly worthless on the 4090; the DSpark drafter queueing-melts on the 32G
consumer card at c16+ (TTFT 167–748s); on the saturated 4090 speculation
morphs into "−11% throughput for 3.2× TPOT" (per-user latency rescue). Modded
cards and the 5090D are labeled — do not align them with stock numbers.

**Boundary statement**: measured mode (actually running probes) is the main
path; model mode (cost-model extrapolation) is experimental — its c64 band
does not cover the batch=64 steady state and omits queueing/prefill, so
cross-hardware extrapolation must not be used as evidence. \*The 4090 is a
modded 48G card on a cu129 build (PRO 6000 ran cu130): it proves the pipeline
crosses hardware and that recommendations flip — it is not a statement about
stock-4090 performance.
