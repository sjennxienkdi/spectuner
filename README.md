# SpecTuner —— 推测解码配置探测定档器

**SpecTuner: probe-and-recommend configuration tuning for vLLM speculative decoding.**

给定一张卡 + 一个模型，15 分钟内自动跑短探针、拟合 draft/verify 成本曲线，
输出该环境推荐的 vLLM `--speculative-config`（固定 K / dynamic 调度表 /
关投机阈值）和依据报告。**vLLM 的 spec 配置不该靠手调和他人的硬件数据。**

## 实测样例（RTX PRO 6000 × Qwen3.8-27B-FP8，2026-10-25 真机探针 8 分钟）

| 并发 | 建议 | 预期吞吐 | vs 基线 | TPOT |
|---|---|---|---|---|
| c1 | K=3 | 69.4 tok/s | +59.7% | 12.7ms |
| c16 | K=3 | 492.1 tok/s | +31.7% | 18.2ms |
| c64 | **关投机** | 763.2 tok/s | — | 35.0ms |

输出附带：可直接粘贴的 vLLM 启动配置 JSON（dynamic 调度表）与运维警告
（c64 以上开投机 = 吞吐 + TTFT 排队双杀，实测 TTFT 13-14s vs 基线 1.7-5.3s）。
完整样例见 [examples/report_measured.md](examples/report_measured.md)。

## 用法

```bash
# 离线推荐：拿 confirm 矩阵形态的实测数据直接出配置（无需 GPU）
spectuner recommend --matrix <results根目录> --out report.md

# 成本模型拟合：探针 timing.jsonl → T_draft/T_verify/T_other 曲线
spectuner fit --timing timing_mtp3.jsonl

# GPU 探针（12-15 分钟）
spectuner probe --quant fp8

# 离线回归（不依赖 pytest）
spectuner selftest
```

在仓库内（未安装）：`PYTHONPATH=. python -m spectuner.cli <cmd>`。
测试：`python -m pytest tests`（21 项：接受率模型 / 成本模型恢复 / 推荐
判据 / confirm 矩阵真值回归；以 CI 实跑为准）。

## 架构

```
probe/probe_run.sh        GPU 探针：K=0/3/1 三臂 × c1/c16/c64 小样本（12-15min）
  │  服务端 bench/patch_timing.py 分项计时（CUDA event，env 门控）
  ├─ serve_c*.json        vllm bench serve：吞吐/TPOT/TTFT/接受长度/分位接受率
  └─ timing_*.jsonl       每步 fwd/sampler/draft GPU 分项 + host cadence
spectuner/
  parse_bench.py          bench 文本表解析（0.32 无 --output-json 的口径）
  fixtures.py             confirm 矩阵装载 = 离线验收夹具（手调最优真值）
  acceptance.py           E[emitted|K]：分位链乘 + 实测锚校准（链乘低估联合）
  costmodel.py            T_step(batch,K) 分项模型；分组中位 + 线性补洞
  recommend.py            档内 argmax + ≥5% 噪声守卫 + disable 阈值 + L0 表
  report.py               Markdown 依据报告
  cli.py                  probe / fit / recommend / selftest
```

## 基准数据集（benchmarks/）

仓库附带本工具的校准与验收数据——**单卡 RTX PRO 6000 Blackwell 96GB 上
Qwen3.8-27B（vLLM 0.32.0）的系统性能对照矩阵**，此硬件档位无公开等价物：

- `benchmarks/confirm/`：24 配置臂 × 3 次重复确认矩阵（FP8/INT4 × MTP/DSpark/
  调度 × c1/c16/c64；指标口径 output throughput / TPOT / TTFT，非 total throughput）
- `benchmarks/w2/`：真实内容负载复核四臂
- `benchmarks/timing/`：draft/verify/sampler 分项 GPU 计时（成本模型燃料）
- `benchmarks/manifests/` + `BASELINE.md`：每次运行的全环境记录（可溯源）
- 置信度 per-position 数据（9MB）不在本仓库，可向作者索取

口径注意：**本数据集所有吞吐均为 output token throughput**（我们踩过把
total throughput 当解码性能的坑，见 B1 博客）。

## 配套工具

配置推荐只管"快"，不管"对"——投机解码正确性验证请配合
[SpecSafe](https://github.com/sjennxienkdi/specsafe)（margin-aware 差分测试器，
已用它抓到 vLLM 真实状态污染 bug
[#63593](https://github.com/vllm-project/vllm/issues/63593)）。

## 判据（工程化转译自预注册协议）

- 档内最优 = argmax out_tps（K=0 基线直接参与竞争）
- 投机档保留条件：吞吐增益 ≥ 5%（噪声守卫；低于噪声地板的赢不开投机）
- c64 关投机在 FP8 上的依据：确认矩阵 A6/B2 TTFT 13-14s vs 基线 1.7-5.3s
  （吞吐+排队延迟双杀）
- 调度表 batch-size≈并发 的映射假设明示于报告（饱和 decode 成立）

## 已验证 / 未验证

- ✅ 21 项离线测试（CI 实跑口径）；真实 confirm 矩阵上推荐 = c1/c16 K=3、c64 关投机，
  与人工结论一致（A2_L0 82.9 的异常优势作为参照臂保留）
- ✅ **2026-10-25 GPU 首测全绿**（`results/probe_st_20261025/`）：端到端
  8m04s（预算 15min）；TPOT 预测 MAPE 8.8%（验收 ≤20%）；探针 measured
  推荐 = confirm 真值 {1:3,16:3,64:0}；计时开/关无可检测开销
- ⚠️ model 模式在 c64 档偏离（选 K=1）：探针 np=32 没打满真 batch=64 +
  tokens>128 无 full-cg 样本 + 排队/prefill 不在 step 模型口径内——
  c64 档以 measured 为准；v2 改进项见提交历史
- ✅ **2026-10-26 跨硬件验收（W4）**：RTX 4090（魔改 48G，sm89）与
  RTX 5090D（32G，sm120 消费级）探针端到端通过——**且推荐结论实质分化**，
  实锤"配置不可跨卡照抄"（详见下方三硬件对照）
- ⬜ 更多硬件欢迎贡献探针数据（benchmarks/ 提 PR 即可）

## 三硬件推荐对照（同模型 Qwen3.8-27B，vLLM 0.32.0）

| 卡 | c1 | c16 | c64 | 数据 |
|---|---|---|---|---|
| PRO 6000 96GB（FP8） | K=3 +60% | K=3 +32% | 关投机 | benchmarks/confirm |
| 4090 魔改 48G（FP8，cu129 构建*） | K=3 +62% | **K=1 +12%** | 关投机 | benchmarks/4090 |
| 4090 魔改 48G（INT4，cu129 构建*） | **K=1 仅+8%** | 关投机 | 关投机 | benchmarks/4090 |
| 5090D 32G（INT4） | MTP K=3 +23.1%（DSpark 锚点仅 +12.9%） | **关投机（DSpark 雪崩 −95%）** | 关投机 | benchmarks/5090d |

三个跨硬件实证：PRO 6000 上的单流之王 INT4 配方在 4090 上几乎无利可图；
DSpark drafter 在 32G 消费卡 c16+ 排队熔断（TTFT 飙至 167-748s）；4090 饱和时
投机解码的形态变为"吞吐 −11% 换 TPOT 3.2×"（per-user 延迟救人）。
魔改卡与 5090D 数字均带标注，不与 stock 卡直接对齐。

**边界声明**：measured 模式（真跑探针）是本工具的主路径；model 模式（成本模型外推）
为实验性——其 c64 档不覆盖 batch=64 稳态、且不含排队/prefill 项，跨硬件外推不可作为
结论依据。\*4090 为魔改 48G 卡 + cu129 构建（PRO 6000 为 cu130），仅证"流程可跨硬件
跑通、推荐会发生反转"，不构成 stock 4090 性能结论。
