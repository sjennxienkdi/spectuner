# 5090D 32G 锚点数据（2026-10-26 窗口）

**硬件**：RTX 5090 D 32G（sm120 消费级，UUID GPU-5cbfb21b-d880-d4c1-4beb-3d2a419488f5，
driver 580.178.04）单卡 TP=1。所有数字标注 **5090D/32G**，与 stock 5090（28G 带宽
更宽）不可直接对齐。

**栈**：vLLM 0.32.0（PyPI，与 PRO 6000 主实验同 cu130 构建）+ torch 2.13.0+cu130 +
flashinfer 0.7.0.post1 + jit-cache-sm120f（560 预编译模块，本地中转安装）。
attention 路径：FlashInfer backend（INT4 的 fp8-KV 配方在 sm120 排除 FLASH_ATTN）
+ `--attention-config.use_trtllm_attention 0`（关 XQA/TRTLLM decode，详见
projects/03_specsafe/results/5090d_window/env_manifest_5090d.txt）。vllm 代码零补丁。

**模型**：RedHatAI/Qwen3.8-27B-INT4（compressed-tensors，含 fp8 KV 配方；
+FP8 同源 vocab.json/merges.txt 修复 tokenizer）。负载：random 2048/256，
REASONING=off，async-scheduling on，np=16/64/128（c1/c16/c64）。

## 锚点结果（一轮，均值口径 Output throughput / Median TTFT / Mean TPOT）

| 臂 | c1 out_tps | c1 TTFT/TPOT | c16 out_tps | c16 TTFT/TPOT | c64 out_tps | c64 TTFT/TPOT |
|---|---|---|---|---|---|---|
| A0（INT4, none） | 71.56 | 535ms / 11.9ms | 398.93 | 1561ms / 31.6ms | 388.96 | 12413ms / 92.3ms |
| INT4+DSpark (K=7) | 80.77 (+12.9%) | 574ms / 10.2ms | 20.51 (−94.9%) | 167050ms / 95.9ms | 20.77 (−94.7%) | 747816ms / 95.0ms |

要点：
- 32G 卡 c16→c64 吞吐不升反平（398.9→389.0），TTFT 中位 1.6s→12.4s——
  排队/prefill 瓶颈在 28G 之前先到
- DSpark 只在 c1 有正收益（+12.9%，低于探针 mtp3 的 +23.1%）；c16/c64
  灾难性回退（−95%，TTFT 中位 167s/748s）——32G 消费卡高并发禁开外部 drafter
- 与 4090W4-INT4 对照：c1 基线 71.6 vs 38.5（~1.86×），投机增益 +23% vs +8.1%
  ——sm120 消费卡的 INT4+MTP 低并发增益是真实的

数据：`5090d_int4_none/`、`5090d_int4_dspark/`（serve_c*.json + server.log +
manifest.json）；`run_manifest.jsonl` 为本窗口追加的两条记录；
`anchor_{none,dspark}_run.log` 为驱动日志。
