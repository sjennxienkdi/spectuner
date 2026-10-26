# benchmarks/4090/ — RTX 4090（魔改 48G）锚点与跨硬件验证归档

> ⚠️ **所有数字带 "modded 48G" 标注使用**：本卡为魔改 48GB RTX 4090
> （49140 MiB，sm89），内存时序/带宽可能与 stock 4090 不同——绝对吞吐
> 不与 stock 4090 或其他卡直接对齐，只用于本机内的相对比较与跨硬件
> 结论形态对照。

## 环境（详见 env/ 与 run_manifest.jsonl）

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 49140 MiB（modded 48G），sm89 |
| 驱动 | 570.211.01（CUDA 12.8；PyPI cu130 栈不可用） |
| vLLM | 0.32.0 **+cu129**（官方 GitHub release wheel） |
| torch | 2.13.0+cu129（download.pytorch.org） |
| flashinfer | 0.7.0.post1 + jit-cache-sm89（预编译，无 nvcc JIT） |
| 模型 | Qwen3.8-27B-FP8（ModelScope，66 shards + mtp.safetensors）；Qwen3.8-27B-INT4（hf-mirror + model_mtp 本地中转，sha256 一致） |
| 说明 | vLLM 代码版本与 PRO 6000 主实验一致（0.32.0），CUDA 构建变体不同（cu129 vs cu130） |

## ⑤ 锚点臂（fp8，random 2048/256，async on，每档单轮）

| 臂 | c | out tok/s | TPOT ms | TTFT 中位 ms | acc_len |
|---|---|---|---|---|---|
| A0 none | 1 | 24.5 | 37.6 | 873 | — |
| A0 none | 16 | 185.4 | 83.1 | 3032 | — |
| A0 none | 64 | 211.7 | 214.5 | 17247 | — |
| mtp3 | 1 | 40.4（+65%） | 20.9 | 943 | 2.40 |
| mtp3 | 16 | 197.8（+6.7%） | 66.6 | 1926 | 2.53 |
| mtp3 | 64 | 188.1（-11%） | **67.2（3.2× 优于基线）** | 64907 | 2.40 |

（c64 mtp3 TTFT 中位 64.9s：128 条排队恶化，高并发投机的排队代价在
48G 卡上被放大；吞吐 -11% 但延迟收益极大——延迟敏感场景的取舍留给
下游，SpecTuner 吞吐口径下仍推荐 c64 关投机。）

## 同窗口其它产出（归档位置）

- SpecSafe drift-k7 S7 跨硬件复现：`projects/03_specsafe/results/4090_window/drift_min/`
  （**sm89-clean：6/6 pass 零分歧，2 独立 boot**）
- SpecSafe rebuild_floor ×2：`projects/03_specsafe/results/4090_window/floor_4090_r{1,2}/`
  （sm89 重建地板不可测=0；PRO 6000 为 0–0.1875 nat）
- SpecTuner W4 探针 FP8/INT4 + 推荐：`projects/04_spectuner/results/4090_w4/`
  （FP8 推荐 c1=K3/c16=K1/c64 关——与 PRO 6000 的 c16 反转）
- vLLM #63593 补充评论草稿（未发布）：`projects/03_specsafe/issue_63593_sm89_comment_draft.md`
