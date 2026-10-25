# SpecTuner 配置推荐报告

- 模式：实测矩阵（measured）
- 模型：`Qwen3.8-27B-FP8@PRO6000`
- 数据源：`../../results/probe_st_20261025/probe_st1`

## 每并发档建议

| 并发 | 建议 | 预期 out_tps | vs 基线 | TPOT(ms) | TTFT 中位(ms) |
|---|---|---|---|---|---|
| c1 | K=3 | 69.4 | +59.7% | 12.7 | 259 |
| c16 | K=3 | 492.1 | +31.7% | 18.2 | 1604 |
| c64 | 关投机 | 763.2 | +0.0% | 35.0 | 1727 |

## vLLM 启动配置

```json
{
  "method": "mtp",
  "num_speculative_tokens": 3,
  "num_speculative_tokens_per_batch_size": [
    [
      1,
      16,
      3
    ],
    [
      17,
      64,
      1
    ]
  ]
}
```

**运维提示**：并发 ≥ 64 时建议关闭投机解码（该档投机最优配置对基线无 ≥5% 净增益；且投机在高并发同时伤害吞吐与 TTFT——确认矩阵 A6/B2 c64 实测 TTFT 13-14s vs 基线 1.7-5.3s）。
