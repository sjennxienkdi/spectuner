# BASELINE.md — 环境锁定文件（docs/08 P0 / docs/07 阶段 A1）

> 冻结日期：2026-10-24（GPU 窗口内，确认性矩阵发车前定格）
> 本文件是正式实验的环境唯一事实源。任何环境变更须追加变更记录段，不得原地改写。
> 配套：`docs/08_完整试验方案.md`（协议）、`docs/09_PR53929核读与移植评估.md`（上游移植决策）。

## 1. 硬件

| 项 | 值 |
|---|---|
| GPU | 1× NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB（SM120） |
| GPU UUID | `GPU-5501484a-a35a-3e26-876e-e4c7663b43e3` |
| 驱动 | 595.71.05 |
| CUDA（nvidia-smi 报告 / 系统 toolkit） | 13.2 / /usr/local/cuda-13.0 |
| CPU / RAM | 208 核 / 1007 GB |
| 实例 | AutoDL westd，数据盘 /root/autodl-tmp 106GB（本锁定仅对该实例/UUID 有效） |

## 2. 软件栈（基线 venv：`/root/autodl-tmp/venv`）

| 包 | 版本 |
|---|---|
| vllm | 0.32.0（PyPI wheel；对应源码 ref `5df63a7203cdb9328d824a27b682e77c7b9928be` = tag v0.32.0，本地 `vllm-src/`） |
| torch | 2.13.0 |
| flashinfer-python | 0.7.0.post1 |
| triton | 3.7.1 |
| transformers | 5.19.0 |
| numpy | 2.3.5 |
| Python | 3.12（venv） |

**基线 venv 本地补丁（env 门控，默认关闭，行为等价原版）：**
- `vllm/v1/core/sched/scheduler.py`：patch_l1.py v4（L1 EMA 键控动态 SD）。
  门控变量 `VLLM_L1_SD`（阈值:K 表）与 `VLLM_L1_MAX_SPEC_BATCH`（batch>K 时 K 下限 1）。
  备份 `scheduler.py.l1bak`。不设环境变量 = L0/原版行为。
- `bench/sanity_check.py` v2（客户端脚本，不影响服务端）：启动时经 `/v1/models` 自动发现 model id，
  修复 DAY1 "404 端口误伤" 的真因（model 名 `default` 与 served_model_name 不匹配）。

## 3. 模型资产（均为本地下载，路径即身份）

| 角色 | 来源 repo | 本地路径 | config.json md5 |
|---|---|---|---|
| 主目标 FP8 | Qwen/Qwen3.8-27B-FP8（ModelScope 镜像，layers-N 分片） | /root/autodl-tmp/models/qwen3.8-27b-fp8 | `fa7068d074f68606785e4f2d872d1480` |
| 主目标 INT4 | RedHatAI/Qwen3.8-27B-INT4 | /root/autodl-tmp/models/qwen3.8-27b-int4 | （分组实验，单独记录） |
| 主目标 NVFP4 | nvidia/Qwen3.8-27B-NVFP4 | /root/autodl-tmp/models/qwen3.8-27b-nvfp4 | 已知上游双 bug（in_proj_ba FP4 scale + 融合层 scale 映射），**排除出正式性能结论** |
| DSpark drafter | Doopeworld/Qwen3.8-27B-DSpark-vLLM（RadixArk 1.86B 的 vLLM config 补丁版） | /root/autodl-tmp/models/drafter-dspark | `c3bc222140c3290f93cb4aab7f6958ff` |

架构事实（FP8 config 实读）：`Qwen3_5ForConditionalGeneration` / model_type `qwen3_5`，
48 层 GatedDeltaNet + 16 层 GQA，vocab 248,320，原生 1 层 MTP，`--language-model-only` 生效。
drafter 权重含 `confidence_head`（AcceptRatePredictor）与 `markov_head`——0.32.0 的 DSpark
adaptive 路径**优先使用该训练 head**（dspark/speculator.py:106-113），仅无 head 时退到
draft-logits 在线估计器（acceptance_estimator.py）。三信号源（训练 head / 估计器 / 静态先验）
在阶段 B 校准中分开记录。

## 4. 服务参数（冻结，全部经 run_all.sh 传递）

```
vllm serve <model> \
  --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 128 \
  --language-model-only --reasoning-parser qwen3        # int4/nvfp4 臂 REASONING=off（第三方量化缺 think token）
  [--async-scheduling]                                   # 按 臂 ASYNC on/off
  [--kv-cache-dtype fp8]                                 # 仅 bf16 臂曾用；FP8/INT4 臂 auto（bf16 KV）
  --speculative-config {...}                             # 按臂：none / mtp{1,2,3} / mtpdyn / dspark(K=7, probabilistic)
```

引擎实测行为（server.log 确认）：V1 引擎、seed=0、prefix caching on、chunked prefill on、
compilation VLLM_COMPILE / cudagraph FULL_AND_PIECEWISE、capture sizes [1..256]、
TP1/DP1、kv_cache_dtype auto（=bf16）。

## 5. 负载（冻结）

`vllm bench serve --backend vllm --dataset-name random`：
input_len=2048、output_len=256、**seed=0（bench 默认，stdout Namespace 可验证）**、
并发档 c1/c16/c64 → num_prompts 16/64/128（沿用 DAY1/DAY2 pilot 口径，保证可比）。

已知口径限制（预注册，不追改）：0.32 的 bench serve 无 `--output-json`，serve_c*.json 为
stdout 文本表 + Namespace 行；无独立客户端预热（服务端 graph capture 预热在启动内完成，
各臂流程完全一致）；结果文件解析用尾部文本表。

## 6. 外部工作 SHA 快照（2026-10-24 定格）

| 引用 | 定格点 |
|---|---|
| vLLM #53929（GDN adaptive DSpark） | head `4955ffdb9c`（base=main@`4a6a3272e8`，2026-08-26/27） |
| Suppressor72 修复（缺陷1+2） | `3462da7907`（branch fix/gdn-adaptive-capture-geometry，叠加在 4955ffdb9c 上） |
| Suppressor72 rebase（本项目采用） | `c4539fe708`（10 commits on main@`7b569ec895`，2026-10-20；单测过，GPU E2E 未重验） |
| 关联上游 | #51869（GDN varlen 主 issue）、#60068（MRV2 confidence-stop）、#62304（GDN 部分接受正确性）、#35150（SGLang GDN 状态漂移）、SGLang #46182/#46187/#46190 |

## 7. adaptive 实验隔离（版本纪律执行方式）

- 基线 venv 的 `site-packages/vllm` **保持原样**（仅含 §2 的 env 门控 L1 补丁）。
- adaptive 变体 = `site-packages/vllm_adaptive`（整包拷贝 + docs/09 §4 的 5 文件改动；
  4 patch 原样 + gdn_attn.py 锚点适配，全部纯 Python）。
- 切换 = 目录重命名，切换动作写入 manifest；每臂启动后从 server.log 校验加载的包路径。
- adaptive 臂的全注意力组固定 **TRITON_ATTN**（FlashInfer 在 SM120 明确 opt-out device-CPU
  mismatch）；配对对照 A6'（TRITON_ATTN + static dspark K=7）同后端对比，不与 FlashInfer
  基线直接混表。

## 8. run manifest schema（confirm_manifest.jsonl，每臂一条）

```json
{"run_id":"confirm_r{轮}_{臂名}", "round":1, "arm":"A0_none", "ts_start":"ISO-UTC",
 "ts_end":"ISO-UTC", "exit_code":0, "json_files":3, "results_dir":"results/confirm_r1_A0_none",
 "config":{"QUANT":"fp8","SPEC":"none","ASYNC":"on","REASONING":"on","VLLM_L1_SD":"","VLLM_L1_MAX_SPEC_BATCH":""},
 "gpu_start":{"temp_c":34,"power_w":35,"mem_mb":0}, "gpu_uuid":"GPU-5501484a-...",
 "driver":"595.71.05", "vllm":"0.32.0",
 "workload":{"dataset":"random","input_len":2048,"output_len":256,"seed":0,"concurrency":[[1,16],[16,64],[64,128]]}}
```

 DAY1/DAY2 数据 = pilot（单次，不作确认性证据）；`confirm_r*` = 确认性数据。

## 9. 冻结判据（引自 docs/08 §11，此处为执行副本）

- 产品门：正确性 G0 全过后，adaptive 候选 W1 c64 output throughput ≥ A0 的 95%，c1 ≥ 1.5×；
  TTFT/TPOT P95 相对最佳静态策略不恶化超 5%。
- 机制门：cost+confidence 相对 cost-only 的 goodput 净提升 ≥5%（或吞吐非劣下 P95 明确下降）。
- 信号门：代理相对静态先验 ΔAUC≥0.10 且校准可用，才接入调度。
- 差异 <5% 或跨重复方向不稳 → 标"不确定"；n=3 是最低工程复核，不作显著性宣称。

## 10. P0-P3 预备冻结（2026-10-24 晚，GPU 窗口前；项目书05 §五 全项落地）

### 10.1 W2 固定真实内容负载（P0-b）

- 构建：`bench/w2_build.py`（确定性规则：sha256(text) 升序 + token∈[16,1024] 过滤，每类取前 30，无 RNG）
- 三类来源（许可明确，hf-mirror 定格）：
  | 类别 | 数据集@SHA | 许可 |
  |---|---|---|
  | general_qa | databricks/databricks-dolly-15k@bdd27f4d（open_qa 子集） | CC-BY-SA-3.0 |
  | code | google-research-datasets/mbpp@4bb6404f（full/train） | CC-BY-4.0 |
  | math | openai/gsm8k@740312ad（main/train） | MIT |
- 冻结文件（sha256 已互验，`w2_build.py --check` 可复核）：
  - `bench/w2_prompts.jsonl` = `009f7e03e055183e240926dd53556c54f18836d87b0a031e06c97e3892efd13c`（90 条）
  - `bench/w2_sharegpt.jsonl` = `6d5f5081012700529f024f94f7722b22cae3f87664d6ac680e5dbad2cd9b838a`（90 条）
  - token 统计（Qwen3.8 tokenizer@017b9c7a，裸文本）：qa 16-42（均值18.9）/ code 16-35（20.0）/ math 28-119（58.9）
- 运行口径（全臂统一）：`--dataset-name sharegpt --dataset-path w2_sharegpt.jsonl --sharegpt-output-len 256 --temperature 0 --seed 0`，
  90 条全送（c1/c64 同集同序）；输出上限 256 含 reasoning token。chat template 开销全臂一致，不另计。

### 10.2 A0 噪声 CV 与 n 决定

confirm r1-r3 A0 out_tps CV：c1=0.01% / c16=0.02% / c64=0.05%——全部远低于 2.5% 上限，
**n=3 维持，不上调**（预授权未触发）。证据：`results/p0p3_prep/prep_analysis.md` §1。

### 10.3 H1 容差与预备判定

- 容差冻结：|acc_len(c64)/acc_len(c1) − 1| ≤ **5%**（与 §9 的 5% 噪声口径一致）
- 预备判定（正式判定在窗口二 adaptive 复核后一并归档）：主判臂 A1_mtp3
  c1 2.47 [2.47~2.66] vs c64 2.46 [2.42~2.55]（n=3），Δ=-0.4% → **支持 H1**；
  逐位置接受率 c1 64/49/34 vs c64 62/47/37 交叉验证成立。
- 边界记录：A6_dspark（固定 K=7）c64 -10.6% 为真实收窄；A2_L0 的 -40% 属策略性砍 K
  （pos0 反升），不入 H1 判定；B2_int4_dspark +18.8% 为 INT4 独立观察。

### 10.4 best-fixed K 冻结（W1/W2 对照统一）

**K=3（mtp3）**。依据：c1 72.8 / c16 543.8 均最优；c64 上 mtp2 单次 +3.8% 在 5% 噪声门内
且 async 口径混杂 → 记"不确定"，按 docs/08 §11 不以单次分档挑 K。窗口二 W2 复核的
best-fixed 臂即 A1_mtp3。

### 10.5 窗口二确认性臂清单冻结

| 组 | 臂 | 轮 | 后端/包 |
|---|---|---|---|
| W1 追加 | A4_adaptive（dspark K=7 + enable_adaptive_verification + TRITON_ATTN） | ×3 | vllm_adaptive |
| W1 追加 | A6'_static_triton（同后端 static 对照） | ×3 | vllm_adaptive |
| W2 复核 | A0_none / A1_mtp3 / A4_adaptive × c1/c64 | ×3 | stock / stock / vllm_adaptive（跨后端不混表，§7 纪律） |

顺序轮转（r2 反序）防温度/时钟漂移；A4 入阵依据 = docs/09 §9（移植成功 + G0 margin 门 PASS）。
前置门（window2_confirm.sh 强制）：窗口一 spot-check ±5% + SpecSafe margin-aware 改判非 FAIL 主导 + 钩子开销已量化。

### 10.6 插桩部署纪律（stageb + timing 双钩子）

- `patch_stageb.py`（置信度/接受）+ `patch_timing.py`（分项计时，项目04 共享资产）可叠加
  （锚点互不重叠，已本地验证：锚点唯一/编译/幂等/共存）。
- 部署：仅窗口一采集跑前 apply 到**当前激活包**（stock），首次打前存 `model_runner.py.pristine`，
  采集毕 revert 回 pristine 并 compile 校验；正式性能臂钩子 env 全程未设（零行为）。
- 采集目录独立（`results/w1_stageb_*`、`/root/autodl-tmp/{stageb,timing}_out/`），不进性能表。

## 变更记录

- 2026-10-24 12:04：初始冻结。同日确认性矩阵（matrix_confirm.sh）发车，24 臂×3 档并发。
- 2026-10-24 晚：**P0-P3 预备冻结追加**（GPU 窗口前本地完成，项目书05 §五）——详见 §10。
- 2026-10-25 03:00-05:50：**P0-P3 GPU 窗口执行**（新实例：数据盘迁移，GPU UUID
  `GPU-d313bb1a-…`/驱动 580.82.09 ≠ 冻结值——环境变更经 spot-check 门仲裁：A0/mtp3
  与 confirm 中位偏差 ≤2.1%，可比性成立）。执行详情与判读：
  `results/p0p3_prep/GPU窗口执行报告.md`。要点：
  - 窗口一三门 PASS（环境/开销/SpecSafe——static32 ALL PASS，drift 1 fail 为
    #35150 形状独立发现）；插桩采集 10,462 条（stageb+timing，两静默 bug 当场修复）
  - 窗口二：A4/A6' W1 确认 ×3（配对 c1 0.948×/c16 1.083×/c64 1.036×）；
    W2 真实内容复核 ×3（mtp3 c1 2.325×/c64 1.686×；A4 c1 2.922×/c64 1.112×，TTFT 门未过）
  - W2 bench 格式勘误（§10.1 更正）：0.32.0 无 `--sharegpt-output-len`（--output-len
    映射）；ShareGPTDataset 需单 JSON 数组且 conversations≥2 轮——冻结产物改为
    `bench/w2_sharegpt.json`（sha256 见 bench/w2_manifest.json），prompt 内容与
    `w2_prompts.jsonl` 规范集不变
  - H1 正式判定：支持（W1 Δ−0.4%、W2 Δ−3%，均 ±5% 容差内）
  - 云机已于 05:5x 关机停费
