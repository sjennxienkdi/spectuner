"""SpecTuner：推测解码配置探测定档器（项目 04）。

给定 GPU + 模型，跑短探针、拟合 draft/verify 成本曲线，输出该环境推荐的
vLLM speculative 配置（K 值 / dynamic 调度表 / 关投机阈值）与依据报告。
"""
__version__ = "0.1.0"
