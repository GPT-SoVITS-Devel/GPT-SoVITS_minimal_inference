<div align="center">

# ⚡ GPT-SoVITS Minimal Inference
**High-Performance | Production-Ready | Zero-Copy Pipeline**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/) [![GPU](https://img.shields.io/badge/CUDA-11.8+-orange.svg)](https://developer.nvidia.com/cuda-zone) [![ONNX](https://img.shields.io/badge/ONNX-Optimized-brightgreen.svg)](https://onnxruntime.ai/) [![TensorRT](https://img.shields.io/badge/TensorRT-Enabled-76B900.svg)](https://developer.nvidia.com/tensorrt) 

[简体中文](./README_zh.md) | [English](./README.md)

**“不仅是代码重构，更是对 GPT-SoVITS 潜力的深度压榨。”**

---
**Engineered for Speed**: A completely refactored inference engine for GPT-SoVITS, featuring ONNX/TensorRT support, KV-Cache optimization, and zero-copy streaming.
</div>

---

## 🌟 核心愿景 (Core Vision)

在不破坏原模型精度、不重新训练的前提下，通过底层算子重写与架构解耦，彻底解决 GPT-SoVITS 在生产环境中的性能瓶颈。

我们追求的是：**快速 (Fast)**、**轻量 (Lightweight)**、**高兼容 (Compatible)**、**可移植 (Portable)**。

## 🚀 性能对比 (Performance Benchmarks)

*测试环境: I7 12700 | RTX 2080TI (22G) | CUDA 12.9 | FP16 精度*

| Metric                      | Native PyTorch | ONNX (fp16) | ONNX Stream | TensorRT (FP16)      |
|:----------------------------|:---------------|:------------|:------------|:---------------------|
| **First Token Latency (↓)** | 2.524 s        | 1.983 s     | **1.000 s** | 2.022 s              |
| **Inference Speed (↑)**     | 144.8 tok/s    | 172.4 tok/s | 167.5 tok/s | **291.6 tok/s** (🤯) |
| **RTF (↓)**                 | 0.3434         | 0.3325      | 0.3100      | **0.2096**           |
| **VRAM Usage (↓)**          | 2.8 G          | 3.9 G       | 4.5 G       | 4.8 G                |

---

## 🛠️ 深度分析：为何重构？ (The "Why")

### 1. 消除动态图与 Python 开销
原版 `GPT-SoVITS` 基于 PyTorch 动态图，在 AR 解码阶段，每生成一个 Token 都会产生显著的 Python 解释器调度开销。在长文本场景下，这种线性累积的延迟是生产环境的噩梦。

### 2. 极致的显存管理优化
*   **KV-Cache 预分配**：规避了 ONNX 导出后常见的 `torch.cat` 导致的空转与频繁内存拷贝。
*   **静态维度对齐**：针对 TensorRT 进行了优化，确保静态执行计划的稳定性，规避动态 Shape 导致的 Re-build 问题。

---

## 💎 核心黑科技 (Core Optimizations)

### 1. 手术刀级算子重写
我们将 GPT 模型拆解为两个独立的计算图：
*   **`GPTEncoder` (Context Phase)**: 一次性处理 Prompt 与 BERT 特征。
*   **`GPTStep` (Decoding Phase)**: 执行 $O(1)$ 复杂度的单步解码，并将 **Top-K Sampling** 下沉至 ONNX 图内部，巨量减少 GPU->CPU 数据传输。

### 2. 全链路 Zero-Copy Pipeline
利用 ONNX Runtime 的 `IOBinding` 技术：
*   **显存驻留**：输入输出直接绑定显存地址，上一轮的 `new_k_cache` 直接作为下轮输入，彻底消除 PCIe 带宽瓶颈。

### 3. 流式推理去伪影 (Artifact-Free)
独创 **Lookahead + History Window** 机制：
*   在 Chunk 边界进行线性加权融合 (Cross-Fade)，彻底消除传统流式推理常见的“咔哒”声。

---

## 🏁 快速开始 (Quick Start)

### 1. 导出模型 (Export)
```bash
python export_onnx.py \
    --gpt_path "weights/gpt.ckpt" \
    --sovits_path "weights/sovits.pth" \
    --output_dir "onnx_export/optimized" \
    --max_len 1000
```

### 2. 精度转换 (Optional)
```bash
python onnx_to_fp16.py \
    --input_dir "onnx_export/optimized" \
    --output_dir "onnx_export/optimized_fp16"
```

### 3. 开启极速推理 (Run)
```bash
# 纯流式推理
python run_onnx_streaming_inference.py --onnx_dir "onnx_export/optimized_fp16" --text "你好，这是一段极速测试。"

# 启动全特性 WebUI
python run_optimized_inference.py --webui
```

### ONNX 优化FP16

```bash
# onnx下对fp16的加速不太明显,但是对显存优化拥有极大好处
python onnx_to_fp16.py --input_dir "onnx_export/optimized" \
  --output_dir "onnx_export/optimized_fp16"
```

### 导出trt

> 编译trt时间较久是正常情况,每台机器在cuda/trt版本不一致时一定要重新编译!!!

```bash
# linux
onnx2trt.sh <onnx_input_dir> <output_dir>
# windows
onnx2trt.bat <onnx_input_dir> <output_dir>
```

---

## 🗺️ 路线图 (Roadmap)

- [x] **V2 / V2ProPlus** 完整支持
- [x] **TensorRT** 静态引擎加速
- [x] **Zero-Copy** IOBinding 优化
- [ ] **Multi-Language Binding**:
    - [ ] C++ SDK (研发中)
    - [ ] Rust / Golang / Android Wrapper
- [ ] **V3 / V4** 模型快速适配
- [ ] **Docker** 一键部署镜像

---

## 🤝 致谢

感谢 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 团队提供的卓越底座。本项目致力于在工程化道路上更进一步。

**如果本项目对你有帮助，请点一个 ⭐，这是我们持续优化的动力！🤗**
