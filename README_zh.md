<div align="center">

# ⚡ GPT-SoVITS Minimal Inference

**High-Performance | Production-Ready | Zero-Copy Pipeline**

[![License](https://img.shields.io/badge/license-apache-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![GPU](https://img.shields.io/badge/CUDA-12.6+-orange.svg)](https://developer.nvidia.com/cuda-zone)
[![ONNX](https://img.shields.io/badge/ONNX-Optimized-brightgreen.svg)](https://onnxruntime.ai/)
[![TensorRT](https://img.shields.io/badge/TensorRT-Enabled-76B900.svg)](https://developer.nvidia.com/tensorrt)

[简体中文](./README_zh.md) | [English](./README.md)

**“不仅是代码重构，更是对 GPT-SoVITS 潜力的深度压榨。”**

---
**Engineered for Speed**: A completely refactored inference engine for GPT-SoVITS, featuring ONNX/TensorRT support,
KV-Cache optimization, and zero-copy streaming.
</div>

---

## 🌟 核心愿景 (Core Vision)

我们的核心愿景极其单纯：在绝对不破坏原模型精度，并且兼容性直接拉满的前提下，只办三件事——**快速**，**快速**，还是TM的**快速**！😤

我们不整虚的，就是追求：**快到模糊 (Fast AF)🤯**、**空间换时间 (Space-Time Tradeoff)😤**、**兼容一切 (Compatible)😡**、**润得飞起 (Portable)🤓**。

## 🚀 性能对比 (Performance Benchmarks)

*测试环境: I7 12700 | RTX 2080TI (22G) | CUDA 12.9 | FP16 精度*

*测试模型: GPT-SoVITS V2 PRO PLUS*

| Metric                      | 原生PyTorch(原仓库)  | 原生PyTorch(本仓库) | ONNX        | ONNX Stream | TensorRT(fitted优化)   |
|:----------------------------|:----------------|:---------------|:------------|:------------|:---------------------|
| **First Token Latency (↓)** | 5.417s          | 2.424 s        | 2.683 s     | **1.000 s** | 2.022 s              |
| **Inference Speed (↑)**     | 148.65 tokens/s | 144.8 tok/s    | 172.4 tok/s | 167.5 tok/s | **291.6 tok/s** (🤯) |
| **RTF (↓)**                 | 0.5229          | 0.3434         | 0.3325      | 0.3100      | **0.2096**           |
| **VRAM Usage (↓)**          | 3 G             | 2.8 G          | 3.9 G       | 4.5 G       | 3.4 G                |

---

## 🛠️ 深度分析：为何重构？ (The "Why")

### 1. 消除动态图与 Python 开销

原版 `GPT-SoVITS` 基于 PyTorch 动态图，在 AR 解码阶段，每生成一个 Token 都会产生显著的 Python
解释器调度开销。在长文本场景下，这种线性累积的延迟是生产环境的噩梦。

### 2. 极致的显存管理优化

* **KV-Cache 预分配**：规避了 ONNX 导出后常见的 `torch.cat` 导致的空转与频繁内存拷贝。
* **静态维度对齐**：针对 TensorRT 进行了优化，确保静态执行计划的稳定性，规避动态 Shape 导致的 Re-build 问题。

---

## 💎 核心黑科技 (Core Optimizations)

### 1. 手术刀级算子重写

我们将 GPT 模型拆解为两个独立的计算图：

* **`GPTEncoder` (Context Phase)**: 一次性处理 Prompt 与 BERT 特征。
* **`GPTStep` (Decoding Phase)**: 执行 $O(1)$ 复杂度的单步解码，并将 **Top-K Sampling** 下沉至 ONNX 图内部，巨量减少 GPU->
  CPU 数据传输。

### 2. 全链路 Zero-Copy Pipeline

利用 ONNX Runtime 的 `IOBinding` 技术：

* **显存驻留**：输入输出直接绑定显存地址，上一轮的 `new_k_cache` 直接作为下轮输入，彻底消除 PCIe 带宽瓶颈。

### 3. 流式推理去伪影 (Artifact-Free)

独创 **Lookahead + History Window** 机制：

* 在 Chunk 边界进行线性加权融合 (Cross-Fade)，彻底消除传统流式推理常见的“咔哒”声。

---

## 🏁 快速开始 (Quick Start)

### 1. 导出模型 (Export)

```bash
python export_onnx.py \
    --gpt_path "pretrained_models\GPT_weights_v2ProPlus/firefly_v2_pp-e25.ckpt"
    --sovits_path "pretrained_models\SoVITS_weights_v2ProPlus/firefly_v2_pp_e10_s590.pth"
    --cnhubert_base_path pretrained_models\chinese-hubert-base
    --bert_path pretrained_models\chinese-roberta-wwm-ext-large
    --output_dir  "onnx_export/firefly_v2_proplus"
    --max_len 1000 # 缩小能加速吞吐与减小预分配的显存大小,但需要修改参数,通常来说1000能在大部分场景(长短文本)下找到一个相对还行的平衡点
```

### 2. 精度转换 (Optional)

```bash
python onnx_to_fp16.py \
    --input_dir "onnx_export/firefly_v2_proplus" \
    --output_dir "onnx_export/firefly_v2_proplus_fp16"
```

### 3. 开启极速推理 (Run)

```bash
# 纯流式推理
python run_onnx_streaming_inference.py \
    --onnx_dir onnx_export/firefly_v2_proplus_fp16 \
    --ref_audio "pretrained_models\看，这尊雕像就是匹诺康尼大名鼎鼎的卡通人物钟表小子.wav" \
    --ref_text "看，这尊雕像就是匹诺康尼大名鼎鼎的卡通人物“钟表小子" \
    --ref_lang "zh" \
    --text "范肖有一项奇特的能力，可以把自己的运气像钱一样攒起来用。攒的越多，越能撞大运。比如攒一个月，就能中彩票。那么，攒到极限会发生什么呢？"
     --lang "zh" --output "out_onnx_stream.wav"

# 启动全特性 WebUI
python run_optimized_inference.py --onnx_dir onnx_export/firefly_v2_proplus_fp16 --webui
```

### 导出 TensorRT Engine

> 编译 TRT 时间较久是正常情况，每台机器在 CUDA/TRT 版本不一致时一定要重新编译！

```bash
# 自动检测 GPU 显存，选择最优 shape profile
python onnx2trt.py \
    --input_dir onnx_export/firefly_v2_proplus_fp16 \
    --output_dir onnx_export/firefly_v2_proplus_fp16

# 显存紧张，使用更紧凑的 profile
python onnx2trt.py \
    --input_dir onnx_export/firefly_v2_proplus_fp16 \
    --output_dir onnx_export/firefly_v2_proplus_fp16 \
    --shape_profile fitted --opt_level 2

# 查看所有选项
python onnx2trt.py --help
```

可用的 Shape Profile：

| Profile | sovits 最大语义长度 | 单段最长音频 | 推荐显存         |
|---------|-----------------|------------|--------------|
| `small` | 150 | ~6秒 | <=12GB       |
| `fitted` | 250 | ~10秒 | 8-24GB（实测定制） |
| `medium` | 400 | ~16秒 | 16-24GB（默认）  |
| `large` | 1000 | ~40秒 | >=32GB       |

> 建议：先用 ONNX 推理跑一遍，查看输出的 **Shape Profile Summary**，再选择最合适的 profile。`fitted` profile 是基于真实推理数据优化的最佳选择。

---

## 🌐 API 服务

如果您不想每天对着终端敲命令，或者想让您的后端程序直接调用。我们整出了兼容 **OpenAI 协议** 的 API 服务，支持流式输出。

*   **PyTorch 稳定版**: `python api_server.py` (默认 8000 端口，适合还没折腾 ONNX 的你)
*   **ONNX 极速版**: `python api_server_onnx.py` (默认 8001 端口，CPU 用户的福音，部署简单)
*   **TensorRT 究极版**: `python api_server_trt.py` (默认 8002 端口，显卡在尖叫，性能在狂飙)

👉 **[点击查阅 API 详细文档](./API_USAGE.md)** —— 求求了，看一眼文档吧，都在这里了。

---

## 🛠️ SDK

C++: [GPT-SoVITS-Devel/GPT-SoVITS-cpp](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp)

---

## 🗺️ 路线图 (Roadmap)

- [x] **V2 / V2ProPlus** 完整支持
- [x] **TensorRT** 静态引擎加速
- [x] **Zero-Copy** IOBinding 优化
- [ ] **Multi-Language Binding**:
    - [x] C++ SDK
    - [ ] Rust / Golang / Android Wrapper
- [ ] **V3 / V4** 模型快速适配
- [ ] **Docker** 一键部署镜像

---

## 🤝 致谢

感谢 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 团队提供的卓越底座。本项目致力于在工程化道路上更进一步。

**如果本项目对你有帮助，请点一个 ⭐，这是我们持续优化的动力！🤗**
