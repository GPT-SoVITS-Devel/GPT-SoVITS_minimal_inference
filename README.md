<div align="center">

# ⚡ GPT-SoVITS Minimal Inference

**High-Performance | Production-Ready | Zero-Copy Pipeline**

[![License](https://img.shields.io/badge/license-apache-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![GPU](https://img.shields.io/badge/CUDA-12.6+-orange.svg)](https://developer.nvidia.com/cuda-zone)
[![ONNX](https://img.shields.io/badge/ONNX-Optimized-brightgreen.svg)](https://onnxruntime.ai/)
[![TensorRT](https://img.shields.io/badge/TensorRT-Enabled-76B900.svg)](https://developer.nvidia.com/tensorrt)

[简体中文](./README_zh.md) | [English](./README.md)

**"Not just a refactor, but a deep squeeze of GPT-SoVITS potential."**

---
**Engineered for Speed**: A completely refactored inference engine for GPT-SoVITS, featuring ONNX/TensorRT support,
KV-Cache optimization, and zero-copy streaming.
</div>

---

## 🌟 Core Vision

We ain't here to nerf your model accuracy or break your production setup with retraining nonsense. We are here to smash
those bottlenecks into oblivion.

Our goal is simple: **Make GPU go brrr**. We strive for: **Fast AF 🏎️**, **Space-Time Tradeoff ⚖️**, **Compatible AF 🤝
**, and **Portable 🌍**. No cap, just pure speed. 😤

## 🚀 Performance Benchmarks

*Environment: I7 12700 | RTX 2080TI (22G) | CUDA 12.9 | FP16 Precision*

| Metric                      | Native PyTorch(Original Project) | Native PyTorch(This Project) | ONNX        | ONNX Stream | TensorRT             |
|:----------------------------|:---------------------------------|:-----------------------------|:------------|:------------|:---------------------|
| **First Token Latency (↓)** | 5.417s                           | 2.424 s                      | 2.683 s     | **1.000 s** | 2.022 s              |
| **Inference Speed (↑)**     | 148.65 tokens/s                  | 144.8 tok/s                  | 172.4 tok/s | 167.5 tok/s | **291.6 tok/s** (🤯) |
| **RTF (↓)**                 | 0.5229                           | 0.3434                       | 0.3325      | 0.3100      | **0.2096**           |
| **VRAM Usage (↓)**          | 3 G                              | 2.8 G                        | 3.9 G       | 4.5 G       | 4.8 G                |

---

## 🛠️ Deep Analysis: Why Refactor?

### 1. Eliminating Dynamic Graph & Python Overhead

The original `GPT-SoVITS` is based on PyTorch dynamic graphs. During the AR decoding stage, generating each token incurs
significant Python interpreter scheduling overhead. In long-text scenarios, this linear accumulation of latency is a
nightmare for production.

### 2. Extreme VRAM Management Optimization

* **KV-Cache Pre-allocation**: Avoids the "idling" and frequent memory copies caused by `torch.cat` after ONNX export.
* **Static Dimension Alignment**: Optimized for TensorRT to ensure stable static execution plans and avoid re-build
  issues caused by dynamic shapes.

---

## 💎 Core Optimizations

### 1. "Surgical" Operator Rewriting

We decoupled the GPT model into two independent computational graphs:

* **`GPTEncoder` (Context Phase)**: Processes prompts and BERT features in one go.
* **`GPTStep` (Decoding Phase)**: Executes single-step decoding with $O(1)$ complexity and sinks **Top-K Sampling** into
  the ONNX graph, drastically reducing GPU->CPU data transfer.

### 2. Full Pipeline Zero-Copy

Utilizing ONNX Runtime's `IOBinding` technology:

* **VRAM Residency**: Input/output are bound directly to VRAM addresses. The `new_k_cache` from the previous round is
  used directly as the next round's input, eliminating PCIe bandwidth bottlenecks.

### 3. Artifact-Free Streaming

Original **Lookahead + History Window** mechanism:

* Performs linear weighted fusion (Cross-Fade) at chunk boundaries, completely eliminating the "clicking" sounds common
  in traditional streaming inference.

---

## 🏁 Quick Start

### 1. Export Model

```bash
python export_onnx.py \
    --gpt_path "pretrained_models\GPT_weights_v2ProPlus/firefly_v2_pp-e25.ckpt"
    --sovits_path "pretrained_models\SoVITS_weights_v2ProPlus/firefly_v2_pp_e10_s590.pth"
    --cnhubert_base_path pretrained_models\chinese-hubert-base
    --bert_path pretrained_models\chinese-roberta-wwm-ext-large
    --output_dir  "onnx_export/firefly_v2_proplus"
    --max_len 1000 # Reducing the size of the GPU can speed up throughput and decrease the pre-allocated video memory, but it requires parameter modification. Generally speaking, 1000 can find a relatively acceptable balance in most scenarios (text of varying lengths).
```

### 2. FP16 Optimization (Optional)

```bash
python onnx_to_fp16.py \
    --input_dir "onnx_export/firefly_v2_proplus" \
    --output_dir "onnx_export/firefly_v2_proplus_fp16"
```

### 3. Run High-Performance Inference

```bash
# Pure streaming inference
python run_onnx_streaming_inference.py \
    --onnx_dir onnx_export/firefly_v2_proplus_fp16 \
    --ref_audio "pretrained_models\看，这尊雕像就是匹诺康尼大名鼎鼎的卡通人物钟表小子.wav" \
    --ref_text "看，这尊雕像就是匹诺康尼大名鼎鼎的卡通人物“钟表小子" \
    --ref_lang "zh" \
    --text "范肖有一项奇特的能力，可以把自己的运气像钱一样攒起来用。攒的越多，越能撞大运。比如攒一个月，就能中彩票。那么，攒到极限会发生什么呢？"
     --lang "zh" --output "out_onnx_stream.wav"

# Launch full-featured WebUI
python run_optimized_inference.py --onnx_dir onnx_export/firefly_v2_proplus_fp16 --webui
```

### ONNX optimize FP16

```bash
# onnx下对fp16的加速不太明显,但是对显存优化拥有极大好处
python onnx_to_fp16.py --input_dir "onnx_export/firefly_v2_proplus" \
  --output_dir "onnx_export/firefly_v2_proplus_fp16"
```

### Export TensorRT Engine

> Note: Compiling TRT engines takes time and must be done for each specific hardware/CUDA/TRT version combination.

```bash
```bash
# Linux
onnx2trt.sh <onnx_input_dir> <output_dir>
# Windows
onnx2trt.bat <onnx_input_dir> <output_dir>
```

---

## 🌐 API Service

If you're tired of staring at the terminal or want your backend to talk to this beast directly, we've squeezed out an **OpenAI-compatible** API service with streaming support. It's basically "Plug and Play".

*   **PyTorch (Stable)**: `python api_server.py` (Port 8000, for the traditionalists)
*   **ONNX (Turbo)**: `python api_server_onnx.py` (Port 8001, CPU users' salvation, easy deployment)
*   **TensorRT (Godspeed)**: `python api_server_trt.py` (Port 8002, GPU screaming, performance peaking)

👉 **[Check the API Documentation](./API_USAGE.md)** — Please, just read the docs. I beg you. Everything is in there.

---

## 🗺️ Roadmap

- [x] **V2 / V2ProPlus** full support
- [x] **TensorRT** static engine acceleration
- [x] **Zero-Copy** IOBinding optimization
- [ ] **Multi-Language Binding**:
    - [ ] C++ SDK (In development)
    - [ ] Rust / Golang / Android Wrapper
- [ ] **V3 / V4** model adaptation
- [ ] **Docker** one-click deployment image

---

## 🤝 Acknowledgments

Special thanks to the [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) team for providing an excellent foundation.
This project aims to push its engineering capabilities even further.

**If this project helps you, please give us a ⭐! It keeps us motivated! 🤗**