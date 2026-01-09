<div align="center">

# ⚡ GPT-SoVITS Minimal Inference

**High-Performance | Production-Ready | Zero-Copy Pipeline**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/) [![GPU](https://img.shields.io/badge/CUDA-11.8+-orange.svg)](https://developer.nvidia.com/cuda-zone) [![ONNX](https://img.shields.io/badge/ONNX-Optimized-brightgreen.svg)](https://onnxruntime.ai/) [![TensorRT](https://img.shields.io/badge/TensorRT-Enabled-76B900.svg)](https://developer.nvidia.com/tensorrt)

[简体中文](./README_zh.md) | [English](./README.md)

**"Not just a refactor, but a deep squeeze of GPT-SoVITS potential."**

---
**Engineered for Speed**: A completely refactored inference engine for GPT-SoVITS, featuring ONNX/TensorRT support,
KV-Cache optimization, and zero-copy streaming.
</div>

---

## 🌟 Core Vision

To solve the performance bottlenecks of GPT-SoVITS in production environments through low-level operator rewriting and
architectural decoupling, without compromising model accuracy or requiring retraining.

We strive for: **Fast**, **Lightweight**, **High Compatibility**, and **Portability**.

## 🚀 Performance Benchmarks

*Environment: I7 12700 | RTX 2080TI (22G) | CUDA 12.9 | FP16 Precision*

| Metric                      | Native PyTorch | ONNX (fp16) | ONNX Stream | TensorRT (FP16)      |
|:----------------------------|:---------------|:------------|:------------|:---------------------|
| **First Token Latency (↓)** | 2.524 s        | 1.983 s     | **1.000 s** | 2.022 s              |
| **Inference Speed (↑)**     | 144.8 tok/s    | 172.4 tok/s | 167.5 tok/s | **291.6 tok/s** (🤯) |
| **RTF (↓)**                 | 0.3434         | 0.3325      | 0.3100      | **0.2096**           |
| **VRAM Usage (↓)**          | 2.8 G          | 3.9 G       | 4.5 G       | 4.8 G                |

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
    --gpt_path "weights/gpt.ckpt" \
    --sovits_path "weights/sovits.pth" \
    --output_dir "onnx_export/optimized" \
    --max_len 1000
```

### 2. FP16 Optimization (Optional)

```bash
python onnx_to_fp16.py \
    --input_dir "onnx_export/optimized" \
    --output_dir "onnx_export/optimized_fp16"
```

### 3. Run High-Performance Inference

```bash
# Pure streaming inference
python run_onnx_streaming_inference.py --onnx_dir "onnx_export/optimized_fp16" --text "Hello, this is a high-speed test."

# Launch full-featured WebUI
python run_optimized_inference.py --webui
```

### 4. Export TensorRT Engine

> Note: Compiling TRT engines takes time and must be done for each specific hardware/CUDA/TRT version combination.

```bash
# Linux
onnx2trt.sh <onnx_input_dir> <output_dir>
# Windows
onnx2trt.bat <onnx_input_dir> <output_dir>
```

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