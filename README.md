# GPT-SoVITS Minimal Inference: High-Performance & Production-Ready

> **Engineered for Speed**: A completely refactored inference engine for GPT-SoVITS, featuring ONNX/TensorRT support,
> KV-Cache optimization, and zero-copy streaming.

本仓库是 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 的深度重构版本，旨在解决原项目在生产环境部署中的性能瓶颈与架构限制。通过底层算子重写与架构解耦，实现了
**毫秒级首包延迟**与**工业级稳定性**，支持GPT-SoVITS V2/V2 ProPlus。  
本项目的重点方向是在不重新训练的前提下挖掘探索GPT-SoVITS的潜力,优化代码,着重强调：快速，轻量，高兼容，可移植。

---

## 1. 深度分析：原仓库的工程瓶颈 (Why Refactor?)

原项目 (`GPT-SoVITS`) 虽然表现优异，但在尝试工程落地时拥有很多问题：

### 1.1 动态图与 Python 开销 (Dynamic Graph Overhead)

原推理逻辑基于 PyTorch 动态图。在 AR (Auto-Regressive) 解码阶段，每生成一个 Token 都需要经过 Python 解释器的调度开销。对于
500+ Tokens 的长文本，这种开销是线性的，严重制约了吞吐量。

### 1.2 显存管理的低效 (Inefficient VRAM Management)

* **KV Cache 未优化**：原实现的kvcache控制在onnx上的表现是灾难性的，对torchscript导出后的模型进行火焰图分析能发现模型中出现很多莫名其妙的空转,实际上这时流已经转到cpu了。
* **动态扩容**：每次 Step 生成都会导致 KV Cache 的拼接 (`torch.cat`) 与显存重分配，引发碎片化与频繁的内存拷贝，并且这种行为对于导出onnx与trt后跨平台非常不友好。

---

## 2. 核心重构与优化 (Core Refactoring & Optimizations)

针对上述问题，我进行了“手术刀级”的重构：

### 2.1 算子重写与模型拆分 (Operator Rewrite & Disaggregation)

为了支持 ONNX/TensorRT 静态图导出，我重写了 GPT 模型的前向逻辑，将其解耦为两个独立的计算图：

1. **`GPTEncoder` (Context Phase)**:
    * **功能**：一次性处理 Prompt 和 BERT 特征，计算初始 KV Cache。
    * **优化**：引入 **KV Cache 预分配 (Pre-allocation)** 机制。不再动态拼接 Cache，而是预先申请最大长度（如 `max_len=1000`
      ）的显存块，通过 `F.pad` 进行对其。这对 TensorRT 的内存复用至关重要。

2. **`GPTStep` (Decoding Phase)**:
    * **功能**：执行单步解码，$O(1)$ 复杂度。
    * **算子下沉**：将 **Top-K Sampling** 逻辑从 Python 层下沉到了 ONNX 图内部。模型直接输出 `topk_values` 和
      `topk_indices` (Shape: `[B, 50]`) 而非全量 Logits (Shape: `[B, Vocab_Size]`)。
    * **收益**：将每步 GPU->CPU 的数据传输量巨量减少，极大缓解了 PCIe 带宽压力(虽然对于小模型来说这点带宽不算什么)。

### 2.2 全链路 GPU 无 Copy (Zero-Copy Pipeline)

在 `run_onnx_inference.py` 中，利用 ONNX Runtime 的 `IOBinding` 技术实现了全链路显存驻留：

* **输入绑定**：KV Cache 的读写直接在显存地址上进行操作。
* **输出绑定**：上一轮的 `new_k_cache` 直接作为下一轮的 `k_cache` 输入，无需回传 CPU。
* **Fallback 阻断**：严格检查算子兼容性，确保全流程无 CPU Fallback。

```python
# 代码片段：Zero-Copy 的循环推理
io_binding.bind_ortvalue_input("k_cache", k_cache_gpu_ptr)
io_binding.bind_ortvalue_output("k_cache_new", next_k_cache_gpu_ptr)
sess.run_with_iobinding(io_binding)
# 数据全程不离开显存，彻底消除 PCIe 瓶颈
```

### 2.3 TensorRT 兼容性 Hack (TRT Compatibility)

为了让 TensorRT 能成功构建 Engine，我处理了多个“顽疾”：

* **`EuclideanCodebook` 初始化绕过**：在导出时动态替换了 VQ 模型中包含随机初始化的逻辑，防止 TRT 构建失败。
* **静态维度对齐**：通过 Padding 策略规避了 TRT 对动态 Shape 支持不佳导致的 Re-build 问题。

### 2.4 流式推理去伪影 (Artifact-Free Streaming)

传统的流式推理容易在 Chunk 边界产生“咔哒”声。我实现了一套 **Lookahead + History Window** 机制：

* **Lookahead (前瞻)**：解码 Chunk $N$ 时，预先计算 $N+1$ 的 Token 语义。
* **History (回溯)**：保留 $N-1$ 的 KV Cache 和 Acoustic Context。
* **Cross-Fade**：在重叠区域进行线性加权融合，实现了完全无感知的流式拼接。

---

## 3. 性能对比 (Performance Benchmarks)

*测试环境:*

```text
CPU: I7 12700
内存: 128G 3600MHz
显卡: NVIDIA RTX 2080TI 22G
CUDA: 12.9
精度: FP16
```

*测试句子:
范肖有一项奇特的能力，可以把自己的运气像钱一样攒起来用。攒的越多，越能撞大运。比如攒一个月，就能中彩票。那么，攒到极限会发生什么呢？*

| 指标           | 原生 PyTorch      | ONNX(fp16)          | ONNX Stream(fp16)   | TensorRT(FP16)           |
|:-------------|:----------------|:--------------------|:--------------------|:-------------------------|
| **首包延迟 (↓)** | 2.524 s         | **1.983 s**         | **1 s**             | **2.022 s**              |
| **推理速度 (↑)** | 144.83 tokens/s | **172.41 tokens/s** | **167.58 tokens/s** | **291.60 tokens/s** (🤯) |
| **RTF (↓)**  | 0.3434          | **0.3325**          | **0.31**            | **0.2096**               |
| **显存占用 (↓)** | 2.8 G           | **4 G**             | **4.5 G**           | **4.8 G**                |



---

## 4. 使用指南 (Usage)

### 4.1 导出 ONNX 模型

首先需要将 PyTorch Checkpoint 导出为优化后的 ONNX 图：

```bash
python export_onnx.py \
    --gpt_path "GPT_weights/your_gpt.ckpt" \
    --sovits_path "SoVITS_weights/your_sovits.pth" \
    --output_dir "onnx_export/optimized" \
    --max_len 1000 # 后续预分配的长度(不推荐太长,有超长上下文优化,通常来说真正用到的上下文长度不会超过300~500)
```

### 4.2 启动优化版 WebUI

```bash
# onnx下对fp16的加速不太明显,但是对显存优化拥有极大好处
python onnx_to_fp16.py --input_dir "onnx_export/optimized" \
  --output_dir "onnx_export/optimized_fp16"
```

### 4.3 运行推理

```bash
# 纯流式推理
python run_onnx_streaming_inference.py \
    --onnx_dir "onnx_export/optimized_fp16" \
    --ref_audio "ref.wav" \
    --text "你好，这是一段测试文本。"
    # --webui # 或者可以使用webui
```

### 4.4 启动优化版 WebUI

```bash
# 包含所有优化特性的 WebUI
python run_optimized_inference.py --webui
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

## 5. 结语

本项目不仅是一个推理脚本集合，更是一套完整的 **GPT-SoVITS 生产化解决方案**。通过对计算图的深度定制与内存管理的极致优化，证明了在保持
Python 灵活性的同时，也能达到接近 C++ 推理引擎的性能表现。

## 路线图

- [ ] 多语言绑定推理包.
    - [ ] C++(编写中)
    - [ ] Rust
    - [ ] Golang
    - [ ] Android
    - [ ] ...
- [ ] api加速推理
- [ ] windows一键包/docker部署
- [ ] 支持 `V3`,`V4`.

## 感谢

[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)