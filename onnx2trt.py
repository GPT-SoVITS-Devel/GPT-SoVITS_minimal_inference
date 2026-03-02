"""
GPT-SoVITS ONNX → TensorRT Engine 编译脚本

自动检测 GPU 能力，为每个模型选择最优精度策略：
  - INT8 QDQ 模型 → --fp16 --int8 (混合精度，QDQ 节点指导量化)
  - FP16 模型    → --fp16
  - FP8 模型     → --fp8 (仅 Ada/Hopper 架构, compute capability >= 8.9)

Shape profiles 控制动态轴的 max shapes，直接影响构建时显存消耗：
  - fitted : 经过多次profile后得到的调优版本,需要配合分句
  - small  : 适合 <=12GB 显存，单句 TTS（最长 ~6秒）
  - medium : 适合 16-24GB 显存，中等长度（最长 ~15秒）（默认）
  - large  : 适合 >=32GB 显存或分段推理场景（最长 ~40秒）

Usage:
    python onnx2trt.py --input_dir onnx_export/model_fp16 --output_dir onnx_export/model_fp16
    python onnx2trt.py --input_dir onnx_export/model_fp16 --output_dir onnx_export/model_fp16 --shape_profile small
    python onnx2trt.py --input_dir onnx_export/model_int8 --output_dir onnx_export/model_int8 --precision int8 --opt_level 3
"""

import os
import re
import sys
import json
import argparse
import subprocess
import shutil


SHAPE_PROFILES = {
    # small
    # sovits max: sem=150 → y_len=300
    "small": {
        "bert": {
            "min": "input_ids:1x1,attention_mask:1x1,token_type_ids:1x1",
            "opt": "input_ids:1x64,attention_mask:1x64,token_type_ids:1x64",
            "max": "input_ids:1x256,attention_mask:1x256,token_type_ids:1x256",
        },
        "gpt_encoder": {
            "min": "phoneme_ids:1x1,prompts:1x1,bert_feature:1x1024x1",
            "opt": "phoneme_ids:1x64,prompts:1x30,bert_feature:1x1024x64",
            "max": "phoneme_ids:1x256,prompts:1x256,bert_feature:1x1024x256",
        },
        "gpt_step": {
            "min": "samples:1x1,k_cache:24x1x1000x512,v_cache:24x1x1000x512,x_len:1,y_len:1,idx:1",
            "opt": "samples:1x1,k_cache:24x1x1000x512,v_cache:24x1x1000x512,x_len:1,y_len:1,idx:1",
            "max": "samples:1x1,k_cache:24x1x1000x512,v_cache:24x1x1000x512,x_len:1,y_len:1,idx:1",
        },
        "sovits": {
            "min": "pred_semantic:1x1x1,text_seq:1x1,refer_spec:1x1025x1",
            "opt": "pred_semantic:1x1x100,text_seq:1x64,refer_spec:1x1025x200",
            "max": "pred_semantic:1x1x150,text_seq:1x128,refer_spec:1x1025x400",
        },
        "ssl": {
            "min": "audio:1x16000",
            "opt": "audio:1x80000",
            "max": "audio:1x320000",
        },
        "vq_encoder": {
            "min": "ssl_content:1x768x50",
            "opt": "ssl_content:1x768x250",
            "max": "ssl_content:1x768x1000",
        },
        "spectrogram": {
            "min": "audio:1x1",
            "opt": "audio:1x48000",
            "max": "audio:1x480000",
        },
        "sv_embedding": {
            "min": "audio:1x16000",
            "opt": "audio:1x48000",
            "max": "audio:1x160000",
        },
    },

    # fitted: 基于 ShapeProfiler 采集的真实推理数据定制
    # opt shapes 匹配 P95 实测值，max shapes 留 ~2x 安全余量
    # sovits max: sem=250 → y_len=500, 约 10秒音频/segment
    # 参考音频 3-8秒，分句推理的典型场景
    "fitted": {
        "bert": {
            "min": "input_ids:1x1,attention_mask:1x1,token_type_ids:1x1",
            "opt": "input_ids:1x100,attention_mask:1x100,token_type_ids:1x100",
            "max": "input_ids:1x256,attention_mask:1x256,token_type_ids:1x256",
        },
        "gpt_encoder": {
            "min": "phoneme_ids:1x1,prompts:1x1,bert_feature:1x1024x1",
            "opt": "phoneme_ids:1x100,prompts:1x150,bert_feature:1x1024x100",
            "max": "phoneme_ids:1x256,prompts:1x300,bert_feature:1x1024x256",
        },
        "gpt_step": {
            "min": "samples:1x1,k_cache:24x1x1000x512,v_cache:24x1x1000x512,x_len:1,y_len:1,idx:1",
            "opt": "samples:1x1,k_cache:24x1x1000x512,v_cache:24x1x1000x512,x_len:1,y_len:1,idx:1",
            "max": "samples:1x1,k_cache:24x1x1000x512,v_cache:24x1x1000x512,x_len:1,y_len:1,idx:1",
        },
        "sovits": {
            "min": "pred_semantic:1x1x1,text_seq:1x1,refer_spec:1x1025x1",
            "opt": "pred_semantic:1x1x120,text_seq:1x50,refer_spec:1x1025x280",
            "max": "pred_semantic:1x1x250,text_seq:1x100,refer_spec:1x1025x400",
        },
        "ssl": {
            "min": "audio:1x16000",
            "opt": "audio:1x96000",
            "max": "audio:1x200000",
        },
        "vq_encoder": {
            "min": "ssl_content:1x768x50",
            "opt": "ssl_content:1x768x300",
            "max": "ssl_content:1x768x700",
        },
        "spectrogram": {
            "min": "audio:1x1",
            "opt": "audio:1x180000",
            "max": "audio:1x400000",
        },
        "sv_embedding": {
            "min": "audio:1x16000",
            "opt": "audio:1x90000",
            "max": "audio:1x180000",
        },
    },

    # medium 默认
    # sovits max: sem=400 → y_len=800
    "medium": {
        "bert": {
            "min": "input_ids:1x1,attention_mask:1x1,token_type_ids:1x1",
            "opt": "input_ids:1x128,attention_mask:1x128,token_type_ids:1x128",
            "max": "input_ids:1x512,attention_mask:1x512,token_type_ids:1x512",
        },
        "gpt_encoder": {
            "min": "phoneme_ids:1x1,prompts:1x1,bert_feature:1x1024x1",
            "opt": "phoneme_ids:1x100,prompts:1x50,bert_feature:1x1024x100",
            "max": "phoneme_ids:1x512,prompts:1x512,bert_feature:1x1024x512",
        },
        "gpt_step": {
            "min": "samples:1x1,k_cache:24x1x1000x512,v_cache:24x1x1000x512,x_len:1,y_len:1,idx:1",
            "opt": "samples:1x1,k_cache:24x1x1000x512,v_cache:24x1x1000x512,x_len:1,y_len:1,idx:1",
            "max": "samples:1x1,k_cache:24x1x1000x512,v_cache:24x1x1000x512,x_len:1,y_len:1,idx:1",
        },
        "sovits": {
            "min": "pred_semantic:1x1x1,text_seq:1x1,refer_spec:1x1025x1",
            "opt": "pred_semantic:1x1x200,text_seq:1x100,refer_spec:1x1025x200",
            "max": "pred_semantic:1x1x400,text_seq:1x256,refer_spec:1x1025x400",
        },
        "ssl": {
            "min": "audio:1x16000",
            "opt": "audio:1x160000",
            "max": "audio:1x480000",
        },
        "vq_encoder": {
            "min": "ssl_content:1x768x50",
            "opt": "ssl_content:1x768x500",
            "max": "ssl_content:1x768x2000",
        },
        "spectrogram": {
            "min": "audio:1x1",
            "opt": "audio:1x48000",
            "max": "audio:1x480000",
        },
        "sv_embedding": {
            "min": "audio:1x16000",
            "opt": "audio:1x48000",
            "max": "audio:1x160000",
        },
    },

    # large
    # sovits max: sem=1000 → y_len=2000
    "large": {
        "bert": {
            "min": "input_ids:1x1,attention_mask:1x1,token_type_ids:1x1",
            "opt": "input_ids:1x128,attention_mask:1x128,token_type_ids:1x128",
            "max": "input_ids:1x512,attention_mask:1x512,token_type_ids:1x512",
        },
        "gpt_encoder": {
            "min": "phoneme_ids:1x1,prompts:1x1,bert_feature:1x1024x1",
            "opt": "phoneme_ids:1x100,prompts:1x50,bert_feature:1x1024x100",
            "max": "phoneme_ids:1x512,prompts:1x512,bert_feature:1x1024x512",
        },
        "gpt_step": {
            "min": "samples:1x1,k_cache:24x1x1000x512,v_cache:24x1x1000x512,x_len:1,y_len:1,idx:1",
            "opt": "samples:1x1,k_cache:24x1x1000x512,v_cache:24x1x1000x512,x_len:1,y_len:1,idx:1",
            "max": "samples:1x1,k_cache:24x1x1000x512,v_cache:24x1x1000x512,x_len:1,y_len:1,idx:1",
        },
        "sovits": {
            "min": "pred_semantic:1x1x1,text_seq:1x1,refer_spec:1x1025x1",
            "opt": "pred_semantic:1x1x200,text_seq:1x100,refer_spec:1x1025x200",
            "max": "pred_semantic:1x1x1000,text_seq:1x512,refer_spec:1x1025x1000",
        },
        "ssl": {
            "min": "audio:1x16000",
            "opt": "audio:1x160000",
            "max": "audio:1x800000",
        },
        "vq_encoder": {
            "min": "ssl_content:1x768x50",
            "opt": "ssl_content:1x768x500",
            "max": "ssl_content:1x768x5000",
        },
        "spectrogram": {
            "min": "audio:1x1",
            "opt": "audio:1x48000",
            "max": "audio:1x960000",
        },
        "sv_embedding": {
            "min": "audio:1x16000",
            "opt": "audio:1x48000",
            "max": "audio:1x160000",
        },
    },
}

MODEL_SHAPES = SHAPE_PROFILES["medium"]


def detect_gpu_capability():
    """检测 GPU compute capability，决定可用精度"""
    capabilities = {
        "fp16": False,
        "int8": False,
        "fp8": False,
        "compute_capability": (0, 0),
        "gpu_name": "Unknown",
    }

    try:
        import torch
        if not torch.cuda.is_available():
            print("[Warn] No CUDA GPU detected. Defaulting to FP16 only.")
            capabilities["fp16"] = True
            return capabilities

        props = torch.cuda.get_device_properties(0)
        cc = (props.major, props.minor)
        capabilities["compute_capability"] = cc
        capabilities["gpu_name"] = props.name

        # FP16: compute capability >= 5.3 (Maxwell+)
        capabilities["fp16"] = cc >= (5, 3)
        # INT8: compute capability >= 6.1 (Pascal+)
        capabilities["int8"] = cc >= (6, 1)
        # FP8: compute capability >= 8.9 (Ada Lovelace / Hopper)
        capabilities["fp8"] = cc >= (8, 9)

    except ImportError:
        # 没有 torch，尝试 nvidia-smi
        print("[Warn] PyTorch not available for GPU detection. Assuming FP16 + INT8 support.")
        capabilities["fp16"] = True
        capabilities["int8"] = True

    return capabilities


def has_qdq_nodes(onnx_path):
    """检测 ONNX 模型是否包含 QDQ 量化节点"""
    try:
        import onnx
        model = onnx.load(onnx_path, load_external_data=False)
        for node in model.graph.node:
            if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
                return True
        return False
    except Exception:
        return False


def get_precision_flags(model_name, onnx_path, gpu_caps, user_precision):
    """为单个模型决定 trtexec 精度标志

    返回: (flags_list, precision_label)
    """
    is_qdq = has_qdq_nodes(onnx_path)

    if user_precision == "fp8":
        if not gpu_caps["fp8"]:
            print(f"  [Warn] {model_name}: FP8 not supported on {gpu_caps['gpu_name']} "
                  f"(CC {gpu_caps['compute_capability'][0]}.{gpu_caps['compute_capability'][1]}). "
                  f"Falling back to {'INT8' if is_qdq and gpu_caps['int8'] else 'FP16'}.")
            if is_qdq and gpu_caps["int8"]:
                return ["--fp16", "--int8"], "FP16+INT8 (QDQ)"
            return ["--fp16"], "FP16"
        return ["--fp16", "--fp8"], "FP16+FP8"

    if user_precision == "int8":
        if is_qdq and gpu_caps["int8"]:
            return ["--fp16", "--int8"], "FP16+INT8 (QDQ)"
        elif not is_qdq:
            print(f"  [Info] {model_name}: No QDQ nodes found, using FP16.")
            return ["--fp16"], "FP16"
        else:
            print(f"  [Warn] {model_name}: INT8 not supported on this GPU. Using FP16.")
            return ["--fp16"], "FP16"

    if user_precision == "fp16":
        return ["--fp16"], "FP16"

    # auto 模式: 根据模型内容和 GPU 能力自动选择
    if is_qdq and gpu_caps["int8"]:
        return ["--fp16", "--int8"], "FP16+INT8 (QDQ, auto)"
    return ["--fp16"], "FP16 (auto)"


def build_trtexec_cmd(onnx_path, engine_path, model_key, precision_flags,
                      workspace_mb=2048, opt_level=None, timing_cache=None,
                      shape_dict=None):
    """构建 trtexec 命令"""
    cmd = ["trtexec"]
    cmd.extend(precision_flags)
    cmd.extend([f"--onnx={onnx_path}", f"--saveEngine={engine_path}"])

    shapes = shape_dict or MODEL_SHAPES
    if model_key in shapes:
        s = shapes[model_key]
        cmd.append(f"--minShapes={s['min']}")
        cmd.append(f"--optShapes={s['opt']}")
        cmd.append(f"--maxShapes={s['max']}")

    cmd.append(f"--memPoolSize=workspace:{workspace_mb}M")

    if opt_level is not None:
        cmd.append(f"--builderOptimizationLevel={opt_level}")

    if timing_cache:
        cmd.append(f"--timingCacheFile={timing_cache}")

    return cmd


def find_trtexec():
    """查找 trtexec 可执行文件"""
    # 1. PATH 中查找
    trtexec = shutil.which("trtexec")
    if trtexec:
        return trtexec

    # 2. 常见安装路径
    common_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\bin\trtexec.exe",
        "/usr/bin/trtexec",
        "/usr/local/bin/trtexec",
    ]
    # TensorRT pip 包路径
    try:
        import tensorrt
        trt_dir = os.path.dirname(tensorrt.__file__)
        common_paths.append(os.path.join(trt_dir, "trtexec"))
        common_paths.append(os.path.join(trt_dir, "trtexec.exe"))
        # pip 安装的 tensorrt 通常在 site-packages/tensorrt/
        bin_dir = os.path.join(os.path.dirname(trt_dir), "tensorrt_libs")
        common_paths.append(os.path.join(bin_dir, "trtexec"))
        common_paths.append(os.path.join(bin_dir, "trtexec.exe"))
    except ImportError:
        pass

    for p in common_paths:
        if os.path.isfile(p):
            return p

    return None


def get_gpu_vram_mb():
    """获取 GPU 显存大小（MB），用于自动选择 shape profile"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except ImportError:
        pass
    return 0


def auto_select_profile(vram_mb):
    """根据显存自动选择 shape profile"""
    if vram_mb <= 0:
        return "medium"
    if vram_mb <= 12288:      # <=12GB
        return "small"
    elif vram_mb <= 26624:    # <=26GB (24GB 卡 + 余量)
        return "medium"
    else:
        return "large"


def patch_gpt_step_shapes(shape_dict, max_len):
    """根据导出时的 max_len 修正 gpt_step 的 KV cache 静态维度。

    GPTEncoder 导出时将 KV cache pad 到 [layers, B, max_len, hidden_dim]，
    其中 dim 2 是静态的。TRT 的 shape 配置必须与 ONNX 中的静态维度一致，
    否则构建会失败或产生错误的 engine。
    """
    if "gpt_step" not in shape_dict:
        return shape_dict

    old = shape_dict["gpt_step"]["min"]
    # 从现有 shape 字符串中提取 layers 和 hidden_dim
    # 格式: "samples:1x1,k_cache:24x1x1000x512,v_cache:24x1x1000x512,..."
    m = re.search(r'k_cache:(\d+)x(\d+)x(\d+)x(\d+)', old)
    if not m:
        return shape_dict

    layers, batch, old_len, hidden = m.group(1), m.group(2), m.group(3), m.group(4)
    if int(old_len) == max_len:
        return shape_dict

    print(f"  [Patch] gpt_step KV cache dim2: {old_len} → {max_len} (from config.json max_len)")

    patched = dict(shape_dict)
    patched["gpt_step"] = {}
    for level in ("min", "opt", "max"):
        s = shape_dict["gpt_step"][level]
        s = re.sub(
            r'(k_cache:\d+x\d+x)\d+(x\d+)',
            rf'\g<1>{max_len}\2', s
        )
        s = re.sub(
            r'(v_cache:\d+x\d+x)\d+(x\d+)',
            rf'\g<1>{max_len}\2', s
        )
        patched["gpt_step"][level] = s

    return patched


def process_directory(input_dir, output_dir, precision="auto", workspace_mb=2048,
                      shape_profile="auto", opt_level=None, timing_cache=None):
    # 设置环境变量优化显存占用
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    # 检测 GPU
    gpu_caps = detect_gpu_capability()
    cc = gpu_caps["compute_capability"]
    vram_mb = get_gpu_vram_mb()
    print(f"GPU: {gpu_caps['gpu_name']} (CC {cc[0]}.{cc[1]}, VRAM {vram_mb}MB)")
    print(f"  FP16: {'Yes' if gpu_caps['fp16'] else 'No'}")
    print(f"  INT8: {'Yes' if gpu_caps['int8'] else 'No'}")
    print(f"  FP8:  {'Yes' if gpu_caps['fp8'] else 'No'}")

    # 选择 shape profile
    if shape_profile == "auto":
        shape_profile = auto_select_profile(vram_mb)
        print(f"  Shape profile: {shape_profile} (auto-selected for {vram_mb}MB VRAM)")
    else:
        print(f"  Shape profile: {shape_profile}")

    shape_dict = dict(SHAPE_PROFILES[shape_profile])

    # 从 config.json 读取 max_len，修正 gpt_step KV cache 维度
    config_path = os.path.join(input_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        max_len = config.get("data", {}).get("max_len", 1000)
        print(f"  KV cache max_len: {max_len} (from config.json)")
        shape_dict = patch_gpt_step_shapes(shape_dict, max_len)
    else:
        print(f"  [Warn] config.json not found in {input_dir}, using default gpt_step shapes")

    if opt_level is not None:
        print(f"  Builder optimization level: {opt_level}")
    if timing_cache:
        print(f"  Timing cache: {timing_cache}")
    print()

    # 查找 trtexec
    trtexec_path = find_trtexec()
    if not trtexec_path:
        print("[ERROR] trtexec not found. Please install TensorRT and ensure trtexec is in PATH.")
        sys.exit(1)
    print(f"Using trtexec: {trtexec_path}\n")

    os.makedirs(output_dir, exist_ok=True)

    # timing cache 路径
    cache_path = timing_cache
    if cache_path is None:
        cache_path = os.path.join(output_dir, "timing.cache")
    print(f"Timing cache: {cache_path} (shared across all models)\n")

    # 收集所有 onnx 文件
    onnx_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".onnx")])
    if not onnx_files:
        print(f"[ERROR] No .onnx files found in {input_dir}")
        sys.exit(1)

    results = []

    for filename in onnx_files:
        onnx_path = os.path.join(input_dir, filename)
        engine_name = filename.replace(".onnx", ".engine")
        engine_path = os.path.join(output_dir, engine_name)

        # 匹配模型 key
        model_key = None
        for key in shape_dict:
            if key in filename:
                model_key = key
                break

        if model_key is None:
            print(f"[Warn] Unknown model: {filename}, skipping.")
            results.append((filename, "Skipped", "Unknown model"))
            continue

        # 决定精度
        precision_flags, precision_label = get_precision_flags(
            filename, onnx_path, gpu_caps, precision
        )

        sovits_max = shape_dict.get(model_key, {}).get("max", "")
        print(f"Building: {filename} → {engine_name}")
        print(f"  Precision: {precision_label} | Max: {sovits_max}")

        # 构建命令
        cmd = build_trtexec_cmd(
            onnx_path, engine_path, model_key, precision_flags,
            workspace_mb=workspace_mb, opt_level=opt_level,
            timing_cache=cache_path, shape_dict=shape_dict,
        )
        cmd[0] = trtexec_path

        print(f"  Command: {' '.join(cmd)}")

        try:
            env = os.environ.copy()
            env.setdefault("CUDA_MODULE_LOADING", "LAZY")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
            if result.returncode == 0:
                engine_size = os.path.getsize(engine_path) / (1024 * 1024)
                results.append((filename, "OK", f"{precision_label} ({engine_size:.1f}MB)"))
                print(f"  Done: {engine_size:.1f}MB\n")
            else:
                stderr_lines = result.stderr.strip().split("\n")[-10:]
                stdout_lines = result.stdout.strip().split("\n")[-5:]
                print(f"  [FAILED] Return code: {result.returncode}")
                for line in stderr_lines + stdout_lines:
                    print(f"    {line}")
                results.append((filename, "FAILED", f"rc={result.returncode}"))
                print()
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] Build exceeded 60 minutes")
            results.append((filename, "TIMEOUT", ""))
            print()
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append((filename, "ERROR", str(e)))
            print()

    # 复制 config.json
    config_src = os.path.join(input_dir, "config.json")
    if os.path.exists(config_src):
        shutil.copy(config_src, os.path.join(output_dir, "config.json"))

    # 汇总
    print(f"\n{'='*70}")
    print(f"{'Model':<25} | {'Status':<10} | Details")
    print(f"{'-'*25}-+-{'-'*10}-+-{'-'*30}")
    for name, status, detail in results:
        print(f"{name:<25} | {status:<10} | {detail}")
    print(f"{'='*70}")

    ok_count = sum(1 for _, s, _ in results if s == "OK")
    total = len(results)
    print(f"\n{ok_count}/{total} engines built successfully → {output_dir}")
    if ok_count < total:
        failed = [n for n, s, _ in results if s != "OK"]
        print(f"\nTips for failed models:")
        print(f"  - Try a smaller shape profile: --shape_profile small")
        print(f"  - Try lower optimization level: --opt_level 2")
        print(f"  - Reduce workspace: --workspace 1024")
        print(f"  - Failed: {', '.join(failed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPT-SoVITS ONNX → TensorRT Engine (auto precision & shape detection)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Precision modes:
  auto  - Auto-detect: QDQ models → INT8, others → FP16 (default)
  fp16  - Force FP16 for all models
  int8  - QDQ models → INT8+FP16, non-QDQ → FP16
  fp8   - FP8+FP16 (requires Ada/Hopper GPU, CC >= 8.9)

Shape profiles (控制 max shapes，直接影响构建时显存消耗):
  auto   - 根据 GPU 显存自动选择 (default)
  small  - <=12GB VRAM, sovits max sem=150 (~6s 音频)
  fitted - 基于 ShapeProfiler 实测数据定制, sovits max sem=250 (~10s)
  medium - 16-24GB VRAM, sovits max sem=400 (~16s 音频)
  large  - >=32GB VRAM, sovits max sem=1000 (~40s 音频)

Builder optimization level (TRT 8.6+):
  0-5, 越低构建越快、显存越少，但运行时性能可能略差
  默认不设置 (TRT 自己选择，通常为 3)
  显存不足时建议设为 2

Examples:
  # 自动检测，适用于大多数场景
  python onnx2trt.py --input_dir onnx_export/model_fp16 --output_dir onnx_export/model_fp16

  # 显存紧张 (<=24GB)，使用保守配置
  python onnx2trt.py --input_dir onnx_export/model_fp16 --output_dir onnx_export/model_fp16 --shape_profile small --opt_level 2 --workspace 1024

  # 大显存卡，最大化推理能力
  python onnx2trt.py --input_dir onnx_export/model_fp16 --output_dir onnx_export/model_fp16 --shape_profile large --opt_level 4
        """
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing ONNX models")
    parser.add_argument("--output_dir", required=True, help="Output directory for TRT engines")
    parser.add_argument("--precision", default="auto", choices=["auto", "fp16", "int8", "fp8"],
                        help="Precision mode (default: auto)")
    parser.add_argument("--workspace", type=int, default=2048, help="Workspace size in MB (default: 2048)")
    parser.add_argument("--shape_profile", default="fitted", choices=["auto", "small", "fitted", "medium", "large"],
                        help="Shape profile controlling max dynamic shapes (default: fitted)")
    parser.add_argument("--opt_level", type=int, default=None, choices=[0, 1, 2, 3, 4, 5],
                        help="TRT builder optimization level 0-5 (default: TRT default ~3)")
    parser.add_argument("--timing_cache", default=None,
                        help="Path to timing cache file (default: <output_dir>/timing.cache)")

    args = parser.parse_args()

    print("=" * 60)
    print("GPT-SoVITS ONNX → TensorRT Engine Builder")
    print(f"  Input:          {args.input_dir}")
    print(f"  Output:         {args.output_dir}")
    print(f"  Precision:      {args.precision}")
    print(f"  Shape profile:  {args.shape_profile}")
    print(f"  Workspace:      {args.workspace}MB")
    print(f"  Opt level:      {args.opt_level if args.opt_level is not None else 'default'}")
    print(f"  Timing cache:   {args.timing_cache or 'auto'}")
    print("=" * 60 + "\n")

    process_directory(
        args.input_dir, args.output_dir, args.precision, args.workspace,
        shape_profile=args.shape_profile, opt_level=args.opt_level,
        timing_cache=args.timing_cache,
    )
