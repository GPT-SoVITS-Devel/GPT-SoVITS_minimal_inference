import os
import onnx
import onnx.helper
from onnx import TensorProto
from onnxconverter_common.float16 import convert_float_to_float16
import argparse
from onnxsim import simplify
import numpy as np

# --- 配置区 ---
MODEL_CONFIGS = {
    "vq_encoder": {"fp16": False, "sensitive": []},
    "bert": {"fp16": True, "sensitive": ["LayerNormalization", "Mean"]},
    "ssl": {"fp16": True, "sensitive": ["LayerNormalization", "Mean"]},
    "gpt_encoder": {"fp16": True, "sensitive": ["Pow", "Exp", "Mean", "ReduceMean", "LayerNormalization"]},
    "gpt_step": {"fp16": True, "sensitive": ["Pow", "Exp", "MatMulInteger", "LayerNormalization"]},
    "sovits": {"fp16": True, "sensitive": ["InstanceNormalization", "Resize", "Mean", "Sum", "Exp"]},
}

# 全局通用黑名单
GLOBAL_SENSITIVE_OPS = [
    "Softmax",
    "LayerNormalization",
    "InstanceNormalization",
    "ReduceMean",
    "Pow",
    "Exp",
    "Resize",
    "Mean",
    "Sum",
]


def get_tensor_type(name, type_map, initializer_map, graph_input_map):
    if name in initializer_map: return initializer_map[name]
    if name in graph_input_map: return graph_input_map[name]
    if name in type_map: return type_map[name]
    return TensorProto.UNDEFINED


def fix_mixed_types_robust(model):
    """
    用于修复 GPT/SoVITS 转换 FP16 后遗留的类型不匹配问题
    """
    initializer_map = {init.name: init.data_type for init in model.graph.initializer}
    graph_input_map = {}
    for inp in model.graph.input:
        if inp.type.HasField("tensor_type"):
            graph_input_map[inp.name] = inp.type.tensor_type.elem_type

    try:
        model = onnx.shape_inference.infer_shapes(model)
    except:
        pass

    type_map = {}
    for vi in model.graph.value_info:
        if vi.type.HasField("tensor_type"):
            type_map[vi.name] = vi.type.tensor_type.elem_type

    new_nodes = []
    ops_to_check = ["MatMul", "Gemm", "Conv"]

    for node in model.graph.node:
        if node.op_type in ops_to_check and len(node.input) >= 2:
            data_name = node.input[0]
            weight_name = node.input[1]
            t_data = get_tensor_type(data_name, type_map, initializer_map, graph_input_map)
            t_weight = get_tensor_type(weight_name, type_map, initializer_map, graph_input_map)

            need_cast = False
            # 权重是 FP16 但数据是 FP32 或 未知 -> 强制 Cast 数据
            if t_weight == TensorProto.FLOAT16 and (t_data == TensorProto.FLOAT or t_data == TensorProto.UNDEFINED):
                need_cast = True

            if need_cast:
                cast_name = f"{data_name}_cast_fp16_fix_{node.name}"
                if any(n.name == cast_name for n in new_nodes): cast_name += "_dup"
                cast_node = onnx.helper.make_node(
                    "Cast", inputs=[data_name], outputs=[cast_name],
                    to=TensorProto.FLOAT16, name=cast_name
                )
                new_nodes.append(cast_node)
                node.input[0] = cast_name

    if new_nodes:
        model.graph.node.extend(new_nodes)
        print(f"    [Robust Fix] Inserted {len(new_nodes)} Cast nodes.")
    return model


def fix_broken_attributes(model):
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except:
        print("    [Warn] Shape inference failed inside fix_broken_attributes, relying on partial info.")

    # 构建类型映射 (Name -> DataType)
    type_map = {}
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.type.HasField("tensor_type"):
            type_map[vi.name] = vi.type.tensor_type.elem_type

    cnt = 0
    # 需要检查属性的算子列表
    random_ops = ["RandomNormal", "RandomUniform", "RandomNormalLike", "RandomUniformLike"]

    for node in model.graph.node:
        # --- 修复 Random 系列算子 ---
        if node.op_type in random_ops:
            out_name = node.output[0]
            # 只有当我们确切知道该输出应该是 FP16 时才动手
            if out_name in type_map:
                real_dtype = type_map[out_name]

                # 检查是否已有 dtype 属性
                found_dtype = False
                for attr in node.attribute:
                    if attr.name == "dtype":
                        found_dtype = True
                        if attr.i != real_dtype:
                            attr.i = real_dtype  # 强制修正属性
                            cnt += 1

                # 如果没有 dtype 属性，且输出要是 FP16，必须显式添加 dtype=10 (FLOAT16)
                # 因为 RandomNormal 默认通常是 Float(1)
                if not found_dtype and real_dtype == TensorProto.FLOAT16:
                    new_attr = onnx.helper.make_attribute("dtype", TensorProto.FLOAT16)
                    node.attribute.extend([new_attr])
                    cnt += 1

        # --- 修复 Cast 算子 ---
        elif node.op_type == "Cast":
            out_name = node.output[0]
            if out_name in type_map:
                real_dtype = type_map[out_name]
                for attr in node.attribute:
                    if attr.name == "to" and attr.i != real_dtype:
                        attr.i = real_dtype
                        cnt += 1

    if cnt > 0:
        print(f"    [Attribute Fix] Fixed {cnt} attributes (Random/Cast mismatch).")
    return model


def optimize_single_model(input_path, output_path):
    filename = os.path.basename(input_path)
    model_name_key = None

    # 匹配策略
    for key in MODEL_CONFIGS:
        if key in filename:
            model_name_key = key
            break

    # 默认策略：如果不匹配（如 unknown.onnx），默认保持 FP32 以求稳
    config = MODEL_CONFIGS.get(model_name_key, {"fp16": False, "sensitive": []})

    print(f"Processing: {filename} | Strategy: {'FP16' if config['fp16'] else 'FP32 (Keep)'}")

    model = onnx.load(input_path)

    # 如果启用 FP16，执行转换和修复
    if config["fp16"]:
        print("  Converting to FP16...")
        block_list = GLOBAL_SENSITIVE_OPS + config["sensitive"]
        model = convert_float_to_float16(
            model,
            keep_io_types=False,
            op_block_list=block_list
        )
        # 仅在 FP16 模式下需要修复混合精度
        model = fix_mixed_types_robust(model)
        model = fix_broken_attributes(model)
    else:
        print("  Skipping FP16 conversion (Sensitivity/Low-Cost).")

    # 通用 Simplification (无论 FP16 还是 FP32 都需要简化)
    print("  Simplifying...")
    try:
        model, check = simplify(model)
    except Exception as e:
        print(f"  [Warn] Simplify failed/warned: {e}")

    onnx.save(model, output_path)
    print(f"  Saved: {output_path}")

import shutil
def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".onnx"):
            optimize_single_model(
                os.path.join(input_dir, filename),
                os.path.join(output_dir, filename)
            )
            # 复制 .data 文件 (如果有)
            dfile = os.path.join(input_dir, filename + ".data")
            if os.path.exists(dfile):

                shutil.copy(dfile, os.path.join(output_dir, filename + ".data"))
    shutil.copy(os.path.join(input_dir, "config.json"),os.path.join(output_dir, "config.json"))
    print(f"\nOptimization complete: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)
