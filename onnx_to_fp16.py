import os
import onnx
from onnxconverter_common.float16 import convert_float_to_float16
import argparse
from onnxsim import simplify

def fix_fp16_types(model):
    """
    Fixes type mismatches in Cast, Random, and ConstantOfShape nodes
    that often occur after FP16 conversion.
    """
    graph = model.graph
    
    # Map from output name to elem_type in value_info
    name_to_type = {}
    for vi in list(graph.value_info) + list(graph.output) + list(graph.input):
        if vi.type.HasField("tensor_type"):
            name_to_type[vi.name] = vi.type.tensor_type.elem_type
            
    fixed_count = 0
    for node in graph.node:
        output_name = node.output[0] if len(node.output) > 0 else None
        if not output_name or output_name not in name_to_type:
            continue
            
        expected_type = name_to_type[output_name]
        
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to":
                    # If cast 'to' is FP32 but expected is FP16 (or vice versa)
                    if attr.i != expected_type and attr.i in [1, 10] and expected_type in [1, 10]:
                        attr.i = expected_type
                        fixed_count += 1
        
        elif node.op_type in ["RandomNormalLike", "RandomUniformLike", "RandomNormal", "RandomUniform"]:
            for attr in node.attribute:
                if attr.name == "dtype":
                    if attr.i != expected_type and attr.i in [1, 10] and expected_type in [1, 10]:
                        attr.i = expected_type
                        fixed_count += 1
                        
        elif node.op_type == "ConstantOfShape":
            for attr in node.attribute:
                if attr.name == "value":
                    if attr.t.data_type != expected_type and attr.t.data_type in [1, 10] and expected_type in [1, 10]:
                        attr.t.data_type = expected_type
                        fixed_count += 1

    if fixed_count > 0:
        print(f"  Applied {fixed_count} type fixes to model")
    return model

def optimize_and_convert(input_path, output_path):
    print(f"Processing: {os.path.basename(input_path)}")
    
    # 1. Load the model
    model = onnx.load(input_path)
    
    # # 2. Initial simplification (FP32)
    # print("  Simplifying FP32 model...")
    # try:
    #     model, check = simplify(model)
    #     if not check:
    #         print("  Warning: FP32 simplification check failed")
    # except Exception as e:
    #     print(f"  Simplification failed: {e}")

    # 3. Convert to FP16
    print("  Converting to FP16...")
    # keep_io_types=True ensures external interfaces remain compatible (usually FP32 for inputs/outputs if required)
    # Set to False if you want pure FP16 throughout (requires inference script to handle FP16 inputs)
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)
    
    # 4. Fix weight nodes and type mismatches
    model_fp16 = fix_fp16_types(model_fp16)
    
    # 5. Final simplification (FP16)
    print("  Simplifying FP16 model...")
    try:
        model_fp16, check = simplify(model_fp16)
        if not check:
            print("  Warning: FP16 simplification check failed")
    except Exception as e:
        print(f"  FP16 Simplification failed: {e}")

    # 6. Save the model
    onnx.save(model_fp16, output_path)
    print(f"  Saved optimized FP16 model to: {output_path}")

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".onnx"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            optimize_and_convert(input_path, output_path)
            
            # Copy large model data if it exists
            data_file = input_path + ".data"
            if os.path.exists(data_file):
                import shutil
                shutil.copy(data_file, output_path + ".data")
                print(f"  Copied {filename}.data")

    print(f"\nOptimization complete. Models saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize ONNX models and convert to FP16")
    parser.add_argument("--input_dir", required=True, help="Directory containing input ONNX models")
    parser.add_argument("--output_dir", required=True, help="Directory to save optimized FP16 models")
    
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)