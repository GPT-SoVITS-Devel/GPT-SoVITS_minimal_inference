import os
import onnx
from onnxconverter_common.float16 import convert_float_to_float16
import argparse

def fix_fp16_types(model_path):
    model = onnx.load(model_path)
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
        onnx.save(model, model_path)
        print(f"  Applied {fixed_count} type fixes to {os.path.basename(model_path)}")

def convert_to_fp16(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".onnx"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f"Converting {filename} to FP16...")
            
            # Load the model
            model = onnx.load(input_path)
            
            # Convert to FP16
            # keep_io_types=True helps maintain compatibility with external inputs/outputs
            model_fp16 = convert_float_to_float16(model, keep_io_types=True)
            
            # Save the model
            onnx.save(model_fp16, output_path)
            
            # Post-process to fix type mismatches introduced by onnxconverter-common
            fix_fp16_types(output_path)
            
            # Copy .data files if they exist (for large models)
            data_file = input_path + ".data"
            if os.path.exists(data_file):
                import shutil
                shutil.copy(data_file, output_path + ".data")
                print(f"Copied {filename}.data")

    print(f"All models in {input_dir} converted to FP16 and saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX models to FP16")
    parser.add_argument("--input_dir", required=True, help="Directory containing float32 ONNX models")
    parser.add_argument("--output_dir", required=True, help="Directory to save FP16 ONNX models")
    
    args = parser.parse_args()
    convert_to_fp16(args.input_dir, args.output_dir)
