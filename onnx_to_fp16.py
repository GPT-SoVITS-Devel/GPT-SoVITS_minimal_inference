import os
import onnx
from onnxconverter_common.float16 import convert_float_to_float16
import argparse

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
