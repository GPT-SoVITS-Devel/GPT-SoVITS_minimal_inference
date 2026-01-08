#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>" >&2
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

trtexec --fp16 \
    --onnx="${INPUT_DIR}/bert.onnx" \
    --saveEngine="${OUTPUT_DIR}/bert.engine" \
    --minShapes=input_ids:1x1,attention_mask:1x1,token_type_ids:1x1 \
    --optShapes=input_ids:1x128,attention_mask:1x128,token_type_ids:1x128 \
    --maxShapes=input_ids:1x512,attention_mask:1x512,token_type_ids:1x512 \
    --memPoolSize=workspace:2048M

trtexec --fp16 \
    --onnx="${INPUT_DIR}/gpt_encoder.onnx" \
    --saveEngine="${OUTPUT_DIR}/gpt_encoder.engine" \
    --minShapes=phoneme_ids:1x1,prompts:1x1,bert_feature:1x1024x1 \
    --optShapes=phoneme_ids:1x100,prompts:1x50,bert_feature:1x1024x100 \
    --maxShapes=phoneme_ids:1x512,prompts:1x512,bert_feature:1x1024x512 \
    --memPoolSize=workspace:2048M

trtexec --fp16 \
    --onnx="${INPUT_DIR}/gpt_step.onnx" \
    --saveEngine="${OUTPUT_DIR}/gpt_step.engine" \
    --minShapes=samples:1x1,k_cache:24x1x1000x512,v_cache:24x1x1000x512,x_len:1,y_len:1,idx:1 \
    --optShapes=samples:1x1,k_cache:24x1x1000x512,v_cache:24x1x1000x512,x_len:1,y_len:1,idx:1 \
    --maxShapes=samples:1x1,k_cache:24x1x1000x512,v_cache:24x1x1000x512,x_len:1,y_len:1,idx:1 \
    --memPoolSize=workspace:2048M

trtexec --fp16 \
    --onnx="${INPUT_DIR}/sovits.onnx" \
    --saveEngine="${OUTPUT_DIR}/sovits.engine" \
    --minShapes=pred_semantic:1x1x1,text_seq:1x1,refer_spec:1x1025x1 \
    --optShapes=pred_semantic:1x1x200,text_seq:1x100,refer_spec:1x1025x200 \
    --maxShapes=pred_semantic:1x1x1000,text_seq:1x512,refer_spec:1x1025x1000 \
    --memPoolSize=workspace:2048M

trtexec --fp16 \
    --onnx="${INPUT_DIR}/ssl.onnx" \
    --saveEngine="${OUTPUT_DIR}/ssl.engine" \
    --minShapes=audio:1x16000 \
    --optShapes=audio:1x160000 \
    --maxShapes=audio:1x800000 \
    --memPoolSize=workspace:2048M

trtexec --fp16 \
    --onnx="${INPUT_DIR}/vq_encoder.onnx" \
    --saveEngine="${OUTPUT_DIR}/vq_encoder.engine" \
    --minShapes=ssl_content:1x768x50 \
    --optShapes=ssl_content:1x768x500 \
    --maxShapes=ssl_content:1x768x5000 \
    --memPoolSize=workspace:2048M