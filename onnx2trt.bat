trtexec --fp16 --onnx=onnx_export\firefly_v2_proplus\bert.onnx --saveEngine=onnx_export\firefly_v2_proplus\bert.engine
trtexec --fp16 --onnx=onnx_export\firefly_v2_proplus\gpt_encoder.onnx --saveEngine=onnx_export\firefly_v2_proplus\gpt_encoder.engine
trtexec --fp16 --onnx=onnx_export\firefly_v2_proplus\gpt_step.onnx --saveEngine=onnx_export\firefly_v2_proplus\gpt_step.engine
trtexec --fp16 --onnx=onnx_export\firefly_v2_proplus\sovits.onnx --saveEngine=onnx_export\firefly_v2_proplus\sovits.engine
trtexec --fp16 --onnx=onnx_export\firefly_v2_proplus\ssl.onnx --saveEngine=onnx_export\firefly_v2_proplus\ssl.engine --minShapes=audio:1x16000 --optShapes=audio:1x160000 --maxShapes=audio:1x800000
trtexec --fp16 --onnx=onnx_export\firefly_v2_proplus\vq_encoder.onnx --saveEngine=onnx_export\firefly_v2_proplus\vq_encoder.engine --minShapes=ssl_content:1x768x50 --optShapes=ssl_content:1x768x500 --maxShapes=ssl_content:1x768x5000
