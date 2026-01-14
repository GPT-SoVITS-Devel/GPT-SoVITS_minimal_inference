import os
import sys
import torch
import numpy as np
import io
import json
import wave
import logging
import argparse
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gpt-sovits-onnx-api")

# Add current and GPT_SoVITS paths
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "GPT_SoVITS"))

try:
    from run_onnx_streaming_inference import GPTSoVITS_ONNX_Streaming_Inference
except ImportError:
    logger.error("Failed to import GPTSoVITS_ONNX_Streaming_Inference. Ensure run_onnx_streaming_inference.py exists.")
    raise

# --- Config Management ---
parser = argparse.ArgumentParser(description="GPT-SoVITS ONNX API Server", add_help=False)
parser.add_argument("--bert_path", default=os.environ.get("BERT_PATH", "pretrained_models/chinese-roberta-wwm-ext-large"))
parser.add_argument("--voices_config", default=os.environ.get("VOICES_CONFIG", "config/voices.json"))
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
args, _ = parser.parse_known_args()

class ModelManager:
    """Manages loading and caching of GPT-SoVITS ONNX engines."""
    def __init__(self, bert_path: str, device: str):
        self.engines: Dict[str, GPTSoVITS_ONNX_Streaming_Inference] = {}
        self.bert_path = bert_path
        self.device = device

    def get_engine(self, onnx_dir: str) -> GPTSoVITS_ONNX_Streaming_Inference:
        if onnx_dir not in self.engines:
            logger.info(f"Loading new ONNX model engine from: {onnx_dir}")
            full_onnx_dir = onnx_dir if os.path.isabs(onnx_dir) else os.path.normpath(os.path.join(cwd, onnx_dir))
            
            engine = GPTSoVITS_ONNX_Streaming_Inference(
                onnx_dir=full_onnx_dir,
                bert_path=os.path.normpath(os.path.join(cwd, self.bert_path)),
                device=self.device
            )
            self.engines[onnx_dir] = engine
        return self.engines[onnx_dir]

class VoiceManager:
    """Manages voice configurations from voices.json."""
    def __init__(self, config_path: str):
        self.config_path = os.path.normpath(os.path.join(cwd, config_path))
        self.voices: Dict[str, Any] = {}
        self.load_configs()

    def load_configs(self):
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file {self.config_path} not found.")
            self.voices = {}
        else:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.voices = json.load(f)
        logger.info(f"Loaded {len(self.voices)} voice configurations.")

    def get_voice(self, name: str) -> Dict[str, Any]:
        if name not in self.voices:
            self.load_configs()
        if name not in self.voices:
            raise HTTPException(status_code=404, detail=f"Voice '{name}' not found.")
        return self.voices[name]

# Initialize global managers
model_manager = ModelManager(args.bert_path, args.device)
voice_manager = VoiceManager(args.voices_config)

app = FastAPI(title="GPT-SoVITS ONNX API")

class SpeechRequest(BaseModel):
    input: str
    voice: str = "default"
    model: str = "gpt-sovits-v2"
    response_format: str = "wav"
    speed: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    text_lang: str = "auto"
    chunk_length: int = 24
    pause_length: Optional[float] = None
    noise_scale: Optional[float] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    ref_lang: Optional[str] = None

def audio_array_to_wav_chunk(audio_data: np.ndarray, sr: int):
    audio_int16 = (audio_data * 32767).astype(np.int16)
    return audio_int16.tobytes()

@app.post("/v1/audio/speech")
async def text_to_speech(request: SpeechRequest):
    voice_config = voice_manager.get_voice(request.voice)
    defaults = voice_config.get("defaults", {})
    
    speed = request.speed if request.speed is not None else defaults.get("speed", 1.0)
    top_k = request.top_k if request.top_k is not None else defaults.get("top_k", 15)
    temperature = request.temperature if request.temperature is not None else defaults.get("temperature", 1.0)
    pause_length = request.pause_length if request.pause_length is not None else defaults.get("pause_length", 0.3)
    noise_scale = request.noise_scale if request.noise_scale is not None else defaults.get("noise_scale", 0.35)
    
    ref_audio = request.ref_audio or voice_config.get("ref_audio")
    ref_text = request.ref_text or voice_config.get("ref_text")
    ref_lang = request.ref_lang or voice_config.get("ref_lang", "zh")

    onnx_path = voice_config.get("onnx_path")
    if not onnx_path:
        raise HTTPException(status_code=400, detail="onnx_path is missing in voice config.")

    if not ref_audio:
        raise HTTPException(status_code=400, detail="Reference audio path is missing.")

    try:
        engine = model_manager.get_engine(onnx_path)
    except Exception as e:
        logger.error(f"Error loading ONNX engine: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load ONNX model: {str(e)}")

    def generate_audio():
        try:
            abs_ref_audio = ref_audio if os.path.isabs(ref_audio) else os.path.join(cwd, ref_audio)
            # Use infer_stream for ONNX
            gen = engine.infer_stream(
                ref_wav_path=abs_ref_audio, prompt_text=ref_text, prompt_lang=ref_lang,
                text=request.input, text_lang=request.text_lang, top_k=top_k,
                temperature=temperature, speed=speed, chunk_length=request.chunk_length,
                noise_scale=noise_scale, pause_length=pause_length
            )
            sr = engine.hps["data"]["sampling_rate"]
            if request.response_format == "wav":
                header = io.BytesIO()
                with wave.open(header, "wb") as wav_file:
                    wav_file.setnchannels(1); wav_file.setsampwidth(2); wav_file.setframerate(sr); wav_file.writeframes(b"")
                yield header.getvalue()
            for chunk in gen:
                if chunk is not None and len(chunk) > 0:
                    yield audio_array_to_wav_chunk(chunk, sr)
        except Exception as e:
            logger.error(f"ONNX Inference error: {e}")

    return StreamingResponse(generate_audio(), media_type="audio/wav" if request.response_format == "wav" else "audio/mpeg")

@app.get("/v1/models")
async def list_models():
    models_list = [{"id": name, "object": "model", "created": 1700000000, "owned_by": "gpt-sovits-onnx", "description": config.get("description", "")} 
                   for name, config in voice_manager.voices.items()]
    return {"object": "list", "data": models_list}

@app.get("/v1/voices")
async def list_voices():
    return voice_manager.voices

@app.post("/v1/voices/reload")
async def reload_voices():
    voice_manager.load_configs()
    return {"status": "success", "count": len(voice_manager.voices)}

if __name__ == "__main__":
    import uvicorn
    final_parser = argparse.ArgumentParser(description="GPT-SoVITS ONNX API Server")
    final_parser.add_argument("--host", default="0.0.0.0")
    final_parser.add_argument("--port", type=int, default=8001)
    final_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    final_parser.add_argument("--bert_path", help="Path to BERT model")
    final_parser.add_argument("--voices_config", help="Path to voices configuration file")
    
    final_args = final_parser.parse_args()
    if final_args.device: model_manager.device = final_args.device
    if final_args.bert_path: model_manager.bert_path = final_args.bert_path
    if final_args.voices_config: 
        voice_manager.config_path = os.path.normpath(os.path.join(cwd, final_args.voices_config))
        voice_manager.load_configs()

    uvicorn.run(app, host=final_args.host, port=final_args.port)
