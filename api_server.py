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
logger = logging.getLogger("gpt-sovits-api")

# Add current and GPT_SoVITS paths
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "GPT_SoVITS"))

try:
    from run_optimized_inference import GPTSoVITSOptimizedInference
except ImportError:
    logger.error("Failed to import GPTSoVITSOptimizedInference. Ensure run_optimized_inference.py exists.")
    raise

# --- Config Management ---
parser = argparse.ArgumentParser(description="GPT-SoVITS API Server", add_help=False)
parser.add_argument("--cnhubert_path", default=os.environ.get("CNHUBERT_PATH", "pretrained_models/chinese-hubert-base"))
parser.add_argument("--bert_path", default=os.environ.get("BERT_PATH", "pretrained_models/chinese-roberta-wwm-ext-large"))
parser.add_argument("--voices_config", default=os.environ.get("VOICES_CONFIG", "config/voices.json"))
args, _ = parser.parse_known_args()

class ModelManager:
    """Manages loading and caching of GPT-SoVITS model engines."""
    def __init__(self, cnhubert_base_path: str, bert_path: str):
        self.engines: Dict[str, GPTSoVITSOptimizedInference] = {}
        self.cnhubert_base_path = cnhubert_base_path
        self.bert_path = bert_path

    def get_engine(self, gpt_path: str, sovits_path: str) -> GPTSoVITSOptimizedInference:
        key = f"{gpt_path}|{sovits_path}"
        if key not in self.engines:
            logger.info(f"Loading new model engine: GPT={gpt_path}, SoVITS={sovits_path}")
            full_gpt_path = gpt_path if os.path.isabs(gpt_path) else os.path.normpath(os.path.join(cwd, gpt_path))
            full_sovits_path = sovits_path if os.path.isabs(sovits_path) else os.path.normpath(os.path.join(cwd, sovits_path))
            
            engine = GPTSoVITSOptimizedInference(
                gpt_path=full_gpt_path,
                sovits_path=full_sovits_path,
                cnhubert_base_path=os.path.normpath(os.path.join(cwd, self.cnhubert_base_path)),
                bert_path=os.path.normpath(os.path.join(cwd, self.bert_path))
            )
            self.engines[key] = engine
        return self.engines[key]

class VoiceManager:
    """Manages voice configurations from voices.json."""
    def __init__(self, config_path: str):
        self.config_path = os.path.normpath(os.path.join(cwd, config_path))
        self.voices: Dict[str, Any] = {}
        self.load_configs()

    def load_configs(self):
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file {self.config_path} not found. Initializing empty config with template.")
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            self.voices = {}
            example_config = {
                "example": {
                    "gpt_path": "pretrained_models/example_gpt.ckpt",
                    "sovits_path": "pretrained_models/example_sovits.pth",
                    "ref_audio": "pretrained_models/example_ref.wav",
                    "ref_text": "参考音频内容",
                    "ref_lang": "zh",
                    "description": "示例语音配置",
                    "defaults": {"speed": 1.0, "top_k": 15, "top_p": 1.0, "temperature": 1.0, "pause_length": 0.3}
                }
            }
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(example_config, f, indent=4, ensure_ascii=False)
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

# Initialize global managers using parsed args or env
model_manager = ModelManager(args.cnhubert_path, args.bert_path)
voice_manager = VoiceManager(args.voices_config)

app = FastAPI(title="GPT-SoVITS Optimized OpenAI API")

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
    history_window: int = 4
    pause_length: Optional[float] = None
    noise_scale: Optional[float] = None
    sid: Optional[int] = None
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
    top_p = request.top_p if request.top_p is not None else defaults.get("top_p", 1.0)
    temperature = request.temperature if request.temperature is not None else defaults.get("temperature", 1.0)
    pause_length = request.pause_length if request.pause_length is not None else defaults.get("pause_length", 0.3)
    noise_scale = request.noise_scale if request.noise_scale is not None else defaults.get("noise_scale", 0.35)
    
    ref_audio = request.ref_audio or voice_config.get("ref_audio")
    ref_text = request.ref_text or voice_config.get("ref_text")
    ref_lang = request.ref_lang or voice_config.get("ref_lang", "zh")

    if not ref_audio:
        raise HTTPException(status_code=400, detail="Reference audio path is missing.")

    try:
        engine = model_manager.get_engine(voice_config["gpt_path"], voice_config["sovits_path"])
    except Exception as e:
        logger.error(f"Error loading engine: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    def generate_audio():
        try:
            abs_ref_audio = ref_audio if os.path.isabs(ref_audio) else os.path.join(cwd, ref_audio)
            gen = engine.infer_optimized(
                ref_wav_path=abs_ref_audio, prompt_text=ref_text, prompt_lang=ref_lang,
                text=request.input, text_lang=request.text_lang, top_k=top_k, top_p=top_p,
                temperature=temperature, speed=speed, chunk_length=request.chunk_length,
                noise_scale=noise_scale, history_window=request.history_window, pause_length=pause_length
            )
            sr = engine.hps.data.sampling_rate
            if request.response_format == "wav":
                header = io.BytesIO()
                with wave.open(header, "wb") as wav_file:
                    wav_file.setnchannels(1); wav_file.setsampwidth(2); wav_file.setframerate(sr); wav_file.writeframes(b"")
                yield header.getvalue()
            for chunk in gen:
                if chunk is not None and len(chunk) > 0:
                    yield audio_array_to_wav_chunk(chunk, sr)
        except Exception as e:
            logger.error(f"Inference error: {e}")

    return StreamingResponse(generate_audio(), media_type="audio/wav" if request.response_format == "wav" else "audio/mpeg")

@app.get("/v1/models")
async def list_models():
    models_list = [{"id": name, "object": "model", "created": 1700000000, "owned_by": "gpt-sovits", "description": config.get("description", "")} 
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
    final_parser = argparse.ArgumentParser(description="GPT-SoVITS API Server (OpenAI Compatible)")
    final_parser.add_argument("--host", default="0.0.0.0")
    final_parser.add_argument("--port", type=int, default=8000)
    final_parser.add_argument("--reload", action="store_true")
    final_parser.add_argument("--cnhubert_path", help="Path to CNHubert model")
    final_parser.add_argument("--bert_path", help="Path to BERT model")
    final_parser.add_argument("--voices_config", help="Path to voices configuration file")
    
    final_args = final_parser.parse_args()
    uvicorn.run("api_server:app", host=final_args.host, port=final_args.port, reload=final_args.reload)