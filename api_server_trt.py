import os
import sys
import torch
import numpy as np
import io
import json
import wave
import logging
import argparse
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gpt-sovits-trt-api")

# Add current and GPT_SoVITS paths
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "GPT_SoVITS"))

try:
    from run_trt_inference import GPTSoVITS_TRT_Inference, split_text, spectrogram_torch
    import librosa
except ImportError:
    logger.error("Failed to import GPTSoVITS_TRT_Inference. Ensure run_trt_inference.py exists and dependencies are installed.")
    raise

class GPTSoVITS_TRT_Streaming_Inference(GPTSoVITS_TRT_Inference):
    """Subclass of TRT Inference to provide a true token-level streaming generator."""
    
    def infer_stream(self, ref_wav_path, prompt_text, prompt_lang, text, text_lang,
                     top_k=5, temperature=1.0, noise_scale=0.5, speed=1.0, chunk_length=24, pause_length=0.3):

        mute_matrix_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPT_SoVITS/pretrained_models/gpts1_mute_emb_sim_matrix.pt")
        mute_matrix = torch.load(mute_matrix_path, map_location=self.device) if os.path.exists(mute_matrix_path) else None

        with torch.cuda.stream(self.stream):
            wav16k, _ = librosa.load(ref_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k).to(self.device).to(self.precision)
            zero_wav = torch.zeros(int(16000 * 0.3), device=self.device, dtype=self.precision)
            wav16k_padded = torch.cat([wav16k, zero_wav])[None, :]

            ssl_content = self.model_ssl({"audio": wav16k_padded})["last_hidden_state"]
            codes = self.model_vq({"ssl_content": ssl_content})["codes"]
            prompt_semantic = codes[0, 0][None, :]

            segments = split_text(text)
            if not segments:
                return

            sr = self.hps["data"]["sampling_rate"]
            samples_per_token_fixed = sr // 25
            h_len, l_len, fade_len = 512, 16, 1280
            prev_fade_out = None

            wav_ref, _ = librosa.load(ref_wav_path, sr=sr)
            spec = spectrogram_torch(torch.from_numpy(wav_ref)[None, :], 
                                     self.hps["data"]["filter_length"], 
                                     self.hps["data"]["sampling_rate"],
                                     self.hps["data"]["hop_length"], 
                                     self.hps["data"]["win_length"], 
                                     center=False).to(self.device).to(self.precision)

            wav16k_sv, _ = librosa.load(ref_wav_path, sr=16000)
            sv_emb = self.sv_model.compute_embedding3(torch.from_numpy(wav16k_sv).to(self.device)[None, :]).detach()
            
            sv_size = 20480 if "Pro" in self.version else 512
            if sv_emb.shape[-1] != sv_size:
                tmp = torch.zeros((1, sv_size), device=self.device, dtype=torch.float32)
                tmp[:, :min(sv_emb.shape[-1], sv_size)] = sv_emb[:, :min(sv_emb.shape[-1], sv_size)]
                sv_emb = tmp
            sv_emb = sv_emb.to(self.precision)

            phones1, bert1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_lang, self.version)

            from run_trt_inference import sample_topk
            
            for seg_idx, seg in enumerate(segments):
                # Text Segment
                phones2, bert2, norm_text2 = self.get_phones_and_bert(seg, text_lang, self.version)
                bert = torch.cat([bert1, bert2], dim=1)[None, :, :].to(self.precision)
                all_phoneme_ids = torch.tensor(phones1 + phones2, dtype=torch.int64, device=self.device)[None, :]
                all_phoneme_len = torch.tensor([all_phoneme_ids.shape[1]], dtype=torch.int64, device=self.device)

                # GPT Encoder
                gpt_enc_out = self.model_gpt_enc({
                    "phoneme_ids": all_phoneme_ids,
                    "phoneme_ids_len": all_phoneme_len,
                    "prompts": prompt_semantic.to(torch.int64),
                    "bert_feature": bert
                })
                
                topk_values = gpt_enc_out["topk_values"].detach().cpu()
                topk_indices = gpt_enc_out["topk_indices"].detach().cpu()
                k_cache = gpt_enc_out["k_cache"]
                v_cache = gpt_enc_out["v_cache"]
                x_len = gpt_enc_out["x_len"]
                y_len = gpt_enc_out["y_len"]

                current_samples = sample_topk(topk_values, topk_indices, temperature=temperature).to(self.device)
                prompt_semantic_gpu = prompt_semantic.to(self.device)
                
                # GPT Step Setup
                kv_max_len = k_cache.shape[2]
                base_len = int(x_len.item() + y_len.item())
                max_gen_len = kv_max_len - base_len - 1
                max_steps = min(1500, max_gen_len) if max_gen_len > 0 else 1
                
                k_cache_0, v_cache_0 = k_cache.clone(), v_cache.clone()
                k_cache_1, v_cache_1 = torch.zeros_like(k_cache_0), torch.zeros_like(v_cache_0)
                
                def prepare_tensor(name, tensor, module):
                    import tensorrt as trt
                    target_loc = module.tensor_location.get(name, trt.TensorLocation.DEVICE)
                    return tensor.detach().cpu().to(torch.int64) if target_loc == trt.TensorLocation.HOST else tensor.detach().to(self.device).to(torch.int64)

                x_len_opt = prepare_tensor("x_len", x_len, self.model_gpt_step)
                y_len_opt = prepare_tensor("y_len", y_len, self.model_gpt_step)
                caches = [(k_cache_0, v_cache_0), (k_cache_1, v_cache_1)]
                
                import tensorrt as trt
                idx_loc = self.model_gpt_step.tensor_location.get("idx", trt.TensorLocation.DEVICE)
                idx_device = "cpu" if idx_loc == trt.TensorLocation.HOST else self.device
                idx_tensors = [torch.tensor([i], dtype=torch.int64, device=idx_device) for i in range(max_steps)]
                step_outputs = {"k_cache_new": None, "v_cache_new": None}

                chunk_queue = []
                tokens_buffer = [current_samples]
                history_tokens = None
                token_counter = 1

                def decode_chunk(chunk_tokens, hist, lookahead):
                    inp_list = []
                    if hist is not None: inp_list.append(hist[:, -h_len:])
                    inp_list.append(chunk_tokens)
                    if lookahead is not None: inp_list.append(lookahead[:, :l_len])
                    
                    full_sem = torch.cat(inp_list, dim=1)[:, None, :]
                    sovits_inputs = {
                        "pred_semantic": full_sem.to(torch.int64),
                        "text_seq": torch.tensor(phones2, dtype=torch.int64, device=self.device)[None, :],
                        "refer_spec": spec, "sv_emb": sv_emb,
                        "noise_scale": torch.tensor([noise_scale], dtype=torch.float32, device=self.device),
                        "speed": torch.tensor([speed], dtype=torch.float32, device=self.device),
                    }
                    sovits_inputs = {k: v for k, v in sovits_inputs.items() if k in self.model_sovits.input_names}
                    audio = self.model_sovits(sovits_inputs)["audio"]
                    
                    # 使用浮点数计算以保持切片精度
                    actual_samples_per_token = samples_per_token_fixed / speed
                    h_samples = int((hist[:, -h_len:].shape[1] if hist is not None else 0) * actual_samples_per_token)
                    c_samples = int(chunk_tokens.shape[1] * actual_samples_per_token)
                    res = audio.flatten()[h_samples : h_samples + c_samples]
                    return res.detach().cpu().numpy()

                for i in range(max_steps):
                    idx_tensor = idx_tensors[i]
                    src_cache, dst_cache = caches[i % 2], caches[(i + 1) % 2]
                    step_outputs["k_cache_new"], step_outputs["v_cache_new"] = dst_cache[0], dst_cache[1]
                    
                    step_out = self.model_gpt_step({
                        "samples": current_samples.to(torch.int64),
                        "k_cache": src_cache[0], "v_cache": src_cache[1],
                        "idx": idx_tensor, "x_len": x_len_opt, "y_len": y_len_opt
                    }, outputs=step_outputs, sync=False)
                    
                    topk_v, topk_i = step_out["topk_values"].detach().cpu(), step_out["topk_indices"].detach().cpu()
                    current_samples = sample_topk(topk_v, topk_indices=topk_i, temperature=temperature).to(self.device)
                    
                    if current_samples[0, 0] == 1024: break
                    
                    tokens_buffer.append(current_samples)
                    token_counter += 1

                    is_split = False
                    if mute_matrix is not None and token_counter >= chunk_length + 2:
                        recent_tokens = torch.cat(tokens_buffer, dim=1).flatten()
                        scores = mute_matrix[recent_tokens] - 0.3
                        scores[scores < 0] = -1
                        scores[:-1] += scores[1:]
                        argmax_idx = torch.argmax(scores).item()
                        if scores[argmax_idx] >= 0 and argmax_idx + 1 >= chunk_length:
                            split_idx = argmax_idx + 1
                            chunk_queue.append(torch.cat(tokens_buffer[:split_idx], dim=1))
                            tokens_buffer = tokens_buffer[split_idx:]
                            token_counter -= split_idx
                            is_split = True
                    elif mute_matrix is None and token_counter >= chunk_length:
                        chunk_queue.append(torch.cat(tokens_buffer, dim=1))
                        tokens_buffer = []
                        token_counter = 0
                        is_split = True

                    while len(chunk_queue) > 1:
                        curr = chunk_queue.pop(0)
                        audio_data = decode_chunk(curr, history_tokens, chunk_queue[0])
                        
                        if prev_fade_out is not None:
                            fade_in = np.linspace(0, 1, fade_len)
                            audio_data[:fade_len] = audio_data[:fade_len] * fade_in + prev_fade_out * (1 - fade_in)
                        
                        prev_fade_out = audio_data[-fade_len:]
                        yield audio_data[:-fade_len]
                        history_tokens = curr if history_tokens is None else torch.cat([history_tokens, curr], dim=1)[:, -h_len:]

                if tokens_buffer:
                    chunk_queue.append(torch.cat(tokens_buffer, dim=1))
                
                while chunk_queue:
                    curr = chunk_queue.pop(0)
                    next_chunk = chunk_queue[0] if chunk_queue else None
                    audio_data = decode_chunk(curr, history_tokens, next_chunk)
                    
                    if prev_fade_out is not None:
                        fade_in = np.linspace(0, 1, fade_len)
                        audio_data[:fade_len] = audio_data[:fade_len] * fade_in + prev_fade_out * (1 - fade_in)
                    
                    if next_chunk is not None:
                        yield audio_data[:-fade_len]
                        prev_fade_out = audio_data[-fade_len:]
                    else:
                        yield audio_data
                        prev_fade_out = None
                    
                    history_tokens = curr if history_tokens is None else torch.cat([history_tokens, curr], dim=1)[:, -h_len:]

                if seg_idx < len(segments) - 1 and pause_length > 0:
                    yield np.zeros(int(sr * pause_length))

# --- Config Management ---
parser = argparse.ArgumentParser(description="GPT-SoVITS TensorRT API Server", add_help=False)
parser.add_argument("--bert_path", default=os.environ.get("BERT_PATH", "pretrained_models/chinese-roberta-wwm-ext-large"))
parser.add_argument("--voices_config", default=os.environ.get("VOICES_CONFIG", "config/voices.json"))
parser.add_argument("--device", default="cuda")
args, _ = parser.parse_known_args()

class ModelManager:
    """Manages loading and caching of GPT-SoVITS TensorRT engines."""
    def __init__(self, bert_path: str, device: str):
        self.engines: Dict[str, GPTSoVITS_TRT_Streaming_Inference] = {}
        self.bert_path = bert_path
        self.device = device

    def get_engine(self, trt_dir: str) -> GPTSoVITS_TRT_Streaming_Inference:
        if trt_dir not in self.engines:
            logger.info(f"Loading new TensorRT model engine from: {trt_dir}")
            full_trt_dir = trt_dir if os.path.isabs(trt_dir) else os.path.normpath(os.path.join(cwd, trt_dir))
            
            engine = GPTSoVITS_TRT_Streaming_Inference(
                trt_dir=full_trt_dir,
                bert_path=os.path.normpath(os.path.join(cwd, self.bert_path)),
                device=self.device
            )
            self.engines[trt_dir] = engine
        return self.engines[trt_dir]

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

app = FastAPI(title="GPT-SoVITS TensorRT API")

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
    top_k = request.top_k if request.top_k is not None else defaults.get("top_k", 5)
    temperature = request.temperature if request.temperature is not None else defaults.get("temperature", 1.0)
    pause_length = request.pause_length if request.pause_length is not None else defaults.get("pause_length", 0.3)
    noise_scale = request.noise_scale if request.noise_scale is not None else defaults.get("noise_scale", 0.5)
    
    ref_audio = request.ref_audio or voice_config.get("ref_audio")
    ref_text = request.ref_text or voice_config.get("ref_text")
    ref_lang = request.ref_lang or voice_config.get("ref_lang", "zh")

    trt_path = voice_config.get("trt_path")
    if not trt_path:
        raise HTTPException(status_code=400, detail="trt_path is missing in voice config.")

    if not ref_audio:
        raise HTTPException(status_code=400, detail="Reference audio path is missing.")

    try:
        engine = model_manager.get_engine(trt_path)
    except Exception as e:
        logger.error(f"Error loading TensorRT engine: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load TensorRT model: {str(e)}")

    def generate_audio():
        try:
            abs_ref_audio = ref_audio if os.path.isabs(ref_audio) else os.path.join(cwd, ref_audio)
            gen = engine.infer_stream(
                ref_wav_path=abs_ref_audio, prompt_text=ref_text, prompt_lang=ref_lang,
                text=request.input, text_lang=request.text_lang, top_k=top_k,
                temperature=temperature, speed=speed, noise_scale=noise_scale, 
                chunk_length=request.chunk_length, pause_length=pause_length
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
            logger.error(f"TensorRT Inference error: {e}")

    return StreamingResponse(generate_audio(), media_type="audio/wav" if request.response_format == "wav" else "audio/mpeg")

@app.get("/v1/models")
async def list_models():
    models_list = [{"id": name, "object": "model", "created": 1700000000, "owned_by": "gpt-sovits-trt", "description": config.get("description", "")} 
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
    final_parser = argparse.ArgumentParser(description="GPT-SoVITS TensorRT API Server")
    final_parser.add_argument("--host", default="0.0.0.0")
    final_parser.add_argument("--port", type=int, default=8002)
    final_parser.add_argument("--device", default="cuda")
    final_parser.add_argument("--bert_path", help="Path to BERT model")
    final_parser.add_argument("--voices_config", help="Path to voices configuration file")
    
    final_args = final_parser.parse_args()
    if final_args.device: model_manager.device = final_args.device
    if final_args.bert_path: model_manager.bert_path = final_args.bert_path
    if final_args.voices_config: 
        voice_manager.config_path = os.path.normpath(os.path.join(cwd, final_args.voices_config))
        voice_manager.load_configs()

    uvicorn.run(app, host=final_args.host, port=final_args.port)
