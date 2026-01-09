import re
import tensorrt as trt
import torch
import numpy as np
import argparse
import os
import librosa
import soundfile as sf
import sys
import time
from transformers import AutoTokenizer

# Setup paths
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "GPT_SoVITS"))

from GPT_SoVITS.text.LangSegmenter import LangSegmenter
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from GPT_SoVITS.module.mel_processing import spectrogram_torch
from GPT_SoVITS.sv import SV

def split_text(text):
    text = text.strip("\n")
    if not text:
        return []
    sentence_delimiters = r'([。！？.!?…\n])'
    parts = re.split(sentence_delimiters, text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentences.append(parts[i] + parts[i + 1])
    if len(parts) % 2 == 1:
        sentences.append(parts[-1])
    sentences = [s.strip() for s in sentences if s.strip()]
    merged = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < 20:
            current += s
        else:
            if current:
                merged.append(current)
            current = s
    if current:
        merged.append(current)
    return merged

# Global TRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def sample_topk(topk_values, topk_indices, temperature=1.0):
    # Support both CPU and GPU tensors
    device = topk_values.device
    if temperature != 1.0:
        topk_values = topk_values / temperature
    
    # Softmax over top-k
    probs = torch.softmax(topk_values, dim=-1)
    
    # Sample using multinomial
    # For small sizes (K=50), CPU is often faster on Windows due to launch overhead
    if device.type == "cuda":
        # Multinomial is a sync point anyway if we inspect it
        indices_of_indices = torch.multinomial(probs, num_samples=1)
    else:
        indices_of_indices = torch.multinomial(probs, num_samples=1)
        
    samples = torch.gather(topk_indices, -1, indices_of_indices)
    
    return samples

def trt_dtype_to_torch(trt_dtype):
    if trt_dtype == trt.float32: return torch.float32
    if trt_dtype == trt.float16: return torch.float16
    if trt_dtype == trt.int32: return torch.int32
    if trt_dtype == trt.int64: return torch.int64
    if trt_dtype == trt.int8: return torch.int8
    if trt_dtype == trt.bool: return torch.bool
    return torch.float32

class TRTModule:
    def __init__(self, engine_path, device="cuda", stream=None):
        self.device = torch.device(device)
        self.stream = stream if stream is not None else torch.cuda.Stream(device=self.device)
        # Use shared global logger
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        self.input_names = []
        self.output_names = []
        self.output_tensors = {}
        self.is_dynamic = {}
        self.tensor_location = {}
        self.tensor_dtype = {}
        
        try:
            self.num_io = self.engine.num_io_tensors
            self.use_new_api = True
        except AttributeError:
            self.num_io = self.engine.num_bindings
            self.use_new_api = False

        for i in range(self.num_io):
            if self.use_new_api:
                name = self.engine.get_tensor_name(i)
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                shape = self.engine.get_tensor_shape(name)
                self.is_dynamic[name] = any(d < 0 for d in shape)
                self.tensor_location[name] = self.engine.get_tensor_location(name)
                self.tensor_dtype[name] = trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            else:
                name = self.engine.get_binding_name(i)
                is_input = self.engine.binding_is_input(i)
                self.is_dynamic[name] = self.engine.binding_is_variable(i) if hasattr(self.engine, "binding_is_variable") else True
                self.tensor_location[name] = trt.TensorLocation.DEVICE
                self.tensor_dtype[name] = trt_dtype_to_torch(self.engine.get_binding_dtype(i))
            
            if is_input:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        
        print(f"  - Loaded {os.path.basename(engine_path)}: Inputs={self.input_names}, Outputs={self.output_names}")

    def __call__(self, inputs, outputs=None, sync=True):
        bindings = [None] * (self.engine.num_bindings if not self.use_new_api else self.num_io)
        held_tensors = [] # Keep references
        
        # 1. Set Input Tensors & Shapes
        for name in self.input_names:
            if name not in inputs:
                continue
            tensor = inputs[name]
            
            # Auto-cast dtype if necessary
            target_dtype = self.tensor_dtype[name]
            if tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
            
            # Ensure correct location
            target_loc = self.tensor_location.get(name, trt.TensorLocation.DEVICE)
            if target_loc == trt.TensorLocation.HOST and tensor.is_cuda:
                tensor = tensor.detach().cpu()
            elif target_loc == trt.TensorLocation.DEVICE and not tensor.is_cuda:
                tensor = tensor.to(self.device)
            
            held_tensors.append(tensor)

            if self.use_new_api:
                if self.is_dynamic[name]:
                    self.context.set_input_shape(name, tensor.shape)
                self.context.set_tensor_address(name, tensor.data_ptr())
            else:
                idx_io = self.engine.get_binding_index(name)
                if self.is_dynamic[name]:
                    self.context.set_binding_shape(idx_io, tensor.shape)
                bindings[idx_io] = tensor.data_ptr()
        
        # 2. Prepare Output Tensors
        if outputs is None:
            outputs = {}
        
        for name in self.output_names:
            target_loc = self.tensor_location.get(name, trt.TensorLocation.DEVICE)
            
            if self.use_new_api:
                shape = self.context.get_tensor_shape(name)
                if name not in outputs:
                    if name not in self.output_tensors or tuple(self.output_tensors[name].shape) != tuple(shape):
                        # Allocation must match required location
                        out_device = "cpu" if target_loc == trt.TensorLocation.HOST else self.device
                        self.output_tensors[name] = torch.empty(tuple(shape), dtype=self.tensor_dtype[name], device=out_device)
                    outputs[name] = self.output_tensors[name]
                
                output_tensor = outputs[name]
                self.context.set_tensor_address(name, output_tensor.data_ptr())
            else:
                idx = self.engine.get_binding_index(name)
                shape = self.context.get_binding_shape(idx)
                if name not in outputs:
                    if name not in self.output_tensors or tuple(self.output_tensors[name].shape) != tuple(shape):
                        self.output_tensors[name] = torch.empty(tuple(shape), dtype=self.tensor_dtype[name], device=self.device)
                    outputs[name] = self.output_tensors[name]
                
                output_tensor = outputs[name]
                bindings[idx] = output_tensor.data_ptr()
            
        # 3. Execute
        try:
            if hasattr(self.context, 'execute_async_v3'):
                self.context.execute_async_v3(self.stream.cuda_stream)
            elif hasattr(self.context, 'execute_v3'):
                self.context.execute_v3(self.stream.cuda_stream)
            elif hasattr(self.context, 'execute_async_v2'):
                self.context.execute_async_v2(bindings, self.stream.cuda_stream)
            else:
                self.context.execute_v2(bindings)
            
            if sync:
                self.stream.synchronize()
        except Exception as e:
            print(f"Error during TRT execution: {e}")
            raise e
            
        return outputs

class GPTSoVITS_TRT_Inference:
    def __init__(self, trt_dir, bert_path, sovits_path, device="cuda"):
        self.trt_dir = trt_dir
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        
        print(f"Loading TensorRT engines from {trt_dir} on {device}...")
        self.model_ssl = TRTModule(f"{trt_dir}/ssl.engine", device, self.stream)
        self.model_bert = TRTModule(f"{trt_dir}/bert.engine", device, self.stream)
        self.model_vq = TRTModule(f"{trt_dir}/vq_encoder.engine", device, self.stream)
        self.model_gpt_enc = TRTModule(f"{trt_dir}/gpt_encoder.engine", device, self.stream)
        self.model_gpt_step = TRTModule(f"{trt_dir}/gpt_step.engine", device, self.stream)
        self.model_sovits = TRTModule(f"{trt_dir}/sovits.engine", device, self.stream)

        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        
        from GPT_SoVITS.process_ckpt import load_sovits_new, get_sovits_version_from_path_fast
        dict_s2 = load_sovits_new(sovits_path)
        self.hps = dict_s2["config"]
        _, self.version, _ = get_sovits_version_from_path_fast(sovits_path)
        print(f"Detected model version: {self.version}")

        self.hps["model"]["version"] = self.version
        self.hps["model"]["semantic_frame_rate"] = "25hz"

        self.sv_model = SV(device, False)

        # Detect precision from gpt_encoder engine as a proxy
        gpt_enc_dtype = self.model_gpt_enc.tensor_dtype["bert_feature"]
        sovits_dtype = self.model_sovits.tensor_dtype["refer_spec"]
        
        print(f"Engine precision detection:")
        print(f"  - GPT Encoder (bert_feature): {gpt_enc_dtype}")
        print(f"  - SoVITS (refer_spec): {sovits_dtype}")

        self.precision = gpt_enc_dtype
        print(f"Using {self.precision} for inference.")

        self.warmup()

    def warmup(self):
        print("Warming up models...")
        # Text Cleaner Warmup
        try:
            _ = clean_text("预热", "zh", self.version)
        except:
            pass
        
        # Real Engine Warmup
        try:
            print("  - Warming up SSL & VQ...")
            dummy_audio = torch.zeros((1, 48000), device=self.device, dtype=self.precision)
            ssl_out = self.model_ssl({"audio": dummy_audio})["last_hidden_state"]
            self.model_vq({"ssl_content": ssl_out})
            
            print("  - Warming up BERT...")
            dummy_ids = torch.zeros((1, 32), dtype=torch.int64, device=self.device)
            self.model_bert({"input_ids": dummy_ids, "attention_mask": dummy_ids, "token_type_ids": dummy_ids})
            
            print("  - Warming up GPT...")
            dummy_phones = torch.zeros((1, 32), dtype=torch.int64, device=self.device)
            dummy_bert = torch.zeros((1, 1024, 32), dtype=self.precision, device=self.device)
            dummy_prompt = torch.zeros((1, 20), dtype=torch.int64, device=self.device)
            self.model_gpt_enc({"phoneme_ids": dummy_phones, "prompts": dummy_prompt, "bert_feature": dummy_bert})
            
            print("  - Warming up SoVITS...")
            dummy_sem = torch.zeros((1, 1, 32), dtype=torch.int64, device=self.device)
            dummy_seq = torch.zeros((1, 32), dtype=torch.int64, device=self.device)
            dummy_spec = torch.zeros((1, 1025, 32), dtype=self.precision, device=self.device)
            sv_size = 20480 if "Pro" in self.version else 512
            dummy_emb = torch.zeros((1, sv_size), dtype=self.precision, device=self.device)
            self.model_sovits({
                "pred_semantic": dummy_sem, "text_seq": dummy_seq, "refer_spec": dummy_spec, 
                "sv_emb": dummy_emb, "noise_scale": torch.tensor([0.5], device=self.device), 
                "speed": torch.tensor([1.0], device=self.device)
            })
        except Exception as e:
            print(f"Warmup warning (some engines might have strict shapes): {e}")
            
        print("Warmup complete.")

    def get_bert_feature(self, text, word2ph, language):
        if language != "zh": return None
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        token_type_ids = inputs["token_type_ids"].to(self.device)
        
        outputs = self.model_bert({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        })
        hidden_states = outputs["hidden_states"]
        
        res = hidden_states[0][1:-1]
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language = language.replace("all_", "")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph, language)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=self.precision,
                device=self.device
            )
        return bert

    def get_phones_and_bert(self, text, language, version):
        import re
        text = re.sub(r' {2,}', ' ', text)
        textlist = []
        langlist = []
        if language == "all_zh":
            for tmp in LangSegmenter.getTexts(text, "zh"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_yue":
            for tmp in LangSegmenter.getTexts(text, "zh"):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_ja":
            for tmp in LangSegmenter.getTexts(text, "ja"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_ko":
            for tmp in LangSegmenter.getTexts(text, "ko"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "en":
            langlist.append("en")
            textlist.append(text)
        elif language == "auto":
            for tmp in LangSegmenter.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegmenter.getTexts(text):
                if langlist:
                    if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                        textlist[-1] += tmp["text"]
                        continue
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    langlist.append(language.replace("all_", ""))
                textlist.append(tmp["text"])

        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text(textlist[i], lang, version)
            phones = cleaned_text_to_sequence(phones, version)
            bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)

        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)

        return phones, bert.to(self.precision), norm_text

    def infer(self, ref_wav_path, prompt_text, prompt_lang, text, text_lang,
              top_k=5, temperature=1.0, noise_scale=0.5, speed=1.0, output_path="out.wav", pause_length=0.3):

        # Timers
        t_ref_audio = 0.0
        t_text_proc = 0.0
        t_gpt_enc = 0.0
        t_gpt_dec = 0.0
        t_sovits = 0.0
        t_first_segment = 0.0
        total_steps = 0

        if self.device.type == "cuda": torch.cuda.synchronize()
        t_total_start = time.perf_counter()

        with torch.cuda.stream(self.stream):
            # Audio
            t_start = time.perf_counter()
            wav16k, _ = librosa.load(ref_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k).to(self.device).to(self.precision)
            zero_wav = torch.zeros(int(16000 * 0.3), device=self.device, dtype=self.precision)
            wav16k_padded = torch.cat([wav16k, zero_wav])[None, :]

            ssl_content = self.model_ssl({"audio": wav16k_padded})["last_hidden_state"]
            codes = self.model_vq({"ssl_content": ssl_content})["codes"]
            prompt_semantic = codes[0, 0][None, :]
            if self.device.type == "cuda": torch.cuda.synchronize()
            t_ref_audio = time.perf_counter() - t_start

            # Text segments
            segments = split_text(text)
            if not segments:
                return

            final_audios = []
            sr = self.hps["data"]["sampling_rate"]

            # SoVITS Setup
            wav_ref, _ = librosa.load(ref_wav_path, sr=sr)
            spec = spectrogram_torch(torch.from_numpy(wav_ref)[None, :], self.hps["data"]["filter_length"], self.hps["data"]["sampling_rate"],
                                     self.hps["data"]["hop_length"], self.hps["data"]["win_length"], center=False).to(self.device).to(self.precision)

            wav16k_sv, _ = librosa.load(ref_wav_path, sr=16000)
            sv_emb = self.sv_model.compute_embedding3(torch.from_numpy(wav16k_sv).to(self.device)[None, :]).detach()

            sv_size = 20480 if "Pro" in self.version else 512
            if sv_emb.shape[-1] != sv_size:
                tmp = torch.zeros((1, sv_size), device=self.device, dtype=torch.float32)
                tmp[:, :min(sv_emb.shape[-1], sv_size)] = sv_emb[:, :min(sv_emb.shape[-1], sv_size)]
                sv_emb = tmp
            sv_emb = sv_emb.to(self.precision)

            # Process Reference Text
            t_start = time.perf_counter()
            phones1, bert1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_lang, self.version)
            if self.device.type == "cuda": torch.cuda.synchronize()
            t_text_proc += time.perf_counter() - t_start

            for seg_idx, seg in enumerate(segments):
                print(f"Processing segment {seg_idx+1}/{len(segments)}: {seg}")
                
                # Text Segment
                t_seg_start = time.perf_counter()
                phones2, bert2, norm_text2 = self.get_phones_and_bert(seg, text_lang, self.version)
                bert = torch.cat([bert1, bert2], dim=1)[None, :, :].to(self.precision)
                all_phoneme_ids = torch.tensor(phones1 + phones2, dtype=torch.int64, device=self.device)[None, :]
                all_phoneme_len = torch.tensor([all_phoneme_ids.shape[1]], dtype=torch.int64, device=self.device)
                if self.device.type == "cuda": torch.cuda.synchronize()
                t_text_proc += time.perf_counter() - t_seg_start

                # GPT Encoder
                t_enc_start = time.perf_counter()
                gpt_enc_out = self.model_gpt_enc({
                    "phoneme_ids": all_phoneme_ids,
                    "phoneme_ids_len": all_phoneme_len,
                    "prompts": prompt_semantic.to(torch.int64),
                    "bert_feature": bert
                })
                if self.device.type == "cuda": torch.cuda.synchronize()
                t_gpt_enc += time.perf_counter() - t_enc_start
                
                topk_values = gpt_enc_out["topk_values"].detach().cpu()
                topk_indices = gpt_enc_out["topk_indices"].detach().cpu()
                k_cache = gpt_enc_out["k_cache"]
                v_cache = gpt_enc_out["v_cache"]
                x_len = gpt_enc_out["x_len"]
                y_len = gpt_enc_out["y_len"]

                current_samples = sample_topk(topk_values, topk_indices, temperature=temperature).to(self.device)
                prompt_semantic_gpu = prompt_semantic.to(self.device)
                decoded_semantic_list = [prompt_semantic_gpu, current_samples]

                # GPT Step
                t_dec_start = time.perf_counter()
                kv_max_len = k_cache.shape[2]
                base_len = int(x_len.item() + y_len.item())
                max_gen_len = kv_max_len - base_len - 1
                max_steps = min(1000, max_gen_len) if max_gen_len > 0 else 1
                
                k_cache_0, v_cache_0 = k_cache.clone(), v_cache.clone()
                k_cache_1, v_cache_1 = torch.zeros_like(k_cache_0), torch.zeros_like(v_cache_0)
                
                def prepare_tensor(name, tensor, module):
                    target_loc = module.tensor_location.get(name, trt.TensorLocation.DEVICE)
                    return tensor.detach().cpu().to(torch.int64) if target_loc == trt.TensorLocation.HOST else tensor.detach().to(self.device).to(torch.int64)

                x_len_opt = prepare_tensor("x_len", x_len, self.model_gpt_step)
                y_len_opt = prepare_tensor("y_len", y_len, self.model_gpt_step)
                caches = [(k_cache_0, v_cache_0), (k_cache_1, v_cache_1)]
                
                seg_steps = 0
                idx_loc = self.model_gpt_step.tensor_location.get("idx", trt.TensorLocation.DEVICE)
                idx_device = "cpu" if idx_loc == trt.TensorLocation.HOST else self.device
                idx_tensors = [torch.tensor([i], dtype=torch.int64, device=idx_device) for i in range(max_steps)]
                step_outputs = {"k_cache_new": None, "v_cache_new": None}

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
                    if temperature != 1.0: topk_v = topk_v / temperature
                    probs = torch.softmax(topk_v, dim=-1)
                    indices_of_indices = torch.multinomial(probs, num_samples=1)
                    current_samples = torch.gather(topk_i, -1, indices_of_indices).to(self.device)
                    
                    decoded_semantic_list.append(current_samples)
                    seg_steps += 1
                    if current_samples[0, 0] == 1024: break
                
                if self.device.type == "cuda": torch.cuda.synchronize()
                t_gpt_dec += time.perf_counter() - t_dec_start
                total_steps += seg_steps

                pred_semantic = torch.cat(decoded_semantic_list, dim=1)
                generated_sem = pred_semantic[:, prompt_semantic.shape[1]:]
                if generated_sem[0, -1] == 1024: generated_sem = generated_sem[:, :-1]
                generated_sem = generated_sem[:, None, :]

                # SoVITS
                t_sov_start = time.perf_counter()
                sovits_inputs = {
                    "pred_semantic": generated_sem.to(torch.int64),
                    "text_seq": torch.tensor(phones2, dtype=torch.int64, device=self.device)[None, :],
                    "refer_spec": spec, "sv_emb": sv_emb,
                    "noise_scale": torch.tensor([noise_scale], dtype=torch.float32, device=self.device),
                    "speed": torch.tensor([speed], dtype=torch.float32, device=self.device),
                }
                sovits_inputs = {k: v for k, v in sovits_inputs.items() if k in self.model_sovits.input_names}
                audio = self.model_sovits(sovits_inputs)["audio"]
                if self.device.type == "cuda": torch.cuda.synchronize()
                t_sovits += time.perf_counter() - t_sov_start
                
                audio_np = audio.squeeze().detach().cpu().numpy()
                # Remove DC offset per segment to prevent drift
                audio_np = audio_np - np.mean(audio_np)
                final_audios.append(audio_np)
                
                if seg_idx == 0:
                    t_first_segment = time.perf_counter() - t_total_start

                if seg_idx < len(segments) - 1 and pause_length > 0:
                    final_audios.append(np.zeros(int(sr * pause_length)))

        t_total = time.perf_counter() - t_total_start
        
        full_audio = np.concatenate(final_audios).astype(np.float32)
        
        # Global Peak Normalization
        max_amp = np.abs(full_audio).max()
        if max_amp > 1e-5:
            full_audio = full_audio / max_amp * 0.9
            
        sf.write(output_path, full_audio, sr)
        print(f"Saved audio to {output_path}")

        # Performance Summary
        total_audio_duration = len(full_audio) / sr
        rtf = t_total / total_audio_duration if total_audio_duration > 0 else 0
        gpt_tps = total_steps / t_gpt_dec if t_gpt_dec > 0 else 0

        print("\n--- Inference Performance Summary (TensorRT) ---")
        print(f"Reference Processing:  {t_ref_audio:.3f}s")
        print(f"Target Text Cleaning:  {t_text_proc:.3f}s")
        print(f"GPT Semantic Gen:      {t_gpt_enc + t_gpt_dec:.3f}s ({gpt_tps:.2f} tokens/s)")
        print(f"SoVITS Audio Decode:   {t_sovits:.3f}s")
        print(f"First Segment Latency: {t_first_segment:.3f}s")
        print(f"Total Audio Duration:  {total_audio_duration:.3f}s")
        print(f"Total Inference Time:  {t_total:.3f}s")
        print(f"Real Time Factor (RTF): {rtf:.4f}")
        print("----------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trt_dir", default="onnx_export/firefly_v2_proplus")
    parser.add_argument("--gpt_path", required=True)
    parser.add_argument("--sovits_path", required=True)
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--ref_text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="output_trt.wav")
    parser.add_argument("--ref_lang", default="zh")
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--bert_path", default="pretrained_models/chinese-roberta-wwm-ext-large")
    parser.add_argument("--pause_length", type=float, default=0.3)
    args = parser.parse_args()

    GPTSoVITS_TRT_Inference(
        args.trt_dir, args.bert_path, args.sovits_path
    ).infer(
        args.ref_audio, args.ref_text, args.ref_lang, args.text, args.lang, 
        output_path=args.output, pause_length=args.pause_length
    )
