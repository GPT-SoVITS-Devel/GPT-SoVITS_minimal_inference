import re

import onnxruntime
import torch
import numpy as np
import argparse
import os

os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
import librosa
import soundfile as sf
import sys
import time
import json
from transformers import AutoTokenizer

# os.environ["http_proxy"]='http://127.0.0.1:10809'
# os.environ["https_proxy"]='http://127.0.0.1:10809'
# import nltk
# nltk.download('averaged_perceptron_tagger_eng')

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

def sample_topk(topk_values, topk_indices, temperature=1.0):
    # topk_values: [B, K], topk_indices: [B, K]
    if temperature != 1.0:
        topk_values = topk_values / temperature
    
    # Softmax over top-k
    topk_values = topk_values - np.max(topk_values, axis=-1, keepdims=True)
    probs = np.exp(topk_values)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    
    samples = []
    for i in range(probs.shape[0]):
        choice = np.random.choice(len(probs[i]), p=probs[i])
        samples.append(topk_indices[i, choice])
    return np.array(samples, dtype=np.int64)[:, None]

class GPTSoVITS_ONNX_Inference:
    def __init__(self, onnx_dir, bert_path, device="cpu"):
        self.onnx_dir = onnx_dir
        self.device = device
        
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        sol = onnxruntime.SessionOptions()
        sol.log_severity_level=1

        if device == "cuda":
            self.providers = [("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"]
        else:
            self.providers = ["CPUExecutionProvider"]
        
        print(f"Loading ONNX models from {onnx_dir} on {device}...")
        self.sess_ssl = onnxruntime.InferenceSession(f"{onnx_dir}/ssl.onnx", sess_options=so, providers=self.providers)
        self.sess_bert = onnxruntime.InferenceSession(f"{onnx_dir}/bert.onnx", sess_options=so,
                                                      providers=self.providers)
        self.sess_vq = onnxruntime.InferenceSession(f"{onnx_dir}/vq_encoder.onnx", sess_options=so,
                                                    providers=self.providers)
        self.sess_gpt_enc = onnxruntime.InferenceSession(f"{onnx_dir}/gpt_encoder.onnx", sess_options=so,
                                                         providers=self.providers)

        self.sess_gpt_step = onnxruntime.InferenceSession(f"{onnx_dir}/gpt_step.onnx", sess_options=so,
                                                          providers=self.providers)

        self.sess_sovits = onnxruntime.InferenceSession(f"{onnx_dir}/sovits.onnx", sess_options=so,
                                                        providers=self.providers)

        self.step_inputs_info = {node.name: node.type for node in self.sess_gpt_step.get_inputs()}
        self.step_outputs_info = {node.name: node.type for node in self.sess_gpt_step.get_outputs()}
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        
        # Pre-allocate IOBinding for GPT Step to reuse
        self.step_io_binding = self.sess_gpt_step.io_binding()

        # Load Config for Native ONNX Inference
        config_path = f"{onnx_dir}/config.json"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}. Please export ONNX with the latest export_onnx.py.")
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.hps = json.load(f)
        
        self.version = self.hps.get("model", {}).get("version", "v2")
        print(f"Detected model version: {self.version}")

        self.hps["model"]["semantic_frame_rate"] = "25hz"

        self.sv_model = SV(device, False)

        # Detect Model Precision from GPT Encoder inputs
        self.precision = np.float32
        try:
            for node in self.sess_gpt_enc.get_inputs():
                if node.name == "bert_feature" and node.type == 'tensor(float16)':
                    self.precision = np.float16
                    print("Detected FP16 model inputs. Enabling FP16 inference mode.")
                    break
        except:
            pass

        self.warmup()

    def warmup(self):
        """
        全链路预热模型
        包含 Text Cleaner (Jieba/CMU Dict加载) 和 ONNX Session 初始化
        """
        print("Warming up all components (Text Cleaner + ONNX)...")

        # Check input precision for GPT Step (as a representative)
        step_inputs = self.sess_gpt_step.get_inputs()
        for inp in step_inputs:
            print(f"  - Model Input '{inp.name}': {inp.type}")

        # Text Cleaner Warmup (加载字典)
        # 针对中英双语分别调用一次，触发 Lazy Loading
        try:
            # 简单词汇即可，覆盖 zh 和 en 逻辑分支
            _ = clean_text("预热", "zh", self.version)
            _ = clean_text("Warmup", "en", self.version)
        except Exception as e:
            print(f"Text Cleaner Warmup Warning: {e}")

        # ONNX Models Warmup
        try:
            # SSL & VQ
            dummy_audio = np.zeros((1, 48000), dtype=self.precision)
            self.run_sess(self.sess_ssl, {"audio": dummy_audio})
            dummy_ssl = np.zeros((1, 768, 150), dtype=self.precision)
            self.run_sess(self.sess_vq, {"ssl_content": dummy_ssl})

            # BERT
            dummy_ids = np.zeros((1, 256), dtype=np.int64)
            self.sess_bert.run(["hidden_states"], {
                "input_ids": dummy_ids,
                "attention_mask": dummy_ids,
                "token_type_ids": dummy_ids
            })

            # GPT Encoder
            dummy_phones = np.zeros((1, 256), dtype=np.int64)
            dummy_bert = np.zeros((1, 1024, 256), dtype=self.precision)
            dummy_prompt = np.zeros((1, 50), dtype=np.int64)
            dummy_len = np.array([256], dtype=np.int64)
            self.run_sess(self.sess_gpt_enc, {
                "phoneme_ids": dummy_phones,
                "phoneme_ids_len": dummy_len,
                "prompts": dummy_prompt,
                "bert_feature": dummy_bert
            })

            # SV Model Warmup (PyTorch)
            print("  - Warming up SV model...")
            dummy_wav_sv = torch.zeros((1, 16000), device=self.device)
            self.sv_model.compute_embedding3(dummy_wav_sv)

            # SoVITS Warmup
            print("  - Warming up SoVITS...")
            dummy_sem = np.zeros((1, 1, 128), dtype=np.int64)
            dummy_seq = np.zeros((1, 64), dtype=np.int64)
            dummy_spec = np.zeros((1, 1025, 100), dtype=self.precision)
            
            sv_size = 20480 if "Pro" in self.version else 512
            dummy_emb = np.zeros((1, sv_size), dtype=self.precision)

            self.run_sess(self.sess_sovits, {
                "pred_semantic": dummy_sem,
                "text_seq": dummy_seq,
                "refer_spec": dummy_spec,
                "sv_emb": dummy_emb,
                "noise_scale": np.array([0.5], dtype=np.float32),
                "speed": np.array([1.0], dtype=np.float32),
            })

        except Exception as e:
            print(f"ONNX Warmup partial warning: {e}")

        print("Warmup complete.")

    def run_sess(self, sess, inputs):
        input_meta = sess.get_inputs()
        actual_inputs = {}
        for i in input_meta:
            if i.name in inputs:
                val = inputs[i.name]
                if isinstance(val, np.ndarray):
                    if i.type == 'tensor(float16)' and val.dtype != np.float16:
                        val = val.astype(np.float16)
                    elif i.type == 'tensor(float)' and val.dtype != np.float32:
                        val = val.astype(np.float32)
                    elif i.type == 'tensor(int64)' and val.dtype != np.int64:
                        val = val.astype(np.int64)
                actual_inputs[i.name] = val
        outputs = sess.run(None, actual_inputs)
        return outputs

    def get_bert_feature(self, text, word2ph, language):
        if language != "zh": return None
        inputs = self.tokenizer(text, return_tensors="np")
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)
        token_type_ids = inputs["token_type_ids"].astype(np.int64)
        hidden_states = self.sess_bert.run(["hidden_states"], {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        })[0]
        res = hidden_states[0][1:-1]
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = np.tile(res[i], (word2ph[i], 1))
            phone_level_feature.append(repeat_feature)
        phone_level_feature = np.concatenate(phone_level_feature, axis=0)
        return phone_level_feature.T

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language = language.replace("all_", "")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph, language)
        else:
            bert = np.zeros(
                (1024, len(phones)),
                dtype=self.precision
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

        print(f"Text segments: {textlist}")
        print(f"Language segments: {langlist}")

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

        bert = np.concatenate(bert_list, axis=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)

        return phones, bert.astype(self.precision), norm_text

    def _to_gpu_ort(self, data):
        if self.device != "cuda": return onnxruntime.OrtValue.ortvalue_from_numpy(data)
        return onnxruntime.OrtValue.ortvalue_from_numpy(data, "cuda", 0)

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

        t_total_start = time.perf_counter()

        # Audio
        t_start = time.perf_counter()
        wav16k, _ = librosa.load(ref_wav_path, sr=16000)
        wav16k = wav16k.astype(self.precision)
        zero_wav = np.zeros(int(16000 * 0.3), dtype=self.precision)
        wav16k_padded = np.concatenate([wav16k, zero_wav])[None, :]

        ssl_content = self.run_sess(self.sess_ssl, {"audio": wav16k_padded})[0]
        codes = self.run_sess(self.sess_vq, {"ssl_content": ssl_content})[0]
        prompt_semantic = codes[0, 0][None, :]
        t_ref_audio = time.perf_counter() - t_start

        # Text segments
        segments = split_text(text)
        if not segments:
            return

        final_audios = []
        sr = self.hps["data"]["sampling_rate"]

        # 4. SoVITS Setup
        wav_ref, _ = librosa.load(ref_wav_path, sr=sr)
        spec = spectrogram_torch(torch.from_numpy(wav_ref)[None, :], self.hps["data"]["filter_length"], self.hps["data"]["sampling_rate"],
                                 self.hps["data"]["hop_length"], self.hps["data"]["win_length"], center=False).numpy()

        wav16k_sv, _ = librosa.load(ref_wav_path, sr=16000)
        sv_emb = self.sv_model.compute_embedding3(torch.from_numpy(wav16k_sv)[None, :]).detach().cpu().numpy()

        sv_size = 20480 if "Pro" in self.version else 512
        if sv_emb.shape[-1] != sv_size:
            tmp = np.zeros((1, sv_size), dtype=np.float32)
            tmp[:, :min(sv_emb.shape[-1], sv_size)] = sv_emb[:, :min(sv_emb.shape[-1], sv_size)]
            sv_emb = tmp

        # 2. Process Reference Text (Once)
        t_start = time.perf_counter()
        phones1, bert1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_lang, self.version)
        t_text_proc += time.perf_counter() - t_start

        for seg_idx, seg in enumerate(segments):
            print(f"Processing segment {seg_idx+1}/{len(segments)}: {seg}")
            
            # Text Segment
            t_seg_start = time.perf_counter()
            phones2, bert2, norm_text2 = self.get_phones_and_bert(seg, text_lang, self.version)
            bert = np.concatenate([bert1, bert2], axis=1)[None, :, :].astype(self.precision)
            all_phoneme_ids = np.array(phones1 + phones2, dtype=np.int64)[None, :]
            all_phoneme_len = np.array([all_phoneme_ids.shape[1]], dtype=np.int64)
            t_text_proc += time.perf_counter() - t_seg_start

            # GPT Encoder
            t_enc_start = time.perf_counter()
            topk_values, topk_indices, k_cache, v_cache, x_len, y_len = self.run_sess(self.sess_gpt_enc, {
                "phoneme_ids": all_phoneme_ids,
                "phoneme_ids_len": all_phoneme_len,
                "prompts": prompt_semantic.astype(np.int64),
                "bert_feature": bert
            })
            t_gpt_enc += time.perf_counter() - t_enc_start

            current_samples = sample_topk(topk_values, topk_indices, temperature=temperature)
            decoded_semantic_list = [prompt_semantic, current_samples]

            # GPT Step
            t_dec_start = time.perf_counter()
            max_steps = 1500
            cache_dtype = np.float32
            if self.step_inputs_info.get("k_cache", "").find("float16") != -1:
                cache_dtype = np.float16

            k_cache_ort_0 = self._to_gpu_ort(k_cache.astype(cache_dtype))
            v_cache_ort_0 = self._to_gpu_ort(v_cache.astype(cache_dtype))
            k_cache_ort_1 = self._to_gpu_ort(k_cache.astype(cache_dtype))
            v_cache_ort_1 = self._to_gpu_ort(v_cache.astype(cache_dtype))
            caches = [(k_cache_ort_0, v_cache_ort_0), (k_cache_ort_1, v_cache_ort_1)]
            idx_ort_list = [onnxruntime.OrtValue.ortvalue_from_numpy(np.array([i], dtype=np.int64)) for i in range(max_steps)]
            x_len_ort = onnxruntime.OrtValue.ortvalue_from_numpy(x_len.astype(np.int64))
            y_len_ort = onnxruntime.OrtValue.ortvalue_from_numpy(y_len.astype(np.int64))
            io_binding = self.step_io_binding

            seg_steps = 0
            for i in range(max_steps):
                src_cache = caches[i % 2]
                dst_cache = caches[(i + 1) % 2]
                samples_ort = self._to_gpu_ort(current_samples.astype(np.int64))
                idx_ort = idx_ort_list[i]
                io_binding.bind_ortvalue_input("samples", samples_ort)
                io_binding.bind_ortvalue_input("k_cache", src_cache[0])
                io_binding.bind_ortvalue_input("v_cache", src_cache[1])
                io_binding.bind_ortvalue_input("idx", idx_ort)
                io_binding.bind_ortvalue_input("x_len", x_len_ort)
                io_binding.bind_ortvalue_input("y_len", y_len_ort)
                io_binding.bind_output("topk_values", "cpu")
                io_binding.bind_output("topk_indices", "cpu")
                io_binding.bind_ortvalue_output("k_cache_new", dst_cache[0])
                io_binding.bind_ortvalue_output("v_cache_new", dst_cache[1])
                self.sess_gpt_step.run_with_iobinding(io_binding)
                outputs = io_binding.get_outputs()
                topk_values = outputs[0].numpy()
                topk_indices = outputs[1].numpy()
                current_samples = sample_topk(topk_values, topk_indices, temperature=temperature)
                decoded_semantic_list.append(current_samples)
                seg_steps += 1
                if current_samples[0, 0] == 1024:
                    break
            t_gpt_dec += time.perf_counter() - t_dec_start
            total_steps += seg_steps

            pred_semantic = np.concatenate(decoded_semantic_list, axis=1)
            generated_sem = pred_semantic[:, prompt_semantic.shape[1]:]
            if generated_sem[0, -1] == 1024: generated_sem = generated_sem[:, :-1]
            generated_sem = generated_sem[:, None, :]

            # 5. SoVITS
            t_sov_start = time.perf_counter()
            audio = self.run_sess(self.sess_sovits, {
                "pred_semantic": generated_sem.astype(np.int64),
                "text_seq": np.array(phones2, dtype=np.int64)[None, :],
                "refer_spec": spec.astype(self.precision),
                "sv_emb": sv_emb.astype(self.precision),
                "noise_scale": np.array([noise_scale], dtype=np.float32),
                "speed": np.array([speed], dtype=np.float32),
            })[0]
            t_sovits += time.perf_counter() - t_sov_start
            
            audio_np = audio.squeeze()
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

        print("\n--- Inference Performance Summary (ONNX) ---")
        print(f"Reference Processing:  {t_ref_audio:.3f}s")
        print(f"Target Text Cleaning:  {t_text_proc:.3f}s")
        print(f"GPT Semantic Gen:      {t_gpt_enc + t_gpt_dec:.3f}s ({gpt_tps:.2f} tokens/s)")
        print(f"SoVITS Audio Decode:   {t_sovits:.3f}s")
        print(f"First Segment Latency: {t_first_segment:.3f}s")
        print(f"Total Audio Duration:  {total_audio_duration:.3f}s")
        print(f"Total Inference Time:  {t_total:.3f}s")
        print(f"Real Time Factor (RTF): {rtf:.4f}")
        print("-------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", default="onnx_export/firefly_v2_proplus_fp16")
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--ref_text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="output_onnx.wav")
    parser.add_argument("--ref_lang", default="zh")
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--bert_path", default="pretrained_models/chinese-roberta-wwm-ext-large")
    parser.add_argument("--pause_length", type=float, default=0.3)
    args = parser.parse_args()

    GPTSoVITS_ONNX_Inference(
        args.onnx_dir, args.bert_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).infer(
        args.ref_audio, args.ref_text, args.ref_lang, args.text, args.lang, 
        output_path=args.output, pause_length=args.pause_length
    )
