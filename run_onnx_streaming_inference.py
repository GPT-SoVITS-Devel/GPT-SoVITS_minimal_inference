import onnxruntime
import torch
import numpy as np
import argparse
import os
import sys
import time
import librosa
import soundfile as sf
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

def sample_topk(topk_values, topk_indices, temperature=1.0):
    if temperature != 1.0:
        topk_values = topk_values / temperature
    topk_values = topk_values - np.max(topk_values, axis=-1, keepdims=True)
    probs = np.exp(topk_values)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    samples = []
    for i in range(probs.shape[0]):
        choice = np.random.choice(len(probs[i]), p=probs[i])
        samples.append(topk_indices[i, choice])
    return np.array(samples, dtype=np.int64)[:, None]

class GPTSoVITS_ONNX_Streaming_Inference:
    def __init__(self, onnx_dir, bert_path, sovits_path, device="cpu"):
        self.onnx_dir = onnx_dir
        self.device = device
        self.providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.sess_ssl = onnxruntime.InferenceSession(f"{onnx_dir}/ssl.onnx", sess_options=so, providers=self.providers)
        self.sess_bert = onnxruntime.InferenceSession(f"{onnx_dir}/bert.onnx", sess_options=so, providers=self.providers)
        self.sess_vq = onnxruntime.InferenceSession(f"{onnx_dir}/vq_encoder.onnx", sess_options=so, providers=self.providers)
        self.sess_gpt_enc = onnxruntime.InferenceSession(f"{onnx_dir}/gpt_encoder.onnx", sess_options=so, providers=self.providers)
        self.sess_gpt_step = onnxruntime.InferenceSession(f"{onnx_dir}/gpt_step.onnx", sess_options=so, providers=self.providers)
        self.sess_sovits = onnxruntime.InferenceSession(f"{onnx_dir}/sovits.onnx", sess_options=so, providers=self.providers)
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        from GPT_SoVITS.process_ckpt import load_sovits_new, get_sovits_version_from_path_fast
        dict_s2 = load_sovits_new(sovits_path)
        self.hps = dict_s2["config"]
        _, self.version, _ = get_sovits_version_from_path_fast(sovits_path)
        self.sv_model = SV(device, False)

        self.precision = np.float16 if any(node.type == 'tensor(float16)' for node in self.sess_gpt_enc.get_inputs() if node.name == "bert_feature") else np.float32
        self.cache_dtype = np.float16 if any(node.type == 'tensor(float16)' for node in self.sess_gpt_step.get_inputs() if node.name == "k_cache") else np.float32

    def run_sess(self, sess, inputs):
        input_meta = sess.get_inputs()
        actual_inputs = {i.name: inputs[i.name].astype(np.float16 if i.type == 'tensor(float16)' else (np.float32 if i.type == 'tensor(float)' else np.int64)) 
                         for i in input_meta if i.name in inputs}
        return sess.run(None, actual_inputs)

    def get_phones_and_bert(self, text, language, version):
        import re
        text = re.sub(r' {2,}', ' ', text)
        textlist, langlist = [], []
        for tmp in LangSegmenter.getTexts(text, language.replace("all_", "")):
            if langlist and ((tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en")):
                textlist[-1] += tmp["text"]
            else:
                langlist.append(tmp["lang"] if tmp["lang"] == "en" else language.replace("all_", ""))
                textlist.append(tmp["text"])
        
        phones_list, bert_list = [], []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text(textlist[i], lang, version)
            phones = cleaned_text_to_sequence(phones, version)
            if lang == "zh":
                inputs = self.tokenizer(norm_text, return_tensors="np")
                hidden_states = self.sess_bert.run(["hidden_states"], {k: v.astype(np.int64) for k, v in inputs.items()})[0]
                res = hidden_states[0][1:-1]
                bert = np.concatenate([np.tile(res[j], (word2ph[j], 1)) for j in range(len(word2ph))], axis=0).T
            else:
                bert = np.zeros((1024, len(phones)), dtype=self.precision)
            phones_list.append(phones)
            bert_list.append(bert)
        return sum(phones_list, []), np.concatenate(bert_list, axis=1).astype(self.precision)

    def _to_ort(self, data):
        if self.device != "cuda": return onnxruntime.OrtValue.ortvalue_from_numpy(data)
        return onnxruntime.OrtValue.ortvalue_from_numpy(data, "cuda", 0)

    def infer_stream(self, ref_wav_path, prompt_text, prompt_lang, text, text_lang,
                     top_k=15, temperature=1.0, noise_scale=0.35, speed=1.0, chunk_length=24, pause_length=0.3):
        
        t_ref_audio = 0.0
        t_text_proc = 0.0
        t_gpt_enc = 0.0
        t_gpt_dec = 0.0
        t_sovits = 0.0
        steps = 0
        t_total_start = time.perf_counter()

        # SSL + VQ
        t_start = time.perf_counter()
        wav16k, _ = librosa.load(ref_wav_path, sr=16000)
        wav16k_padded = np.concatenate([wav16k.astype(self.precision), np.zeros(int(16000 * 0.3), dtype=self.precision)])[None, :]
        ssl_content = self.run_sess(self.sess_ssl, {"audio": wav16k_padded})[0]
        codes = self.run_sess(self.sess_vq, {"ssl_content": ssl_content})[0]
        t_ref_audio += time.perf_counter() - t_start

        # Text Prep
        t_start = time.perf_counter()
        ref_phones, ref_bert = self.get_phones_and_bert(prompt_text, prompt_lang, self.version)
        
        # SoVITS Prep
        wav_ref, _ = librosa.load(ref_wav_path, sr=self.hps["data"]["sampling_rate"])
        spec = spectrogram_torch(torch.from_numpy(wav_ref)[None, :], self.hps["data"]["filter_length"], self.hps["data"]["sampling_rate"],
                                 self.hps["data"]["hop_length"], self.hps["data"]["win_length"], center=False).numpy().astype(self.precision)
        wav16k_sv, _ = librosa.load(ref_wav_path, sr=16000)
        sv_emb = self.sv_model.compute_embedding3(torch.from_numpy(wav16k_sv)[None, :]).detach().cpu().numpy().astype(self.precision)
        sv_size = 20480 if "Pro" in self.version else 512
        if sv_emb.shape[-1] != sv_size:
            tmp = np.zeros((1, sv_size), dtype=self.precision)
            tmp[:, :min(sv_emb.shape[-1], sv_size)] = sv_emb[:, :min(sv_emb.shape[-1], sv_size)]
            sv_emb = tmp
        t_text_proc += time.perf_counter() - t_start

        segments = split_text(text)
        sr = self.hps["data"]["sampling_rate"]
        samples_per_token = int((sr // 25) / speed)
        h_len, l_len, fade_len = 512, 16, 1280
        prev_fade_out = None

        # Mute Matrix
        mute_matrix_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPT_SoVITS/pretrained_models/gpts1_mute_emb_sim_matrix.pt")
        mute_matrix = torch.load(mute_matrix_path, map_location="cpu").numpy() if os.path.exists(mute_matrix_path) else None

        for seg_idx, seg_text in enumerate(segments):
            print(f"Processing segment {seg_idx+1}/{len(segments)}: {seg_text}")
            curr_phones, curr_bert = self.get_phones_and_bert(seg_text, text_lang, self.version)
            bert = np.concatenate([ref_bert, curr_bert], axis=1)[None, :, :]
            all_phones = np.array(ref_phones + curr_phones, dtype=np.int64)[None, :]
            
            t_start = time.perf_counter()
            topk_v, topk_i, k_cache, v_cache, x_len, y_len = self.run_sess(self.sess_gpt_enc, {
                "phoneme_ids": all_phones, "phoneme_ids_len": np.array([all_phones.shape[1]], dtype=np.int64),
                "prompts": prompt_semantic.astype(np.int64), "bert_feature": bert
            })
            t_gpt_enc += time.perf_counter() - t_start
            
            current_token = sample_topk(topk_v, topk_i, temperature=temperature)
            tokens = [current_token]
            
            # IO Binding setup for step
            io_binding = self.sess_gpt_step.io_binding()
            k_cache_ort = [self._to_ort(k_cache.astype(self.cache_dtype)), self._to_ort(k_cache.astype(self.cache_dtype))]
            v_cache_ort = [self._to_ort(v_cache.astype(self.cache_dtype)), self._to_ort(v_cache.astype(self.cache_dtype))]
            x_len_ort = onnxruntime.OrtValue.ortvalue_from_numpy(x_len.astype(np.int64))
            y_len_ort = onnxruntime.OrtValue.ortvalue_from_numpy(y_len.astype(np.int64))

            history_tokens = None
            chunk_queue = []
            token_counter = 0

            def decode_chunk(chunk_tokens, hist, lookahead):
                nonlocal t_sovits
                t_s = time.perf_counter()
                inp_list = []
                if hist is not None: inp_list.append(hist[:, -h_len:])
                inp_list.append(chunk_tokens)
                if lookahead is not None: inp_list.append(lookahead[:, :l_len])
                full_sem = np.concatenate(inp_list, axis=1)[:, None, :]
                
                audio = self.run_sess(self.sess_sovits, {
                    "pred_semantic": full_sem.astype(np.int64), "text_seq": np.array(curr_phones, dtype=np.int64)[None, :],
                    "refer_spec": spec, "sv_emb": sv_emb, "noise_scale": np.array([noise_scale], dtype=np.float32), "speed": np.array([speed], dtype=np.float32)
                })[0]
                
                h_samples = (hist[:, -h_len:].shape[1] if hist is not None else 0) * samples_per_token
                c_samples = chunk_tokens.shape[1] * samples_per_token
                res = audio.flatten()[h_samples : h_samples + c_samples]
                this_sovits_time = time.perf_counter() - t_s
                t_sovits += this_sovits_time
                return res, this_sovits_time

        t_dec_start = time.perf_counter()
        t_chunk_gpt_start = t_dec_start
        chunk_idx = 0
        for i in range(1500):
            steps += 1
            src_idx = i % 2
            dst_idx = (i + 1) % 2
            
            io_binding.bind_ortvalue_input("samples", self._to_ort(current_token.astype(np.int64)))
            io_binding.bind_ortvalue_input("k_cache", k_cache_ort[src_idx])
            io_binding.bind_ortvalue_input("v_cache", v_cache_ort[src_idx])
            io_binding.bind_ortvalue_input("idx", onnxruntime.OrtValue.ortvalue_from_numpy(np.array([i], dtype=np.int64)))
            io_binding.bind_ortvalue_input("x_len", x_len_ort)
            io_binding.bind_ortvalue_input("y_len", y_len_ort)
            io_binding.bind_output("topk_values", "cpu")
            io_binding.bind_output("topk_indices", "cpu")
            io_binding.bind_ortvalue_output("k_cache_new", k_cache_ort[dst_idx])
            io_binding.bind_ortvalue_output("v_cache_new", v_cache_ort[dst_idx])
            
            self.sess_gpt_step.run_with_iobinding(io_binding)
            outputs = io_binding.get_outputs()
            current_token = sample_topk(outputs[0].numpy(), outputs[1].numpy(), temperature=temperature)
            
            if current_token[0, 0] == 1024: break
            tokens.append(current_token)

            # Streaming split logic
            is_split = False
            if mute_matrix is not None and token_counter >= chunk_length + 2:
                recent_tokens = np.concatenate(tokens[-token_counter:], axis=1).flatten()
                scores = mute_matrix[recent_tokens] - 0.3
                scores[scores < 0] = -1
                scores[:-1] += scores[1:]
                argmax_idx = np.argmax(scores)
                if scores[argmax_idx] >= 0 and argmax_idx + 1 >= chunk_length:
                    split_idx = argmax_idx + 1
                    chunk_queue.append(np.concatenate(tokens[-token_counter : -token_counter + split_idx], axis=1))
                    token_counter -= split_idx
                    is_split = True
            elif mute_matrix is None and token_counter >= chunk_length:
                chunk_queue.append(np.concatenate(tokens[-token_counter:], axis=1))
                token_counter = 0
                is_split = True

            if is_split and len(chunk_queue) > 1:
                t_now = time.perf_counter()
                t_gpt_chunk = t_now - t_chunk_gpt_start
                t_gpt_dec += t_gpt_chunk
                
                curr = chunk_queue.pop(0)
                audio_data, t_sov_chunk = decode_chunk(curr, history_tokens, chunk_queue[0])
                if prev_fade_out is not None:
                    fade_in = np.linspace(0, 1, fade_len)
                    audio_data[:fade_len] = audio_data[:fade_len] * fade_in + prev_fade_out * (1 - fade_in)
                prev_fade_out = audio_data[-fade_len:]
                
                chunk_idx += 1
                audio_dur = len(audio_data) / sr
                print(f"Chunk {chunk_idx:02d} | GPT: {t_gpt_chunk:.4f}s | SoVITS: {t_sov_chunk:.4f}s | Audio: {audio_dur:.2f}s | RTF: {(t_gpt_chunk+t_sov_chunk)/audio_dur:.4f}")
                
                yield audio_data[:-fade_len]
                history_tokens = curr if history_tokens is None else np.concatenate([history_tokens, curr], axis=1)[:, -h_len:]
                t_chunk_gpt_start = time.perf_counter()

        # Handle remaining tokens
        t_now = time.perf_counter()
        t_gpt_final = t_now - t_chunk_gpt_start
        t_gpt_dec += t_gpt_final
        
        if token_counter > 0: chunk_queue.append(np.concatenate(tokens[-token_counter:], axis=1))
        while chunk_queue:
            curr = chunk_queue.pop(0)
            next_chunk = chunk_queue[0] if chunk_queue else None
            audio_data, t_sov_chunk = decode_chunk(curr, history_tokens, next_chunk)
            if prev_fade_out is not None:
                fade_in = np.linspace(0, 1, fade_len)
                audio_data[:fade_len] = audio_data[:fade_len] * fade_in + prev_fade_out * (1 - fade_in)
            
            chunk_idx += 1
            audio_dur = len(audio_data) / sr
            # 最后的 GPT 耗时只计入第一个剩余 chunk
            this_gpt_time = t_gpt_final if chunk_idx == (chunk_idx if not is_split else chunk_idx) else 0 
            # 剩余部分统一输出
            print(f"Chunk {chunk_idx:02d} | GPT: {this_gpt_time:.4f}s | SoVITS: {t_sov_chunk:.4f}s | Audio: {audio_dur:.2f}s | RTF: {(this_gpt_time+t_sov_chunk)/audio_dur:.4f}")

            if next_chunk is not None:
                prev_fade_out = audio_data[-fade_len:]
                yield audio_data[:-fade_len]
            else:
                yield audio_data
            history_tokens = curr if history_tokens is None else np.concatenate([history_tokens, curr], axis=1)[:, -h_len:]
            t_gpt_final = 0 # 仅计入一次

        t_total = time.perf_counter() - t_total_start
        t_step_avg = t_gpt_dec / steps if steps > 0 else 0.0
        print("\n--- Inference Timings (Streaming) ---")
        print(f"Ref Audio (SSL+VQ):   {t_ref_audio:.4f}s")
        print(f"Text (Cleaning+BERT): {t_text_proc:.4f}s")
        print(f"GPT Encoder:          {t_gpt_enc:.4f}s")
        print(f"GPT Decoding:         {t_gpt_dec:.4f}s ({steps} steps, {t_step_avg:.5f}s/step, {1/t_step_avg:.2f}step/s)")
        print(f"SoVITS Decoder:       {t_sovits:.4f}s")
        print(f"Total Time:           {t_total:.4f}s")
        print("-------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", default="onnx_export/firefly_v2_proplus_fp16")
    parser.add_argument("--gpt_path", required=True)
    parser.add_argument("--sovits_path", required=True)
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--ref_text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="out_onnx_stream.wav")
    parser.add_argument("--ref_lang", default="zh")
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--bert_path", default="pretrained_models/chinese-roberta-wwm-ext-large")
    parser.add_argument("--pause_length", type=float, default=0.3)
    args = parser.parse_args()

    infer = GPTSoVITS_ONNX_Streaming_Inference(args.onnx_dir, args.bert_path, args.sovits_path, device="cuda" if torch.cuda.is_available() else "cpu")
    full_audio = []
    for chunk in infer.infer_stream(args.ref_audio, args.ref_text, args.ref_lang, args.text, args.lang, pause_length=args.pause_length):
        full_audio.append(chunk)
    if full_audio:
        sf.write(args.output, np.concatenate(full_audio), infer.hps["data"]["sampling_rate"])
        print(f"Saved to {args.output}")
