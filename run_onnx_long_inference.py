import onnxruntime
import torch
import numpy as np
import argparse
import os
import sys
import re
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
    sentence_delimiters = r'([。！？.!?…\n])'
    parts = re.split(sentence_delimiters, text.strip("\n"))
    sentences = [parts[i] + parts[i+1] for i in range(0, len(parts)-1, 2)]
    if len(parts) % 2 == 1: sentences.append(parts[-1])
    merged, current = [], ""
    for s in sentences:
        if len(current) + len(s) < 20: current += s
        else:
            if current: merged.append(current)
            current = s
    if current: merged.append(current)
    return merged

def sample_topk(topk_values, topk_indices, temperature=1.0):
    if temperature != 1.0: topk_values = topk_values / temperature
    topk_values -= np.max(topk_values, axis=-1, keepdims=True)
    probs = np.exp(topk_values)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return np.array([topk_indices[i, np.random.choice(len(probs[i]), p=probs[i])] for i in range(probs.shape[0])], dtype=np.int64)[:, None]

class GPTSoVITS_ONNX_Long_Inference:
    def __init__(self, onnx_dir, bert_path, sovits_path, device="cpu"):
        self.onnx_dir, self.device = onnx_dir, device
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

    def get_phones_and_bert(self, text, language):
        phones, word2ph, norm_text = clean_text(text, language, self.version)
        phones = cleaned_text_to_sequence(phones, self.version)
        if language == "zh":
            inputs = self.tokenizer(norm_text, return_tensors="np")
            hidden_states = self.sess_bert.run(["hidden_states"], {k: v.astype(np.int64) for k, v in inputs.items()})[0]
            res = hidden_states[0][1:-1]
            bert = np.concatenate([np.tile(res[j], (word2ph[j], 1)) for j in range(len(word2ph))], axis=0).T
        else:
            bert = np.zeros((1024, len(phones)), dtype=self.precision)
        return phones, bert.astype(self.precision)

    def _to_ort(self, data):
        return onnxruntime.OrtValue.ortvalue_from_numpy(data, "cuda", 0) if self.device == "cuda" else onnxruntime.OrtValue.ortvalue_from_numpy(data)

    def infer_long(self, ref_wav_path, prompt_text, prompt_lang, text, text_lang,
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
        ssl_content = self.sess_ssl.run(None, {"audio": wav16k_padded})[0]
        prompt_semantic = self.sess_vq.run(None, {"ssl_content": ssl_content.astype(self.precision)})[0][0, 0][None, :]
        t_ref_audio += time.perf_counter() - t_start
        
        # Text Prep
        t_start = time.perf_counter()
        ref_phones, ref_bert = self.get_phones_and_bert(prompt_text, prompt_lang)
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
        history_phones, history_bert, history_tokens = ref_phones, ref_bert, prompt_semantic
        prev_fade_out, fade_len, sr = None, 1280, self.hps["data"]["sampling_rate"]
        samples_per_token = int((sr // 25) / speed)
        h_len, l_len = 16, 16
        
        mute_matrix_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPT_SoVITS/pretrained_models/gpts1_mute_emb_sim_matrix.pt")
        mute_matrix = torch.load(mute_matrix_path, map_location="cpu").numpy() if os.path.exists(mute_matrix_path) else None

        for i, seg_text in enumerate(segments):
            # Text segment
            t_start = time.perf_counter()
            curr_phones, curr_bert = self.get_phones_and_bert(seg_text, text_lang)
            if i == 0:
                inp_phones, inp_bert, inp_prompt = ref_phones + curr_phones, np.concatenate([ref_bert, curr_bert], 1), prompt_semantic
            else:
                inp_phones, inp_bert, inp_prompt = ref_phones + history_phones + curr_phones, np.concatenate([ref_bert, history_bert, curr_bert], 1), np.concatenate([prompt_semantic, history_tokens], 1)
            
            inp_phones_tensor = np.array(inp_phones, dtype=np.int64)[None, :]
            t_text_proc += time.perf_counter() - t_start

            # GPT Encoder
            t_start = time.perf_counter()
            topk_v, topk_i, k_cache, v_cache, x_len, y_len = self.sess_gpt_enc.run(None, {
                "phoneme_ids": inp_phones_tensor, "phoneme_ids_len": np.array([inp_phones_tensor.shape[1]], dtype=np.int64),
                "prompts": inp_prompt.astype(np.int64), "bert_feature": inp_bert[None, :, :].astype(self.precision)
            })
            t_gpt_enc += time.perf_counter() - t_start

            seg_tokens_list = []
            current_token = sample_topk(topk_v, topk_i, temperature=temperature)
            
            io_binding = self.sess_gpt_step.io_binding()
            k_cache_ort = [self._to_ort(k_cache.astype(self.cache_dtype)), self._to_ort(k_cache.astype(self.cache_dtype))]
            v_cache_ort = [self._to_ort(v_cache.astype(self.cache_dtype)), self._to_ort(v_cache.astype(self.cache_dtype))]
            x_len_ort, y_len_ort = onnxruntime.OrtValue.ortvalue_from_numpy(x_len.astype(np.int64)), onnxruntime.OrtValue.ortvalue_from_numpy(y_len.astype(np.int64))

            chunk_queue, seg_history_tokens, token_counter = [], None, 0
            
            # GPT Step loop
            t_dec_start = time.perf_counter()
            t_chunk_gpt_start = t_dec_start
            chunk_idx = 0
            for j in range(1500):
                steps += 1
                token_counter += 1
                src_idx, dst_idx = j % 2, (j + 1) % 2
                io_binding.bind_ortvalue_input("samples", self._to_ort(current_token.astype(np.int64)))
                io_binding.bind_ortvalue_input("k_cache", k_cache_ort[src_idx])
                io_binding.bind_ortvalue_input("v_cache", v_cache_ort[src_idx])
                io_binding.bind_ortvalue_input("idx", onnxruntime.OrtValue.ortvalue_from_numpy(np.array([j], dtype=np.int64)))
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
                seg_tokens_list.append(current_token)

                is_split = False
                if mute_matrix is not None and token_counter >= chunk_length + 2:
                    recent = np.concatenate(seg_tokens_list[-token_counter:], axis=1).flatten()
                    scores = mute_matrix[recent] - 0.3
                    scores[scores < 0] = -1
                    scores[:-1] += scores[1:]
                    argmax_idx = np.argmax(scores)
                    if scores[argmax_idx] >= 0 and argmax_idx + 1 >= chunk_length:
                        split_idx = argmax_idx + 1
                        chunk_queue.append(np.concatenate(seg_tokens_list[-token_counter : -token_counter + split_idx], axis=1))
                        token_counter -= split_idx
                        is_split = True
                elif token_counter >= chunk_length:
                    chunk_queue.append(np.concatenate(seg_tokens_list[-token_counter:], axis=1))
                    token_counter, is_split = 0, True

                if is_split and len(chunk_queue) > 1:
                    t_now = time.perf_counter()
                    t_gpt_chunk = t_now - t_chunk_gpt_start
                    t_gpt_dec += t_gpt_chunk
                    
                    t_start = time.perf_counter()
                    curr = chunk_queue.pop(0)
                    full_sem = np.concatenate([seg_history_tokens[:, -h_len:] if seg_history_tokens is not None else np.zeros((1,0)), curr, chunk_queue[0][:, :l_len]], axis=1)[:, None, :]
                    audio = self.sess_sovits.run(None, {"pred_semantic": full_sem.astype(np.int64), "text_seq": np.array(curr_phones, dtype=np.int64)[None, :], "refer_spec": spec, "sv_emb": sv_emb, "noise_scale": np.array([noise_scale], dtype=np.float32), "speed": np.array([speed], dtype=np.float32)})[0]
                    h_s = (h_len if seg_history_tokens is not None else 0) * samples_per_token
                    audio_data = audio.flatten()[h_s : h_s + curr.shape[1] * samples_per_token]
                    if prev_fade_out is not None:
                        audio_data[:fade_len] = audio_data[:fade_len] * np.linspace(0, 1, fade_len) + prev_fade_out * np.linspace(1, 0, fade_len)
                    prev_fade_out = audio_data[-fade_len:]
                    
                    this_sov_time = time.perf_counter() - t_start
                    t_sovits += this_sov_time
                    
                    chunk_idx += 1
                    audio_dur = len(audio_data) / sr
                    print(f"Seg {i:02d} Chunk {chunk_idx:02d} | GPT: {t_gpt_chunk:.4f}s | SoVITS: {this_sov_time:.4f}s | Audio: {audio_dur:.2f}s | RTF: {(t_gpt_chunk+this_sov_time)/audio_dur:.4f}")

                    yield audio_data[:-fade_len]
                    seg_history_tokens = curr
                    t_chunk_gpt_start = time.perf_counter()

            t_now = time.perf_counter()
            t_gpt_final = t_now - t_chunk_gpt_start
            t_gpt_dec += t_gpt_final

            # End of segment handling
            if token_counter > 0: chunk_queue.append(np.concatenate(seg_tokens_list[-token_counter:], axis=1))
            while chunk_queue:
                t_start = time.perf_counter()
                curr = chunk_queue.pop(0)
                next_c = chunk_queue[0] if chunk_queue else None
                full_sem = np.concatenate([seg_history_tokens[:, -h_len:] if seg_history_tokens is not None else np.zeros((1,0)), curr, next_c[:, :l_len] if next_c is not None else np.zeros((1,0))], axis=1)[:, None, :]
                audio = self.sess_sovits.run(None, {"pred_semantic": full_sem.astype(np.int64), "text_seq": np.array(curr_phones, dtype=np.int64)[None, :], "refer_spec": spec, "sv_emb": sv_emb, "noise_scale": np.array([noise_scale], dtype=np.float32), "speed": np.array([speed], dtype=np.float32)})[0]
                h_s = (h_len if seg_history_tokens is not None else 0) * samples_per_token
                audio_data = audio.flatten()[h_s : h_s + curr.shape[1] * samples_per_token]
                if prev_fade_out is not None:
                    audio_data[:fade_len] = audio_data[:fade_len] * np.linspace(0, 1, fade_len) + prev_fade_out * (1 - np.linspace(0, 1, fade_len))
                
                this_sov_time = time.perf_counter() - t_start
                t_sovits += this_sov_time
                
                chunk_idx += 1
                audio_dur = len(audio_data) / sr
                this_gpt_time = t_gpt_final if chunk_idx == (chunk_idx if not is_split else chunk_idx) else 0
                print(f"Seg {i:02d} Chunk {chunk_idx:02d} | GPT: {this_gpt_time:.4f}s | SoVITS: {this_sov_time:.4f}s | Audio: {audio_dur:.2f}s | RTF: {(this_gpt_time+this_sov_time)/audio_dur:.4f}")

                prev_fade_out = audio_data[-fade_len:]
                yield audio_data[:-fade_len]
                seg_history_tokens = curr
                t_gpt_final = 0

            history_phones, history_bert, history_tokens = curr_phones, curr_bert, np.concatenate(seg_tokens_list, 1)
            if history_tokens.shape[1] > 125:
                history_phones, history_tokens = history_phones[-75:], history_tokens[:, -125:]
                history_bert = history_bert[:, -len(history_phones):]
            
            # Add pause between segments
            if i < len(segments) - 1 and pause_length > 0:
                if prev_fade_out is not None:
                    yield prev_fade_out
                    prev_fade_out = None
                yield np.zeros(int(sr * pause_length), dtype=np.float32)
        
        if prev_fade_out is not None: yield prev_fade_out
        
        t_total = time.perf_counter() - t_total_start
        t_step_avg = t_gpt_dec / steps if steps > 0 else 0.0
        print("\n--- Inference Timings (Long) ---")
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
    parser.add_argument("--output", default="out_onnx_long.wav")
    parser.add_argument("--ref_lang", default="zh")
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--bert_path", default="pretrained_models/chinese-roberta-wwm-ext-large")
    parser.add_argument("--pause_length", type=float, default=0.3)
    args = parser.parse_args()

    infer = GPTSoVITS_ONNX_Long_Inference(args.onnx_dir, args.bert_path, args.sovits_path, device="cuda" if torch.cuda.is_available() else "cpu")
    full_audio = [chunk for chunk in infer.infer_long(args.ref_audio, args.ref_text, args.ref_lang, args.text, args.lang, pause_length=args.pause_length)]
    if full_audio:
        sf.write(args.output, np.concatenate(full_audio), infer.hps["data"]["sampling_rate"])
        print(f"Saved to {args.output}")
