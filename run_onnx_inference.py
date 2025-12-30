import onnxruntime
import torch
import numpy as np
import argparse
import os
import librosa
import soundfile as sf
import sys
import torchaudio
import time


so = onnxruntime.SessionOptions()
so.log_severity_level = 1

# Setup paths
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "GPT_SoVITS"))

from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast
from GPT_SoVITS.module.mel_processing import spectrogram_torch
from GPT_SoVITS.sv import SV
from transformers import AutoTokenizer

def sample_logits(logits, top_k=5, top_p=1.0, temperature=1.0):
    # logits: [B, Vocab]
    if temperature != 1.0:
        logits = logits / temperature
    
    # Top-K
    if top_k > 0:
        k_th_value = np.partition(logits, -top_k, axis=-1)[:, -top_k][:, None]
        indices_to_remove = logits < k_th_value
        logits[indices_to_remove] = -np.inf
    
    # Softmax
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    
    # Sample
    samples = []
    for i in range(probs.shape[0]):
        p = probs[i]
        # In case of rounding errors
        p = p / p.sum()
        sample = np.random.choice(probs.shape[1], p=p)
        samples.append(sample)
    return np.array(samples, dtype=np.int64)[:, None]

class GPTSoVITS_ONNX_Inference:
    def __init__(self, onnx_dir, bert_path, sovits_path, device="cpu"):
        self.onnx_dir = onnx_dir
        self.bert_path = bert_path
        self.device = device
        
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        
        print(f"Loading ONNX models from {onnx_dir} on {device}...")
        self.sess_ssl = onnxruntime.InferenceSession(f"{onnx_dir}/ssl.onnx", providers=providers)
        self.sess_bert = onnxruntime.InferenceSession(f"{onnx_dir}/bert.onnx", providers=providers)
        self.sess_vq = onnxruntime.InferenceSession(f"{onnx_dir}/vq_encoder.onnx", providers=providers)
        self.sess_gpt_enc = onnxruntime.InferenceSession(f"{onnx_dir}/gpt_encoder.onnx", providers=providers)
        self.sess_gpt_step = onnxruntime.InferenceSession(f"{onnx_dir}/gpt_step.onnx", providers=providers)
        self.sess_sovits = onnxruntime.InferenceSession(f"{onnx_dir}/sovits.onnx", providers=providers)
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        
        # Determine version from original sovits path if provided
        from GPT_SoVITS.process_ckpt import load_sovits_new, get_sovits_version_from_path_fast
        dict_s2 = load_sovits_new(sovits_path)
        self.hps = dict_s2["config"]
        _, self.version, _ = get_sovits_version_from_path_fast(sovits_path)
        print(f"Detected model version: {self.version}")
        
        # Ensure hps matches what was used during export
        self.hps["model"]["version"] = self.version
        self.hps["model"]["semantic_frame_rate"] = "25hz"
        
        # Load SV model (PyTorch for now as it's not in ONNX)
        self.sv_model = SV(device, False)
        
        # Warm up models to avoid first-run latency
        self.warmup()

    def warmup(self):
        print("Warming up ONNX models...")
        # Synthetic inputs for warmup (matching export_onnx.py dummy shapes)
        audio = np.zeros((1, 32000), dtype=np.float32)
        input_ids = np.zeros((1, 20), dtype=np.int64)
        attention_mask = np.ones((1, 20), dtype=np.int64)
        token_type_ids = np.zeros((1, 20), dtype=np.int64)
        ssl_content = np.zeros((1, 768, 100), dtype=np.float32)
        phoneme_ids = np.zeros((1, 50), dtype=np.int64)
        phoneme_ids_len = np.array([50], dtype=np.int64)
        prompts = np.zeros((1, 20), dtype=np.int64)
        bert_feature = np.zeros((1, 1024, 50), dtype=np.float32)
        
        # Version-specific shapes
        spec_channels = 1025 if self.version != "v1" else 513
        sv_emb_size = 20480 if "Pro" in self.version else 512
        
        # Warm up SSL, BERT, VQ
        for _ in range(2):
            _ = self.run_sess(self.sess_ssl, {"audio": audio})
            _ = self.run_sess(self.sess_bert, {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids})
            _ = self.run_sess(self.sess_vq, {"ssl_content": ssl_content})
        
        # Warm up GPT Encoder
        res_enc = self.run_sess(self.sess_gpt_enc, {
            "phoneme_ids": phoneme_ids,
            "phoneme_ids_len": phoneme_ids_len,
            "prompts": prompts,
            "bert_feature": bert_feature
        })
        logits, k_cache, v_cache, x_len, y_len = res_enc
        
        # Warm up GPT Step (run a few steps)
        samples = np.zeros((1, 1), dtype=np.int64)
        for i in range(5):
            res_step = self.run_sess(self.sess_gpt_step, {
                "samples": samples,
                "k_cache": k_cache,
                "v_cache": v_cache,
                "x_len": x_len,
                "y_len": y_len,
                "idx": np.array(i, dtype=np.int64)
            })
            _, k_cache, v_cache = res_step
        
        # Warm up SoVITS
        pred_semantic = np.zeros((1, 1, 150), dtype=np.int64)
        text_seq = np.zeros((1, 50), dtype=np.int64)
        refer_spec = np.zeros((1, spec_channels, 200), dtype=np.float32)
        sv_emb = np.zeros((1, sv_emb_size), dtype=np.float32)
        for _ in range(2):
            _ = self.run_sess(self.sess_sovits, {
                "pred_semantic": pred_semantic,
                "text_seq": text_seq,
                "refer_spec": refer_spec,
                "sv_emb": sv_emb
            })
        print("Warmup complete.")

    def get_bert_feature(self, text, word2ph, language):
        if language != "zh":
            # For non-zh, return zero features matching the phoneme length
            # Note: Clean text should have been called before this to get phones
            return None 

        inputs = self.tokenizer(text, return_tensors="np")
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)
        token_type_ids = inputs["token_type_ids"].astype(np.int64)
        
        hidden_states = self.sess_bert.run(["hidden_states"], {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        })[0] # [1, T, 1024]
        
        res = hidden_states[0][1:-1]
        
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = np.tile(res[i], (word2ph[i], 1))
            phone_level_feature.append(repeat_feature)
        
        phone_level_feature = np.concatenate(phone_level_feature, axis=0)
        return phone_level_feature.T # [1024, T]

    def run_sess(self, sess, inputs):
        expected = [i.name for i in sess.get_inputs()]
        actual_inputs = {k: v for k, v in inputs.items() if k in expected}
        return sess.run(None, actual_inputs)

    def infer(self, ref_wav_path, prompt_text, prompt_lang, text, text_lang, 
              top_k=5, temperature=1.0, output_path="out.wav"):
        
        t_total_start = time.perf_counter()
        
        print("Processing reference audio...")
        t_ref_start = time.perf_counter()
        wav16k, _ = librosa.load(ref_wav_path, sr=16000)
        zero_wav = np.zeros(int(16000 * 0.3), dtype=np.float32)
        wav16k_padded = np.concatenate([wav16k, zero_wav])[None, :]
        
        ssl_content = self.run_sess(self.sess_ssl, {"audio": wav16k_padded})[0]
        ssl_content = ssl_content.transpose(0, 2, 1) # [1, 768, T]
        
        codes = self.run_sess(self.sess_vq, {"ssl_content": ssl_content})[0]
        prompt_semantic = codes[0, 0][None, :] # [1, T]
        t_ref_end = time.perf_counter()
        
        print("Cleaning text and extracting BERT features...")
        t_text_start = time.perf_counter()
        phones1, word2ph1, norm_text1 = clean_text(prompt_text, prompt_lang, self.version)
        phones1 = cleaned_text_to_sequence(phones1, self.version)
        
        phones2, word2ph2, norm_text2 = clean_text(text, text_lang, self.version)
        phones2 = cleaned_text_to_sequence(phones2, self.version)
        
        bert1 = self.get_bert_feature(norm_text1, word2ph1, prompt_lang)
        bert2 = self.get_bert_feature(norm_text2, word2ph2, text_lang)
        
        if bert1 is None: bert1 = np.zeros((1024, len(phones1)), dtype=np.float32)
        if bert2 is None: bert2 = np.zeros((1024, len(phones2)), dtype=np.float32)
        
        bert = np.concatenate([bert1, bert2], axis=1)[None, :, :].astype(np.float32)
        
        all_phoneme_ids = np.array(phones1 + phones2, dtype=np.int64)[None, :]
        all_phoneme_len = np.array([all_phoneme_ids.shape[1]], dtype=np.int64)
        t_text_end = time.perf_counter()
        
        print("Running GPT Encoder...")
        t_gpt_enc_start = time.perf_counter()
        logits, k_cache, v_cache, x_len, y_len = self.run_sess(self.sess_gpt_enc, {
            "phoneme_ids": all_phoneme_ids,
            "phoneme_ids_len": all_phoneme_len,
            "prompts": prompt_semantic.astype(np.int64),
            "bert_feature": bert
        })
        t_gpt_enc_end = time.perf_counter()
        
        current_samples = sample_logits(logits, top_k=top_k, temperature=temperature)
        decoded_semantic = [prompt_semantic, current_samples]
        
        print("Running GPT Step loop...")
        t_gpt_dec_start = time.perf_counter()
        max_steps = 1500
        reached_eos = False
        steps = 0
        for i in range(max_steps):
            step_outputs = self.run_sess(self.sess_gpt_step, {
                "samples": current_samples,
                "k_cache": k_cache,
                "v_cache": v_cache,
                "x_len": x_len,
                "y_len": y_len,
                "idx": np.array(i, dtype=np.int64)
            })
            logits, k_cache, v_cache = step_outputs
            
            current_samples = sample_logits(logits, top_k=top_k, temperature=temperature)
            decoded_semantic.append(current_samples)
            steps += 1
            if current_samples[0, 0] == 1024:
                print(f"Reached EOS at step {i}")
                reached_eos = True
                break
        t_gpt_dec_end = time.perf_counter()
        
        pred_semantic = np.concatenate(decoded_semantic, axis=1)
        generated_sem = pred_semantic[:, prompt_semantic.shape[1]:]
        if generated_sem[0, -1] == 1024:
            generated_sem = generated_sem[:, :-1]
        generated_sem = generated_sem[:, None, :] # [1, 1, T]
        
        print("Running SoVITS Decoder...")
        t_sovits_start = time.perf_counter()
        # Get refer spectrogram
        # Ensure hps is object-like if it's a dict
        class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super(AttrDict, self).__init__(*args, **kwargs)
                self.__dict__ = self
        
        hps = AttrDict(self.hps)
        if isinstance(hps.data, dict): hps.data = AttrDict(hps.data)

        wav_ref, sr_ref = librosa.load(ref_wav_path, sr=hps.data.sampling_rate)
        wav_ref_tensor = torch.from_numpy(wav_ref)[None, :]
        spec = spectrogram_torch(
            wav_ref_tensor, 
            hps.data.filter_length, 
            hps.data.sampling_rate, 
            hps.data.hop_length, 
            hps.data.win_length, 
            center=False
        ).numpy()
        
        # Get SV Embedding
        wav16k_sv, _ = librosa.load(ref_wav_path, sr=16000)
        sv_emb = self.sv_model.compute_embedding3(torch.from_numpy(wav16k_sv)[None, :]).detach().cpu().numpy()
        
        # Align sv_emb size for Pro models if needed
        sv_emb_expected = 20480 if "Pro" in self.version else 512
        if sv_emb.shape[-1] != sv_emb_expected:
            padded_sv_emb = np.zeros((1, sv_emb_expected), dtype=np.float32)
            # Copy available embedding data
            min_dim = min(sv_emb.shape[-1], sv_emb_expected)
            padded_sv_emb[:, :min_dim] = sv_emb[:, :min_dim]
            sv_emb = padded_sv_emb

        sovits_inputs = {
            "pred_semantic": generated_sem.astype(np.int64),
            "text_seq": np.array(phones2, dtype=np.int64)[None, :],
            "refer_spec": spec.astype(np.float32),
            "sv_emb": sv_emb.astype(np.float32),
        }
        
        audio = self.run_sess(self.sess_sovits, sovits_inputs)[0]
        t_sovits_end = time.perf_counter()
        
        t_total_end = time.perf_counter()
        
        print(f"\n--- Inference Timings (Warmed Up) ---")
        print(f"Ref Audio (SSL+VQ): {t_ref_end - t_ref_start:.4f}s")
        print(f"Text (Cleaning+BERT): {t_text_end - t_text_start:.4f}s")
        print(f"GPT Encoder: {t_gpt_enc_end - t_gpt_enc_start:.4f}s")
        print(f"GPT Decoding: {t_gpt_dec_end - t_gpt_dec_start:.4f}s ({steps} steps, {(t_gpt_dec_end - t_gpt_dec_start)/max(1, steps):.5f}s/step)")
        print(f"SoVITS Decoder: {t_sovits_end - t_sovits_start:.4f}s")
        print(f"Total Time: {t_total_end - t_total_start:.4f}s")
        print(f"-------------------------------------\n")

        
        sf.write(output_path, audio.squeeze(), self.hps["data"]["sampling_rate"])
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS ONNX Inference")
    parser.add_argument("--onnx_dir", default="onnx_export", help="Path to ONNX models directory")
    parser.add_argument("--gpt_path", required=True, help="Original GPT path (for version detection)")
    parser.add_argument("--sovits_path", required=True, help="Original SoVITS path (for version detection)")
    parser.add_argument("--bert_path", default="pretrained_models/chinese-roberta-wwm-ext-large")
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--ref_text", required=True)
    parser.add_argument("--ref_lang", default="zh")
    parser.add_argument("--text", required=True)
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--output", default="output_onnx.wav")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference = GPTSoVITS_ONNX_Inference(args.onnx_dir, args.bert_path, args.sovits_path, device=device)
    inference.infer(
        args.ref_audio,
        args.ref_text,
        args.ref_lang,
        args.text,
        args.lang,
        output_path=args.output
    )
