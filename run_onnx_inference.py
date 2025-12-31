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
from transformers import AutoTokenizer

# os.environ["http_proxy"]='http://127.0.0.1:10809'
# os.environ["https_proxy"]='http://127.0.0.1:10809'
# import nltk
# nltk.download('averaged_perceptron_tagger_eng')

# Setup paths
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "GPT_SoVITS"))

from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from GPT_SoVITS.module.mel_processing import spectrogram_torch
from GPT_SoVITS.sv import SV


def sample_logits(logits, top_k=5, top_p=1.0, temperature=1.0):
    logits = logits.astype(np.float32)
    if temperature != 1.0:
        logits = logits / temperature
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    if top_k > 0:
        k_th_value = np.partition(logits, -top_k, axis=-1)[:, -top_k][:, None]
        indices_to_remove = logits < k_th_value
        logits[indices_to_remove] = -np.inf
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    samples = []
    for i in range(probs.shape[0]):
        p = probs[i]
        if np.isnan(p).any():
            p = np.ones_like(p) / len(p)
        p = p / p.sum()
        sample = np.random.choice(probs.shape[1], p=p)
        samples.append(sample)
    return np.array(samples, dtype=np.int64)[:, None]


class GPTSoVITS_ONNX_Inference:
    def __init__(self, onnx_dir, bert_path, sovits_path, device="cpu"):
        self.onnx_dir = onnx_dir
        self.device = device
        
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 4
        so = None
        sol = onnxruntime.SessionOptions()
        sol.log_severity_level = 1
        # sol = None
        # Prevent aggressive CPU fallback heuristics by limiting optimization level
        # sol.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        if device == "cuda":
            self.providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
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

        self.sess_sovits = onnxruntime.InferenceSession(f"{onnx_dir}/sovits.onnx", sess_options=sol,
                                                        providers=self.providers)

        self.step_inputs_info = {node.name: node.type for node in self.sess_gpt_step.get_inputs()}
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)

        from GPT_SoVITS.process_ckpt import load_sovits_new, get_sovits_version_from_path_fast
        dict_s2 = load_sovits_new(sovits_path)
        self.hps = dict_s2["config"]
        _, self.version, _ = get_sovits_version_from_path_fast(sovits_path)
        print(f"Detected model version: {self.version}")

        self.hps["model"]["version"] = self.version
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

        # 1. Text Cleaner Warmup (加载字典)
        # 针对中英双语分别调用一次，触发 Lazy Loading
        try:
            # 简单词汇即可，覆盖 zh 和 en 逻辑分支
            _ = clean_text("预热", "zh", self.version)
            _ = clean_text("Warmup", "en", self.version)
        except Exception as e:
            print(f"Text Cleaner Warmup Warning: {e}")

        # 2. ONNX Models Warmup
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

            # GPT Step (Init Cache)
            cache_dtype = np.float32
            if self.step_inputs_info.get("k_cache", "").find("float16") != -1:
                cache_dtype = np.float16

            # 触发一次即可分配显存
            # 构造最小 Dummy 输入以通过 Shape 检查
            # 假设 standard layer/head 配置，如有特定维度需求可调整
            dummy_cache = np.zeros((32, 2, 1, 16, 0, 64), dtype=cache_dtype)

            # SoVITS
            dummy_sem = np.zeros((1, 256), dtype=np.int64)
            dummy_seq = np.zeros((1, 256), dtype=np.int64)
            dummy_spec = np.zeros((1, 1025, 100), dtype=self.precision)
            dummy_emb = np.zeros((1, 256), dtype=self.precision)

            # 运行 SoVITS 空跑 (如果有维度报错可忽略，重点是加载库)
            pass

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

    def _to_gpu_ort(self, data):
        if self.device != "cuda": return onnxruntime.OrtValue.ortvalue_from_numpy(data)
        return onnxruntime.OrtValue.ortvalue_from_numpy(data, "cuda", 0)

    def infer(self, ref_wav_path, prompt_text, prompt_lang, text, text_lang,
              top_k=5, temperature=1.0, output_path="out.wav"):

        # Timers
        t_ref_audio = 0.0
        t_text_proc = 0.0
        t_gpt_enc = 0.0
        t_gpt_dec = 0.0
        t_sovits = 0.0
        steps = 0

        t_total_start = time.perf_counter()

        # 1. Audio
        t_start = time.perf_counter()
        wav16k, _ = librosa.load(ref_wav_path, sr=16000)
        wav16k = wav16k.astype(self.precision)
        zero_wav = np.zeros(int(16000 * 0.3), dtype=self.precision)
        wav16k_padded = np.concatenate([wav16k, zero_wav])[None, :]

        ssl_content = self.run_sess(self.sess_ssl, {"audio": wav16k_padded})[0]
        ssl_content = ssl_content.transpose(0, 2, 1)

        codes = self.run_sess(self.sess_vq, {"ssl_content": ssl_content})[0]
        prompt_semantic = codes[0, 0][None, :]
        t_ref_audio = time.perf_counter() - t_start

        # 2. Text
        t_start = time.perf_counter()
        phones1, word2ph1, norm_text1 = clean_text(prompt_text, prompt_lang, self.version)
        phones1 = cleaned_text_to_sequence(phones1, self.version)
        phones2, word2ph2, norm_text2 = clean_text(text, text_lang, self.version)
        phones2 = cleaned_text_to_sequence(phones2, self.version)

        bert1 = self.get_bert_feature(norm_text1, word2ph1, prompt_lang)
        bert2 = self.get_bert_feature(norm_text2, word2ph2, text_lang)
        if bert1 is None: bert1 = np.zeros((1024, len(phones1)), dtype=self.precision)
        if bert2 is None: bert2 = np.zeros((1024, len(phones2)), dtype=self.precision)
        bert = np.concatenate([bert1, bert2], axis=1)[None, :, :].astype(self.precision)

        all_phoneme_ids = np.array(phones1 + phones2, dtype=np.int64)[None, :]
        all_phoneme_len = np.array([all_phoneme_ids.shape[1]], dtype=np.int64)
        t_text_proc = time.perf_counter() - t_start

        # 3. GPT Encoder
        print("Running GPT Encoder...")
        t_start = time.perf_counter()
        logits, k_cache, v_cache, x_len, y_len = self.run_sess(self.sess_gpt_enc, {
            "phoneme_ids": all_phoneme_ids,
            "phoneme_ids_len": all_phoneme_len,
            "prompts": prompt_semantic.astype(np.int64),
            "bert_feature": bert
        })
        t_gpt_enc = time.perf_counter() - t_start

        current_samples = sample_logits(logits, top_k=top_k, temperature=temperature)
        decoded_semantic_list = [prompt_semantic, current_samples]

        # 4. GPT Step
        print(f"Running GPT Step (IO Binding Mode)...")
        t_dec_start = time.perf_counter()

        max_steps = 1500

        cache_dtype = np.float32
        if self.step_inputs_info.get("k_cache", "").find("float16") != -1:
            cache_dtype = np.float16

        k_cache_ort = self._to_gpu_ort(k_cache.astype(cache_dtype))
        v_cache_ort = self._to_gpu_ort(v_cache.astype(cache_dtype))

        x_len_ort = None
        y_len_ort = None
        if "x_len" in self.step_inputs_info: x_len_ort = self._to_gpu_ort(x_len)
        if "y_len" in self.step_inputs_info: y_len_ort = self._to_gpu_ort(y_len)

        device_type_binding = "cuda" if self.device == "cuda" else "cpu"
        device_id_binding = 0

        for i in range(max_steps):
            io_binding = self.sess_gpt_step.io_binding()

            samples_ort = self._to_gpu_ort(current_samples)
            io_binding.bind_ortvalue_input("samples", samples_ort)
            io_binding.bind_ortvalue_input("k_cache", k_cache_ort)
            io_binding.bind_ortvalue_input("v_cache", v_cache_ort)

            idx_ort = self._to_gpu_ort(np.array([i], dtype=np.int64))
            io_binding.bind_ortvalue_input("idx", idx_ort)

            if x_len_ort: io_binding.bind_ortvalue_input("x_len", x_len_ort)
            if y_len_ort: io_binding.bind_ortvalue_input("y_len", y_len_ort)

            io_binding.bind_output("logits", "cpu")
            io_binding.bind_output("k_cache_new", device_type_binding, device_id=device_id_binding)
            io_binding.bind_output("v_cache_new", device_type_binding, device_id=device_id_binding)

            self.sess_gpt_step.run_with_iobinding(io_binding)

            outputs = io_binding.get_outputs()
            logits = outputs[0].numpy()
            k_cache_ort = outputs[1]
            v_cache_ort = outputs[2]

            current_samples = sample_logits(logits, top_k=top_k, temperature=temperature)
            decoded_semantic_list.append(current_samples)
            steps += 1
            if current_samples[0, 0] == 1024:
                break

        t_dec_end = time.perf_counter()
        t_gpt_dec = t_dec_end - t_dec_start

        pred_semantic = np.concatenate(decoded_semantic_list, axis=1)
        generated_sem = pred_semantic[:, prompt_semantic.shape[1]:]
        if generated_sem[0, -1] == 1024: generated_sem = generated_sem[:, :-1]
        generated_sem = generated_sem[:, None, :]

        # 5. SoVITS
        print("Running SoVITS Decoder...")
        t_start = time.perf_counter()

        class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super(AttrDict, self).__init__(*args, **kwargs)
                self.__dict__ = self

        hps = AttrDict(self.hps)
        if isinstance(hps.data, dict): hps.data = AttrDict(hps.data)

        wav_ref, _ = librosa.load(ref_wav_path, sr=hps.data.sampling_rate)
        spec = spectrogram_torch(torch.from_numpy(wav_ref)[None, :], hps.data.filter_length, hps.data.sampling_rate,
                                 hps.data.hop_length, hps.data.win_length, center=False).numpy()

        wav16k_sv, _ = librosa.load(ref_wav_path, sr=16000)
        sv_emb = self.sv_model.compute_embedding3(torch.from_numpy(wav16k_sv)[None, :]).detach().cpu().numpy()

        sv_size = 20480 if "Pro" in self.version else 512
        if sv_emb.shape[-1] != sv_size:
            tmp = np.zeros((1, sv_size), dtype=np.float32)
            tmp[:, :min(sv_emb.shape[-1], sv_size)] = sv_emb[:, :min(sv_emb.shape[-1], sv_size)]
            sv_emb = tmp

        audio = self.run_sess(self.sess_sovits, {
            "pred_semantic": generated_sem.astype(np.int64),
            "text_seq": np.array(phones2, dtype=np.int64)[None, :],
            "refer_spec": spec.astype(self.precision),
            "sv_emb": sv_emb.astype(self.precision),
        })[0]

        t_sovits = time.perf_counter() - t_start
        t_total = time.perf_counter() - t_total_start

        sf.write(output_path, audio.squeeze().astype(np.float32), hps.data.sampling_rate)
        print(f"write {output_path}")
        # Report
        t_step_avg = t_gpt_dec / steps if steps > 0 else 0.0
        print("\n--- Inference Timings (Warmed Up) ---")
        print(f"Ref Audio (SSL+VQ):   {t_ref_audio:.4f}s")
        print(f"Text (Cleaning+BERT): {t_text_proc:.4f}s")
        print(f"GPT Encoder:          {t_gpt_enc:.4f}s")
        print(f"GPT Decoding:         {t_gpt_dec:.4f}s ({steps} steps, {t_step_avg:.5f}s/step)")
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
    parser.add_argument("--output", default="output_onnx.wav")
    parser.add_argument("--ref_lang", default="zh")
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--bert_path", default="pretrained_models/chinese-roberta-wwm-ext-large")
    args = parser.parse_args()

    GPTSoVITS_ONNX_Inference(
        args.onnx_dir, args.bert_path, args.sovits_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).infer(
        args.ref_audio, args.ref_text, args.ref_lang, args.text, args.lang, output_path=args.output
    )
