import onnxruntime
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

from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from GPT_SoVITS.module.mel_processing import spectrogram_torch
from GPT_SoVITS.sv import SV

# Global Options
so = onnxruntime.SessionOptions()
so.log_severity_level = 1


def sample_logits(logits, top_k=5, top_p=1.0, temperature=1.0):
    # 确保在 float32 下进行概率计算，防止 FP16 溢出
    logits = logits.astype(np.float32)

    if temperature != 1.0:
        logits = logits / temperature

    # 数值稳定性处理：减去最大值
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
        # 最后的防线：如果概率分布计算出 NaN，回退到均匀分布或 argmax
        if np.isnan(p).any():
            # print("Warning: NaN in probabilities, using uniform distribution.")
            p = np.ones_like(p) / len(p)

        # 归一化以防万一
        p = p / p.sum()
        sample = np.random.choice(probs.shape[1], p=p)
        samples.append(sample)
    return np.array(samples, dtype=np.int64)[:, None]


class GPTSoVITS_ONNX_Inference:
    def __init__(self, onnx_dir, bert_path, sovits_path, device="cpu"):
        self.onnx_dir = onnx_dir
        self.device = device
        # 显式指定 Provider，避免不必要的警告
        if device == "cuda":
            self.providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        else:
            self.providers = ["CPUExecutionProvider"]

        print(f"Loading ONNX models from {onnx_dir} on {device}...")

        # 加载模型
        self.sess_ssl = onnxruntime.InferenceSession(f"{onnx_dir}/ssl.onnx", providers=self.providers)
        self.sess_bert = onnxruntime.InferenceSession(f"{onnx_dir}/bert.onnx", providers=self.providers)
        self.sess_vq = onnxruntime.InferenceSession(f"{onnx_dir}/vq_encoder.onnx", providers=self.providers)
        self.sess_gpt_enc = onnxruntime.InferenceSession(f"{onnx_dir}/gpt_encoder.onnx", providers=self.providers)
        self.sess_gpt_step = onnxruntime.InferenceSession(f"{onnx_dir}/gpt_step.onnx", providers=self.providers)
        self.sess_sovits = onnxruntime.InferenceSession(f"{onnx_dir}/sovits.onnx", providers=self.providers)

        # 缓存 GPT Step 的输入元数据，用于类型检查
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
        self.warmup()

    def warmup(self):
        """预热模型，消除首次推理的 JIT 编译延迟"""
        print("Warming up...")
        audio = np.zeros((1, 32000), dtype=np.float32)
        # 仅预热 SSL 即可触发大部分库加载
        try:
            _ = self.run_sess(self.sess_ssl, {"audio": audio})
        except Exception as e:
            print(f"Warmup warning: {e}")
        print("Warmup complete.")

    def run_sess(self, sess, inputs):
        """
        通用的 Session 运行包装器
        自动处理 Numpy 类型对齐 (FP32 <-> FP16)
        """
        input_meta = sess.get_inputs()
        actual_inputs = {}
        for i in input_meta:
            if i.name in inputs:
                val = inputs[i.name]
                if isinstance(val, np.ndarray):
                    # 如果模型要 FP16 但给了 FP32 -> 转 FP16
                    if i.type == 'tensor(float16)' and val.dtype != np.float16:
                        val = val.astype(np.float16)
                    # 如果模型要 FP32 但给了 FP16 -> 转 FP32 (兼容性)
                    elif i.type == 'tensor(float)' and val.dtype != np.float32:
                        val = val.astype(np.float32)
                    elif i.type == 'tensor(int64)' and val.dtype != np.int64:
                        val = val.astype(np.int64)
                actual_inputs[i.name] = val

        # 运行推理 (Standard Mode: CPU <-> GPU Copy)
        outputs = sess.run(None, actual_inputs)

        # 将 FP16 输出转回 FP32，方便 Python 处理和防止后续溢出
        processed_outputs = []
        for out in outputs:
            if isinstance(out, np.ndarray) and out.dtype == np.float16:
                processed_outputs.append(out.astype(np.float32))
            else:
                processed_outputs.append(out)
        return processed_outputs

    def get_bert_feature(self, text, word2ph, language):
        if language != "zh":
            return None

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

    def infer(self, ref_wav_path, prompt_text, prompt_lang, text, text_lang,
              top_k=5, temperature=1.0, output_path="out.wav"):

        t_total_start = time.perf_counter()

        # 1. Audio Processing
        wav16k, _ = librosa.load(ref_wav_path, sr=16000)
        zero_wav = np.zeros(int(16000 * 0.3), dtype=np.float32)
        wav16k_padded = np.concatenate([wav16k, zero_wav])[None, :]

        ssl_content = self.run_sess(self.sess_ssl, {"audio": wav16k_padded})[0]
        ssl_content = ssl_content.transpose(0, 2, 1)

        codes = self.run_sess(self.sess_vq, {"ssl_content": ssl_content})[0]
        prompt_semantic = codes[0, 0][None, :]

        # 2. Text Processing
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

        # 3. GPT Encoder
        print("Running GPT Encoder...")
        # gpt_encoder 通常只有一次，FP16 模式下 sess.run 足够快
        logits, k_cache, v_cache, x_len, y_len = self.run_sess(self.sess_gpt_enc, {
            "phoneme_ids": all_phoneme_ids,
            "phoneme_ids_len": all_phoneme_len,
            "prompts": prompt_semantic.astype(np.int64),
            "bert_feature": bert
        })

        current_samples = sample_logits(logits, top_k=top_k, temperature=temperature)
        decoded_semantic_list = [prompt_semantic, current_samples]

        # 4. GPT Step (Autoregressive Loop)
        # [决策] 使用 Standard sess.run 循环
        # 原因：IOBinding 在纯 FP16 循环下极易因累积误差导致 NaN，而 sess.run 的 CPU 往返隐式清洗了数据，保证了稳定性。
        # 且对于小 Batch (1) 和小 Cache，PCIe 带宽不是瓶颈。
        print(f"Running GPT Step (Standard Mode)...")
        t_dec_start = time.perf_counter()

        steps = 0
        max_steps = 1500

        for i in range(max_steps):
            inputs = {
                "samples": current_samples,
                "k_cache": k_cache,
                "v_cache": v_cache,
                "idx": np.array([i], dtype=np.int64)
            }
            # 兼容不同导出版本的输入名
            if "x_len" in self.step_inputs_info: inputs["x_len"] = x_len
            if "y_len" in self.step_inputs_info: inputs["y_len"] = y_len

            # Run (这里会自动处理 FP16<->FP32 转换)
            logits, k_cache, v_cache = self.run_sess(self.sess_gpt_step, inputs)

            current_samples = sample_logits(logits, top_k=top_k, temperature=temperature)
            decoded_semantic_list.append(current_samples)
            steps += 1
            if current_samples[0, 0] == 1024:
                break

        t_dec_end = time.perf_counter()

        pred_semantic = np.concatenate(decoded_semantic_list, axis=1)
        generated_sem = pred_semantic[:, prompt_semantic.shape[1]:]
        if generated_sem[0, -1] == 1024: generated_sem = generated_sem[:, :-1]
        generated_sem = generated_sem[:, None, :]

        # 5. SoVITS Decode
        print("Running SoVITS Decoder...")

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

        # 简单的 Embedding Padding 逻辑
        sv_size = 20480 if "Pro" in self.version else 512
        if sv_emb.shape[-1] != sv_size:
            tmp = np.zeros((1, sv_size), dtype=np.float32)
            tmp[:, :min(sv_emb.shape[-1], sv_size)] = sv_emb[:, :min(sv_emb.shape[-1], sv_size)]
            sv_emb = tmp

        audio = self.run_sess(self.sess_sovits, {
            "pred_semantic": generated_sem.astype(np.int64),
            "text_seq": np.array(phones2, dtype=np.int64)[None, :],
            "refer_spec": spec.astype(np.float32),
            "sv_emb": sv_emb.astype(np.float32),
        })[0]

        sf.write(output_path, audio.squeeze(), hps.data.sampling_rate)
        print(
            f"Done. GPT Steps: {steps}, Time: {t_dec_end - t_dec_start:.3f}s, Total: {time.perf_counter() - t_total_start:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", default="onnx_export/firefly_v2_proplus_fp16")  # 你的导出目录
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