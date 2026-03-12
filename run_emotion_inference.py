"""
Multi-Emotion TTS Inference for GPT-SoVITS v2/v2Pro

核心思路:
- ge (global emotion) 是从参考音频频谱提取的风格向量 [B, 512, 1]
- 预计算每种情感音频的 ge 向量并缓存
- 解析 <emotion>text</emotion> 标签，按 span 拆分
- 每个片段用对应情感的 ge 解码
- 相邻情感之间在 phoneme 级别做 crossfade 插值，实现平滑过渡
- sv_emb 始终使用主音色(neutral)，保证音色一致性，只让 ge 控制情感
"""

import os
import re
import sys
import time
import json
import torch
import numpy as np
import librosa
import argparse
import torchaudio
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ONNX runtime for emotion predictor
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "GPT_SoVITS"))

from GPT_SoVITS.module.models import SynthesizerTrn
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from GPT_SoVITS.feature_extractor import cnhubert
from transformers import AutoModelForMaskedLM, AutoTokenizer
from GPT_SoVITS.text.LangSegmenter import LangSegmenter
from GPT_SoVITS.module.mel_processing import spectrogram_torch
from GPT_SoVITS.process_ckpt import load_sovits_new, get_sovits_version_from_path_fast
from GPT_SoVITS.sv import SV
from GPT_SoVITS.utils import load_audio_equivalent
from GPT_SoVITS.module import commons

device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = True if device == "cuda" else False


# ─── Text Parsing ───────────────────────────────────────────────────────────

def has_emotion_tags(text):
    """
    检查文本是否包含情感标签。

    情感标签格式: <emotion>text</emotion>
    例如: <happy>你好</happy><sad>再见</sad>
    """
    tag_pattern = re.compile(r"<([a-zA-Z_][\w-]*)>.*?</\1>", re.DOTALL)
    return bool(tag_pattern.search(text))


def tagged_text_to_json(s: str):
    """
    解析带情感标签的文本，例如:
    "<doubt>哎呀，你说我看完</doubt><sad>我就记得它里面那个歌。</sad>"
    
    特殊标签 <|> 表示情感重置点（情绪断点），清除前面的情感残留。
    例如: "<angry>你怎么回事！</angry><|><neutral>好吧算了。</neutral>"
    表示说话人在两句之间平复了情绪，后面的 neutral 不受 angry 残留影响。
    
    返回:
    {
        "text": "哎呀，你说我看完我就记得它里面那个歌。",
        "spans": [
            {"start": 0, "end": 7, "label": "doubt"},
            {"start": 7, "end": 17, "label": "sad"}
        ],
        "resets": [7]  # 情感重置点的字符偏移列表
    }
    """
    # 先提取所有 <|> 的位置，然后移除它们再解析标签
    # 分两步：先找 reset 标记，再解析情感标签
    
    # Step 1: 找到 <|> 并记录它们在"去标签后纯文本"中的位置
    # 先把 <|> 替换成占位符，解析完再处理
    RESET_PLACEHOLDER = "\x00RESET\x00"
    s_with_placeholder = s.replace("<|>", RESET_PLACEHOLDER)
    
    tag_pattern = re.compile(r"<([a-zA-Z_][\w-]*)>(.*?)</\1>", re.DOTALL)
    text_parts = []
    spans = []
    resets = []
    out_pos = 0
    last_end = 0

    for m in tag_pattern.finditer(s_with_placeholder):
        # 标签前的普通文本 → 默认 neutral
        plain = s_with_placeholder[last_end:m.start()]
        if plain:
            # 处理 plain 中可能包含的 reset 占位符
            sub_parts = plain.split(RESET_PLACEHOLDER)
            for k, sub in enumerate(sub_parts):
                if k > 0:
                    resets.append(out_pos)
                if sub:
                    spans.append({"start": out_pos, "end": out_pos + len(sub), "label": "neutral"})
                    text_parts.append(sub)
                    out_pos += len(sub)

        label = m.group(1)
        content = m.group(2)
        if content:
            # content 内部也可能有 reset 占位符
            sub_parts = content.split(RESET_PLACEHOLDER)
            content_start = out_pos
            for k, sub in enumerate(sub_parts):
                if k > 0:
                    resets.append(out_pos)
                if sub:
                    text_parts.append(sub)
                    out_pos += len(sub)
            if out_pos > content_start:
                spans.append({"start": content_start, "end": out_pos, "label": label})
        last_end = m.end()

    # 尾部普通文本
    tail = s_with_placeholder[last_end:]
    if tail:
        sub_parts = tail.split(RESET_PLACEHOLDER)
        for k, sub in enumerate(sub_parts):
            if k > 0:
                resets.append(out_pos)
            if sub:
                spans.append({"start": out_pos, "end": out_pos + len(sub), "label": "neutral"})
                text_parts.append(sub)
                out_pos += len(sub)

    return {"text": "".join(text_parts), "spans": spans, "resets": resets}



def split_text_keep_structure(text, spans=None):
    """
    按句子分割文本，但保留字符偏移信息。
    如果提供 spans，在情感边界处强制断句，不合并跨情感的短句。
    """
    text = text.strip("\n")
    if not text:
        return []

    # 收集情感边界位置 (情感切换的字符偏移)
    emotion_boundaries = set()
    if spans:
        for span in spans:
            emotion_boundaries.add(span["start"])
            emotion_boundaries.add(span["end"])

    delimiters = r'([。！？.!?…\n])'
    parts = re.split(delimiters, text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentences.append(parts[i] + parts[i + 1])
    if len(parts) % 2 == 1:
        sentences.append(parts[-1])
    sentences = [s.strip() for s in sentences if s.strip()]

    # 合并短句，但不跨情感边界合并
    merged = []
    current = ""
    current_start = 0  # 当前累积段在原文中的起始偏移

    # 计算每个句子在原文中的偏移
    offset = 0
    for s in sentences:
        s_start = text.find(s, offset)
        if s_start == -1:
            s_start = offset
        s_end = s_start + len(s)

        # 检查合并后是否会跨越情感边界
        would_cross_boundary = False
        if emotion_boundaries and current:
            for b in emotion_boundaries:
                if current_start < b <= s_start:
                    would_cross_boundary = True
                    break

        if would_cross_boundary or len(current) + len(s) >= 20:
            if current:
                merged.append(current)
            current = s
            current_start = s_start
        else:
            if not current:
                current_start = s_start
            current += s

        offset = s_end

    if current:
        merged.append(current)
    return merged


def map_chars_to_phones(text, phones, word2ph):
    """
    建立字符→phone 的映射。
    word2ph[i] 表示第 i 个字符对应多少个 phone。
    返回 char_to_phone_range: list of (phone_start, phone_end) for each char.
    """
    ranges = []
    phone_idx = 0
    for i, count in enumerate(word2ph):
        ranges.append((phone_idx, phone_idx + count))
        phone_idx += count
    return ranges


def compute_phone_level_emotions(spans, text, char_to_phone_ranges, total_phones, emotion_labels):
    """
    将字符级别的情感标注映射到 phone 级别。
    返回 phone_emotions: list of str, 每个 phone 对应的情感标签。
    """
    # 先给每个字符分配情感
    char_emotions = ["neutral"] * len(text)
    for span in spans:
        for ci in range(span["start"], min(span["end"], len(text))):
            char_emotions[ci] = span["label"]

    # 映射到 phone 级别
    phone_emotions = ["neutral"] * total_phones
    for ci, (ps, pe) in enumerate(char_to_phone_ranges):
        if ci < len(char_emotions):
            for pi in range(ps, pe):
                if pi < total_phones:
                    phone_emotions[pi] = char_emotions[ci]
    return phone_emotions


# ─── DictToAttrRecursive (same as run_inference.py) ─────────────────────────

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


# ─── Emotion Predictor (ONNX) ─────────────────────────────────────────────────

EMOTION_PREDICTOR_MAX_LEN = 128
EMOTION_PREDICTOR_NUM_FUSE_LAYERS = 4  # BERT 最后几层 hidden states 作为输入


class EmotionPredictor:
    """
    基于 BERT + ONNX 的情感预测器。
    自动对文本进行情感标注，返回 spans 格式。
    """

    def __init__(self, bert_path, onnx_dir=None, device="cpu"):
        """
        Args:
            bert_path: BERT 模型路径 (chinese-roberta-wwm-ext-large)
            onnx_dir: emotion_head.onnx 所在目录，默认为 GPT_SoVITS/emotion_predictor/
            device: "cpu" 或 "cuda"
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime 未安装，请运行: pip install onnxruntime")

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32

        # 加载 BERT (用于情感预测的独立实例)
        from transformers import AutoModel, AutoTokenizer
        print(f"[EmotionPredictor] Loading BERT from {bert_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModel.from_pretrained(bert_path)
        if device == "cuda":
            self.bert_model = self.bert_model.half()
        self.bert_model = self.bert_model.to(device)
        self.bert_model.eval()

        # 加载 ONNX emotion head
        if onnx_dir is None:
            onnx_dir = Path(__file__).parent / "GPT_SoVITS" / "emotion_predictor"
        onnx_dir = Path(onnx_dir)
        onnx_path = onnx_dir / "emotion_head.onnx"
        meta_path = onnx_dir / "meta.json"

        if not onnx_path.exists():
            raise FileNotFoundError(f"未找到 {onnx_path}，请确保 emotion_head.onnx 存在")
        if not meta_path.exists():
            raise FileNotFoundError(f"未找到 {meta_path}，请确保 meta.json 存在")

        # 加载 id2label 映射
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.id2label = meta["id2label"]

        # 创建 ONNX session
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        self.ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
        print(f"[EmotionPredictor] Loaded emotion_head.onnx from {onnx_path}")

    def _bert_encode(self, input_ids, attention_mask, token_type_ids):
        """用 transformers 跑 BERT，返回 hidden_states numpy array。"""
        with torch.no_grad():
            out = self.bert_model(
                input_ids=torch.from_numpy(input_ids).to(self.device),
                attention_mask=torch.from_numpy(attention_mask).to(self.device),
                token_type_ids=torch.from_numpy(token_type_ids).to(self.device),
                output_hidden_states=True,
            )
        # 取最后 N 层，stack 为 (B, N, L, H)
        layers = [h.cpu().numpy() for h in out.hidden_states[-EMOTION_PREDICTOR_NUM_FUSE_LAYERS:]]
        return np.stack(layers, axis=1)

    def _decode_spans(self, tag_ids, offset_mapping, text):
        """将 tag_ids 解码为字符级 span 列表。"""
        # 提取每个 token 的 emotion（去掉 BIOES 前缀，O 保持 O）
        emotion_seq = []
        for i, tid in enumerate(tag_ids):
            om = offset_mapping[i]
            if om[0] == 0 and om[1] == 0:
                emotion_seq.append("O")
                continue
            label = self.id2label.get(str(tid), "O")
            if label == "O" or "-" not in label:
                emotion_seq.append("O")
            else:
                emotion_seq.append(label.split("-", 1)[1])

        # 连续相同 emotion 合并为 span
        spans, i = [], 0
        while i < len(emotion_seq):
            lbl = emotion_seq[i]
            if lbl == "O":
                i += 1
                continue
            j = i + 1
            while j < len(emotion_seq) and emotion_seq[j] == lbl:
                j += 1
            # 转为字符偏移
            valid = [offset_mapping[k] for k in range(i, j)
                     if not (offset_mapping[k][0] == 0 and offset_mapping[k][1] == 0)]
            if valid:
                cs, ce = valid[0][0], valid[-1][1]
                if cs < ce:
                    spans.append({"start": cs, "end": ce, "label": lbl, "text": text[cs:ce]})
            i = j
        return spans

    def _fill_neutral(self, text, spans):
        """用 neutral 填充未被标注的区域。"""
        filled, cursor = [], 0
        for sp in sorted(spans, key=lambda s: s["start"]):
            if sp["start"] > cursor:
                filled.append({"start": cursor, "end": sp["start"],
                               "label": "neutral", "text": text[cursor:sp["start"]]})
            filled.append(sp)
            cursor = sp["end"]
        if cursor < len(text):
            filled.append({"start": cursor, "end": len(text),
                           "label": "neutral", "text": text[cursor:]})
        return filled

    def predict(self, text, fill_neutral=True):
        """
        预测文本的情感 spans。

        Args:
            text: 输入文本
            fill_neutral: 是否用 neutral 填充未标注区域

        Returns:
            {"text": text, "spans": [{"start": int, "end": int, "label": str, "text": str}, ...]}
        """
        # Tokenize
        enc = self.tokenizer(
            text,
            max_length=EMOTION_PREDICTOR_MAX_LEN,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="np",
        )
        offset_mapping = enc.pop("offset_mapping")[0].tolist()
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)
        token_type_ids = enc.get("token_type_ids",
                                 np.zeros_like(input_ids)).astype(np.int64)

        seq_len = input_ids.shape[1]

        # BERT 编码
        hidden_states = self._bert_encode(input_ids, attention_mask, token_type_ids)

        # 零填充到 MAX_LEN
        pad_len = EMOTION_PREDICTOR_MAX_LEN - seq_len
        hidden_states = np.pad(hidden_states, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        mask_padded = np.pad(attention_mask, ((0, 0), (0, pad_len)))

        # ONNX 推理
        tag_ids = self.ort_session.run(
            ["tag_ids"],
            {"hidden_states": hidden_states.astype(np.float32),
             "attention_mask": mask_padded.astype(np.int64)},
        )[0][0].tolist()

        # 只取有效长度部分
        tag_ids = tag_ids[:seq_len]
        offset_mapping = offset_mapping[:seq_len]

        # 解码为 spans
        spans = self._decode_spans(tag_ids, offset_mapping, text)
        if fill_neutral:
            spans = self._fill_neutral(text, spans)

        return {"text": text, "spans": spans}


# ─── Main Inference Class ───────────────────────────────────────────────────

class EmotionTTSInference:
    """
    多情感 TTS 推理引擎。
    
    架构要点:
    - ge (global emotion): 从 ref_enc(spectrogram) 提取的风格向量 [B, gin_channels, 1]
      它同时编码了音色和情感信息
    - sv_emb: 从 SV 模型提取的说话人验证嵌入，在 v2Pro 中用于增强音色一致性
    - 在 decode 时，ge 被注入到 enc_p(通过MRTE)、flow、dec 三个模块
    
    多情感策略:
    1. 预计算每种情感参考音频的 ge_emotion 和主音色的 ge_neutral
    2. sv_emb 始终使用主音色音频，保证音色底色一致
    3. 对于 v2Pro: ge = ref_enc(spec) + sv_emb，我们分离情感和音色:
       - ge_base = sv_emb 部分 (音色)
       - ge_style = ref_enc 部分 (情感+音色混合)
       我们用不同情感的 ge_style 替换，但保持 sv_emb 不变
    4. 在 phoneme 级别对相邻情感 span 的边界做 crossfade 插值
    """

    def __init__(self, gpt_path, sovits_path, cnhubert_base_path, bert_path,
                 emotion_predictor_onnx_dir=None):
        self.device = device
        self.is_half = is_half
        self.dtype = torch.float16 if is_half else torch.float32

        print(f"[EmotionTTS] Loading models on {device}...")

        # Load CNHubert
        cnhubert.cnhubert_base_path = cnhubert_base_path
        self.ssl_model = cnhubert.get_model()
        if is_half:
            self.ssl_model = self.ssl_model.half()
        self.ssl_model = self.ssl_model.to(device)

        # Load BERT (用于 TTS)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        if is_half:
            self.bert_model = self.bert_model.half()
        self.bert_model = self.bert_model.to(device)

        # Emotion Predictor (可选，用于自动情感标注)
        self.emotion_predictor = None
        self.emotion_predictor_enabled = False
        if emotion_predictor_onnx_dir is not None or ONNX_AVAILABLE:
            try:
                predictor_onnx_dir = emotion_predictor_onnx_dir or (
                    Path(__file__).parent / "GPT_SoVITS" / "emotion_predictor"
                )
                self.emotion_predictor = EmotionPredictor(
                    bert_path=bert_path,
                    onnx_dir=predictor_onnx_dir,
                    device=device
                )
                self.emotion_predictor_enabled = True
                print("[EmotionTTS] Emotion predictor enabled (auto-tagging)")
            except Exception as e:
                print(f"[EmotionTTS] Warning: Failed to load emotion predictor: {e}")
                print("[EmotionTTS] Auto-tagging disabled, manual tags required")

        # Load GPT
        dict_s1 = torch.load(gpt_path, map_location="cpu")
        self.config = dict_s1["config"]
        self.t2s_model = Text2SemanticLightningModule(self.config, "****", is_train=False)
        self.t2s_model.load_state_dict(dict_s1["weight"])
        if is_half:
            self.t2s_model = self.t2s_model.half()
        self.t2s_model = self.t2s_model.to(device)
        self.t2s_model.eval()

        # Load SoVITS
        dict_s2 = load_sovits_new(sovits_path)
        self.hps = DictToAttrRecursive(dict_s2["config"])
        self.hps.model.semantic_frame_rate = "25hz"

        _, model_version, _ = get_sovits_version_from_path_fast(sovits_path)
        if "config" in dict_s2 and "model" in dict_s2["config"] and "version" in dict_s2["config"]["model"]:
            model_version = dict_s2["config"]["model"]["version"]
        elif "sv_emb.weight" in dict_s2["weight"]:
            model_version = "v2Pro"
        self.hps.model.version = model_version
        print(f"[EmotionTTS] SoVITS version: {model_version}")

        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        )
        if is_half:
            self.vq_model = self.vq_model.half()
        self.vq_model = self.vq_model.to(device)
        self.vq_model.eval()
        self.vq_model.load_state_dict(dict_s2["weight"], strict=False)

        self.sv_model = SV(device, is_half)

        # 情感 ge 缓存: {emotion_label: ge_tensor}
        self.emotion_ge_cache = {}
        # 情感 prompt 缓存: {emotion_label: (prompt_semantic, phones, bert)}
        # 每种情感用自己的 prompt_semantic，让 GPT 生成时就带上对应韵律
        self.emotion_prompt_cache = {}
        # 情感语速倍率: {emotion_label: float}
        # 基于情感的自然语速特征，1.0 = 不变
        self.emotion_speed = {
            "neutral": 1.0,
            "happy":   1.05,   # 开心时略快
            "angry":   1.05,   # 愤怒时明显加速
            "sad":     0.95,   # 悲伤时放慢
            "anxious": 1.10,   # 焦虑时加速
            "doubt":   0.95,   # 疑惑时略慢，带犹豫
        }
        # 主音色缓存
        self.primary_sv_emb = None
        self.primary_ge = None
        # 主音色参考信息 (用于 GPT prompt fallback)
        self.primary_ref_spec = None
        self.primary_prompt_semantic = None
        self.primary_phones = None
        self.primary_bert = None

    # ─── ge 提取 ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_ge_from_audio(self, wav_path):
        """
        从音频文件提取 ge 向量 (ref_enc 输出) 和 prompt_semantic。
        返回: ge [1, gin_channels, 1], sv_emb [1, 20480], prompt_semantic [1, T]
        """
        refer_spec, refer_audio = self._get_spepc(wav_path)

        if refer_audio.shape[0] > 1:
            refer_audio = refer_audio[0].unsqueeze(0)

        # ge from ref_enc
        refer_len = torch.tensor(refer_spec.size(2), device=self.device).unsqueeze(0)
        refer_mask = torch.unsqueeze(
            commons.sequence_mask(refer_len, refer_spec.size(2)), 1
        ).to(refer_spec.dtype)

        if self.hps.model.version == "v1":
            ge = self.vq_model.ref_enc(refer_spec * refer_mask, refer_mask)
        else:
            ge = self.vq_model.ref_enc(refer_spec[:, :704] * refer_mask, refer_mask)

        # sv_emb
        if self.hps.data.sampling_rate != 16000:
            audio_16k = torchaudio.transforms.Resample(
                self.hps.data.sampling_rate, 16000
            ).to(self.device)(refer_audio)
        else:
            audio_16k = refer_audio
        sv_emb = self.sv_model.compute_embedding3(audio_16k)

        # prompt_semantic
        with torch.no_grad():
            zero_wav_16k = torch.zeros(int(16000 * 0.3), dtype=self.dtype).to(self.device)
            wav16k, _ = librosa.load(wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k).to(self.device)
            if self.is_half:
                wav16k = wav16k.half()
            wav16k = torch.cat([wav16k, zero_wav_16k])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            codes = self.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0].unsqueeze(0).to(self.device)

        return ge, sv_emb, prompt_semantic

    @torch.no_grad()
    def _compute_full_ge(self, ge_raw, sv_emb):
        """
        对于 v2Pro，将 ref_enc 输出和 sv_emb 合并得到完整 ge。
        对于非 Pro 版本，直接返回 ge_raw。
        """
        if self.vq_model.is_v2pro:
            sv_proj = self.vq_model.sv_emb(sv_emb)  # [B, gin_channels]
            ge = ge_raw + sv_proj.unsqueeze(-1)
            ge = self.vq_model.prelu(ge)
            return ge
        return ge_raw

    def _get_spepc(self, filename):
        audio, sr = load_audio_equivalent(filename, self.device)
        if sr != self.hps.data.sampling_rate:
            audio = torchaudio.transforms.Resample(sr, self.hps.data.sampling_rate).to(self.device)(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        spec = spectrogram_torch(
            audio, self.hps.data.filter_length, self.hps.data.sampling_rate,
            self.hps.data.hop_length, self.hps.data.win_length, center=False
        )
        if self.is_half:
            spec = spec.half()
        return spec, audio

    # ─── 预设情感加载 ────────────────────────────────────────────────────────

    def load_emotion_presets(self, emotion_audio_map, primary_ref_audio, primary_ref_text, primary_ref_lang):
        """
        加载情感预设 (从音频文件提取 ge + prompt_semantic)。
        
        Args:
            emotion_audio_map: dict, {emotion_label: wav_path}
                或 {emotion_label: (wav_path, lab_text)}
            primary_ref_audio: str, 主音色参考音频路径 (neutral)
            primary_ref_text: str, 主音色参考文本
            primary_ref_lang: str, 主音色参考语言
        """
        print(f"[EmotionTTS] Loading {len(emotion_audio_map)} emotion presets...")

        # 1. 提取主音色的 sv_emb (所有情感共用)
        primary_ge_raw, self.primary_sv_emb, primary_prompt = self._extract_ge_from_audio(primary_ref_audio)
        self.primary_ge = self._compute_full_ge(primary_ge_raw, self.primary_sv_emb)

        self._load_primary_prompt(primary_ref_audio, primary_ref_text, primary_ref_lang)

        # 3. 缓存 neutral 的 ge 和 prompt
        self.emotion_ge_cache["neutral"] = self.primary_ge.clone()
        self.emotion_prompt_cache["neutral"] = (
            self.primary_prompt_semantic,
            self.primary_phones,
            self.primary_bert,
        )

        # 4. 提取每种情感的 ge + prompt_semantic
        for label, value in emotion_audio_map.items():
            if label == "neutral":
                continue  # 已经有了

            # 支持两种格式: wav_path 或 (wav_path, lab_text)
            if isinstance(value, (list, tuple)):
                wav_path, lab_text = value[0], value[1]
            else:
                wav_path, lab_text = value, ""

            ge_raw, _, prompt_semantic = self._extract_ge_from_audio(wav_path)
            # 关键: 用主音色的 sv_emb 而不是情感音频的 sv_emb
            # 这样保持音色一致，只改变情感色彩
            ge_full = self._compute_full_ge(ge_raw, self.primary_sv_emb)
            self.emotion_ge_cache[label] = ge_full

            # 提取该情感的 phones/bert (需要 lab 文本)
            if lab_text:
                phones, bert, _, _ = self.get_phones_and_bert(
                    lab_text, primary_ref_lang, self.hps.model.version
                )
                self.emotion_prompt_cache[label] = (prompt_semantic, phones, bert)
                print(f"  [✓] {label}: {wav_path} (prompt+ge)")
            else:
                # 没有 lab 文本，fallback 到主音色 prompt
                self.emotion_prompt_cache[label] = (
                    self.primary_prompt_semantic,
                    self.primary_phones,
                    self.primary_bert,
                )
                print(f"  [✓] {label}: {wav_path} (ge only, no lab → fallback prompt)")

        print(f"[EmotionTTS] Loaded emotions: {list(self.emotion_ge_cache.keys())}")
        prompts_with_own = [l for l, v in self.emotion_prompt_cache.items()
                           if v[0] is not self.primary_prompt_semantic or l == "neutral"]
        print(f"[EmotionTTS] Emotions with own prompt_semantic: {prompts_with_own}")

    def load_emotion_presets_from_cache(self, ge_cache_path, primary_ref_audio, primary_ref_text, primary_ref_lang):
        """
        从预计算的 ge 缓存文件加载情感预设。
        由 tools/build_emotion_presets.py 生成。
        
        缓存格式支持两种:
        - 旧格式: {emotion: ge_tensor}
        - 新格式: {emotion: {"ge": ge_tensor, "prompt_semantic": tensor, "phones": list, "bert": tensor}}
        
        Args:
            ge_cache_path: str, .pt 文件路径
            primary_ref_audio: str, 主音色参考音频路径 (用于 GPT prompt fallback)
            primary_ref_text: str, 主音色参考文本
            primary_ref_lang: str, 主音色参考语言
        """
        ge_cache = torch.load(ge_cache_path, map_location=self.device)

        # 提取 sv_emb (仍需要主音色音频)
        _, self.primary_sv_emb, _ = self._extract_ge_from_audio(primary_ref_audio)

        self._load_primary_prompt(primary_ref_audio, primary_ref_text, primary_ref_lang)

        for label, data in ge_cache.items():
            if isinstance(data, dict):
                # 新格式: 包含 ge + prompt
                self.emotion_ge_cache[label] = data["ge"].to(self.device).to(self.dtype)
                if "prompt_semantic" in data and "phones" in data and "bert" in data:
                    self.emotion_prompt_cache[label] = (
                        data["prompt_semantic"].to(self.device).to(self.dtype),
                        data["phones"],
                        data["bert"].to(self.device).to(self.dtype),
                    )
                else:
                    self.emotion_prompt_cache[label] = (
                        self.primary_prompt_semantic,
                        self.primary_phones,
                        self.primary_bert,
                    )
            else:
                # 旧格式: 只有 ge tensor
                self.emotion_ge_cache[label] = data.to(self.device).to(self.dtype)
                self.emotion_prompt_cache[label] = (
                    self.primary_prompt_semantic,
                    self.primary_phones,
                    self.primary_bert,
                )

        if "neutral" in self.emotion_ge_cache:
            self.primary_ge = self.emotion_ge_cache["neutral"]
        else:
            raise ValueError("ge cache 中必须包含 'neutral'")

        print(f"[EmotionTTS] Loaded {len(self.emotion_ge_cache)} emotions from cache: "
              f"{list(self.emotion_ge_cache.keys())}")

    def _load_primary_prompt(self, primary_ref_audio, primary_ref_text, primary_ref_lang):
        """提取主音色的 GPT prompt 和 phones/bert (内部方法)"""
        with torch.no_grad():
            zero_wav_16k = torch.zeros(int(16000 * 0.3), dtype=self.dtype).to(self.device)
            wav16k, _ = librosa.load(primary_ref_audio, sr=16000)
            wav16k = torch.from_numpy(wav16k).to(self.device)
            if self.is_half:
                wav16k = wav16k.half()
            wav16k = torch.cat([wav16k, zero_wav_16k])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            codes = self.vq_model.extract_latent(ssl_content)
            self.primary_prompt_semantic = codes[0, 0].unsqueeze(0).to(self.device)

        self.primary_ref_spec, _ = self._get_spepc(primary_ref_audio)
        self.primary_phones, self.primary_bert, _, _ = self.get_phones_and_bert(
            primary_ref_text, primary_ref_lang, self.hps.model.version
        )

    def _spans_to_tagged_text(self, text, spans):
        """
        将 spans 转换为带标签的文本格式。

        例如:
            text = "你好世界"
            spans = [{"start": 0, "end": 2, "label": "happy"},
                     {"start": 2, "end": 4, "label": "sad"}]
            返回 = "<happy>你好</happy><sad>世界</sad>"
        """
        if not spans:
            return f"<neutral>{text}</neutral>"

        result = []
        for span in sorted(spans, key=lambda s: s["start"]):
            label = span["label"]
            content = text[span["start"]:span["end"]]
            result.append(f"<{label}>{content}</{label}>")
        return "".join(result)


    # ─── BERT / Phones (复用 run_inference.py 逻辑) ──────────────────────────

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language = language.replace("all_", "")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
        else:
            bert = torch.zeros(
                (1024, len(phones)), dtype=self.dtype
            ).to(self.device)
        return bert

    def get_phones_and_bert(self, text, language, version, default_lang=None):
        text = re.sub(r' {2,}', ' ', text)
        textlist, langlist = [], []

        if language in ("all_zh", "zh"):
            if language == "zh":
                langlist.append("zh")
                textlist.append(text)
            else:
                for tmp in LangSegmenter.getTexts(text, "zh"):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
        elif language == "en":
            langlist.append("en")
            textlist.append(text)
        elif language == "auto":
            for tmp in LangSegmenter.getTexts(text, default_lang=default_lang):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegmenter.getTexts(text):
                if langlist:
                    if (tmp["lang"] == "en" and langlist[-1] == "en") or \
                       (tmp["lang"] != "en" and langlist[-1] != "en"):
                        textlist[-1] += tmp["text"]
                        continue
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    langlist.append(language.replace("all_", ""))
                textlist.append(tmp["text"])

        phones_list, bert_list, norm_text_list = [], [], []
        word2ph_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text(textlist[i], lang, version)
            phones = cleaned_text_to_sequence(phones, version)
            bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
            word2ph_list.extend(word2ph)

        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)
        return phones, bert.to(self.dtype), norm_text, word2ph_list

    # ─── 多情感 ge 插值 ───────────────────────────────────────────────

    def _build_phone_ge_sequence(self, phone_emotions, transition_phones=4):
        """
        根据每个 phone 的情感标签，构建 phone 级别的 ge 序列。
        使用情感惯性模型: 当前标注区为主导，前面的情感作为衰减残留自然混入。
        
        Args:
            phone_emotions: list[str], 每个 phone 对应的情感标签
            transition_phones: int, 情感切换后残留衰减的 phone 数量
        
        Returns:
            ge_sequence: Tensor [1, gin_channels, num_phones] 或 [1, gin_channels, 1]
        """
        n_phones = len(phone_emotions)
        if n_phones == 0:
            return self.primary_ge

        gin_channels = self.primary_ge.shape[1]

        # 检查是否只有一种情感
        unique_emotions = set(phone_emotions)
        if len(unique_emotions) == 1:
            label = phone_emotions[0]
            ge = self.emotion_ge_cache.get(label, self.primary_ge)
            return ge

        # 构建每个 phone 的 ge，带情感惯性
        ge_per_phone = torch.zeros(1, gin_channels, n_phones, dtype=self.dtype, device=self.device)

        prev_label = None
        carry_ge = None
        carry_strength = 0.0
        CARRY_DECAY_PER_PHONE = 0.85  # 每个 phone 残留衰减

        for i in range(n_phones):
            label = phone_emotions[i]
            ge_current = self.emotion_ge_cache.get(label, self.primary_ge)[:, :, 0]

            if label != prev_label and prev_label is not None:
                # 情感切换: 前一个情感成为新的残留
                if carry_ge is not None:
                    carry_ge = 0.5 * carry_ge + 0.5 * self.emotion_ge_cache.get(
                        prev_label, self.primary_ge)[:, :, 0]
                else:
                    carry_ge = self.emotion_ge_cache.get(prev_label, self.primary_ge)[:, :, 0].clone()
                carry_strength = 0.15  # 切换瞬间的残留权重

            if carry_ge is not None and carry_strength > 0.005:
                ge_per_phone[:, :, i] = (1 - carry_strength) * ge_current + carry_strength * carry_ge
                carry_strength *= CARRY_DECAY_PER_PHONE
            else:
                ge_per_phone[:, :, i] = ge_current

            prev_label = label

        return ge_per_phone


    # ─── 自定义 decode: 支持 phone 级别 ge ────────────────────────────────────

    @torch.no_grad()
    def _decode_with_phonelevel_ge(self, codes, text_phones, ge_per_phone,
                                   speed=1, noise_scale=0.5):
        """
        精细版 decode: 支持 phone 级别的 ge 注入。
        
        核心发现 (通过阅读模型源码确认):
        ─────────────────────────────────────────────────────────────
        ge 在模型中有三个注入点，全部通过 Conv1d(kernel=1) + 广播加法:
        
        1. MRTE:  x = cross_attn(...) + ssl_enc + ge
           ge [B,C,1] 广播加到 [B,C,T_ssl]
           
        2. flow (WN.forward):  g = self.cond_layer(g)  # Conv1d(512, N, 1)
           然后 g_l 逐层加到 x_in [B,C,T]
           
        3. dec (Generator.forward):  x = x + self.cond(g)  # Conv1d(512, ch, 1)
           加到 [B,ch,T_audio]
        
        关键: Conv1d(kernel=1) 对每个时间步独立运算，等价于逐帧的线性变换。
        所以如果我们传入 ge [B,C,T] 而不是 [B,C,1]，数学上完全兼容:
        - Conv1d(k=1) 对 [B,C,T] 的每个 t 独立做 W*ge[:,t] + b
        - 加法 x + ge 变成逐帧相加而非广播
        
        唯一需要处理的是时间维度对齐:
        - enc_p 内部: ge 作用于 T_ssl (= codes * 2) 帧
        - flow: 作用于 T_ssl 帧 (经过 speed 调整后)
        - dec: 作用于 T_audio 帧 (经过上采样后，远大于 T_ssl)
        
        策略:
        - phone 级别 ge [B,C,N_phones] → 插值到各阶段需要的时间分辨率
        - 对 flow 和 dec 使用插值后的 ge_temporal
        ─────────────────────────────────────────────────────────────
        
        Args:
            codes: [1, 1, T_codes] semantic codes
            text_phones: [1, N_phones] phone ids
            ge_per_phone: [1, C, N_phones] 每个 phone 的 ge 向量
            speed: float
            noise_scale: float
        """
        import torch.nn.functional as F

        n_phones = text_phones.shape[-1]
        y_lengths = (torch.tensor(codes.shape[2], device=self.device) * 2).reshape(1)
        text_lengths = torch.tensor(n_phones, device=self.device).reshape(1)
        T_ssl = y_lengths.item()  # = codes * 2

        quantized = self.vq_model.quantizer.decode(codes)
        if self.vq_model.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, scale_factor=2.0, mode="nearest")

        # ── Step 1: 将 phone 级 ge 插值到 T_ssl 分辨率 ──
        # ge_per_phone: [1, C, N_phones] → [1, C, T_ssl]
        ge_ssl = F.interpolate(
            ge_per_phone.float(), size=T_ssl, mode="linear", align_corners=False
        ).to(self.dtype)

        # ── Step 2: enc_p (MRTE 用 ge_ssl) ──
        # enc_p 内部 MRTE 做 x = cross_attn(...) + ssl_enc + ge
        # ge [B,C,T_ssl] 直接逐帧加，完美兼容
        ge_for_enc = self.vq_model.ge_to512(ge_ssl.transpose(2, 1)).transpose(2, 1) \
            if self.vq_model.is_v2pro else ge_ssl

        x, m_p, logs_p, y_mask, _, _ = self.vq_model.enc_p(
            quantized, y_lengths, text_phones, text_lengths, ge_for_enc, speed
        )

        # enc_p 输出的实际时间长度 (可能因 speed 调整)
        T_enc = m_p.shape[2]

        # ── Step 3: flow (WN 用 ge_flow) ──
        # flow 内部 WN: g = cond_layer(g) 是 Conv1d(k=1)，逐帧独立
        # 需要 ge 的时间维度 = T_enc
        ge_flow = F.interpolate(
            ge_per_phone.float(), size=T_enc, mode="linear", align_corners=False
        ).to(self.dtype)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.vq_model.flow(z_p, y_mask, g=ge_flow, reverse=True)

        # ── Step 4: dec (Generator 用 ge_dec) ──
        # dec 内部: x = conv_pre(z) + cond(g)，cond 是 Conv1d(k=1)
        # dec 输入 z 的时间维度 = T_enc，但 cond(g) 加在 conv_pre 之后、
        # 上采样之前，所以 ge 需要 T_enc 的时间维度
        # (上采样发生在 cond 之后，所以 ge 不需要匹配音频采样率)
        ge_dec = ge_flow  # 同样的时间分辨率

        o = self.vq_model.dec((z * y_mask)[:, :, :], g=ge_dec)
        return o

    # ─── 主推理入口 ─────────────────────────────────────────────────────────

    def infer(self, tagged_text, text_lang="zh",
              top_k=5, top_p=1, temperature=1, speed=1,
              pause_length=0.3, transition_phones=4,
              segment_level=True, phoneme_level=False,
              auto_predict_emotion=None):
        """
        多情感推理。

        Args:
            tagged_text: str, 带情感标签的文本
                例如 "<doubt>哎呀，你说我看完</doubt><sad>我就记得那个歌。</sad>"
                如果不带标签且 auto_predict_emotion 为 True，则自动预测情感
            text_lang: str, 文本语言
            transition_phones: int, 情感过渡区域的 phone 数 (每侧)
            segment_level: bool, True=按句子级别切换情感(更稳定)
            phoneme_level: bool, True=phone 级别精细 ge 注入(最精细，实验性)
                          当 phoneme_level=True 时，segment_level 参数被忽略
            auto_predict_emotion: bool or None, 是否自动预测情感
                None = 自动检测 (无标签时自动预测)
                True = 强制使用预测器 (忽略已有标签)
                False = 禁用自动预测 (必须有标签)

        Returns:
            audio: np.ndarray, 音频数据
            sr: int, 采样率
        """
        assert self.primary_prompt_semantic is not None, \
            "请先调用 load_emotion_presets() 加载情感预设"

        t_start = time.perf_counter()

        # ── 自动情感预测逻辑 ──
        text_has_tags = has_emotion_tags(tagged_text)

        if auto_predict_emotion is None:
            # 自动检测: 无标签时自动预测
            use_predictor = not text_has_tags and self.emotion_predictor_enabled
        else:
            use_predictor = auto_predict_emotion and self.emotion_predictor_enabled

        if use_predictor and self.emotion_predictor is not None:
            # 使用预测器自动标注情感
            print(f"[EmotionTTS] Auto-predicting emotions for: {tagged_text}")
            predicted = self.emotion_predictor.predict(tagged_text, fill_neutral=True)
            # 将预测结果转换为 tagged_text 格式 (重新构建带标签的文本)
            tagged_text = self._spans_to_tagged_text(predicted["text"], predicted["spans"])
            print(f"[EmotionTTS] Predicted: {tagged_text}")

        # 1. 解析标签
        parsed = tagged_text_to_json(tagged_text)
        full_text = parsed["text"]
        spans = parsed["spans"]
        resets = parsed.get("resets", [])
        print(f"[EmotionTTS] Text: {full_text}")
        print(f"[EmotionTTS] Spans: {spans}")
        if resets:
            print(f"[EmotionTTS] Resets: {resets}")

        if not full_text.strip():
            return np.zeros(0), self.hps.data.sampling_rate

        sr = self.hps.data.sampling_rate
        final_audios = []

        if phoneme_level:
            # ─── phone 级别精细 ge 注入 ─────────────────────────
            # 利用 Conv1d(k=1) 的逐帧独立性，直接传入 [B,C,T] 的 ge
            audio, sr = self._infer_phoneme_level(
                full_text, spans, text_lang, top_k, top_p, temperature,
                speed, pause_length, transition_phones
            )
        elif segment_level:
            # ─── 按句子级别切换情感 ──────────────────────────────
            # 更稳定，每个句子用该句子主导情感的 ge
            audio, sr = self._infer_segment_level(
                full_text, spans, text_lang, top_k, top_p, temperature,
                speed, pause_length, transition_phones, resets=resets
            )
        else:
            # ─── 整段推理，用加权 ge + 情感惯性 ─────────────────
            audio, sr = self._infer_blended(
                full_text, spans, text_lang, top_k, top_p, temperature,
                speed, pause_length, transition_phones, resets=resets
            )

        t_end = time.perf_counter()
        duration = len(audio) / sr if len(audio) > 0 else 0
        print(f"[EmotionTTS] Done: {duration:.2f}s audio in {t_end - t_start:.2f}s")
        return audio, sr

    def _infer_segment_level(self, full_text, spans, text_lang,
                             top_k, top_p, temperature, speed, pause_length,
                             transition_phones, resets=None):
        """
        按句子级别切换情感，带情感惯性模型。
        
        情感惯性: 人说话时情绪有残留，前面的情绪不会瞬间消失，
        而是像余温一样慢慢衰减。当前标注区是主导，前面的情绪只是"还没缓过来"。
        
        模型:
          ge_effective = current_weight * ge_current + carry_weight * ge_carry
          其中 carry 是前面累积的情感残留，每经过一个 segment 衰减一次。
          遇到 <|> 重置标记时，carry 清零（情绪断点，比如停顿、深呼吸）。
        
        Args:
            resets: list[int], 情感重置点的字符偏移列表
        """
        sr = self.hps.data.sampling_rate
        segments = split_text_keep_structure(full_text, spans)
        if not segments:
            return np.zeros(0), sr
        if resets is None:
            resets = []

        # 计算每个 segment 在原文中的字符偏移
        seg_offsets = []
        offset = 0
        for seg in segments:
            idx = full_text.find(seg, offset)
            if idx == -1:
                idx = offset
            seg_offsets.append((idx, idx + len(seg)))
            offset = idx + len(seg)

        # 确定每个 segment 的主导情感
        seg_emotions = []
        for s_start, s_end in seg_offsets:
            emotion_counts = {}
            for span in spans:
                overlap_start = max(s_start, span["start"])
                overlap_end = min(s_end, span["end"])
                if overlap_end > overlap_start:
                    label = span["label"]
                    emotion_counts[label] = emotion_counts.get(label, 0) + (overlap_end - overlap_start)
            if emotion_counts:
                dominant = max(emotion_counts, key=emotion_counts.get)
            else:
                dominant = "neutral"
            seg_emotions.append(dominant)

        # 判断每个 segment 前是否有 reset 点
        seg_has_reset = []
        for i, (s_start, s_end) in enumerate(seg_offsets):
            has_reset = False
            prev_end = seg_offsets[i - 1][1] if i > 0 else 0
            for r in resets:
                if prev_end <= r <= s_start:
                    has_reset = True
                    break
            seg_has_reset.append(has_reset)

        print(f"[EmotionTTS] Segments: {list(zip(segments, seg_emotions))}")
        if resets:
            print(f"[EmotionTTS] Reset points: {resets}")

        # ─── 情感惯性推理 ──────────────────────────────────────────
        # carry_ge: 前面累积的情感残留
        # decay: 每个 segment 后残留衰减的比例 (0.3 = 保留30%的前一段情感)
        CARRY_DECAY = 0.3      # 残留衰减率: 每过一个 segment，前面的残留保留多少
        CARRY_WEIGHT = 0.15    # 残留混入当前情感的最大权重
        # 即: ge = (1 - effective_carry) * ge_current + effective_carry * ge_carry
        # effective_carry = CARRY_WEIGHT * carry_strength
        # carry_strength 从 1.0 开始，每个 segment 乘以 CARRY_DECAY

        carry_ge = None        # 累积的情感残留向量
        carry_strength = 0.0   # 残留强度 (0~1)

        segment_audios = []

        for i, (seg, emotion) in enumerate(zip(segments, seg_emotions)):
            ge_current = self.emotion_ge_cache.get(emotion, self.primary_ge)

            # 遇到 reset → 清除残留
            if seg_has_reset[i]:
                carry_ge = None
                carry_strength = 0.0
                print(f"  [{i+1}/{len(segments)}] ({emotion}) {seg}  "
                      f"[RESET, speed: {speed * self.emotion_speed.get(emotion, 1.0):.2f}x]")
            else:
                print(f"  [{i+1}/{len(segments)}] ({emotion}) {seg}  "
                      f"[carry: {carry_strength:.2f}, speed: {speed * self.emotion_speed.get(emotion, 1.0):.2f}x]")

            # 混合: 当前情感为主，残留为辅
            if carry_ge is not None and carry_strength > 0.01:
                effective_carry = CARRY_WEIGHT * carry_strength
                ge = (1 - effective_carry) * ge_current + effective_carry * carry_ge
            else:
                ge = ge_current

            # 计算该 segment 的实际语速: 基础语速 × 情感语速倍率
            emotion_speed_mult = self.emotion_speed.get(emotion, 1.0)
            seg_speed = speed * emotion_speed_mult

            audio_seg = self._infer_single_segment(
                seg, text_lang, ge, top_k, top_p, temperature, seg_speed,
                emotion=emotion
            )
            segment_audios.append(audio_seg)

            # 更新残留: 当前情感成为新的残留，叠加衰减后的旧残留
            if carry_ge is not None and carry_strength > 0.01:
                # 旧残留继续衰减，加上当前情感作为新残留
                carry_ge = CARRY_DECAY * carry_strength * carry_ge + (1 - CARRY_DECAY * carry_strength) * ge_current
                carry_ge = carry_ge / (carry_ge.norm() + 1e-8) * ge_current.norm()  # 保持模长稳定
                carry_strength = min(1.0, CARRY_DECAY * carry_strength + 1.0)
            else:
                carry_ge = ge_current.clone()
                carry_strength = 1.0

            # 残留衰减
            carry_strength *= CARRY_DECAY

        # 拼接音频，在情感切换处做轻微 crossfade
        final_audios = []
        crossfade_samples = int(sr * 0.03)  # 30ms，仅防止硬切爆音

        for i, audio_seg in enumerate(segment_audios):
            if i > 0 and seg_emotions[i] != seg_emotions[i - 1] and crossfade_samples > 0:
                prev = final_audios[-1] if final_audios else None
                if prev is not None and len(prev) > crossfade_samples and len(audio_seg) > crossfade_samples:
                    fade_len = min(crossfade_samples, len(prev), len(audio_seg))
                    fade_out = np.linspace(1, 0, fade_len)
                    fade_in = np.linspace(0, 1, fade_len)
                    final_audios[-1] = prev.copy()
                    final_audios[-1][-fade_len:] = prev[-fade_len:] * fade_out + audio_seg[:fade_len] * fade_in
                    audio_seg = audio_seg[fade_len:]

            final_audios.append(audio_seg)

            if i < len(segment_audios) - 1 and pause_length > 0:
                final_audios.append(np.zeros(int(sr * pause_length)))

        if not final_audios:
            return np.zeros(0), sr

        audio_final = np.concatenate(final_audios)
        max_amp = np.abs(audio_final).max()
        if max_amp > 1e-5:
            audio_final = audio_final / max_amp * 0.9
        return audio_final, sr

    def _infer_blended(self, full_text, spans, text_lang,
                       top_k, top_p, temperature, speed, pause_length,
                       transition_phones, resets=None):
        """
        整段推理，用字符比例加权的 ge + 跨段情感惯性。
        
        情感惯性模型 (与 segment_level 一致):
          人说话时情绪有残留——从一个情绪跳到另一个时，前面的情绪不会瞬间消失。
          当情绪波动剧烈时 (如 doubt→angry)，愤怒中会带一丝之前的疑惑。
          
          ge_effective = (1 - carry_w) * ge_blended + carry_w * ge_carry
          carry 每经过一个 segment 按 CARRY_DECAY 衰减。
          遇到 <|> 重置标记时 carry 清零。
        """
        sr = self.hps.data.sampling_rate
        segments = split_text_keep_structure(full_text, spans)
        if not segments:
            return np.zeros(0), sr
        if resets is None:
            resets = []

        # ── 计算每个 segment 在原文中的字符偏移 ──
        seg_offsets = []
        offset = 0
        for seg in segments:
            idx = full_text.find(seg, offset)
            if idx == -1:
                idx = offset
            seg_offsets.append((idx, idx + len(seg)))
            offset = idx + len(seg)

        # ── 判断每个 segment 前是否有 reset 点 ──
        seg_has_reset = []
        for i, (s_start, _) in enumerate(seg_offsets):
            has_reset = False
            prev_end = seg_offsets[i - 1][1] if i > 0 else 0
            for r in resets:
                if prev_end <= r <= s_start:
                    has_reset = True
                    break
            seg_has_reset.append(has_reset)

        # ── 情感惯性参数 ──
        CARRY_DECAY = 0.3       # 残留衰减率
        CARRY_WEIGHT_MAX = 0.15 # 残留混入上限

        carry_ge = None
        carry_strength = 0.0

        segment_audios = []
        seg_dominant_emotions = []  # 用于 crossfade 判断

        for i, seg in enumerate(segments):
            seg_start, seg_end = seg_offsets[i]

            # 找到覆盖这个 segment 的 spans
            seg_spans = []
            for span in spans:
                overlap_start = max(seg_start, span["start"])
                overlap_end = min(seg_end, span["end"])
                if overlap_end > overlap_start:
                    seg_spans.append({
                        "start": overlap_start - seg_start,
                        "end": overlap_end - seg_start,
                        "label": span["label"]
                    })

            # 计算情感权重 (按字符比例)
            emotion_weights = {}
            total_chars = len(seg)
            for sp in seg_spans:
                label = sp["label"]
                weight = (sp["end"] - sp["start"]) / max(total_chars, 1)
                emotion_weights[label] = emotion_weights.get(label, 0) + weight

            if not emotion_weights:
                emotion_weights["neutral"] = 1.0

            # 主导情感 (权重最大的)
            dominant = max(emotion_weights, key=emotion_weights.get)
            seg_dominant_emotions.append(dominant)

            # 加权混合 ge
            ge_blended = torch.zeros_like(self.primary_ge)
            for label, weight in emotion_weights.items():
                ge_emo = self.emotion_ge_cache.get(label, self.primary_ge)
                ge_blended += weight * ge_emo

            # ── 情感惯性: 混入前段残留 ──
            if seg_has_reset[i]:
                carry_ge = None
                carry_strength = 0.0

            if carry_ge is not None and carry_strength > 0.01:
                effective_carry = CARRY_WEIGHT_MAX * carry_strength
                ge_final = (1 - effective_carry) * ge_blended + effective_carry * carry_ge
            else:
                ge_final = ge_blended

            carry_info = f"carry: {carry_strength:.2f}" if not seg_has_reset[i] else "RESET"
            print(f"  [{i+1}/{len(segments)}] ({dominant}) {seg} | "
                  f"weights: {emotion_weights} | {carry_info}")

            # 加权语速
            seg_speed_mult = sum(
                self.emotion_speed.get(label, 1.0) * w
                for label, w in emotion_weights.items()
            )
            seg_speed = speed * seg_speed_mult

            audio_seg = self._infer_single_segment(
                seg, text_lang, ge_final, top_k, top_p, temperature, seg_speed,
                emotion=dominant
            )
            segment_audios.append(audio_seg)

            # ── 更新残留 ──
            if carry_ge is not None and carry_strength > 0.01:
                carry_ge = CARRY_DECAY * carry_strength * carry_ge + \
                           (1 - CARRY_DECAY * carry_strength) * ge_blended
                carry_ge = carry_ge / (carry_ge.norm() + 1e-8) * ge_blended.norm()
                carry_strength = min(1.0, CARRY_DECAY * carry_strength + 1.0)
            else:
                carry_ge = ge_blended.clone()
                carry_strength = 1.0
            carry_strength *= CARRY_DECAY

        # ── 拼接音频，情感切换处做 crossfade 防爆音 ──
        final_audios = []
        crossfade_samples = int(sr * 0.03)

        for i, audio_seg in enumerate(segment_audios):
            if (i > 0 and seg_dominant_emotions[i] != seg_dominant_emotions[i - 1]
                    and crossfade_samples > 0):
                prev = final_audios[-1] if final_audios else None
                if (prev is not None and len(prev) > crossfade_samples
                        and len(audio_seg) > crossfade_samples):
                    fade_len = min(crossfade_samples, len(prev), len(audio_seg))
                    fade_out = np.linspace(1, 0, fade_len)
                    fade_in = np.linspace(0, 1, fade_len)
                    final_audios[-1] = prev.copy()
                    final_audios[-1][-fade_len:] = (
                        prev[-fade_len:] * fade_out + audio_seg[:fade_len] * fade_in
                    )
                    audio_seg = audio_seg[fade_len:]

            final_audios.append(audio_seg)

            if i < len(segment_audios) - 1 and pause_length > 0:
                final_audios.append(np.zeros(int(sr * pause_length)))

        if not final_audios:
            return np.zeros(0), sr

        audio_final = np.concatenate(final_audios)
        max_amp = np.abs(audio_final).max()
        if max_amp > 1e-5:
            audio_final = audio_final / max_amp * 0.9
        return audio_final, sr

    def _infer_phoneme_level(self, full_text, spans, text_lang,
                             top_k, top_p, temperature, speed, pause_length,
                             transition_phones):
        """
        Phone 级别精细情感注入。

        模型中 ge 的三个注入点 (MRTE, flow/WN, dec/Generator) 全部使用
        Conv1d(kernel_size=1) 处理 ge，这意味着对每个时间步独立运算。
        原始 ge [B,C,1] 通过广播变成每帧相同的条件。
        如果我们传入 ge [B,C,T]，Conv1d(k=1) 会对每帧独立做 W*ge[:,t]+b，
        加法也变成逐帧相加。数学上完全兼容，不需要改模型代码。
        
        流程:
        1. 对每个 segment，计算 phones 和 word2ph
        2. 通过 word2ph 建立 char→phone 映射
        3. 通过 spans 建立 char→emotion 映射
        4. 组合得到 phone→emotion 映射
        5. 在情感边界处做 crossfade 插值
        6. 构建 ge [B,C,N_phones]，插值到各阶段时间分辨率
        7. 传入自定义 decode
        """
        sr = self.hps.data.sampling_rate
        segments = split_text_keep_structure(full_text, spans)
        if not segments:
            return np.zeros(0), sr

        final_audios = []
        char_offset = 0

        for i, seg in enumerate(segments):
            seg_start = full_text.find(seg, char_offset)
            if seg_start == -1:
                seg_start = char_offset
            seg_end = seg_start + len(seg)
            char_offset = seg_end

            # 找到覆盖这个 segment 的 spans (相对于 segment 的偏移)
            seg_spans = []
            for span in spans:
                overlap_start = max(seg_start, span["start"])
                overlap_end = min(seg_end, span["end"])
                if overlap_end > overlap_start:
                    seg_spans.append({
                        "start": overlap_start - seg_start,
                        "end": overlap_end - seg_start,
                        "label": span["label"]
                    })

            # 获取 phones 和 word2ph
            phones2, bert2, norm_text2, word2ph2 = self.get_phones_and_bert(
                seg, text_lang, self.hps.model.version, default_lang="zh"
            )
            n_phones = len(phones2)

            # 建立 char→phone 映射
            char_to_phone = map_chars_to_phones(norm_text2, phones2, word2ph2)

            # 建立 phone→emotion 映射
            phone_emotions = compute_phone_level_emotions(
                seg_spans, norm_text2, char_to_phone, n_phones,
                list(self.emotion_ge_cache.keys())
            )

            # 构建 phone 级别 ge [1, C, N_phones]
            ge_per_phone = self._build_phone_ge_sequence(phone_emotions, transition_phones)

            # 打印情感分布
            emotion_dist = {}
            for e in phone_emotions:
                emotion_dist[e] = emotion_dist.get(e, 0) + 1

            # 按 phone 数量加权计算语速
            total = max(sum(emotion_dist.values()), 1)
            seg_speed_mult = sum(
                self.emotion_speed.get(emo, 1.0) * cnt / total
                for emo, cnt in emotion_dist.items()
            )
            seg_speed = speed * seg_speed_mult
            print(f"  [{i+1}/{len(segments)}] {seg} | phone emotions: {emotion_dist} | speed: {seg_speed:.2f}x")

            # 确定该 segment 的主导情感 (用于选择 prompt_semantic)
            dominant = max(emotion_dist, key=emotion_dist.get)
            if dominant in self.emotion_prompt_cache:
                prompt_semantic, ref_phones, ref_bert = self.emotion_prompt_cache[dominant]
            else:
                prompt_semantic = self.primary_prompt_semantic
                ref_phones = self.primary_phones
                ref_bert = self.primary_bert

            # GPT inference (用主导情感的 prompt)
            bert = torch.cat([ref_bert, bert2], 1).unsqueeze(0).to(self.device)
            all_phoneme_ids = torch.LongTensor(ref_phones + phones2).to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

            with torch.no_grad():
                pred_semantic, idx = self.t2s_model.model.infer_panel(
                    all_phoneme_ids, all_phoneme_len,
                    prompt_semantic, bert,
                    top_k=top_k, top_p=top_p, temperature=temperature,
                    early_stop_num=50 * 30
                )
                prefix_len = prompt_semantic.shape[1]
                pred_semantic = pred_semantic[:, prefix_len:].unsqueeze(0)

            # SoVITS decode with phone-level ge
            text_tensor = torch.LongTensor(phones2).to(self.device).unsqueeze(0)
            with torch.no_grad():
                audio = self._decode_with_phonelevel_ge(
                    pred_semantic, text_tensor, ge_per_phone,
                    speed=seg_speed, noise_scale=0.5
                )[0][0]

            audio_np = audio.cpu().float().numpy()
            audio_np = audio_np - np.mean(audio_np)
            final_audios.append(audio_np)

            if i < len(segments) - 1 and pause_length > 0:
                final_audios.append(np.zeros(int(sr * pause_length)))

        if not final_audios:
            return np.zeros(0), sr

        audio_final = np.concatenate(final_audios)
        max_amp = np.abs(audio_final).max()
        if max_amp > 1e-5:
            audio_final = audio_final / max_amp * 0.9
        return audio_final, sr

    def _infer_single_segment(self, text, text_lang, ge, top_k, top_p, temperature, speed,
                              emotion=None):
        """
        用指定的 ge 推理单个文本片段。
        GPT 部分使用对应情感的 prompt_semantic（韵律骨架），SoVITS 部分使用传入的 ge。
        
        Args:
            emotion: str or None, 情感标签。如果提供且该情感有独立的 prompt_semantic，
                     则用该情感的 prompt 驱动 GPT，让生成的语义 token 自带对应韵律。
                     如果为 None 或该情感没有独立 prompt，fallback 到主音色 prompt。
        """
        # 选择 prompt: 优先用情感专属的，没有则 fallback
        if emotion and emotion in self.emotion_prompt_cache:
            prompt_semantic, ref_phones, ref_bert = self.emotion_prompt_cache[emotion]
        else:
            prompt_semantic = self.primary_prompt_semantic
            ref_phones = self.primary_phones
            ref_bert = self.primary_bert

        # Text processing
        phones2, bert2, norm_text2, _ = self.get_phones_and_bert(
            text, text_lang, self.hps.model.version, default_lang="zh"
        )

        bert = torch.cat([ref_bert, bert2], 1).unsqueeze(0).to(self.device)
        all_phoneme_ids = torch.LongTensor(ref_phones + phones2).to(self.device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

        # GPT inference (用情感对应的 prompt_semantic)
        with torch.no_grad():
            pred_semantic, idx = self.t2s_model.model.infer_panel(
                all_phoneme_ids, all_phoneme_len,
                prompt_semantic, bert,
                top_k=top_k, top_p=top_p, temperature=temperature,
                early_stop_num=50 * 30
            )
            prefix_len = prompt_semantic.shape[1]
            pred_semantic = pred_semantic[:, prefix_len:].unsqueeze(0)

        # SoVITS decode with emotion-specific ge
        with torch.no_grad():
            # 直接调用底层 decode 逻辑，但用我们的 ge
            y_lengths = (torch.tensor(pred_semantic.shape[2], device=self.device) * 2).reshape(1)
            text_lengths = torch.tensor(len(phones2), device=self.device).reshape(1)
            text_tensor = torch.LongTensor(phones2).to(self.device).unsqueeze(0)

            quantized = self.vq_model.quantizer.decode(pred_semantic)
            if self.vq_model.semantic_frame_rate == "25hz":
                quantized = torch.nn.functional.interpolate(quantized, scale_factor=2.0, mode="nearest")

            ge_for_enc = self.vq_model.ge_to512(ge.transpose(2, 1)).transpose(2, 1) \
                if self.vq_model.is_v2pro else ge

            x, m_p, logs_p, y_mask, _, _ = self.vq_model.enc_p(
                quantized, y_lengths, text_tensor, text_lengths, ge_for_enc, speed
            )
            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * 0.5
            z = self.vq_model.flow(z_p, y_mask, g=ge, reverse=True)
            audio = self.vq_model.dec((z * y_mask)[:, :, :], g=ge)[0][0]

        audio_np = audio.cpu().float().numpy()
        audio_np = audio_np - np.mean(audio_np)
        return audio_np


# ─── CLI ────────────────────────────────────────────────────────────────────

def scan_emotion_dir(emotion_dir):
    """
    扫描情感音频目录，自动发现 *.wav + *.lab 配对。
    文件名即情感标签，例如 happy.wav + happy.lab。
    
    返回:
        emotion_audio_map: {label: (wav_path, lab_text)}
        primary_audio: neutral.wav 路径
        primary_text: neutral.lab 内容
    """
    emotion_audio_map = {}
    primary_audio = None
    primary_text = None

    for f in os.listdir(emotion_dir):
        if not f.endswith(".wav"):
            continue
        label = os.path.splitext(f)[0]
        wav_path = os.path.join(emotion_dir, f)
        lab_path = os.path.join(emotion_dir, f"{label}.lab")

        lab_text = ""
        if os.path.exists(lab_path):
            with open(lab_path, "r", encoding="utf-8") as lf:
                lab_text = lf.read().strip()

        emotion_audio_map[label] = (wav_path, lab_text)

        if label == "neutral":
            primary_audio = wav_path
            primary_text = lab_text

    if not primary_audio:
        raise FileNotFoundError(f"在 {emotion_dir} 中未找到 neutral.wav")
    if not primary_text:
        raise FileNotFoundError(f"在 {emotion_dir} 中未找到 neutral.lab 或内容为空")

    print(f"[scan] 发现 {len(emotion_audio_map)} 种情感: {list(emotion_audio_map.keys())}")
    return emotion_audio_map, primary_audio, primary_text


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Emotion GPT-SoVITS Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 (使用 --emotion_dir 自动扫描):

  python run_emotion_inference.py \\
    --gpt_path pretrained_models/GPT_weights_v2ProPlus/firefly_v2_pp-e25.ckpt \\
    --sovits_path pretrained_models/SoVITS_weights_v2ProPlus/firefly_v2_pp_e10_s590.pth \\
    --emotion_dir pretrained_models/audios/firefly \\
    --text "<doubt>哎呀，你说我看完，我就记得什么啊？</doubt><sad>我就记得它里面那个歌。</sad>" \\
    --output output_emotion.wav

示例 (自动情感预测，无需手动标注):

  python run_emotion_inference.py \\
    --gpt_path pretrained_models/GPT_weights_v2ProPlus/firefly_v2_pp-e25.ckpt \\
    --sovits_path pretrained_models/SoVITS_weights_v2ProPlus/firefly_v2_pp_e10_s590.pth \\
    --emotion_dir pretrained_models/audios/firefly \\
    --text "你终于回来啦！咦？你身后藏着什么？" \\
    --auto_predict \\
    --output output_emotion.wav

目录结构要求:
  emotion_dir/
    neutral.wav + neutral.lab   ← 主音色 (必须)
    happy.wav   + happy.lab     ← 情感预设
    sad.wav     + sad.lab
    angry.wav   + angry.lab
    ...

自动情感预测需要:
  GPT_SoVITS/emotion_predictor/
    emotion_head.onnx  ← 情感预测模型
    meta.json          ← 标签映射
        """
    )
    parser.add_argument("--gpt_path", required=True)
    parser.add_argument("--sovits_path", required=True)
    parser.add_argument("--cnhubert_base_path", default="pretrained_models/chinese-hubert-base")
    parser.add_argument("--bert_path", default="pretrained_models/chinese-roberta-wwm-ext-large")

    # 情感预测器 ONNX 目录 (可选)
    parser.add_argument("--emotion_predictor_dir",
                        help="Directory containing emotion_head.onnx and meta.json "
                             "(default: GPT_SoVITS/emotion_predictor)")

    # 情感预设来源 (三选一)
    preset_group = parser.add_mutually_exclusive_group(required=True)
    preset_group.add_argument("--emotion_dir",
                              help="情感音频目录，自动扫描 *.wav + *.lab (推荐)")
    preset_group.add_argument("--emotion_config",
                              help="JSON file mapping emotion labels to audio paths")
    preset_group.add_argument("--ge_cache",
                              help="Pre-computed ge cache .pt file")

    # 当使用 --emotion_config 或 --ge_cache 时需要手动指定主音色
    parser.add_argument("--primary_audio", help="Primary (neutral) reference audio")
    parser.add_argument("--primary_text", help="Primary reference text")
    parser.add_argument("--primary_lang", default="zh")

    parser.add_argument("--text", required=True,
                        help='Tagged text, e.g. "<doubt>哎呀</doubt><sad>好难过</sad>"')
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--output", default="output_emotion.wav")

    # 推理模式 (三选一，默认 segment)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--blended", action="store_true",
                            help="按字符比例加权混合 ge (适合一句内多情感混合)")
    mode_group.add_argument("--phoneme_level", action="store_true",
                            help="Phone 级别精细 ge 注入 (实验性，最精细)")
    # 不加任何 flag = 默认 segment_level 模式

    parser.add_argument("--transition_phones", type=int, default=4,
                        help="Number of phones for emotion transition crossfade")
    parser.add_argument("--pause_length", type=float, default=0.3)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Base speed multiplier (default: 1.0)")
    parser.add_argument("--emotion_speeds", type=str, default=None,
                        help='Override emotion speed multipliers as JSON, '
                             'e.g. \'{"angry": 1.2, "sad": 0.85}\'')

    # 自动情感预测
    parser.add_argument("--auto_predict", action="store_true",
                        help="Auto-predict emotions from text using emotion_head.onnx "
                             "(requires emotion_head.onnx and meta.json)")
    parser.add_argument("--force_predict", action="store_true",
                        help="Force use predictor even if text has tags (implies --auto_predict)")

    args = parser.parse_args()

    # 决定是否启用情感预测器
    predictor_dir = None
    if args.auto_predict or args.force_predict or args.emotion_predictor_dir:
        predictor_dir = args.emotion_predictor_dir  # None 会使用默认路径

    engine = EmotionTTSInference(
        args.gpt_path, args.sovits_path,
        args.cnhubert_base_path, args.bert_path,
        emotion_predictor_onnx_dir=predictor_dir
    )

    # 覆盖情感语速倍率
    if args.emotion_speeds:
        overrides = json.loads(args.emotion_speeds)
        engine.emotion_speed.update(overrides)
        print(f"[EmotionTTS] Emotion speeds: {engine.emotion_speed}")

    # Load emotion presets
    if args.emotion_dir:
        emotion_audio_map, primary_audio, primary_text = scan_emotion_dir(args.emotion_dir)
        primary_lang = args.primary_lang
        engine.load_emotion_presets(
            emotion_audio_map, primary_audio, primary_text, primary_lang
        )
    elif args.ge_cache:
        if not args.primary_audio or not args.primary_text:
            print("错误: 使用 --ge_cache 时必须指定 --primary_audio 和 --primary_text")
            sys.exit(1)
        engine.load_emotion_presets_from_cache(
            args.ge_cache,
            args.primary_audio, args.primary_text, args.primary_lang
        )
    elif args.emotion_config:
        if not args.primary_audio or not args.primary_text:
            print("错误: 使用 --emotion_config 时必须指定 --primary_audio 和 --primary_text")
            sys.exit(1)
        with open(args.emotion_config, "r", encoding="utf-8") as f:
            emotion_audio_map = json.load(f)
        engine.load_emotion_presets(
            emotion_audio_map,
            args.primary_audio, args.primary_text, args.primary_lang
        )

    use_segment = not args.blended and not args.phoneme_level

    # 确定自动预测模式
    if args.force_predict:
        auto_predict = True   # 强制使用预测器
    elif args.auto_predict:
        auto_predict = None   # 自动检测 (无标签时预测)
    else:
        auto_predict = False  # 禁用

    audio, sr = engine.infer(
        args.text, text_lang=args.lang,
        speed=args.speed, pause_length=args.pause_length,
        transition_phones=args.transition_phones,
        segment_level=use_segment,
        phoneme_level=args.phoneme_level,
        auto_predict_emotion=auto_predict
    )

    import soundfile as sf
    sf.write(args.output, audio, sr)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
