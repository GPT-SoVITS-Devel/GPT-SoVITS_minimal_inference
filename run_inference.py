import os
import re
import time

os.environ['http_proxy'] = "http://192.168.1.50:10809"
os.environ['https_proxy'] = "http://192.168.1.50:10809"
import sys
import torch
import numpy as np
import librosa
import argparse
from GPT_SoVITS.utils import load_audio_equivalent
import warnings
import torchaudio

# Filter warnings
warnings.filterwarnings("ignore")

# Setup paths
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

device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = True if device == "cuda" else False


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


class GPTSoVITSInference:
    def __init__(self, gpt_path, sovits_path, cnhubert_base_path, bert_path):
        self.device = device
        self.is_half = is_half

        print(f"Loading models on {device} (half precision: {is_half})...")

        # Load CNHubert
        cnhubert.cnhubert_base_path = cnhubert_base_path
        self.ssl_model = cnhubert.get_model()
        if is_half:
            self.ssl_model = self.ssl_model.half()
        self.ssl_model = self.ssl_model.to(device)

        # Load BERT
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        if is_half:
            self.bert_model = self.bert_model.half()
        self.bert_model = self.bert_model.to(device)

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

        # Determine version
        _, model_version, _ = get_sovits_version_from_path_fast(sovits_path)
        if "config" in dict_s2 and "model" in dict_s2["config"] and "version" in dict_s2["config"]["model"]:
            model_version = dict_s2["config"]["model"]["version"]
        elif "sv_emb.weight" in dict_s2["weight"]:
            model_version = "v2Pro"
        
        self.hps.model.version = model_version
        print(f"Detected SoVITS model version: {model_version}")

        # Check for Pro/Plus
        # Heuristic: User should provide v2 models as requested.
        # Ideally check metadata or file hash, but for minimal script we assume config matches.

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

        # SV Model (for v2Pro/Plus)
        # We initialize it if needed or just always init for simplicity in this minimal script
        self.sv_model = SV(device, is_half)

        self.warmup()

    def warmup(self):
        print("Warming up models (tokenizer, BERT, etc.)...")
        # Warmup text cleaning and BERT (common sources of initial delay)
        try:
            _ = self.get_phones_and_bert("Warmup text.", "en", self.hps.model.version)
            _ = self.get_phones_and_bert("你好，预热文本。", "zh", self.hps.model.version)
        except Exception as e:
            print(f"Warmup failed (non-critical): {e}")

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
                (1024, len(phones)),
                dtype=torch.float16 if self.is_half else torch.float32,
            ).to(self.device)

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
        
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)

        return phones, bert.to(torch.float16 if self.is_half else torch.float32), norm_text

    def get_spepc(self, filename):
        audio, sr = load_audio_equivalent(filename, self.device)
        if sr != self.hps.data.sampling_rate:
            audio = torchaudio.transforms.Resample(sr, self.hps.data.sampling_rate).to(self.device)(audio)

        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)

        spec = spectrogram_torch(
            audio,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False
        )
        if self.is_half:
            spec = spec.half()
        return spec, audio

    def infer(self, ref_wav_path, prompt_text, prompt_lang, text, text_lang,
              top_k=5, top_p=1, temperature=1, speed=1, pause_length=0.3):

        print(f"Inferencing: {text} ({text_lang})")
        
        t_all_start = time.perf_counter()
        segments = split_text(text)
        if not segments:
            return np.zeros(0), self.hps.data.sampling_rate
            
        final_audios = []
        sr = self.hps.data.sampling_rate

        # Process Reference
        t_ref_start = time.perf_counter()
        zero_wav = torch.zeros(
            int(self.hps.data.sampling_rate * 0.3),
            dtype=torch.float16 if self.is_half else torch.float32
        ).to(self.device)

        with torch.no_grad():
            wav16k, _ = librosa.load(ref_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k).to(self.device)
            if self.is_half: wav16k = wav16k.half()

            # Extract SSL
            wav16k = torch.cat([wav16k, zero_wav])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            codes = self.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(self.device)

            # SoVITS Inference Setup
            refer_spec, refer_audio = self.get_spepc(ref_wav_path)
            if refer_audio.shape[0] > 1: refer_audio = refer_audio[0].unsqueeze(0)
            if self.hps.data.sampling_rate != 16000:
                audio_16k = torchaudio.transforms.Resample(self.hps.data.sampling_rate, 16000).to(self.device)(refer_audio)
            else:
                audio_16k = refer_audio
            sv_emb = self.sv_model.compute_embedding3(audio_16k)

        # Process Reference Text (Once)
        phones1, bert1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_lang, self.hps.model.version)
        t_ref_end = time.perf_counter()

        total_text_time = 0
        total_gpt_time = 0
        total_sovits_time = 0

        for i, seg in enumerate(segments):
            print(f"Processing segment {i+1}/{len(segments)}: {seg}")
            
            # Process Text Segment
            t_seg_text_start = time.perf_counter()
            phones2, bert2, norm_text2 = self.get_phones_and_bert(seg, text_lang, self.hps.model.version)
            t_seg_text_end = time.perf_counter()
            total_text_time += (t_seg_text_end - t_seg_text_start)

            bert = torch.cat([bert1, bert2], 1).unsqueeze(0).to(self.device)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

            # GPT Inference
            t_gpt_start = time.perf_counter()
            with torch.no_grad():
                pred_semantic, idx = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    bert,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=50 * 30  # max_sec
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            t_gpt_end = time.perf_counter()
            total_gpt_time += (t_gpt_end - t_gpt_start)

            # SoVITS Decoding
            t_sovits_start = time.perf_counter()
            with torch.no_grad():
                # Standard V2/V2Pro decoding
                audio = self.vq_model.decode(
                    pred_semantic,
                    torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                    [refer_spec],  # List of refers
                    speed=speed,
                    sv_emb=[sv_emb]  # List of sv_embs
                )[0][0]
            t_sovits_end = time.perf_counter()
            total_sovits_time += (t_sovits_end - t_sovits_start)
                
            final_audios.append(audio.cpu().float().numpy())
            if i < len(segments) - 1 and pause_length > 0:
                final_audios.append(np.zeros(int(sr * pause_length)))

        t_all_end = time.perf_counter()
        
        print(f"\n--- Inference Performance Summary ---")
        print(f"Reference Processing:  {t_ref_end - t_ref_start:.3f}s")
        print(f"Target Text Cleaning:  {total_text_time:.3f}s")
        print(f"GPT Semantic Gen:      {total_gpt_time:.3f}s")
        print(f"SoVITS Audio Decode:   {total_sovits_time:.3f}s")
        print(f"Total Inference Time:  {t_all_end - t_all_start:.3f}s")
        print(f"-------------------------------------\n")

        return np.concatenate(final_audios), sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS Minimal Inference (v2/v2Pro)")
    parser.add_argument("--gpt_path", required=True, help="Path to GPT model (.ckpt)")
    parser.add_argument("--sovits_path", required=True, help="Path to SoVITS model (.pth)")
    parser.add_argument("--cnhubert_base_path", default="pretrained_models/chinese-hubert-base",
                        help="Path to CNHubert base")
    parser.add_argument("--bert_path", default="pretrained_models/chinese-roberta-wwm-ext-large",
                        help="Path to BERT model")
    parser.add_argument("--ref_audio", required=True, help="Reference audio path")
    parser.add_argument("--ref_text", required=True, help="Reference audio text")
    parser.add_argument("--ref_lang", default="zh", help="Reference language (zh, en, ja, ko, yue)")
    parser.add_argument("--text", required=True, help="Target text")
    parser.add_argument("--lang", default="zh", help="Target language (zh, en, ja, ko, yue)")
    parser.add_argument("--output", default="output.wav", help="Output filename")
    parser.add_argument("--pause_length", type=float, default=0.3, help="Pause length between sentences (seconds)")

    args = parser.parse_args()

    inference = GPTSoVITSInference(
        args.gpt_path,
        args.sovits_path,
        args.cnhubert_base_path,
        args.bert_path
    )

    audio, sr = inference.infer(
        args.ref_audio,
        args.ref_text,
        args.ref_lang,
        args.text,
        args.lang,
        pause_length=args.pause_length
    )

    import soundfile as sf

    sf.write(args.output, audio, sr)
    print(f"Saved to {args.output}")