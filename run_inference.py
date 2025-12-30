import os

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

    def get_phones_and_bert(self, text, language):
        # Minimal clean_text_inf logic
        # Assuming input is just text, language specified
        # Language: "zh", "en", "ja", "yue", "ko"

        if language in ["en", "all_zh", "all_ja", "all_ko", "all_yue"]:
            lang_code = language.replace("all_", "")
        else:
            lang_code = language

        phones, word2ph, norm_text = clean_text(text, lang_code, self.hps.model.version)
        phones = cleaned_text_to_sequence(phones, self.hps.model.version)

        dtype = torch.float16 if self.is_half else torch.float32

        if lang_code == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=dtype
            ).to(self.device)

        return phones, bert, norm_text

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
              top_k=5, top_p=1, temperature=1, speed=1):

        print(f"Inferencing: {text} ({text_lang})")

        # 1. Process Reference
        zero_wav = torch.zeros(
            int(self.hps.data.sampling_rate * 0.3),
            dtype=torch.float16 if self.is_half else torch.float32
        ).to(self.device)

        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k).to(self.device)
            if self.is_half: wav16k = wav16k.half()

            # Extract SSL
            wav16k = torch.cat([wav16k, zero_wav])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            codes = self.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(self.device)

        # 2. Process Text
        phones1, bert1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_lang)
        phones2, bert2, norm_text2 = self.get_phones_and_bert(text, text_lang)

        bert = torch.cat([bert1, bert2], 1).unsqueeze(0).to(self.device)
        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

        # 3. GPT Inference
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

        # 4. SoVITS Inference
        refer_spec, refer_audio = self.get_spepc(ref_wav_path)

        # SV Embedding (v2Pro)
        # We need 16k audio for SV
        if refer_audio.shape[0] > 1: refer_audio = refer_audio[0].unsqueeze(0)

        # Resample for SV (16k)
        if self.hps.data.sampling_rate != 16000:
            audio_16k = torchaudio.transforms.Resample(self.hps.data.sampling_rate, 16000).to(self.device)(refer_audio)
        else:
            audio_16k = refer_audio

        sv_emb = self.sv_model.compute_embedding3(audio_16k)

        with torch.no_grad():
            # Standard V2/V2Pro decoding
            audio = self.vq_model.decode(
                pred_semantic,
                torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                [refer_spec],  # List of refers
                speed=speed,
                sv_emb=[sv_emb]  # List of sv_embs
            )[0][0]

        return audio.cpu().float().numpy(), self.hps.data.sampling_rate


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
        args.lang
    )

    import soundfile as sf

    sf.write(args.output, audio, sr)
    print(f"Saved to {args.output}")