import os
import sys
import torch
import numpy as np
import librosa
import argparse
import warnings
import torchaudio
import soundfile as sf
from tqdm import tqdm

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
from GPT_SoVITS.utils import load_audio_equivalent

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


class GPTSoVITSStreamingInference:
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

        # SV Model
        self.sv_model = SV(device, is_half)

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
                if tmp["lang"] == "zh": tmp["lang"] = "yue"
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
                if tmp["lang"] == "zh": tmp["lang"] = "yue"
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
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text(textlist[i], lang, version)
            phones = cleaned_text_to_sequence(phones, version)
            
            if lang in ["zh", "yue"]:
                with torch.no_grad():
                    inputs = self.tokenizer(norm_text, return_tensors="pt")
                    for k in inputs: inputs[k] = inputs[k].to(self.device)
                    res = self.bert_model(**inputs, output_hidden_states=True)
                    res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
                
                phone_level_feature = []
                for j in range(len(word2ph)):
                    phone_level_feature.append(res[j].repeat(word2ph[j], 1))
                bert = torch.cat(phone_level_feature, dim=0).T
            else:
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float16 if self.is_half else torch.float32,
                )

            phones_list.append(phones)
            bert_list.append(bert)
        
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        return phones, bert.to(torch.float16 if self.is_half else torch.float32)

    def get_spepc(self, filename):
        audio, sr = load_audio_equivalent(filename, self.device)
        if sr != self.hps.data.sampling_rate:
            audio = torchaudio.transforms.Resample(sr, self.hps.data.sampling_rate).to(self.device)(audio)
        if audio.shape[0] > 1: audio = audio.mean(0, keepdim=True)
        spec = spectrogram_torch(audio, self.hps.data.filter_length, self.hps.data.sampling_rate,
                                 self.hps.data.hop_length, self.hps.data.win_length, center=False)
        if self.is_half: spec = spec.half()
        return spec, audio

    def infer_stream(self, ref_wav_path, prompt_text, prompt_lang, text, text_lang,
                    top_k=5, top_p=1, temperature=1, speed=1, chunk_length=24):

        # Load Mute Matrix
        mute_matrix_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPT_SoVITS/pretrained_models/gpts1_mute_emb_sim_matrix.pt")
        mute_emb_sim_matrix = torch.load(mute_matrix_path, map_location=self.device) if os.path.exists(mute_matrix_path) else None

        # Process Reference
        with torch.no_grad():
            wav16k, _ = librosa.load(ref_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k).to(self.device)
            if self.is_half: wav16k = wav16k.half()
            zero_wav = torch.zeros(int(16000 * 0.3), dtype=wav16k.dtype, device=self.device)
            wav16k = torch.cat([wav16k, zero_wav])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            prompt_semantic = self.vq_model.extract_latent(ssl_content)[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(self.device)

        # Process Text
        phones1, bert1 = self.get_phones_and_bert(prompt_text, prompt_lang, self.hps.model.version)
        phones2, bert2 = self.get_phones_and_bert(text, text_lang, self.hps.model.version)
        bert = torch.cat([bert1, bert2], 1).unsqueeze(0).to(self.device)
        all_phones = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
        all_phones_len = torch.tensor([all_phones.shape[-1]]).to(self.device)

        # SoVITS Preparations
        refer_spec, refer_audio = self.get_spepc(ref_wav_path)
        if refer_audio.shape[0] > 1: refer_audio = refer_audio[0].unsqueeze(0)
        audio_16k = torchaudio.transforms.Resample(self.hps.data.sampling_rate, 16000).to(self.device)(refer_audio) if self.hps.data.sampling_rate != 16000 else refer_audio
        sv_emb = self.sv_model.compute_embedding3(audio_16k)

        # GPT Streaming Generator
        token_generator = self.t2s_model.model.infer_panel_naive(
            all_phones, all_phones_len, prompt, bert, top_k=top_k, top_p=top_p,
            temperature=temperature, early_stop_num=50 * 30, streaming_mode=True,
            chunk_length=chunk_length, mute_emb_sim_matrix=mute_emb_sim_matrix
        )

        phones2_tensor = torch.LongTensor(phones2).to(self.device).unsqueeze(0)
        sr = self.hps.data.sampling_rate
        samples_per_token = sr // 25
        h_len, l_len, fade_len = 16, 16, 160
        history_tokens, prev_fade_out = None, None
        
        # Buffer queue for lookahead
        chunk_queue = []

        def decode_and_crop(tokens, hist, lookahead):
            input_list = []
            if hist is not None: input_list.append(hist[:, -h_len:])
            input_list.append(tokens)
            if lookahead is not None: input_list.append(lookahead[:, :l_len])
            
            full_chunk = torch.cat(input_list, dim=1)
            with torch.no_grad():
                # We use manual cropping because our models.py modification 
                # handles context-aware Transformer encoding perfectly.
                audio = self.vq_model.decode(full_chunk.unsqueeze(0), phones2_tensor, [refer_spec], noise_scale=0, speed=speed, sv_emb=[sv_emb])
            
            h_samples = min(hist.shape[1] if hist is not None else 0, h_len) * samples_per_token
            c_samples = tokens.shape[1] * samples_per_token
            return audio[0, 0].cpu().float().numpy()[h_samples : h_samples + c_samples]

        for chunk, is_last in token_generator:
            if chunk is not None:
                chunk_queue.append(chunk)
            
            # While we have at least one chunk to output and one for lookahead
            while len(chunk_queue) > 1:
                curr = chunk_queue.pop(0)
                next_chunk = chunk_queue[0]
                
                audio_data = decode_and_crop(curr, history_tokens, next_chunk)
                
                # Crossfade
                if prev_fade_out is not None:
                    fade_in = np.linspace(0, 1, fade_len)
                    audio_data[:fade_len] = audio_data[:fade_len] * fade_in + prev_fade_out * (1 - fade_in)
                
                prev_fade_out = audio_data[-fade_len:]
                yield audio_data[:-fade_len]
                history_tokens = curr

            if is_last and chunk_queue:
                # Handle the final chunk in queue
                final_curr = chunk_queue.pop(0)
                audio_data = decode_and_crop(final_curr, history_tokens, None)
                
                if prev_fade_out is not None:
                    fade_in = np.linspace(0, 1, fade_len)
                    audio_data[:fade_len] = audio_data[:fade_len] * fade_in + prev_fade_out * (1 - fade_in)
                
                yield audio_data

        print("Streaming Inference Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS Streaming Inference")
    parser.add_argument("--gpt_path", required=True)
    parser.add_argument("--sovits_path", required=True)
    parser.add_argument("--cnhubert_base_path", default="pretrained_models/chinese-hubert-base")
    parser.add_argument("--bert_path", default="pretrained_models/chinese-roberta-wwm-ext-large")
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--ref_text", required=True)
    parser.add_argument("--ref_lang", default="zh")
    parser.add_argument("--text", required=True)
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--output", default="out_streaming.wav")
    parser.add_argument("--chunk_length", type=int, default=24)

    args = parser.parse_args()
    inference = GPTSoVITSStreamingInference(args.gpt_path, args.sovits_path, args.cnhubert_base_path, args.bert_path)
    full_audio = []
    for audio_chunk in inference.infer_stream(args.ref_audio, args.ref_text, args.ref_lang, args.text, args.lang, chunk_length=args.chunk_length):
        full_audio.append(audio_chunk)
        print(f"Yielded audio chunk: {len(audio_chunk)} samples")
    if full_audio:
        sf.write(args.output, np.concatenate(full_audio), inference.hps.data.sampling_rate)
        print(f"Saved to {args.output}")