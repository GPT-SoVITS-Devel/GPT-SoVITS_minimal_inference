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

        # SV Model (for v2Pro/Plus)
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

    def infer_stream(self, ref_wav_path, prompt_text, prompt_lang, text, text_lang,
                    top_k=5, top_p=1, temperature=1, speed=1, chunk_length=24):

        print(f"Streaming Inference: {text} ({text_lang})")

        # Load Mute Matrix for natural splits
        mute_matrix_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPT_SoVITS/pretrained_models/gpts1_mute_emb_sim_matrix.pt")
        mute_emb_sim_matrix = None
        if os.path.exists(mute_matrix_path):
            mute_emb_sim_matrix = torch.load(mute_matrix_path, map_location=self.device)

        # Process Reference
        zero_wav = torch.zeros(
            int(self.hps.data.sampling_rate * 0.3),
            dtype=torch.float16 if self.is_half else torch.float32
        ).to(self.device)

        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k).to(self.device)
            if self.is_half: wav16k = wav16k.half()
            wav16k = torch.cat([wav16k, zero_wav])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            codes = self.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(self.device)

        # Process Text
        phones1, bert1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_lang, self.hps.model.version)
        phones2, bert2, norm_text2 = self.get_phones_and_bert(text, text_lang, self.hps.model.version)

        bert = torch.cat([bert1, bert2], 1).unsqueeze(0).to(self.device)
        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

        # SoVITS Preparations
        refer_spec, refer_audio = self.get_spepc(ref_wav_path)
        if refer_audio.shape[0] > 1: refer_audio = refer_audio[0].unsqueeze(0)
        if self.hps.data.sampling_rate != 16000:
            audio_16k = torchaudio.transforms.Resample(self.hps.data.sampling_rate, 16000).to(self.device)(refer_audio)
        else:
            audio_16k = refer_audio
        sv_emb = self.sv_model.compute_embedding3(audio_16k)

        # 4. GPT Streaming Loop with Context
        token_generator = self.t2s_model.model.infer_panel_naive(
            all_phoneme_ids,
            all_phoneme_len,
            prompt,
            bert,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            early_stop_num=50 * 30,
            streaming_mode=True,
            chunk_length=chunk_length,
            mute_emb_sim_matrix=mute_emb_sim_matrix
        )

        phones2_tensor = torch.LongTensor(phones2).to(self.device).unsqueeze(0)
        
        history_tokens = None
        current_tokens = None
        
        # h_len: context from past to stabilize the start of current chunk
        # l_len: context from future to stabilize the end of current chunk
        h_len = 16 
        l_len = 16

        for tokens_chunk, is_last in token_generator:
            if tokens_chunk is None: continue
            
            if current_tokens is None:
                current_tokens = tokens_chunk
                if not is_last:
                    continue
            
            # Now we have current_tokens to decode.
            # tokens_chunk serves as the lookahead (future context).
            
            # Construct context input: history + current + lookahead
            input_list = []
            if history_tokens is not None:
                input_list.append(history_tokens[:, -h_len:])
            input_list.append(current_tokens)
            
            # If it's the last chunk, we have no more lookahead
            # If not, use the start of the next chunk as lookahead
            actual_l_len = 0
            if not is_last:
                input_list.append(tokens_chunk[:, :l_len])
                actual_l_len = min(tokens_chunk.shape[1], l_len)
            
            full_chunk = torch.cat(input_list, dim=1)
            
            # result_length: the part we want to KEEP (current_tokens)
            # padding_length: the part we want to DISCARD from the end (lookahead)
            # Units: Semantic Tokens (decode_streaming will handle Hz conversion)
            r_len = current_tokens.shape[1]
            p_len = actual_l_len
            
            with torch.no_grad():
                audio_chunk, _, _ = self.vq_model.decode_streaming(
                    codes=full_chunk.unsqueeze(0),
                    text=phones2_tensor,
                    refer=[refer_spec],
                    speed=speed,
                    sv_emb=[sv_emb],
                    overlap_frames=None, 
                    padding_length=p_len,
                    result_length=r_len
                )
            
            yield audio_chunk[0, 0].cpu().float().numpy()
            
            # Prepare for next iteration
            history_tokens = current_tokens
            current_tokens = tokens_chunk
            
            if is_last:
                # The very last chunk (which was tokens_chunk) needs to be decoded
                input_list = []
                if history_tokens is not None:
                    input_list.append(history_tokens[:, -h_len:])
                input_list.append(current_tokens)
                full_chunk = torch.cat(input_list, dim=1)
                
                with torch.no_grad():
                    audio_chunk, _, _ = self.vq_model.decode_streaming(
                        codes=full_chunk.unsqueeze(0),
                        text=phones2_tensor,
                        refer=[refer_spec],
                        speed=speed,
                        sv_emb=[sv_emb],
                        overlap_frames=None,
                        padding_length=0,
                        result_length=current_tokens.shape[1]
                    )
                yield audio_chunk[0, 0].cpu().float().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS Streaming Inference")
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
    parser.add_argument("--output", default="out_streaming.wav", help="Output filename")
    parser.add_argument("--chunk_length", type=int, default=24, help="Semantic chunk length")

    args = parser.parse_args()

    inference = GPTSoVITSStreamingInference(
        args.gpt_path,
        args.sovits_path,
        args.cnhubert_base_path,
        args.bert_path
    )

    full_audio = []
    sr = inference.hps.data.sampling_rate
    
    for audio_chunk in inference.infer_stream(
        args.ref_audio,
        args.ref_text,
        args.ref_lang,
        args.text,
        args.lang,
        chunk_length=args.chunk_length
    ):
        full_audio.append(audio_chunk)
        # In a real streaming scenario, you would send audio_chunk to the client here.
        print(f"Yielded audio chunk: {len(audio_chunk)} samples")

    if full_audio:
        combined_audio = np.concatenate(full_audio)
        sf.write(args.output, combined_audio, sr)
        print(f"Saved full streaming result to {args.output}")
