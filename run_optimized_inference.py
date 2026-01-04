import os
import sys
import torch
import numpy as np
import librosa
import argparse
import warnings
import torchaudio
import soundfile as sf
import re
import time
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


def split_text(text):
    text = text.strip("\n")
    if not text:
        return []
    
    # Split by common sentence endings and keep them
    sentence_delimiters = r'([。！？.!?…\n])'
    parts = re.split(sentence_delimiters, text)
    
    # Reconstruct sentences
    sentences = []
    for i in range(0, len(parts)-1, 2):
        sentences.append(parts[i] + parts[i+1])
    if len(parts) % 2 == 1:
        sentences.append(parts[-1])
    
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Merge very short segments to avoid artifacts
    merged = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < 10: 
            current += s
        else:
            if current:
                merged.append(current)
            current = s
    if current:
        merged.append(current)
    
    return merged


class GPTSoVITSOptimizedInference:
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
                    for k in inputs:
                        inputs[k] = inputs[k].to(self.device)
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

    def infer_optimized(self, ref_wav_path, prompt_text, prompt_lang, text, text_lang,
                        top_k=15, top_p=1, temperature=1, speed=1, chunk_length=24, noise_scale=0.35,
                        history_window=4, pause_length=0.3):
        """
        Optimized inference:
        1. First sentence uses streaming for low latency.
        2. Subsequent sentences use previously generated tokens as context (long-context).
        3. Sliding window for history to maintain performance.
        """
        mute_matrix_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPT_SoVITS/pretrained_models/gpts1_mute_emb_sim_matrix.pt")
        mute_emb_sim_matrix = torch.load(mute_matrix_path, map_location=self.device) if os.path.exists(mute_matrix_path) else None

        with torch.no_grad():
            # Process Reference Audio (Once)
            wav16k, _ = librosa.load(ref_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k).to(self.device)
            if self.is_half: wav16k = wav16k.half()
            zero_wav = torch.zeros(int(16000 * 0.3), dtype=wav16k.dtype, device=self.device)
            wav16k = torch.cat([wav16k, zero_wav])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            prompt_semantic = self.vq_model.extract_latent(ssl_content)[0, 0]
            ref_tokens = prompt_semantic.unsqueeze(0).to(self.device)

            ref_phones, ref_bert = self.get_phones_and_bert(prompt_text, prompt_lang, self.hps.model.version)

            refer_spec, refer_audio = self.get_spepc(ref_wav_path)
            if refer_audio.shape[0] > 1: refer_audio = refer_audio[0].unsqueeze(0)
            audio_16k = torchaudio.transforms.Resample(self.hps.data.sampling_rate, 16000).to(self.device)(refer_audio) if self.hps.data.sampling_rate != 16000 else refer_audio
            sv_emb = self.sv_model.compute_embedding3(audio_16k)

            # Split Target Text
            segments = split_text(text)
            if not segments: return

            # History Management
            # sliding_history stores list of (phones, bert, tokens) for each segment in the window
            sliding_history = [] 

            sr = self.hps.data.sampling_rate
            samples_per_token = sr // 25
            # Crossfade params
            h_len, l_len, fade_len = 16, 16, 640 # Reduced fade_len for lower latency
            prev_fade_out = None

            for i, seg_text in enumerate(segments):
                # Process Current Segment Text
                curr_phones, curr_bert = self.get_phones_and_bert(seg_text, text_lang, self.hps.model.version)
                
                # Construct Input with History
                hist_phones_list = []
                hist_bert_list = []
                hist_tokens_list = []
                
                for hp, hb, ht in sliding_history:
                    hist_phones_list += hp
                    hist_bert_list.append(hb)
                    hist_tokens_list.append(ht)
                
                if not sliding_history:
                    inp_phones = torch.LongTensor(ref_phones + curr_phones).to(self.device).unsqueeze(0)
                    inp_bert = torch.cat([ref_bert, curr_bert], 1).unsqueeze(0).to(self.device)
                    inp_prompt = ref_tokens
                else:
                    inp_phones = torch.LongTensor(ref_phones + hist_phones_list + curr_phones).to(self.device).unsqueeze(0)
                    inp_bert = torch.cat([ref_bert] + hist_bert_list + [curr_bert], 1).unsqueeze(0).to(self.device)
                    inp_prompt = torch.cat([ref_tokens] + hist_tokens_list, 1)

                inp_phones_len = torch.tensor([inp_phones.shape[-1]]).to(self.device)

                # GPT Generator (Streaming)
                token_generator = self.t2s_model.model.infer_panel_naive(
                    inp_phones, inp_phones_len, inp_prompt, inp_bert, top_k=top_k, top_p=top_p,
                    temperature=temperature, early_stop_num=50 * 30, streaming_mode=True,
                    chunk_length=chunk_length, mute_emb_sim_matrix=mute_emb_sim_matrix
                )

                curr_phones_tensor = torch.LongTensor(curr_phones).to(self.device).unsqueeze(0)
                
                chunk_queue = []
                seg_tokens_collected = []
                seg_history_tokens = None # inner-segment history for lookahead decoding

                def decode_and_crop(tokens, hist, lookahead):
                    input_list = []
                    if hist is not None: 
                        input_list.append(hist[:, -h_len:])
                    input_list.append(tokens)
                    if lookahead is not None: 
                        input_list.append(lookahead[:, :l_len])
                    
                    full_chunk = torch.cat(input_list, dim=1)
                    audio = self.vq_model.decode(full_chunk.unsqueeze(0), curr_phones_tensor, [refer_spec], noise_scale=noise_scale, speed=speed, sv_emb=[sv_emb])
                    
                    # h_samples = min(hist.shape[1] if hist is not None else 0, h_len) * samples_per_token
                    # Correct sample calculation with speed
                    actual_samples_per_token = samples_per_token / speed
                    h_samples = int((min(hist.shape[1] if hist is not None else 0, h_len)) * actual_samples_per_token)
                    c_samples = int(tokens.shape[1] * actual_samples_per_token)
                    
                    return audio[0, 0].cpu().float().numpy()[h_samples : h_samples + c_samples]

                for chunk, is_last in token_generator:
                    if chunk is not None:
                        chunk_queue.append(chunk)
                        seg_tokens_collected.append(chunk)
                    
                    while len(chunk_queue) > 1:
                        curr = chunk_queue.pop(0)
                        next_chunk = chunk_queue[0]
                        audio_data = decode_and_crop(curr, seg_history_tokens, next_chunk)
                        
                        if prev_fade_out is not None:
                            fade_in = np.linspace(0, 1, fade_len)
                            # Handle potential size mismatch in extreme cases
                            actual_fade = min(fade_len, len(audio_data))
                            audio_data[:actual_fade] = audio_data[:actual_fade] * fade_in[:actual_fade] + prev_fade_out[:actual_fade] * (1 - fade_in[:actual_fade])
                        
                        prev_fade_out = audio_data[-fade_len:]
                        yield audio_data[:-fade_len]
                        seg_history_tokens = curr

                    if is_last and chunk_queue:
                        final_curr = chunk_queue.pop(0)
                        audio_data = decode_and_crop(final_curr, seg_history_tokens, None)
                        
                        if prev_fade_out is not None:
                            fade_in = np.linspace(0, 1, fade_len)
                            actual_fade = min(fade_len, len(audio_data))
                            audio_data[:actual_fade] = audio_data[:actual_fade] * fade_in[:actual_fade] + prev_fade_out[:actual_fade] * (1 - fade_in[:actual_fade])
                        
                        # For the very last chunk of a segment, we might want to keep the fade_out for the next segment
                        yield audio_data[:-fade_len]
                        prev_fade_out = audio_data[-fade_len:]

                # Update sliding window history
                seg_tokens_tensor = torch.cat(seg_tokens_collected, 1) if seg_tokens_collected else torch.zeros((1, 0), device=self.device)
                sliding_history.append((curr_phones, curr_bert, seg_tokens_tensor))
                if len(sliding_history) > history_window:
                    sliding_history.pop(0)

                # Add pause between segments
                if i < len(segments) - 1 and pause_length > 0:
                    if prev_fade_out is not None:
                        yield prev_fade_out
                        prev_fade_out = None
                    yield np.zeros(int(sr * pause_length), dtype=np.float32)

            # Yield remaining
            if prev_fade_out is not None:
                yield prev_fade_out


def launch_webui(args):
    import gradio as gr
    inference = GPTSoVITSOptimizedInference(args.gpt_path, args.sovits_path, args.cnhubert_base_path, args.bert_path)
    
    def predict(ref_audio, ref_text, ref_lang, text, lang, top_k, top_p, temperature, speed, chunk_length, noise_scale, history_window, pause_length):
        if ref_audio is None or not text:
            return
        
        start_time = time.time()
        first_packet_time = None
        total_samples = 0
        
        gen = inference.infer_optimized(
            ref_audio, ref_text, ref_lang, text, lang,
            top_k=top_k, top_p=top_p, temperature=temperature, 
            speed=speed, chunk_length=chunk_length, noise_scale=noise_scale,
            history_window=int(history_window), pause_length=pause_length
        )
        
        sr = inference.hps.data.sampling_rate
        last_chunk = None
        for audio_chunk in gen:
            if first_packet_time is None:
                first_packet_time = time.time()
                latency = first_packet_time - start_time
            
            total_samples += len(audio_chunk)
            if last_chunk is not None:
                yield (sr, (last_chunk * 32768).astype(np.int16)), f"{latency:.3f}s", "", ""
            last_chunk = audio_chunk

        if last_chunk is not None:
            end_time = time.time()
            total_duration = total_samples / sr
            total_inference_time = end_time - start_time
            rtf = total_inference_time / total_duration if total_duration > 0 else 0
            yield (sr, (last_chunk * 32768).astype(np.int16)), f"{latency:.3f}s", f"{total_duration:.2f}s", f"{rtf:.3f}"

    with gr.Blocks(title="GPT-SoVITS Optimized Inference (Streaming + Long Context)") as app:
        gr.Markdown("# GPT-SoVITS Optimized Inference")
        gr.Markdown("Combined Streaming (for low latency) and Long-Context (for better cross-sentence consistency).")
        
        with gr.Row():
            with gr.Column():
                ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                ref_text = gr.Textbox(label="Reference Text", value=args.ref_text)
                ref_lang = gr.Dropdown(label="Reference Language", choices=["zh", "en", "ja", "ko", "yue"], value=args.ref_lang)
                target_text = gr.Textbox(label="Target Text", lines=5, value=args.text)
                target_lang = gr.Dropdown(label="Target Language", choices=["zh", "en", "ja", "ko", "yue", "auto", "auto_yue"], value=args.lang)
                
                with gr.Accordion("Optimization Settings", open=True):
                    history_window = gr.Slider(label="History Window (Sentences)", minimum=0, maximum=10, step=1, value=4)
                    chunk_length = gr.Slider(label="GPT Chunk Length", minimum=10, maximum=100, step=1, value=24)
                    speed = gr.Slider(label="Speed", minimum=0.5, maximum=2.0, step=0.1, value=1.0)
                    pause_length = gr.Slider(label="Pause Length", minimum=0.0, maximum=1.0, step=0.05, value=0.3)

                with gr.Accordion("Advanced Model Settings", open=False):
                    top_k = gr.Slider(label="Top K", minimum=1, maximum=100, step=1, value=15)
                    top_p = gr.Slider(label="Top P", minimum=0.1, maximum=1.0, step=0.05, value=1.0)
                    temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, step=0.05, value=1.0)
                    noise_scale = gr.Slider(label="Noise Scale", minimum=0.0, maximum=1.0, step=0.05, value=0.35)
                
                btn = gr.Button("Generate", variant="primary")
            
            with gr.Column():
                audio_output = gr.Audio(label="Output Audio", streaming=True, autoplay=True)
                with gr.Row():
                    latency_label = gr.Textbox(label="首包延迟 (First Packet Latency)", interactive=False)
                    duration_label = gr.Textbox(label="音频总时长 (Total Duration)", interactive=False)
                    rtf_label = gr.Textbox(label="实时率 (RTF)", interactive=False)

        btn.click(
            predict,
            inputs=[ref_audio, ref_text, ref_lang, target_text, target_lang, top_k, top_p, temperature, speed, chunk_length, noise_scale, history_window, pause_length],
            outputs=[audio_output, latency_label, duration_label, rtf_label]
        )

    app.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS Optimized Inference")
    parser.add_argument("--gpt_path", required=True)
    parser.add_argument("--sovits_path", required=True)
    parser.add_argument("--cnhubert_base_path", default="pretrained_models/chinese-hubert-base")
    parser.add_argument("--bert_path", default="pretrained_models/chinese-roberta-wwm-ext-large")
    parser.add_argument("--ref_audio", default=None)
    parser.add_argument("--ref_text", default="")
    parser.add_argument("--ref_lang", default="zh")
    parser.add_argument("--text", default="")
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--output", default="out_optimized.wav")
    parser.add_argument("--history_window", type=int, default=4)
    parser.add_argument("--pause_length", type=float, default=0.3)
    
    # WebUI arguments
    parser.add_argument("--webui", action="store_true", help="Launch WebUI")
    parser.add_argument("--share", action="store_true", help="Share Gradio app")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9882)

    args = parser.parse_args()

    if args.webui:
        launch_webui(args)
    else:
        if not args.ref_audio or not args.text:
            parser.error("--ref_audio and --text are required in CLI mode")
        inference = GPTSoVITSOptimizedInference(args.gpt_path, args.sovits_path, args.cnhubert_base_path, args.bert_path)
        full_audio = []
        for audio_chunk in inference.infer_optimized(args.ref_audio, args.ref_text, args.ref_lang, args.text, args.lang, history_window=args.history_window, pause_length=args.pause_length):
            full_audio.append(audio_chunk)
        
        if full_audio:
            sf.write(args.output, np.concatenate(full_audio), inference.hps.data.sampling_rate)
            print(f"Saved to {args.output}")