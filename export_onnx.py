import argparse
import os
import sys
import torch
import json
from torch import nn
from torch.nn import functional as F
from GPT_SoVITS.process_ckpt import load_sovits_new, get_sovits_version_from_path_fast
from GPT_SoVITS.feature_extractor import cnhubert
from GPT_SoVITS.text import _symbol_to_id_v2
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.module.models import SynthesizerTrn
from transformers import AutoModelForMaskedLM, AutoTokenizer
import logging

# ONNX校验
import onnx_validation

logging.getLogger("torch.onnx").setLevel(logging.WARN)
logging.getLogger("onnx").setLevel(logging.WARN)
logging.getLogger("onnx_ir").setLevel(logging.WARN)
logging.getLogger("onnxscript").setLevel(logging.WARN)

# Wrappers for ONNX Export

class T2SEncoder(nn.Module):
    def __init__(self, t2s_model):
        super().__init__()
        self.t2s_model = t2s_model

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        pass

class GPTEncoder(nn.Module):
    def __init__(self, t2s_model, max_len=2000):
        super().__init__()
        self.t2s_model = t2s_model
        self.max_len = max_len

    def forward(self, phoneme_ids, prompts, bert_feature):
        # Wrapper for infer_first_stage
        # Returns: logits, k_cache (stacked), v_cache (stacked), x_len, y_len
        
        actual_len = phoneme_ids.shape[1]
        
        logits, k_cache, v_cache, x_len, y_len = self.t2s_model.model.infer_first_stage(
            phoneme_ids, prompts, bert_feature
        )
        
        # Stack caches: List[Tensor] -> Tensor [Layers, B, T, D]
        k_cache_stacked = torch.stack(k_cache, dim=0)
        v_cache_stacked = torch.stack(v_cache, dim=0)
        
        # Pad to max length for pre-allocation
        k_cache_padded = F.pad(k_cache_stacked, (0, 0, 0, self.max_len - k_cache_stacked.shape[2]))
        v_cache_padded = F.pad(v_cache_stacked, (0, 0, 0, self.max_len - v_cache_stacked.shape[2]))
        
        # Optimization: Return Top-K instead of full logits
        topk_values, topk_indices = torch.topk(logits, k=50, dim=-1)
        
        # Ensure x_len and y_len are rank-1 tensors for ONNX export dynamic axes
        if not isinstance(x_len, torch.Tensor):
            x_len = torch.tensor([x_len], dtype=torch.long)
        else:
            x_len = x_len.reshape(1)
        if not isinstance(y_len, torch.Tensor):
            y_len = torch.tensor([y_len], dtype=torch.long)
        else:
            y_len = y_len.reshape(1)
        
        return topk_values, topk_indices, k_cache_padded, v_cache_padded, x_len, y_len

class GPTStep(nn.Module):
    def __init__(self, t2s_model):
        super().__init__()
        self.t2s_model = t2s_model

    def forward(self, samples, k_cache, v_cache, x_len, y_len, idx):
        # Wrapper for infer_next_stage
        # k_cache, v_cache are stacked [Layers, B, T_max, D]
        
        # Ensure x_len, y_len, idx are scalars for the underlying model if they come as rank-1
        x_len_s = x_len[0] if x_len.ndim > 0 else x_len
        y_len_s = y_len[0] if y_len.ndim > 0 else y_len
        idx_s = idx[0] if idx.ndim > 0 else idx

        # Unstack to list
        k_cache_list = [t for t in k_cache]
        v_cache_list = [t for t in v_cache]
        
        logits, k_cache_new, v_cache_new = self.t2s_model.model.infer_next_stage(
            samples, k_cache_list, v_cache_list, x_len_s, y_len_s, idx_s
        )
        
        # Stack again (they should still be the same tensors if updated in-place)
        k_cache_stacked = torch.stack(k_cache_new, dim=0)
        v_cache_stacked = torch.stack(v_cache_new, dim=0)
        
        # Optimization: Return Top-K instead of full logits to reduce GPU->CPU transfer
        # SoVITS vocabulary is 1025. Returning Top-50 is enough for high-quality sampling.
        topk_values, topk_indices = torch.topk(logits, k=50, dim=-1)
        
        return topk_values, topk_indices, k_cache_stacked, v_cache_stacked

class SoVITS(nn.Module):
    def __init__(self, vq_model, version):
        super().__init__()
        self.vq_model = vq_model
        self.version = version

    def forward(self, pred_semantic, text_seq, refer_spec, sv_emb=None, noise_scale=0.5, speed=1.0):
        # Reconstruct list for decode
        refer_list = [refer_spec]
        sv_emb_list = [sv_emb] if sv_emb is not None else None
        
        return self.vq_model.decode(
            pred_semantic, text_seq, refer_list, sv_emb=sv_emb_list, noise_scale=noise_scale, speed=speed
        )

class VQEncoder(nn.Module):
    def __init__(self, vq_model):
        super().__init__()
        self.vq_model = vq_model
    
    def forward(self, ssl_content):
        # ssl_content: [1, 768, T]
        codes = self.vq_model.extract_latent(ssl_content)
        # codes: [1, 1, T] (indices)
        return codes

class SpectrogramWrapper(nn.Module):
    def __init__(self, filter_length, hop_length, win_length, sampling_rate):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.register_buffer("hann_window", torch.hann_window(win_length))

    def forward(self, y):
        # y: [1, T] audio waveform
        if torch.min(y) < -1.2:
            print("min value is ", torch.min(y))
        if torch.max(y) > 1.2:
            print("max value is ", torch.max(y))

        n_fft = self.filter_length
        hop_size = self.hop_length
        win_size = self.win_length

        # Convert to float32 for STFT (TensorRT requires float32 input for STFT)
        y_stft = y.to(torch.float32)

        # Pad audio for STFT
        y_padded = torch.nn.functional.pad(
            y_stft.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
        )
        y_padded = y_padded.squeeze(1)

        # Compute STFT
        spec = torch.stft(
            y_padded,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=self.hann_window.to(torch.float32),
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )

        # Compute magnitude spectrum
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-8)
        return spec

class SVEmbeddingWrapper(nn.Module):
    """
    Wrapper for SV model compute_embedding3 to enable ONNX export.
    """
    def __init__(self, sv_model):
        super().__init__()
        self.embedding_model = sv_model
        self.num_mel_bins = 80
        self.sample_frequency = 16000.0
        self.frame_length = 25.0  # ms
        self.frame_shift = 10.0    # ms
        self.dither = 0.0
        self.low_freq = 20.0
        self.high_freq = 0.0  # 0 means Nyquist
        self.window_type = "povey"
        self.remove_dc_offset = True
        self.preemphasis_coefficient = 0.97
        self.energy_floor = 1.0
        self.raw_energy = True

        # Pre-compute window and mel filter bank (ONNX compatible)
        self.window_size = int(self.sample_frequency * self.frame_length * 0.001)
        self.window_shift = int(self.sample_frequency * self.frame_shift * 0.001)
        self.padded_window_size = 2 ** ((self.window_size - 1).bit_length())
        num_fft_bins = self.padded_window_size // 2 + 1

        # Pre-compute Povey window (Hanning window^0.85)
        self.register_buffer("window", torch.hann_window(self.window_size) ** 0.85)

        self.register_buffer("mel_filterbank", self._create_mel_filterbank_kaldi(num_fft_bins))

    def _create_mel_filterbank_kaldi(self, num_fft_bins):
        """
        Create mel filter bank matching Kaldi implementation.
        Kaldi uses triangular mel filters with specific edge frequencies.
        """
        import math

        high_freq = self.high_freq
        if high_freq <= 0.0:
            high_freq = self.sample_frequency / 2.0

        # Mel scale conversion functions (matching Kaldi)
        def mel_scale(freq):
            return 1127.0 * math.log(1.0 + freq / 700.0)

        def inverse_mel_scale(mel_freq):
            return 700.0 * (math.exp(mel_freq / 1127.0) - 1.0)

        # Calculate mel frequencies
        mel_low_freq = mel_scale(self.low_freq)
        mel_high_freq = mel_scale(high_freq)

        # Divide by num_bins+1 due to end-effects where bins spread to sides
        mel_freq_delta = (mel_high_freq - mel_low_freq) / (self.num_mel_bins + 1)

        # Create mel filter bank
        bins = torch.zeros(self.num_mel_bins, num_fft_bins)

        # FFT bin width
        fft_bin_width = self.sample_frequency / self.padded_window_size

        # For each mel bin
        for i in range(self.num_mel_bins):
            # Calculate left, center, right mel frequencies
            left_mel = mel_low_freq + i * mel_freq_delta
            center_mel = mel_low_freq + (i + 1.0) * mel_freq_delta
            right_mel = mel_low_freq + (i + 2.0) * mel_freq_delta

            # Convert to Hz
            left_hz = inverse_mel_scale(left_mel)
            center_hz = inverse_mel_scale(center_mel)
            right_hz = inverse_mel_scale(right_mel)

            # Calculate which FFT bins these correspond to
            left_bin = int(round(left_hz / fft_bin_width))
            center_bin = int(round(center_hz / fft_bin_width))
            right_bin = int(round(right_hz / fft_bin_width))

            # Create triangular filter
            # Left slope: from left_bin to center_bin
            if center_bin > left_bin:
                bins[i, left_bin:center_bin] = torch.linspace(0, 1, center_bin - left_bin)

            # Right slope: from center_bin to right_bin
            if right_bin > center_bin:
                bins[i, center_bin:right_bin] = torch.linspace(1, 0, right_bin - center_bin)

        return bins

    def forward(self, wav):
        # wav: [B, T] audio waveform at 16kHz
        B = wav.shape[0]
        device = wav.device
        dtype = wav.dtype

        # Convert to float32 for STFT (TensorRT requires float32 input for STFT)
        wav_stft = wav.to(torch.float32)

        # Use STFT directly (ONNX compatible) to extract frames and compute FFT
        # STFT will handle framing, windowing, and FFT in one operation
        stft_result = torch.stft(
            wav_stft,
            n_fft=self.padded_window_size,
            hop_length=self.window_shift,
            win_length=self.window_size,
            window=self.window.to(device=device, dtype=torch.float32),
            center=False,  # Kaldi doesn't center
            normalized=False,
            onesided=True,
            return_complex=False
        )  # [B, num_freq_bins, num_frames, 2]

        # Extract real and imag parts
        real_part = stft_result[:, :, :, 0]  # [B, num_freq_bins, num_frames]
        imag_part = stft_result[:, :, :, 1]  # [B, num_freq_bins, num_frames]

        # Compute power spectrum: [B, num_freq_bins, num_frames]
        spectrum = real_part.pow(2) + imag_part.pow(2)

        # Transpose to [B, num_frames, num_freq_bins] for mel filter bank
        spectrum = spectrum.transpose(1, 2)  # [B, num_frames, num_freq_bins]

        # Apply mel filter bank
        # spectrum: [B, num_frames, num_fft_bins]
        # mel_filterbank: [num_mel_bins, num_fft_bins]
        # result: [B, num_frames, num_mel_bins]
        mel_energies = torch.matmul(spectrum, self.mel_filterbank.T)

        # Log compression
        epsilon_tensor = torch.tensor(torch.finfo(torch.float32).eps, device=device, dtype=torch.float32)
        mel_energies = torch.clamp_min(mel_energies, epsilon_tensor).log()

        # mel_energies: [B, T, F] where F=80 (already in correct format)
        feat = mel_energies

        # Pass through ERes2NetV2 forward3
        # forward3 returns [B, 20480] regardless of input length
        sv_emb = self.embedding_model.forward3(feat)

        return sv_emb

def hparams_to_dict(hp):
    if hasattr(hp, "__dict__"):
        return {k: hparams_to_dict(v) for k, v in hp.__dict__.items()}
    elif isinstance(hp, dict):
        return {k: hparams_to_dict(v) for k, v in hp.items()}
    elif isinstance(hp, (list, tuple)):
        return [hparams_to_dict(v) for v in hp]
    else:
        return hp

def export_onnx(args):
    torch.set_grad_enabled(False)
    device = "cpu" # Export on CPU usually safer for dynamic axes

    # 初始化校验器（如果启用）
    validator = None
    if args.validate:
        print(f"\n{'='*60}")
        print("启用ONNX导出精度校验")
        print(f"{'='*60}\n")
        validator = onnx_validation.ONNXValidator(
            output_dir=args.output_dir,
            onnx_device=args.validation_device
        )

    print("Loading models...")
    # SSL
    cnhubert.cnhubert_base_path = args.cnhubert_base_path
    ssl_model = cnhubert.get_model()
    ssl_model = ssl_model.to(device)
    ssl_model.eval()
    
    # BERT
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(args.bert_path)
    bert_model = bert_model.to(device)
    bert_model.eval()

    # GPT
    dict_s1 = torch.load(args.gpt_path, map_location="cpu")
    config = dict_s1["config"]
    t2s_model = Text2SemanticLightningModule(config, "output", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    t2s_model.eval()
    
    # SoVITS
    dict_s2 = load_sovits_new(args.sovits_path)
    hps = dict_s2["config"]
    # Handle DictToAttrRecursive logic manually or using the class if available. 
    class AttrDict(object):
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, AttrDict(v))
                else:
                    setattr(self, k, v)
    
    hps_obj = AttrDict(hps)
    hps_obj.model.semantic_frame_rate = "25hz"
    _, model_version, _ = get_sovits_version_from_path_fast(args.sovits_path)
    hps_obj.model.version = model_version
    
    # Update the original hps dict as well to ensure SynthesizerTrn gets the right values
    hps["model"]["version"] = model_version
    hps["model"]["semantic_frame_rate"] = "25hz"
    
    vq_model = SynthesizerTrn(
        hps_obj.data.filter_length // 2 + 1,
        hps_obj.train.segment_size // hps_obj.data.hop_length,
        n_speakers=hps_obj.data.n_speakers,
        **hps["model"]
    )
    vq_model.eval()
    vq_model.load_state_dict(dict_s2["weight"], strict=False)
    
    # Patch EuclideanCodebook.init_embed_ to avoid export error
    for name, module in vq_model.named_modules():
        if "EuclideanCodebook" in module.__class__.__name__:
            import types
            module.init_embed_ = types.MethodType(lambda self, data: None, module)

    # SV Model (for speaker embedding)
    sv_path = os.environ.get("SV_MODEL_PATH", "pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt")
    sys.path.append(os.path.join(os.path.dirname(__file__), "GPT_SoVITS", "eres2net"))
    from ERes2NetV2 import ERes2NetV2
    pretrained_state = torch.load(sv_path, map_location="cpu", weights_only=False)
    sv_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
    sv_model.load_state_dict(pretrained_state)
    sv_model.eval()
    sv_model = sv_model.to(device)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Exporting to {output_dir}...")
    print("Exporting SSL...")
    class SSLWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, audio):
            # HubertModel returns [B, T, C], we need [B, C, T] for VQEncoder
            return self.model(audio).last_hidden_state.transpose(1, 2)

    ssl_wrapper = SSLWrapper(ssl_model.model)
    # Input: [1, T] audio 16k
    dummy_audio = torch.randn(1, 16000 * 2)
    torch.onnx.export(
        ssl_wrapper,
        (dummy_audio,),
        f"{output_dir}/ssl.onnx",
        input_names=["audio"],
        output_names=["last_hidden_state"],
        dynamic_axes={"audio": {1: "time"}, "last_hidden_state": {2: "time"}},
        opset_version=18,
        dynamo=False
    )

    # 校验SSL模型
    if validator:
        validator.validate_model(
            model_name="SSL",
            onnx_path=f"{output_dir}/ssl.onnx",
            pytorch_model=ssl_wrapper,
            dummy_inputs={"audio": dummy_audio},
            output_names=["last_hidden_state"],
            rtol=1e-3,
            atol=1e-5
        )
    
    print("Exporting BERT...")
    # Input: input_ids [1, T], attention_mask [1, T], token_type_ids [1, T]
    dummy_input_ids = torch.randint(0, 100, (1, 20), dtype=torch.long)
    dummy_attn_mask = torch.ones(1, 20, dtype=torch.long)
    dummy_token_type = torch.zeros(1, 20, dtype=torch.long)
    # Wrapper for BERT to return only what we need
    class BERTWrapper(nn.Module):
        def __init__(self, bert):
            super().__init__()
            self.bert = bert
        def forward(self, input_ids, attention_mask, token_type_ids):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            return torch.cat(outputs.hidden_states[-3:-2], -1)

    bert_wrapper = BERTWrapper(bert_model)
    torch.onnx.export(
        bert_wrapper,
        (dummy_input_ids, dummy_attn_mask, dummy_token_type),
        f"{output_dir}/bert.onnx",
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["hidden_states"],
        dynamic_axes={"input_ids": {1: "seq_len"}, "attention_mask": {1: "seq_len"}, "token_type_ids": {1: "seq_len"}, "hidden_states": {1: "seq_len"}},
        opset_version=20,
        dynamo=False
    )

    # 校验BERT模型
    if validator:
        validator.validate_model(
            model_name="BERT",
            onnx_path=f"{output_dir}/bert.onnx",
            pytorch_model=bert_wrapper,
            dummy_inputs={
                "input_ids": dummy_input_ids,
                "attention_mask": dummy_attn_mask,
                "token_type_ids": dummy_token_type
            },
            output_names=["hidden_states"],
            rtol=1e-3,
            atol=1e-5
        )
    
    print("Exporting VQEncoder...")
    vq_enc = VQEncoder(vq_model)
    # ssl_content: [1, 768, T]
    dummy_ssl = torch.randn(1, 768, 100)
    torch.onnx.export(
        vq_enc,
        (dummy_ssl,),
        f"{output_dir}/vq_encoder.onnx",
        input_names=["ssl_content"],
        output_names=["codes"],
        dynamic_axes={"ssl_content": {2: "time"}, "codes": {2: "time"}},
        opset_version=20,
        dynamo=False
    )

    # 校验VQEncoder模型
    if validator:
        validator.validate_model(
            model_name="VQEncoder",
            onnx_path=f"{output_dir}/vq_encoder.onnx",
            pytorch_model=vq_enc,
            dummy_inputs={"ssl_content": dummy_ssl},
            output_names=["codes"],
            rtol=1e-3,
            atol=1e-5
        )

    print("Exporting GPT Encoder...")
    gpt_enc = GPTEncoder(t2s_model, max_len=args.max_len)
    # Dummies
    phoneme_ids = torch.randint(0, 512, (1, 50), dtype=torch.long)
    phoneme_ids_len = torch.tensor([50], dtype=torch.long)
    prompts = torch.randint(0, 1024, (1, 20), dtype=torch.long)
    bert_feature = torch.randn(1, 1024, 50)
    
    dynamic_axes_gpt = {
        "phoneme_ids": {1: "text_len"},
        "prompts": {1: "prompt_len"},
        "bert_feature": {2: "text_len"},
        "k_cache": {1: "batch_size"},
        "v_cache": {1: "batch_size"},
        "x_len": {0: "one"},
        "y_len": {0: "one"},
    }
    
    torch.onnx.export(
        gpt_enc,
        (phoneme_ids, prompts, bert_feature),
        f"{output_dir}/gpt_encoder.onnx",
        input_names=["phoneme_ids", "prompts", "bert_feature"],
        output_names=["topk_values", "topk_indices", "k_cache", "v_cache", "x_len", "y_len"],
        dynamic_axes=dynamic_axes_gpt,
        opset_version=20,
        dynamo=False
    )
    
    print("Exporting GPT Step...")
    # Get outputs from encoder to feed to step
    with torch.no_grad():
        topk_v_dummy, topk_i_dummy, k_cache, v_cache, x_len, y_len = gpt_enc(phoneme_ids, prompts, bert_feature)
    
    gpt_step = GPTStep(t2s_model)
    idx = torch.tensor([0], dtype=torch.long)
    # samples input for step is indices [B, 1]
    samples = torch.randint(0, 1024, (1, 1), dtype=torch.long)
    
    dynamic_axes_step = {
        "k_cache": {1: "batch_size"},
        "v_cache": {1: "batch_size"},
        "x_len": {0: "one"},
        "y_len": {0: "one"},
        "idx": {0: "one"},
    }
    
    torch.onnx.export(
        gpt_step,
        (samples, k_cache, v_cache, x_len, y_len, idx),
        f"{output_dir}/gpt_step.onnx",
        input_names=["samples", "k_cache", "v_cache", "x_len", "y_len", "idx"],
        output_names=["topk_values", "topk_indices", "k_cache_new", "v_cache_new"],
        dynamic_axes=dynamic_axes_step,
        opset_version=20,
        dynamo=False
    )
    
    print("Exporting SoVITS...")
    sovits_wrapper = SoVITS(vq_model, model_version)
    # Dummies
    # pred_semantic: [1, 1, T_sem] -> [1, 1, 150]
    pred_semantic = torch.randint(0, 1024, (1, 1, 150), dtype=torch.long)
    text_seq = torch.randint(0, 512, (1, 50), dtype=torch.long)
    # refer_spec: [1, C, T_ref] -> [1, 1025, 200]
    refer_spec = torch.randn(1, 1025, 200)
    noise_scale = torch.tensor([0.5], dtype=torch.float32)
    speed = torch.tensor([1.0], dtype=torch.float32)
    
    args_sovits = [pred_semantic, text_seq, refer_spec]
    input_names = ["pred_semantic", "text_seq", "refer_spec"]
    
    if "Pro" in model_version:
        sv_emb = torch.randn(1, 20480)
        args_sovits.append(sv_emb)
        input_names.append("sv_emb")

    args_sovits.extend([noise_scale, speed])
    input_names.extend(["noise_scale", "speed"])
    
    dynamic_axes_sovits = {
        "pred_semantic": {2: "sem_len"},
        "text_seq": {1: "text_len"},
        "refer_spec": {2: "ref_len"},
    }
    
    torch.onnx.export(
        sovits_wrapper,
        tuple(args_sovits),
        f"{output_dir}/sovits.onnx",
        input_names=input_names,
        output_names=["audio"],
        dynamic_axes=dynamic_axes_sovits,
        opset_version=20,
        dynamo=False
    )

    # Export SpectrogramWrapper
    print("Exporting Spectrogram...")
    spec_wrapper = SpectrogramWrapper(
        filter_length=hps_obj.data.filter_length,
        hop_length=hps_obj.data.hop_length,
        win_length=hps_obj.data.win_length,
        sampling_rate=hps_obj.data.sampling_rate
    )
    # Input: [1, T] audio waveform at sampling_rate
    dummy_wav = torch.randn(1, 48000)
    torch.onnx.export(
        spec_wrapper,
        (dummy_wav,),
        f"{output_dir}/spectrogram.onnx",
        input_names=["audio"],
        output_names=["spectrogram"],
        dynamic_axes={"audio": {1: "time"}, "spectrogram": {2: "time"}},
        opset_version=20,
        dynamo=False
    )

    # 校验Spectrogram模型
    if validator:
        validator.validate_model(
            model_name="Spectrogram",
            onnx_path=f"{output_dir}/spectrogram.onnx",
            pytorch_model=spec_wrapper,
            dummy_inputs={"audio": dummy_wav},
            output_names=["spectrogram"],
            rtol=1e-4,
            atol=1e-6
        )

    # Export SVEmbeddingWrapper
    print("Exporting SV Embedding...")
    sv_wrapper = SVEmbeddingWrapper(sv_model)
    # Input: [B, T] audio waveform at 16kHz
    dummy_wav_16k = torch.randn(1, 16000 * 3)
    torch.onnx.export(
        sv_wrapper,
        (dummy_wav_16k,),
        f"{output_dir}/sv_embedding.onnx",
        input_names=["audio"],
        output_names=["sv_embedding"],
        dynamic_axes={"audio": {1: "time"}},
        opset_version=20,
        dynamo=False
    )

    # 校验SVEmbedding模型
    if validator:
        validator.validate_model(
            model_name="SVEmbedding",
            onnx_path=f"{output_dir}/sv_embedding.onnx",
            pytorch_model=sv_wrapper,
            dummy_inputs={"audio": dummy_wav_16k},
            output_names=["sv_embedding"],
            rtol=1e-3,
            atol=1e-5
        )

    config_dict = hparams_to_dict(hps)
    config_dict["symbol_to_id"] = _symbol_to_id_v2
    config_dict["spectrogram"] = {
        "filter_length": hps_obj.data.filter_length,
        "hop_length": hps_obj.data.hop_length,
        "win_length": hps_obj.data.win_length,
        "sampling_rate": hps_obj.data.sampling_rate
    }
    config_dict["data"]['max_len'] = args.max_len
    config_dict["sv_embedding"] = {
        "embedding_size": 20480 if "Pro" in model_version else 512,
        "model_version": model_version
    }
    with open(f"{output_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)

    print(f"Export complete! Config saved to {output_dir}/config.json")

    # 校验摘要
    if validator:
        validator.print_summary()
        validator.save_report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS ONNX Export")
    parser.add_argument("--gpt_path", required=True)
    parser.add_argument("--sovits_path", required=True)
    parser.add_argument("--cnhubert_base_path", default="pretrained_models/chinese-hubert-base")
    parser.add_argument("--bert_path", default="pretrained_models/chinese-roberta-wwm-ext-large")
    parser.add_argument("--sv_path", default=None, help="Path to SV model (default: pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt)")
    parser.add_argument("--max_len", type=int, default=2000, help="Pre-allocated KV cache length")
    parser.add_argument("--output_dir", default="onnx_export", help="Output directory for ONNX models")
    parser.add_argument("--validate", action="store_true", help="Enable ONNX export accuracy validation")
    parser.add_argument("--validation_device", default="cpu", choices=["cpu", "cuda"], help="Device for ONNX validation (default: cpu)")

    args = parser.parse_args()

    # Set SV model path if provided
    if args.sv_path:
        os.environ["SV_MODEL_PATH"] = args.sv_path

    export_onnx(args)