import argparse
import os
import sys
import torch
from torch import nn
from torch.nn import functional as F
from GPT_SoVITS.process_ckpt import load_sovits_new, get_sovits_version_from_path_fast
from GPT_SoVITS.feature_extractor import cnhubert
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.module.models import SynthesizerTrn
from transformers import AutoModelForMaskedLM, AutoTokenizer
import logging
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
    def __init__(self, t2s_model):
        super().__init__()
        self.t2s_model = t2s_model

    def forward(self, phoneme_ids, phoneme_ids_len, prompts, bert_feature):
        # Wrapper for infer_first_stage
        # Returns: logits, k_cache (stacked), v_cache (stacked), x_len, y_len
        logits, k_cache, v_cache, x_len, y_len = self.t2s_model.model.infer_first_stage(
            phoneme_ids, phoneme_ids_len, prompts, bert_feature
        )
        
        # Stack caches: List[Tensor] -> Tensor [Layers, B, T, D]
        k_cache_stacked = torch.stack(k_cache, dim=0)
        v_cache_stacked = torch.stack(v_cache, dim=0)
        
        # Pad to max length for pre-allocation
        # Use a large enough constant or make it configurable. 1500 + max_prompt_len
        max_len = 2000
        k_cache_padded = F.pad(k_cache_stacked, (0, 0, 0, max_len - k_cache_stacked.shape[2]))
        v_cache_padded = F.pad(v_cache_stacked, (0, 0, 0, max_len - v_cache_stacked.shape[2]))
        
        return logits, k_cache_padded, v_cache_padded, x_len, y_len

class GPTStep(nn.Module):
    def __init__(self, t2s_model):
        super().__init__()
        self.t2s_model = t2s_model

    def forward(self, samples, k_cache, v_cache, x_len, y_len, idx):
        # Wrapper for infer_next_stage
        # k_cache, v_cache are stacked [Layers, B, T_max, D]
        
        # Unstack to list
        k_cache_list = [t for t in k_cache]
        v_cache_list = [t for t in v_cache]
        
        logits, k_cache_new, v_cache_new = self.t2s_model.model.infer_next_stage(
            samples, k_cache_list, v_cache_list, x_len, y_len, idx
        )
        
        # Stack again (they should still be the same tensors if updated in-place)
        k_cache_stacked = torch.stack(k_cache_new, dim=0)
        v_cache_stacked = torch.stack(v_cache_new, dim=0)
        
        return logits, k_cache_stacked, v_cache_stacked

class SoVITS(nn.Module):
    def __init__(self, vq_model, version):
        super().__init__()
        self.vq_model = vq_model
        self.version = version

    def forward(self, pred_semantic, text_seq, refer_spec, sv_emb=None):
        # Reconstruct list for decode
        refer_list = [refer_spec]
        sv_emb_list = [sv_emb] if sv_emb is not None else None
        
        return self.vq_model.decode(
            pred_semantic, text_seq, refer_list, sv_emb=sv_emb_list
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

def export_onnx(args):
    torch.set_grad_enabled(False)
    device = "cpu" # Export on CPU usually safer for dynamic axes
    
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

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Exporting to {output_dir}...")
    print("Exporting SSL...")
    # Input: [1, T] audio 16k
    dummy_audio = torch.randn(1, 16000*2)
    torch.onnx.export(
        ssl_model.model,
        (dummy_audio,),
        f"{output_dir}/ssl.onnx",
        input_names=["audio"],
        output_names=["last_hidden_state"],
        dynamic_axes={"audio": {1: "time"}, "last_hidden_state": {1: "time"}},
        opset_version=18,
        dynamo=False
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

    print("Exporting GPT Encoder...")
    gpt_enc = GPTEncoder(t2s_model)
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
    }
    
    torch.onnx.export(
        gpt_enc,
        (phoneme_ids, phoneme_ids_len, prompts, bert_feature),
        f"{output_dir}/gpt_encoder.onnx",
        input_names=["phoneme_ids", "phoneme_ids_len", "prompts", "bert_feature"],
        output_names=["logits", "k_cache", "v_cache", "x_len", "y_len"],
        dynamic_axes=dynamic_axes_gpt,
        opset_version=20,
        dynamo=False
    )
    
    print("Exporting GPT Step...")
    # Get outputs from encoder to feed to step
    with torch.no_grad():
        logits, k_cache, v_cache, x_len, y_len = gpt_enc(phoneme_ids, phoneme_ids_len, prompts, bert_feature)
    
    gpt_step = GPTStep(t2s_model)
    idx = torch.tensor([0], dtype=torch.long)
    # samples input for step is indices [B, 1]
    samples = torch.randint(0, 1024, (1, 1), dtype=torch.long)
    
    dynamic_axes_step = {
        "k_cache": {1: "batch_size"},
        "v_cache": {1: "batch_size"},
    }
    
    torch.onnx.export(
        gpt_step,
        (samples, k_cache, v_cache, x_len, y_len, idx),
        f"{output_dir}/gpt_step.onnx",
        input_names=["samples", "k_cache", "v_cache", "x_len", "y_len", "idx"],
        output_names=["logits", "k_cache_new", "v_cache_new"],
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
    
    args_sovits = (pred_semantic, text_seq, refer_spec)
    input_names = ["pred_semantic", "text_seq", "refer_spec"]
    
    dynamic_axes_sovits = {
        "pred_semantic": {2: "sem_len"},
        "text_seq": {1: "text_len"},
        "refer_spec": {2: "ref_len"},
    }
    
    if "Pro" in model_version:
        sv_emb = torch.randn(1, 20480)
        args_sovits += (sv_emb,)
        input_names.append("sv_emb")
    
    torch.onnx.export(
        sovits_wrapper,
        args_sovits,
        f"{output_dir}/sovits.onnx",
        input_names=input_names,
        output_names=["audio"],
        dynamic_axes=dynamic_axes_sovits,
        opset_version=20,
        dynamo=False
    )

    print("Export complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS ONNX Export")
    parser.add_argument("--gpt_path", required=True)
    parser.add_argument("--sovits_path", required=True)
    parser.add_argument("--cnhubert_base_path", default="pretrained_models/chinese-hubert-base")
    parser.add_argument("--bert_path", default="pretrained_models/chinese-roberta-wwm-ext-large")
    parser.add_argument("--output_dir", default="onnx_export", help="Output directory for ONNX models")
    
    args = parser.parse_args()
    export_onnx(args)