"""
example.py — 独立推理示例，脱离原始项目运行

依赖:
    pip install onnxruntime transformers numpy

用法:
    python example.py --bert_dir /path/to/chinese-roberta-wwm-ext-large \
                      --onnx_dir /path/to/this/directory \
                      --text "今天心情很好，但有点担心明天的考试"

说明:
    emotion_head.onnx 接收 BERT 的 hidden_states 和 attention_mask，
    直接输出 tag_ids（已含 Viterbi 解码），无需任何额外库。
    BERT 本身通过 onnxruntime + transformers tokenizer 调用，
    如果你的项目已有 BERT 的 ONNX，可替换 _bert_encode 部分。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModel
import torch

MAX_LEN = 128
NUM_FUSE_LAYERS = 4  # BERT 最后几层 hidden states 作为输入


def _bert_encode(bert_model, input_ids, attention_mask, token_type_ids, num_fuse_layers):
    """用 transformers 跑 BERT，返回 hidden_states numpy array。"""
    with torch.no_grad():
        out = bert_model(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            token_type_ids=torch.from_numpy(token_type_ids),
            output_hidden_states=(num_fuse_layers > 1),
        )
    if num_fuse_layers > 1:
        # 取最后 N 层，stack 为 (B, N, L, H)
        layers = [h.numpy() for h in out.hidden_states[-num_fuse_layers:]]
        return np.stack(layers, axis=1)
    else:
        return out.last_hidden_state.numpy()


def decode_spans(tag_ids, offset_mapping, text, id2label):
    """将 tag_ids 解码为字符级 span 列表。"""
    # 提取每个 token 的 emotion（去掉 BIOES 前缀，O 保持 O）
    emotion_seq = []
    for i, tid in enumerate(tag_ids):
        om = offset_mapping[i]
        if om[0] == 0 and om[1] == 0:
            emotion_seq.append("O")
            continue
        label = id2label.get(str(tid), "O")
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


def fill_neutral(text, spans):
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


def predict(text, tokenizer, bert_model, head_session, id2label, fill_gaps=True):
    # 1) Tokenize — 不 pad，和 PyTorch 推理路径保持一致
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        truncation=True,
        padding=False,
        return_offsets_mapping=True,
        return_tensors="np",
    )
    offset_mapping = enc.pop("offset_mapping")[0].tolist()
    input_ids      = enc["input_ids"].astype(np.int64)       # (1, seq_len)
    attention_mask = enc["attention_mask"].astype(np.int64)   # (1, seq_len)
    token_type_ids = enc.get("token_type_ids",
                             np.zeros_like(input_ids)).astype(np.int64)

    seq_len = input_ids.shape[1]

    # 2) BERT 编码（不 pad，保证 hidden_states 与 PyTorch 路径一致）
    hidden_states = _bert_encode(bert_model, input_ids, attention_mask,
                                 token_type_ids, NUM_FUSE_LAYERS)

    # 3) 零填充到 MAX_LEN 再喂给 ONNX head（head 导出时 L 固定为 MAX_LEN）
    pad_len = MAX_LEN - seq_len
    if NUM_FUSE_LAYERS > 1:
        # hidden_states: (1, N, seq_len, H)
        hidden_states = np.pad(hidden_states,
                               ((0,0), (0,0), (0,pad_len), (0,0)))
    else:
        # hidden_states: (1, seq_len, H)
        hidden_states = np.pad(hidden_states,
                               ((0,0), (0,pad_len), (0,0)))
    mask_padded = np.pad(attention_mask, ((0,0), (0,pad_len)))  # (1, MAX_LEN)

    # 4) Head 推理（含 Viterbi）
    tag_ids = head_session.run(
        ["tag_ids"],
        {"hidden_states": hidden_states.astype(np.float32),
          "attention_mask": mask_padded.astype(np.int64)},
    )[0][0].tolist()  # (MAX_LEN,)

    # 只取有效长度部分
    tag_ids = tag_ids[:seq_len]
    offset_mapping = offset_mapping[:seq_len]

    spans = decode_spans(tag_ids, offset_mapping, text, id2label)
    if fill_gaps:
        spans = fill_neutral(text, spans)
    return {"text": text, "spans": spans}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_dir", required=True,
                        help="BERT 模型目录（含 config.json / pytorch_model.bin）")
    parser.add_argument("--onnx_dir", default=str(Path(__file__).parent),
                        help="emotion_head.onnx 所在目录（默认为本文件所在目录）")
    parser.add_argument("--text", default="今天心情很好，但有点担心明天的考试")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    meta = json.loads((onnx_dir / "meta.json").read_text(encoding="utf-8"))
    id2label = meta["id2label"]

    print(f"[example] loading tokenizer & BERT from {args.bert_dir} ...")
    tokenizer  = AutoTokenizer.from_pretrained(args.bert_dir)
    bert_model = AutoModel.from_pretrained(args.bert_dir).eval()

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.device == "cuda" else ["CPUExecutionProvider"])
    head_session = ort.InferenceSession(
        str(onnx_dir / "emotion_head.onnx"), providers=providers
    )
    print(f"[example] emotion_head.onnx loaded")

    result = predict(args.text, tokenizer, bert_model, head_session, id2label)
    print(f"\n输入: {result['text']}")
    print("情感 spans:")
    for sp in result["spans"]:
        print(f"  [{sp['label']:8s}] {sp['text']!r:30s}  char {sp['start']}~{sp['end']}")


if __name__ == "__main__":
    main()
