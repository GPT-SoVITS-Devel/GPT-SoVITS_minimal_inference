"""
情感预设构建工具

三种模式:
1. --mode extract:  从同一说话人的长音频中，用 emotion_predictor 自动挖掘情感片段
2. --mode synthetic: 从 neutral ge 出发，用"情感供体"音频做向量算术合成情感 ge
3. --mode manual:   直接指定每种情感的参考音频路径

原理 (synthetic 模式):
  ge 是一个 512 维向量，同时编码了音色和情感。
  如果我们有:
    - ge_donor_neutral: 供体说话人的 neutral ge
    - ge_donor_happy:   供体说话人的 happy ge
    - ge_target_neutral: 目标说话人的 neutral ge
  
  那么 "happy 方向" = ge_donor_happy - ge_donor_neutral
  目标说话人的 happy ge ≈ ge_target_neutral + scale * (ge_donor_happy - ge_donor_neutral)
  
  这类似于 word2vec 的向量算术，在风格迁移中是常见做法。
  scale 参数控制情感强度 (0.5=轻微, 1.0=正常, 1.5=夸张)
"""

import os
import sys
import json
import torch
import argparse
import numpy as np

cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "GPT_SoVITS"))


def build_synthetic_presets(args):
    """
    向量算术合成情感 ge。
    
    需要:
    - 目标说话人的 neutral 音频 (你的主音色)
    - 一组"情感供体"音频 (可以是任何人，只要情感表达清晰)
    - 供体的 neutral 音频
    
    输出:
    - emotion_ge_cache.pt: 预计算的 ge 向量字典
    """
    from run_emotion_inference import EmotionTTSInference

    engine = EmotionTTSInference(
        args.gpt_path, args.sovits_path,
        args.cnhubert_base_path, args.bert_path
    )

    # 提取目标说话人的 neutral ge
    target_ge_raw, target_sv_emb, _ = engine._extract_ge_from_audio(args.target_neutral)
    target_ge = engine._compute_full_ge(target_ge_raw, target_sv_emb)
    print(f"[target] neutral ge extracted, shape: {target_ge.shape}")

    # 加载供体配置
    with open(args.donor_config, "r", encoding="utf-8") as f:
        donor_config = json.load(f)
    # donor_config 格式:
    # {
    #   "neutral": "donor_audio/neutral.wav",
    #   "happy": "donor_audio/happy.wav",
    #   "sad": "donor_audio/sad.wav",
    #   ...
    # }

    # 提取供体的 neutral ge
    donor_neutral_path = donor_config.get("neutral")
    if not donor_neutral_path:
        raise ValueError("donor_config 必须包含 'neutral' 条目")

    donor_neutral_ge_raw, _, _ = engine._extract_ge_from_audio(donor_neutral_path)
    donor_neutral_ge = engine._compute_full_ge(donor_neutral_ge_raw, target_sv_emb)
    print(f"[donor] neutral ge extracted")

    scale = args.emotion_scale
    ge_cache = {"neutral": target_ge.cpu()}

    for label, wav_path in donor_config.items():
        if label == "neutral":
            continue

        donor_ge_raw, _, _ = engine._extract_ge_from_audio(wav_path)
        donor_ge = engine._compute_full_ge(donor_ge_raw, target_sv_emb)

        # 向量算术: target_emotion = target_neutral + scale * (donor_emotion - donor_neutral)
        emotion_direction = donor_ge - donor_neutral_ge
        synthetic_ge = target_ge + scale * emotion_direction

        ge_cache[label] = synthetic_ge.cpu()

        # 计算方向向量的模长，作为情感强度的参考
        direction_norm = emotion_direction.norm().item()
        print(f"  [✓] {label}: direction norm = {direction_norm:.4f}")

    # 保存
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    torch.save(ge_cache, args.output)
    print(f"\nSaved {len(ge_cache)} emotion ge vectors to {args.output}")
    print(f"Emotion scale: {scale}")
    print("提示: 如果情感太弱，增大 --emotion_scale; 太夸张则减小")


def build_extract_presets(args):
    """
    从长音频中自动挖掘情感片段。
    
    需要:
    - 说话人的长音频 + 对应文本
    - emotion_predictor 模型
    
    流程:
    1. 用 emotion_predictor 对文本做情感标注
    2. 找到每种情感最长/最典型的片段
    3. 根据时间戳切出对应音频片段
    4. 提取 ge 并保存
    """
    import librosa
    import soundfile as sf
    from pathlib import Path

    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("需要安装: pip install onnxruntime transformers")
        return

    # 加载 emotion predictor
    onnx_dir = Path(os.path.join(cwd, "GPT_SoVITS/emotion_predictor"))
    meta = json.loads((onnx_dir / "meta.json").read_text(encoding="utf-8"))
    id2label = meta["id2label"]

    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    bert_model = AutoModel.from_pretrained(args.bert_path).eval()
    head_session = ort.InferenceSession(
        str(onnx_dir / "emotion_head.onnx"),
        providers=["CPUExecutionProvider"]
    )

    # 导入 predict 函数
    sys.path.insert(0, str(onnx_dir))
    from example import predict

    # 读取文本列表 (每行: 起始秒|结束秒|文本)
    # 或者简单模式: 每行一句文本 (需要配合 ASR 时间戳)
    segments = []
    with open(args.transcript, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 3:
                start_sec, end_sec, text = float(parts[0]), float(parts[1]), parts[2]
                segments.append({"start": start_sec, "end": end_sec, "text": text})
            else:
                # 纯文本模式，只做情感分析不切音频
                segments.append({"text": line})

    # 对每个片段做情感预测
    emotion_segments = {e: [] for e in ["happy", "angry", "sad", "anxious", "doubt"]}

    for seg in segments:
        result = predict(seg["text"], tokenizer, bert_model, head_session, id2label)
        # 找主导情感
        emotion_chars = {}
        for span in result["spans"]:
            label = span["label"]
            if label != "neutral":
                chars = span["end"] - span["start"]
                emotion_chars[label] = emotion_chars.get(label, 0) + chars

        if emotion_chars:
            dominant = max(emotion_chars, key=emotion_chars.get)
            purity = emotion_chars[dominant] / len(seg["text"])
            seg["emotion"] = dominant
            seg["purity"] = purity
            if purity > 0.5:  # 情感纯度 > 50% 才采用
                emotion_segments[dominant].append(seg)

    # 为每种情感选最佳片段
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    has_timestamps = all("start" in s for s in segments if segments)
    selected = {}

    for emotion, segs in emotion_segments.items():
        if not segs:
            print(f"  [!] {emotion}: 未找到合适片段")
            continue

        # 按纯度排序，取最纯的
        segs.sort(key=lambda s: s["purity"], reverse=True)
        best = segs[0]
        print(f"  [✓] {emotion}: \"{best['text']}\" (purity: {best['purity']:.2f})")

        if has_timestamps and "start" in best:
            # 切出音频片段
            audio, sr = librosa.load(args.audio, sr=None,
                                     offset=best["start"],
                                     duration=best["end"] - best["start"])
            out_path = os.path.join(output_dir, f"{emotion}.wav")
            sf.write(out_path, audio, sr)
            selected[emotion] = out_path
        else:
            selected[emotion] = f"<需要手动切出: {best['text']}>"

    # 保存配置
    config_path = os.path.join(output_dir, "emotion_presets.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"\nSaved config to {config_path}")

    if not has_timestamps:
        print("\n注意: 输入文本没有时间戳，无法自动切音频。")
        print("请手动切出上述片段，或提供格式为 '起始秒|结束秒|文本' 的 transcript 文件。")


def build_manual_presets(args):
    """模式3: 直接从指定音频提取 ge + prompt_semantic 并保存"""
    from run_emotion_inference import EmotionTTSInference

    engine = EmotionTTSInference(
        args.gpt_path, args.sovits_path,
        args.cnhubert_base_path, args.bert_path
    )

    with open(args.emotion_config, "r", encoding="utf-8") as f:
        emotion_audio_map = json.load(f)

    target_ge_raw, target_sv_emb, target_prompt = engine._extract_ge_from_audio(args.target_neutral)
    target_ge = engine._compute_full_ge(target_ge_raw, target_sv_emb)

    ge_cache = {"neutral": target_ge.cpu()}

    for label, wav_path in emotion_audio_map.items():
        if label == "neutral":
            continue
        ge_raw, _, prompt_semantic = engine._extract_ge_from_audio(wav_path)
        ge_full = engine._compute_full_ge(ge_raw, target_sv_emb)
        ge_cache[label] = ge_full.cpu()
        print(f"  [✓] {label}: {wav_path}")

    torch.save(ge_cache, args.output)
    print(f"\nSaved {len(ge_cache)} emotion ge vectors to {args.output}")



def main():
    parser = argparse.ArgumentParser(
        description="情感预设构建工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

  # 模式1: 从长音频自动挖掘情感片段
  python tools/build_emotion_presets.py --mode extract \\
    --audio long_speech.wav --transcript transcript.txt \\
    --bert_path pretrained_models/chinese-roberta-wwm-ext-large \\
    --output_dir emotion_refs/

  # 模式2: 向量算术合成 (推荐，不需要目标说话人的情感音频)
  python tools/build_emotion_presets.py --mode synthetic \\
    --gpt_path ... --sovits_path ... \\
    --target_neutral pretrained_models/neutral.wav \\
    --donor_config config/donor_emotions.json \\
    --emotion_scale 0.8 \\
    --output emotion_ge_cache.pt

  # 模式3: 手动指定音频
  python tools/build_emotion_presets.py --mode manual \\
    --gpt_path ... --sovits_path ... \\
    --target_neutral pretrained_models/neutral.wav \\
    --emotion_config config/emotion_presets.json \\
    --output emotion_ge_cache.pt

donor_config.json 格式 (供体音频，可以是任何人):
  {
    "neutral": "donor_audio/neutral.wav",
    "happy":   "donor_audio/happy.wav",
    "sad":     "donor_audio/sad.wav",
    "angry":   "donor_audio/angry.wav",
    "anxious": "donor_audio/anxious.wav",
    "doubt":   "donor_audio/doubt.wav"
  }

transcript.txt 格式 (带时间戳):
  0.0|3.5|今天天气真好啊
  3.5|6.2|我好难过，什么都不想做
  ...
        """
    )

    parser.add_argument("--mode", required=True, choices=["extract", "synthetic", "manual"],
                        help="构建模式: extract/synthetic/manual")

    # 通用参数
    parser.add_argument("--gpt_path", help="GPT model path")
    parser.add_argument("--sovits_path", help="SoVITS model path")
    parser.add_argument("--cnhubert_base_path", default="pretrained_models/chinese-hubert-base")
    parser.add_argument("--bert_path", default="pretrained_models/chinese-roberta-wwm-ext-large")

    # extract 模式
    parser.add_argument("--audio", help="[extract] 长音频路径")
    parser.add_argument("--transcript", help="[extract] 文本/时间戳文件")
    parser.add_argument("--output_dir", default="emotion_refs/",
                        help="[extract] 输出目录")

    # synthetic 模式
    parser.add_argument("--target_neutral", help="[synthetic/manual] 目标说话人 neutral 音频")
    parser.add_argument("--donor_config", help="[synthetic] 供体情感音频配置 JSON")
    parser.add_argument("--emotion_scale", type=float, default=0.8,
                        help="[synthetic] 情感强度 (0.5=轻微, 0.8=适中, 1.0=正常, 1.5=夸张)")

    # manual 模式
    parser.add_argument("--emotion_config", help="[manual] 情感音频映射 JSON")

    # 输出
    parser.add_argument("--output", default="emotion_ge_cache.pt",
                        help="[synthetic/manual] 输出 ge 缓存文件")

    args = parser.parse_args()

    if args.mode == "extract":
        build_extract_presets(args)
    elif args.mode == "synthetic":
        build_synthetic_presets(args)
    elif args.mode == "manual":
        build_manual_presets(args)


if __name__ == "__main__":
    main()
