# GPT-SoVITS API 使用文档

本项目提供了一个兼容 OpenAI `v1/audio/speech` 接口的 FastAPI 服务，支持流式输出和多角色配置。

## 快速启动

### 1. 标准推理 (PyTorch)
```bash
python api_server.py --host 0.0.0.0 --port 8000 --voices_config config/voices.json
```

### 2. ONNX 推理
```bash
python api_server_onnx.py --host 0.0.0.0 --port 8001 --device cpu
```

### 3. TensorRT 推理
```bash
python api_server_trt.py --host 0.0.0.0 --port 8002 --device cuda
```

### 命令行参数

| 参数                | 说明                      | 默认值                                               | 适用服务            |
|:------------------|:------------------------|:--------------------------------------------------|:----------------|
| `--host`          | 监听地址                    | `0.0.0.0`                                         | 全部              |
| `--port`          | 监听端口                    | `8000`/`8001`/`8002`                              | 全部              |
| `--device`        | 推理设备 (`cpu`, `cuda`)    | `cpu`(ONNX) / `cuda`(TRT)                         | ONNX, TRT       |
| `--cnhubert_path` | CNHubert 模型路径           | `pretrained_models/chinese-hubert-base`           | 标准              |
| `--bert_path`     | BERT 模型路径               | `pretrained_models/chinese-roberta-wwm-ext-large` | 全部              |
| `--voices_config` | 语音配置文件路径                | `config/voices.json`                              | 全部              |
| `--reload`        | 开发模式(热重载)               | `False`                                           | 标准              |

---

## 配置文件 (`voices.json`)

根据所使用的推理引擎，在 `voices.json` 中配置对应的模型路径。

### 1. 标准引擎配置
```json
  "char_pytorch": {
    "gpt_path": "models/my_gpt.ckpt",
    "sovits_path": "models/my_sovits.pth",
    "ref_audio": "data/ref.wav",
    "ref_text": "参考文本",
    "ref_lang": "zh"
  }
```

### 2. ONNX 引擎配置
`onnx_path` 应指向包含 `ssl.onnx`, `gpt_encoder.onnx` 等文件的目录。
```json
  "char_onnx": {
    "onnx_path": "onnx_export/my_character",
    "ref_audio": "data/ref.wav",
    "ref_text": "参考文本",
    "ref_lang": "zh"
  }
```

### 3. TensorRT 引擎配置
`trt_path` 应指向包含 `.engine` 文件和 `config.json` 的目录。
```json
  "char_trt": {
    "trt_path": "onnx_export/my_character_trt",
    "ref_audio": "data/ref.wav",
    "ref_text": "参考文本",
    "ref_lang": "zh"
  }
```

---

## 接口说明

### 1. 语音合成 (OpenAI 兼容)

**POST** `/v1/audio/speech`

#### 请求参数 (JSON)

| 字段                | 类型     | 必选 | 说明                                             |
|:------------------|:-------|:---|:-----------------------------------------------|
| `input`           | string | 是  | 要合成的目标文本                                       |
| `voice`           | string | 否  | `voices.json` 中定义的角色名，默认为 `default`            |
| `response_format` | string | 否  | 支持 `wav` (默认), `pcm`                           |
| `speed`           | float  | 否  | 语速 (0.5 - 2.0)                                 |
| `temperature`     | float  | 否  | 采样温度                                           |
| `text_lang`       | string | 否  | 目标文本语言 (`zh`, `en`, `ja`, `ko`, `yue`, `auto`) |

#### 示例 (Curl)

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "你好，这是测试音频。",
    "voice": "my_character",
    "speed": 1.0
  }' \
  --output test.wav
```

### 2. 获取模型列表

**GET** `/v1/models`
返回当前加载的所有可用角色列表。

### 3. 热重载配置

**POST** `/v1/voices/reload`
在修改 `voices.json` 后无需重启服务，调用此接口即可生效。

---

## 注意事项

1. **首次加载延迟**：首次请求某个角色时，系统会自动加载对应的 GPT/SoVITS 模型到显存，可能会有几秒延迟。后续请求将使用缓存的引擎。
2. **绝对路径**：配置文件中的路径建议使用绝对路径，或相对于 `api_server.py` 的相对路径。
3. **流式输出**：接口采用流式响应 (`StreamingResponse`)，适合低延迟实时合成场景。
