# GPT-SoVITS API 使用文档

本项目提供了一个兼容 OpenAI `v1/audio/speech` 接口的 FastAPI 服务，支持流式输出和多角色配置。

## 快速启动

```bash
python api_server.py --host 0.0.0.0 --port 8000 --voices_config config/voices.json
```

### 命令行参数

| 参数                | 说明            | 默认值                                               |
|:------------------|:--------------|:--------------------------------------------------|
| `--host`          | 监听地址          | `0.0.0.0`                                         |
| `--port`          | 监听端口          | `8000`                                            |
| `--cnhubert_path` | CNHubert 模型路径 | `pretrained_models/chinese-hubert-base`           |
| `--bert_path`     | BERT 模型路径     | `pretrained_models/chinese-roberta-wwm-ext-large` |
| `--voices_config` | 语音配置文件路径      | `config/voices.json`                              |
| `--reload`        | 开发模式(热重载)     | `False`                                           |

环境变量支持：`CNHUBERT_PATH`, `BERT_PATH`, `VOICES_CONFIG`。

---

## 配置文件 (`voices.json`)

通过配置文件定义多个角色及其对应的模型路径和参考音频。

```json
{
  "my_character": {
    "gpt_path": "models/my_gpt.ckpt",
    "sovits_path": "models/my_sovits.pth",
    "ref_audio": "data/ref.wav",
    "ref_text": "参考音频的文本内容",
    "ref_lang": "zh",
    "description": "角色描述信息",
    "defaults": {
      "speed": 1.0,
      "top_k": 15,
      "top_p": 1.0,
      "temperature": 1.0,
      "pause_length": 0.3
    }
  }
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
