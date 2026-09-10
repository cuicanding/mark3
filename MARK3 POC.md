# AI大促队长 后端 Demo 设计（感知+对话+语音输出，最小依赖版）
> 目标：一个 Python 后端 + 一个网页 = 能跑在 Mac mini 上的可对话 demo。  
验收：浏览器打开 → 允许麦克风摄像头 → 说话"小队长，当前支付成功率是多少" → 屏幕出现识别文字 + 听到队长语音回答。  
所有 API 均已在沙箱实测（sherpa-onnx 1.13.7 / opencv 4.11），非纸上谈兵。
>

---

## 0. 核心设计决策（为什么这样最轻）
| 决策 | 理由 |
| --- | --- |
| **浏览器就是感知设备**（getUserMedia 采麦克风+摄像头） | 免掉 macOS 麦克风权限/PortAudio/多设备采集的全部坑；demo 在任何机器浏览器里就能跑；现场形态再切本地采集也不动协议 |
| **视觉用 OpenCV 自带 YuNet+SFace**，不装 InsightFace | cv2 4.5.4+ 内置，零额外依赖（已实测跑通）；认 10 个熟人精度足够；InsightFace 留作二阶段精度升级 |
| **语音全家桶 sherpa-onnx 一个包** | VAD+ASR（SenseVoice）+TTS（vits-melo）+KWS+声纹全在；已实测 VAD 引擎与 SenseVoice 工厂可用 |
| **对话引擎先用规则模板**，LLM 网关做成可选注入 | Demo 不等网关审批；接口按 OpenAI 兼容留好，网关批了一行配置切换 |
| **全链路 MOCK 开关**（环境变量 MOCK=1） | 模型没下好/无摄像头环境也能演示文字对话，D1 第一天就能看到效果 |
| Web 页面由后端直接托管 | 打开 [http://localhost:8000](http://localhost:8000) 即用，不存在"前端工程" |


**依赖共 6 个**：`fastapi uvicorn websockets opencv-python-headless numpy sherpa-onnx`（+可选 huggingface_hub 下载模型）

---

## 1. 工程结构
```plain
ai-captain-demo/
├── server.py           # FastAPI：/ 页面、WS /ws、REST /say /state
├── pipeline.py         # VAD/ASR/TTS/人脸 封装（惰性加载，全部可 MOCK）
├── brain.py            # 规则对话引擎 + LLM 网关可选项 + 会话上下文
├── norm.py             # 数字/百分比→中文读法（已单测通过）
├── static/index.html   # 对讲页面：采音→WS→显示→播放
├── face_db.json        # 注册的人脸向量 {"name": [[128维],...]}（只存向量）
├── register_face.py    # 对着摄像头拍 3 张注册自己（一条命令）
└── requirements.txt
```

## 2. 数据流（一张图）
```plain
浏览器                    Python 后端
│ getUserMedia 麦克风        │
│ 16k单声道PCM16 ─binary──▶ │ VAD切段 → SenseVoice ASR → 文本
│ 定时JPEG快照  ──json────▶ │ YuNet人脸检测 → SFace 128维 → 脸库余弦比对
│                           │        ↓
│                           │ brain.py：意图识别(规则) → 模板回复(带身份/指标mock)
│                           │        ↓
│ ◀─json {type:reply}────── │ 回复文本（字幕显示）
│ ◀─binary wav(实时TTS)──── │ vits-melo 合成（无模型时降级浏览器语音）
```

## 3. WS 协议（前后端唯一契约）
```plain
客户端→服务端  binary   ：PCM16 16kHz 单声道音频块（转int16后直接accept）
客户端→服务端  json    ：{"type":"shot","img":"<base64 jpeg>"}   # 每2秒一张
                      ：{"type":"text","text":"..."}            # MOCK文字输入
服务端→客户端  json    ：{"type":"asr","text":"..."}             # 识别结果
                      ：{"type":"identity","name":"...","score":0.87}
                      ：{"type":"reply","text":"...","person":"..."}
                      ：{"type":"tts_mode","mode":"server|browser"}
服务端→客户端  binary   ：WAV 字节（先发 {"type":"audio","bytes":N} 再发N字节）
```

## 4. 模块职责
### pipeline.py（感知+发声，全部可 MOCK）
+ `AudioPipe`：silero VAD（阈值0.5，500ms静音切段）→ SenseVoice-Small(int8) ASR  
⚠ 实测注意：`VadModelConfig().sample_rate=16000` 必须设置；SenseVoice 工厂是 `OfflineRecognizer.from_sense_voice(...)`
+ `TtsPipe`：vits-melo 中文（HF: `k2-fsa/sherpa-onnx-melo-tts-zh_en`，取 model.onnx+lexicon+tokens+dict）  
⚠ 实测注意：**没有** `OfflineTts.from_vits_melo`，用 `OfflineTts(OfflineTtsConfig(vits=OfflineTtsVitsModelConfig(...), ...))`；sid 选中文音色
+ `FacePipe`：YuNet 检测（`FaceDetectorYN.create(模型路径,"",(w,h),0.6)`，**inputSize=实际帧尺寸**，实测踩过）→ SFace 特征 → 与 face_db 余弦，阈值 0.363，连续 3 帧确认才发 identity
+ 全部模块 `MOCK=1` 时返回固定结果，保证零模型环境可跑



### brain.py（认知）
+ 意图规则表（正则→回复模板）：指标类/值班类/问候类/身份类/闲聊兜底，回复全部带 mock 数据
+ 会话上下文：保留最近 6 轮；识别到 identity 时注入"当前对话者：XXX"，问候与称呼随身份变化
+ LLM 注入点：`brain.LLM_BASE_URL/LLM_API_KEY` 配置后，规则未命中自动转 OpenAI 兼容 chat 接口（内部网关/ollama 都行），系统提示词限制"只回答大促保障相关问题"



### server.py（编排）
+ `WS /ws`：收音频/快照/文字，回识别/身份/回复/音频；事件全部落盘 `events.jsonl`（回放审计）
+ `POST /say`：遥控直输（curl 一句话让队长播报，演示后门）
+ 静态托管 index.html；`GET /state` 看当前会话状态



### static/index.html（表现层，单页）
+ 两栏：左边状态卡（身份/识别文本/回复字幕）+ 文字输入框；右边人脸注册按钮 + 音频可视化小波形
+ 音频：AudioWorklet/ScriptProcessor 采 48k → 重采样 16k → WS；播放直接 `audio.play(blob)`
+ TTS 降级开关：收到 `tts_mode:browser` 时用 `speechSynthesis` 念（模型未就绪时Demo不断声）



## 5. 启动步骤（真机 D1/D2 执行）
```bash
# 0) 环境
conda create -n captain python=3.11 -y && conda activate captain
pip install fastapi uvicorn websockets opencv-python-headless numpy sherpa-onnx huggingface_hub

# 1) MOCK 模式先跑通链路（D1 第一天就能看到效果）
MOCK=1 uvicorn server:app --port 8000   # 打开 http://localhost:8000 文字对话

# 2) 下模型（HF 网络慢用 hf-mirror.com）
export HF_ENDPOINT=https://hf-mirror.com
python -c "
from huggingface_hub import snapshot_download
snapshot_download('k2-fsa/sherpa-onnx-sense-voice-zh-en-ja-ku-yue-2024-07-17', local_dir='models/asr')
snapshot_download('csukuangfj/vits-melo-tts-zh_en', local_dir='models/tts')
"
# VAD/YuNet/SFace 三个小模型（40MB）下载脚本照 sherpa-onnx/opencv_zoo 官方 README，或内网镜像

# 3) 真语音+真识别
uvicorn server:app --port 8000
# 4) 注册自己的人脸
python register_face.py --name 存孝   # 对着摄像头拍3张 → face_db.json
```

## 6. 验收标准（每档都独立可演示）
| 档位 | 操作 | 通过标准 |
| --- | --- | --- |
| L1 MOCK | MOCK=1 打开页面，文字输入"小队长，当前支付成功率" | 2秒内返回模板回复+mock数据 |
| L2 语音 | 正常启动，对麦克风说同一句话 | 页面显示识别文本，听到队长语音回答 |
| L3 视觉 | 注册后站到摄像头前说话 | identity 事件命中，回复带称呼"存孝" |
| L4 遥控 | `curl -X POST localhost:8000/say -d '{"text":"现在应该安静"}'` | 页面播报该句 |
| L5 兜底 | 删掉 TTS 模型再启动 | 自动降级浏览器语音，对话不断 |


## 7. 已知坑（沙箱实测踩过，直接抄结论）
1. **VAD 构造必须设** `vad_cfg.sample_rate=16000`，否则引擎静默不工作
2. **YuNet 的 inputSize 必须等于实际帧宽高**（创建后 detect 前不要改尺寸），否则抛 Size not match
3. **TTS 没有 from_vits_melo 工厂**（1.13.7 实测），用 OfflineTtsVitsModelConfig 组装；GeneratedAudio 在顶层命名空间不存在，取返回对象的 .samples/.sample_rate 自己写 wav
4. SFace 对纯噪声图余弦会虚高（分布外），**真人比对才用 0.363 阈值**；注册时存 3 张向量取平均更稳
5. opencv 必装 headless 版（带 GUI 版在 mini 上会拖 Qt 依赖）
6. 浏览器 getUserMedia 需要 localhost 或 HTTPS；局域网 MacBook 采音时用 `--host 0.0.0.0` + Chrome 加站点豁免，或干脆 USB 线直连



## 8. 与正式版的衔接
+ 本 demo 的 WS 协议 = 设计文档 §3.2 事件协议的子集，正式版大屏 v7 直接消费同一协议
+ brain.py 的规则表 → 正式版的说话权闸门+台本；LLM 注入点 → 内部网关
+ FacePipe → 感知模块（InsightFace 升级位），register_face.py → 志愿者注册流程
+ 演示完成即可进 W1 正式开发，代码全部可搬

