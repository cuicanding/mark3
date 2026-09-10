# AI 大促队长 · 后端 Demo（MVP v0.3）

按《MARK3 设计文档 v0.3》实现：P1 后台治理 + 感知升级已落地；P2/P3 关键接口与数据库结构已就位。

## 快速开始

```bash
cd ai-captain-demo
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# L1 MOCK：零模型，文字对话（第一天就能看到效果）
MOCK=1 .venv/bin/python -m uvicorn server:app --port 8000
# 打开 http://localhost:8000   对讲页
# 后台 http://localhost:8000/admin

# 下模型（约 700-800MB，含声纹模型；走 gh-proxy）
.venv/bin/python scripts/download_models.py

# L2-L5 真实模式：语音 + 人脸 + 服务端 TTS + 声纹锁
.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

## 新增能力（v0.3）

| 模块 | 新增内容 |
| ---- | -------- |
| **数据层** | `captain.db` SQLite：配置、值班表、台本、人格、模型 provider、技能 |
| **后台管理** | `/admin` 七页管理台：总览、人员、值班表、声音、台本、模型、对话 |
| **人脸精度** | 5 姿态注册 + 逐张质量门控 + max-cosine + top1-top2 margin ≥0.05 |
| **声纹锁** | WeSpeaker 中文说话人嵌入；VAD 段过声纹门才放行 ASR；PTT 通道豁免 |
| **收音模式** | PTT（默认）/ 声纹持续监听 / hybrid；页面可切 |
| **双档 TTS** | melo 现役；kokoro 接口位已预留（下载/配置后切换） |
| **台本播报** | CRUD + 简单 cron 调度 + 预生成音频接口 |
| **说话权闸门** | 播报内容 = 工具数据 + 模板组装；LLM 兜底走后台模型页配置 |

## 验收档位（POC 五档 + 新增）

| 档位 | 操作 | 通过标准 |
| ---- | ---- | -------- |
| L1 MOCK | `MOCK=1` 起服务，页面文字问"小队长，当前支付成功率" | 2s 内模板回复 + mock 数据 |
| L2 语音 | 正常启动，开启麦克风说同一句话 | 页面显示识别文本 + 听到回答 |
| L3 视觉 | 后台/页面对 5 姿态注册后到镜头前 | identity 命中，回复带称呼 |
| L4 遥控 | `curl -X POST localhost:8000/api/say -H 'Content-Type: application/json' -d '{"text":"现在应该安静"}'` | 页面播报 |
| L5 兜底 | 删掉 models/tts 再启动 | 自动降级浏览器语音，对话不断 |
| 新增 P1 | 后台值班表增删改查 → 页面再问"支付组预案巡检到哪了" | 返回库中真实进度 |
| 新增 P1 | PTT 模式：按住说话→松开→回答 | ≤2.5s 出声 |
| 新增 P1 | 声纹注册后切 voiceprint 模式 | 非注册人声被拦截，页面显示拒绝 |

## 后台管理台

浏览器打开 `http://localhost:8000/admin`：

1. **总览**：链路状态、当前身份、实时事件提示
2. **人员**：添加成员、人脸 5 姿态注册、声纹录音注册、删除
3. **值班表**：CRUD + 进度/状态更新
4. **声音**：收音模式、声纹开关与阈值、TTS 引擎与语速
5. **台本**：播报文本 + cron + 启用/停用
6. **模型**：LLM provider 配置（默认 DeepSeek），api_key 留空保持原值
7. **对话**：System Prompt 人格编辑

## 无头验证（不开浏览器）

```bash
.venv/bin/python scripts/selftest.py     # ASR/TTS/人脸/声纹/Store/API 全量
.venv/bin/python scripts/test_admin.py   # 后台管理 API 集成测试
.venv/bin/python scripts/regression.py   # L1/L4/L5 回归 + 意图规则
.venv/bin/python norm.py                 # 数字读法单测
.venv/bin/python brain.py                # 规则引擎自问自答
```

## 配置项

| 变量 | 说明 | 默认 |
| ---- | ---- | ---- |
| `MOCK=1` | 全链路 mock，零模型跑文字对话 | 关 |
| `MODEL_DIR` | 模型目录 | `models` |

运行期配置热生效（改后台管理台不重启）：DB + ConfigStore revision。

## 安全/隐私红线

- 人脸只存 128 维向量（`face_db.json`），不存原始照片。
- 声纹只存嵌入向量（`voiceprint_db.json`），不上传云端。
- 模型 provider API key 只存 SQLite，不进代码库；查询接口不回传 key。
- 播报内容 = 工具数据 + 模板，LLM 只做槽位填充与未命中兜底（限时/限 tokens）。

## P2/P3 进度（代码结构已就位，模型/精调待完成）

- [x] kokoro TTS 接入：`pipeline.py` 已支持 melo/kokoro 双档，后台可切换音色/语速；下载脚本已加入 `tts_kokoro`
- [x] LLM 薄循环工具调用：`brain.py` 已支持 ≤2 轮 tool calling、总 timeout 1.5s（工具只返回真实数据）
- [x] 可编辑意图规则表：后台「对话」页可增删改正则→模板，页面内测试
- [x] 模型连通性测试：后台「模型」页一键测试 LLM provider 延迟
- [x] 台本批量预生成：`/api/playbook/generate-all` + 单条 `/api/playbook/{id}/generate`
- [x] 总览页事件流：`/api/events` + 后台实时轮询已实现
- [x] 台本 cron 调度器：每分钟检查、命中即播报
- [x] L1/L4/L5 回归测试：`scripts/regression.py`
- [ ] kokoro 实测验证：等待 `models/tts_kokoro/` 下载完成
- [ ] 声纹模型实测验证：等待 `models/speaker/` 下载完成
- [ ] KWS 唤醒词 hybrid 模式：待接入 zipformer-wenetspeech 关键词模型与 keywords_file
