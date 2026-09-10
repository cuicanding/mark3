# AI 大促队长 · 后台管理与能力升级 设计文档 v0.3

## 一、总纲：三层智能分级（本设计的灵魂）

系统的所有能力按"触发可枚举性 / 路径可枚举性 / 出错代价"归入三层，**AI 自主配比 = 99% 确定性 + 1% 白名单对话**：

| 层 | 内容 | AI 含量 | 实现 |
| --- | --- | --- | --- |
| **L1 调度层（零 AI）** | 定时定点播报、阈值触发告警、静默令 | 0% | APScheduler + 台本表，100% 可预测 |
| **L2 编排层（笼内 AI）** | 定时巡检出稿、自由问答、身份问候、人脸应用 | 笼内 | 事件→意图路由→工具取数→LLM 槽位填充→**说话权闸门**→播报 |
| **L3 自主层（暂封存）** | 故障排查辅助、多步工具循环 | ≤1% | 仅白名单自由问答启用（≤2 轮工具循环，全程落盘）；故障排查辅助列为二期候选 |

说话权闸门红线不变：播报内容 = 工具查询结果 + 预审模板组装，LLM 只做槽位填充与分析出稿（出稿走人审后才播报），禁止自由生成直接上嘴。

## 二、多音色 TTS 成本评估（实测，回应"比 melo 大多少"）

| 模型 | 包大小 | 音色数（中文） | 磁盘 | 内存占用 | 合成延迟* | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| vits-melo-zh_en（现役） | 159MB | 1 | 197MB | ~300MB | ~0.3-0.5s/句 | 保留作低延迟兜底 |
| **kokoro-multi-lang-v1_0（选型）** | **333MB（≈2.1×）** | **8（4男4女）** | ~420MB | ~700MB | ~1-2s/句 | 主力：音色试听/切换用它 |
| vits-zh-aishell3（备选） | 140MB | 174 | ~180MB | ~350MB | ~0.5s/句 | 音色多但机械感重，不推荐首发 |

*延迟为 M 系列 CPU 估值，实施时实测校验。

**双档发声策略**（化解 kokoro 延迟与 ≤2s 预算的矛盾）：
- 互动问答（实时路径）：默认 kokoro；若实测 RTF 超标，自动回落 melo（`VoiceProfile` 一行配置）
- 台本播报（L1 路径）：kokoro **预生成** wav 存盘，播报时直接放音频，零实时合成
- 成本总账：磁盘 +420MB、内存峰值 +700MB，16GB Mac mini 无压力；下载走已验证的 ghfast.top 通道约 10 分钟

## 三、架构：六层 + 薄循环大脑（不引任何 agent 框架）

```
感知层   浏览器采音 → [浏览器NS] → [可选GTCRN降噪] → silero VAD → 声纹门控 → [可选KWS唤醒] → SenseVoice ASR
决策层   会话状态机 → IntentRouter(规则表) → 工具取数 → TalkGate(模板渲染) → [未命中→LLM薄循环≤2轮/超时1.5s] → 人格注入
表达层   文本 → 数字归一化(norm.py) → VoiceProfile(kokoro音色/语速) → 逐句合成 → 字幕+大屏事件
技能层   SkillRegistry：metrics / roster / broadcast / inspect（装饰器注册，元数据供后台启停与配参）
治理层   后台管理台（MVP 七页）→ ConfigStore(SQLite，版本号热生效，改配置不重启)
数据层   captain.db(配置) + events.jsonl(审计) + face_db/voiceprint_db(生物特征，只存向量)
```

**大脑薄循环**（已确认不引框架）：`openai` SDK（DeepSeek 兼容协议）+ ~30 行 tool-calling 循环（封顶 2 轮、总超时 1.5s、失败回兜底模板）+ pydantic 校验技能参数。说话权闸门在代码层强制，不委托给 prompt。

**Pi agent 定位**：MVP 不接入；治理层预留「Agent 任务」适配位（子进程方式跑长任务：巡检报告、批量提醒），二期启用。

## 四、分模块关键改动

### 4.1 感知层（痛点 1/2：嘈杂环境 + 持续误识别）
- **三种收音模式**（后台可切，页面同步显示）：`ptt` 按住说话（默认，最稳）/ `voiceprint` 声纹锁持续监听 / `hybrid` 声纹+唤醒词
- **声纹锁**：新增 `VoiceprintGate`——sherpa SpeakerEmbeddingExtractor（3D-Speaker/WeSpeaker 中文模型，~30MB，走 ghfast 通道）；注册 3-5 句取均值向量；VAD 段→提纹→cosine≥阈值(默认 0.55 可配)才放行 ASR；PTT 通道豁免声纹
- **唤醒词**（hybrid 模式用）：KWS zipformer-wenetspeech-3.3M + keywords_file 自定义"小队长"（拼音分词，免训练），误触指标 30 分钟 <3 次
- **降噪**：浏览器 noiseSuppression 已开；服务端 GTCRN 为可选档位，默认关（省 CPU）
- 前端：新增「按住说话」大按钮（Space 键也可按住）、收音模式开关、声纹通过/拦截状态灯

### 4.2 人脸精度修复（痛点 3：相似度 0.49 偏低）
根因：注册 3 张是 0.8s 连拍≈同一帧 + 480px 低清 JPEG + 平均向量对姿态敏感。修复（零成本策略，不换模型）：
- 注册升级 **5 姿态采集**（正/左/右/微仰/微俯），逐张质量门控（人脸≥120px、Laplacian 清晰度、亮度），每张显示质量分
- 比对改 **max-cosine**（保留每条向量，不再平均）+ **top1−top2 margin ≥0.05** 防多人串脸
- 识别帧质量门控（太小/太糊跳过）+ 阈值可配（默认 0.45）
- 应用范围按决定：MVP 只做问候+称呼；代码留 `identity_hooks` 扩展点（二期在场看板/权限门）
- 升级位：InsightFace buffalo_s 二期替换 SFace，接口预留

### 4.3 后台管理台（MVP 七页，无前端构建链，vanilla JS 沿用 POC 视觉）
| 页 | 内容 |
| --- | --- |
| ①总览 | 链路状态卡（ASR/TTS/声纹/人脸/LLM provider）、实时延迟、当前身份、一键静音、最近事件流 |
| ②对话与技能 | 人格编辑（system prompt/称呼/语气）、技能启停+参数、意图规则表（正则+模板，页面内可测） |
| ③声音 | 音色试听（kokoro 8 音色逐个试听选定）、语速/音量、收音模式、声纹锁开关+阈值、唤醒词管理 |
| ④人员 | 人脸 5 姿态注册、声纹 3 句注册、成员列表（质量分）、删除、权限标记（预留） |
| ⑤值班表 | 人员/时段/任务/状态/backup CRUD + 进度更新（替掉 brain.py 硬编码 ROSTER） |
| ⑥台本播报 | 台本 CRUD（文本+cron+kokoro 预生成）、播报队列 confirm 放行、遥控直输、静默令 |
| ⑦模型 | LLM provider 配置（base_url/api_key/model，默认 DeepSeek）、连通性测试、超时/兜底开关 |

### 4.4 数据与协议
- 新增 `captain.db`（SQLite，stdlib sqlite3）：members / face_vectors / voiceprints / roster / tasks / playbook / skills / personas / settings / model_provider
- `ROSTER`/`Metrics` 从 brain.py 迁入库；`events.jsonl` 保留作审计
- WS 协议向后兼容扩展：`mode`、`ptt_start/ptt_stop`、`voiceprint{status,score}`、`playbook` 事件；POC 的 L1-L5 契约不破坏
- 配置热生效：ConfigStore 版本号 + 变更广播，改音色/阈值/规则不重启

## 五、测试与验收

**回归红线**：POC 的 L1-MOCK / L2-语音 / L3-视觉 / L4-遥控 / L5-降级 五档必须全部不退化。

**新增验收**：
1. 声纹门控：本人说话放行率 ≥90%；他人说话、背景视频音 30 分钟拦截率 ≥95%
2. PTT：按住说话→松开→≤2.5s 出声回答；未按住时持续说话不触发
3. 唤醒词（hybrid）：TV 背景音 30 分钟误触 <3 次
4. 人脸：5 姿态注册后本人相似度均值提升、他人 <0.36、无串脸（margin 校验生效）
5. 音色：后台切换 kokoro 音色/语速后，下一条播报生效；台本预生成可离线播放
6. 台本：改系统时间 ±2 分钟触发整点播报全链路出声（沿用 W1 D4 口径）
7. LLM：DeepSeek 连通性测试通过；断网/超时 1.5s 回兜底模板不断答
8. 闸门：播报内容抽检 = 工具数据+模板组装，无 LLM 自由发挥

**性能**：说完→文本 ≤1s；互动问答端到端 ≤2.5s；改配置热生效 ≤1s。

## 六、实施分期（建议）

- **P1（2-3 天）**：ConfigStore+SQLite、值班表 CRUD、人脸精度修复、声纹锁+PTT、后台①④⑤页 → 解决全部已感知痛点
- **P2（2-3 天）**：kokoro 接入+双档发声、声音页③、台本播报⑥、对话与技能②、LLM 薄循环+DeepSeek → 完整 MVP
- **P3（1-2 天）**：总览①、模型页⑦、KWS hybrid、联调+八项验收

## 七、假设与默认值（无人反对即生效）

1. 大脑=自研薄循环（openai SDK），不引 LangChain/pydantic-ai/Pi；Pi 仅留二期适配位
2. TTS 主力 kokoro-multi-lang-v1_0；实测延迟不达标时互动路径回落 melo
3. 声纹模型选 3D-Speaker/WeSpeaker 中文款（~30MB）；阈值默认 0.55、人脸阈值默认 0.45，均可后台调
4. 默认收音模式 = PTT（办公室嘈杂场景最稳），声纹锁为一键切换
5. DeepSeek API Key 由你提供（后台模型页填写，存 SQLite 不进代码库）
6. 人脸应用 MVP 只做问候；在场看板/权限门/InsightFace/Agent 任务面板 = 二期清单
7. 交付物：本设计将落成 `MARK3 设计文档 v0.3.md` + 按 P1→P3 实施代码
