"""感知 + 发声封装：VAD/ASR/TTS/人脸/声纹。

相对 POC 的关键升级：
- 人脸：5 姿态保留全部向量 + max-cosine + top1-top2 margin + 质量门控。
- 声纹：SpeakerEmbeddingExtractor 注册/比对，VAD 段过声纹锁才放行 ASR；PTT 可豁免。
- TTS：后续接入 kokoro，此处保留 melo 老路并加 VoiceProfile 占位。
"""
from pathlib import Path
import base64
import io
import json
import os
import re
import time
import wave

import numpy as np

MOCK = os.environ.get("MOCK") == "1"
MODELS = os.environ.get("MODEL_DIR", "models")

TAG = re.compile(r"<\|[^|]*\|>")

# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


# ---------------------------------------------------------------------------
# 语音输入：VAD + ASR
# ---------------------------------------------------------------------------
class AudioPipe:
    """silero VAD 切段 -> SenseVoice ASR。feed() 收 PCM16 bytes，吐出识别完的句子。"""

    WINDOW = 512  # 32ms @16k

    def __init__(self, voiceprint_gate=None, voiceprint_required=False, voiceprint_threshold=0.55,
                 kws_gate=None, kws_required=False):
        self.ok = False
        self._pending = np.zeros(0, dtype=np.float32)
        self._voiceprint_gate = voiceprint_gate
        self._voiceprint_required = voiceprint_required
        self._voiceprint_threshold = voiceprint_threshold
        self._rejected_signal = None  # 调用方可设置 cb(samples, score, name)
        self._accepted_signal = None  # 调用方可设置 cb(samples, score, name)：声纹放行时带上命中人
        self._kws_gate = kws_gate
        self._kws_required = kws_required  # hybrid 模式：需唤醒词
        self._is_awake = None  # 调用方注入 cb() -> bool：唤醒窗口是否打开
        self._wake_signal = None  # 调用方注入 cb()：命中纯唤醒词时回调（应答+开窗）
        if MOCK:
            print("[AudioPipe] MOCK 模式，语音输入关闭，用页面文字输入")
            return
        try:
            import sherpa_onnx

            vad_cfg = sherpa_onnx.VadModelConfig(
                silero_vad=sherpa_onnx.SileroVadModelConfig(
                    model=f"{MODELS}/vad/silero_vad.onnx",
                    threshold=0.5,
                    min_speech_duration=0.25,
                    min_silence_duration=0.5,
                    max_speech_duration=20.0,
                    window_size=self.WINDOW,
                ),
                sample_rate=16000,
                num_threads=1,
                provider="cpu",
                debug=False,
            )
            self.vad = sherpa_onnx.VoiceActivityDetector(vad_cfg)
            self.asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=f"{MODELS}/asr/model.int8.onnx",
                tokens=f"{MODELS}/asr/tokens.txt",
                use_itn=True,
            )
            self.ok = True
            print("[AudioPipe] VAD + SenseVoice 就绪")
        except Exception as e:
            print(f"[AudioPipe] 初始化失败，语音输入关闭: {e}")

    def feed(self, pcm16: bytes) -> list[str]:
        """喂一块 PCM16 16kHz 单声道，返回识别文本列表。"""
        if not self.ok:
            return []
        chunk = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self._pending = np.concatenate([self._pending, chunk])

        results = []
        while len(self._pending) >= self.WINDOW:
            window, self._pending = self._pending[: self.WINDOW], self._pending[self.WINDOW :]
            self.vad.accept_waveform(window)
            results.extend(self._drain())
        return results

    def flush(self) -> list[str]:
        if not self.ok:
            return []
        if len(self._pending):
            pad = np.zeros(self.WINDOW - len(self._pending), dtype=np.float32)
            self.vad.accept_waveform(np.concatenate([self._pending, pad]))
            self._pending = np.zeros(0, dtype=np.float32)
        self.vad.flush()
        return self._drain()

    def _drain(self) -> list[str]:
        results = []
        while not self.vad.empty():
            samples = np.array(self.vad.front.samples)
            self.vad.pop()
            if len(samples) == 0:
                continue

            # 唤醒词状态机（hybrid）：
            #   未唤醒：段落须命中唤醒词；短段=纯唤醒 → 回调应答+开窗；长段=命令直接识别
            #   已唤醒：短段若是再次唤醒则刷新窗口；其余直接进入识别
            if self._kws_required and self._kws_gate:
                awake = bool(self._is_awake and self._is_awake())
                short = len(samples) < 16000 * 2.0
                if not awake:
                    if not self._kws_gate.detect(samples):
                        continue
                    if short:
                        if self._wake_signal:
                            self._wake_signal()
                        continue
                elif short and self._kws_gate.detect(samples):
                    if self._wake_signal:
                        self._wake_signal()
                    continue

            # 声纹门控：连续监听模式下只放行已注册说话人
            if self._voiceprint_required and self._voiceprint_gate and self._voiceprint_gate.ok and self._voiceprint_gate._avg:
                ok, score, name = self._voiceprint_gate.verify(samples, self._voiceprint_threshold)
                if not ok:
                    if self._rejected_signal:
                        self._rejected_signal(samples, score, name)
                    continue
                if self._accepted_signal:
                    self._accepted_signal(samples, score, name)

            text = self._decode(samples)
            if text:
                results.append(text)
        return results

    def _decode(self, samples: np.ndarray) -> str:
        stream = self.asr.create_stream()
        stream.accept_waveform(16000, samples)
        self.asr.decode_stream(stream)
        return TAG.sub("", stream.result.text).strip(" ，。,.!?！？").strip()


# ---------------------------------------------------------------------------
# 声纹门控
# ---------------------------------------------------------------------------
class VoiceprintGate:
    """基于 sherpa-onnx SpeakerEmbeddingExtractor 的声纹锁。"""

    DB_FILE = "voiceprint_db.json"
    POSES = ["normal"]

    def __init__(self):
        self.ok = False
        self.db: dict[str, list[list[float]]] = {}
        self._avg: dict[str, np.ndarray] = {}
        self._load()
        if MOCK:
            print("[VoiceprintGate] MOCK 模式，声纹锁关闭")
            return
        try:
            import sherpa_onnx

            # 3d-speaker 中文说话人嵌入模型（sherpa-onnx release）
            self.sid_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=f"{MODELS}/speaker/wespeaker_zh_cnceleb_resnet34.onnx",
                num_threads=2,
                provider="cpu",
            )
            self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(self.sid_cfg)
            self.ok = True
            print(f"[VoiceprintGate] 声纹锁就绪，已注册 {list(self.db.keys())}")
        except Exception as e:
            print(f"[VoiceprintGate] 初始化失败，声纹锁关闭: {e}")

    def _load(self):
        if os.path.exists(self.DB_FILE):
            try:
                with open(self.DB_FILE, encoding="utf-8") as f:
                    self.db = json.load(f)
            except Exception as e:
                print(f"[VoiceprintGate] db 读取失败: {e}")
        self._avg = {
            n: _norm(np.mean([np.array(v, dtype=np.float32) for v in vs], axis=0))
            for n, vs in self.db.items()
            if vs
        }

    def _save(self):
        with open(self.DB_FILE, "w", encoding="utf-8") as f:
            json.dump(self.db, f, ensure_ascii=False)
        self._avg = {
            n: _norm(np.mean([np.array(v, dtype=np.float32) for v in vs], axis=0))
            for n, vs in self.db.items()
            if vs
        }

    def _extract(self, samples: np.ndarray) -> np.ndarray | None:
        if not self.ok:
            return None
        import sherpa_onnx

        # 构造一个包含单条语音的 OnlineStream
        stream = self.extractor.create_stream()
        stream.accept_waveform(16000, samples)
        stream.input_finished()
        # 等待模型 ready；设上限防止异常段死循环占满 CPU
        waited = 0
        while not self.extractor.is_ready(stream):
            time.sleep(0.01)
            waited += 1
            if waited > 200:  # 约 2s 仍未就绪，放弃本次提取
                return None
        # 注意：1.13.x 的 SpeakerEmbeddingExtractor 没有 release_stream，stream 交给 GC
        # compute() 返回 list[float]，需转 ndarray
        embedding = self.extractor.compute(stream)
        return _norm(np.asarray(embedding, dtype=np.float32).flatten())

    def register(self, samples: np.ndarray, name: str) -> bool:
        if not self.ok or not name:
            return False
        feat = self._extract(samples)
        if feat is None:
            return False
        self.db.setdefault(name, []).append([round(float(x), 6) for x in feat])
        self.db[name] = self.db[name][-5:]  # 最多保留 5 条
        self._save()
        print(f"[VoiceprintGate] 已注册 {name}（{len(self.db[name])} 条）")
        return True

    def verify(self, samples: np.ndarray, threshold: float = 0.55) -> tuple[bool, float, str | None]:
        """VAD 段 → 提取声纹 → 返回 (是否通过, 最高分, 命中人名)。"""
        if not self.ok or not self._avg:
            return True, 0.0, None  # 未启用/无注册人时默认放行
        feat = self._extract(samples)
        if feat is None:
            return False, 0.0, None
        scores = {n: float(feat @ v) for n, v in self._avg.items()}
        name = max(scores, key=scores.get)
        score = scores[name]
        return score >= threshold, score, name


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# KWS 唤醒词门控（P3 占位）
# ---------------------------------------------------------------------------
class KwsGate:
    """关键词唤醒（hybrid 模式）：sherpa-onnx KeywordSpotter + wenetspeech zipformer。
    keywords 为逗号/顿号分隔的中文唤醒词（配置项 kws_keywords），空则不启用。"""

    KW_DIR = f"{MODELS}/kws"

    def __init__(self, keywords: str = ""):
        self.ok = False
        self.keywords = [k.strip() for k in re.split(r"[,，、/]", keywords or "") if k.strip()]
        if MOCK or not self.keywords:
            return
        d = self.KW_DIR
        enc = f"{d}/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
        dec = f"{d}/decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
        join = f"{d}/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
        tokens_path = f"{d}/tokens.txt"
        if not all(Path(p).exists() for p in (enc, dec, join, tokens_path)):
            print("[KwsGate] 未检测到 kws 模型，先跑 scripts/download_models.py kws")
            return
        try:
            valid = self._load_tokens(tokens_path)
            lines = []
            for kw in self.keywords:
                toks = self._kw_tokens(kw, valid)
                if toks is None:
                    print(f"[KwsGate] 唤醒词无法转成拼音 token，跳过: {kw}")
                    continue
                lines.append(" ".join(toks) + f" @{kw}")
            if not lines:
                print("[KwsGate] 没有有效唤醒词，关闭")
                return
            kwfile = Path(d) / "keywords_custom.txt"
            kwfile.write_text("\n".join(lines) + "\n", encoding="utf-8")

            import sherpa_onnx

            self.kws = sherpa_onnx.KeywordSpotter(
                tokens=tokens_path, encoder=enc, decoder=dec, joiner=join,
                keywords_file=str(kwfile), num_threads=1, provider="cpu",
            )
            self.ok = True
            print(f"[KwsGate] 唤醒词就绪: {self.keywords}")
        except Exception as e:
            print(f"[KwsGate] 初始化失败: {e}")

    @staticmethod
    def _load_tokens(path: str) -> set[str]:
        with open(path, encoding="utf-8") as f:
            return {line.split()[0] for line in f if line.strip()}

    @staticmethod
    def _kw_tokens(word: str, valid: set[str]) -> list[str] | None:
        """汉字 → 拼音声母+带调韵母 token 序列；出现未知 token 返回 None。"""
        from pypinyin import lazy_pinyin, Style

        syls = lazy_pinyin(word, style=Style.TONE)
        inis = lazy_pinyin(word, style=Style.INITIALS, strict=False)
        toks: list[str] = []
        for s, i in zip(syls, inis):
            for part in ([i] if i else []) + [s[len(i):]]:
                if part not in valid:
                    return None
                toks.append(part)
        return toks or None

    def detect(self, samples: np.ndarray) -> bool:
        """一段 VAD 切出的音频是否命中任一唤醒词。
        注意：get_result 是瞬态的，必须在解码循环内检查（官方示例同款写法）；
        结尾补 0.66s 静音让 keyword 边界得以封口。"""
        if not self.ok:
            return False
        stream = self.kws.create_stream()
        stream.accept_waveform(16000, samples)
        stream.accept_waveform(16000, np.zeros(int(16000 * 0.66), dtype=np.float32))
        stream.input_finished()
        while self.kws.is_ready(stream):
            self.kws.decode_stream(stream)
            if self.kws.get_result(stream):
                return True
        return False

class TtsPipe:
    """vits-melo / kokoro 双档 TTS。模型缺失时 ok=False，由前端降级浏览器语音。"""

    # kokoro v1.0 voices.bin 的音色名 → sid（取自 model.onnx 元数据 speaker2id）
    KOKORO_VOICE2ID = {
        "af_alloy": 0, "af_aoede": 1, "af_bella": 2, "af_heart": 3, "af_jessica": 4,
        "af_kore": 5, "af_nicole": 6, "af_nova": 7, "af_river": 8, "af_sarah": 9,
        "af_sky": 10, "am_adam": 11, "am_echo": 12, "am_eric": 13, "am_fenrir": 14,
        "am_liam": 15, "am_michael": 16, "am_onyx": 17, "am_puck": 18, "am_santa": 19,
        "bf_alice": 20, "bf_emma": 21, "bf_isabella": 22, "bf_lily": 23, "bm_daniel": 24,
        "bm_fable": 25, "bm_george": 26, "bm_lewis": 27, "ef_dora": 28, "em_alex": 29,
        "ff_siwis": 30, "hf_alpha": 31, "hf_beta": 32, "hm_omega": 33, "hm_psi": 34,
        "if_sara": 35, "im_nicola": 36, "jf_alpha": 37, "jf_gongitsune": 38,
        "jf_nezumi": 39, "jf_tebukuro": 40, "jm_kumo": 41, "pf_dora": 42,
        "pm_alex": 43, "pm_santa": 44, "zf_xiaobei": 45, "zf_xiaoni": 46,
        "zf_xiaoxiao": 47, "zf_xiaoyi": 48, "zm_yunjian": 49, "zm_yunxi": 50,
        "zm_yunxia": 51, "zm_yunyang": 52, "em_santa": 53,
    }

    def __init__(self, engine: str = "melo", voice: str | None = None, speed: float = 1.0):
        self.ok = False
        self.engine = engine
        self.voice = voice
        self.speed = speed
        self.sid = 0
        if MOCK:
            print("[TtsPipe] MOCK 模式，服务端 TTS 关闭，前端用浏览器语音")
            return
        self._init_engine(engine, voice)

    def _init_engine(self, engine: str, voice: str | None):
        import sherpa_onnx

        if engine == "kokoro":
            d = f"{MODELS}/tts_kokoro"
            if not Path(d).exists():
                print("[TtsPipe] kokoro 模型目录不存在，尝试回落 melo")
                engine = "melo"
            else:
                try:
                    # voices.bin 是 v1.0 发布包的实际文件名（不是 voices.json）
                    voices_path = f"{d}/voices.bin" if (Path(d) / "voices.bin").exists() else ""
                    data_dir = f"{d}/espeak-ng-data" if (Path(d) / "espeak-ng-data").exists() else f"{d}"
                    lexicon = ",".join(
                        p for p in (f"{d}/lexicon-zh.txt", f"{d}/lexicon-us-en.txt")
                        if Path(p).exists()
                    )
                    kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model=f"{d}/model.onnx",
                        voices=voices_path,
                        tokens=f"{d}/tokens.txt",
                        lexicon=lexicon,
                        data_dir=data_dir,
                    )
                    cfg = sherpa_onnx.OfflineTtsConfig(
                        model=sherpa_onnx.OfflineTtsModelConfig(kokoro=kokoro, num_threads=2, provider="cpu"),
                        max_num_sentences=1,
                    )
                    if not cfg.validate():
                        raise RuntimeError("kokoro 配置校验失败")
                    self.tts = sherpa_onnx.OfflineTts(cfg)
                    self.engine = "kokoro"
                    # 新版 sherpa-onnx 去掉了 voice= 参数，音色用 sid 索引（见 KOKORO_VOICE2ID）
                    if voice not in self.KOKORO_VOICE2ID:
                        voice = "af_alloy"
                    self.voice = voice
                    self.ok = True
                    print(f"[TtsPipe] kokoro 就绪 (voice={self.voice})")
                    return
                except Exception as e:
                    print(f"[TtsPipe] kokoro 加载失败，回落 melo: {e}")
                    engine = "melo"

        if engine == "melo":
            d = f"{MODELS}/tts"
            try:
                vits = sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=f"{d}/model.onnx",
                    lexicon=f"{d}/lexicon.txt",
                    tokens=f"{d}/tokens.txt",
                )
                cfg = sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=2, provider="cpu"),
                    max_num_sentences=1,
                )
                if not cfg.validate():
                    raise RuntimeError("melo 配置校验失败")
                self.tts = sherpa_onnx.OfflineTts(cfg)
                self.engine = "melo"
                self.voice = None
                self.sid = 0
                self.ok = True
                print("[TtsPipe] vits-melo 就绪")
            except Exception as e:
                print(f"[TtsPipe] 初始化失败，自动降级浏览器语音: {e}")

    def synth(self, text: str) -> bytes | None:
        if not self.ok:
            return None
        sentences = [s.strip() for s in re.split(r"[。！？；!?;\n]+", text) if s.strip()]
        if not sentences:
            return None
        pieces, sr = [], None
        for s in sentences:
            try:
                if self.engine == "kokoro":
                    audio = self.tts.generate(
                        s, sid=self.KOKORO_VOICE2ID.get(self.voice, 0), speed=self.speed
                    )
                else:
                    audio = self.tts.generate(s, sid=self.sid, speed=self.speed)
            except TypeError:
                audio = self.tts.generate(s)
            except Exception as e:
                print(f"[TtsPipe] 句子合成失败，跳过: {s!r} {e}")
                continue
            sr = audio.sample_rate
            pieces.append(np.asarray(audio.samples, dtype=np.float32))
            pieces.append(np.zeros(int(0.15 * sr), dtype=np.float32))
        if not pieces:
            return None
        return to_wav(np.concatenate(pieces), sr)

class FacePipe:
    """YuNet 检测 + SFace 128 维特征，max-cosine + margin 校验。"""

    DB_FILE = "face_db.json"
    POSES = ["front", "left", "right", "up", "down"]
    MIN_SIZE = 120
    DEFAULT_QUALITY = 0.0

    def __init__(self):
        self.ok = False
        self.db: dict[str, list[dict]] = {}
        if os.path.exists(self.DB_FILE):
            try:
                with open(self.DB_FILE, encoding="utf-8") as f:
                    self.db = json.load(f)
            except Exception as e:
                print(f"[FacePipe] face_db.json 读取失败: {e}")
        if MOCK:
            print("[FacePipe] MOCK 模式，视觉关闭")
            return
        try:
            import cv2

            self.cv2 = cv2
            self.rec = cv2.FaceRecognizerSF.create(f"{MODELS}/face/face_recognition_sface_2021dec.onnx", "")
            self._detectors = {}
            self.ok = True
            registered = [n for n, vs in self.db.items() if vs]
            print(f"[FacePipe] YuNet + SFace 就绪，已注册 {registered}")
        except Exception as e:
            print(f"[FacePipe] 初始化失败，视觉关闭: {e}")

    def _detector(self, w: int, h: int):
        if (w, h) not in self._detectors:
            self._detectors[(w, h)] = self.cv2.FaceDetectorYN.create(
                f"{MODELS}/face/face_detection_yunet_2023mar.onnx", "", (w, h), 0.6
            )
        return self._detectors[(w, h)]

    def _decode(self, img_b64: str):
        raw = base64.b64decode(img_b64)
        return self.cv2.imdecode(np.frombuffer(raw, np.uint8), self.cv2.IMREAD_COLOR)

    def _quality(self, img, face) -> float:
        """人脸质量分：尺寸 + 清晰度(Laplacian) + 亮度。"""
        x, y, w, h = face[:4]
        if w < self.MIN_SIZE or h < self.MIN_SIZE:
            return 0.0
        x, y, w, h = max(0, int(x)), max(0, int(y)), max(1, int(w)), max(1, int(h))
        roi = img[y : y + h, x : x + w]
        if roi.size == 0:
            return 0.0
        gray = self.cv2.cvtColor(roi, self.cv2.COLOR_BGR2GRAY)
        lap_var = self.cv2.Laplacian(gray, self.cv2.CV_64F).var()
        brightness = float(gray.mean())
        # 亮度居中(128)加分，清晰度越高越好
        bright_score = 1.0 - min(abs(brightness - 128) / 128, 1.0)
        sharp_score = min(lap_var / 300.0, 1.0)
        size_score = min(min(w, h) / 200.0, 1.0)
        return round(0.3 * bright_score + 0.4 * sharp_score + 0.3 * size_score, 3)

    def detect(self, img_b64: str):
        """返回最大人脸的 (face_obj, quality, w, h)。"""
        if not self.ok:
            return None, 0.0, 0, 0
        img = self._decode(img_b64)
        h, w = img.shape[:2]
        _, faces = self._detector(w, h).detect(img)
        if faces is None or len(faces) == 0:
            return None, 0.0, 0, 0
        face = max(faces, key=lambda f: f[2] * f[3])
        q = self._quality(img, face)
        return img, face, q, int(face[2]), int(face[3])

    def _feature(self, img, face) -> np.ndarray | None:
        aligned = self.rec.alignCrop(img, face)
        return _norm(self.rec.feature(aligned).flatten())

    def identify(self, img_b64: str, threshold: float | None = None, margin: float = 0.05) -> tuple[str | None, float]:
        """max-cosine + top1-top2 margin。返回 (名字, 得分)。"""
        if not self.ok:
            return None, 0.0
        try:
            img, face, q, w, h = self.detect(img_b64)
            if face is None or q == 0.0:
                return None, 0.0
            threshold = threshold if threshold is not None else float(os.environ.get("FACE_THRESHOLD", "0.45"))
            feat = self._feature(img, face)
            if feat is None:
                return None, 0.0

            # 只取质量分够格的注册向量
            all_scores = []
            for name, records in self.db.items():
                for rec in records:
                    if isinstance(rec, dict) and "v" in rec:
                        vec = np.array(rec["v"], dtype=np.float32)
                        all_scores.append((name, float(feat @ vec)))
                    elif isinstance(rec, list):
                        vec = np.array(rec, dtype=np.float32)
                        all_scores.append((name, float(feat @ vec)))
            if not all_scores:
                return None, 0.0

            per_name = {}
            for n, s in all_scores:
                per_name[n] = max(per_name.get(n, -1.0), s)
            ranked = sorted(per_name.items(), key=lambda x: x[1], reverse=True)
            top1_name, top1_score = ranked[0]
            top2_score = ranked[1][1] if len(ranked) > 1 else 0.0
            if top1_score >= threshold and (top1_score - top2_score) >= margin:
                return top1_name, round(top1_score, 3)
            return None, round(top1_score, 3)
        except Exception as e:
            print(f"[FacePipe] identify 异常: {e}")
            return None, 0.0

    def register(self, name: str, img_b64: str, pose: str = "front") -> dict:
        """注册一张人脸。返回结果 dict。"""
        out = {"ok": False, "name": name, "pose": pose, "quality": 0.0, "count": 0}
        if not self.ok or not name:
            return out
        try:
            img, face, q, w, h = self.detect(img_b64)
            if face is None:
                out["error"] = "未检测到人脸"
                return out
            if q == 0.0 or w < self.MIN_SIZE or h < self.MIN_SIZE:
                out["error"] = f"人脸质量不达标：尺寸 {w}x{h} 质量 {q}"
                return out
            feat = self._feature(img, face)
            if feat is None:
                out["error"] = "特征提取失败"
                return out
            record = {"v": [round(float(x), 6) for x in feat], "pose": pose, "q": q}
            # 如果已有旧格式 list，先保留；否则追加 record
            self.db.setdefault(name, []).append(record)
            self.db[name] = self.db[name][-10:]
            with open(self.DB_FILE, "w", encoding="utf-8") as f:
                json.dump(self.db, f, ensure_ascii=False)
            out.update({"ok": True, "quality": q, "count": len(self.db[name])})
            print(f"[FacePipe] 已注册 {name}/{pose} 质量{q}")
        except Exception as e:
            out["error"] = str(e)
            print(f"[FacePipe] register 异常: {e}")
        return out

    def delete_person(self, name: str) -> bool:
        if name in self.db:
            del self.db[name]
            with open(self.DB_FILE, "w", encoding="utf-8") as f:
                json.dump(self.db, f, ensure_ascii=False)
            print(f"[FacePipe] 已删除 {name}")
            return True
        return False
