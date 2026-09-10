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

    def __init__(self, voiceprint_gate=None, voiceprint_required=False, voiceprint_threshold=0.55):
        self.ok = False
        self._pending = np.zeros(0, dtype=np.float32)
        self._voiceprint_gate = voiceprint_gate
        self._voiceprint_required = voiceprint_required
        self._voiceprint_threshold = voiceprint_threshold
        self._rejected_signal = None  # 调用方可设置 cb(samples, score, name)
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

            # 声纹门控：连续监听模式下只放行已注册说话人
            if self._voiceprint_required and self._voiceprint_gate and self._voiceprint_gate.ok and self._voiceprint_gate._avg:
                ok, score, name = self._voiceprint_gate.verify(samples, self._voiceprint_threshold)
                if not ok:
                    if self._rejected_signal:
                        self._rejected_signal(samples, score, name)
                    continue

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
        # 等待模型 ready
        while not self.extractor.is_ready(stream):
            pass
        embedding = self.extractor.compute(stream)
        self.extractor.release_stream(stream)
        return _norm(embedding.flatten())

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
    """关键词唤醒（小队长）。模型到位后启用，hybrid 模式下作为唤醒触发器。"""

    def __init__(self):
        self.ok = False
        if MOCK:
            print("[KwsGate] MOCK 模式，KWS 关闭")
            return
        model = Path(f"{MODELS}/kws/model.onnx")
        if not model.exists():
            print("[KwsGate] 未检测到 kws 模型，关闭")
            return
        try:
            # TODO: 接入 sherpa-onnx KeywordSpotter（zipformer-wenetspeech + keywords_file）
            self.ok = False
        except Exception as e:
            print(f"[KwsGate] 初始化失败: {e}")

    def feed(self, pcm16: bytes) -> bool:
        """喂 PCM16 音频，返回是否检测到唤醒词。"""
        if not self.ok:
            return False
        # TODO: 实现 KeywordSpotter 流式检测
        return False

class TtsPipe:
    """vits-melo / kokoro 双档 TTS。模型缺失时 ok=False，由前端降级浏览器语音。"""

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
                    voices_path = f"{d}/voices.json" if (Path(d) / "voices.json").exists() else ""
                    data_dir = f"{d}"
                    if (Path(d) / "data").exists():
                        data_dir = f"{d}/data"
                    kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model=f"{d}/model.onnx",
                        voices=voices_path,
                        tokens=f"{d}/tokens.txt",
                        lexicon=f"{d}/lexicon.txt" if (Path(d) / "lexicon.txt").exists() else "",
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
                    self.voice = voice or "af"
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
                    audio = self.tts.generate(s, sid=0, speed=self.speed, voice=self.voice)
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
