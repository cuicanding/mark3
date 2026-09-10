"""服务端能力自检（不开浏览器）：ASR / TTS / 人脸 / 声纹 / Store / API。
用法: python scripts/selftest.py
"""
import os
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from fastapi.testclient import TestClient
from norm import normalize_for_tts
from pipeline import AudioPipe, TtsPipe, FacePipe, VoiceprintGate
from store import store
import server


def read_wav_int16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000, f"测试音频需 16k，实际 {w.getframerate()}"
        assert w.getnchannels() == 1, "测试音频需单声道"
        return w.readframes(w.getnframes())


def main():
    print("Store 自检")
    print("  settings:", list(store.settings_dict().keys())[:5], "...")
    print("  roster rows:", len(store.roster_list()))

    print("ASR 自检")
    audio = AudioPipe()
    print("  ok:", audio.ok)
    if audio.ok:
        for name in ("zh.wav", "en.wav"):
            p = ROOT / "models" / "asr" / "test_wavs" / name
            if not p.exists():
                continue
            texts = audio.feed(read_wav_int16(p)) + audio.flush()
            print(f"  {name} -> {texts}")

    print("TTS 自检")
    tts = TtsPipe()
    print("  ok:", tts.ok)
    if tts.ok:
        text = "当前支付成功率 99.95%，水位正常"
        wav = tts.synth(normalize_for_tts(text))
        out = Path("/tmp/captain_tts_test.wav")
        out.write_bytes(wav)
        print(f"  {text}")
        print(f"  -> {out} {len(wav) // 1024} KB")

    print("FacePipe 自检")
    face = FacePipe()
    print("  ok:", face.ok, "registered:", list(face.db.keys()))

    print("VoiceprintGate 自检")
    vp = VoiceprintGate()
    print("  ok:", vp.ok, "registered:", list(vp.db.keys()))

    print("API 自检 (FastAPI TestClient)")
    client = TestClient(server.app)
    print("  /api/state", client.get("/api/state").status_code)
    print("  /api/roster", client.get("/api/roster").status_code)
    print("  /api/settings", client.get("/api/settings").status_code)
    print("  /admin page", client.get("/admin").status_code)

    print("\n自检完成")


if __name__ == "__main__":
    main()
