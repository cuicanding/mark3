"""一键下载 MVP 全部模型（GitHub releases + gh-proxy 镜像，约 700-800MB）。
用法: python scripts/download_models.py [all、asr、tts、tts_kokoro、vad、face、speaker]
背景: hf-mirror 大文件链路在本网络不通，而 sherpa-onnx 官方把全部模型
      打包放在 GitHub releases，走 gh-proxy 可达（实测 500KB/s）。
"""
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "models"
TMP = M / ".tmp"

GH = "https://github.com"
PROXIES = ["https://ghfast.top/", "https://gh-proxy.com/", ""]

ASR_TAR = ("asr-models", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17")
TTS_TAR = ("tts-models", "vits-melo-tts-zh_en")
KOKORO_TAR = ("tts-models", "kokoro-multi-lang-v1_0")
SPEAKER_MODEL = "wespeaker_zh_cnceleb_resnet34.onnx"


def curl(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"[skip] {dest.name} 已存在 ({dest.stat().st_size // 1024} KB)")
        return True
    for prefix in PROXIES:
        u = prefix + url
        print(f"[dl ] {u}")
        r = subprocess.run(
            ["curl", "-sL", "--fail", "-C", "-", "--retry", "2", "-o", str(dest), u],
            timeout=1800,
        )
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
            print(f"[ ok ] {dest.name} ({dest.stat().st_size // 1024} KB)")
            return True
        print(f"[fail] rc={r.returncode}")
    dest.unlink(missing_ok=True)
    return False


def fetch_release_tar(tag: str, name: str, dest_dir: Path):
    """下载 sherpa-onnx release tar.bz2 并把内容平铺到 dest_dir。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = TMP / f"{name}.tar.bz2"
    if not curl(f"{GH}/k2-fsa/sherpa-onnx/releases/download/{tag}/{name}.tar.bz2", marker):
        raise SystemExit(f"下载失败: {name}")
    print(f"[tar] 解压 {marker.name} ...")
    with tarfile.open(marker) as t:
        t.extractall(TMP)
    src = TMP / name
    for f in src.rglob("*"):
        if f.is_file():
            target = dest_dir / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(target))
    shutil.rmtree(src)
    marker.unlink(missing_ok=True)


def fetch_speaker_model(dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"{GH}/k2-fsa/sherpa-onnx/releases/download/speaker-recognition-models/{SPEAKER_MODEL}"
    ok = curl(url, dest_dir / SPEAKER_MODEL)
    if not ok:
        raise SystemExit(f"下载失败: {SPEAKER_MODEL}")


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    TMP.mkdir(parents=True, exist_ok=True)

    if only in ("all", "asr"):
        fetch_release_tar(*ASR_TAR, M / "asr")
        assert (M / "asr" / "model.int8.onnx").exists(), "asr 模型缺失"
    if only in ("all", "tts"):
        fetch_release_tar(*TTS_TAR, M / "tts")
        assert (M / "tts" / "model.onnx").exists(), "tts 模型缺失"
    if only in ("all", "tts_kokoro"):
        fetch_release_tar(*KOKORO_TAR, M / "tts_kokoro")
        assert (M / "tts_kokoro" / "model.onnx").exists(), "kokoro 模型缺失"
    if only in ("all", "vad"):
        ok = curl(f"{GH}/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
                  M / "vad" / "silero_vad.onnx")
        if not ok:
            raise SystemExit("下载失败: silero_vad")
    if only in ("all", "speaker"):
        fetch_speaker_model(M / "speaker")
    if only in ("all", "face"):
        zoo = f"{GH}/opencv/opencv_zoo/raw/main/models"
        ok1 = curl(f"{zoo}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                   M / "face" / "face_detection_yunet_2023mar.onnx")
        ok2 = curl(f"{zoo}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
                   M / "face" / "face_recognition_sface_2021dec.onnx")
        if not (ok1 and ok2):
            raise SystemExit("下载失败: face 模型")

    print("模型目录:")
    for p in sorted(M.rglob("*")):
        if p.is_file() and ".tmp" not in str(p) and ".cache" not in str(p):
            print(f"  {p.relative_to(ROOT)}  {p.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
