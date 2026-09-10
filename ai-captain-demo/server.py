"""FastAPI 编排层：POC 页面、后台管理、WS /ws、REST API。

WS 协议（向后兼容 POC）：
  客户端->服务端 binary：PCM16 16kHz 单声道音频块
  客户端->服务端 json：{"type":"text"} / {"type":"shot"} / {"type":"register"}
                        {"type":"mode","mode":"ptt|voiceprint|hybrid"}
                        {"type":"ptt_start"} / {"type":"ptt_stop"}
  服务端->客户端 json：asr / identity / reply / tts_mode / voiceprint...
  服务端->客户端 binary：先发 {"type":"audio","bytes":N} 再发 N 字节 WAV
"""
import asyncio
import base64
import io
import json
import os
import time
import wave
from pathlib import Path

import numpy as np
import urllib.request
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from brain import Brain
from norm import normalize_for_tts
from pipeline import AudioPipe, FacePipe, TtsPipe, VoiceprintGate, KwsGate, MOCK
from store import ConfigStore

ROOT = Path(__file__).resolve().parent

store = ConfigStore()
brain = Brain()

# 托管数据层：先注册成员，再生成人脸/声纹向量
voiceprint = VoiceprintGate()
face_pipe = FacePipe()
tts_pipe = TtsPipe()
kws_gate = KwsGate()

clients: set[WebSocket] = set()
EVENTS_FILE = ROOT / "events.jsonl"

# 全局事件循环引用，供声纹拒绝回调触发广播
_event_loop: asyncio.AbstractEventLoop | None = None


# --------------------------------------------------------------------------- seed
POC_ROSTER = [
    {"team": "支付组", "task": "支付链路预案巡检", "owner": "存孝", "backup": "建国", "progress": 80, "status": "进行中"},
    {"team": "风控组", "task": "限流降级预案演练", "owner": "李雷", "backup": "赵敏", "progress": 100, "status": "已完成"},
    {"team": "搜索组", "task": "搜索引擎扩容检查", "owner": "韩梅梅", "backup": "周杰", "progress": 45, "status": "进行中"},
    {"team": "物流组", "task": "履约时效监控核对", "owner": "魏龙", "backup": "郑爽", "progress": 10, "status": "进行中"},
]


def _seed_if_empty():
    """首次启动把 POC 值班表灌入 DB，避免空跑。"""
    if not store.roster_list():
        for r in POC_ROSTER:
            store.roster_add(r["team"], r["task"], r["owner"], r["backup"])
            store.roster_update(store.roster_list()[-1].id, progress=r["progress"], status=r["status"])
        print("[server] 已写入默认值班表")


_seed_if_empty()


# --------------------------------------------------------------------------- helpers
def log_event(direction: str, payload: dict):
    try:
        with open(EVENTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": round(time.time(), 3), "dir": direction, **payload}, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[events] 落盘失败: {e}")


async def broadcast(payload: dict):
    log_event("server", payload)
    for ws in list(clients):
        try:
            await ws.send_json(payload)
        except Exception:
            clients.discard(ws)




# TTS 与连续监听管道按配置动态切换（改后台管理台音色/引擎不重启）
_tts_handle = {"pipe": None, "cfg": None}
_audio_handle = {"pipe": None, "cfg": None}


def get_tts_pipe() -> TtsPipe:
    engine = store.get("tts_engine", "melo")
    voice = store.get("tts_kokoro_voice", "af")
    speed = store.get_float("tts_speed", 1.0)
    cfg = (engine, voice, speed)
    if _tts_handle["pipe"] is None or _tts_handle["cfg"] != cfg:
        _tts_handle["pipe"] = TtsPipe(engine, voice, speed)
        _tts_handle["cfg"] = cfg
    return _tts_handle["pipe"]


def get_continuous_pipe() -> AudioPipe:
    mode, vp_enabled, threshold = current_audio_config()
    vp_required = mode in ("voiceprint", "hybrid") and vp_enabled and bool(voiceprint._avg)
    cfg = (mode, vp_enabled, threshold)
    if _audio_handle["pipe"] is None or _audio_handle["cfg"] != cfg:
        pipe = AudioPipe(voiceprint, vp_required, threshold)
        pipe._rejected_signal = lambda samples, score, name: _notify_voiceprint_rejected(score, name)
        _audio_handle["pipe"] = pipe
        _audio_handle["cfg"] = cfg
        print(f"[server] 连续监听管道重建: mode={mode} vp={vp_enabled} threshold={threshold}")
    return _audio_handle["pipe"]


async def broadcast_audio(text: str):
    pipe = get_tts_pipe()
    if not pipe.ok:
        return
    wav = await asyncio.to_thread(pipe.synth, normalize_for_tts(text))
    if not wav:
        return
    for ws in list(clients):
        try:
            await ws.send_json({"type": "audio", "bytes": len(wav)})
            await ws.send_bytes(wav)
        except Exception:
            clients.discard(ws)


def _notify_voiceprint_rejected(score: float, name: str | None):
    if _event_loop is None:
        return
    asyncio.run_coroutine_threadsafe(
        broadcast({"type": "voiceprint", "status": "rejected", "score": round(score, 3), "name": name}),
        _event_loop,
    )


def current_audio_config() -> tuple[str, bool, float]:
    mode = store.get("audio_mode", "ptt")
    vp_enabled = store.get_bool("voiceprint_enabled")
    threshold = store.get_float("voiceprint_threshold", 0.55)
    return mode, vp_enabled, threshold



# --------------------------------------------------------------------------- identity
class IdentityTracker:
    CONFIRM = 3
    LEAVE = 5

    def __init__(self):
        self.current = None
        self._cand, self._hits, self._miss = None, 0, 0

    def update(self, name: str | None) -> str | None:
        if name:
            self._miss = 0
            if name == self.current:
                return None
            self._cand, self._hits = name, self._hits + 1 if name == self._cand else 1
            if self._hits >= self.CONFIRM:
                self.current, self._cand, self._hits = name, None, 0
                return self.current
        else:
            self._cand, self._hits = None, 0
            self._miss += 1
            if self._miss >= self.LEAVE and self.current:
                self.current = None
                return "leave"
        return None


tracker = IdentityTracker()


# --------------------------------------------------------------------------- app / pages


# --------------------------------------------------------------------------- playbook scheduler
async def playbook_scheduler():
    """每分钟检查一次 playbook，命中则播报。简单 cron 支持 * / 数字。"""
    await asyncio.sleep(3)
    while True:
        try:
            now = time.localtime()
            for row in store.playbook_list():
                if not row.enabled:
                    continue
                if _match_cron(row.cron, now):
                    # 播放音频（预生成优先）
                    audio_path = row.audio_path
                    if audio_path and Path(audio_path).exists():
                        wav = Path(audio_path).read_bytes()
                        for ws in list(clients):
                            try:
                                await ws.send_json({"type": "audio", "bytes": len(wav)})
                                await ws.send_bytes(wav)
                                await ws.send_json({"type": "reply", "text": row.text, "person": tracker.current, "source": "playbook"})
                            except Exception:
                                clients.discard(ws)
                    else:
                        # 实时 TTS
                        await broadcast({"type": "reply", "text": row.text, "person": tracker.current, "source": "playbook"})
                        await broadcast_audio(row.text)
                    log_event("playbook", {"text": row.text, "cron": row.cron})
        except Exception as e:
            print(f"[playbook scheduler] error: {e}")
        await asyncio.sleep(60)


def _match_cron(expr: str, now) -> bool:
    """expr: 分 时 日 月 周，每项为 * 或整数。"""
    parts = expr.split()
    if len(parts) < 5:
        return False
    checks = [(parts[0], now.tm_min), (parts[1], now.tm_hour), (parts[2], now.tm_mday), (parts[3], now.tm_mon), (parts[4], now.tm_wday)]
    for pat, val in checks:
        if pat != "*" and int(pat) != val:
            return False
    return True


# --------------------------------------------------------------------------- TTS 预生成（台本）
async def pregenerate_playbook(pid: int) -> bool:
    pipe = get_tts_pipe()
    if not pipe.ok:
        return False
    rows = [p for p in store.playbook_list() if p.id == pid]
    if not rows:
        return False
    wav = await asyncio.to_thread(pipe.synth, normalize_for_tts(rows[0].text))
    if not wav:
        return False
    out = ROOT / "audio" / f"playbook_{pid}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(wav)
    store.playbook_update(pid, audio_path=str(out))
    return True


app = FastAPI(title="AI 大促队长 MVP")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.on_event("startup")
async def startup():
    asyncio.create_task(playbook_scheduler())

@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/admin")
async def admin_page():
    return FileResponse(ROOT / "static" / "admin.html")


# --------------------------------------------------------------------------- REST API
@app.get("/state")
async def compat_state():
    return await api_state()


@app.get("/api/state")
async def api_state():
    mode, vp_enabled, threshold = current_audio_config()
    return {
        "mock": MOCK,
        "tts_mode": "server" if get_tts_pipe().ok else "browser",
        "tts_engine": get_tts_pipe().engine if get_tts_pipe().ok else None,
        "tts_voice": get_tts_pipe().voice if get_tts_pipe().ok else None,
        "tts_speed": get_tts_pipe().speed if get_tts_pipe().ok else None,
        "asr": get_continuous_pipe().ok,
        "voiceprint": voiceprint.ok,
        "voiceprint_enabled": vp_enabled,
        "kws": kws_gate.ok,
        "voiceprint_registered": list(voiceprint.db.keys()),
        "face": face_pipe.ok,
        "registered_faces": list(face_pipe.db.keys()),
        "identity": tracker.current,
        "clients": len(clients),
        "audio_mode": mode,
        "revision": store.revision(),
        "history": list(brain.history),
    }


@app.post("/say")
async def compat_say(payload: dict):
    return await api_say(payload)


@app.post("/api/say")
async def api_say(payload: dict):
    text = (payload.get("text") or "").strip()
    if not text:
        return {"ok": False, "error": "empty text"}
    log_event("say", {"text": text})
    await broadcast({"type": "reply", "text": text, "person": tracker.current, "source": "say"})
    await broadcast_audio(text)
    return {"ok": True, "text": text}


@app.post("/hush")
async def compat_hush():
    return await api_hush()


@app.post("/api/hush")
async def api_hush():
    await broadcast({"type": "hush"})
    return {"ok": True}


# settings
@app.get("/api/settings")
async def api_settings():
    return store.settings_dict()


@app.post("/api/settings")
async def api_settings_update(payload: dict):
    for k, v in payload.items():
        store.set(k, v)
    return {"ok": True, "revision": store.revision()}


# roster
@app.get("/api/roster")
async def api_roster():
    return [{"id": r.id, **r.__dict__} for r in store.roster_list()]


@app.post("/api/roster")
async def api_roster_add(payload: dict):
    rid = store.roster_add(
        payload.get("team", ""), payload.get("task", ""),
        payload.get("owner", ""), payload.get("backup", "")
    )
    return {"ok": True, "id": rid}


@app.post("/api/roster/{rid}")
async def api_roster_update(rid: int, payload: dict):
    store.roster_update(rid, **payload)
    return {"ok": True}


@app.delete("/api/roster/{rid}")
async def api_roster_delete(rid: int):
    store.roster_delete(rid)
    return {"ok": True}


# members / face / voiceprint
@app.get("/api/members")
async def api_members():
    members = store.member_list()
    out = []
    for m in members:
        d = {"id": m.id, "name": m.name, "face": [], "voiceprint": False}
        if m.name in face_pipe.db:
            d["face"] = [{"pose": (rec.get("pose") if isinstance(rec, dict) else None), "q": (rec.get("q") if isinstance(rec, dict) else None)} for rec in face_pipe.db[m.name]]
        d["voiceprint"] = m.name in voiceprint.db
        out.append(d)
    return out


@app.post("/api/members")
async def api_member_add(payload: dict):
    name = (payload.get("name") or "").strip()
    if not name:
        return {"ok": False, "error": "empty name"}
    mid = store.member_add(name)
    return {"ok": True, "id": mid}


@app.delete("/api/members/{mid}")
async def api_member_delete(mid: int):
    for m in store.member_list():
        if m.id == mid:
            face_pipe.delete_person(m.name)
            if m.name in voiceprint.db:
                del voiceprint.db[m.name]
                voiceprint._save()
            store.member_delete(mid)
            return {"ok": True}
    return {"ok": False, "error": "not found"}


@app.post("/api/members/{mid}/face")
async def api_face_register(mid: int, payload: dict):
    for m in store.member_list():
        if m.id == mid:
            result = face_pipe.register(m.name, payload.get("img", ""), pose=payload.get("pose", "front"))
            return {"ok": result["ok"], **result}
    return {"ok": False, "error": "member not found"}


def _read_wav_bytes(b64: str) -> np.ndarray | None:
    try:
        raw = base64.b64decode(b64)
        with wave.open(io.BytesIO(raw), "rb") as w:
            if w.getnchannels() != 1 or w.getframerate() != 16000:
                return None
            frames = w.readframes(w.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception as e:
        print(f"[voiceprint] wav 解析失败: {e}")
        return None


@app.post("/api/members/{mid}/voiceprint")
async def api_voiceprint_register(mid: int, payload: dict):
    for m in store.member_list():
        if m.id == mid:
            samples = _read_wav_bytes(payload.get("wav", ""))
            if samples is None:
                return {"ok": False, "error": "wav must be 16kHz mono PCM16"}
            ok = voiceprint.register(samples, m.name)
            return {"ok": ok, "count": len(voiceprint.db.get(m.name, []))}
    return {"ok": False, "error": "member not found"}


# personas
@app.get("/api/personas")
async def api_personas():
    return store.persona_list()


@app.post("/api/personas")
async def api_persona_upsert(payload: dict):
    store.persona_upsert(payload.get("name", ""), payload.get("system_prompt", ""), payload.get("is_default", False))
    return {"ok": True}


# model provider
@app.get("/api/provider")
async def api_provider():
    p = store.provider_default()
    if p:
        p = dict(p)
        p.pop("api_key", None)  #  never return key
    return p or {}


@app.post("/api/provider")
async def api_provider_upsert(payload: dict):
    name = payload.get("name", "")
    existing = store.provider_default() or {}
    key = payload.get("api_key")
    if not key and existing.get("name") == name:
        key = existing.get("api_key", "")
    store.provider_upsert(
        name, payload.get("base_url", ""), key or "",
        payload.get("model", ""), payload.get("timeout_ms", 1500), payload.get("is_default", False)
    )
    return {"ok": True}


# intents
@app.get("/api/intents")
async def api_intents():
    return store.intent_list()


@app.post("/api/intents/test")
async def api_intent_test(payload: dict):
    return brain.test_intent(payload.get("text", ""))


@app.post("/api/intents")
async def api_intent_add(payload: dict):
    iid = store.intent_add(
        payload.get("name", ""), payload.get("pattern", ""), payload.get("template", ""),
        payload.get("priority", 0)
    )
    brain.reload_intents()
    return {"ok": True, "id": iid}


@app.post("/api/intents/{iid}")
async def api_intent_update(iid: int, payload: dict):
    store.intent_update(iid, **payload)
    brain.reload_intents()
    return {"ok": True}


@app.delete("/api/intents/{iid}")
async def api_intent_delete(iid: int):
    store.intent_delete(iid)
    brain.reload_intents()
    return {"ok": True}


# provider test
@app.post("/api/provider/test")
async def api_provider_test():
    provider = store.provider_default() or {}
    base = (provider.get("base_url") or "").rstrip("/")
    if not base:
        return {"ok": False, "error": "未配置 provider"}
    key = provider.get("api_key", "")
    model = provider.get("model", "deepseek-chat")
    timeout_ms = provider.get("timeout_ms", 1500)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
        "stream": False,
    })
    req = urllib.request.Request(
        base + "/chat/completions",
        data=body.encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_ms / 1000.0) as resp:
            resp.read()
        return {"ok": True, "latency_ms": round((time.time() - t0) * 1000, 1)}
    except Exception as e:
        return {"ok": False, "error": str(e), "latency_ms": round((time.time() - t0) * 1000, 1)}


# playbook generate-all
@app.post("/api/playbook/generate-all")
async def api_playbook_generate_all():
    results = []
    for row in store.playbook_list():
        ok = await pregenerate_playbook(row.id)
        results.append({"id": row.id, "ok": ok})
    return {"ok": True, "results": results}


# skills - P2
@app.get("/api/skills")
async def api_skills():
    return store.skill_list()


@app.post("/api/skills/{name}")
async def api_skill_update(name: str, payload: dict):
    store.skill_set(name, payload.get("enabled"), payload.get("config"))
    return {"ok": True}


# playbook - P2 skeleton
@app.get("/api/playbook")
async def api_playbook():
    return [{"id": p.id, **p.__dict__} for p in store.playbook_list()]


@app.post("/api/playbook")
async def api_playbook_add(payload: dict):
    pid = store.playbook_add(payload.get("text", ""), payload.get("cron", ""), payload.get("audio_path", ""), payload.get("enabled", 1))
    return {"ok": True, "id": pid}


@app.post("/api/playbook/{pid}")
async def api_playbook_update(pid: int, payload: dict):
    store.playbook_update(pid, **payload)
    return {"ok": True}


@app.delete("/api/playbook/{pid}")
async def api_playbook_delete(pid: int):
    store.playbook_delete(pid)
    return {"ok": True}


@app.get("/api/events")
async def api_events(n: int = 30):
    """返回最近 n 条审计事件。"""
    lines = []
    if EVENTS_FILE.exists():
        with open(EVENTS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    events = []
    for line in lines[-n:]:
        try:
            events.append(json.loads(line))
        except Exception:
            pass
    return events



@app.post("/api/playbook/{pid}/generate")
async def api_playbook_generate(pid: int):
    ok = await pregenerate_playbook(pid)
    return {"ok": ok}


# --------------------------------------------------------------------------- WebSocket
ptt_pipe = AudioPipe(voiceprint, False)


@app.websocket("/ws")
async def ws_handler(ws: WebSocket):
    global _event_loop
    if _event_loop is None:
        _event_loop = asyncio.get_running_loop()

    await ws.accept()
    clients.add(ws)
    log_event("server", {"type": "connect"})

    mode, vp_enabled, vp_threshold = current_audio_config()
    await ws.send_json({"type": "config", "audio_mode": mode, "voiceprint_enabled": vp_enabled})
    await ws.send_json({"type": "tts_mode", "mode": "server" if get_tts_pipe().ok else "browser", "engine": get_tts_pipe().engine})
    if MOCK:
        await ws.send_json({"type": "reply", "text": "MOCK 链路已就绪，先用下方文字框问我。", "person": None})

    ws_state = {"mode": mode, "ptt_active": False, "ptt_buffer": b""}
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                await _handle_audio(ws, ws_state, msg["bytes"])
            elif msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                t = data.get("type")
                if t == "text":
                    await handle_user_text(data.get("text", ""), source="text")
                elif t == "shot":
                    await _handle_shot(data.get("img", ""))
                elif t == "register":
                    await _handle_register(ws, data)
                elif t == "mode":
                    await _handle_mode(ws, ws_state, data.get("mode", "ptt"))
                elif t == "ptt_start":
                    await _handle_ptt_start(ws_state)
                elif t == "ptt_stop":
                    await _handle_ptt_stop(ws_state)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws] 连接异常: {e}")
    finally:
        clients.discard(ws)


async def _handle_audio(_ws, state: dict, pcm16: bytes):
    if state["mode"] == "ptt":
        if state["ptt_active"]:
            state["ptt_buffer"] += pcm16
        return

    # voiceprint / hybrid continuous
    texts = await asyncio.to_thread(get_continuous_pipe().feed, pcm16)
    for text in texts:
        await handle_user_text(text, source="asr")


async def _handle_ptt_start(state: dict):
    state["ptt_active"] = True
    state["ptt_buffer"] = b""
    await broadcast({"type": "ptt", "status": "start"})


async def _handle_ptt_stop(state: dict):
    state["ptt_active"] = False
    buf = state["ptt_buffer"]
    state["ptt_buffer"] = b""
    await broadcast({"type": "ptt", "status": "stop"})
    if not buf:
        return
    texts = await asyncio.to_thread(ptt_pipe.feed, buf)
    texts += await asyncio.to_thread(ptt_pipe.flush)
    for text in texts:
        await handle_user_text(text, source="asr")


async def _handle_mode(ws, state: dict, mode: str):
    mode = mode if mode in ("ptt", "voiceprint", "hybrid") else "ptt"
    state["mode"] = mode
    store.set("audio_mode", mode)
    get_continuous_pipe()  # 重建管道
    await ws.send_json({"type": "config", "audio_mode": mode, "voiceprint_enabled": store.get_bool("voiceprint_enabled")})


async def _handle_register(ws, data: dict):
    pose = data.get("pose", "front")
    res = face_pipe.register(data.get("name", ""), data.get("img", ""), pose=pose)
    await ws.send_json({"type": "register_result", **res})


async def handle_user_text(text: str, source: str):
    text = (text or "").strip()
    if not text:
        return
    log_event("user", {"source": source, "text": text, "identity": tracker.current})
    if source == "asr":
        await broadcast({"type": "asr", "text": text})
    reply = await asyncio.to_thread(brain.respond, text, tracker.current)
    await broadcast({"type": "reply", "text": reply, "person": tracker.current})
    await broadcast_audio(reply)


async def _handle_shot(img_b64: str):
    if not img_b64 or not face_pipe.ok:
        return
    # 同步调用（人脸检测较轻）
    name, score = await asyncio.to_thread(face_pipe.identify, img_b64)
    event = tracker.update(name)
    if event == "leave":
        log_event("face", {"type": "leave"})
        await broadcast({"type": "identity", "name": None, "score": 0})
    elif event:
        log_event("face", {"type": "enter", "name": event, "score": round(score, 3)})
        await broadcast({"type": "identity", "name": event, "score": round(score, 3)})
        greeting = brain.greet(event)
        await broadcast({"type": "reply", "text": greeting, "person": event})
        await broadcast_audio(greeting)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
