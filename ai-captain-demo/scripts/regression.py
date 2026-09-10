"""L1-L4 回归测试（无浏览器、无真实外设）。
用法: python scripts/regression.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from brain import Brain
import server


def main():
    failures = []

    # L4 REST /say
    client = TestClient(server.app)
    r = client.post("/say", json={"text": "值班表情况如何"})
    if r.status_code != 200 or not r.json().get("ok"):
        failures.append(f"REST /say failed: {r.status_code}")
    else:
        print("L4 REST /say OK")

    # L1 对话逻辑
    b = Brain()
    for q in ["小队长，当前支付成功率是多少", "支付组预案巡检到哪了"]:
        answer = b.respond(q, identity="存孝")
        print(f"L1 logic '{q[:20]}...' -> {answer[:40]}...")

    # L5 TTS 状态
    s = client.get("/api/state").json()
    print(f"L5 TTS mode: {s.get('tts_mode')}")

    # API 可用性
    for ep in ["/api/settings", "/api/roster", "/api/intents", "/api/members", "/admin"]:
        if client.get(ep).status_code != 200:
            failures.append(f"GET {ep} failed")

    print("L2 语音 / L3 视觉 需要真实外设，回归脚本未覆盖")

    if not failures:
        print("\n回归测试全部通过")
        return 0
    else:
        print("\n回归失败:", failures)
        return 1


if __name__ == "__main__":
    sys.exit(main())
