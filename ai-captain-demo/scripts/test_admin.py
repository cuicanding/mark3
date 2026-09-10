"""后台管理 API 集成测试（无浏览器、无真实模型）。
用法: python scripts/test_admin.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
import server


def test_lifecycle():
    client = TestClient(server.app)

    s = client.get("/api/state")
    assert s.status_code == 200
    print("state", s.json()["tts_engine"], s.json()["audio_mode"])

    client.post("/api/settings", json={"face_threshold": "0.5"})
    assert client.get("/api/settings").json()["face_threshold"] == "0.5"

    # roster
    r = client.post("/api/roster", json={"team": "冒烟", "task": "API 测试", "owner": "我", "backup": "他"})
    rid = r.json()["id"]
    rows = client.get("/api/roster").json()
    assert any(row["id"] == rid for row in rows)
    client.delete(f"/api/roster/{rid}")

    # members
    m = client.post("/api/members", json={"name": "测试成员"})
    mid = m.json()["id"]
    assert any(member["id"] == mid for member in client.get("/api/members").json())
    client.delete(f"/api/members/{mid}")

    # intents
    i = client.post("/api/intents", json={"name": "测试意图", "pattern": r"测试意图", "template": "匹配成功 {identity}"})
    assert i.status_code == 200
    iid = i.json()["id"]
    test = client.post("/api/intents/test", json={"text": "这是测试意图句子"})
    assert test.json()["matched"] is True
    assert "匹配成功" in test.json()["rendered"]
    client.delete(f"/api/intents/{iid}")

    # provider key 保留
    client.post("/api/provider", json={"name": "deepseek", "base_url": "https://a", "api_key": "super-secret", "model": "m", "is_default": True})
    client.post("/api/provider", json={"name": "deepseek", "base_url": "https://a", "api_key": "", "model": "m2", "is_default": True})
    assert server.store.provider_default()["api_key"] == "super-secret"

    print("test_admin 全部通过")


if __name__ == "__main__":
    test_lifecycle()
