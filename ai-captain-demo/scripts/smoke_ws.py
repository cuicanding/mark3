"""无头冒烟验证：连 WS -> 发文字 -> 收 reply（+ 可选 WAV）。
用法: python scripts/smoke_ws.py [ws://localhost:8000/ws] ["问题文本"]
退出码 0 = 收到 reply；用于 L1/L2 验收时不开浏览器先验证链路。
"""
import asyncio
import json
import sys

import websockets


async def main() -> int:
    uri = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/ws"
    text = sys.argv[2] if len(sys.argv) > 2 else "小队长，当前支付成功率是多少"
    got_reply, got_audio = None, 0

    async with websockets.connect(uri, proxy=None) as ws:
        await ws.send(json.dumps({"type": "text", "text": text}))
        print(f">> 已发送: {text}")
        deadline = asyncio.get_event_loop().time() + 15
        while asyncio.get_event_loop().time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=3)
            except asyncio.TimeoutError:
                if got_reply:
                    break
                continue
            if isinstance(msg, bytes):
                got_audio = len(msg)
                print(f"<< [binary] WAV {got_audio} bytes")
                continue
            d = json.loads(msg)
            print(f"<< [{d.get('type')}] {d.get('text') or d.get('mode') or d.get('bytes') or ''}")
            if d.get("type") == "reply":
                got_reply = d.get("text", "")

    print(f"结果: reply={'OK' if got_reply else 'MISSING'}  audio={got_audio} bytes")
    return 0 if got_reply else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
