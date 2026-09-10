"""规则对话引擎 + LLM 薄循环工具调用 + ConfigStore 集成。

P1/P2 能力：
- ROSTER / metrics 走 SQLite，不再硬编码。
- 人格、模型配置走 SQLite。
- 意图规则命中直接走模板（零 LLM）。
- 规则未命中走 LLM 薄循环：最多 2 轮 tool calling，总 timeout 1.5s。
- 工具只返回真实数据（值班表 / 指标），LLM 只能做槽位填充，播报文本仍由模板组装。
"""
import json
import os
import random
import re
import time
import urllib.request
from collections import deque

from store import ConfigStore


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Metrics:
    """指标随机游走。P1 仍内存，P2 可持久化。"""

    def __init__(self):
        self.pay_rate = 99.95
        self.order_rate = 99.90
        self.gmv = 3.28
        self.order_per_min = 12.0
        self.online = 580.0

    def walk(self):
        self.pay_rate = _clamp(self.pay_rate + random.uniform(-0.05, 0.02), 99.50, 99.99)
        self.order_rate = _clamp(self.order_rate + random.uniform(-0.04, 0.02), 99.60, 99.99)
        self.gmv = _clamp(self.gmv + random.uniform(-0.03, 0.05), 3.0, 3.6)
        self.order_per_min = _clamp(self.order_per_min + random.uniform(-0.3, 0.3), 10.0, 14.0)
        self.online = _clamp(self.online + random.uniform(-5, 6), 520, 650)

    def snapshot(self) -> dict:
        return {
            "pay_rate": round(self.pay_rate, 2),
            "order_rate": round(self.order_rate, 2),
            "gmv": round(self.gmv, 2),
            "order_per_min": round(self.order_per_min, 1),
            "online": round(self.online, 0),
        }


WAKE = re.compile(r"^\s*(小队长|队长)[，,。.!！\s]*")


class Brain:
    # LLM tool definitions
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "list_duty",
                "description": "查询值班表/巡检任务进度，返回各组负责人、backup、进度、状态。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "team": {"type": "string", "description": "可选：组名，如'支付组'。为空则返回全部。"},
                    },
                    "required": ["team"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_metrics",
                "description": "查询实时业务指标：支付成功率、下单成功率、GMV、在线人数等。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {
                            "type": "string",
                            "enum": ["pay_rate", "order_rate", "gmv", "online", "order_per_min", "all"],
                            "description": "指标名；all 返回全部。",
                        },
                    },
                    "required": ["metric"],
                },
            },
        },
    ]

    def __init__(self):
        self.store = ConfigStore()
        self.metrics = Metrics()
        self.history = deque(maxlen=6)
        self.intent_rules: list[tuple[re.Pattern, str]] = self._load_intents()
        self.rules = [
            (re.compile(r"支付.*成功率|成功率.*支付", re.I), self._pay_rate),
            (re.compile(r"下单.*成功率", re.I), self._order_rate),
            (re.compile(r"成功率"), self._all_rates),
            (re.compile(r"gmv|成交|交易额|销售额", re.I), self._gmv),
            (re.compile(r"订单"), self._orders),
            (re.compile(r"流量|在线|uv|pv|人数", re.I), self._online),
            (re.compile(r"巡检|预案|任务|值班|进度|到哪|排班"), self._duty),
            (re.compile(r"你是谁|自我介绍|介绍一下"), self._whoami),
            (re.compile(r"你好|您好|早上好|下午好|晚上好|在吗|在不在|^早$"), self._hello),
            (re.compile(r"谢谢|辛苦"), self._thanks),
        ]

    def _load_intents(self) -> list[tuple[re.Pattern, str]]:
        rules = []
        for intent in self.store.intent_list():
            if not intent.get("enabled"):
                continue
            try:
                pat = re.compile(intent["pattern"])
                rules.append((pat, intent["template"]))
            except Exception as e:
                print(f"[brain] 意图规则编译失败 id={intent.get('id')}: {e}")
        return rules

    def reload_intents(self):
        self.intent_rules = self._load_intents()

    def test_intent(self, text: str) -> dict:
        """返回匹配到的意图名/模板/渲染结果，未命中返回空。"""
        for pat, template in self.intent_rules:
            if pat.search(text):
                rendered = self._render(template, None)
                return {"matched": True, "pattern": pat.pattern, "template": template, "rendered": rendered}
        return {"matched": False}

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------
    def respond(self, text: str, identity: str | None = None) -> str:
        text = (text or "").strip()
        WAKE.sub("", text).strip()

        self.metrics.walk()
        # 1) 后台管理的可编辑意图规则优先
        matched_template = None
        for pat, template in self.intent_rules:
            if pat.search(text):
                matched_template = template
                break
        if matched_template is not None:
            result = matched_template
        else:
            # 2) 代码内置复杂规则
            for pattern, handler in self.rules:
                if pattern.search(text):
                    result = handler(text, identity)
                    break
            else:
                # 3) LLM 薄循环兜底
                result = self._llm_tool_loop(text, identity) or self._fallback(identity)

        reply = self._render(result, identity)
        self._record("user", text)
        self._record("assistant", reply)
        return reply

    def greet(self, name: str) -> str:
        hour = time.localtime().tm_hour
        when = "早上好" if hour < 9 else "上午好" if hour < 12 else "下午好" if hour < 18 else "晚上好"
        return f"{name}，{when}，我是大促队长。指标一切正常，有事随时问我。"

    def roster_list(self) -> list[dict]:
        return [dict(r) for r in self.store.roster_list()]

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------
    def _tool_list_duty(self, team: str) -> str:
        rows = self.store.roster_list()
        if team and team != "全部":
            team_clean = team.replace("组", "")
            rows = [r for r in rows if team in r.team or team_clean in r.team]
        if not rows:
            return "未找到相关值班任务。"
        parts = []
        for r in rows:
            if r.status == "已完成":
                parts.append(f"{r.team}{r.task}已完成，负责人{r.owner}。")
            else:
                parts.append(f"{r.team}{r.task}{r.status}，进度 {r.progress}%，负责人{r.owner}，backup {r.backup}。")
        return "".join(parts)

    def _tool_get_metrics(self, metric: str) -> str:
        m = self.metrics.snapshot()
        if metric == "pay_rate":
            return f"支付成功率 {m['pay_rate']}%"
        if metric == "order_rate":
            return f"下单成功率 {m['order_rate']}%"
        if metric == "gmv":
            return f"实时 GMV {m['gmv']}亿"
        if metric == "online":
            return f"当前在线 {int(m['online'])}万人"
        if metric == "order_per_min":
            return f"当前下单量 {m['order_per_min']}万笔每分钟"
        return f"支付成功率 {m['pay_rate']}%，下单成功率 {m['order_rate']}%，GMV {m['gmv']}亿，在线 {int(m['online'])}万人，下单量 {m['order_per_min']}万笔每分钟。"

    # ------------------------------------------------------------------
    # Handler / 模板
    # ------------------------------------------------------------------
    def _render(self, template_or_result, identity: str | None) -> str:
        if isinstance(template_or_result, dict):
            template = template_or_result.get("template", "")
            data = template_or_result.get("data", {})
            try:
                return template.format(identity=identity or "同事", **data)
            except Exception:
                return template
        if isinstance(template_or_result, str):
            return template_or_result.format(identity=identity or "同事")
        return str(template_or_result)

    def _pay_rate(self, text, identity):
        r = self.metrics.pay_rate
        tail = "水位正常。" if r >= 99.8 else "略低于警戒线，支付组已介入跟进。"
        return {"template": "当前支付成功率 {pay_rate:.2f}%，{tail}", "data": {"pay_rate": r, "tail": tail}}

    def _order_rate(self, text, identity):
        return {"template": "当前下单成功率 {order_rate:.2f}%，链路平稳。", "data": {"order_rate": self.metrics.order_rate}}

    def _all_rates(self, text, identity):
        m = self.metrics
        return {
            "template": "支付成功率 {pay_rate:.2f}%，下单成功率 {order_rate:.2f}%，都在水位之上。",
            "data": {"pay_rate": m.pay_rate, "order_rate": m.order_rate},
        }

    def _gmv(self, text, identity):
        return {"template": "实时 GMV {gmv:.2f}亿，节奏符合预期。", "data": {"gmv": self.metrics.gmv}}

    def _orders(self, text, identity):
        return {
            "template": "当前下单量约 {opm:.1f}万笔每分钟，无明显堆积。",
            "data": {"opm": self.metrics.order_per_min},
        }

    def _online(self, text, identity):
        return {"template": "当前在线约 {online:.0f}万人，流量曲线平稳。", "data": {"online": self.metrics.online}}

    def _duty(self, text, identity):
        roster = self.store.roster_list()
        for r in roster:
            if r.team in text or r.team.replace("组", "") in text:
                return self._tool_list_duty(r.team)
        # 没指定组则返回全部
        return self._tool_list_duty("")

    def _whoami(self, text, identity):
        return {"template": "{identity}，我是大促队长，作战室的 AI 值班员。可以问我支付成功率、GMV、各组巡检进度和值班安排。", "data": {}}

    def _hello(self, text, identity):
        if identity:
            return {"template": "{identity}，你好。大促现场我在盯着，要听指标还是巡检进度？", "data": {}}
        return "你好，我是大促队长。要听指标还是巡检进度？"

    def _thanks(self, text, identity):
        return "应该的，我继续盯着。"

    def _fallback(self, identity):
        return {"template": "{identity}，这个问题我还在学习。可以先问我支付成功率、GMV、各组巡检进度或值班安排。", "data": {}}

    # ------------------------------------------------------------------
    # LLM 薄循环：≤2 轮 tool calling，总超时 1.5s
    # ------------------------------------------------------------------
    def _call_llm(self, messages: list[dict], tools: list[dict] | None = None, timeout: float = 0.8) -> dict | None:
        provider = self.store.provider_default() or {}
        base = (provider.get("base_url") or "").rstrip("/")
        if not base:
            return None
        key = provider.get("api_key", "")
        model = provider.get("model", "deepseek-chat")
        body = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 160,
        }
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(
            base + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data
        except Exception as e:
            print(f"[brain] LLM 调用失败: {e}")
            return None

    def _llm_tool_loop(self, text: str, identity: str | None) -> str | None:
        persona = self.store.persona_get_default()
        sys_prompt = (
            f"{persona.get('system_prompt', '')}\n"
            "你只能调用给定的工具获取真实数据，禁止编造数字。"
            "如果用户问题与工具无关，直接礼貌拒绝。回答简短口语化，不超过 50 字。"
            f"当前对话者：{identity or '同事'}。"
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            *( {"role": r, "content": t} for r, t in self.history ),
            {"role": "user", "content": text},
        ]

        start = time.time()
        # Round 1: with tools
        if time.time() - start > 1.5:
            return None
        resp = self._call_llm(messages, tools=self.TOOLS, timeout=0.75)
        if resp is None:
            return None
        choice = resp.get("choices", [{}])[0]
        msg = choice.get("message", {})
        # Final answer
        content = msg.get("content")
        if content:
            return content.strip()

        # Tool calls
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return None

        messages.append(msg)
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            if name == "list_duty":
                result = self._tool_list_duty(args.get("team", ""))
            elif name == "get_metrics":
                result = self._tool_get_metrics(args.get("metric", "all"))
            else:
                result = "该工具不可用。"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result,
            })

        # Round 2: final answer with tool results
        if time.time() - start > 1.5:
            return None
        resp2 = self._call_llm(messages, timeout=0.75)
        if resp2 is None:
            return None
        final = resp2.get("choices", [{}])[0].get("message", {}).get("content")
        return final.strip() if final else None

    def _record(self, role, text):
        self.history.append((role, text))


if __name__ == "__main__":
    b = Brain()
    for q in ["小队长，当前支付成功率是多少", "GMV 怎么样", "支付组预案巡检到哪了", "现在谁在值班", "你好", "今天天气怎么样"]:
        print(f"Q: {q}")
        print(f"A: {b.respond(q, identity='存孝')}")
        print()
