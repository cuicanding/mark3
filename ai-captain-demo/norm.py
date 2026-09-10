"""数字/百分比 -> 中文读法（TTS 播报归一化）。

规则（照 POC 文档示例）：
    "3.28亿"  -> "三点二八亿"      （带 万/亿 单位时逐位读）
    "99.95%"  -> "百分之九十九点九五"
    "12"      -> "十二"
"""
import re

D = "零一二三四五六七八九"


def read_digits(s: str) -> str:
    """逐位读：328 -> 三二八"""
    return "".join(D[int(c)] for c in s)


def read_int(n: int) -> str:
    """0 ~ 9999 中文读法：99 -> 九十九，105 -> 一百零五"""
    assert 0 <= n <= 9999, f"read_int out of range: {n}"
    if n < 10:
        return D[n]
    units = ["", "十", "百", "千"]
    digits = [int(c) for c in str(n)]
    out, zero = [], False
    for i, d in enumerate(digits):
        pos = len(digits) - 1 - i
        if d == 0:
            zero = True
            continue
        if zero:
            out.append("零")
            zero = False
        out.append(D[d] + units[pos])
    s = "".join(out)
    if s.startswith("一十"):  # 一十五 -> 十五
        s = s[1:]
    return s


def num_to_zh(num: str, digit_wise: bool = False) -> str:
    """"99.95" -> 九十九点九五；"3.28" digit_wise -> 三点二八"""
    if "." in num:
        ip, fp = num.split(".", 1)
        head = read_digits(ip) if digit_wise else read_int(int(ip))
        return head + "点" + read_digits(fp)
    return read_digits(num) if digit_wise else read_int(int(num))


_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|％)")
_BIG = re.compile(r"(\d+(?:\.\d+)?)\s*([亿万])")
_NUM = re.compile(r"\d+(?:\.\d+)?")


def normalize_for_tts(text: str) -> str:
    """把文本中的数字替换成中文读法，供 TTS 播报。"""
    text = _PCT.sub(lambda m: "百分之" + num_to_zh(m.group(1)), text)
    text = _BIG.sub(lambda m: num_to_zh(m.group(1), digit_wise="." in m.group(1)) + m.group(2), text)
    text = _NUM.sub(lambda m: num_to_zh(m.group(0)), text)
    return text


if __name__ == "__main__":
    cases = {
        "支付成功率 99.95%": "支付成功率 百分之九十九点九五",
        "GMV 3.28亿": "GMV 三点二八亿",
        "订单量 12万": "订单量 十二万",
        "进度 80%": "进度 百分之八十",
        "100.0%": "百分之一百点零",
        "在线 580万人": "在线 五百八十万人",
    }
    for src, want in cases.items():
        got = normalize_for_tts(src)
        assert got == want, f"{src!r}: got {got!r}, want {want!r}"
        print(f"OK  {src}  ->  {got}")
    print("norm.py 全部用例通过")
