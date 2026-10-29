"""Safe MiniMind tool-call parsing and deterministic mock execution."""

from __future__ import annotations

import ast
import json
import operator
import re
from typing import Any

TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
]

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}
_UNIT = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048, "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556}
_WEATHER = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"), "London": ("5°C", "小雨")}
_TIME = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00", "Europe/London": "2025-03-07 06:30:00"}
_RATES = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92}
_TRANSLATE = {("hello world", "chinese"): "你好世界", ("good morning", "chinese"): "早上好", ("你好世界", "english"): "Hello World", ("机器学习很有趣", "english"): "Machine learning is interesting"}


def safe_math_eval(expression: str) -> float | int:
    def evaluate(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](evaluate(node.operand))
        raise ValueError("unsupported math expression")

    return evaluate(ast.parse(str(expression), mode="eval"))


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls = []
    for raw in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            call = json.loads(raw.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict):
            calls.append(call)
    return calls


def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    try:
        if name == "calculate_math":
            return {"result": str(safe_math_eval(args.get("expression", "0")))}
        if name == "unit_converter":
            value = float(args.get("value", 0))
            factor = _UNIT.get(f"{args.get('from_unit', '')}_{args.get('to_unit', '')}", 1.0)
            return {"result": round(value * factor, 4)}
        if name == "get_current_weather":
            location = str(args.get("location", ""))
            temp, condition = _WEATHER.get(location, ("22°C", "晴"))
            return {"city": location, "temperature": temp, "condition": condition}
        if name == "get_current_time":
            tz = str(args.get("timezone", "Asia/Shanghai"))
            return {"datetime": _TIME.get(tz, "2025-03-07 14:30:00"), "timezone": tz}
        if name == "get_exchange_rate":
            key = (str(args.get("from_currency", "")), str(args.get("to_currency", "")))
            return {"from": key[0], "to": key[1], "rate": _RATES.get(key, 1.0)}
        if name == "translate_text":
            key = (str(args.get("text", "")).lower(), str(args.get("target_language", "")).lower())
            return {"translated_text": _TRANSLATE.get(key, str(args.get("text", "")))}
    except (ValueError, TypeError, ZeroDivisionError):
        return None
    return None
