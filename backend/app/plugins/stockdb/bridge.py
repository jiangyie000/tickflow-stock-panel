"""stockdb 桥接:通过 HTTP 调 stockdb K-V 数据库服务。

URL 协议: http://{STOCKDB_URL}/?cmd={get|vals|keys}&t={table}:{key}:{range}
  - cmd=get  → dict (单条) 或 None
  - cmd=vals → list[dict] (匹配的全部值)
  - cmd=keys → list[str] (匹配的全部 key)
"""
from __future__ import annotations

import os

import httpx

_DEFAULT_URL = "http://192.168.31.145:7899"
STOCKDB_URL = os.getenv("STOCKDB_URL", _DEFAULT_URL).rstrip("/")
_TIMEOUT = 30.0


class StockDBBridgeError(RuntimeError):
    """桥接调用失败(连不上 / 非 2xx / JSON 解析失败)。"""


def call(cmd: str, t: str) -> object:
    """调一次 stockdb,返回 JSON 解析后的对象(可能 dict / list / None)。"""
    url = f"{STOCKDB_URL}/?cmd={cmd}&t={t}"
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        raise StockDBBridgeError(f"stockdb HTTP 失败: {url}: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise StockDBBridgeError(f"stockdb 调 {cmd} {t} 失败: {e}") from e
    return data


def availability() -> tuple[bool, str]:
    """探活: 调一次简单查询,成功即 OK。返回 (是否可用, 原因)。"""
    try:
        call("get", "日k:600000:20260101")
    except StockDBBridgeError as e:
        return False, f"{e} (STOCKDB_URL={STOCKDB_URL})"
    return True, f"ok (stockdb @ {STOCKDB_URL})"
