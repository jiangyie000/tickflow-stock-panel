"""stockdb 桥接: 通过本地 SDK 直连 stockdb K-V 数据库服务。

SDK (stock_sdk) 通过编译二进制 (stockdb.so) 直连本地 LevelDB,
默认地址 127.0.0.1:7899, 本地 IPC 无网络延迟。

模块级 rd 单例由 SDK 维护, 首次调用时懒初始化连接 + 预加载复权因子。
"""
from __future__ import annotations


class StockDBBridgeError(RuntimeError):
    """桥接调用失败(SDK 未安装 / 连不上 stockdb 服务)。"""


def _get_rd():
    """获取 SDK 的 rd 单例, 懒初始化。

    放在函数内调用而非模块顶层, 避免 import 时就触发连接 ——
    plugin loader 在 _call_check 阶段调 availability() 才真正连接。
    """
    try:
        from stock_sdk import rd  # noqa: F401 (rd 是 proxy, 属性访问才触发)
        return rd
    except ImportError as e:
        raise StockDBBridgeError(
            "stock_sdk 未安装。请运行 stockdb/pybao/安装.py 安装 SDK。"
        ) from e


def availability() -> tuple[bool, str]:
    """探活: 用 SDK 拉一条数据验证连接。返回 (是否可用, 原因)。

    SDK 默认连 127.0.0.1:7899, socket_timeout=1s (localhost),
    探活天然快, 不需要额外短超时控制。
    """
    try:
        rd = _get_rd()
        data = rd.vals("日k", "600519", "*").do()
        if data:
            return True, "ok (stockdb SDK @ 127.0.0.1:7899)"
        return True, "ok (stockdb SDK 连接成功, 暂无数据)"
    except StockDBBridgeError as e:
        return False, str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"{e}"
