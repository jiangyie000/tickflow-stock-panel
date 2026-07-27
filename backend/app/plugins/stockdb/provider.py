"""stockdb 数据源 provider。

通过 bridge 调 stockdb HTTP 服务拉日K/复权/分钟/实时/标的,
调用 normalizer.* 归一化到内部 schema。方法签名对齐
GenericHTTPProvider / StockSDKProvider, services 层路由零改动。

stockdb 表/字段约定:
  日k    :{amount, amplitude, close, code, date(YYYYMMDD),
           float_mv, float_share, high, is_st, low, name, open, pb,
           pct_chg, pe_ttm, pre_close, total_mv, total_share,
           turnover, vol_ratio, volume}
  分钟k  :{amount, close, code, date(YYYYMMDDHHMMSS),
           high, low, open, volume}
  复权   :{div, give, trans, mult, cum}   键 = "复权:code:date(YYYYMMDD)"
  股票代码:{<市场前缀>: [code, ...]}    0/3/6/9/4/8 等
  退市   :list[code]

注意:
  - 这版 stockdb 行为: `vals(日k, *, date)` 不工作(全市场通配符失效),
    `<` 范围语法不工作,只能用 per-symbol + 日期前缀通配 `YYYYMM*`。
  - realtime 没独立表,且无全市场快照,只能 per-symbol 抽样几只活跃股
    取"最近日"做盘后行情参考(主力指数已包含在抽样里)。
  - 复权表只存除权除息日(非全交易日), 下游需 carry-forward。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import polars as pl

from app.data_providers.normalizer import normalize_adj_factors, normalize_daily
from app.plugins.stockdb import bridge
from app.tickflow.rate_limits import chunked

logger = logging.getLogger(__name__)

_DATASETS = ("daily", "adj_factor", "minute", "realtime", "instruments")
_BATCH = 40  # 每批 symbols 数量,与 stocksdk 保持一致

# realtime 抽样股票: stockdb 没法一次拉全市场快照,只能 per-symbol 拼。
# 抽样几只活跃股取"最近日"作为盘后行情快照,主力指数(000001.SH / 399001.SZ / 399006.SZ)
# 一定要在里面。完整全市场需走别处(见 get_realtime docstring)。
_REALTIME_SAMPLE = [
    "600519",  # 贵州茅台
    "600000",  # 浦发
    "000001",  # 平安
    "000002",  # 万科A
    "300750",  # 宁德
    "601318",  # 中国平安
]

# 股票代码表 → (exchange, asset_type)
_MARKET_PREFIX = {
    "0": ("SZ", "stock"),   # 000xxx 主板 / 002xxx 中小
    "2": ("SZ", "stock"),   # 20xxxx B 股
    "3": ("SZ", "stock"),   # 300xxx 创业板 / 301xxx
    "1": ("SZ", "fund"),
    "5": ("SH", "fund"),
    "6": ("SH", "stock"),   # 600/601/603/605
    "9": ("SH", "stock"),   # 688 科创板
    "4": ("BJ", "stock"),   # 北交所
    "8": ("BJ", "stock"),
    "92": ("BJ", "stock"),
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def code_to_symbol(code: str) -> str:
    """裸 code → app 符号 (600xxx.SH / 000xxx.SZ / 4xxxx.BJ)。"""
    c = str(code)
    if c.startswith(("60", "68", "5", "9", "11", "13")):
        return f"{c}.SH"
    if c.startswith(("00", "30", "12", "15", "20")):
        return f"{c}.SZ"
    if c.startswith(("4", "8", "92")):
        return f"{c}.BJ"
    return f"{c}.SH"  # 兜底


def _ymd_int_to_iso(d: int | str) -> str | None:
    """YYYYMMDD (int 或 8 位字符串) → 'YYYY-MM-DD'。polars.Date 不能直接吃 YYYYMMDD int
    (会按 epoch days 解释成年份 57441),必须先转 ISO 串再让 normalize 解析。"""
    s = str(d)
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return None


def _safe_call(cmd: str, t: str) -> Any:
    """调 bridge.call,失败不抛异常,只 warn 一次。"""
    try:
        return bridge.call(cmd, t)
    except bridge.StockDBBridgeError as e:
        logger.warning("stockdb 调 %s %s 失败: %s", cmd, t, e)
        return None


def _as_list(data: Any) -> list[dict]:
    """vals 多数情况返 list,get 单条返 dict。统一成 list[dict]。"""
    if data is None:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _extract_key_date(key: str, prefix: str) -> str | None:
    """从 '复权:600702:20260527' 这种 key 抽 8 位日期。"""
    parts = key.split(":")
    if len(parts) < 3:
        return None
    d = parts[-1]
    if len(d) >= 8 and d[:8].isdigit():
        return d[:8]
    return None


# ---------------------------------------------------------------------------
# Config shim
# ---------------------------------------------------------------------------

@dataclass
class _StockDBConfig:
    name: str = "stockdb"
    display_name: str = "stockdb（本地数据库）"
    datasets: dict = field(default_factory=lambda: dict.fromkeys(_DATASETS))
    path: None = None
    builtin: bool = True


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class StockDBProvider:
    """stockdb 内置数据源。"""

    name = "stockdb"
    builtin = True

    def __init__(self) -> None:
        self.config = _StockDBConfig()

    def close(self) -> None:  # loader.load_all 会对每个 provider 调 close
        pass

    # ---- daily ----
    # 这版 stockdb 行为:
    #   - 单股单日 `vals(日k, code, YYYYMMDD)` 正常
    #   - 日期前缀通配 `vals(日k, code, YYYYMM*)` 正常 (整月)
    #   - 全市场通配 `vals(日k, *, *)` 不工作 (返 0)
    #   - `<` 范围 `vals(日k, code, A<B)` 不工作 (返 0)
    # 所以只能 per-symbol, 把 [start,end] 拆成 YYYYMM 列表逐月拉, 内存里再按日期裁剪。
    def get_daily(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",  # noqa: ARG002
        on_chunk_done=None,
    ) -> pl.DataFrame:
        if not start_time or not end_time or not symbols:
            return pl.DataFrame()
        s_int = int(start_time.strftime("%Y%m%d"))
        e_int = int(end_time.strftime("%Y%m%d"))
        # 拆月份: [2025-09-01, 2026-07-25] → [202509, 202510, ..., 202607]
        months: list[str] = []
        cur_y, cur_m = start_time.year, start_time.month
        end_y, end_m = end_time.year, end_time.month
        while (cur_y, cur_m) <= (end_y, end_m):
            months.append(f"{cur_y:04d}{cur_m:02d}")
            cur_m += 1
            if cur_m > 12:
                cur_m = 1
                cur_y += 1
        frames: list[pl.DataFrame] = []
        chunks = list(chunked(symbols, _BATCH))
        total = len(chunks) * len(months)
        done = 0
        for chunk in chunks:
            for sym in chunk:
                code = sym.split(".")[0]
                for m in months:
                    data = _safe_call("vals", f"日k:{code}:{m}*")
                    rows: list[dict] = []
                    for item in _as_list(data):
                        d_int = item.get("date")
                        if isinstance(d_int, int) and not (s_int <= d_int <= e_int):
                            continue
                        row = dict(item)
                        # stockdb date 是 int YYYYMMDD → 转 ISO 让 polars.Date 正确解析
                        # (int 20260701 直接 cast pl.Date 会被当 epoch days 解释成年份 57441)
                        if "date" in row and isinstance(row["date"], int):
                            iso = _ymd_int_to_iso(row["date"])
                            if iso:
                                row["date"] = iso
                        row["symbol"] = sym
                        rows.append(row)
                    if rows:
                        df = normalize_daily(rows, source=self.name)
                        if not df.is_empty():
                            frames.append(df)
                done += 1
                if on_chunk_done:
                    on_chunk_done(done, total)
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    # ---- adj_factor ----
    # 复权表 sparse: 只有除权除息日有记录,key 末段是 date。
    # 这版 stockdb 不支持 < 范围,用 `*` 拉全量再 Python 里过滤。
    def get_adj_factors(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",  # noqa: ARG002
        on_chunk_done=None,
    ) -> pl.DataFrame:
        if not symbols:
            return pl.DataFrame()
        s_int = int(start_time.strftime("%Y%m%d")) if start_time else 0
        e_int = int(end_time.strftime("%Y%m%d")) if end_time else 9_999_999_99
        frames: list[pl.DataFrame] = []
        chunks = list(chunked(symbols, _BATCH))
        for i, chunk in enumerate(chunks):
            for sym in chunk:
                code = sym.split(".")[0]
                vals = _safe_call("vals", f"复权:{code}:*")
                keys = _safe_call("keys", f"复权:{code}:*") or []
                rows: list[dict] = []
                for k, item in zip(keys, _as_list(vals)):
                    td_str = _extract_key_date(str(k), "复权")
                    if not td_str:
                        continue
                    td_int = int(td_str)
                    if not (s_int <= td_int <= e_int):
                        continue
                    cum = item.get("cum")
                    if cum is None:
                        continue
                    rows.append({
                        "symbol": sym,
                        "trade_date": _ymd_int_to_iso(td_str) or td_str,
                        "ex_factor": cum,
                    })
                if rows:
                    df = normalize_adj_factors(rows, source=self.name)
                    if not df.is_empty():
                        frames.append(df)
            if on_chunk_done:
                on_chunk_done(i + 1, len(chunks))
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    # ---- minute ----
    # 分钟 K 也只能 per-symbol; `*` 拉全量再按 datetime 过滤。
    def get_minute(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",  # noqa: ARG002
        on_chunk_done=None,
        freq: str = "5m",  # noqa: ARG002 (stockdb 端 freq 不可控,数据原样返回)
    ) -> pl.DataFrame:
        if not symbols:
            return pl.DataFrame()
        s_int = int(start_time.strftime("%Y%m%d%H%M%S")) if start_time else 0
        e_int = int(end_time.strftime("%Y%m%d%H%M%S")) if end_time else 9_999_999_999_999
        _MINUTE_CANONICAL = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]

        frames: list[pl.DataFrame] = []
        chunks = list(chunked(symbols, _BATCH))
        for i, chunk in enumerate(chunks):
            for sym in chunk:
                code = sym.split(".")[0]
                data = _safe_call("vals", f"分钟k:{code}:*")
                items = _as_list(data)
                # 按 datetime 范围过滤
                if start_time or end_time:
                    items = [
                        x for x in items
                        if isinstance(x.get("date"), int)
                        and s_int <= x["date"] <= e_int
                    ]
                if not items:
                    continue
                df = pl.DataFrame(items)
                if "date" in df.columns:
                    df = df.with_columns(
                        pl.col("date").cast(pl.Utf8)
                        .str.to_datetime("%Y%m%d%H%M%S", strict=False)
                        .alias("datetime")
                    )
                df = df.with_columns(pl.lit(sym).alias("symbol"))
                for col in ("open", "high", "low", "close", "volume", "amount"):
                    if col in df.columns:
                        df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
                keep = [c for c in _MINUTE_CANONICAL if c in df.columns]
                if "datetime" not in keep:
                    continue
                df = df.select(keep)
                if not df.is_empty():
                    frames.append(df)
            if on_chunk_done:
                on_chunk_done(i + 1, len(chunks))
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    # ---- realtime ----
    # stockdb 没独立实时表,且 `vals(日k, *, date)` 不工作 (全市场通配符失效)。
    # per-symbol 拉最新一天: keys(*:code:*) → 末位日期 → vals 取当日 bar。
    def get_realtime(self) -> list[dict]:
        if not _REALTIME_SAMPLE:
            return []
        rows: list[dict] = []
        for code in _REALTIME_SAMPLE:
            keys = _safe_call("keys", f"日k:{code}:*")
            if not keys or not isinstance(keys, list) or not keys:
                continue
            last_key = str(keys[-1])
            parts = last_key.split(":")
            if len(parts) < 3:
                continue
            ymd = parts[-1][:8]
            data = _safe_call("vals", f"日k:{code}:{ymd}")
            for item in _as_list(data):
                c = item.get("code")
                if not c:
                    continue
                rows.append({
                    "symbol": code_to_symbol(c),
                    "name": item.get("name"),
                    "last_price": item.get("close"),
                    "prev_close": item.get("pre_close"),
                    "open": item.get("open"),
                    "high": item.get("high"),
                    "low": item.get("low"),
                    "volume": item.get("volume"),
                    "amount": item.get("amount"),
                    "change_pct": item.get("pct_chg"),
                })
        if rows:
            logger.info("stockdb realtime 快照: 抽样 %d 只, 共 %d 行", len(_REALTIME_SAMPLE), len(rows))
        return rows

    # ---- instruments ----
    # 股票代码表: {<市场前缀>: [code, ...]}。
    # 拉一次全表,展平成 tickflow Instrument 形状。
    def get_instruments(self, asset_type: str = "stock") -> list[dict]:
        if asset_type != "stock":
            return []
        data = _safe_call("get", "股票代码")
        if not isinstance(data, dict):
            return []
        rows: list[dict] = []
        for prefix, codes in data.items():
            ex, at = _MARKET_PREFIX.get(str(prefix), ("SH", asset_type))
            if not isinstance(codes, list):
                continue
            for code in codes:
                sym = code_to_symbol(code)
                rows.append({
                    "symbol": sym,
                    "name": sym,  # 股票代码表无 name 字段,先用 sym 占位
                    "code": str(code),
                    "exchange": ex,
                    "region": "CN",
                    "type": at,
                })
        return rows

    # ---- 测试 (设置页试拉) ----
    def test_dataset(self, dataset: str, symbols: list[str] | None = None) -> dict:
        symbols = symbols or ["600519.SH"]
        if dataset == "daily":
            df = self.get_daily(symbols, datetime(2026, 7, 1), datetime(2026, 7, 24))
            return _preview("daily", df)
        if dataset == "adj_factor":
            df = self.get_adj_factors(symbols, datetime(2026, 1, 1), datetime(2026, 7, 24))
            return _preview("adj_factor", df)
        if dataset == "minute":
            df = self.get_minute(symbols, datetime(2026, 7, 24, 9, 30), datetime(2026, 7, 24, 15, 0))
            return _preview("minute", df)
        if dataset == "realtime":
            rows = self.get_realtime()
            head = rows[:5]
            return {
                "provider": self.name,
                "dataset": "realtime",
                "rows": len(rows),
                "columns": list(head[0].keys()) if head else [],
                "preview": head,
            }
        if dataset == "instruments":
            rows = self.get_instruments()
            head = rows[:5]
            return {
                "provider": self.name,
                "dataset": "instruments",
                "rows": len(rows),
                "columns": list(head[0].keys()) if head else [],
                "preview": head,
            }
        raise ValueError(f"stockdb 不支持数据集: {dataset}")


def _preview(dataset: str, df: pl.DataFrame) -> dict:
    # to_dicts 会对 pl.Date / pl.Datetime 列做内部转换,如果数据源混入
    # 非标准日期(int YYYYMMDD 等)会在 Rust 层 panic("year 57441 is out of range")。
    # 预览不需要日期类型,统一转 Utf8 字符串更安全也方便前端展示。
    safe = df
    for col, dtype in df.schema.items():
        if dtype in (pl.Date, pl.Datetime, pl.Time):
            safe = safe.with_columns(pl.col(col).cast(pl.Utf8, strict=False))
    return {
        "provider": "stockdb",
        "dataset": dataset,
        "rows": df.height,
        "columns": df.columns,
        "preview": safe.head(5).to_dicts() if not df.is_empty() else [],
    }
