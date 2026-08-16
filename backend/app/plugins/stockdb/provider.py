"""stockdb 数据源 provider (SDK 直连版)。

通过本地 SDK (stock_sdk) 直连 stockdb K-V 数据库, 调 normalizer.* 归一化到
内部 schema。方法签名对齐 GenericHTTPProvider / StockSDKProvider, services 层
路由零改动。

SDK 核心接口:
  rd.get_data(codes, start, end, frequency, fq) — 批量拉取日K/分钟K, 内置复权
  rd.get("股票代码").do() — 全市场代码表
  rd.vals / rd.keys — 原始 K-V 查询

stockdb 表/字段约定:
  日k    :{amount, amplitude, close, code, date(YYYYMMDD),
           float_mv, float_share, high, is_st, low, name, open, pb,
           pct_chg, pe_ttm, pre_close, total_mv, total_share,
           turnover, vol_ratio, volume}
  分钟k  :{amount, close, code, date(YYYYMMDDHHMMSS),
           high, low, open, volume}
  复权   :{div, give, trans, mult, cum}   键 = "复权:code:date(YYYYMMDD)"
  股票代码:{<市场前缀>: [code, ...]}    0/3/6/9 等

注意:
  - rd.get_data 内部用 rd.pipe() + mget() 批量拉取, 一次传全量代码列表,
    不需要前缀通配 (stockdb 的 K1 前缀通配不工作)。
  - SDK StockDBClient 在 __init__ 时预加载全部复权因子到内存, get_data(fq="qfq")
    内部做内存复权, 不需要单独拉复权表。
  - realtime 没独立表, 用 get_data 拉抽样股的最新日K做盘后快照。
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from app.data_providers.normalizer import normalize_adj_factors, normalize_daily
from app.plugins.stockdb import bridge

logger = logging.getLogger(__name__)

_DATASETS = ("daily", "adj_factor", "minute", "realtime", "instruments")

# 分钟K canonical 列
_MINUTE_CANONICAL = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]

# realtime 抽样股票: stockdb 没有全市场实时快照表, 用最新日K做盘后参考。
# 主力指数和活跃股一定要在里面。
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
    (会按 epoch days 解释成年份 57441),必须先转 ISO 串再让 normalize 解析。
    """
    s = str(d)
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
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
    """stockdb 内置数据源 (SDK 直连版)。"""

    name = "stockdb"
    builtin = True

    def __init__(self) -> None:
        self.config = _StockDBConfig()

    def close(self) -> None:  # loader.load_all 会对每个 provider 调 close
        pass

    def _get_rd(self):
        """获取 SDK rd 单例。"""
        return bridge._get_rd()

    # ---- daily ----
    # SDK 的 rd.get_data(codes, start, end) 内部用 rd.pipe() + mget() 批量拉取,
    # 一次传全量代码列表 (~7500 只), 实测 4-5 秒完成 14 天日K。
    # fq=None 不复权 (下游 kline_daily 存不复权原始价, enriched 阶段再复权)。
    def get_daily(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",  # noqa: ARG002
        on_chunk_done=None,
    ) -> pl.DataFrame:
        if not symbols:
            return pl.DataFrame()
        if not start_time or not end_time:
            return pl.DataFrame()

        rd = self._get_rd()
        # symbols 是 "600519.SH" 格式, SDK 用裸 code "600519"
        code_to_sym: dict[str, str] = {}
        codes: list[str] = []
        for sym in symbols:
            code = sym.split(".")[0]
            code_to_sym[code] = sym
            codes.append(code)

        start_str = start_time.strftime("%Y%m%d")
        end_str = end_time.strftime("%Y%m%d")

        # SDK 批量拉取: 返回 {code: [row_dict, ...]}
        data = rd.get_data(codes, start=start_str, end=end_str, frequency="1d", fq=None)

        # 展平为行列表, 补 symbol 字段
        rows: list[dict] = []
        for code, records in data.items():
            sym = code_to_sym.get(code, code)
            for r in records:
                if not isinstance(r, dict):
                    continue
                row = dict(r)
                # stockdb date 是 int YYYYMMDD → 转 ISO 让 polars.Date 正确解析
                if "date" in row and isinstance(row["date"], int):
                    iso = _ymd_int_to_iso(row["date"])
                    if iso:
                        row["date"] = iso
                row["symbol"] = sym
                rows.append(row)

        if on_chunk_done:
            on_chunk_done(1, 1)

        if not rows:
            return pl.DataFrame()
        # stockdb volume 单位是"股", 项目约定 volume 为"手"(1手=100股)。
        # 除以 100 与 tickflow / stocksdk 两个内置源对齐。
        for r in rows:
            v = r.get("volume")
            if v is not None:
                try:
                    r["volume"] = float(v) / 100.0
                except (TypeError, ValueError):  # noqa: PERF203
                    pass
        df = normalize_daily(rows, source=self.name)
        return df

    # ---- adj_factor ----
    # SDK StockDBClient 在 __init__ 时一次性预加载全部复权因子到内存
    # (_fq_dates: {code: [date_str, ...]}, _fq_cums: {code: [cum_float, ...]})。
    # 直接从内存提取, 不需要任何网络请求。
    #
    # 关键: stockdb 的 cum 是**累积复权因子** (从上市至今的累积比值),
    # 而 _apply_adj_factor 期望 ex_factor 是**单次除权事件比值** (非累积),
    # 然后自己做 cum_prod 来还原累积因子。
    # 如果直接把 cum 当 ex_factor 传入, cum_prod 会做二次累积导致复权错误。
    # 因此这里把 cum 转换为相邻比值: ex_factor[t] = cum[t] / cum[t-1],
    # 第一次除权 ex_factor[0] = cum[0] (从 1.0 开始累积)。
    #
    # 同时, 前复权计算需要完整的因子链 (从最早到最新), 否则 ratio=1 不生效。
    # 因此不按 start_time/end_time 过滤, 始终返回全部历史因子。
    # sync_adj_factor 内部 merge+unique 幂等, 重复拉取无副作用。
    def get_adj_factors(
        self,
        symbols: list[str],
        start_time: datetime | None,  # noqa: ARG002 (忽略, 始终返回全量)
        end_time: datetime | None,  # noqa: ARG002
        asset_type: str = "stock",  # noqa: ARG002
        on_chunk_done=None,
    ) -> pl.DataFrame:
        if not symbols:
            return pl.DataFrame()

        # 获取 SDK 的 StockDBClient 单例 (复权因子已预加载)
        from stock_sdk import get_default_client
        client = get_default_client()

        rows: list[dict] = []
        total = len(symbols)
        for i, sym in enumerate(symbols):
            code = sym.split(".")[0]
            dates = client._fq_dates.get(code)
            cums = client._fq_cums.get(code)
            if not dates or not cums:
                continue
            prev_cum = 1.0  # 累积因子从 1.0 开始
            for date_str, cum_val in zip(dates, cums):
                # cum 是累积因子, 转为单次事件比值
                ex_factor = cum_val / prev_cum if prev_cum else cum_val
                prev_cum = cum_val
                rows.append({
                    "symbol": sym,
                    "trade_date": _ymd_int_to_iso(date_str) or date_str,
                    "ex_factor": ex_factor,
                })
            if on_chunk_done and (i + 1) % 100 == 0:
                on_chunk_done(i + 1, total)

        if on_chunk_done:
            on_chunk_done(total, total)

        if not rows:
            return pl.DataFrame()
        df = normalize_adj_factors(rows, source=self.name)
        return df

    # ---- minute ----
    # SDK 的 rd.get_data(codes, start, end, frequency="1m") 批量拉取分钟K,
    # build_time_query_for_retrieval 自动把 8 位日期补成 14 位时间戳。
    def get_minute(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",  # noqa: ARG002
        on_chunk_done=None,
        freq: str = "1m",  # noqa: ARG002 (SDK 端 frequency 固定 1m, 数据原样返回)
    ) -> pl.DataFrame:
        if not symbols:
            return pl.DataFrame()
        if not start_time or not end_time:
            return pl.DataFrame()

        rd = self._get_rd()
        code_to_sym: dict[str, str] = {}
        codes: list[str] = []
        for sym in symbols:
            code = sym.split(".")[0]
            code_to_sym[code] = sym
            codes.append(code)

        # 分钟K: SDK 用 14 位时间戳, build_time_query_for_retrieval 自动补全
        start_str = start_time.strftime("%Y%m%d")
        end_str = end_time.strftime("%Y%m%d")

        data = rd.get_data(codes, start=start_str, end=end_str, frequency="1m", fq=None)

        frames: list[pl.DataFrame] = []
        for code, records in data.items():
            sym = code_to_sym.get(code, code)
            if not records:
                continue
            df = pl.DataFrame(records)
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
            on_chunk_done(1, 1)

        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    # ---- realtime ----
    # stockdb 没独立实时表, 用 get_data 拉抽样股的最新日K做盘后快照。
    def get_realtime(self) -> list[dict]:
        if not _REALTIME_SAMPLE:
            return []
        rd = self._get_rd()
        # 拉抽样股最近 2 天日K, 取最后一天作为快照
        data = rd.get_data(_REALTIME_SAMPLE, start="20260101", end="N", frequency="1d", fq=None)
        rows: list[dict] = []
        for code, records in data.items():
            if not records:
                continue
            item = records[-1]  # 最新一天
            c = item.get("code") or code
            rows.append({
                "symbol": code_to_symbol(c),
                "name": item.get("name"),
                "last_price": item.get("close"),
                "prev_close": item.get("pre_close"),
                "open": item.get("open"),
                "high": item.get("high"),
                "low": item.get("low"),
                "volume": (item.get("volume") or 0) / 100.0,  # 股 → 手
                "amount": item.get("amount"),
                # stockdb pct_chg 字段是「百分比」单位 (5.32 = +5.32%);
                # 系统 change_pct 约定是「小数」 (0.0532 = +5.32%)。
                "change_pct": (item.get("pct_chg") or 0) / 100.0,
            })
        if rows:
            logger.info("stockdb realtime 快照: 抽样 %d 只, 共 %d 行", len(_REALTIME_SAMPLE), len(rows))
        return rows

    # ---- instruments ----
    # 股票代码表: {<市场前缀>: [code, ...]}。
    # 拉一次全表, 展平成 tickflow Instrument 形状。
    # name 从日K最新行提取 (日K每行都带 name 字段)。
    def get_instruments(self, asset_type: str = "stock") -> list[dict]:
        if asset_type != "stock":
            return []
        rd = self._get_rd()
        data = rd.get("股票代码").do()
        if not isinstance(data, dict):
            return []
        # 用 get_data 拉最近日K, 从中提取 (code → name) 映射
        all_codes: list[str] = []
        for codes in data.values():
            if isinstance(codes, list):
                all_codes.extend(codes)
        name_map: dict[str, str] = {}
        if all_codes:
            try:
                kdata = rd.get_data(all_codes, start="20260101", end="N", frequency="1d", fq=None)
                for code, records in kdata.items():
                    if records:
                        n = records[-1].get("name")
                        if n:
                            name_map[code] = n
            except Exception as e:  # noqa: BLE001
                logger.warning("instruments name 补全失败 (不影响主流程): %s", e)

        rows: list[dict] = []
        for prefix, codes in data.items():
            ex, at = _MARKET_PREFIX.get(str(prefix), ("SH", asset_type))
            if not isinstance(codes, list):
                continue
            for code in codes:
                sym = code_to_symbol(code)
                rows.append({
                    "symbol": sym,
                    "name": name_map.get(code, sym),
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
