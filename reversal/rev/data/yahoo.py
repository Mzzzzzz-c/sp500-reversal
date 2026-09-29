"""雅虎财经行情（免费，无需账号）。

- 日线：拆股复权的 OHLCV + 分红复权收盘价 adjclose + 分红/拆股事件，可取到上市以来全部历史
- 60 分钟 K 线：只能取最近约 730 天；每根 K 线的开盘价 = 该整点半的成交价
  （美东 9:30、10:30 … 15:30 各一根，15:30 这根的开盘价就是尾盘买入时刻的价格）
- 实时快照：用于每日 15:20 计算信号（雅虎对美股通常是实时成交价，少数情况下有延迟，实盘建议用券商行情）

国内网络如果打不开雅虎，设置环境变量 HTTPS_PROXY（例如 http://127.0.0.1:1087）即可。
"""
from __future__ import annotations

import time
import random
import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd
import requests

BASE = "https://query1.finance.yahoo.com/v8/finance/chart/"
# 注意：完整的浏览器 UA 反而会被雅虎要求 cookie 并返回 429，用最简 UA 即可
HEADERS = {"User-Agent": "Mozilla/5.0"}
NY = "America/New_York"


def yahoo_symbol(sym: str) -> str:
    """BRK.B / BRK-B → BRK-B；指数保持 ^VIX 形式。"""
    return sym.replace(".", "-").upper()


def _get(url: str, params: dict, retries: int = 5, session: Optional[requests.Session] = None) -> Optional[dict]:
    s = session or requests
    for k in range(retries):
        try:
            r = s.get(url, params=params, headers=HEADERS, timeout=30)
        except requests.RequestException:
            time.sleep(2 * (k + 1))
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                return None
        if r.status_code == 404:
            return None
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(3 * (k + 1) + random.random() * 2)
            continue
        return None
    return None


def daily(sym: str, start: str = "2010-01-01", end: Optional[str] = None,
          session: Optional[requests.Session] = None) -> Optional[dict]:
    """返回 {'bars': DataFrame[open,high,low,close,adjclose,volume], 'div': Series, 'split': Series}；无数据返回 None。"""
    p1 = int(pd.Timestamp(start, tz=NY).timestamp())
    p2 = int((pd.Timestamp(end, tz=NY) if end else pd.Timestamp.now(tz=NY) + pd.Timedelta(days=1)).timestamp())
    js = _get(BASE + yahoo_symbol(sym), {"period1": p1, "period2": p2, "interval": "1d",
                                         "events": "div|split", "includeAdjustedClose": "true"}, session=session)
    if not js or not js.get("chart", {}).get("result"):
        return None
    res = js["chart"]["result"][0]
    ts = res.get("timestamp")
    if not ts:
        return None
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(NY).normalize().tz_localize(None)
    df = pd.DataFrame({"open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
                       "close": q.get("close"), "adjclose": adj if adj else q.get("close"),
                       "volume": q.get("volume")}, index=idx, dtype="float64")
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"])
    ev = res.get("events", {}) or {}
    div = pd.Series({pd.to_datetime(int(k), unit="s", utc=True).tz_convert(NY).normalize().tz_localize(None):
                     v["amount"] for k, v in (ev.get("dividends") or {}).items()}, dtype="float64").sort_index()
    spl = pd.Series({pd.to_datetime(int(k), unit="s", utc=True).tz_convert(NY).normalize().tz_localize(None):
                     v["numerator"] / v["denominator"] for k, v in (ev.get("splits") or {}).items()},
                    dtype="float64").sort_index()
    meta = res.get("meta", {})
    return {"bars": df, "div": div, "split": spl, "meta": meta}


def intraday(sym: str, interval: str = "60m", range_: str = "730d",
             session: Optional[requests.Session] = None) -> Optional[pd.DataFrame]:
    """日内 K 线（美东时间，只含常规交易时段）。列：open high low close volume。"""
    js = _get(BASE + yahoo_symbol(sym), {"range": range_, "interval": interval, "includePrePost": "false"},
              session=session)
    if not js or not js.get("chart", {}).get("result"):
        return None
    res = js["chart"]["result"][0]
    ts = res.get("timestamp")
    if not ts:
        return None
    q = res["indicators"]["quote"][0]
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(NY).tz_localize(None)
    df = pd.DataFrame({"open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
                       "close": q.get("close"), "volume": q.get("volume")}, index=idx, dtype="float64")
    df = df[~df.index.duplicated(keep="last")]
    df = df[(df.index.time >= dt.time(9, 30)) & (df.index.time < dt.time(16, 0))]
    return df.dropna(subset=["open"])


def snapshot(sym: str, session: Optional[requests.Session] = None) -> Optional[dict]:
    """实时快照：最新价、昨收、今日开盘、今日累计成交量、行情时间。"""
    js = _get(BASE + yahoo_symbol(sym), {"range": "1d", "interval": "5m", "includePrePost": "false"},
              retries=3, session=session)
    if not js or not js.get("chart", {}).get("result"):
        return None
    res = js["chart"]["result"][0]
    m = res.get("meta", {})
    q = res["indicators"]["quote"][0]
    vol = np.nansum(np.array(q.get("volume") or [], dtype="float64"))
    opens = [x for x in (q.get("open") or []) if x is not None]
    return {"symbol": sym, "price": m.get("regularMarketPrice"),
            "prev_close": m.get("chartPreviousClose") if m.get("previousClose") is None else m.get("previousClose"),
            "open": opens[0] if opens else None, "volume": float(vol),
            "time": pd.to_datetime(m.get("regularMarketTime"), unit="s", utc=True).tz_convert(NY).tz_localize(None)
            if m.get("regularMarketTime") else None}
