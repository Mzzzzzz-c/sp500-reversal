"""把本地缓存整理成回测需要的全部数据（一次性加载，存成 pickle 加速）。"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from .config import path
from .data.store import Store, ETFS
from .data import universe as U
from .data import sec_earnings


def load(cfg: dict, use_cache: bool = True, log=print) -> dict:
    ddir = path(cfg["data"]["dir"])
    cache = os.path.join(ddir, "dataset.pkl")
    if use_cache and os.path.exists(cache):
        newest = max(os.path.getmtime(os.path.join(ddir, "daily", f)) for f in os.listdir(os.path.join(ddir, "daily")))
        if os.path.getmtime(cache) > newest:
            return pd.read_pickle(cache)
    st = Store(ddir)
    log("读取日线 …")
    P = st.load_daily_panel()
    close = P["close"]
    td = close["SPY"].dropna().index
    # 盘中下载的日线最后一行是“今天到目前为止”，不是收盘价 → 去掉
    now = pd.Timestamp.now(tz="America/New_York")
    if td[-1] == now.normalize().tz_localize(None) and (now.hour, now.minute) < (16, 30):
        td = td[:-1]
    P = {k: v.reindex(td) for k, v in P.items()}

    log("整理历史成分股 …")
    hist = U.fetch_history(os.path.join(ddir, "meta", "sp500_history.csv"))
    cur = U.fetch_current(os.path.join(ddir, "meta", "sp500_current.csv"))
    M = U.membership(hist, cur, td, cfg["data"]["start"]).reindex(columns=close.columns, fill_value=False)
    M = M.reindex(td, fill_value=False)
    M[[c for c in M.columns if c in ETFS or c.startswith("^")]] = False
    M, bad = U.drop_junk(M, P["close"], P["volume"])
    if bad:
        log(f"  剔除疑似代码被重用的数据：{bad}")

    info_path = os.path.join(ddir, "sec", "company_info.json")
    sec_info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    extra = None
    ep = os.path.join(ddir, "meta", "sectors_extra.csv")
    if os.path.exists(ep):
        extra = pd.read_csv(ep)
    sectors = U.sectors(close.columns, cur, extra, sec_info)

    log("整理财报日期 …")
    evp = os.path.join(ddir, "earnings_events.csv")
    ev = pd.read_csv(evp) if os.path.exists(evp) else pd.DataFrame(columns=["symbol", "accept_utc", "filing_date"])
    rd = sec_earnings.reaction_days(ev, td)
    earn_react = _flag_panel(rd, "react_day", td, close.columns)
    earn_file = _flag_panel(rd, "filing_day", td, close.columns)
    has_sec = pd.Series(close.columns.isin(rd["symbol"].unique()), index=close.columns)

    log("读取 60 分钟线 …")
    I = st.load_intraday_panel(at=cfg["signal"]["time"])
    I = {k: v.reindex(columns=close.columns) for k, v in I.items()}
    # 只保留日内数据完整的交易日（至少 90% 的有日线的股票有 15:30 价格）
    if len(I["px"]):
        have = I["px"].notna().sum(axis=1)
        icols = I["px"].columns[I["px"].notna().any()]
        need = close.reindex(index=I["px"].index, columns=icols).notna().sum(axis=1)
        good = have >= 0.9 * need
        I = {k: v.reindex(good[good].index) for k, v in I.items()}

    rf = _tbill(td)
    out = {"P": P, "member": M, "sectors": sectors, "earn_react": earn_react, "earn_file": earn_file,
           "has_sec": has_sec, "intraday": I, "rf": rf, "dates": td}
    pd.to_pickle(out, cache)
    return out


def _flag_panel(rd: pd.DataFrame, col: str, td, cols) -> pd.DataFrame:
    f = pd.DataFrame(False, index=td, columns=cols)
    rd = rd[rd["symbol"].isin(cols)]
    for sym, g in rd.groupby("symbol"):
        f.loc[f.index.intersection(pd.DatetimeIndex(g[col])), sym] = True
    return f


def _tbill(td) -> pd.Series:
    """3 个月国债年化利率（小数）。优先用 qsys 已下载的 data/meta/tbill_3m.csv。"""
    for p in (os.path.join(os.path.dirname(path()), "data", "meta", "tbill_3m.csv"), path("data", "meta", "tbill_3m.csv")):
        if os.path.exists(p):
            s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
            s = pd.to_numeric(s, errors="coerce")
            if s.dropna().median() > 0.5:          # 百分数 → 小数
                s = s / 100
            return s.reindex(td, method="ffill").fillna(0)
    return pd.Series(0.0, index=td)
