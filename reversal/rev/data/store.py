"""本地缓存：每只股票一个文件，合并成面板（日期 × 股票）。"""
from __future__ import annotations

import os
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional

import pandas as pd
import requests

from . import yahoo

ETFS = ["SPY", "XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]
INDEXES = ["^VIX"]


def _fname(sym: str) -> str:
    return sym.replace("^", "_").replace("/", "_") + ".csv"


class Store:
    def __init__(self, root: str):
        self.root = root
        for sub in ("daily", "events", "intraday60", "meta"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)

    # ---------------- 下载 ----------------
    def _fresh(self, path: str, hours: float) -> bool:
        return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < hours * 3600

    def download_daily(self, symbols: Iterable[str], start: str = "2010-01-01", workers: int = 6,
                       refresh_hours: float = 12, log=print) -> dict:
        symbols = sorted(set(symbols))
        status = {}
        todo = [s for s in symbols if not self._fresh(os.path.join(self.root, "daily", _fname(s)), refresh_hours)]
        log(f"日线：共 {len(symbols)} 只，需要下载 {len(todo)} 只")
        sess = requests.Session()

        def one(sym):
            r = yahoo.daily(sym, start=start, session=sess)
            if r is None or r["bars"].empty:
                return sym, "missing"
            r["bars"].to_csv(os.path.join(self.root, "daily", _fname(sym)), index_label="date", float_format="%.7g")
            ev = pd.concat({"div": r["div"], "split": r["split"]}, axis=1)
            ev.to_csv(os.path.join(self.root, "events", _fname(sym)), index_label="date")
            return sym, "ok"

        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(one, s) for s in todo]
            for i, f in enumerate(as_completed(futs), 1):
                sym, st = f.result()
                status[sym] = st
                if i % 100 == 0:
                    log(f"  已完成 {i}/{len(todo)}")
        miss = sorted(k for k, v in status.items() if v != "ok")
        if miss:
            log(f"  雅虎没有数据的代码 {len(miss)} 只（多为已退市/已改名）：{', '.join(miss[:30])}{' …' if len(miss) > 30 else ''}")
        path = os.path.join(self.root, "meta", "daily_missing.json")
        old = json.load(open(path)) if os.path.exists(path) else []
        json.dump(sorted((set(old) | set(miss)) - {k for k, v in status.items() if v == "ok"}), open(path, "w"))
        return status

    def download_intraday(self, symbols: Iterable[str], workers: int = 6, refresh_hours: float = 12,
                          log=print) -> dict:
        """60 分钟 K 线。雅虎只保留约 730 天，所以每次下载后与旧文件合并，历史会越攒越长。"""
        symbols = sorted(set(symbols))
        todo = [s for s in symbols if not self._fresh(os.path.join(self.root, "intraday60", _fname(s)), refresh_hours)]
        log(f"60分钟线：共 {len(symbols)} 只，需要下载 {len(todo)} 只")
        status = {}
        sess = requests.Session()

        def one(sym):
            df = yahoo.intraday(sym, "60m", "730d", session=sess)
            if df is None or df.empty:
                return sym, "missing"
            path = os.path.join(self.root, "intraday60", _fname(sym))
            if os.path.exists(path):
                old = pd.read_csv(path, index_col=0, parse_dates=True)
                df = pd.concat([old[old.index < df.index.min()], df])
            df.to_csv(path, index_label="time", float_format="%.7g")
            return sym, "ok"

        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(one, s) for s in todo]
            for i, f in enumerate(as_completed(futs), 1):
                sym, st = f.result()
                status[sym] = st
                if i % 100 == 0:
                    log(f"  已完成 {i}/{len(todo)}")
        return status

    # ---------------- 读取 ----------------
    def available(self, kind: str = "daily") -> list:
        d = os.path.join(self.root, kind)
        return sorted(f[:-4].replace("_", "^", 1) if f.startswith("_") else f[:-4] for f in os.listdir(d)
                      if f.endswith(".csv"))

    def load_daily_panel(self, symbols: Optional[Iterable[str]] = None) -> dict:
        """返回 {'open','high','low','close','adjclose','volume','div'}，每个都是 日期 × 股票 的 DataFrame。"""
        symbols = sorted(set(symbols)) if symbols is not None else self.available("daily")
        cols = {k: {} for k in ("open", "high", "low", "close", "adjclose", "volume")}
        divs, spls = {}, {}
        for s in symbols:
            p = os.path.join(self.root, "daily", _fname(s))
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            for k in cols:
                cols[k][s] = df[k]
            pe = os.path.join(self.root, "events", _fname(s))
            if os.path.exists(pe):
                ev = pd.read_csv(pe, index_col=0, parse_dates=True)
                if "div" in ev and ev["div"].notna().any():
                    divs[s] = ev["div"].dropna()
                if "split" in ev and ev["split"].notna().any():
                    spls[s] = ev["split"].dropna()
        out = {k: pd.DataFrame(v).sort_index() for k, v in cols.items()}
        idx = out["close"].index
        out["div"] = pd.DataFrame({s: v.groupby(level=0).sum().reindex(idx) for s, v in divs.items()},
                                  index=idx).reindex(columns=out["close"].columns)
        out["split"] = pd.DataFrame({s: v.groupby(level=0).prod().reindex(idx) for s, v in spls.items()},
                                    index=idx).reindex(columns=out["close"].columns)
        return out

    def load_intraday_panel(self, symbols: Optional[Iterable[str]] = None, at: str = "15:30") -> dict:
        """从 60 分钟线取出每天指定整点半的价格和截至该时刻的累计成交量。
        返回 {'px': 该时刻价格（该 K 线开盘价）, 'cumvol': 该时刻之前的累计成交量}。"""
        symbols = sorted(set(symbols)) if symbols is not None else self.available("intraday60")
        hh, mm = map(int, at.split(":"))
        px, cv = {}, {}
        for s in symbols:
            p = os.path.join(self.root, "intraday60", _fname(s))
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            t = df.index
            day = t.normalize()
            sel = (t.hour == hh) & (t.minute == mm)
            px[s] = pd.Series(df["open"].values[sel], index=day[sel])
            before = (t.hour * 60 + t.minute) < (hh * 60 + mm)
            cv[s] = df["volume"].where(before, 0).groupby(day).sum()
        return {"px": pd.DataFrame(px).sort_index(), "cumvol": pd.DataFrame(cv).sort_index()}
