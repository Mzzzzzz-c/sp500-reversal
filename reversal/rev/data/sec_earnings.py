"""财报发布时间（SEC EDGAR 官方、免费）。

上市公司发布季度业绩时，按规定要提交 8-K 表格的 Item 2.02（Results of Operations），
EDGAR 记录了精确到秒的受理时间（UTC）。据此可以判断财报是盘前、盘中还是盘后发布，
从而得到“市场对财报做出反应的交易日”。

注意：这只能得到【已经发生】的财报。实盘里要回避“明天盘前 / 今天盘后发财报”的股票，
需要未来的财报日历，见 live.py 里的 upcoming_earnings()。
"""
from __future__ import annotations

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional

import pandas as pd
import requests

NY = "America/New_York"
SUB = "https://data.sec.gov/submissions/"


def _get(url: str, ua: str, retries: int = 4) -> Optional[dict]:
    for k in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": ua}, timeout=30)
        except requests.RequestException:
            time.sleep(2 * (k + 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        time.sleep(2 * (k + 1))
    return None


def ticker_cik_map(ua: str) -> dict:
    js = _get("https://www.sec.gov/files/company_tickers.json", ua)
    out = {}
    for v in (js or {}).values():
        out[v["ticker"].upper().replace(".", "-")] = int(v["cik_str"])
    return out


def _rows(block: dict) -> pd.DataFrame:
    df = pd.DataFrame({k: block.get(k, []) for k in ("form", "items", "acceptanceDateTime", "filingDate")})
    return df


def fetch_company(cik: int, ua: str, since: str = "2010-01-01") -> dict:
    """返回 {'events': DataFrame[accept_utc, filing_date], 'sic': str, 'name': str}"""
    js = _get(f"{SUB}CIK{cik:010d}.json", ua)
    if js is None:
        return {"events": pd.DataFrame(), "sic": None, "name": None}
    parts = [_rows(js["filings"]["recent"])]
    for f in js["filings"].get("files", []):
        if f.get("filingTo", "9999") >= since:
            more = _get(SUB + f["name"], ua)
            if more:
                parts.append(_rows(more))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["form"].isin(["8-K", "8-K/A"]) & df["items"].fillna("").str.contains("2.02", regex=False)]
    df = df[df["filingDate"] >= since]
    ev = pd.DataFrame({"accept_utc": df["acceptanceDateTime"].values, "filing_date": df["filingDate"].values})
    return {"events": ev, "sic": js.get("sic"), "name": js.get("name")}


def download(tickers: Iterable[str], out_dir: str, ua: str, extra_ciks: Optional[dict] = None,
             workers: int = 6, since: str = "2010-01-01", refresh_hours: float = 72, log=print) -> pd.DataFrame:
    """下载所有股票的财报 8-K 时间，合并成一张表 events.csv（symbol, accept_utc, filing_date）。"""
    os.makedirs(os.path.join(out_dir, "sec"), exist_ok=True)
    cmap = ticker_cik_map(ua)
    if extra_ciks:
        cmap.update(extra_ciks)
    tickers = sorted(set(tickers))
    info_path = os.path.join(out_dir, "sec", "company_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    todo = []
    for t in tickers:
        p = os.path.join(out_dir, "sec", f"{t}.csv")
        if os.path.exists(p) and time.time() - os.path.getmtime(p) < refresh_hours * 3600:
            continue
        if t in cmap:
            todo.append(t)
    log(f"SEC 财报时间：{len(tickers)} 只中 {sum(t in cmap for t in tickers)} 只找到 CIK，需要下载 {len(todo)} 只")

    def one(t):
        r = fetch_company(cmap[t], ua, since)
        r["events"].to_csv(os.path.join(out_dir, "sec", f"{t}.csv"), index=False)
        time.sleep(0.5)          # SEC 上限每秒 10 次
        return t, r["sic"], r["name"]

    with ThreadPoolExecutor(workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(one, t) for t in todo]), 1):
            t, sic, name = f.result()
            info[t] = {"cik": cmap[t], "sic": sic, "name": name}
            if i % 100 == 0:
                log(f"  已完成 {i}/{len(todo)}")
    json.dump(info, open(info_path, "w"))
    frames = []
    for t in tickers:
        p = os.path.join(out_dir, "sec", f"{t}.csv")
        if os.path.exists(p) and os.path.getsize(p) > 30:
            d = pd.read_csv(p)
            d.insert(0, "symbol", t)
            frames.append(d)
    ev = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["symbol", "accept_utc", "filing_date"])
    ev.to_csv(os.path.join(out_dir, "earnings_events.csv"), index=False)
    return ev


def reaction_days(events: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """把每条财报 8-K 换算成“市场反应日”：
    - 美东 9:30 前受理 → 当天（非交易日则顺延）
    - 9:30–16:00 受理 → 当天
    - 16:00 之后受理 → 下一个交易日
    返回 DataFrame[symbol, react_day, filing_day]（filing_day 为受理日期所在或之后的第一个交易日，做保险用）。"""
    if events.empty:
        return pd.DataFrame(columns=["symbol", "react_day", "filing_day"])
    t = pd.to_datetime(events["accept_utc"], utc=True).dt.tz_convert(NY).dt.tz_localize(None)
    d = t.dt.normalize()
    after_close = (t.dt.hour >= 16)
    td = pd.DatetimeIndex(trading_days).sort_values()
    pos = td.searchsorted(d.values, side="left")            # 当天或之后的第一个交易日
    pos_react = pos + (after_close.values & (td[pos.clip(max=len(td) - 1)] == d.values)).astype(int)
    ok = pos_react < len(td)
    out = pd.DataFrame({"symbol": events["symbol"].values[ok],
                        "react_day": td[pos_react[ok]],
                        "filing_day": td[pos[ok]]})
    return out.drop_duplicates()
