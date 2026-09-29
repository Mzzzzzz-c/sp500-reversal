"""实盘/模拟盘：在美东 15:30 左右运行，用实时行情计算当天信号，生成收盘价市价单（MOC）清单。

只生成清单，不会自动下单。

行情来源：
  sina  —— 新浪美股实时行情（国内直连，推荐）
  yahoo —— 雅虎（需要能访问雅虎，可配代理）
财报日历、除息日历：Nasdaq 官网公开接口。
"""
from __future__ import annotations

import datetime as dt
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests

from . import strategy as ST
from .config import path
from .data import yahoo
from .data import universe as U

NY = "America/New_York"
UA = {"User-Agent": "Mozilla/5.0"}


# ----------------------------------------------------------------------------
# 交易日历（NYSE）
# ----------------------------------------------------------------------------
def _easter(y: int) -> dt.date:
    a = y % 19
    b, c = divmod(y, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return dt.date(y, month, day)


def _nth_weekday(y, month, weekday, n):
    d = dt.date(y, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def _last_weekday(y, month, weekday):
    d = dt.date(y, month + 1, 1) - dt.timedelta(days=1) if month < 12 else dt.date(y, 12, 31)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: dt.date) -> dt.date:
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def nyse_holidays(y: int) -> set:
    h = {_observed(dt.date(y, 1, 1)), _nth_weekday(y, 1, 0, 3), _nth_weekday(y, 2, 0, 3),
         _easter(y) - dt.timedelta(days=2), _last_weekday(y, 5, 0), _observed(dt.date(y, 7, 4)),
         _nth_weekday(y, 9, 0, 1), _nth_weekday(y, 11, 3, 4), _observed(dt.date(y, 12, 25))}
    if y >= 2022:
        h.add(_observed(dt.date(y, 6, 19)))
    return {d for d in h if d.year == y}


def is_trading_day(d: dt.date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def shift_trading_day(d: dt.date, n: int) -> dt.date:
    step = 1 if n > 0 else -1
    k = 0
    while k < abs(n):
        d = d + dt.timedelta(days=step)
        if is_trading_day(d):
            k += 1
    return d


def now_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY)


# ----------------------------------------------------------------------------
# 实时行情
# ----------------------------------------------------------------------------
def _sina_code(sym: str) -> str:
    return "gb_" + sym.lower().replace("-", "$").replace(".", "$")


def sina_quotes(symbols: Iterable[str], batch: int = 80) -> pd.DataFrame:
    symbols = list(dict.fromkeys(symbols))
    rows = {}
    head = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}
    for i in range(0, len(symbols), batch):
        part = symbols[i:i + batch]
        codes = ",".join(_sina_code(s) for s in part)
        r = requests.get("https://hq.sinajs.cn/list=" + codes, headers=head, timeout=20)
        r.encoding = "gbk"
        back = {_sina_code(s): s for s in part}
        for m in re.finditer(r'var hq_str_(\S+?)="(.*?)";', r.text):
            code, body = m.group(1), m.group(2)
            if not body or code not in back:
                continue
            f = body.split(",")
            try:
                price, chg, opn, vol = float(f[1]), float(f[4]), float(f[5]), float(f[10])
            except (ValueError, IndexError):
                continue
            if price <= 0:
                continue
            t = f[25] if len(f) > 25 else ""
            rows[back[code]] = {"price": price, "prev_close": price - chg, "open": opn, "volume": vol,
                                "quote_time": t}
    return pd.DataFrame.from_dict(rows, orient="index")


def yahoo_quotes(symbols: Iterable[str], workers: int = 8) -> pd.DataFrame:
    symbols = list(dict.fromkeys(symbols))
    sess = requests.Session()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda s: yahoo.snapshot(s, session=sess), symbols))
    rows = {r["symbol"]: {"price": r["price"], "prev_close": r["prev_close"], "open": r["open"],
                          "volume": r["volume"], "quote_time": str(r["time"])} for r in res if r and r.get("price")}
    return pd.DataFrame.from_dict(rows, orient="index")


def quotes(symbols, provider: str = "sina") -> pd.DataFrame:
    q = sina_quotes(symbols) if provider == "sina" else yahoo_quotes(symbols)
    missing = [s for s in symbols if s not in q.index]
    if missing and provider == "sina":
        try:                                   # 新浪没有的用雅虎补
            q2 = yahoo_quotes(missing)
            q = pd.concat([q, q2])
        except Exception:
            pass
    return q


# ----------------------------------------------------------------------------
# 财报 / 除息日历（Nasdaq）
# ----------------------------------------------------------------------------
def nasdaq_earnings(day: dt.date) -> pd.DataFrame:
    r = requests.get("https://api.nasdaq.com/api/calendar/earnings", params={"date": day.isoformat()},
                     headers={**UA, "Accept": "application/json"}, timeout=20)
    rows = ((r.json().get("data") or {}).get("rows")) or []
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "time"])
    df["symbol"] = df["symbol"].str.upper().str.replace(".", "-", regex=False)
    return df[["symbol", "time"]]


def nasdaq_exdiv(day: dt.date) -> set:
    r = requests.get("https://api.nasdaq.com/api/calendar/dividends", params={"date": day.isoformat()},
                     headers={**UA, "Accept": "application/json"}, timeout=20)
    cal = (r.json().get("data") or {}).get("calendar") or {}
    rows = cal.get("rows") or []
    out = set()
    for x in rows:
        try:
            ex = pd.to_datetime(x.get("dividend_Ex_Date")).date()
        except Exception:
            continue
        if ex == day:
            out.add(str(x.get("symbol", "")).upper().replace(".", "-"))
    return out


def earnings_blocks(today: dt.date) -> tuple:
    """返回 (recent, upcoming)：
    recent   = 反应日为昨天或今天（昨天盘前/盘后、今天盘前/盘中发布）
    upcoming = 今天盘后或下一个交易日发布（拿着过夜就会碰上财报）"""
    prev_d, next_d = shift_trading_day(today, -1), shift_trading_day(today, 1)
    prev2 = shift_trading_day(today, -2)
    e_prev2, e_prev, e_today, e_next = (nasdaq_earnings(d) for d in (prev2, prev_d, today, next_d))
    amc = "time-after-hours"
    recent = set(e_prev["symbol"]) | set(e_today.loc[e_today["time"] != amc, "symbol"]) | \
        set(e_prev2.loc[e_prev2["time"] == amc, "symbol"])
    upcoming = set(e_today.loc[e_today["time"] == amc, "symbol"]) | set(e_next["symbol"])
    return recent, upcoming


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def build_live_dataset(D: dict, members: list, q: pd.DataFrame, today: pd.Timestamp, recent: set,
                       upcoming: set, exdiv: set, lookback: int = 300) -> dict:
    """在历史数据后面追加“今天”这一行，让回测用的同一套函数直接算出今天的信号。"""
    P = {k: v[v.index < today].iloc[-lookback:] for k, v in D["P"].items()}
    cols = P["close"].columns.union(pd.Index(members)).union(pd.Index(q.index))
    P = {k: v.reindex(columns=cols) for k, v in P.items()}
    row = pd.DataFrame(np.nan, index=[today], columns=cols)
    Pn = {}
    for k, v in P.items():
        r = row.copy()
        if k == "open":
            r.loc[today, q.index] = q["open"].values
        if k == "div":
            r.loc[today, [s for s in exdiv if s in cols]] = 1.0
        Pn[k] = pd.concat([v, r])
    # 昨收以实时行情为准（防止历史数据没更新到昨天）
    prev_idx = Pn["close"].index[-2]
    for s, pc in q["prev_close"].items():
        if pd.notna(pc) and s in cols:
            old = Pn["close"].at[prev_idx, s]
            if pd.isna(old) or abs(old / pc - 1) > 0.002:
                Pn["close"].at[prev_idx, s] = pc
    idx = Pn["close"].index
    member = pd.DataFrame(False, index=idx, columns=cols)
    member.loc[today, [m for m in members if m in cols]] = True
    er = pd.DataFrame(False, index=idx, columns=cols)
    rec = er.copy()
    rec.loc[today, [s for s in recent if s in cols]] = True
    nxt = er.copy()
    nxt.loc[today, [s for s in upcoming if s in cols]] = True
    px = pd.DataFrame([q["price"].reindex(cols).values], index=[today], columns=cols)
    cv = pd.DataFrame([q["volume"].reindex(cols).values], index=[today], columns=cols)
    return {"P": Pn, "member": member, "sectors": D["sectors"].reindex(cols).fillna("Unknown"),
            "earn_react": er, "earn_file": er, "earn_recent_flag": rec, "earn_next_flag": nxt,
            "intraday": {"px": px, "cumvol": cv}, "rf": D["rf"], "dates": idx}


def live_signal(D: dict, cfg: dict, provider: str = "sina", account_value: Optional[float] = None,
                positions: Optional[pd.Series] = None, members: Optional[list] = None,
                today: Optional[pd.Timestamp] = None, vol_frac: float = 0.8, log=print) -> dict:
    t_ny = now_ny()
    today = pd.Timestamp(today or t_ny.date())
    if not is_trading_day(today.date()):
        log(f"{today.date()} 不是美股交易日")
    if members is None:
        cur = U.fetch_current(path(cfg["data"]["dir"], "meta", "sp500_current.csv"))
        members = list(cur["Symbol"].str.replace(".", "-", regex=False))
    etfs = ["SPY"] + (list(U.GICS_ETF.values()) if cfg["signal"]["benchmark"] == "sector" else [])
    log(f"获取 {len(members)} 只成分股实时行情（{provider}）…")
    q = quotes(members + etfs, provider)
    log(f"  拿到 {len(q)} 只；行情时间示例：{q['quote_time'].iloc[0] if len(q) else '无'}")
    try:
        recent, upcoming = earnings_blocks(today.date())
        exdiv = nasdaq_exdiv(today.date())
        log(f"  财报回避：近期 {len(recent)} 只、即将发布 {len(upcoming)} 只；今日除息 {len(exdiv)} 只")
    except Exception as e:                                 # 日历拿不到时宁可不交易这些过滤
        log(f"  ⚠️ 财报/除息日历获取失败：{e}（本次不做 F2/F3/F4 过滤，请人工检查）")
        recent, upcoming, exdiv = set(), set(), set()
    L = build_live_dataset(D, members, q, today, recent, upcoming, exdiv)
    H = ST.history_features(L, cfg)
    S = ST.signal_features(L, H, cfg, "intraday")
    # 成交量异常：今天截至目前的成交量 ÷ (20日日均量 × 信号时刻前通常完成的比例)
    avgv = L["P"]["volume"].iloc[:-1].tail(20).mean()
    S["vratio"] = (L["intraday"]["cumvol"] / (avgv * vol_frac)).reindex(S["idx"])
    if "^VIX" in D["P"]["close"]:
        vix_prev = D["P"]["close"]["^VIX"].dropna().iloc[-1]
        try:
            snap = yahoo.snapshot("^VIX")
            vix_now = snap["price"] if snap and snap.get("price") else vix_prev
        except Exception:
            vix_now = vix_prev
        S["vix"] = pd.Series(vix_now, index=S["idx"])
    ok, reasons = ST.eligibility(L, H, S, cfg)
    av = account_value or cfg["account"]["value"]
    W, sel = ST.select(L, H, S, ok, cfg, nav=av)
    sel = sel[sel["date"] == today]
    w = W.loc[today] if today in W.index else pd.Series(dtype=float)
    w = w[w > 0]
    # 风控（VIX）
    rc = cfg["risk"]
    vix = float(S["vix"].iloc[-1]) if S["vix"].notna().any() else np.nan
    scale = 1.0
    if not np.isnan(vix):
        if rc.get("vix_stop") and vix >= rc["vix_stop"]:
            scale = 0.0
        elif rc.get("vix_half") and vix >= rc["vix_half"]:
            scale = 0.5
    spy_r = float(S["spy_r"].iloc[-1]) if pd.notna(S["spy_r"].iloc[-1]) else 0.0
    if rc.get("spy_drop_half") is not None and spy_r <= rc["spy_drop_half"]:
        scale *= 0.5
    w = w * scale
    orders = make_orders(w, q["price"], positions if positions is not None else pd.Series(dtype=float), av,
                         cfg["account"].get("fractional", False))
    snap = pd.DataFrame({"r": S["r"].loc[today], "e": S["e"].loc[today], "z": S["z"].loc[today],
                         "eligible": ok.loc[today]}).dropna(subset=["z"]).sort_values("z")
    return {"date": today, "time_ny": t_ny, "weights": w, "selection": sel, "orders": orders, "vix": vix,
            "spy_r": spy_r, "scale": scale, "reasons": reasons.loc[today] if today in reasons.index else None,
            "snapshot": snap, "quotes": q}


def make_orders(w: pd.Series, price: pd.Series, pos: pd.Series, account_value: float,
                fractional: bool = False) -> pd.DataFrame:
    syms = w.index.union(pos.index)
    px = price.reindex(syms)
    raw = (w.reindex(syms).fillna(0) * account_value / px)
    if fractional:
        tgt = raw.round(4)
    else:
        # 整数股：四舍五入，但实际金额不超过目标的 1.5 倍（小账户买不起的高价股会被放弃）
        tgt = np.round(raw)
        tgt = tgt.where(tgt * px <= 1.5 * raw * px + 1e-9, np.floor(raw))
    cur = pos.reindex(syms).fillna(0)
    d = (tgt - cur).fillna(0)
    df = pd.DataFrame({"action": np.where(d > 0, "BUY", np.where(d < 0, "SELL", "HOLD")),
                       "quantity": d.abs(), "order_type": "MOC", "ref_price": px.round(2),
                       "est_value": (d * px).round(2), "target_weight": w.reindex(syms).fillna(0).round(4),
                       "current_qty": cur, "target_qty": tgt})
    df = df[df["quantity"] > 0]
    return df.sort_values(["action", "est_value"], ascending=[False, True])
