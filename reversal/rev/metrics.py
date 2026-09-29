from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def perf(daily: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None, nav0: float = 1.0) -> dict:
    d = daily
    if start:
        d = d[d.index >= pd.Timestamp(start)]
    if end:
        d = d[d.index <= pd.Timestamp(end)]
    if len(d) < 5:
        return {}
    r = d["ret"].fillna(0)
    rf = d["rf"].fillna(0) if "rf" in d else 0.0
    ex = r - rf
    yrs = len(r) / 252
    nav = (1 + r).cumprod()
    dd = 1 - nav / nav.cummax()
    cagr = nav.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    sharpe = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else np.nan
    invested = d["gross"].shift(1).fillna(0) > 0 if "gross" in d else r != 0
    out = {
        "起止": f"{d.index[0].date()} ~ {d.index[-1].date()}",
        "年化收益": cagr,
        "年化波动": vol,
        "夏普(超额)": sharpe,
        "最大回撤": dd.max(),
        "卡玛比率": cagr / dd.max() if dd.max() > 0 else np.nan,
        "总收益": nav.iloc[-1] - 1,
        "平均仓位": d["gross"].mean() if "gross" in d else np.nan,
        "平均持股数": d["n"].mean() if "n" in d else np.nan,
        "有持仓天数占比": invested.mean(),
        "持仓日胜率": (r[invested] > 0).mean() if invested.any() else np.nan,
        "年换手(倍)": d["turnover"].sum() / yrs if "turnover" in d else np.nan,
        "年成本(占净值)": (d["cost"] / d["nav"].shift(1).fillna(nav0)).sum() / yrs if "cost" in d and d["cost"].notna().any() else np.nan,
    }
    return out


def yearly(daily: pd.DataFrame) -> pd.Series:
    r = daily["ret"].fillna(0)
    return (1 + r).groupby(r.index.year).prod() - 1


def benchmark(D: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    spy = D["P"]["adjclose"]["SPY"].pct_change(fill_method=None).reindex(index).fillna(0)
    rf = D["rf"].reindex(index).fillna(0) / 252
    return pd.DataFrame({"ret": spy, "rf": rf, "gross": 1.0, "n": 1, "turnover": 0.0, "cost": 0.0,
                         "nav": (1 + spy).cumprod()})


def trade_stats(D: dict, sel: pd.DataFrame, H: dict) -> pd.DataFrame:
    """每笔入选的次日（收盘→收盘）收益、相对 SPY 超额、相对 β×SPY 的残差收益。"""
    if sel.empty:
        return sel
    adj = D["P"]["adjclose"]
    nxt = adj.shift(-1) / adj - 1
    spy = nxt["SPY"]
    st = nxt.stack()
    key = pd.MultiIndex.from_arrays([sel["date"], sel["symbol"]])
    out = sel.copy()
    out["next_ret"] = st.reindex(key).values
    out["spy_next"] = spy.reindex(sel["date"]).values
    b = H["beta_spy"].stack().reindex(key).values
    out["next_excess"] = out["next_ret"] - out["spy_next"]
    out["next_resid"] = out["next_ret"] - b * out["spy_next"]
    return out


def summarize_trades(t: pd.DataFrame, by: Optional[str] = None) -> pd.DataFrame:
    if t.empty:
        return pd.DataFrame()
    g = t.groupby(by) if by else t.assign(_all="全部").groupby("_all")
    res = g.agg(笔数=("next_ret", "count"),
                次日收益_bp=("next_ret", lambda s: s.mean() * 1e4),
                次日超额_bp=("next_excess", lambda s: s.mean() * 1e4),
                次日残差_bp=("next_resid", lambda s: s.mean() * 1e4),
                上涨概率=("next_ret", lambda s: (s > 0).mean()),
                跑赢SPY概率=("next_excess", lambda s: (s > 0).mean()))
    se = g["next_resid"].agg(lambda s: s.std() / np.sqrt(max(len(s), 1)) * 1e4)
    res["残差t值"] = res["次日残差_bp"] / se
    return res


def fmt(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (float, np.floating)):
            if k in ("夏普(超额)", "卡玛比率", "平均持股数", "年换手(倍)"):
                out[k] = f"{v:.2f}"
            else:
                out[k] = f"{v:.1%}"
        else:
            out[k] = v
    return out
