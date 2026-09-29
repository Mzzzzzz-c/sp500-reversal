"""信号、过滤、选股、定权 —— 回测与实盘共用同一套函数。

记号：第 t 天在信号时刻（默认美东 15:30）取价 P_sig，
  原始跌幅   r   = P_sig / 昨收 − 1
  残差跌幅   e   = r − β × r_基准（基准 = SPY 或行业 ETF 同一时刻的涨跌）
  信号值     z   = e / σ（σ = 过去 60 天日残差收益的标准差）
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .data.universe import GICS_ETF


# ----------------------------------------------------------------------------
# 1. 只依赖历史日线的量（截至 t-1，盘中就能算好）
# ----------------------------------------------------------------------------
def history_features(D: dict, cfg: dict) -> dict:
    P = D["P"]
    sc = cfg["signal"]
    n = int(sc["beta_window"])
    lo, hi = sc["beta_clip"]
    close, adj, vol = P["close"], P["adjclose"], P["volume"]
    ra = adj.pct_change(fill_method=None)
    bench = _bench_map(D, sc["benchmark"], close.columns)
    rb = _bench_panel(ra, bench)

    x, y = ra.shift(1), rb.shift(1)                  # 只用到 t-1
    mp = int(n * 0.67)
    mx, my = x.rolling(n, min_periods=mp).mean(), y.rolling(n, min_periods=mp).mean()
    cov = (x * y).rolling(n, min_periods=mp).mean() - mx * my
    var = (y * y).rolling(n, min_periods=mp).mean() - my * my
    beta = (cov / var).clip(lo, hi)
    eps = ra - beta * rb                               # 第 s 天的残差（β 用 s-1 之前的数据）
    sigma = eps.shift(1).rolling(n, min_periods=mp).std()
    tvol = ra.shift(1).rolling(n, min_periods=mp).std()
    # 对 SPY 的 β（对冲用）
    rs = ra["SPY"]
    xs = rs.shift(1)
    cov_s = x.mul(xs, axis=0).rolling(n, min_periods=mp).mean() - mx.mul(xs.rolling(n, min_periods=mp).mean(), axis=0)
    var_s = (xs * xs).rolling(n, min_periods=mp).mean() - xs.rolling(n, min_periods=mp).mean() ** 2
    beta_spy = cov_s.div(var_s, axis=0).clip(lo, hi)

    adv = (close * vol).rolling(20, min_periods=15).mean().shift(1)
    nhist = close.notna().cumsum().shift(1)
    streak_prev = _streak(ra).shift(1)
    return {"ra": ra, "beta": beta, "beta_spy": beta_spy, "sigma": sigma, "tvol": tvol, "adv": adv,
            "nhist": nhist, "streak_prev": streak_prev, "bench": bench}


def _streak(ra: pd.DataFrame) -> pd.DataFrame:
    """截至当天的连跌天数。"""
    d = (ra < 0).astype(int)
    c = d.cumsum()
    reset = c.where(d == 0).ffill().fillna(0)
    return c - reset


def _bench_map(D: dict, mode: str, cols) -> pd.Series:
    if mode == "sector":
        sec = D["sectors"].reindex(cols)
        return sec.map(GICS_ETF).fillna("SPY")
    return pd.Series("SPY", index=cols)


def _bench_panel(ra: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    out = {}
    spy = ra["SPY"]
    for c in ra.columns:
        b = bench.get(c, "SPY")
        s = ra[b] if b in ra.columns else spy
        out[c] = s.where(s.notna(), spy)
    return pd.DataFrame(out, index=ra.index)


# ----------------------------------------------------------------------------
# 2. 信号时刻的量
# ----------------------------------------------------------------------------
def signal_features(D: dict, H: dict, cfg: dict, mode: str = "intraday") -> dict:
    """mode = intraday：用 60 分钟线在信号时刻的价格（真实可交易）；
       mode = daily   ：用收盘价（有前视偏差，只用来做长周期近似和对照）。"""
    P = D["P"]
    close, opn = P["close"], P["open"]
    prev = close.shift(1)
    if mode == "intraday":
        px = D["intraday"]["px"]
        idx = px.index.intersection(close.index)
        px = px.reindex(idx)
        cv = D["intraday"]["cumvol"].reindex(idx)
        cv_all = D["intraday"]["cumvol"]
        cv_avg = cv_all.rolling(20, min_periods=10).mean().shift(1).reindex(idx)
        vratio = cv / cv_avg
        vix = px.get("^VIX") if "^VIX" in px else None
        if vix is None or vix.isna().all():
            vix = close["^VIX"].shift(1).reindex(idx)
    else:
        idx = close.index
        px = close
        vratio = P["volume"] / P["volume"].rolling(20, min_periods=10).mean().shift(1)
        vix = close["^VIX"].shift(1) if "^VIX" in close else pd.Series(np.nan, index=idx)
    prev = prev.reindex(idx)
    r = px.reindex(idx) / prev - 1
    bench = H["bench"]
    rb = _bench_panel(r, bench)
    beta = H["beta"].reindex(idx)
    e = r - beta * rb
    z = e / H["sigma"].reindex(idx)
    gap = opn.reindex(idx) / prev - 1
    below_open = px.reindex(idx) < opn.reindex(idx)
    streak = (H["streak_prev"].reindex(idx) + 1).where(r < 0, 0)
    return {"idx": idx, "px": px.reindex(idx), "r": r, "e": e, "z": z, "gap": gap, "below_open": below_open,
            "vratio": vratio.reindex(idx), "vix": vix.reindex(idx), "spy_r": r["SPY"], "streak": streak}


# ----------------------------------------------------------------------------
# 3. 过滤
# ----------------------------------------------------------------------------
def eligibility(D: dict, H: dict, S: dict, cfg: dict) -> tuple:
    """返回 (可选股票布尔矩阵, 各过滤条件剔除数量的 DataFrame)。"""
    f = cfg["filters"]
    idx = S["idx"]
    P = D["P"]
    cols = P["close"].columns
    ok = D["member"].reindex(index=idx, columns=cols, fill_value=False).copy()
    ok &= S["r"].notna() & S["z"].notna()
    reasons = {}

    def apply(name, bad):
        nonlocal ok
        bad = bad.reindex(index=idx, columns=cols).fillna(False).astype(bool)
        reasons[name] = (ok & bad).sum(axis=1)
        ok &= ~bad

    apply("price", S["px"] < f["min_price"])
    apply("adv", H["adv"].reindex(idx) < f["min_adv_usd"])
    apply("history", H["nhist"].reindex(idx) < f["min_history"])
    er, ef = D["earn_react"], D["earn_file"]
    if f.get("earnings_recent", True):
        rec = D.get("earn_recent_flag")            # 实盘：由财报日历直接给出
        if rec is None:
            rec = er | er.shift(1, fill_value=False) | ef | ef.shift(1, fill_value=False)
        apply("earnings_recent", rec)
    if f.get("earnings_next", True):
        nxt = D.get("earn_next_flag")
        if nxt is None:
            nxt = er.shift(-1, fill_value=False) | ef.shift(-1, fill_value=False)
        apply("earnings_next", nxt)
    if f.get("ex_dividend", True):
        apply("ex_dividend", P["div"].notna())
    g = f.get("gap_news", "keep")
    news = (S["gap"] <= f["gap_threshold"]) & S["below_open"]
    if g == "exclude":
        apply("gap_news", news)
    elif g == "only":
        apply("gap_news", ~news)
    if f.get("volume_spike"):
        apply("volume_spike", S["vratio"] > f["volume_spike"])
    if f.get("min_down_streak"):
        apply("down_streak", S["streak"] < f["min_down_streak"])
    return ok, pd.DataFrame(reasons)


# ----------------------------------------------------------------------------
# 4. 选股与定权
# ----------------------------------------------------------------------------
def select(D: dict, H: dict, S: dict, ok: pd.DataFrame, cfg: dict, nav: Optional[float] = None) -> tuple:
    """返回 (目标权重 DataFrame, 入选明细 DataFrame)。"""
    sc, pc = cfg["signal"], cfg["portfolio"]
    idx = S["idx"]
    cand = ok.copy()
    if sc.get("z_max") is not None:
        cand &= S["z"] <= sc["z_max"]
    if sc.get("resid_max") is not None:
        cand &= S["e"] <= sc["resid_max"]
    if sc.get("raw_max") is not None:
        cand &= S["r"] <= sc["raw_max"]
    key = S["z"] if sc.get("rank_by", "z") == "z" else S["r"]
    sectors = D["sectors"]
    tvol = H["tvol"].reindex(idx)
    adv = H["adv"].reindex(idx)
    nav = nav or cfg["account"]["value"]
    W = {}
    rows = []
    cm = cand.values
    kv = key.values
    cols = np.array(cand.columns)
    for i, d in enumerate(idx):
        js = np.where(cm[i])[0]
        if len(js) == 0:
            continue
        js = js[np.argsort(kv[i, js])]
        picked, cnt = [], {}
        for j in js:
            s = cols[j]
            sec = sectors.get(s, "Unknown")
            if cnt.get(sec, 0) >= sc["sector_max_names"]:
                continue
            cnt[sec] = cnt.get(sec, 0) + 1
            picked.append(s)
            if len(picked) >= sc["max_names"]:
                break
        w = weights(picked, tvol.loc[d], adv.loc[d], sectors, pc, nav)
        W[d] = w
        for s in picked:
            rows.append((d, s, S["r"].at[d, s], S["e"].at[d, s], S["z"].at[d, s], sectors.get(s, "Unknown"),
                         w.get(s, 0.0)))
    Wdf = pd.DataFrame(W).T.reindex(idx).fillna(0.0) if W else pd.DataFrame(0.0, index=idx, columns=[])
    sel = pd.DataFrame(rows, columns=["date", "symbol", "r", "e", "z", "sector", "weight"])
    return Wdf, sel


def weights(picked, tvol_row: pd.Series, adv_row: pd.Series, sectors: pd.Series, pc: dict, nav: float) -> pd.Series:
    if not picked:
        return pd.Series(dtype=float)
    if pc.get("weighting", "inverse_vol") == "inverse_vol":
        iv = 1.0 / tvol_row.reindex(picked).clip(lower=0.005).fillna(tvol_row.median())
        w = iv / iv.sum()
    else:
        w = pd.Series(1.0 / len(picked), index=picked)
    w = w.clip(upper=pc["max_weight"])
    if pc.get("adv_max_frac") and nav:
        w = np.minimum(w, pc["adv_max_frac"] * adv_row.reindex(picked).fillna(0) / nav)
    sec = sectors.reindex(picked).fillna("Unknown")
    tot = w.groupby(sec).transform("sum")
    w = w * np.minimum(1.0, pc["sector_max_weight"] / tot)
    g = w.sum()
    if g > pc["gross_max"]:
        w = w * pc["gross_max"] / g
    return w[w > 0]
