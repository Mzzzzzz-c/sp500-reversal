"""逐日回测引擎。

时间线（第 t 天）：
  15:30 计算信号 → 收盘前用 MOC（收盘价市价单）卖出旧持仓、买入新持仓 → 成交价 = 当天收盘价
  第 t+1 天收盘：按收盘价计算持仓收益（含分红），再做下一轮调仓。
成本：按比例成本（滑点）+ 交易所费用 + IBKR 阶梯佣金（含每笔最低佣金，按账户规模计算）。
风控：VIX、SPY 当日跌幅、影子净值回撤、单日亏损 —— 只作用于【新开仓】。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def run(D: dict, H: dict, S: dict, W: pd.DataFrame, cfg: dict, start: Optional[str] = None,
        end: Optional[str] = None, costs: bool = True, risk: bool = True) -> dict:
    if cfg["portfolio"].get("exit", "close") == "open":
        return run_overnight(D, S, W, cfg, start, end, costs)
    rc = cfg["risk"]
    dd = None
    if risk and (rc.get("dd_half") or rc.get("dd_stop")):
        # 第一遍：不用回撤风控，得到“影子净值”；第二遍用影子净值的回撤（前一天收盘为准）决定降仓
        shadow = _run(D, H, S, W, cfg, start, end, costs, risk, None)["daily"]["nav"]
        dd = (1 - shadow / shadow.cummax()).shift(1).fillna(0)
    return _run(D, H, S, W, cfg, start, end, costs, risk, dd)


def _run(D, H, S, W, cfg, start, end, costs, risk, dd) -> dict:
    pc, cc, rc = cfg["portfolio"], cfg["costs"], cfg["risk"]
    P = D["P"]
    idx = S["idx"]
    if start:
        idx = idx[idx >= pd.Timestamp(start)]
    if end:
        idx = idx[idx <= pd.Timestamp(end)]
    all_days = P["close"].index
    last = idx[-1] if end else all_days[-1]
    days = all_days[(all_days >= idx[0]) & (all_days <= last)]
    is_sig = days.isin(idx)
    cols = W.columns
    ra = P["adjclose"].pct_change(fill_method=None).reindex(index=days, columns=cols)
    spy = P["adjclose"]["SPY"].pct_change(fill_method=None).reindex(days).fillna(0).values
    price = _unadjusted(P).reindex(index=days, columns=cols)
    beta_spy = H["beta_spy"].reindex(index=days, columns=cols).fillna(1.0)
    rf = (D["rf"].reindex(days).fillna(0).values / 252 if cc.get("cash_rate", "tbill") == "tbill"
          else np.zeros(len(days)))
    Wv = W.reindex(index=days).fillna(0.0).values
    vix = S["vix"].reindex(days).values
    spy_sig = S["spy_r"].reindex(days).values
    ddv = dd.reindex(days).fillna(0).values if dd is not None else np.zeros(len(days))
    k = int(pc.get("hold_days", 1))
    band = float(pc.get("rebalance_band", 0.0))
    nav0 = float(cfg["account"]["value"])
    bps = (cc["bps_per_side"] + cc.get("fees_bps", 0)) / 1e4 if costs else 0.0
    comm_on = costs and cc.get("min_commission") is not None and nav0 > 0
    frac = cfg["account"].get("fractional", False)

    n, m = len(days), len(cols)
    w = np.zeros(m)
    h = 0.0
    cohorts = []
    nav = nav0
    halted = False
    last_ret = 0.0
    rec = np.zeros((n, 7))
    rav = ra.values
    pxv = price.values
    bv = beta_spy.values
    for i in range(n):
        # 1) 持仓从 t-1 收盘到 t 收盘的收益
        r = np.nan_to_num(rav[i])
        g0 = w.sum()
        port = float(w @ r) - h * spy[i] + max(0.0, 1.0 - g0) * rf[i]
        nav *= (1 + port)
        if g0 > 0:
            w = w * (1 + r) / (1 + port)
            w[np.isnan(rav[i]) & (w > 0)] = 0.0         # 停牌/退市：按 0 收益清仓
        h = h * (1 + spy[i]) / (1 + port) if h else 0.0

        # 2) 风控系数（只作用于今天新开的一批）
        s = 1.0
        if risk and is_sig[i]:
            v = vix[i]
            if not np.isnan(v):
                if rc.get("vix_stop") and v >= rc["vix_stop"]:
                    s = 0.0
                elif rc.get("vix_half") and v >= rc["vix_half"]:
                    s *= 0.5
            if rc.get("spy_drop_half") is not None and not np.isnan(spy_sig[i]) and spy_sig[i] <= rc["spy_drop_half"]:
                s *= 0.5
            if dd is not None:
                if rc.get("dd_stop") and ddv[i] >= rc["dd_stop"]:
                    halted = True
                elif halted and ddv[i] < rc.get("dd_half", 0.1):
                    halted = False
                if halted:
                    s = 0.0
                elif rc.get("dd_half") and ddv[i] >= rc["dd_half"]:
                    s *= 0.5
            if rc.get("daily_loss_stop") and last_ret <= -rc["daily_loss_stop"]:
                s = 0.0
        last_ret = port

        # 3) 目标权重 = 最近 k 批的平均
        cohorts.append(Wv[i] * s if is_sig[i] else np.zeros(m))
        cohorts = cohorts[-k:]
        tgt = np.sum(cohorts, axis=0) / k
        delta = tgt - w
        if band > 0:
            keep = (w > 0) & (tgt > 0) & (np.abs(delta) < band * tgt)
            delta[keep] = 0.0
        tv_all = np.abs(delta) * nav
        cost = tv_all.sum() * bps
        if comm_on:
            mask = tv_all > 1e-6
            tv = tv_all[mask]
            px = np.nan_to_num(pxv[i][mask], nan=50.0)
            px[px <= 0] = 50.0
            sh = tv / px
            if not frac:
                sh = np.maximum(1, np.round(sh))
            c = np.maximum(cc["min_commission"], cc["commission_per_share"] * sh)
            c = np.minimum(c, cc.get("max_commission_pct", 0.01) * tv)
            cost += c.sum()
        w = w + delta
        if pc.get("hedge"):
            h_new = float(w @ bv[i])
            if costs:
                cost += abs(h_new - h) * nav * cc.get("hedge_bps", 1) / 1e4
            h = h_new
        nav -= cost
        rec[i] = (nav, port, cost, w.sum(), (w > 1e-9).sum(), s, np.abs(delta).sum())
    res = pd.DataFrame(rec, columns=["nav", "ret_gross", "cost", "gross", "n", "scale", "turnover"], index=days)
    res["ret"] = res["nav"].pct_change()
    res.iloc[0, res.columns.get_loc("ret")] = res["nav"].iloc[0] / nav0 - 1
    res["rf"] = rf
    return {"daily": res, "nav0": nav0}


def run_overnight(D: dict, S: dict, W: pd.DataFrame, cfg: dict, start=None, end=None, costs=True) -> dict:
    """变体：收盘买入，次日开盘卖出（只赚隔夜）。资金白天闲置。"""
    P = D["P"]
    idx = S["idx"]
    if start:
        idx = idx[idx >= pd.Timestamp(start)]
    if end:
        idx = idx[idx <= pd.Timestamp(end)]
    cols = W.columns
    close = P["close"][cols]
    nxt_open = P["open"][cols].shift(-1)
    nxt_div = P["div"][cols].shift(-1).fillna(0)
    ron = ((nxt_open + nxt_div) / close - 1).reindex(idx)
    Wd = W.reindex(idx).fillna(0)
    cc = cfg["costs"]
    bps = (cc["bps_per_side"] + cc.get("fees_bps", 0)) / 1e4 if costs else 0.0
    gross = Wd.sum(axis=1)
    pnl = (Wd * ron.fillna(0)).sum(axis=1) - 2 * gross * bps
    # 最低佣金近似：每只 2 笔
    nav0 = float(cfg["account"]["value"])
    if costs and cc.get("min_commission") is not None:
        pnl -= (Wd > 0).sum(axis=1) * 2 * cc["min_commission"] / nav0
    rf = D["rf"].reindex(idx).fillna(0) / 252
    pnl += (1 - gross) * rf
    # 收益记在次日
    days = P["close"].index
    nxt = pd.Series(days[np.minimum(days.searchsorted(idx) + 1, len(days) - 1)], index=idx)
    ret = pd.Series(pnl.values, index=nxt.values).groupby(level=0).sum()
    res = pd.DataFrame({"ret": ret})
    res["nav"] = nav0 * (1 + res["ret"]).cumprod()
    res["gross"] = gross.values[:len(res)] if len(gross) == len(res) else gross.reindex(res.index, method="ffill").values
    res["n"] = (Wd > 0).sum(axis=1).reindex(res.index, method="ffill").values
    res["rf"] = D["rf"].reindex(res.index).fillna(0).values / 252
    res["cost"] = np.nan
    res["turnover"] = 2 * res["gross"]
    return {"daily": res, "nav0": nav0}


def _unadjusted(P: dict) -> pd.DataFrame:
    """把拆股复权价还原成当时的真实价格（计算股数和最低佣金时要用真实价格）。"""
    close = P["close"]
    spl = P.get("split")
    if spl is None or spl.empty:
        return close
    f = spl.reindex(index=close.index, columns=close.columns).fillna(1.0)
    # 第 t 天真实价格 = 复权价 × t 之后所有拆股比例的乘积
    cum_after = f[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    return close * cum_after
