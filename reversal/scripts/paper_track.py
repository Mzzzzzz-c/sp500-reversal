"""模拟盘记账：按 live_signal.py 每天记录的入选股票和权重，用真实收盘价计算
“当天收盘买入、次日收盘卖出”的收益（扣除 config.yaml 里的成本），并和 SPY 对比。

用法（在 reversal 目录下，先运行 update_data.py 更新日线）：python scripts/paper_track.py
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rev.config import load_config, path
from rev import dataset

cfg = load_config()
led = path("results", "live", "paper_ledger.csv")
if not os.path.exists(led):
    sys.exit("还没有模拟盘记录：请先在尾盘运行 scripts/live_signal.py")
L = pd.read_csv(led)
L["date"] = pd.to_datetime(L["date"].astype(str).str[:10])
D = dataset.load(cfg)
adj = D["P"]["adjclose"]
days = adj.index
cc = cfg["costs"]
nav0 = float(cfg["account"]["value"])
rows = []
nav = nav0
for d in sorted(L["date"].unique()):
    d = pd.Timestamp(d)
    if d not in days:
        continue
    i = days.get_loc(d)
    if i + 1 >= len(days):
        rows.append({"date": d.date(), "状态": "等待次日收盘"})
        continue
    g = L[L["date"] == d]
    w = (g.set_index("symbol")["weight"] * g.set_index("symbol")["scale"].fillna(1)).groupby(level=0).sum()
    r = (adj.iloc[i + 1] / adj.iloc[i] - 1).reindex(w.index).fillna(0)
    spy = adj["SPY"].iloc[i + 1] / adj["SPY"].iloc[i] - 1
    gross = float((w * r).sum())
    tv = (w * nav).values
    cost = 2 * tv.sum() * (cc["bps_per_side"] + cc.get("fees_bps", 0)) / 1e4
    if cc.get("min_commission") is not None:
        px = D["P"]["close"].iloc[i].reindex(w.index).fillna(50).values
        sh = np.maximum(1, np.round(tv / px))
        comm = np.minimum(np.maximum(cc["min_commission"], cc["commission_per_share"] * sh),
                          cc.get("max_commission_pct", 0.01) * tv)
        cost += 2 * comm.sum()
    net = gross - cost / nav
    rows.append({"date": d.date(), "持股数": len(w), "仓位": w.sum(), "毛收益": gross, "成本": cost / nav,
                 "净收益": net, "SPY": spy, "净值": nav * (1 + net)})
    nav *= 1 + net
R = pd.DataFrame(rows)
R.to_csv(path("results", "live", "paper_nav.csv"), index=False, encoding="utf-8-sig")
done = R.dropna(subset=["净收益"]) if "净收益" in R else R.iloc[0:0]
print(R.tail(20).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
if len(done):
    tot = done["净值"].iloc[-1] / nav0 - 1
    spy_tot = (1 + done["SPY"]).prod() - 1
    print(f"\n模拟盘 {len(done)} 个交易日：累计 {tot:+.2%}，同期 SPY {spy_tot:+.2%}；"
          f"日均毛收益 {done['毛收益'].mean() * 1e4:.1f}bp，日均成本 {done['成本'].mean() * 1e4:.1f}bp")
