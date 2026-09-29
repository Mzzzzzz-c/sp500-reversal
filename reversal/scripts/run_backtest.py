"""回测：你最初的设想 vs 完善后的计划规格，三档成本，输出 results/report.html。

用法（在 reversal 目录下）：
    python scripts/run_backtest.py            # 主要方案
    python scripts/run_backtest.py --grid     # 另外跑参数网格（只用训练期挑参数，约 5-10 分钟）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rev.config import load_config, override, path
from rev import dataset, strategy as ST, backtest as BT, metrics as MT, report as RP, scenarios as SC

ap = argparse.ArgumentParser()
ap.add_argument("--grid", action="store_true", help="跑参数网格")
ap.add_argument("--no-cache", action="store_true", help="重新整理数据（数据更新后自动重建，一般不用）")
a = ap.parse_args()

t_start = time.time()
cfg = load_config()
out_dir = path("results")
os.makedirs(out_dir, exist_ok=True)
D = dataset.load(cfg, use_cache=not a.no_cache)
bt = cfg["backtest"]
PERIODS = {"训练期": bt["train"], "验证期": bt["valid"], "测试期": bt["test"]}

_cache = {}


def _key(c, parts):
    return json.dumps({p: c.get(p) for p in parts}, sort_keys=True, default=str)


def prepare(c, mode):
    # 历史特征只取决于 β 窗口和基准，缓存键不要包含选股阈值，否则内存会爆
    kH = json.dumps({k: c["signal"].get(k) for k in ("beta_window", "beta_clip", "benchmark", "time")},
                    sort_keys=True, default=str)
    if ("H", kH) not in _cache:
        _cache[("H", kH)] = ST.history_features(D, c)
    H = _cache[("H", kH)]
    kS = (mode, kH)
    if ("S",) + kS not in _cache:
        _cache[("S",) + kS] = ST.signal_features(D, H, c, mode)
    S = _cache[("S",) + kS]
    kO = (mode, kH, _key(c, ["filters"]))
    if ("O",) + kO not in _cache:
        _cache[("O",) + kO] = ST.eligibility(D, H, S, c)
    ok, reasons = _cache[("O",) + kO]
    kW = kO + (_key(c, ["signal", "portfolio", "account"]),)
    if ("W",) + kW not in _cache:
        _cache[("W",) + kW] = ST.select(D, H, S, ok, c)
    W, sel = _cache[("W",) + kW]
    return H, S, ok, reasons, W, sel


def run(c, mode, start=None, end=None):
    H, S, ok, reasons, W, sel = prepare(c, mode)
    res = BT.run(D, H, S, W, c, start=start, end=end)
    return res, sel, H, reasons


rows, yearly, navs, trades = [], {}, {}, []
for mode, start in (("daily", bt["start"]), ("intraday", None)):
    for sname in SC.STRATEGIES:
        for cname in SC.COSTS:
            c = SC.make(cfg, sname, cname)
            res, sel, H, reasons = run(c, mode, start=start)
            d = res["daily"]
            label = f"{sname}｜{cname}"
            periods = PERIODS if mode == "daily" else {"全期": [str(d.index[0].date()), "2099-12-31"]}
            for pn, (p0, p1) in periods.items():
                if cname == SC.SMALL and mode == "daily":
                    # 最低佣金让结果依赖账户规模：每个区间都从 config 里的账户净值重新开始
                    rp = run(c, mode, start=max(p0, start), end=p1)[0]
                    m = MT.perf(rp["daily"], p0, p1, nav0=rp["nav0"])
                else:
                    m = MT.perf(d, p0, p1, nav0=res["nav0"])
                if m:
                    rows.append({"模式": "日线近似(收盘价信号)" if mode == "daily" else "日内真实(15:30信号)",
                                 "方案": sname, "成本": cname, "区间": pn, **m})
            yearly[(mode, label)] = MT.yearly(d)
            navs[(mode, label)] = d["nav"] / res["nav0"]
            if cname == "零成本（理想）" and "开盘" not in sname and "对冲" not in sname and "熔断" not in sname:
                t = MT.trade_stats(D, sel, H)
                t["模式"] = mode
                t["方案"] = sname
                trades.append(t)
        print(f"  [{mode}] {sname} 完成（{time.time() - t_start:.0f}s）")
    spy = MT.benchmark(D, navs[(mode, label)].index)
    periods = PERIODS if mode == "daily" else {"全期": [str(spy.index[0].date()), "2099-12-31"]}
    for pn, (p0, p1) in periods.items():
        rows.append({"模式": "日线近似(收盘价信号)" if mode == "daily" else "日内真实(15:30信号)",
                     "方案": "SPY 买入持有", "成本": "—", "区间": pn, **MT.perf(spy, p0, p1)})
    yearly[(mode, "SPY 买入持有")] = MT.yearly(spy)
    navs[(mode, "SPY 买入持有")] = spy["nav"]

perf = pd.DataFrame(rows)
perf.to_csv(os.path.join(out_dir, "performance.csv"), index=False, encoding="utf-8-sig")
T = pd.concat(trades, ignore_index=True)
T.to_csv(os.path.join(out_dir, "trades.csv"), index=False, encoding="utf-8-sig")

# ---- 逐笔统计：次日相对 SPY 的超额收益 ----
def period_of(d):
    for pn, (p0, p1) in PERIODS.items():
        if pd.Timestamp(p0) <= d <= pd.Timestamp(p1):
            return pn
    return "其他"


T["区间"] = T["date"].map(period_of)
tsum = []
for (mode, sname), g in T.groupby(["模式", "方案"]):
    by = "区间" if mode == "daily" else None
    s = MT.summarize_trades(g, by)
    s.insert(0, "方案", sname)
    s.insert(0, "模式", "日线近似" if mode == "daily" else "日内真实(15:30)")
    tsum.append(s.reset_index().rename(columns={"index": "区间", "_all": "区间"}))
tsum = pd.concat(tsum, ignore_index=True)
tsum.to_csv(os.path.join(out_dir, "trade_summary.csv"), index=False, encoding="utf-8-sig")

# ---- 前视偏差对照：同一时间窗口，收盘价信号 vs 15:30 信号（零成本）----
look = []
i0 = D["intraday"]["px"].index.min()
for sname in ("原始设想：跌幅前30只", "计划规格"):
    for mode in ("daily", "intraday"):
        c = SC.make(cfg, sname, "零成本（理想）")
        res, *_ = run(c, mode, start=str(i0.date()))
        m = MT.perf(res["daily"], str(i0.date()), None, nav0=res["nav0"])
        look.append({"方案": sname, "信号价格": "收盘价（有前视）" if mode == "daily" else "15:30 价格（真实）",
                     "年化收益": m["年化收益"], "夏普(超额)": m["夏普(超额)"], "最大回撤": m["最大回撤"]})
look = pd.DataFrame(look)

# ---- 参数网格（只在训练期挑）----
grid_tbl = None
if not a.grid and os.path.exists(os.path.join(out_dir, "grid.csv")):
    grid_tbl = pd.read_csv(os.path.join(out_dir, "grid.csv"))      # 沿用上一次 --grid 的结果
if a.grid:
    import gc
    for k in [k for k in _cache if k[0] in ("W",)]:
        del _cache[k]
    gc.collect()
    g_rows = []
    base = SC.make(cfg, "计划规格", "机构成本5bp")
    for bench in ("spy", "sector"):
        for z in (-1.5, -2.0, -2.5, -3.0):
            for nmax in (10, 30):
                for hold in (1, 3, 5):
                    for hedge in (False, True):
                        c = override(base, {"signal.benchmark": bench, "signal.z_max": z, "signal.max_names": nmax,
                                            "portfolio.hold_days": hold, "portfolio.hedge": hedge})
                        res, *_ = run(c, "daily", start=bt["start"])
                        r = {"基准": bench, "z阈值": z, "最多持股": nmax, "持有天数": hold, "对冲": hedge}
                        for pn, (p0, p1) in PERIODS.items():
                            m = MT.perf(res["daily"], p0, p1, nav0=res["nav0"])
                            r[pn + "夏普"] = m.get("夏普(超额)")
                            r[pn + "年化"] = m.get("年化收益")
                        g_rows.append(r)
        for k in [k for k in _cache if k[0] in ("W",)]:
            del _cache[k]
        gc.collect()
        print(f"  网格 {bench} 完成（{time.time() - t_start:.0f}s）")
    grid_tbl = pd.DataFrame(g_rows).sort_values("训练期夏普", ascending=False)
    grid_tbl.to_csv(os.path.join(out_dir, "grid.csv"), index=False, encoding="utf-8-sig")


# ---- HTML 报告 ----
def pct_tbl(df, pct_cols, num_cols=()):
    df = df.copy()
    for c in pct_cols:
        if c in df:
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
    for c in num_cols:
        if c in df:
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    return df


show_cols = ["模式", "方案", "成本", "区间", "年化收益", "年化波动", "夏普(超额)", "最大回撤", "平均仓位", "平均持股数",
             "年换手(倍)", "年成本(占净值)"]
ptab = pct_tbl(perf[show_cols], ["年化收益", "年化波动", "最大回撤", "平均仓位", "年成本(占净值)"],
               ["夏普(超额)", "平均持股数", "年换手(倍)"])
sections = [
    '<p class="note">回测区间：日线 {} 至 {}；60 分钟线 {} 至 {}。股票池为当日的标普 500 成分股（含后来被剔除但雅虎仍有行情的股票）。'
    '收益含分红，现金按 3 个月国债利率计息，夏普比率按超额收益计算。</p>'.format(
        D["dates"][0].date(), D["dates"][-1].date(), D["intraday"]["px"].index.min().date(),
        D["intraday"]["px"].index.max().date()),
    "<h2>1. 各方案表现</h2>" + RP.table(ptab.set_index(["模式", "方案", "成本", "区间"])),
    "<h2>2. 逐笔统计：入选股票次日相对 SPY 的超额收益（零成本）</h2>"
    '<p class="note">这是策略最核心的证据：如果“大跌后第二天会反弹”成立，这里的“次日超额”和“次日残差”应该明显大于 0，'
    '而且要大于来回一次的交易成本（机构约 5–10bp，小账户约 15–30bp）。残差 t 值绝对值小于 2 说明和 0 没有显著差别。</p>'
    + RP.table(tsum.set_index(["模式", "方案", "区间"]).round(2)),
    "<h2>3. 前视偏差：收盘价信号 vs 15:30 信号（同一时间窗口、零成本）</h2>"
    + RP.table(pct_tbl(look, ["年化收益", "最大回撤"], ["夏普(超额)"]).set_index(["方案", "信号价格"])),
]
chart = {}
for k in [("daily", "原始设想：跌幅前30只｜零成本（理想）"), ("daily", "原始设想：跌幅前30只｜机构成本5bp"),
          ("daily", "计划规格｜零成本（理想）"), ("daily", "计划规格｜机构成本5bp"), ("daily", "SPY 买入持有")]:
    if k in navs:
        chart[k[1]] = navs[k]
sections.append("<h2>4. 净值曲线（日线近似，对数坐标）</h2>" + RP.svg_lines(chart, "净值（起点 = 1）"))
chart2 = {k[1]: v for k, v in navs.items() if k[0] == "intraday" and ("零成本" in k[1] or SC.SMALL in k[1] or "SPY" in k[1])
          and "开盘" not in k[1] and "跌超" not in k[1] and "对冲" not in k[1]}
sections.append("<h2>5. 净值曲线（日内真实 15:30 信号）</h2>" + RP.svg_lines(chart2, "净值（起点 = 1）"))
Y = pd.DataFrame({f"{m}｜{l}": s for (m, l), s in yearly.items()
                  if m == "daily" and ("机构成本" in l or "SPY" in l or "零成本" in l)}).T
sections.append("<h2>6. 分年收益（日线近似）</h2>" + RP.table(Y.apply(lambda col: col.map(lambda v: "" if pd.isna(v) else f"{v:.1%}"))))
if grid_tbl is not None:
    gt = grid_tbl.head(15).copy()
    for c in gt.columns:
        if c.endswith("年化"):
            gt[c] = gt[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
        if c.endswith("夏普"):
            gt[c] = gt[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    sections.append("<h2>7. 参数网格（机构成本 5bp；按训练期夏普排序的前 15 组）</h2>"
                    '<p class="note">只能用训练期挑参数；验证期、测试期是“事后检验”。训练期最好的参数到了验证期/测试期如果明显变差，就是过拟合。</p>'
                    + RP.table(gt.set_index(["基准", "z阈值", "最多持股", "持有天数", "对冲"])))
open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8").write(RP.page("尾盘超跌反转策略 回测报告", sections))
pd.to_pickle({"perf": perf, "trade_summary": tsum, "look": look, "yearly": yearly, "navs": navs, "grid": grid_tbl},
             os.path.join(out_dir, "summary.pkl"))
print(f"\n完成，用时 {time.time() - t_start:.0f} 秒。报告：{os.path.join(out_dir, 'report.html')}")
key = perf[(perf["区间"].isin(["测试期", "全期"])) & (perf["成本"] != "零成本（理想）")]
print(pct_tbl(key[["模式", "方案", "成本", "区间", "年化收益", "夏普(超额)", "最大回撤"]],
              ["年化收益", "最大回撤"], ["夏普(超额)"]).to_string(index=False))
