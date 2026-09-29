import json, sys
import numpy as np, pandas as pd

R = pd.read_pickle(sys.argv[2] if len(sys.argv) > 2 else "reversal/results/summary.pkl")
perf, tsum, look, yearly, navs, grid = R["perf"], R["trade_summary"], R["look"], R["yearly"], R["navs"], R["grid"]
OUT = sys.argv[1]


def pct(v, d=1):
    return "—" if v is None or pd.isna(v) else f"{v*100:.{d}f}%"


def num(v, d=2):
    return "—" if v is None or pd.isna(v) else f"{v:.{d}f}"


def cls(v):
    if v is None or pd.isna(v):
        return ""
    return "pos" if v > 0 else ("neg" if v < 0 else "")


def P(mode, strat, cost, period):
    m = "日线近似(收盘价信号)" if mode == "daily" else "日内真实(15:30信号)"
    x = perf[(perf["模式"] == m) & (perf["方案"] == strat) & (perf["成本"] == cost) & (perf["区间"] == period)]
    return x.iloc[0] if len(x) else None


# ---------- 1. 逐笔证据 ----------
def trade_rows():
    rows = []
    order = [("daily", "训练期"), ("daily", "验证期"), ("daily", "测试期"), ("intraday", None)]
    for sname in ["原始设想：跌幅前30只", "原始设想：跌超5%", "计划规格"]:
        for mode, per in order:
            mlabel = "日线近似" if mode == "daily" else "日内真实(15:30)"
            x = tsum[(tsum["模式"] == mlabel) & (tsum["方案"] == sname)]
            if per:
                x = x[x["区间"] == per]
            if not len(x):
                continue
            r = x.iloc[0]
            rows.append({"s": sname, "p": (per or "2023-10 起") + ("" if mode == "daily" else "（15:30 真实信号）"),
                         "n": int(r["笔数"]), "ret": r["次日收益_bp"], "ex": r["次日超额_bp"], "res": r["次日残差_bp"],
                         "up": r["上涨概率"], "t": r["残差t值"]})
    return rows


TR = trade_rows()

# ---------- 2. 各方案表现 ----------
def perf_rows(mode, periods, strats, costs):
    rows = []
    for s in strats:
        for c in costs:
            for p in periods:
                r = P(mode, s, c, p)
                if r is None:
                    continue
                rows.append({"s": s, "c": c, "p": p, "cagr": r["年化收益"], "sh": r["夏普(超额)"], "dd": r["最大回撤"],
                             "gross": r["平均仓位"], "n": r["平均持股数"], "cost": r["年成本(占净值)"]})
    for p in periods:
        r = P(mode, "SPY 买入持有", "—", p)
        if r is not None:
            rows.append({"s": "SPY 买入持有", "c": "—", "p": p, "cagr": r["年化收益"], "sh": r["夏普(超额)"],
                         "dd": r["最大回撤"], "gross": 1.0, "n": 1, "cost": 0.0})
    return rows


STRATS = ["原始设想：跌幅前30只", "原始设想：跌超5%", "原始设想：次日开盘卖", "计划规格", "计划规格+对冲SPY", "计划规格+回撤熔断"]
COSTS = ["零成本（理想）", "机构成本5bp", "小账户1万美元"]
PD = perf_rows("daily", ["训练期", "验证期", "测试期"], STRATS, COSTS)
PI = perf_rows("intraday", ["全期"], STRATS, COSTS)


# ---------- 3. 曲线 ----------
def series(mode, labels):
    out = []
    for lab, nm in labels:
        s = navs.get((mode, lab))
        if s is None:
            continue
        s = s.dropna()
        if mode == "daily":
            s = s.resample("W-FRI").last().dropna()
        s = s / s.iloc[0]
        out.append({"name": nm, "pts": [[int(t.timestamp() // 86400), round(float(v), 5)] for t, v in s.items()]})
    return out


CH1 = series("daily", [("原始设想：跌幅前30只｜零成本（理想）", "原始设想 · 零成本"),
                       ("原始设想：跌幅前30只｜机构成本5bp", "原始设想 · 机构成本"),
                       ("计划规格｜零成本（理想）", "计划规格 · 零成本"),
                       ("计划规格｜机构成本5bp", "计划规格 · 机构成本"),
                       ("SPY 买入持有", "SPY 买入持有")])
CH2 = series("intraday", [("原始设想：跌幅前30只｜零成本（理想）", "原始设想 · 零成本"),
                          ("原始设想：跌幅前30只｜小账户1万美元", "原始设想 · 1万美元小账户"),
                          ("计划规格｜零成本（理想）", "计划规格 · 零成本"),
                          ("计划规格｜小账户1万美元", "计划规格 · 1万美元小账户"),
                          ("SPY 买入持有", "SPY 买入持有")])

# ---------- 4. 分年 ----------
YR = []
for lab, nm in [("原始设想：跌幅前30只｜零成本（理想）", "原始设想 · 零成本"), ("原始设想：跌幅前30只｜机构成本5bp", "原始设想 · 机构成本"),
                ("计划规格｜零成本（理想）", "计划规格 · 零成本"), ("计划规格｜机构成本5bp", "计划规格 · 机构成本"),
                ("SPY 买入持有", "SPY 买入持有")]:
    y = yearly.get(("daily", lab))
    if y is not None:
        YR.append({"name": nm, "v": {int(k): float(v) for k, v in y.items() if k >= 2011}})
years = sorted({k for r in YR for k in r["v"]})

# ---------- 5. 网格 ----------
G = []
if grid is not None:
    for _, r in grid.head(10).iterrows():
        G.append({"b": "行业ETF" if r["基准"] == "sector" else "SPY", "z": r["z阈值"], "n": int(r["最多持股"]), "h": int(r["持有天数"]),
                  "hg": "是" if r["对冲"] else "否", "tr": r["训练期夏普"], "va": r["验证期夏普"], "te": r["测试期夏普"],
                  "trc": r["训练期年化"], "vac": r["验证期年化"], "tec": r["测试期年化"]})
    n_pos_train = int((grid["训练期夏普"] > 0).sum())
    n_all = len(grid)
    best = grid.iloc[0]
else:
    n_pos_train, n_all, best = 0, 0, None

json.dump({"TR": TR, "PD": PD, "PI": PI, "CH1": CH1, "CH2": CH2, "YR": YR, "years": years, "G": G,
           "look": look.to_dict("records"), "n_pos_train": n_pos_train, "n_all": n_all},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, default=float)
print("ok", len(TR), len(PD), len(PI), len(G), n_pos_train, n_all)
