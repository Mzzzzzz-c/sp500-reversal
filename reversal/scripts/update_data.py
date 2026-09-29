"""更新数据：历史成分股、日线（雅虎）、60 分钟线（雅虎，自动累积）、财报时间（SEC）。

用法（在 reversal 目录下）：python scripts/update_data.py
建议每天北京时间上午（美股收盘后）运行一次。第一次运行约 5–10 分钟，之后会快一些。
国内访问雅虎如需代理：在 config.yaml 的 data.proxy 填写，例如 http://127.0.0.1:1087
"""
from __future__ import annotations

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd

from rev.config import load_config, path
from rev.data.store import Store, ETFS, INDEXES
from rev.data import universe as U
from rev.data import sec_earnings

t0 = time.time()
cfg = load_config()
ddir = path(cfg["data"]["dir"])
st = Store(ddir)
meta = os.path.join(ddir, "meta")


def stale(p, days):
    return not os.path.exists(p) or time.time() - os.path.getmtime(p) > days * 86400


print("1/4 标普500 成分股名单 …")
hist = U.fetch_history(os.path.join(meta, "sp500_history.csv"), refresh=stale(os.path.join(meta, "sp500_history.csv"), 7))
cur = U.fetch_current(os.path.join(meta, "sp500_current.csv"), refresh=stale(os.path.join(meta, "sp500_current.csv"), 1))
h = hist[hist["date"] >= cfg["data"]["start"]]
syms = set()
for t in h["tickers"]:
    syms |= {x.strip().replace(".", "-") for x in t.split(",")}
syms |= set(cur["Symbol"].str.replace(".", "-", regex=False))
syms = {U.RENAMES.get(s, s) for s in syms} | {v[1] for v in U.REUSED.values() if v[1]}
syms -= {k for k, v in U.REUSED.items() if v[0] >= "2099"}
print(f"   {len(syms)} 只（{cfg['data']['start']} 以来出现过的成分股）")

print("2/4 日线（雅虎）…")
st.download_daily(sorted(syms) + ETFS + INDEXES, start=cfg["data"]["start"], workers=6, refresh_hours=6)

print("3/4 60 分钟线（雅虎只保留约 730 天，本地会一直累积）…")
recent = set()
for t in hist[pd.to_datetime(hist["date"]) >= pd.Timestamp.today() - pd.Timedelta(days=760)]["tickers"]:
    recent |= {x.strip().replace(".", "-") for x in t.split(",")}
recent |= set(cur["Symbol"].str.replace(".", "-", regex=False))
recent = {U.RENAMES.get(s, s) for s in recent}
have = set(st.available("daily"))
st.download_intraday(sorted(recent & have) + ETFS + ["^VIX"], workers=4, refresh_hours=6)

print("4/4 财报发布时间（SEC 8-K Item 2.02）…")
sec_earnings.download(sorted(have - set(ETFS) - set(INDEXES)), ddir, ua=cfg["data"]["sec_user_agent"],
                      workers=5, since=cfg["data"]["start"], refresh_hours=20)
cache = os.path.join(ddir, "dataset.pkl")
if os.path.exists(cache):
    os.remove(cache)
print(f"完成，用时 {time.time() - t0:.0f} 秒")
