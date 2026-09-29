"""尾盘信号：美东 15:30–15:45（北京时间夏令时 03:30–03:45，冬令时 04:30–04:45）运行。
用实时行情计算当天入选股票，生成收盘价市价单（MOC）清单，并记入模拟盘账本。

用法（在 reversal 目录下）：
    python scripts/live_signal.py                        # 新浪实时行情，账户规模取 config.yaml
    python scripts/live_signal.py --account-value 10000 --provider yahoo
    python scripts/live_signal.py --assume-filled        # 模拟盘：假设订单全部成交，自动更新 positions.csv

只生成清单，不会自动下单。NYSE 的 MOC 截止时间是 15:50，Nasdaq 是 15:55。
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd

from rev.config import load_config, path
from rev import dataset, live

ap = argparse.ArgumentParser()
ap.add_argument("--provider", default="sina", choices=["sina", "yahoo"])
ap.add_argument("--account-value", type=float, default=None)
ap.add_argument("--positions", default=path("positions.csv"), help="本策略当前持仓（symbol,quantity），不含你其他策略的股票")
ap.add_argument("--assume-filled", action="store_true", help="模拟盘：假设全部成交并更新持仓文件")
a = ap.parse_args()

cfg = load_config()
BANNER = ("⚠️  回测结论：该策略扣除成本后没有正收益，未通过上线标准（见 results/report.html）。\n"
          "    本清单仅用于模拟跟踪。实盘下单前请先确认模拟盘结果。")
print(BANNER)
t = live.now_ny()
if not live.is_trading_day(t.date()):
    sys.exit(f"今天（美东 {t.date()}）不是美股交易日。")
if not ((t.hour, t.minute) >= (15, 20) and (t.hour, t.minute) <= (15, 50)):
    print(f"提示：现在是美东 {t.strftime('%H:%M')}，策略设计在 15:30 左右运行；现在算出的信号和尾盘会不同。")

D = dataset.load(cfg)
last = D["dates"][-1].date()
expect = live.shift_trading_day(t.date(), -1)
if last < expect:
    print(f"⚠️ 历史日线只到 {last}，应到 {expect}。请先运行 python scripts/update_data.py（本次用实时行情里的昨收补齐）。")

pos = pd.Series(dtype=float)
if os.path.exists(a.positions):
    p = pd.read_csv(a.positions)
    if len(p):
        pos = p.set_index("symbol")["quantity"].astype(float)
av = a.account_value or cfg["account"]["value"]
out = live.live_signal(D, cfg, provider=a.provider, account_value=av, positions=pos)

d = out["date"].date()
print(f"\n美东 {out['time_ny'].strftime('%Y-%m-%d %H:%M')}｜VIX {out['vix']:.1f}｜SPY 当日 {out['spy_r']:+.2%}｜仓位系数 {out['scale']:.1f}")
if out["reasons"] is not None:
    print("各过滤条件剔除数量：" + "，".join(f"{k} {int(v)}" for k, v in out["reasons"].items() if v))
snap = out["snapshot"]
print("\n残差 z 值最低的 15 只（eligible = 是否通过过滤）：")
print(snap.head(15).assign(r=lambda x: x.r.map("{:+.2%}".format), e=lambda x: x.e.map("{:+.2%}".format),
                           z=lambda x: x.z.map("{:.2f}".format)).to_string())
sel = out["selection"]
print(f"\n今日入选 {len(sel)} 只：" + ("无" if sel.empty else ", ".join(f"{s}({w:.1%})" for s, w in zip(sel.symbol, sel.weight))))
od = out["orders"]
live_dir = path("results", "live")
os.makedirs(live_dir, exist_ok=True)
od.to_csv(os.path.join(live_dir, f"orders_{d}.csv"), encoding="utf-8-sig")
sel.to_csv(os.path.join(live_dir, f"picks_{d}.csv"), index=False, encoding="utf-8-sig")
led = os.path.join(live_dir, "paper_ledger.csv")
rec = sel.assign(scale=out["scale"], ref_price=sel["symbol"].map(out["quotes"]["price"]),
                 signal_time=str(out["time_ny"]))
if os.path.exists(led):
    old = pd.read_csv(led)
    old = old[old["date"].astype(str).str[:10] != str(d)]
    rec = pd.concat([old, rec.assign(date=str(d))], ignore_index=True)
else:
    rec = rec.assign(date=str(d))
rec.to_csv(led, index=False, encoding="utf-8-sig")
print("\n收盘价市价单（MOC）清单：")
print("无需交易" if od.empty else od.to_string())
if not od.empty:
    print(f"\n买入约 {od.loc[od.action == 'BUY', 'est_value'].sum():,.0f} 美元，卖出约 {-od.loc[od.action == 'SELL', 'est_value'].sum():,.0f} 美元")
print(f"\n已保存：{os.path.join(live_dir, f'orders_{d}.csv')}；模拟盘账本：{led}")
if a.assume_filled:
    tq = od.set_index(od.index)["target_qty"] if not od.empty else pd.Series(dtype=float)
    newpos = pos.copy()
    for s, q in tq.items():
        newpos[s] = q
    newpos = newpos[newpos > 0]
    pd.DataFrame({"symbol": newpos.index, "quantity": newpos.values}).to_csv(a.positions, index=False)
    print(f"模拟成交：持仓已更新 → {a.positions}")
