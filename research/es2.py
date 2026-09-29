import sys, pandas as pd, numpy as np
sys.path.insert(0,".")
from rev.data import universe as U
P=pd.read_pickle("/home/claude/panel_daily.pkl")
hist = pd.read_csv("/home/claude/fja05680/sp500/S&P 500 Historical Components & Changes (Updated).csv")
cur = pd.read_csv("/mnt/user-data/uploads/quant/data/meta/sp500.csv")
td = P["close"]["SPY"].dropna().index
M = U.membership(hist, cur, td, "2010-01-01").reindex(columns=P["close"].columns, fill_value=False)
M, bad = U.drop_junk(M, P["close"], P["volume"]); print("junk", bad)
pd.to_pickle(M, "/home/claude/member.pkl")
a=P["adjclose"].reindex(M.index); c=P["close"].reindex(M.index)
r=a.pct_change(fill_method=None); spy=r["SPY"]
nxt=r.shift(-1)
ex_nxt=nxt.sub(nxt["SPY"],axis=0)
mask=M & c.notna() & (c>5)
R=r.where(mask); X=ex_nxt.where(mask)
bins=[-1,-0.10,-0.07,-0.05,-0.04,-0.03,-0.02,-0.01,0,0.01,0.02,0.05,1]
df=pd.DataFrame({"r":R.stack(),"x":X.stack()}).dropna()
df["b"]=pd.cut(df.r,bins)
df["yr"]=df.index.get_level_values(0).year
g=df.groupby("b",observed=True).x.agg(["count","mean","median",lambda s:(s>0).mean()])
g.columns=["n","mean_bp","median_bp","win"]; g["mean_bp"]*=1e4; g["median_bp"]*=1e4
print(g.round(2))
# bottom 30 per day by raw return
rk=R.rank(axis=1,method="first")
sel=(rk<=30)
x30=X.where(sel).mean(axis=1)
raw30=nxt.where(mask).where(sel).mean(axis=1)
print("bottom30 next-day excess mean bp:", round(x30.mean()*1e4,2), " raw mean bp:", round(raw30.mean()*1e4,2))
print(pd.DataFrame({"excess_bp":x30.groupby(x30.index.year).mean()*1e4,"raw_bp":raw30.groupby(raw30.index.year).mean()*1e4}).round(1).T.to_string())
